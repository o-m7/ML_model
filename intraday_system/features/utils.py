"""Feature utilities: alignment, resampling, leakage prevention."""

import pandas as pd
import numpy as np
from typing import Optional


def align_timeframes(
    df_low: pd.DataFrame,
    df_high: pd.DataFrame,
    method: str = "ffill"
) -> pd.DataFrame:
    """
    Safely join higher timeframe features to lower timeframe without leakage.
    
    Args:
        df_low: Lower timeframe DataFrame with 'timestamp'
        df_high: Higher timeframe DataFrame with 'timestamp'
        method: Fill method ('ffill' to prevent lookahead)
        
    Returns:
        df_low with HTF features appended
    """
    if 'timestamp' not in df_low.columns or 'timestamp' not in df_high.columns:
        raise ValueError("Both DataFrames must have 'timestamp' column")
    
    # Ensure datetime
    df_low = df_low.copy()
    df_high = df_high.copy()
    df_low['timestamp'] = pd.to_datetime(df_low['timestamp'])
    df_high['timestamp'] = pd.to_datetime(df_high['timestamp'])
    
    # Set index for merging
    df_low_indexed = df_low.set_index('timestamp')
    df_high_indexed = df_high.set_index('timestamp')
    
    # Merge as-of (prevent future leakage)
    merged = pd.merge_asof(
        df_low_indexed.sort_index(),
        df_high_indexed.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward',  # Use only past HTF data
        suffixes=('', '_htf')
    )
    
    merged = merged.reset_index()
    
    return merged


def check_leakage(df: pd.DataFrame, target_col: str = 'target') -> dict:
    """
    Check for potential data leakage in features.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        
    Returns:
        Dictionary with suspicious features
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    if 'timestamp' not in df.columns:
        raise ValueError("'timestamp' column required for temporal checks")
    
    results = {
        'suspicious_correlations': [],
        'future_peeking': [],
        'constant_features': []
    }
    
    # Check for suspiciously high correlations
    feature_cols = [col for col in df.columns 
                   if col not in ['timestamp', target_col, 'open', 'high', 'low', 'close', 'volume']]
    
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Check correlation with target
        corr = df[col].corr(df[target_col])
        if abs(corr) > 0.25:  # Suspiciously high
            results['suspicious_correlations'].append({
                'feature': col,
                'correlation': corr
            })
        
        # Check if feature is constant
        if df[col].nunique() <= 1:
            results['constant_features'].append(col)
    
    # Check for future peeking: feature should not correlate with future returns
    if 'close' in df.columns:
        future_return = df['close'].pct_change().shift(-1)
        
        for col in feature_cols[:50]:  # Sample check
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            corr = df[col].corr(future_return)
            if abs(corr) > 0.15:
                results['future_peeking'].append({
                    'feature': col,
                    'future_return_corr': corr
                })
    
    return results


def winsorize(series: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
    """Winsorize series to handle outliers."""
    lower, upper = series.quantile([limits[0], 1 - limits[1]])
    return series.clip(lower, upper)


def cap_z_scores(df: pd.DataFrame, cols: list, max_z: float = 5.0) -> pd.DataFrame:
    """Cap z-scores at ±max_z to handle outliers."""
    df = df.copy()
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                z_scores = (df[col] - mean) / std
                df[col] = mean + z_scores.clip(-max_z, max_z) * std
    return df


def remove_collinear_features(df: pd.DataFrame, threshold: float = 0.95) -> list:
    """
    Remove highly collinear features.
    
    Args:
        df: DataFrame with features
        threshold: Correlation threshold
        
    Returns:
        List of features to keep
    """
    feature_cols = [col for col in df.columns 
                   if pd.api.types.is_numeric_dtype(df[col])]
    
    corr_matrix = df[feature_cols].corr().abs()
    
    # Upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation > threshold
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    features_to_keep = [col for col in feature_cols if col not in to_drop]
    
    return features_to_keep

