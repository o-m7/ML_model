"""
Live-Style Backtesting Script for Quote-Based ML Models
Tests trained models on out-of-sample data with realistic trade simulation.
No shuffling. No look-ahead bias. Time-ordered execution.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data path
DATA_PATH: str = "feature_store/C:XAU-USD/quotes/C:XAU-USD_5T_quotes_labeled.parquet"

# Column names
TARGET_COL: str = "label"
RET_COL: str = "return_fwd"  # Future return column (in basis points or %)

# Artifact and output directories
ARTIFACTS_DIR: str = "artifacts"
BACKTEST_OUTPUT_DIR: str = "backtest_results"

# Backtest parameters
BACKTEST_RATIO: float = 0.3  # Use last 30% of data as out-of-sample
MIN_PROB: float = 0.55  # Decision threshold for probability (>= MIN_PROB = long)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(path: str, target_col: str, ret_col: str) -> pd.DataFrame:
    """
    Load dataset from parquet or CSV file.
    
    Args:
        path: Path to data file
        target_col: Name of target column
        ret_col: Name of realized return column
    
    Returns:
        Cleaned dataframe sorted by timestamp
    """
    logger.info(f"Loading data from: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    # Load based on extension
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Use .parquet or .csv")
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Parse timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Sorted by timestamp: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Verify required columns
    for col in [target_col, ret_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Drop rows with missing required columns
    df = df.dropna(subset=[target_col, ret_col])
    logger.info(f"Final dataset: {len(df)} rows")
    
    return df


# ============================================================================
# ARTIFACT LOADING
# ============================================================================

def load_feature_names(artifacts_dir: str, model_name: str = None) -> List[str]:
    """
    Load feature names from artifacts directory.
    If model_name is provided, try to load model-specific features first.
    Tries JSON first, then pickle, then .txt files.
    
    Args:
        artifacts_dir: Path to artifacts directory
        model_name: Optional model name to load model-specific features
    
    Returns:
        List of feature column names
    """
    # Try model-specific features first
    if model_name:
        for suffix in ['_quote_', '_ohlcv_', '_1T_', '_5T_', '_15T_', '_30T_']:
            if suffix in model_name:
                model_type = suffix.strip('_')
                for ext in ['.json', '.pkl', '.txt']:
                    feature_file = os.path.join(artifacts_dir, f'quote_features_{model_type}{ext}')
                    if os.path.exists(feature_file):
                        return _load_feature_file(feature_file, model_name)
    
    # Fall back to default feature files
    for fname in ['quote_features.json', 'quote_features.pkl']:
        fpath = os.path.join(artifacts_dir, fname)
        if os.path.exists(fpath):
            return _load_feature_file(fpath, model_name)
    
    # Try any quote_features_* file
    for fname in os.listdir(artifacts_dir):
        if 'quote_features' in fname.lower() and fname.endswith(('.json', '.pkl', '.txt')):
            return _load_feature_file(os.path.join(artifacts_dir, fname), model_name)
    
    raise FileNotFoundError(f"No feature names file found in {artifacts_dir}")


def _load_feature_file(fpath: str, model_name: str = None) -> List[str]:
    """Helper to load features from a file"""
    if fpath.endswith('.json'):
        with open(fpath, 'r') as f:
            features = json.load(f)
    elif fpath.endswith('.pkl'):
        features = joblib_load(fpath)
    else:  # .txt
        with open(fpath, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
    
    fname = os.path.basename(fpath)
    logger.info(f"✓ Loaded {len(features)} features from {fname}")
    return features


def load_scaler(artifacts_dir: str) -> StandardScaler:
    """
    Load the fitted StandardScaler.
    
    Args:
        artifacts_dir: Path to artifacts directory
    
    Returns:
        Fitted StandardScaler
    """
    logger.info("Loading scaler...")
    
    scaler_path = os.path.join(artifacts_dir, 'quote_scaler.pkl')
    
    if not os.path.exists(scaler_path):
        # Try to find any quote_scaler_*.pkl
        for fname in os.listdir(artifacts_dir):
            if 'quote_scaler' in fname.lower() and fname.endswith('.pkl'):
                scaler_path = os.path.join(artifacts_dir, fname)
                break
        else:
            raise FileNotFoundError(f"No scaler file found in {artifacts_dir}")
    
    scaler = joblib_load(scaler_path)
    logger.info("✓ Scaler loaded")
    
    return scaler


def load_models(artifacts_dir: str) -> Dict[str, BaseEstimator]:
    """
    Load all trained models from artifacts directory.
    Any .pkl file with 'model' in the name is treated as a model.
    
    Args:
        artifacts_dir: Path to artifacts directory
    
    Returns:
        Dictionary mapping model names to model instances
    """
    logger.info("Loading models...")
    
    models = {}
    
    for fname in os.listdir(artifacts_dir):
        if fname.endswith('.pkl') and 'model' in fname.lower():
            # Skip scaler and feature files
            if 'scaler' in fname.lower() or 'feature' in fname.lower():
                continue
            
            model_path = os.path.join(artifacts_dir, fname)
            model_name = fname.replace('.pkl', '')
            
            try:
                model = joblib_load(model_path)
                models[model_name] = model
                logger.info(f"✓ Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
    
    if not models:
        raise ValueError(f"No models found in {artifacts_dir}")
    
    logger.info(f"✓ Loaded {len(models)} models total")
    return models


# ============================================================================
# FEATURE SELECTION & SCALING
# ============================================================================

def select_backtest_slice(
    df: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
    ret_col: str,
    backtest_ratio: float
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Select the out-of-sample backtest window.
    
    Args:
        df: Full dataset
        feature_names: List of feature column names to use
        target_col: Name of target column
        ret_col: Name of return column
        backtest_ratio: Fraction of data to use for backtest
    
    Returns:
        X_backtest, y_backtest, r_backtest
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"SELECTING BACKTEST SLICE (last {100*backtest_ratio:.0f}%)")
    logger.info(f"{'='*70}")
    
    # Determine backtest start index
    start_idx = int(len(df) * (1.0 - backtest_ratio))
    df_back = df.iloc[start_idx:].copy()
    
    logger.info(f"Backtest window: rows {start_idx} to {len(df)} ({len(df_back)} bars)")
    
    # Find available features
    available_features = [f for f in feature_names if f in df_back.columns]
    missing_features = [f for f in feature_names if f not in df_back.columns]
    
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features:")
        for feat in missing_features[:5]:
            logger.warning(f"  - {feat}")
        if len(missing_features) > 5:
            logger.warning(f"  ... and {len(missing_features) - 5} more")
    
    if not available_features:
        raise ValueError("No requested features found in backtest data")
    
    logger.info(f"Using {len(available_features)} features")
    
    # Extract features and target
    X_backtest = df_back[available_features].copy()
    y_backtest = df_back[target_col].copy()
    r_backtest = df_back[ret_col].copy()
    
    # Drop NaN
    initial_len = len(X_backtest)
    X_backtest = X_backtest.dropna()
    y_backtest = y_backtest.loc[X_backtest.index]
    r_backtest = r_backtest.loc[X_backtest.index]
    
    dropped = initial_len - len(X_backtest)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN features")
    
    logger.info(f"Backtest data: {len(X_backtest)} bars")
    
    return X_backtest, y_backtest, r_backtest


def scale_features_for_backtest(
    scaler: StandardScaler,
    X_backtest: pd.DataFrame
) -> np.ndarray:
    """
    Transform backtest features using fitted scaler.
    
    Args:
        scaler: Fitted StandardScaler
        X_backtest: Features dataframe
    
    Returns:
        Scaled feature array
    """
    X_back_scaled = scaler.transform(X_backtest)
    logger.info(f"Scaled {X_back_scaled.shape[0]} samples, {X_back_scaled.shape[1]} features")
    
    return X_back_scaled


# ============================================================================
# TRADE SIMULATION
# ============================================================================

def simulate_trades(
    predictions: np.ndarray,
    returns: pd.Series
) -> pd.DataFrame:
    """
    Simulate trades given trade directions and realized returns.
    
    Args:
        predictions: Array of trade directions in {-1, +1}
        returns: Realized trade returns for each bar
    
    Returns:
        DataFrame with pnl, equity, direction, ret columns
    """
    # Compute PnL for each trade
    pnl = predictions * returns.values
    
    # Compute equity curve
    equity = np.cumsum(pnl)
    
    # Build results dataframe
    trades_df = pd.DataFrame({
        'direction': predictions,
        'ret': returns.values,
        'pnl': pnl,
        'equity': equity,
    })
    
    return trades_df


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def compute_performance_stats(
    pnl: pd.Series,
    equity: pd.Series
) -> Dict[str, float]:
    """
    Compute backtest performance metrics.
    
    Args:
        pnl: Per-trade PnL series
        equity: Cumulative equity curve
    
    Returns:
        Dictionary of performance statistics
    """
    # Filter out zero trades (no position taken or no movement)
    pnl_nonzero = pnl[pnl != 0]
    
    num_trades = len(pnl_nonzero)
    
    if num_trades == 0:
        logger.warning("No trades executed")
        return {
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_r': 0.0,
            'max_drawdown': 0.0,
            'sharpe': 0.0,
            'final_equity': equity.iloc[-1] if len(equity) > 0 else 0.0,
            'total_return': equity.iloc[-1] if len(equity) > 0 else 0.0,
        }
    
    # Win rate
    wins = (pnl_nonzero > 0).sum()
    win_rate = wins / num_trades if num_trades > 0 else 0.0
    
    # Profit factor
    sum_pos = pnl_nonzero[pnl_nonzero > 0].sum()
    sum_neg = pnl_nonzero[pnl_nonzero < 0].sum()
    
    if sum_neg == 0:
        profit_factor = np.inf if sum_pos > 0 else 0.0
    else:
        profit_factor = abs(sum_pos / sum_neg) if sum_neg != 0 else np.inf
    
    # Average R per trade
    avg_r = pnl_nonzero.mean()
    
    # Max drawdown
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (per trade, not annualized)
    pnl_std = pnl_nonzero.std()
    sharpe = pnl_nonzero.mean() / pnl_std if pnl_std > 0 else 0.0
    
    # Final equity and total return
    final_equity = equity.iloc[-1] if len(equity) > 0 else 0.0
    
    return {
        'num_trades': int(num_trades),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor) if not np.isinf(profit_factor) else 999.99,
        'avg_r': float(avg_r),
        'max_drawdown': float(max_drawdown),
        'sharpe': float(sharpe),
        'final_equity': float(final_equity),
        'total_return': float(final_equity),
        'sum_wins': float(sum_pos),
        'sum_losses': float(sum_neg),
    }


# ============================================================================
# MODEL BACKTESTING
# ============================================================================

def backtest_model(
    model_name: str,
    model: BaseEstimator,
    X_back_scaled: np.ndarray,
    r_back: pd.Series,
    min_prob: float
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run live-style backtest for a single model.
    
    Args:
        model_name: Name of the model
        model: Trained model instance
        X_back_scaled: Scaled feature array
        r_back: Realized returns series
        min_prob: Probability threshold for long direction
    
    Returns:
        trades_df, stats dictionary
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"BACKTESTING: {model_name}")
    logger.info(f"{'='*70}")
    
    # Generate predictions
    if hasattr(model, 'predict_proba'):
        logger.info("Using predict_proba for predictions")
        probas = model.predict_proba(X_back_scaled)[:, 1]
        directions = np.where(probas >= min_prob, 1, -1)
        logger.info(f"Probability threshold: {min_prob:.2f}")
        logger.info(f"Long trades: {(directions == 1).sum()}, Short trades: {(directions == -1).sum()}")
    else:
        logger.info("Using predict for predictions")
        preds = model.predict(X_back_scaled)
        # Map predictions to directions
        directions = np.where(preds >= 0.5, 1, -1) if preds.max() <= 1 else np.where(preds > 0, 1, -1)
        logger.info(f"Long trades: {(directions == 1).sum()}, Short trades: {(directions == -1).sum()}")
    
    # Simulate trades
    trades_df = simulate_trades(directions, r_back)
    
    # Compute stats
    stats = compute_performance_stats(trades_df['pnl'], trades_df['equity'])
    
    # Print stats
    logger.info("\nPERFORMANCE METRICS:")
    logger.info(f"  Trades:         {stats['num_trades']}")
    logger.info(f"  Win Rate:       {stats['win_rate']:.2%}")
    logger.info(f"  Profit Factor:  {stats['profit_factor']:.2f}")
    logger.info(f"  Avg R:          {stats['avg_r']:.4f}")
    logger.info(f"  Max Drawdown:   {stats['max_drawdown']:.4f}")
    logger.info(f"  Sharpe:         {stats['sharpe']:.4f}")
    logger.info(f"  Final Equity:   {stats['final_equity']:.4f}")
    
    return trades_df, stats


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_backtest_results(
    model_name: str,
    trades_df: pd.DataFrame,
    stats: Dict[str, float],
    out_dir: str = BACKTEST_OUTPUT_DIR
) -> None:
    """
    Save per-model backtest results to CSV and JSON.
    
    Args:
        model_name: Name of model
        trades_df: Trades dataframe
        stats: Performance statistics
        out_dir: Output directory
    """
    Path(out_dir).mkdir(exist_ok=True)
    
    # Save trades CSV
    trades_csv = os.path.join(out_dir, f'{model_name}_trades.csv')
    trades_df.to_csv(trades_csv, index=True)
    logger.info(f"✓ Saved trades: {trades_csv}")
    
    # Save stats JSON
    stats_json = os.path.join(out_dir, f'{model_name}_stats.json')
    with open(stats_json, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Saved stats: {stats_json}")


def save_summary(
    summary_stats: Dict[str, Dict[str, float]],
    out_dir: str = BACKTEST_OUTPUT_DIR
) -> None:
    """
    Save overall summary of all model results to CSV.
    
    Args:
        summary_stats: Dictionary mapping model names to stats
        out_dir: Output directory
    """
    Path(out_dir).mkdir(exist_ok=True)
    
    # Convert to dataframe
    df_summary = pd.DataFrame(summary_stats).T
    
    # Save to CSV
    summary_csv = os.path.join(out_dir, 'backtest_summary.csv')
    df_summary.to_csv(summary_csv)
    logger.info(f"✓ Saved summary: {summary_csv}")
    
    # Print summary table to console
    logger.info("\n" + "="*70)
    logger.info("BACKTEST SUMMARY - ALL MODELS")
    logger.info("="*70)
    logger.info(df_summary.to_string())


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main backtesting pipeline"""
    logger.info("\n" + "#"*70)
    logger.info("# LIVE-STYLE QUOTE MODEL BACKTESTING")
    logger.info("#"*70)
    
    try:
        # Load data
        df = load_data(DATA_PATH, TARGET_COL, RET_COL)
        
        # Load models
        models = load_models(ARTIFACTS_DIR)
        
        # Load full data for selection
        logger.info("Loading full dataset for feature selection...")
        
        # Run backtest for each model with its own features
        summary_stats = {}
        
        for model_name, model in sorted(models.items()):
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"Preparing backtest for: {model_name}")
                
                # Load model-specific features
                model_features = load_feature_names(ARTIFACTS_DIR, model_name)
                
                # Load model-specific scaler
                model_scaler = load_scaler(ARTIFACTS_DIR)
                
                # Select backtest slice with this model's features
                X_back, y_back, r_back = select_backtest_slice(
                    df, model_features, TARGET_COL, RET_COL, BACKTEST_RATIO
                )
                
                # Scale features with this model's scaler
                X_back_scaled = scale_features_for_backtest(model_scaler, X_back)
                
                # Run backtest
                trades_df, stats = backtest_model(
                    model_name=model_name,
                    model=model,
                    X_back_scaled=X_back_scaled,
                    r_back=r_back,
                    min_prob=MIN_PROB,
                )
                
                # Attach timestamp index if available
                if 'timestamp' in df.columns:
                    backtest_start_idx = int(len(df) * (1.0 - BACKTEST_RATIO))
                    backtest_timestamps = df.iloc[backtest_start_idx:]['timestamp'].iloc[:len(trades_df)]
                    trades_df.index = backtest_timestamps
                
                save_backtest_results(model_name, trades_df, stats)
                summary_stats[model_name] = stats
            
            except Exception as e:
                logger.error(f"Failed to backtest {model_name}: {e}")
                continue
        
        # Save and print summary
        save_summary(summary_stats)
        
        logger.info("\n" + "#"*70)
        logger.info("✅ BACKTESTING COMPLETE")
        logger.info("#"*70)
    
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
