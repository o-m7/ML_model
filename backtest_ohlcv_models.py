"""
Live-style OHLCV Backtesting Script
===================================

Loads OHLCV models and runs time-ordered, out-of-sample backtests with no future leakage.
Computes performance metrics (win rate, profit factor, Sharpe, etc.) for each model.
Saves per-model results and summary statistics.

Data & Artifact Assumptions:
- DATA_PATH: labeled OHLCV dataset with OHLCV columns, features, label, and trade_return
- ARTIFACTS_DIR: contains ohlcv_scaler.pkl, ohlcv_feature_names.json, and ohlcv_model_*.pkl files
- BACKTEST_RATIO: use last N% of data as out-of-sample test window
"""

import os
import json
import logging
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH_TEMPLATE: str = "./feature_store/C:XAU-USD/C:XAU-USD_{timeframe}_labeled.parquet"
TARGET_COL: str = "label"
RET_COL: str = "return_fwd"
ARTIFACTS_DIR: str = "artifacts"
BACKTEST_RATIO: float = 0.3  # Use last 30% of data as backtest window
MIN_PROB: float = 0.55  # Probability threshold for long vs short
TIMEFRAMES: List[str] = ["1T", "5T", "15T", "30T"]  # All available timeframes

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(path: str, target_col: str, ret_col: str) -> pd.DataFrame:
    """
    Load the OHLCV dataset from a parquet or CSV file.
    
    Steps:
    - Detect file extension and load accordingly.
    - Parse 'timestamp' column to datetime if present.
    - Sort by 'timestamp' ascending (preserve time order).
    - Drop rows with NaN in target or return columns.
    
    Args:
        path: Path to data file (.parquet or .csv)
        target_col: Name of target/label column
        ret_col: Name of realized trade return column
        
    Returns:
        Cleaned DataFrame sorted by timestamp
    """
    logger.info(f"Loading OHLCV data from {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}. Use .parquet or .csv")
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Parse timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        logger.info(f"Sorted by timestamp: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Validate required columns
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    if ret_col not in df.columns:
        raise ValueError(f"Return column '{ret_col}' not found in dataset. "
                        f"Must be pre-computed in the dataset.")
    
    # Drop rows with NaN in critical columns
    initial_len = len(df)
    df = df.dropna(subset=[target_col, ret_col])
    logger.info(f"Dropped {initial_len - len(df)} rows with NaN in target/return columns")
    
    return df


# ============================================================================
# ARTIFACT LOADING
# ============================================================================

def load_feature_names(artifacts_dir: str, timeframe: str = None) -> List[str]:
    """
    Load OHLCV feature names from artifacts directory.
    
    Try timeframe-specific files first (e.g., ohlcv_features_1T.txt),
    then generic JSON/pickle format.
    
    Args:
        artifacts_dir: Path to artifacts directory
        timeframe: Timeframe string (e.g., "1T", "5T", "15T", "30T") or None for generic
        
    Returns:
        List of feature names used during model training
    """
    # Try timeframe-specific file first (e.g., ohlcv_features_1T.txt)
    if timeframe:
        txt_path = os.path.join(artifacts_dir, f"ohlcv_features_{timeframe}.txt")
        if os.path.exists(txt_path):
            logger.info(f"Loading feature names from {txt_path}")
            with open(txt_path, "r") as f:
                feature_names = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(feature_names)} feature names")
            return feature_names
    
    # Try generic JSON
    json_path = os.path.join(artifacts_dir, "ohlcv_feature_names.json")
    if os.path.exists(json_path):
        logger.info(f"Loading feature names from {json_path}")
        with open(json_path, "r") as f:
            feature_names = json.load(f)
        if not isinstance(feature_names, list):
            raise ValueError("Feature names JSON must be a list of strings")
        logger.info(f"Loaded {len(feature_names)} feature names")
        return feature_names
    
    # Try generic pickle
    pkl_path = os.path.join(artifacts_dir, "ohlcv_feature_names.pkl")
    if os.path.exists(pkl_path):
        logger.info(f"Loading feature names from {pkl_path}")
        feature_names = joblib_load(pkl_path)
        if not isinstance(feature_names, list):
            raise ValueError("Feature names pickle must be a list of strings")
        logger.info(f"Loaded {len(feature_names)} feature names")
        return feature_names
    
    raise FileNotFoundError(
        f"Feature names file not found in {artifacts_dir}. "
        f"Expected: ohlcv_features_{{timeframe}}.txt, ohlcv_feature_names.json, or ohlcv_feature_names.pkl"
    )


def load_scaler(artifacts_dir: str, timeframe: str = None) -> StandardScaler:
    """
    Load the fitted StandardScaler from artifacts.
    
    Try timeframe-specific scaler first (e.g., ohlcv_scaler_1T.pkl),
    then generic ohlcv_scaler.pkl.
    
    Args:
        artifacts_dir: Path to artifacts directory
        timeframe: Timeframe string (e.g., "1T", "5T", "15T", "30T") or None for generic
        
    Returns:
        Fitted StandardScaler instance
    """
    # Try timeframe-specific scaler first
    if timeframe:
        scaler_path = os.path.join(artifacts_dir, f"ohlcv_scaler_{timeframe}.pkl")
        if os.path.exists(scaler_path):
            logger.info(f"Loading scaler from {scaler_path}")
            scaler = joblib_load(scaler_path)
            logger.info("Scaler loaded successfully")
            return scaler
    
    # Try generic scaler
    scaler_path = os.path.join(artifacts_dir, "ohlcv_scaler.pkl")
    if os.path.exists(scaler_path):
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib_load(scaler_path)
        logger.info("Scaler loaded successfully")
        return scaler
    
    raise FileNotFoundError(
        f"Scaler not found in {artifacts_dir}. "
        f"Expected: ohlcv_scaler_{timeframe}.pkl or ohlcv_scaler.pkl"
    )


def load_models(artifacts_dir: str, timeframe_filter: str = None) -> Dict[str, BaseEstimator]:
    """
    Load all OHLCV-specific model .pkl files from artifacts directory.
    
    If timeframe_filter is provided (e.g., "1T"), only loads models matching that timeframe.
    Filters out quote models (which have different feature sets).
    
    Args:
        artifacts_dir: Path to artifacts directory
        timeframe_filter: Optional timeframe to filter by (e.g., "1T", "5T", "15T", "30T")
        
    Returns:
        Dictionary mapping model_name -> model_instance
    """
    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
    
    models: Dict[str, BaseEstimator] = {}
    
    for filename in sorted(os.listdir(artifacts_dir)):
        if not filename.endswith(".pkl"):
            continue
        
        # Skip scaler and feature files
        if filename in ["ohlcv_scaler.pkl", "ohlcv_feature_names.pkl"] or "scaler" in filename or "features" in filename:
            continue
        
        # Only load OHLCV models, skip quote models
        if "ohlcv_model" not in filename.lower():
            continue
        
        # Filter by timeframe if specified
        if timeframe_filter and timeframe_filter not in filename:
            continue
        
        filepath = os.path.join(artifacts_dir, filename)
        model_name = filename[:-4]  # Remove .pkl extension
        
        logger.info(f"Loading model: {model_name}")
        models[model_name] = joblib_load(filepath)
    
    if not models:
        filter_str = f" matching timeframe '{timeframe_filter}'" if timeframe_filter else ""
        raise ValueError(
            f"No OHLCV models found in {artifacts_dir}{filter_str}. "
            f"Expected files matching pattern '*ohlcv_model*.pkl'"
        )
    
    logger.info(f"Loaded {len(models)} OHLCV models: {list(models.keys())}")
    return models


# ============================================================================
# DATA AGGREGATION
# ============================================================================

def aggregate_to_timeframe(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Aggregate 1T OHLCV data to a target timeframe (5T, 15T, 30T, etc.).
    
    Args:
        df: DataFrame with 1T OHLCV data and timestamp index
        target_tf: Target timeframe string (e.g., "5T", "15T", "30T")
        
    Returns:
        Aggregated DataFrame at target timeframe
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column for aggregation")
    
    df.set_index("timestamp", inplace=True)
    
    # Define aggregation rules
    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    
    # Add numeric columns for mean aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in agg_rules:
            agg_rules[col] = "mean"  # Most features aggregate via mean
    
    # Special case: trade_return and return_fwd should use the last value or mean
    if "return_fwd" in agg_rules:
        agg_rules["return_fwd"] = "mean"
    if "trade_return" in agg_rules:
        agg_rules["trade_return"] = "mean"
    
    # Use target timeframe for resampling
    df_agg = df.resample(target_tf).agg(agg_rules)
    
    # Reset index to have timestamp as column again
    df_agg.reset_index(inplace=True)
    
    # Drop rows with all NaN (e.g., overnight gaps)
    df_agg = df_agg.dropna(subset=["close"])
    
    return df_agg


# ============================================================================
# BACKTEST SLICE SELECTION
# ============================================================================

def select_backtest_slice(
    df: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
    ret_col: str,
    backtest_ratio: float
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Select the out-of-sample backtest window from the dataset.
    
    Steps:
    - Use the last `backtest_ratio` fraction of data for testing.
    - Intersect requested features with available columns.
    - Warn about missing features.
    - Drop rows with NaN in selected features.
    - Return: X_backtest, y_backtest, r_backtest
    
    Args:
        df: Full dataset
        feature_names: List of feature names to use
        target_col: Name of target column
        ret_col: Name of return column
        backtest_ratio: Fraction of data to use for backtesting (e.g., 0.3 = last 30%)
        
    Returns:
        Tuple of (X_backtest, y_backtest, r_backtest) DataFrames/Series
    """
    logger.info("\n" + "=" * 70)
    logger.info("SELECTING BACKTEST SLICE (last {:.0%})".format(backtest_ratio))
    logger.info("=" * 70)
    
    start_idx = int(len(df) * (1.0 - backtest_ratio))
    df_back = df.iloc[start_idx:].copy()
    
    logger.info(f"Backtest window: rows {start_idx} to {len(df)} ({len(df_back)} bars)")
    
    # Intersect requested features with available columns
    available_features = [f for f in feature_names if f in df_back.columns]
    missing_features = set(feature_names) - set(available_features)
    
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} requested features: {missing_features}")
    
    if not available_features:
        raise ValueError("No requested features available in the dataset")
    
    logger.info(f"Using {len(available_features)} features")
    
    # Drop rows with NaN in selected features
    initial_len = len(df_back)
    df_back = df_back.dropna(subset=available_features)
    dropped = initial_len - len(df_back)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN in features")
    
    X_backtest = df_back[available_features]
    y_backtest = df_back[target_col]
    r_backtest = df_back[ret_col]
    
    logger.info(f"Backtest data: {len(X_backtest)} bars, {X_backtest.shape[1]} features")
    
    return X_backtest, y_backtest, r_backtest


# ============================================================================
# FEATURE SCALING
# ============================================================================

def scale_features_for_backtest(
    scaler: StandardScaler,
    X_backtest: pd.DataFrame
) -> np.ndarray:
    """
    Transform backtest features using the fitted scaler.
    
    Does NOT refit the scaler; only applies pre-fitted transformation.
    
    Args:
        scaler: Fitted StandardScaler instance
        X_backtest: Features DataFrame to scale
        
    Returns:
        Scaled feature array (numpy array)
    """
    logger.info("Scaling features with fitted scaler...")
    X_scaled = scaler.transform(X_backtest)
    logger.info(f"Scaled {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    return X_scaled


# ============================================================================
# DIRECTION MAPPING & TRADE SIMULATION
# ============================================================================

def to_direction(pred: np.ndarray, positive_label: float = 1.0) -> np.ndarray:
    """
    Map predictions to trading directions in {-1, +1}.
    
    Args:
        pred: Predicted class label or probability
        positive_label: Label value that maps to +1 direction (long)
        
    Returns:
        Array of directions in {-1, +1}
    """
    return np.where(pred == positive_label, 1, -1)


def simulate_trades(
    directions: np.ndarray,
    returns: pd.Series
) -> pd.DataFrame:
    """
    Simulate trades given predicted directions and realized trade returns.
    
    Steps:
    - Multiply direction {-1, +1} by realized return to get P&L per trade.
    - Compute cumulative equity curve.
    - Return DataFrame with direction, ret, pnl, equity indexed by original data.
    
    Args:
        directions: Array of directions in {-1, +1}, same length as returns
        returns: Series of realized trade returns
        
    Returns:
        DataFrame with columns: ['direction', 'ret', 'pnl', 'equity']
    """
    ret_values = returns.values
    pnl = directions * ret_values
    equity = np.cumsum(pnl)
    
    trades_df = pd.DataFrame({
        "direction": directions,
        "ret": ret_values,
        "pnl": pnl,
        "equity": equity,
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
    Compute comprehensive performance metrics for OHLCV backtest.
    
    Metrics:
    - num_trades: Count of trades (non-zero pnl)
    - win_rate: Proportion of winning trades
    - profit_factor: Sum of wins / abs(sum of losses)
    - avg_r: Mean P&L per trade
    - max_drawdown: Maximum peak-to-trough drawdown
    - sharpe: Mean P&L / Std P&L (risk-adjusted return)
    - final_equity: Final cumulative equity
    
    Args:
        pnl: Series of trade P&L values
        equity: Series of cumulative equity curve
        
    Returns:
        Dictionary of performance metrics
    """
    # Count trades and win rate
    pnl_nonzero = pnl[pnl != 0]
    num_trades = len(pnl_nonzero)
    
    if num_trades == 0:
        logger.warning("No non-zero trades found")
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_r": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "final_equity": 0.0,
            "total_return": 0.0,
            "sum_wins": 0.0,
            "sum_losses": 0.0,
        }
    
    wins = np.sum(pnl_nonzero > 0)
    win_rate = wins / num_trades
    
    # Profit factor
    sum_wins = pnl[pnl > 0].sum()
    sum_losses = pnl[pnl < 0].sum()
    
    if sum_losses == 0:
        profit_factor = np.inf if sum_wins > 0 else 0.0
    else:
        profit_factor = sum_wins / abs(sum_losses)
    
    # Average R per trade
    avg_r = pnl.mean()
    
    # Max drawdown
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = float(drawdown.min())
    
    # Sharpe ratio (per-trade)
    if pnl.std() == 0:
        sharpe = 0.0
    else:
        sharpe = pnl.mean() / pnl.std()
    
    # Final equity
    final_equity = float(equity.iloc[-1]) if len(equity) > 0 else 0.0
    total_return = final_equity
    
    return {
        "num_trades": int(num_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_r": float(avg_r),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
        "final_equity": float(final_equity),
        "total_return": float(total_return),
        "sum_wins": float(sum_wins),
        "sum_losses": float(sum_losses),
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
    Run live-style backtest for a single OHLCV model.
    
    Steps:
    1. Generate predictions (prefer predict_proba if available).
    2. Convert predictions to directions in {-1, +1}.
    3. Simulate trades and compute equity curve.
    4. Compute performance metrics.
    
    Args:
        model_name: Name of the model (for logging)
        model: Fitted model instance with predict() or predict_proba()
        X_back_scaled: Scaled backtest features (numpy array)
        r_back: Realized trade returns (Series)
        min_prob: Probability threshold for long vs short
        
    Returns:
        Tuple of (trades_df, stats_dict)
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"BACKTESTING: {model_name}")
    logger.info("=" * 70)
    
    # Generate predictions
    if hasattr(model, "predict_proba"):
        logger.info("Using predict_proba for predictions")
        probas = model.predict_proba(X_back_scaled)[:, 1]
        logger.info(f"Probability threshold: {min_prob}")
        directions = np.where(probas >= min_prob, 1, -1)
    else:
        logger.info("Using predict for predictions")
        preds = model.predict(X_back_scaled)
        
        # Map to directions
        unique_labels = np.unique(preds)
        if len(unique_labels) == 2:
            if set(unique_labels) == {0, 1}:
                directions = np.where(preds == 1, 1, -1)
            elif set(unique_labels) == {-1, 1}:
                directions = preds.astype(int)
            else:
                # Arbitrary mapping: first label -> -1, second -> +1
                sorted_labels = np.sort(unique_labels)
                directions = np.where(preds == sorted_labels[1], 1, -1)
        else:
            logger.warning(f"Unexpected number of unique labels: {unique_labels}")
            directions = np.where(preds > preds.mean(), 1, -1)
    
    # Count trades by direction
    long_trades = np.sum(directions == 1)
    short_trades = np.sum(directions == -1)
    logger.info(f"Long trades: {long_trades}, Short trades: {short_trades}")
    
    # Simulate trades
    trades_df = simulate_trades(directions, r_back)
    
    # Compute stats
    stats = compute_performance_stats(trades_df["pnl"], trades_df["equity"])
    
    # Log stats
    logger.info("\nPERFORMANCE METRICS:")
    logger.info(f"  Trades:         {stats['num_trades']}")
    logger.info(f"  Win Rate:       {stats['win_rate']*100:.2f}%")
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
    out_dir: str = "backtest_results_ohlcv"
) -> None:
    """
    Save per-model backtest results to CSV and JSON.
    
    Args:
        model_name: Name of the model
        trades_df: DataFrame of trade-by-trade results
        stats: Dictionary of performance metrics
        out_dir: Output directory for results
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Save trades CSV
    trades_path = os.path.join(out_dir, f"{model_name}_trades.csv")
    trades_df.to_csv(trades_path)
    logger.info(f"✓ Saved trades: {trades_path}")
    
    # Save stats JSON
    stats_path = os.path.join(out_dir, f"{model_name}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Saved stats: {stats_path}")


def save_summary(
    summary_stats: Dict[str, Dict[str, float]],
    out_dir: str = "backtest_results_ohlcv"
) -> None:
    """
    Save overall summary of all OHLCV models to CSV.
    
    Args:
        summary_stats: Dictionary mapping model_name -> metrics_dict
        out_dir: Output directory for results
    """
    os.makedirs(out_dir, exist_ok=True)
    
    summary_df = pd.DataFrame.from_dict(summary_stats, orient="index")
    summary_path = os.path.join(out_dir, "summary_stats.csv")
    summary_df.to_csv(summary_path)
    logger.info(f"✓ Saved summary: {summary_path}")
    
    return summary_df


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Main OHLCV backtesting workflow.
    
    Backtests OHLCV models at all available timeframes (1T, 5T, 15T, 30T).
    Loads pre-computed labeled data at each timeframe to ensure all features are available.
    """
    try:
        # Track all results across all timeframes
        all_summary_stats: Dict[str, Dict[str, float]] = {}
        
        # Backtest each timeframe
        for tf in TIMEFRAMES:
            try:
                logger.info("\n" + "=" * 70)
                logger.info(f"BACKTESTING OHLCV MODELS AT {tf} TIMEFRAME")
                logger.info("=" * 70)
                
                # Construct path to timeframe-specific data
                data_path = DATA_PATH_TEMPLATE.format(timeframe=tf)
                
                # Load timeframe-specific data
                logger.info(f"Loading {tf} OHLCV data...")
                df = load_data(data_path, TARGET_COL, RET_COL)
                
                # Load timeframe-specific artifacts
                try:
                    feature_names = load_feature_names(ARTIFACTS_DIR, tf)
                    scaler = load_scaler(ARTIFACTS_DIR, tf)
                    models = load_models(ARTIFACTS_DIR, tf)
                except FileNotFoundError as e:
                    logger.warning(f"Skipping {tf}: {e}")
                    continue
                
                # Select backtest slice
                X_back, y_back, r_back = select_backtest_slice(
                    df, feature_names, TARGET_COL, RET_COL, BACKTEST_RATIO
                )
                
                # Scale features
                X_back_scaled = scale_features_for_backtest(scaler, X_back)
                
                # Run backtest for each OHLCV model at this timeframe
                for model_name, model in models.items():
                    try:
                        trades_df, stats = backtest_model(
                            model_name=model_name,
                            model=model,
                            X_back_scaled=X_back_scaled,
                            r_back=r_back,
                            min_prob=MIN_PROB,
                        )
                        
                        # Align index with backtest data
                        trades_df.index = X_back.index
                        
                        # Save results
                        save_backtest_results(model_name, trades_df, stats)
                        all_summary_stats[model_name] = stats
                        
                    except Exception as e:
                        logger.error(f"Error backtesting {model_name}: {str(e)}", exc_info=True)
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing {tf}: {str(e)}", exc_info=True)
                continue
        
        # Save and display overall summary
        if all_summary_stats:
            summary_df = save_summary(all_summary_stats)
            
            # Print summary table
            logger.info("\n" + "=" * 140)
            logger.info("BACKTEST SUMMARY - OHLCV MODELS (ALL TIMEFRAMES)")
            logger.info("=" * 140)
            logger.info("\n" + summary_df.to_string())
            logger.info("\n" + "=" * 140)
            logger.info("✅ BACKTESTING COMPLETE")
            logger.info("=" * 140)
        else:
            logger.error("No models were successfully backtested")
        
    except Exception as e:
        logger.error(f"\n❌ BACKTESTING FAILED: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
