#!/usr/bin/env python3
"""
CITADEL METALS TRAINING SYSTEM
===============================
Production-grade ML training pipeline for XAUUSD and XAGUSD algorithmic trading.

Supports three core strategy archetypes:
1. Short-term mean reversion (5T, 15T)
2. Volatility breakouts (5T, 15T, 30T)
3. Trend following with regime filters (30T, 1H)

Features:
- Proper temporal train/test splits (no data leakage)
- Comprehensive feature engineering for algo trading
- Realistic trade simulation with costs
- Prop-firm validation metrics
- Only saves models that pass strict thresholds

Usage:
    python citadel_training_system.py
"""

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Markets
SYMBOLS = ["XAUUSD", "XAGUSD"]
TIMEFRAMES = ["5T", "15T", "30T", "1H"]

# Data paths
DATA_DIR = Path("feature_store")  # Adjust to your data location
MODELS_DIR = Path("models_citadel")

# Feature horizons for mean reversion
RETURN_HORIZONS = [1, 3, 5, 10, 20]
EMA_PERIODS = [10, 20, 50, 100, 200]
VOLATILITY_WINDOWS = [10, 20, 50]

# Label configuration
LABEL_HORIZONS = {
    "5T": 5,   # 25 minutes forward
    "15T": 3,  # 45 minutes forward
    "30T": 3,  # 90 minutes forward
    "1H": 2    # 2 hours forward
}

# Threshold multipliers (x ATR) for classification
# Lower thresholds = more labeled signals, easier for model to learn
LONG_THRESHOLD_ATR = 0.5   # Need to move +0.5 ATR to be labeled "long"
SHORT_THRESHOLD_ATR = 0.5  # Need to move -0.5 ATR to be labeled "short"

# Model hyperparameters
XGB_PARAMS = {
    'objective': 'multi:softprob',  # 3-class: flat, long, short
    'num_class': 3,
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
    'tree_method': 'hist',
}

# Trading costs per symbol (in points)
TRADING_COSTS = {
    "XAUUSD": {
        "spread": 20.0,      # $2.00 per lot
        "commission": 7.0,   # $7 per round turn per lot
        "point_value": 0.10, # $0.10 per point per lot
    },
    "XAGUSD": {
        "spread": 2.0,       # $0.02 per lot
        "commission": 7.0,
        "point_value": 0.01,
    }
}

# Trade simulation parameters
TP_ATR_MULTIPLE = 2.5   # Take profit at 2.5x ATR
SL_ATR_MULTIPLE = 1.5   # Stop loss at 1.5x ATR

# Model acceptance thresholds (realistic for initial training)
MIN_TRADES_TEST = 150       # At least 150 trades for statistical significance
MIN_WIN_RATE = 0.45         # 45%+ win rate (acceptable with good R:R)
MIN_PROFIT_FACTOR = 1.2     # 1.2+ profit factor (profitable after costs)
MAX_DRAWDOWN_PCT = 0.25     # Max 25% drawdown
MIN_EXPECTANCY = 0.05       # Average R per trade (small positive edge)


# ============================================================================
# DATA LOADING AND RESAMPLING
# ============================================================================

def load_data(symbol: str) -> pd.DataFrame:
    """
    Load raw OHLCV data for a symbol.

    Expects data in feature_store/{symbol}/{symbol}_5T.parquet (or similar).
    You can adapt this to load from your actual data source.

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume
    """
    # Try to load 5T data as base (finest granularity)
    data_path = DATA_DIR / symbol / f"{symbol}_5T.parquet"

    if not data_path.exists():
        # Try other timeframes or raise error
        raise FileNotFoundError(f"Data not found for {symbol} at {data_path}")

    df = pd.read_parquet(data_path)

    # Ensure proper columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

    # Ensure timezone-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    # Sort by time
    df = df.sort_index()

    # Keep only OHLCV
    df = df[required_cols].copy()

    print(f"✅ Loaded {symbol}: {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

    return df


def resample_to_timeframes(df: pd.DataFrame, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Resample OHLCV data to multiple timeframes.

    Args:
        df: Base OHLCV dataframe with DatetimeIndex
        timeframes: List of timeframe strings (e.g., ["5T", "15T", "1H"])

    Returns:
        Dict mapping timeframe -> resampled DataFrame
    """
    resampled = {}

    for tf in timeframes:
        # Resample with proper OHLCV aggregation
        df_tf = df.resample(tf, label='left', closed='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        resampled[tf] = df_tf
        print(f"   Resampled to {tf}: {len(df_tf):,} bars")

    return resampled


# ============================================================================
# FEATURE ENGINEERING FOR ALGO TRADING
# ============================================================================

def build_features(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Build comprehensive feature set for algorithmic trading.

    Supports:
    - Mean reversion features (z-scores, BB, RSI, VWAP)
    - Volatility/breakout features (ATR, range, session breakouts)
    - Trend features (EMAs, slopes, ADX-proxy, higher timeframe)

    Args:
        df: OHLCV dataframe with DatetimeIndex
        symbol: Symbol name
        timeframe: Timeframe string

    Returns:
        DataFrame with original OHLCV + features
    """
    df = df.copy()

    # === PRICE AND RETURNS ===
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Multi-horizon returns
    for h in RETURN_HORIZONS:
        df[f'ret_{h}'] = df['close'].pct_change(h)

    # === VOLATILITY ===
    for w in VOLATILITY_WINDOWS:
        df[f'vol_{w}'] = df['returns'].rolling(w).std()

    # ATR (True Range)
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df['atr14'] = tr.rolling(14).mean()
    df['atr20'] = tr.rolling(20).mean()
    df['atr50'] = tr.rolling(50).mean()

    # === CANDLE FEATURES ===
    candle_range = high - low
    body = abs(df['close'] - df['open'])
    upper_wick = high - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - low

    df['range'] = candle_range
    df['body_pct'] = (body / (candle_range + 1e-10)).fillna(0)
    df['upper_wick_pct'] = (upper_wick / (candle_range + 1e-10)).fillna(0)
    df['lower_wick_pct'] = (lower_wick / (candle_range + 1e-10)).fillna(0)
    df['gap'] = df['open'] - close.shift(1)

    # === TREND / MOMENTUM FEATURES ===
    # EMAs
    for period in EMA_PERIODS:
        ema = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema{period}'] = ema
        df[f'ema{period}_slope'] = (ema - ema.shift(5)) / (ema.shift(5).abs() + 1e-10)
        df[f'price_vs_ema{period}'] = (df['close'] - ema) / (ema + 1e-10)

    # MACD-style
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Trend strength proxy (like ADX)
    df['trend_strength'] = (ema12 - ema26).abs() / (df['atr14'] + 1e-10)

    # === MEAN REVERSION FEATURES ===
    # Z-score vs rolling mean
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['zscore_20'] = (df['close'] - sma20) / (std20 + 1e-10)

    # Bollinger Bands
    df['bb_mid'] = sma20
    df['bb_upper'] = sma20 + (2 * std20)
    df['bb_lower'] = sma20 - (2 * std20)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-10)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi14'] = 100 - (100 / (1 + rs))

    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # VWAP proxy (session rolling VWAP)
    df['vwap_proxy'] = (df['close'] * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)
    df['dist_from_vwap'] = (df['close'] - df['vwap_proxy']) / (df['atr14'] + 1e-10)

    # === VOLATILITY / BREAKOUT FEATURES ===
    # Range vs historical
    median_range_20 = df['range'].rolling(20).median()
    df['range_vs_median'] = df['range'] / (median_range_20 + 1e-10)

    # ATR expansion
    df['atr_expansion'] = df['atr14'] / (df['atr50'] + 1e-10)

    # Breakout flags
    df['breakout_high_10'] = (df['close'] > df['high'].shift(1).rolling(10).max()).astype(int)
    df['breakout_low_10'] = (df['close'] < df['low'].shift(1).rolling(10).min()).astype(int)

    # Distance to recent high/low
    high_50 = df['high'].rolling(50).max()
    low_50 = df['low'].rolling(50).min()
    df['dist_to_high_50'] = (high_50 - df['close']) / (df['atr14'] + 1e-10)
    df['dist_to_low_50'] = (df['close'] - low_50) / (df['atr14'] + 1e-10)

    # === TIME-BASED FEATURES ===
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute

    # Session flags (London: 7-13 UTC, NY: 13-21 UTC)
    df['london_session'] = ((df['hour'] >= 7) & (df['hour'] < 13)).astype(int)
    df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)

    # Opening hour flags (first hour of each session)
    df['london_open_hour'] = ((df['hour'] >= 7) & (df['hour'] < 8)).astype(int)
    df['ny_open_hour'] = ((df['hour'] >= 13) & (df['hour'] < 14)).astype(int)

    # === VOLUME FEATURES ===
    volume_sma20 = df['volume'].rolling(20).mean()
    volume_std20 = df['volume'].rolling(20).std()
    df['volume_ratio'] = df['volume'] / (volume_sma20 + 1e-10)
    df['volume_zscore'] = (df['volume'] - volume_sma20) / (volume_std20 + 1e-10)

    # === HIGHER TIMEFRAME CONTEXT ===
    # For intraday timeframes, add hourly trend context
    if timeframe in ["5T", "15T", "30T"]:
        # Resample to 1H and merge trend indicators
        df_1h = df.resample('1H', label='left', closed='left').agg({
            'close': 'last',
            'ema20': 'last'
        }).fillna(method='ffill')

        df_1h['htf_ema20_slope'] = (df_1h['ema20'] - df_1h['ema20'].shift(1)) / (df_1h['ema20'].shift(1).abs() + 1e-10)
        df_1h['htf_trend'] = np.sign(df_1h['htf_ema20_slope'])

        # Merge back to original timeframe
        df = df.join(df_1h[['htf_trend']], how='left').fillna(method='ffill')
    else:
        df['htf_trend'] = 0

    # Drop initial NaN rows from rolling calculations
    df = df.dropna()

    return df


# ============================================================================
# LABEL GENERATION
# ============================================================================

def build_labels(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Create classification labels based on future price movement.

    Labels:
        0 = FLAT (small move, -thr < ret < +thr)
        1 = LONG (strong up move, ret >= +thr)
        2 = SHORT (strong down move, ret <= -thr)

    Thresholds are scaled by ATR to adapt to volatility.

    Args:
        df: Feature dataframe
        symbol: Symbol name
        timeframe: Timeframe string

    Returns:
        DataFrame with added 'target', 'future_return', 'future_r' columns
    """
    df = df.copy()

    # Determine forward horizon
    horizon = LABEL_HORIZONS.get(timeframe, 3)

    # Calculate future return
    future_close = df['close'].shift(-horizon)
    future_return = (future_close - df['close']) / df['close']

    # Adaptive threshold based on recent ATR
    atr = df['atr14']
    close = df['close']

    # Threshold in percentage terms
    threshold_pct = (LONG_THRESHOLD_ATR * atr) / close

    # Create labels
    labels = np.zeros(len(df), dtype=int)

    # LONG: future return >= +threshold
    labels[future_return >= threshold_pct] = 1

    # SHORT: future return <= -threshold
    labels[future_return <= -threshold_pct] = 2

    # FLAT: everything else (stays 0)

    df['target'] = labels
    df['future_return'] = future_return

    # Calculate future R-multiple (for ranking)
    # Assume stop at SL_ATR_MULTIPLE * ATR
    stop_distance = SL_ATR_MULTIPLE * atr / close
    df['future_r'] = future_return / stop_distance

    # Remove last N bars (no future data)
    df = df.iloc[:-horizon]

    # Class balance check
    label_counts = df['target'].value_counts().sort_index()
    total = len(df)

    print(f"\n📊 Label distribution for {symbol} {timeframe}:")
    for label, count in label_counts.items():
        label_name = ['FLAT', 'LONG', 'SHORT'][label]
        pct = count / total * 100
        print(f"   {label_name} ({label}): {count:,} ({pct:.1f}%)")

    return df


# ============================================================================
# TEMPORAL TRAIN/TEST SPLIT
# ============================================================================

def temporal_split(df: pd.DataFrame, train_pct: float = 0.7, val_pct: float = 0.15):
    """
    Split data chronologically into train/val/test.

    Args:
        df: Full dataset with DatetimeIndex
        train_pct: Fraction for training (default 70%)
        val_pct: Fraction for validation (default 15%)

    Returns:
        train_df, val_df, test_df
    """
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"\n⏰ Temporal split:")
    print(f"   Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df):,} bars)")
    print(f"   Val:   {val_df.index[0]} to {val_df.index[-1]} ({len(val_df):,} bars)")
    print(f"   Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df):,} bars)")

    return train_df, val_df, test_df


# ============================================================================
# TRADE SIMULATOR FOR EVALUATION
# ============================================================================

def simulate_trades(
    df: pd.DataFrame,
    predictions: np.ndarray,
    symbol: str,
    verbose: bool = False
) -> Dict:
    """
    Simulate trades based on model predictions with realistic costs.

    Args:
        df: Test dataframe with OHLCV and features
        predictions: Model predictions (0=flat, 1=long, 2=short)
        symbol: Symbol name
        verbose: Print trade details

    Returns:
        Dict with trade metrics
    """
    costs = TRADING_COSTS[symbol]
    spread_cost = costs['spread'] * costs['point_value']
    commission = costs['commission']
    point_value = costs['point_value']

    trades = []
    position = None  # Current position: None, 'long', or 'short'
    entry_price = None
    entry_time = None
    tp_price = None
    sl_price = None

    for i in range(len(df)):
        row = df.iloc[i]
        pred = predictions[i]
        current_time = row.name
        current_price = row['close']
        atr = row['atr14']

        # Exit logic (if in position)
        if position is not None:
            hit_tp = False
            hit_sl = False
            exit_price = None
            exit_reason = None

            if position == 'long':
                if row['high'] >= tp_price:
                    hit_tp = True
                    exit_price = tp_price
                    exit_reason = 'TP'
                elif row['low'] <= sl_price:
                    hit_sl = True
                    exit_price = sl_price
                    exit_reason = 'SL'
                elif pred != 1:  # Signal changed
                    exit_price = current_price
                    exit_reason = 'SIGNAL'

            elif position == 'short':
                if row['low'] <= tp_price:
                    hit_tp = True
                    exit_price = tp_price
                    exit_reason = 'TP'
                elif row['high'] >= sl_price:
                    hit_sl = True
                    exit_price = sl_price
                    exit_reason = 'SL'
                elif pred != 2:  # Signal changed
                    exit_price = current_price
                    exit_reason = 'SIGNAL'

            # Close position if exit triggered
            if exit_price is not None:
                if position == 'long':
                    gross_pnl = (exit_price - entry_price) * point_value
                else:  # short
                    gross_pnl = (entry_price - exit_price) * point_value

                # Subtract costs
                net_pnl = gross_pnl - spread_cost - commission

                # Calculate R-multiple
                risk = SL_ATR_MULTIPLE * atr * point_value
                r_multiple = net_pnl / risk if risk > 0 else 0

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'direction': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'r_multiple': r_multiple
                })

                if verbose and len(trades) <= 10:
                    sign = '+' if net_pnl >= 0 else ''
                    print(f"   {position.upper()} {entry_time} -> {current_time}: {sign}${net_pnl:.2f} ({exit_reason})")

                # Reset position
                position = None
                entry_price = None
                tp_price = None
                sl_price = None

        # Entry logic (if flat)
        if position is None:
            if pred == 1:  # Long signal
                position = 'long'
                entry_price = current_price
                entry_time = current_time
                tp_price = entry_price + (TP_ATR_MULTIPLE * atr)
                sl_price = entry_price - (SL_ATR_MULTIPLE * atr)

            elif pred == 2:  # Short signal
                position = 'short'
                entry_price = current_price
                entry_time = current_time
                tp_price = entry_price - (TP_ATR_MULTIPLE * atr)
                sl_price = entry_price + (SL_ATR_MULTIPLE * atr)

    # Close any remaining position at end
    if position is not None:
        exit_price = df.iloc[-1]['close']
        if position == 'long':
            gross_pnl = (exit_price - entry_price) * point_value
        else:
            gross_pnl = (entry_price - exit_price) * point_value

        net_pnl = gross_pnl - spread_cost - commission
        atr = df.iloc[-1]['atr14']
        risk = SL_ATR_MULTIPLE * atr * point_value
        r_multiple = net_pnl / risk if risk > 0 else 0

        trades.append({
            'entry_time': entry_time,
            'exit_time': df.index[-1],
            'direction': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': 'EOD',
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'r_multiple': r_multiple
        })

    # Calculate metrics
    if not trades:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'expectancy': 0,
            'sharpe': 0,
            'total_pnl': 0
        }

    trades_df = pd.DataFrame(trades)

    wins = trades_df[trades_df['net_pnl'] > 0]
    losses = trades_df[trades_df['net_pnl'] <= 0]

    num_trades = len(trades_df)
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = num_wins / num_trades if num_trades > 0 else 0

    total_wins = wins['net_pnl'].sum() if len(wins) > 0 else 0
    total_losses = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0

    # Drawdown
    cumulative = trades_df['net_pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

    # Expectancy (average R per trade)
    expectancy = trades_df['r_multiple'].mean()

    # Sharpe-like (mean / std of returns)
    if trades_df['net_pnl'].std() > 0:
        sharpe = trades_df['net_pnl'].mean() / trades_df['net_pnl'].std()
    else:
        sharpe = 0

    total_pnl = trades_df['net_pnl'].sum()

    return {
        'num_trades': num_trades,
        'num_wins': num_wins,
        'num_losses': num_losses,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'expectancy': expectancy,
        'sharpe': sharpe,
        'total_pnl': total_pnl
    }


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model_for(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame
) -> Optional[Dict]:
    """
    Train XGBoost model for a specific symbol/timeframe.

    Args:
        symbol: Symbol name
        timeframe: Timeframe string
        df: Full feature + label dataframe

    Returns:
        Dict with model and metadata, or None if model fails validation
    """
    print(f"\n{'='*80}")
    print(f"TRAINING: {symbol} {timeframe}")
    print(f"{'='*80}")

    # Temporal split
    train_df, val_df, test_df = temporal_split(df)

    # Get feature columns (exclude OHLCV, target, and helper columns)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'future_return', 'future_r']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\n🔧 Features: {len(feature_cols)} columns")

    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values

    # Train XGBoost
    print(f"\n🔥 Training XGBoost...")

    # Update params for newer XGBoost API
    xgb_params = XGB_PARAMS.copy()
    xgb_params['early_stopping_rounds'] = 30
    xgb_params['eval_metric'] = 'mlogloss'

    model = xgb.XGBClassifier(**xgb_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Validation metrics
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)

    val_acc = (y_val_pred == y_val).mean()

    # ROC AUC (one-vs-rest for multiclass)
    try:
        val_auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr', average='weighted')
    except:
        val_auc = 0.0

    print(f"\n✅ Validation Accuracy: {val_acc:.1%}")
    print(f"✅ Validation AUC: {val_auc:.3f}")

    # Test set predictions
    y_test_pred = model.predict(X_test)
    test_acc = (y_test_pred == y_test).mean()

    print(f"✅ Test Accuracy: {test_acc:.1%}")

    print("\n" + classification_report(
        y_test, y_test_pred,
        target_names=['FLAT', 'LONG', 'SHORT'],
        zero_division=0
    ))

    # === TRADING SIMULATION ON TEST SET ===
    print(f"\n💼 Simulating trades on test set...")

    metrics = simulate_trades(test_df, y_test_pred, symbol, verbose=True)

    # Signal distribution on test set
    signal_counts = pd.Series(y_test_pred).value_counts().sort_index()
    total_sigs = len(y_test_pred)

    print(f"\n📊 SIGNAL DISTRIBUTION (Test Set):")
    for sig, count in signal_counts.items():
        sig_name = ['FLAT', 'LONG', 'SHORT'][sig]
        pct = count / total_sigs * 100
        print(f"   {sig_name}: {count:,} ({pct:.1f}%)")

    print(f"\n📊 TRADING METRICS:")
    print(f"   Total trades: {metrics['num_trades']}")
    print(f"   Wins: {metrics['num_wins']} | Losses: {metrics['num_losses']}")
    print(f"   Win rate: {metrics['win_rate']:.1%}")
    print(f"   Profit factor: {metrics['profit_factor']:.2f}")
    print(f"   Avg win: ${metrics['avg_win']:.2f} | Avg loss: ${metrics['avg_loss']:.2f}")
    print(f"   Max drawdown: ${metrics['max_drawdown']:.2f}")
    print(f"   Expectancy (avg R): {metrics['expectancy']:.2f}")
    print(f"   Sharpe-like: {metrics['sharpe']:.2f}")
    print(f"   Total PnL: ${metrics['total_pnl']:.2f}")

    # === VALIDATION: APPLY THRESHOLDS ===
    print(f"\n🎯 Validation against thresholds:")

    passed = True
    reasons = []

    if metrics['num_trades'] < MIN_TRADES_TEST:
        passed = False
        reasons.append(f"Too few trades ({metrics['num_trades']} < {MIN_TRADES_TEST})")
        print(f"   ❌ Too few trades: {metrics['num_trades']} < {MIN_TRADES_TEST}")
    else:
        print(f"   ✅ Sufficient trades: {metrics['num_trades']}")

    if metrics['win_rate'] < MIN_WIN_RATE:
        passed = False
        reasons.append(f"Win rate too low ({metrics['win_rate']:.1%} < {MIN_WIN_RATE:.1%})")
        print(f"   ❌ Win rate too low: {metrics['win_rate']:.1%} < {MIN_WIN_RATE:.1%}")
    else:
        print(f"   ✅ Win rate acceptable: {metrics['win_rate']:.1%}")

    if metrics['profit_factor'] < MIN_PROFIT_FACTOR:
        passed = False
        reasons.append(f"Profit factor too low ({metrics['profit_factor']:.2f} < {MIN_PROFIT_FACTOR})")
        print(f"   ❌ Profit factor too low: {metrics['profit_factor']:.2f} < {MIN_PROFIT_FACTOR}")
    else:
        print(f"   ✅ Profit factor acceptable: {metrics['profit_factor']:.2f}")

    # Drawdown as percentage of starting capital (assume $25k)
    dd_pct = metrics['max_drawdown'] / 25000
    if dd_pct > MAX_DRAWDOWN_PCT:
        passed = False
        reasons.append(f"Drawdown too high ({dd_pct:.1%} > {MAX_DRAWDOWN_PCT:.1%})")
        print(f"   ❌ Drawdown too high: {dd_pct:.1%} > {MAX_DRAWDOWN_PCT:.1%}")
    else:
        print(f"   ✅ Drawdown acceptable: {dd_pct:.1%}")

    if metrics['expectancy'] < MIN_EXPECTANCY:
        passed = False
        reasons.append(f"Expectancy too low ({metrics['expectancy']:.2f} < {MIN_EXPECTANCY})")
        print(f"   ❌ Expectancy too low: {metrics['expectancy']:.2f} < {MIN_EXPECTANCY}")
    else:
        print(f"   ✅ Expectancy acceptable: {metrics['expectancy']:.2f}")

    if not passed:
        print(f"\n❌ MODEL REJECTED - {symbol} {timeframe}")
        print(f"   Reasons: {'; '.join(reasons)}")
        print(f"   This model will NOT be saved.")
        return None

    print(f"\n✅ MODEL PASSED ALL THRESHOLDS - {symbol} {timeframe}")

    # Return model package
    return {
        'model': model,
        'features': feature_cols,
        'symbol': symbol,
        'timeframe': timeframe,
        'metrics': metrics,
        'test_accuracy': test_acc,
        'val_accuracy': val_acc,
        'val_auc': val_auc,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'train_period': f"{train_df.index[0]} to {train_df.index[-1]}",
        'test_period': f"{test_df.index[0]} to {test_df.index[-1]}"
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_all_models():
    """
    Main function: Train models for all symbol/timeframe combinations.
    """
    print("\n" + "="*80)
    print("CITADEL METALS TRAINING SYSTEM")
    print("="*80)
    print(f"\nSymbols: {SYMBOLS}")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"\nModel thresholds:")
    print(f"  Min trades: {MIN_TRADES_TEST}")
    print(f"  Min win rate: {MIN_WIN_RATE:.1%}")
    print(f"  Min profit factor: {MIN_PROFIT_FACTOR}")
    print(f"  Max drawdown: {MAX_DRAWDOWN_PCT:.1%}")
    print(f"  Min expectancy: {MIN_EXPECTANCY}")

    MODELS_DIR.mkdir(exist_ok=True)

    results_summary = []

    for symbol in SYMBOLS:
        print(f"\n{'='*80}")
        print(f"PROCESSING SYMBOL: {symbol}")
        print(f"{'='*80}")

        # Load base data
        try:
            base_df = load_data(symbol)
        except Exception as e:
            print(f"❌ Failed to load data for {symbol}: {e}")
            continue

        # Resample to all timeframes
        resampled_data = resample_to_timeframes(base_df, TIMEFRAMES)

        for timeframe in TIMEFRAMES:
            try:
                df = resampled_data[timeframe]

                # Build features
                print(f"\n🔧 Building features for {symbol} {timeframe}...")
                df = build_features(df, symbol, timeframe)
                print(f"✅ Features built: {len(df.columns)} columns, {len(df):,} bars")

                # Build labels
                print(f"\n🏷️  Creating labels...")
                df = build_labels(df, symbol, timeframe)
                print(f"✅ Labels created: {len(df):,} samples")

                # Train model
                model_package = train_model_for(symbol, timeframe, df)

                if model_package is not None:
                    # Save model
                    model_dir = MODELS_DIR / symbol
                    model_dir.mkdir(exist_ok=True)

                    model_path = model_dir / f"{symbol}_{timeframe}.pkl"
                    joblib.dump(model_package, model_path)
                    print(f"\n💾 Saved model: {model_path}")

                    results_summary.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'status': 'PASSED',
                        'num_trades': model_package['metrics']['num_trades'],
                        'win_rate': model_package['metrics']['win_rate'],
                        'profit_factor': model_package['metrics']['profit_factor'],
                        'expectancy': model_package['metrics']['expectancy'],
                        'test_accuracy': model_package['test_accuracy']
                    })
                else:
                    results_summary.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'status': 'FAILED',
                        'num_trades': 0,
                        'win_rate': 0,
                        'profit_factor': 0,
                        'expectancy': 0,
                        'test_accuracy': 0
                    })

            except Exception as e:
                print(f"\n❌ Error training {symbol} {timeframe}: {e}")
                import traceback
                traceback.print_exc()

                results_summary.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'status': 'ERROR',
                    'num_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'expectancy': 0,
                    'test_accuracy': 0
                })

    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)

    summary_df = pd.DataFrame(results_summary)

    print("\nPASSED MODELS:")
    passed = summary_df[summary_df['status'] == 'PASSED']
    if len(passed) > 0:
        print(passed.to_string(index=False))
    else:
        print("  None")

    print("\nFAILED MODELS:")
    failed = summary_df[summary_df['status'] == 'FAILED']
    if len(failed) > 0:
        print(failed[['symbol', 'timeframe']].to_string(index=False))
    else:
        print("  None")

    print("\nERROR MODELS:")
    errors = summary_df[summary_df['status'] == 'ERROR']
    if len(errors) > 0:
        print(errors[['symbol', 'timeframe']].to_string(index=False))
    else:
        print("  None")

    # Save summary
    summary_path = MODELS_DIR / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Saved training summary: {summary_path}")

    print(f"\n✅ Training pipeline complete!")
    print(f"   Passed: {len(passed)}/{len(summary_df)}")
    print(f"   Models saved to: {MODELS_DIR}/")


if __name__ == "__main__":
    train_all_models()
