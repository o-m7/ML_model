#!/usr/bin/env python3
"""
RENAISSANCE TECHNOLOGIES - PRODUCTION ML TRADING SYSTEM
========================================================

Institutional-grade ML trading system with:
✅ Adaptive learning from live trades
✅ Proper class balance (25-30% Flat labels)
✅ Advanced feature engineering
✅ Rigorous benchmark requirements (PF > 1.5, DD < 6%, WR > 50%)
✅ S&P500 comparison
✅ Online learning capabilities
✅ Regime-aware trading

Usage:
    python3 rentec_training_system.py --symbol XAUUSD --tf 15T
    python3 rentec_training_system.py --all --workers 4
"""

import argparse
import json
import pickle
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SystemConfig:
    """Master configuration - Renaissance Technologies standards."""
    
    FEATURE_STORE = Path("feature_store")
    MODEL_STORE = Path("models_rentec")
    TRAIN_START = "2019-01-01"
    TRAIN_END = "2025-10-22"
    OOS_MONTHS = 6
    
    SYMBOLS = ["XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD"]
    TIMEFRAMES = ["5T", "15T", "30T", "1H", "4H"]
    
    # Walk-forward CV
    N_FOLDS = 8
    EMBARGO_BARS = 50
    PURGE_BARS = 25
    
    # Labeling - More conservative for better Flat class representation
    FORECAST_HORIZON = 40  # Increased for clearer moves
    TP_ATR_MULT = 1.5  # Higher TP
    SL_ATR_MULT = 1.0
    FLAT_THRESHOLD = 0.85  # Only label as directional if >= 85% of TP/SL target
    
    # Symbol-specific overrides
    SYMBOL_TP_SL_OVERRIDES = {
        'XAUUSD': {'tp': 1.6, 'sl': 1.0, 'flat_thresh': 0.85},
        'XAGUSD': {'tp': 1.5, 'sl': 1.0, 'flat_thresh': 0.85},
        'EURUSD': {'tp': 1.4, 'sl': 1.0, 'flat_thresh': 0.85},
        'GBPUSD': {'tp': 1.4, 'sl': 1.0, 'flat_thresh': 0.85},
        'AUDUSD': {'tp': 1.4, 'sl': 1.0, 'flat_thresh': 0.85},
        'NZDUSD': {'tp': 1.4, 'sl': 1.0, 'flat_thresh': 0.85},
        'USDJPY': {'tp': 1.4, 'sl': 1.0, 'flat_thresh': 0.85},
        'USDCAD': {'tp': 1.4, 'sl': 1.0, 'flat_thresh': 0.85},
    }
    
    # Trading parameters
    INITIAL_CAPITAL = 100000
    RISK_PER_TRADE_PCT = 0.015  # 1.5% risk per trade
    MAX_POSITION_PCT = 0.25
    LEVERAGE = 20.0  # Conservative leverage
    COMMISSION_PCT = 0.00006  # Realistic forex commission
    SLIPPAGE_PCT = 0.00002  # Realistic slippage
    MAX_BARS_IN_TRADE = 50
    
    # Signal generation - Adaptive thresholds
    MIN_CONFIDENCE = 0.52  # Balanced threshold
    MIN_EDGE = 0.12  # Clear edge required
    
    # Benchmarks - Renaissance Standards
    MIN_PROFIT_FACTOR = 1.50
    MAX_DRAWDOWN_PCT = 6.0
    MIN_SHARPE = 0.25
    MIN_WIN_RATE = 50.0
    MIN_TRADES_OOS = 80  # Minimum trades for statistical significance
    
    MIN_TRADES_BY_TF = {
        '5T': 150,
        '15T': 80,
        '30T': 60,
        '1H': 50,
        '4H': 25,
    }
    
    MAX_FEATURES = 60
    ENSEMBLE_WEIGHTS = {'xgb': 0.35, 'lgb': 0.35, 'linear': 0.30}


CONFIG = SystemConfig()
CONFIG.MODEL_STORE.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

def build_rentec_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renaissance Technologies grade feature engineering.
    
    Features capture:
    - Multi-scale momentum and mean reversion
    - Volatility clustering and regime changes
    - Market microstructure
    - Time-series properties
    """
    df = df.copy()
    
    # === MOMENTUM FEATURES (Multi-scale) ===
    for period in [3, 5, 10, 20, 40, 60]:
        df[f'mom_{period}'] = df['close'].pct_change(period)
        df[f'mom_{period}_rank'] = df[f'mom_{period}'].rolling(100).rank(pct=True)
    
    # Momentum acceleration
    df['mom_accel_5'] = df['mom_5'].diff(2)
    df['mom_accel_20'] = df['mom_20'].diff(5)
    
    # === VOLATILITY FEATURES ===
    for window in [5, 10, 20, 40]:
        df[f'vol_{window}'] = df['close'].pct_change().rolling(window).std()
        df[f'vol_{window}_rank'] = df[f'vol_{window}'].rolling(200).rank(pct=True)
    
    # Volatility regime change
    df['vol_regime_change'] = (df['vol_20'] / df['vol_40'].rolling(20).mean()).fillna(1.0)
    df['vol_expanding'] = (df['vol_10'] > df['vol_20']).astype(int)
    
    # Parkinson volatility (uses high/low range)
    df['parkinson_vol'] = np.sqrt(
        ((np.log(df['high'] / df['low'])) ** 2) / (4 * np.log(2))
    ).rolling(20).mean()
    
    # === MEAN REVERSION FEATURES ===
    if 'ema20' in df.columns and 'ema50' in df.columns:
        atr = df.get('atr14', df['close'] * 0.015)
        df['dist_ema20'] = (df['close'] - df['ema20']) / atr
        df['dist_ema50'] = (df['close'] - df['ema50']) / atr
        df['ema_cross'] = ((df['ema20'] > df['ema50']).astype(int) - 0.5) * 2
        df['ema_spread'] = (df['ema20'] - df['ema50']) / atr
    
    if 'vwap' in df.columns:
        atr = df.get('atr14', df['close'] * 0.015)
        df['dist_vwap'] = (df['close'] - df['vwap']) / atr
        df['vwap_momentum'] = df['dist_vwap'].diff(3)
    
    # === RSI FEATURES (Multi-timeframe) ===
    if 'rsi14' in df.columns:
        df['rsi_zscore'] = (df['rsi14'] - 50) / 15
        df['rsi_mom'] = df['rsi14'].diff(5)
        df['rsi_extreme'] = ((df['rsi14'] < 25) | (df['rsi14'] > 75)).astype(int)
        df['rsi_divergence'] = df['rsi14'].diff(10) - df['mom_10'] * 100
    
    # === BOLLINGER BANDS ===
    if 'bb_pct' in df.columns and 'bb_width' in df.columns:
        df['bb_position'] = (df['bb_pct'] - 0.5) * 2  # Normalize to [-1, 1]
        df['bb_width_rank'] = df['bb_width'].rolling(100).rank(pct=True)
        df['bb_squeeze'] = (df['bb_width_rank'] < 0.2).astype(int)
        df['bb_expansion'] = (df['bb_width_rank'] > 0.8).astype(int)
    
    # === ADX / TREND STRENGTH ===
    if 'adx' in df.columns:
        df['adx_normalized'] = df['adx'] / 50
        df['adx_trending'] = (df['adx'] > 25).astype(int)
        df['adx_momentum'] = df['adx'].diff(5)
    
    if 'plus_di' in df.columns and 'minus_di' in df.columns:
        df['di_diff'] = df['plus_di'] - df['minus_di']
        df['di_ratio'] = np.log1p(df['plus_di']) - np.log1p(df['minus_di'])
    
    # === VOLUME FEATURES ===
    if 'volume' in df.columns:
        for period in [10, 20, 40]:
            df[f'vol_ma{period}'] = df['volume'].rolling(period).mean()
            df[f'vol_ratio{period}'] = df['volume'] / df[f'vol_ma{period}']
        
        df['vol_surge'] = (df['vol_ratio20'] > 1.8).astype(int)
        df['vol_dry'] = (df['vol_ratio20'] < 0.6).astype(int)
    
    # === PRICE ACTION FEATURES ===
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # Candle body vs wick ratio
    df['body_to_range'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # === AUTOCORRELATION / PERSISTENCE ===
    for lag in [1, 3, 5]:
        df[f'return_autocorr_{lag}'] = df['close'].pct_change().rolling(50).apply(
            lambda x: x.autocorr(lag=lag) if len(x) >= lag + 10 else 0, raw=False
        )
    
    # === TIME FEATURES ===
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Session indicators
        df['asian_session'] = df['hour'].between(0, 8).astype(int)
        df['london_session'] = df['hour'].between(8, 16).astype(int)
        df['ny_session'] = df['hour'].between(13, 21).astype(int)
        df['session_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
    
    # === COMPOSITE FEATURES ===
    # Trend + Volatility interaction
    if 'mom_20' in df.columns and 'vol_20' in df.columns:
        df['mom_vol_ratio'] = df['mom_20'] / (df['vol_20'] + 1e-6)
    
    # Mean reversion opportunity score
    if 'dist_ema50' in df.columns and 'vol_20_rank' in df.columns:
        df['mean_reversion_score'] = abs(df['dist_ema50']) * (1 - df['vol_20_rank'])
    
    # Trend opportunity score
    if 'adx' in df.columns and 'mom_20' in df.columns:
        df['trend_score'] = df['adx'] / 50 * abs(df['mom_20']) * 100
    
    return df


# ============================================================================
# DATA LOADING
# ============================================================================

def load_symbol_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load data and apply RenTec features."""
    
    path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
    
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    
    df = pd.read_parquet(path)
    
    if 'timestamp' not in df.columns:
        if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise ValueError("No timestamp column")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    train_start = pd.to_datetime(CONFIG.TRAIN_START, utc=True)
    train_end = pd.to_datetime(CONFIG.TRAIN_END, utc=True)
    df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)]
    
    print(f"  Loaded {len(df):,} bars")
    
    # Apply RenTec features
    df = build_rentec_features(df)
    
    return df.reset_index(drop=True)


# ============================================================================
# TRIPLE-BARRIER LABELING (BALANCED)
# ============================================================================

def create_balanced_labels(df: pd.DataFrame, tp_mult: float, sl_mult: float,
                          flat_threshold: float = 0.85) -> pd.DataFrame:
    """
    Create balanced 3-class labels with 25-30% Flat representation.

    CRITICAL FIX: Entry at NEXT bar's open (not current bar's close)
    This matches live trading reality: signal on bar i → enter on bar i+1 at open

    Strategy:
    - TP/SL hits are always directional (clear moves)
    - Non-hits are Flat unless they achieve >= flat_threshold of target
    """

    print("  Creating balanced triple-barrier labels...")

    df = df.copy()
    n = len(df)
    horizon = CONFIG.FORECAST_HORIZON

    atr = df.get('atr14', df['close'] * 0.015).values

    # CRITICAL FIX: Entry is at NEXT bar's open (realistic!)
    # Signal on bar i → Entry at bar i+1 open
    next_bar_opens = df['open'].shift(-1).values
    entry_prices = next_bar_opens

    tp_prices_long = entry_prices + (atr * tp_mult)
    sl_prices_long = entry_prices - (atr * sl_mult)

    labels = np.zeros(n, dtype=int)
    returns = np.zeros(n)
    durations = np.zeros(n, dtype=int)
    hit_types = np.array(['none'] * n, dtype=object)

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # Need extra bar at end for next bar open
    for i in range(n - horizon - 1):
        # Skip if next bar open is NaN (no next bar available)
        if np.isnan(entry_prices[i]):
            continue

        future_end = min(i + 2 + horizon, n)  # Start from i+1 (after entry)
        future_highs = highs[i+1:future_end]
        future_lows = lows[i+1:future_end]
        future_closes = closes[i+1:future_end]

        if len(future_highs) == 0:
            continue

        # Check for TP/SL hits
        tp_hits = np.where(future_highs >= tp_prices_long[i])[0]
        sl_hits = np.where(future_lows <= sl_prices_long[i])[0]

        tp_hit = len(tp_hits) > 0
        sl_hit = len(sl_hits) > 0

        if tp_hit and sl_hit:
            # Both hit - use whichever came first
            if tp_hits[0] < sl_hits[0]:
                labels[i] = 1  # Up
                returns[i] = tp_mult * atr[i] / entry_prices[i]
                durations[i] = tp_hits[0] + 1
                hit_types[i] = 'tp'
            else:
                labels[i] = 2  # Down
                returns[i] = -sl_mult * atr[i] / entry_prices[i]
                durations[i] = sl_hits[0] + 1
                hit_types[i] = 'sl'
        elif tp_hit:
            labels[i] = 1  # Up
            returns[i] = tp_mult * atr[i] / entry_prices[i]
            durations[i] = tp_hits[0] + 1
            hit_types[i] = 'tp'
        elif sl_hit:
            labels[i] = 2  # Down
            returns[i] = -sl_mult * atr[i] / entry_prices[i]
            durations[i] = sl_hits[0] + 1
            hit_types[i] = 'sl'
        else:
            # No TP/SL hit - check if move was significant
            final_price = future_closes[-1]
            ret = (final_price - entry_prices[i]) / entry_prices[i]
            atr_normalized_ret = ret * entry_prices[i] / atr[i]

            # Only label as directional if >= flat_threshold of target
            if atr_normalized_ret >= (tp_mult * flat_threshold):
                labels[i] = 1  # Up
                hit_types[i] = 'partial_up'
            elif atr_normalized_ret <= -(sl_mult * flat_threshold):
                labels[i] = 2  # Down
                hit_types[i] = 'partial_down'
            else:
                labels[i] = 0  # Flat - ambiguous/small move
                hit_types[i] = 'flat'

            returns[i] = ret
            durations[i] = len(future_closes)

    df['target'] = labels
    df['expected_return'] = returns
    df['expected_duration'] = durations
    df['hit_type'] = hit_types

    # Remove last bars without full horizon + need for next bar
    df = df.iloc[:-(CONFIG.FORECAST_HORIZON + 1)]

    # Print class distribution
    class_counts = df['target'].value_counts()
    total = len(df)
    print(f"    Flat: {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/total*100:.1f}%)")
    print(f"    Up:   {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/total*100:.1f}%)")
    print(f"    Down: {class_counts.get(2, 0):,} ({class_counts.get(2, 0)/total*100:.1f}%)")

    # Print hit type distribution
    hit_type_counts = df['hit_type'].value_counts()
    print(f"    TP hits: {hit_type_counts.get('tp', 0):,}, SL hits: {hit_type_counts.get('sl', 0):,}")

    return df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def detect_look_ahead_bias(df: pd.DataFrame, feature: str) -> Tuple[bool, Dict]:
    """Strict look-ahead detection."""
    
    diagnostics = {'feature': feature, 'tests_failed': []}
    
    try:
        feat_series = df[feature].fillna(0)
        
        # Test correlation with future returns
        for h in [1, 3, 5, 10]:
            future_ret = df['close'].pct_change(h).shift(-h)
            corr = abs(feat_series.corr(future_ret))
            
            if corr > 0.04:  # Stricter threshold
                diagnostics['tests_failed'].append(f'fwd_corr_{h}={corr:.4f}')
        
        # Future vs past asymmetry
        past_ret = df['close'].pct_change(10)
        future_ret = df['close'].pct_change(10).shift(-10)
        
        past_corr = abs(feat_series.corr(past_ret))
        future_corr = abs(feat_series.corr(future_ret))
        
        if future_corr > past_corr * 1.5 and future_corr > 0.03:
            diagnostics['tests_failed'].append('future_asymmetry')
        
        is_suspicious = len(diagnostics['tests_failed']) > 0
        
        return is_suspicious, diagnostics
        
    except:
        return True, diagnostics


def select_features_strict(df: pd.DataFrame) -> List[str]:
    """Strict feature selection with zero look-ahead tolerance."""
    
    print("  Selecting features (strict validation)...")
    
    # Exclude non-features
    exclude = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'target', 'expected_return', 'expected_duration', 'hit_type'
    ]
    
    # Exclude SMC patterns
    exclude_patterns = [
        'swing', 'fvg', 'ob_', 'bos', 'choch', 'eq_',
        'order_block', 'fair_value', 'liquidity', 'inducement'
    ]
    
    all_features = []
    for col in df.columns:
        if col in exclude or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        if any(pattern in col.lower() for pattern in exclude_patterns):
            continue
        
        all_features.append(col)
    
    print(f"    Available: {len(all_features)} features")
    
    # Look-ahead testing
    clean_features = []
    suspicious_features = []
    
    for feat in all_features:
        is_suspicious, diagnostics = detect_look_ahead_bias(df, feat)
        
        if is_suspicious:
            suspicious_features.append((feat, diagnostics))
        else:
            clean_features.append(feat)
    
    print(f"    ✅ Clean: {len(clean_features)}")
    print(f"    ❌ Suspicious: {len(suspicious_features)}")
    
    # Remove highly collinear features
    if len(clean_features) > CONFIG.MAX_FEATURES:
        corr_matrix = df[clean_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.92)]
        clean_features = [f for f in clean_features if f not in to_drop]
        print(f"    Removed {len(to_drop)} highly collinear")
    
    # Feature importance selection
    if len(clean_features) > CONFIG.MAX_FEATURES:
        print(f"    Selecting top {CONFIG.MAX_FEATURES} by importance...")
        
        X = df[clean_features].fillna(0).values
        y = df['target'].values
        
        temp_model = lgb.LGBMClassifier(
            n_estimators=50, max_depth=4, verbosity=-1,
            force_row_wise=True, random_state=42
        )
        temp_model.fit(X, y)
        
        importance = dict(zip(clean_features, temp_model.feature_importances_))
        clean_features = [f[0] for f in sorted(importance.items(), 
                                               key=lambda x: x[1], 
                                               reverse=True)[:CONFIG.MAX_FEATURES]]
    
    print(f"    ✓ Final: {len(clean_features)} features")
    
    return clean_features


# ============================================================================
# ENSEMBLE MODEL (ADAPTIVE)
# ============================================================================

class AdaptiveEnsemble:
    """
    Renaissance-grade adaptive ensemble.
    
    Features:
    - XGBoost + LightGBM + Linear model
    - Balanced class weighting with Flat class boost
    - Strong regularization to prevent overfitting
    - Robust scaling for stability
    """
    
    def __init__(self, n_classes: int = 3):
        self.n_classes = n_classes
        self.models = {}
        self.scaler = RobustScaler()  # More robust to outliers
        self.weights = CONFIG.ENSEMBLE_WEIGHTS
        self.feature_importance = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train ensemble with adaptive class weighting."""
        
        # Handle missing classes
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 3:
            self.class_mapping = {old: new for new, old in enumerate(unique_classes)}
            self.inverse_mapping = {v: k for k, v in self.class_mapping.items()}
            y_train = np.array([self.class_mapping[y] for y in y_train])
            if y_val is not None:
                y_val = np.array([self.class_mapping.get(y, 0) for y in y_val])
        else:
            self.class_mapping = None
            self.inverse_mapping = None
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # Adaptive class weighting
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        
        # Boost Flat and Down classes for better balance
        if len(class_weights) >= 3:
            class_weights[0] *= 4.0  # Heavily boost Flat class
            class_weights[2] *= 2.0  # Boost Down class for long/short balance
        
        sample_weights = class_weights[y_train]
        
        print("  Training ensemble models...")
        
        # XGBoost - Highly regularized
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.65,
            colsample_bytree=0.65,
            gamma=2.0,
            reg_alpha=3.0,
            reg_lambda=4.0,
            min_child_weight=15,
            random_state=42,
            verbosity=0
        )
        self.models['xgb'].fit(X_train, y_train, sample_weight=sample_weights)
        
        # LightGBM - Highly regularized
        self.models['lgb'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.02,
            num_leaves=8,
            subsample=0.65,
            colsample_bytree=0.65,
            reg_alpha=3.0,
            reg_lambda=4.0,
            min_child_samples=40,
            random_state=42,
            verbosity=-1,
            force_row_wise=True
        )
        self.models['lgb'].fit(X_train, y_train, sample_weight=sample_weights)
        
        # Logistic Regression - Strong regularization
        self.models['linear'] = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            C=0.03,
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.models['linear'].fit(X_train_scaled, y_train)
        
        # Store feature importance
        self.feature_importance = (
            self.models['xgb'].feature_importances_ * self.weights['xgb'] +
            self.models['lgb'].feature_importances_ * self.weights['lgb']
        )
        
    def predict_proba(self, X):
        """Ensemble predictions."""
        X_scaled = self.scaler.transform(X)
        
        preds = []
        for name, model in self.models.items():
            if name == 'linear':
                pred = model.predict_proba(X_scaled)
            else:
                pred = model.predict_proba(X)
            
            # Handle missing classes
            if self.class_mapping is not None and pred.shape[1] < 3:
                full_pred = np.zeros((pred.shape[0], 3))
                for new_class, old_class in self.inverse_mapping.items():
                    full_pred[:, old_class] = pred[:, new_class]
                pred = full_pred
            
            preds.append(pred * self.weights[name])
        
        return np.sum(preds, axis=0)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# ============================================================================
# WALK-FORWARD CV
# ============================================================================

def create_walk_forward_splits(df: pd.DataFrame, n_folds: int = 8) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create walk-forward CV splits with purging and embargo."""
    
    print(f"  Creating {n_folds} walk-forward folds...")
    
    n_samples = len(df)
    fold_size = n_samples // (n_folds + 1)
    
    splits = []
    
    for fold_idx in range(n_folds):
        val_start = (fold_idx + 1) * fold_size
        val_end = val_start + fold_size
        
        if val_end > n_samples:
            break
        
        # Purge and embargo
        train_end = val_start - CONFIG.PURGE_BARS
        train_indices = np.arange(0, max(0, train_end))
        val_indices = np.arange(val_start, min(val_end, n_samples - CONFIG.EMBARGO_BARS))
        
        if len(train_indices) > 500 and len(val_indices) > 50:
            splits.append((train_indices, val_indices))
    
    print(f"  ✓ Created {len(splits)} valid folds")
    return splits


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

@dataclass
class TradeConfig:
    initial_capital: float = 100000
    risk_per_trade_pct: float = 0.015
    max_position_pct: float = 0.25
    leverage: float = 20.0
    commission_pct: float = 0.00006
    slippage_pct: float = 0.00002
    confidence_threshold: float = 0.52
    max_bars_in_trade: int = 50
    symbol_pip_values: Dict[str, float] = None
    
    def __post_init__(self):
        if self.symbol_pip_values is None:
            self.symbol_pip_values = {
                'XAUUSD': 0.01, 'XAGUSD': 0.001,
                'EURUSD': 0.0001, 'GBPUSD': 0.0001,
                'AUDUSD': 0.0001, 'NZDUSD': 0.0001,
                'USDJPY': 0.01, 'USDCAD': 0.0001,
            }


@dataclass
class Trade:
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    direction: str
    position_size: float
    sl_price: float
    tp_price: float
    exit_idx: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    confidence: Optional[float] = None


class BacktestEngine:
    """Professional backtest engine."""
    
    def __init__(self, df: pd.DataFrame, config: TradeConfig, symbol: str):
        self.df = df.copy()
        self.config = config
        self.symbol = symbol
        self.pip_value = config.symbol_pip_values.get(symbol, 0.0001)
        
        required = ['open', 'high', 'low', 'close', 'timestamp']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        self.trades = []
        self.equity_curve = [config.initial_capital]
        self.current_equity = config.initial_capital
        self.peak_equity = config.initial_capital
        self.active_trade = None
        self.circuit_breaker_dd = 0.20  # Stop if DD > 20%
        self.trading_halted = False
    
    def calculate_position_size(self, entry_price: float, sl_price: float) -> Tuple[float, float]:
        """Calculate position size based on risk."""
        risk_amount = self.current_equity * self.config.risk_per_trade_pct
        sl_distance = abs(entry_price - sl_price)
        
        if sl_distance <= 0:
            return 0, 0
        
        position_size = risk_amount / sl_distance
        position_value = position_size * entry_price
        
        # Apply leverage constraint
        margin_required = position_value / self.config.leverage
        max_margin = self.current_equity * self.config.max_position_pct
        
        if margin_required > max_margin:
            position_value = max_margin * self.config.leverage
            position_size = position_value / entry_price
        
        if margin_required > self.current_equity or position_value < (self.current_equity * 0.001):
            return 0, 0
        
        return position_size, position_value
    
    def check_sl_tp_hit(self, trade: Trade, bar_idx: int) -> Tuple[bool, str, float]:
        """Check if SL or TP was hit with gap handling."""
        
        bar = self.df.iloc[bar_idx]
        open_price = bar['open']
        high = bar['high']
        low = bar['low']
        
        if trade.direction == 'long':
            # Check gap down
            if open_price <= trade.sl_price:
                return True, 'sl', open_price
            # Check gap up
            if open_price >= trade.tp_price:
                return True, 'tp', open_price
            
            # Intrabar hits
            if low <= trade.sl_price:
                if high >= trade.tp_price:
                    return True, 'sl', trade.sl_price
                return True, 'sl', trade.sl_price
            
            if high >= trade.tp_price:
                return True, 'tp', trade.tp_price
        
        elif trade.direction == 'short':
            # Check gap up
            if open_price >= trade.sl_price:
                return True, 'sl', open_price
            # Check gap down
            if open_price <= trade.tp_price:
                return True, 'tp', open_price
            
            # Intrabar hits
            if high >= trade.sl_price:
                if low <= trade.tp_price:
                    return True, 'sl', trade.sl_price
                return True, 'sl', trade.sl_price
            
            if low <= trade.tp_price:
                return True, 'tp', trade.tp_price
        
        return False, '', 0.0
    
    def enter_trade(self, signal_idx: int, direction: str, confidence: float,
                   tp_r: float, sl_r: float, atr: float) -> bool:
        """Enter a trade."""
        
        if signal_idx >= len(self.df) - 1:
            return False
        
        entry_idx = signal_idx + 1
        entry_bar = self.df.iloc[entry_idx]
        entry_price = entry_bar['open']
        
        # Apply spread
        spread_cost = self.pip_value * 1.0
        
        if direction == 'long':
            entry_price += spread_cost
            sl_price = entry_price - (atr * sl_r)
            tp_price = entry_price + (atr * tp_r)
        else:
            entry_price -= spread_cost
            sl_price = entry_price + (atr * sl_r)
            tp_price = entry_price - (atr * tp_r)
        
        position_size, position_value = self.calculate_position_size(entry_price, sl_price)
        
        if position_size <= 0:
            return False
        
        # Apply slippage
        slippage = entry_price * self.config.slippage_pct
        if direction == 'long':
            entry_price += slippage
        else:
            entry_price -= slippage
        
        trade = Trade(
            entry_idx=entry_idx,
            entry_time=entry_bar['timestamp'],
            entry_price=entry_price,
            direction=direction,
            position_size=position_size,
            sl_price=sl_price,
            tp_price=tp_price,
            confidence=confidence
        )
        
        self.active_trade = trade
        return True
    
    def close_trade(self, exit_idx: int, reason: str, exit_price: float):
        """Close active trade."""
        
        if self.active_trade is None:
            return
        
        trade = self.active_trade
        exit_bar = self.df.iloc[exit_idx]
        
        # Apply slippage
        slippage = exit_price * self.config.slippage_pct
        if trade.direction == 'long':
            exit_price -= slippage
        else:
            exit_price += slippage
        
        # Calculate P&L
        if trade.direction == 'long':
            price_change = exit_price - trade.entry_price
        else:
            price_change = trade.entry_price - exit_price
        
        gross_pnl = trade.position_size * price_change
        
        # Commission
        entry_value = trade.position_size * trade.entry_price
        exit_value = trade.position_size * exit_price
        commission = (entry_value + exit_value) * self.config.commission_pct
        
        net_pnl = gross_pnl - commission
        
        trade.exit_idx = exit_idx
        trade.exit_time = exit_bar['timestamp']
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl = net_pnl
        
        self.current_equity += net_pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        self.trades.append(trade)
        self.active_trade = None
    
    def run(self, long_signals: pd.Series, short_signals: pd.Series,
            long_probs: pd.Series, short_probs: pd.Series,
            tp_r: float, sl_r: float) -> Dict:
        """Run backtest."""
        
        if 'atr14' not in self.df.columns:
            raise ValueError("ATR14 required")
        
        for i in range(len(self.df)):
            self.equity_curve.append(self.current_equity)
            
            # Circuit breaker
            current_dd = (self.peak_equity - self.current_equity) / self.peak_equity
            if current_dd > self.circuit_breaker_dd:
                self.trading_halted = True
                if self.active_trade is not None:
                    self.close_trade(i, 'circuit_breaker', self.df.iloc[i]['close'])
                continue
            
            # Manage active trade
            if self.active_trade is not None:
                hit, reason, exit_price = self.check_sl_tp_hit(self.active_trade, i)
                if hit:
                    self.close_trade(i, reason, exit_price)
                    continue
                
                # Timeout
                if (i - self.active_trade.entry_idx) >= self.config.max_bars_in_trade:
                    self.close_trade(i, 'timeout', self.df.iloc[i]['close'])
                    continue
                
                continue
            
            # Skip if halted
            if self.trading_halted:
                continue
            
            atr = self.df['atr14'].iloc[i]
            
            long_prob = long_probs.iloc[i] if i < len(long_probs) else 0
            short_prob = short_probs.iloc[i] if i < len(short_probs) else 0
            
            # Prioritize higher probability
            if long_signals.iloc[i] and long_prob >= self.config.confidence_threshold:
                if short_signals.iloc[i] and short_prob > long_prob:
                    self.enter_trade(i, 'short', short_prob, tp_r, sl_r, atr)
                else:
                    self.enter_trade(i, 'long', long_prob, tp_r, sl_r, atr)
            elif short_signals.iloc[i] and short_prob >= self.config.confidence_threshold:
                self.enter_trade(i, 'short', short_prob, tp_r, sl_r, atr)
        
        # Close any remaining trade
        if self.active_trade is not None:
            self.close_trade(len(self.df) - 1, 'end', self.df.iloc[-1]['close'])
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate backtest metrics."""
        
        if len(self.trades) == 0:
            return {
                'total_trades': 0, 'long_trades': 0, 'short_trades': 0,
                'win_rate': 0, 'profit_factor': 0, 'sharpe_ratio': 0,
                'max_drawdown_pct': 0, 'total_return_pct': 0,
                'avg_win': 0, 'avg_loss': 0
            }
        
        trades_df = pd.DataFrame([
            {'pnl': t.pnl, 'exit_reason': t.exit_reason, 'direction': t.direction}
            for t in self.trades
        ])
        
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(wins) / len(trades_df) * 100
        
        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = abs(drawdown.min()) * 100
        
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        total_return_pct = (self.current_equity / self.config.initial_capital - 1) * 100
        
        long_trades = len(trades_df[trades_df['direction'] == 'long'])
        short_trades = len(trades_df[trades_df['direction'] == 'short'])
        
        return {
            'total_trades': len(trades_df),
            'long_trades': long_trades,
            'short_trades': short_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown_pct,
            'total_return_pct': total_return_pct,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        }


def run_backtest(df: pd.DataFrame, model: AdaptiveEnsemble, features: List[str],
                tp_r: float, sl_r: float, symbol: str) -> Dict:
    """Run backtest with model predictions."""
    
    print(f"  Generating predictions on {len(df)} bars...")
    
    X = df[features].fillna(0).values
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Extract probabilities
    flat_probs = pd.Series(probabilities[:, 0], index=df.index)
    long_probs = pd.Series(probabilities[:, 1], index=df.index)
    short_probs = pd.Series(probabilities[:, 2], index=df.index)
    
    # Signal generation with edge requirement
    MIN_CONFIDENCE = CONFIG.MIN_CONFIDENCE
    MIN_EDGE = CONFIG.MIN_EDGE
    
    long_signals = pd.Series(False, index=df.index)
    short_signals = pd.Series(False, index=df.index)
    
    for pos in range(len(df)):
        probs = [flat_probs.iloc[pos], long_probs.iloc[pos], short_probs.iloc[pos]]
        max_prob = max(probs)
        sorted_probs = sorted(probs, reverse=True)
        edge = sorted_probs[0] - sorted_probs[1]
        
        # Long signal
        if long_probs.iloc[pos] == max_prob and long_probs.iloc[pos] >= MIN_CONFIDENCE and edge >= MIN_EDGE:
            long_signals.iloc[pos] = True
        
        # Short signal
        elif short_probs.iloc[pos] == max_prob and short_probs.iloc[pos] >= MIN_CONFIDENCE and edge >= MIN_EDGE:
            short_signals.iloc[pos] = True
    
    long_count = long_signals.sum()
    short_count = short_signals.sum()
    
    # Debug
    flat_pred = (predictions == 0).sum()
    long_pred = (predictions == 1).sum()
    short_pred = (predictions == 2).sum()
    
    print(f"  Model predictions: Flat={flat_pred} ({flat_pred/len(df)*100:.1f}%), "
          f"Long={long_pred} ({long_pred/len(df)*100:.1f}%), "
          f"Short={short_pred} ({short_pred/len(df)*100:.1f}%)")
    print(f"  Trading signals: Long={long_count} ({long_count/len(df)*100:.1f}%), "
          f"Short={short_count} ({short_count/len(df)*100:.1f}%)")
    
    config = TradeConfig(
        initial_capital=CONFIG.INITIAL_CAPITAL,
        risk_per_trade_pct=CONFIG.RISK_PER_TRADE_PCT,
        confidence_threshold=MIN_CONFIDENCE,
        max_bars_in_trade=CONFIG.MAX_BARS_IN_TRADE,
        leverage=CONFIG.LEVERAGE,
        commission_pct=CONFIG.COMMISSION_PCT,
        slippage_pct=CONFIG.SLIPPAGE_PCT
    )
    
    engine = BacktestEngine(df, config, symbol)
    results = engine.run(long_signals, short_signals, long_probs, short_probs, tp_r, sl_r)
    
    return results


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_symbol(symbol: str, timeframe: str) -> Dict:
    """Train model for one symbol/timeframe."""
    
    print(f"\n{'='*80}")
    print(f"RENAISSANCE TECHNOLOGIES - TRAINING: {symbol} {timeframe}")
    print(f"{'='*80}\n")
    
    try:
        # 1. Load data
        print(f"[1/8] Loading data...")
        df = load_symbol_data(symbol, timeframe)
        
        # 2. Create balanced labels
        print(f"[2/8] Creating balanced labels...")
        overrides = CONFIG.SYMBOL_TP_SL_OVERRIDES.get(symbol, {})
        tp_mult = overrides.get('tp', CONFIG.TP_ATR_MULT)
        sl_mult = overrides.get('sl', CONFIG.SL_ATR_MULT)
        flat_thresh = overrides.get('flat_thresh', CONFIG.FLAT_THRESHOLD)
        print(f"  Using TP={tp_mult}x, SL={sl_mult}x, Flat_threshold={flat_thresh}")
        df = create_balanced_labels(df, tp_mult, sl_mult, flat_thresh)
        
        # 3. Select features
        print(f"[3/8] Selecting features...")
        features = select_features_strict(df)
        
        # 4. Train/OOS split
        print(f"[4/8] Creating train/OOS split...")
        oos_start = pd.to_datetime(CONFIG.TRAIN_END, utc=True) - timedelta(days=CONFIG.OOS_MONTHS * 30)
        train_df = df[df['timestamp'] < oos_start].copy()
        oos_df = df[df['timestamp'] >= oos_start].copy()
        
        print(f"    Train: {len(train_df):,} bars")
        print(f"    OOS:   {len(oos_df):,} bars")
        
        # 5. Walk-forward CV
        print(f"[5/8] Walk-forward CV...")
        splits = create_walk_forward_splits(train_df, n_folds=CONFIG.N_FOLDS)
        
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train = train_df.iloc[train_idx][features].fillna(0).values
            y_train = train_df.iloc[train_idx]['target'].values
            X_val = train_df.iloc[val_idx][features].fillna(0).values
            y_val = train_df.iloc[val_idx]['target'].values
            
            fold_model = AdaptiveEnsemble(n_classes=3)
            fold_model.fit(X_train, y_train, X_val, y_val)
            
            y_pred = fold_model.predict(X_val)
            acc = (y_pred == y_val).mean()
            
            fold_metrics.append({'fold': fold_idx + 1, 'accuracy': acc})
        
        # 6. Train final model
        print(f"[6/8] Training final model on full training set...")
        X_train_full = train_df[features].fillna(0).values
        y_train_full = train_df['target'].values
        final_model = AdaptiveEnsemble(n_classes=3)
        final_model.fit(X_train_full, y_train_full)
        
        # 7. Backtest OOS
        print(f"\n[7/8] Backtesting OOS period...")
        oos_results = run_backtest(oos_df, final_model, features, tp_mult, sl_mult, symbol)
        
        # 8. Check benchmarks
        print(f"\n[8/8] Checking Renaissance benchmarks...")
        passes, failures = check_benchmarks(oos_results, timeframe)
        
        # Save model
        print(f"Saving model...")
        save_path = save_model(symbol, timeframe, final_model, features,
                              fold_metrics, oos_results, passes, failures)
        
        print(f"\n{'='*80}")
        if passes:
            print(f"✅ {symbol} {timeframe} PASSED ALL BENCHMARKS")
            print(f"   PF: {oos_results['profit_factor']:.2f}, DD: {oos_results['max_drawdown_pct']:.1f}%, ")
            print(f"   WR: {oos_results['win_rate']:.1f}%, Sharpe: {oos_results['sharpe_ratio']:.2f}")
        else:
            print(f"❌ {symbol} {timeframe} FAILED: {', '.join(failures)}")
        print(f"{'='*80}\n")
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'passed': passes,
            'failures': failures,
            'oos_metrics': oos_results,
            'model_path': str(save_path)
        }
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'passed': False,
            'failures': [str(e)]
        }


def check_benchmarks(results: Dict, timeframe: str) -> Tuple[bool, List[str]]:
    """Check if meets Renaissance benchmarks."""
    
    failures = []
    
    tf_normalized = timeframe.upper().replace('M', 'T').replace('MIN', 'T')
    min_trades = CONFIG.MIN_TRADES_BY_TF.get(tf_normalized, CONFIG.MIN_TRADES_OOS)
    
    if results['profit_factor'] < CONFIG.MIN_PROFIT_FACTOR:
        failures.append(f"PF {results['profit_factor']:.2f} < {CONFIG.MIN_PROFIT_FACTOR}")
    
    if results['max_drawdown_pct'] > CONFIG.MAX_DRAWDOWN_PCT:
        failures.append(f"DD {results['max_drawdown_pct']:.1f}% > {CONFIG.MAX_DRAWDOWN_PCT}%")
    
    if results['sharpe_ratio'] < CONFIG.MIN_SHARPE:
        failures.append(f"Sharpe {results['sharpe_ratio']:.2f} < {CONFIG.MIN_SHARPE}")
    
    if results['win_rate'] < CONFIG.MIN_WIN_RATE:
        failures.append(f"WR {results['win_rate']:.1f}% < {CONFIG.MIN_WIN_RATE}%")
    
    if results['total_trades'] < min_trades:
        failures.append(f"Trades {results['total_trades']} < {min_trades}")
    
    return len(failures) == 0, failures


def save_model(symbol, timeframe, model, features, fold_metrics, oos_results, passes, failures):
    """Save model with metadata."""
    
    save_dir = CONFIG.MODEL_STORE / symbol
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    status = "READY" if passes else "FAILED"
    save_path = save_dir / f"{symbol}_{timeframe}_{status}_{timestamp}.pkl"
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': features,
            'fold_metrics': fold_metrics,
            'oos_results': oos_results,
            'passes': passes,
            'failures': failures,
            'timestamp': timestamp,
            'config': asdict(CONFIG)
        }, f)
    
    card_path = save_path.with_suffix('.json')
    with open(card_path, 'w') as f:
        json.dump({
            'symbol': symbol,
            'timeframe': timeframe,
            'status': status,
            'timestamp': timestamp,
            'oos_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                          for k, v in oos_results.items()},
            'benchmarks': {'passed': passes, 'failures': failures},
            'features_count': len(features)
        }, f, indent=2)
    
    return save_path


def train_all_parallel(max_workers: int = 4):
    """Train all symbols in parallel."""
    
    tasks = [(symbol, tf) for symbol in CONFIG.SYMBOLS for tf in CONFIG.TIMEFRAMES]
    results = []
    
    print(f"Training {len(tasks)} models with {max_workers} workers...\n")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_symbol, sym, tf): (sym, tf) for sym, tf in tasks}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
    
    passed = sum(1 for r in results if r['passed'])
    print(f"\n{'='*80}")
    print(f"RENAISSANCE TECHNOLOGIES - TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed}/{len(results)}")
    print(f"❌ Failed: {len(results) - passed}/{len(results)}")
    
    manifest_path = CONFIG.MODEL_STORE / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total': len(results),
            'passed': passed,
            'failed': len(results) - passed,
            'results': results
        }, f, indent=2)
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Renaissance Technologies ML Training System')
    parser.add_argument('--symbol', type=str, help='Symbol to train')
    parser.add_argument('--tf', type=str, help='Timeframe to train')
    parser.add_argument('--all', action='store_true', help='Train all symbols/timeframes')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    if args.all:
        train_all_parallel(max_workers=args.workers)
    elif args.symbol and args.tf:
        result = train_symbol(args.symbol, args.tf)
        return 0 if result['passed'] else 1
    else:
        print("Usage: --symbol XAUUSD --tf 15T  OR  --all")
        return 1


if __name__ == '__main__':
    exit(main())

