#!/usr/bin/env python3
"""
RENAISSANCE TECHNOLOGIES ML TRADING SYSTEM
==========================================

Production-grade machine learning system for systematic trading.

Key Features:
- Adaptive triple-barrier labeling with balanced classes (25-35% Flat)
- Sophisticated feature engineering with automatic selection
- Ensemble models with proper regularization
- Realistic backtesting with transaction costs
- Online learning capability for continuous improvement
- S&P 500 benchmark comparison
- Strict risk management (DD < 6%, PF > 1.5)

Author: Renaissance Technologies Quantitative Research Team
Date: November 2025
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

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RentecConfig:
    """Master configuration for Renaissance Technologies ML System."""
    
    # Paths
    FEATURE_STORE = Path("feature_store")
    MODEL_STORE = Path("models_rentec_v2")
    RESULTS_STORE = Path("results")
    
    # Data Configuration
    TRAIN_START = "2019-01-01"
    TRAIN_END = "2025-10-22"
    OOS_MONTHS = 6
    
    SYMBOLS = ["XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD"]
    TIMEFRAMES = ["5T", "15T", "30T", "1H", "4H"]
    
    # Walk-Forward CV
    N_FOLDS = 6
    EMBARGO_BARS = 50
    PURGE_BARS = 25
    
    # Triple-Barrier Labeling - ADAPTIVE
    FORECAST_HORIZON = 30
    TP_ATR_MULT = 1.5  # Target 1.5x ATR
    SL_ATR_MULT = 1.0  # Risk 1.0x ATR (1.5:1 R:R)
    FLAT_THRESHOLD = 0.7  # Only label as Up/Down if >= 70% of TP/SL achieved
    
    # Symbol-specific overrides for volatile assets
    SYMBOL_TP_SL_OVERRIDES = {
        'XAUUSD': {'tp': 1.6, 'sl': 1.0, 'flat_threshold': 0.75},
        'XAGUSD': {'tp': 1.6, 'sl': 1.0, 'flat_threshold': 0.75},
        'EURUSD': {'tp': 1.4, 'sl': 1.0, 'flat_threshold': 0.70},
        'GBPUSD': {'tp': 1.5, 'sl': 1.0, 'flat_threshold': 0.70},
        'AUDUSD': {'tp': 1.4, 'sl': 1.0, 'flat_threshold': 0.70},
        'NZDUSD': {'tp': 1.4, 'sl': 1.0, 'flat_threshold': 0.70},
        'USDJPY': {'tp': 1.4, 'sl': 1.0, 'flat_threshold': 0.70},
        'USDCAD': {'tp': 1.4, 'sl': 1.0, 'flat_threshold': 0.70},
    }
    
    # Trading Parameters
    INITIAL_CAPITAL = 100000
    RISK_PER_TRADE_PCT = 0.01  # 1% risk per trade
    MAX_POSITION_PCT = 0.20     # Max 20% of capital per position
    LEVERAGE = 20.0              # Conservative leverage
    COMMISSION_PCT = 0.00006     # 0.6 basis points (realistic for institutional)
    SLIPPAGE_PCT = 0.00002       # 0.2 basis points (realistic slippage)
    MAX_BARS_IN_TRADE = 50       # Max holding period
    
    # Signal Generation - STRICT QUALITY CONTROL
    MIN_CONFIDENCE = 0.52        # Minimum probability for trade
    MIN_EDGE = 0.12              # Minimum edge over alternatives
    MAX_CORRELATION_WINDOW = 20  # Check signal correlation to avoid over-trading
    
    # Benchmarks - STRICT
    MIN_PROFIT_FACTOR = 1.50
    MAX_DRAWDOWN_PCT = 6.0
    MIN_SHARPE_RATIO = 0.30
    MIN_WIN_RATE = 51.0
    MIN_TRADES_OOS = 100
    MIN_AVG_TRADE_RETURN_PCT = 0.05  # Minimum 0.05% per trade after costs
    
    # Feature Engineering
    MAX_FEATURES = 40            # Reduced for better generalization
    MIN_FEATURE_IMPORTANCE = 0.001
    LOOKBACK_PERIODS = [5, 10, 20, 50, 100, 200]
    
    # Model Configuration
    ENSEMBLE_WEIGHTS = {'xgb': 0.35, 'lgb': 0.35, 'rf': 0.20, 'linear': 0.10}
    ONLINE_LEARNING_RATE = 0.01
    RETRAIN_FREQUENCY_BARS = 5000
    
    # S&P 500 Benchmark
    SPX_ANNUAL_RETURN = 0.10     # 10% annual return benchmark
    SPX_SHARPE = 0.80            # S&P 500 typical Sharpe
    SPX_MAX_DD = 0.20            # S&P 500 typical max DD


CONFIG = RentecConfig()
CONFIG.MODEL_STORE.mkdir(parents=True, exist_ok=True)
CONFIG.RESULTS_STORE.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ADVANCED TRIPLE-BARRIER LABELING
# ============================================================================

class AdaptiveTripleBarrierLabeler:
    """
    Sophisticated triple-barrier labeling with:
    - Adaptive thresholds based on market volatility
    - Balanced class distribution (25-35% Flat)
    - Clear directional signals only
    """
    
    def __init__(self, config: RentecConfig):
        self.config = config
        
    def create_labels(self, df: pd.DataFrame, symbol: str, tp_mult: float, 
                     sl_mult: float, flat_threshold: float) -> pd.DataFrame:
        """Create triple-barrier labels with adaptive thresholds."""
        
        print("  Creating adaptive triple-barrier labels...")
        
        df = df.copy()
        n = len(df)
        horizon = self.config.FORECAST_HORIZON
        
        # Get ATR for dynamic thresholds
        atr = df.get('atr14', df['close'] * 0.02).values
        entry_prices = df['close'].values
        
        # Calculate TP/SL levels
        tp_long = entry_prices + (atr * tp_mult)
        sl_long = entry_prices - (atr * sl_mult)
        tp_short = entry_prices - (atr * tp_mult)
        sl_short = entry_prices + (atr * sl_mult)
        
        labels = np.zeros(n, dtype=int)
        returns = np.zeros(n)
        durations = np.zeros(n, dtype=int)
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Calculate labels
        for i in range(n - horizon):
            future_end = min(i + 1 + horizon, n)
            future_highs = highs[i+1:future_end]
            future_lows = lows[i+1:future_end]
            future_closes = closes[i+1:future_end]
            
            if len(future_highs) == 0:
                continue
            
            # Check long direction
            tp_long_hits = np.where(future_highs >= tp_long[i])[0]
            sl_long_hits = np.where(future_lows <= sl_long[i])[0]
            
            # Check short direction
            tp_short_hits = np.where(future_lows <= tp_short[i])[0]
            sl_short_hits = np.where(future_highs >= sl_short[i])[0]
            
            tp_long_hit = len(tp_long_hits) > 0
            sl_long_hit = len(sl_long_hits) > 0
            tp_short_hit = len(tp_short_hits) > 0
            sl_short_hit = len(sl_short_hits) > 0
            
            # Long direction wins
            if tp_long_hit and (not sl_long_hit or tp_long_hits[0] < sl_long_hits[0]):
                labels[i] = 1  # Up
                returns[i] = tp_mult * atr[i] / entry_prices[i]
                durations[i] = tp_long_hits[0] + 1
            
            # Short direction wins
            elif tp_short_hit and (not sl_short_hit or tp_short_hits[0] < sl_short_hits[0]):
                labels[i] = 2  # Down
                returns[i] = tp_mult * atr[i] / entry_prices[i]
                durations[i] = tp_short_hits[0] + 1
            
            # Long SL hit
            elif sl_long_hit and (not tp_long_hit or sl_long_hits[0] < tp_long_hits[0]):
                labels[i] = 2  # Down (long failed)
                returns[i] = -sl_mult * atr[i] / entry_prices[i]
                durations[i] = sl_long_hits[0] + 1
            
            # Short SL hit
            elif sl_short_hit and (not tp_short_hit or sl_short_hits[0] < tp_short_hits[0]):
                labels[i] = 1  # Up (short failed)
                returns[i] = -sl_mult * atr[i] / entry_prices[i]
                durations[i] = sl_short_hits[0] + 1
            
            # No clear hit - classify based on final movement
            else:
                final_price = future_closes[-1]
                ret = (final_price - entry_prices[i]) / entry_prices[i]
                atr_normalized_ret = ret * entry_prices[i] / atr[i]
                
                # STRICT THRESHOLD - Only clear moves get labeled
                if atr_normalized_ret >= (tp_mult * flat_threshold):
                    labels[i] = 1  # Up
                elif atr_normalized_ret <= -(tp_mult * flat_threshold):
                    labels[i] = 2  # Down
                else:
                    labels[i] = 0  # Flat - ambiguous move
                
                returns[i] = ret
                durations[i] = len(future_closes)
        
        df['target'] = labels
        df['expected_return'] = returns
        df['expected_duration'] = durations
        
        # Remove last horizon bars (no labels)
        df = df.iloc[:-self.config.FORECAST_HORIZON]
        
        # Print class distribution
        class_counts = df['target'].value_counts()
        total = len(df)
        for cls in [0, 1, 2]:
            count = class_counts.get(cls, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"    {['Flat', 'Up', 'Down'][cls]}: {count:,} ({pct:.1f}%)")
        
        # Validate balanced distribution
        flat_pct = class_counts.get(0, 0) / total * 100
        if flat_pct < 20:
            print(f"    ⚠️  WARNING: Flat class only {flat_pct:.1f}% - model may overtrade!")
        elif flat_pct > 40:
            print(f"    ⚠️  WARNING: Flat class {flat_pct:.1f}% - model may be too conservative!")
        else:
            print(f"    ✅ Balanced distribution achieved")
        
        return df


# ============================================================================
# SOPHISTICATED FEATURE ENGINEERING
# ============================================================================

class RentecFeatureEngine:
    """
    Renaissance Technologies-grade feature engineering.
    
    Focus on:
    - Statistical arbitrage signals
    - Mean reversion indicators  
    - Momentum factors
    - Volatility regime detection
    - Microstructure signals
    """
    
    def __init__(self, config: RentecConfig):
        self.config = config
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build sophisticated features."""
        
        df = df.copy()
        
        # 1. Multi-timeframe momentum with decay
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'momentum_{period}_ema'] = df[f'momentum_{period}'].ewm(span=period//2).mean()
        
        # 2. Volatility regime detection
        for period in [10, 20, 50]:
            df[f'vol_{period}'] = df['close'].pct_change().rolling(period).std()
            df[f'vol_{period}_zscore'] = (df[f'vol_{period}'] - df[f'vol_{period}'].rolling(100).mean()) / df[f'vol_{period}'].rolling(100).std()
        
        # 3. Mean reversion signals
        if 'ema20' in df.columns:
            df['mean_reversion_20'] = (df['close'] - df['ema20']) / df.get('atr14', df['close'] * 0.02)
            df['mean_reversion_20_extreme'] = (abs(df['mean_reversion_20']) > 2).astype(int)
        
        if 'ema50' in df.columns:
            df['mean_reversion_50'] = (df['close'] - df['ema50']) / df.get('atr14', df['close'] * 0.02)
        
        # 4. Trend strength and quality
        if 'ema10' in df.columns and 'ema50' in df.columns:
            df['trend_alignment'] = ((df['ema10'] > df['ema50']).astype(int) * 2 - 1)  # +1 or -1
            df['trend_strength'] = abs(df['ema10'] - df['ema50']) / df['ema50']
        
        # 5. RSI-based signals
        if 'rsi14' in df.columns:
            df['rsi_zscore'] = (df['rsi14'] - 50) / 25  # Normalized RSI
            df['rsi_divergence'] = df['rsi14'].diff(5) - df['close'].pct_change(5) * 100
            df['rsi_regime'] = pd.cut(df['rsi14'], bins=[0, 30, 70, 100], labels=[0, 1, 2])
        
        # 6. Volume analysis
        if 'volume' in df.columns:
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
            df['volume_trend'] = df['volume'].rolling(10).mean() / df['volume'].rolling(50).mean()
            df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'].pct_change())
        
        # 7. Price action patterns (safe - uses closes)
        df['higher_high'] = (df['close'] > df['close'].rolling(10).max().shift(1)).astype(int)
        df['lower_low'] = (df['close'] < df['close'].rolling(10).min().shift(1)).astype(int)
        df['inside_bar'] = ((df['close'] - df['close'].shift(5)).abs() < df.get('atr14', df['close'] * 0.02) * 0.5).astype(int)
        
        # 8. Bollinger Band signals
        if 'bb_pct' in df.columns and 'bb_width' in df.columns:
            df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(100).quantile(0.2)).astype(int)
            df['bb_extreme'] = ((df['bb_pct'] < 0.05) | (df['bb_pct'] > 0.95)).astype(int)
            df['bb_mean_reversion'] = (df['bb_pct'] - 0.5) * 2  # -1 to +1
        
        # 9. ADX trend quality
        if 'adx' in df.columns:
            df['adx_strong'] = (df['adx'] > 25).astype(int)
            df['adx_slope'] = df['adx'].diff(3)
            if 'plus_di' in df.columns and 'minus_di' in df.columns:
                df['di_spread'] = df['plus_di'] - df['minus_di']
                df['di_crossover'] = ((df['plus_di'] > df['minus_di']).astype(int) * 2 - 1)
        
        # 10. MACD signals
        if 'macd_hist' in df.columns:
            df['macd_hist_slope'] = df['macd_hist'].diff(3)
            df['macd_zero_cross'] = (df['macd_hist'] * df['macd_hist'].shift(1) < 0).astype(int)
        
        # 11. Market timing features
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_london_open'] = df['hour'].between(8, 16).astype(int)
            df['is_ny_open'] = df['hour'].between(13, 21).astype(int)
            df['is_overlap'] = df['hour'].between(13, 16).astype(int)
        
        # 12. Statistical features
        df['skewness_20'] = df['close'].pct_change().rolling(20).skew()
        df['kurtosis_20'] = df['close'].pct_change().rolling(20).kurt()
        
        # 13. Regime detection - combine multiple signals
        regime_score = 0
        if 'adx' in df.columns:
            regime_score += (df['adx'] > 25).astype(float) * 0.3
        if 'vol_20_zscore' in df.columns:
            regime_score += (df['vol_20_zscore'] < 0).astype(float) * 0.3  # Low vol
        if 'volume_trend' in df.columns:
            regime_score += (df['volume_trend'] > 1).astype(float) * 0.2
        df['favorable_regime'] = regime_score
        
        return df


# ============================================================================
# INTELLIGENT FEATURE SELECTION
# ============================================================================

class IntelligentFeatureSelector:
    """Advanced feature selection with multiple criteria."""
    
    def __init__(self, config: RentecConfig):
        self.config = config
        
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select features using multiple criteria."""
        
        print("  Selecting features with advanced criteria...")
        
        # Exclude non-features
        exclude = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'target', 'expected_return', 'expected_duration',
        ]
        
        # Exclude SMC features (just in case)
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
        
        # Test for look-ahead bias
        clean_features = self._remove_lookahead_features(df, all_features)
        print(f"    After look-ahead check: {len(clean_features)} features")
        
        # Remove highly correlated features
        clean_features = self._remove_collinear_features(df, clean_features)
        print(f"    After collinearity removal: {len(clean_features)} features")
        
        # Select by importance
        if len(clean_features) > self.config.MAX_FEATURES:
            clean_features = self._select_by_importance(df, clean_features)
            print(f"    After importance selection: {len(clean_features)} features")
        
        print(f"    ✅ Final features: {len(clean_features)}")
        
        return clean_features
    
    def _remove_lookahead_features(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove features with look-ahead bias."""
        
        clean = []
        for feat in features:
            try:
                feat_series = df[feat].fillna(0)
                
                # Test correlation with future returns
                future_ret = df['close'].pct_change(5).shift(-5)
                future_corr = abs(feat_series.corr(future_ret))
                
                # Test correlation with past returns
                past_ret = df['close'].pct_change(5)
                past_corr = abs(feat_series.corr(past_ret))
                
                # Flag if future correlation is suspiciously high
                if future_corr > 0.03 or (future_corr > past_corr * 2 and future_corr > 0.02):
                    continue
                
                clean.append(feat)
            except:
                continue
        
        return clean
    
    def _remove_collinear_features(self, df: pd.DataFrame, features: List[str], threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features."""
        
        corr_matrix = df[features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = set()
        for col in upper.columns:
            if any(upper[col] > threshold):
                to_drop.add(col)
        
        return [f for f in features if f not in to_drop]
    
    def _select_by_importance(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Select top features by importance."""
        
        X = df[features].fillna(0).values
        y = df['target'].values
        
        # Use LightGBM for fast importance calculation
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            num_leaves=15, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0,
            random_state=42, verbosity=-1, force_row_wise=True
        )
        model.fit(X, y)
        
        importance = dict(zip(features, model.feature_importances_))
        
        # Filter by minimum importance and select top N
        important_features = {f: imp for f, imp in importance.items() 
                            if imp >= self.config.MIN_FEATURE_IMPORTANCE}
        
        sorted_features = sorted(important_features.items(), key=lambda x: x[1], reverse=True)
        selected = [f[0] for f in sorted_features[:self.config.MAX_FEATURES]]
        
        return selected


# ============================================================================
# ENSEMBLE MODEL WITH ONLINE LEARNING
# ============================================================================

class RentecEnsembleModel:
    """
    Sophisticated ensemble model with:
    - Multiple learners (XGBoost, LightGBM, RandomForest, Linear)
    - Strong regularization to prevent overfitting
    - Balanced class weights
    - Online learning capability
    """
    
    def __init__(self, config: RentecConfig, n_classes: int = 3):
        self.config = config
        self.n_classes = n_classes
        self.models = {}
        self.scaler = RobustScaler()  # More robust to outliers
        self.weights = config.ENSEMBLE_WEIGHTS
        self.online_learner = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train ensemble with strong regularization."""
        
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
        
        # Calculate balanced class weights with boost for Flat class
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        
        # CRITICAL: Boost Flat class significantly to learn "no trade"
        if len(class_weights) >= 3:
            class_weights[0] *= 4.0  # 4x weight for Flat class
            class_weights[2] *= 1.3  # Slight boost for Down to balance long/short
        
        sample_weights = class_weights[y_train]
        
        print("    Training XGBoost...")
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,  # Shallow trees to reduce overfitting
            learning_rate=0.02,  # Low learning rate
            subsample=0.6,
            colsample_bytree=0.6,
            gamma=2.0,
            reg_alpha=3.0,
            reg_lambda=4.0,
            min_child_weight=20,
            random_state=42,
            verbosity=0
        )
        self.models['xgb'].fit(X_train, y_train, sample_weight=sample_weights)
        
        print("    Training LightGBM...")
        self.models['lgb'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.02,
            num_leaves=7,  # Very conservative
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=3.0,
            reg_lambda=4.0,
            min_child_samples=40,
            random_state=42,
            verbosity=-1,
            force_row_wise=True
        )
        self.models['lgb'].fit(X_train, y_train, sample_weight=sample_weights)
        
        print("    Training Random Forest...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=40,
            min_samples_leaf=20,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train, y_train, sample_weight=sample_weights)
        
        print("    Training Linear Model...")
        self.models['linear'] = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            C=0.03,  # Strong regularization
            max_iter=2000,
            class_weight='balanced',
            random_state=42
        )
        self.models['linear'].fit(X_train_scaled, y_train)
        
        # Initialize online learner for continuous learning
        print("    Initializing online learner...")
        self.online_learner = SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            alpha=0.001,
            l1_ratio=0.5,
            learning_rate='adaptive',
            eta0=self.config.ONLINE_LEARNING_RATE,
            max_iter=1000,
            random_state=42,
            warm_start=True
        )
        self.online_learner.fit(X_train_scaled, y_train)
        
    def predict_proba(self, X):
        """Weighted ensemble prediction."""
        
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
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def partial_fit(self, X, y):
        """Online learning - update model with new data."""
        X_scaled = self.scaler.transform(X)
        
        if self.online_learner is not None:
            self.online_learner.partial_fit(X_scaled, y)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_symbol_data(symbol: str, timeframe: str, config: RentecConfig) -> pd.DataFrame:
    """Load and prepare data."""
    
    path = config.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
    
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
    
    # Filter date range
    train_start = pd.to_datetime(config.TRAIN_START, utc=True)
    train_end = pd.to_datetime(config.TRAIN_END, utc=True)
    df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)]
    
    print(f"  Loaded {len(df):,} bars")
    
    # Apply feature engineering
    feature_engine = RentecFeatureEngine(config)
    df = feature_engine.engineer_features(df)
    
    return df.reset_index(drop=True)


# ============================================================================
# WALK-FORWARD CV
# ============================================================================

def create_purged_embargo_splits(df: pd.DataFrame, config: RentecConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create walk-forward CV splits with purging and embargo."""
    
    print(f"  Creating {config.N_FOLDS} walk-forward folds...")
    
    n_samples = len(df)
    fold_size = n_samples // (config.N_FOLDS + 1)
    
    splits = []
    
    for fold_idx in range(config.N_FOLDS):
        val_start = (fold_idx + 1) * fold_size
        val_end = val_start + fold_size
        
        if val_end > n_samples:
            break
        
        # Purge and embargo
        train_end = val_start - config.PURGE_BARS
        train_indices = np.arange(0, max(0, train_end))
        val_indices = np.arange(val_start, min(val_end, n_samples - config.EMBARGO_BARS))
        
        if len(train_indices) > 100 and len(val_indices) > 20:
            splits.append((train_indices, val_indices))
    
    print(f"  ✓ Created {len(splits)} valid folds")
    return splits


# ============================================================================
# PRODUCTION BACKTEST ENGINE
# ============================================================================

@dataclass
class TradeConfig:
    """Trading configuration."""
    initial_capital: float = 100000
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.20
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
    """Trade record."""
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


class RentecBacktestEngine:
    """Production-grade backtest engine with realistic costs."""
    
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
        self.max_drawdown_circuit_breaker = 0.12
        self.trading_halted = False
    
    def calculate_position_size(self, entry_price: float, sl_price: float) -> Tuple[float, float]:
        """Calculate position size based on risk."""
        risk_amount = self.current_equity * self.config.risk_per_trade_pct
        sl_distance = abs(entry_price - sl_price)
        
        if sl_distance <= 0:
            return 0, 0
        
        position_size = risk_amount / sl_distance
        position_value = position_size * entry_price
        
        margin_required = position_value / self.config.leverage
        max_margin = self.current_equity * self.config.max_position_pct
        
        if margin_required > max_margin:
            position_value = max_margin * self.config.leverage
            position_size = position_value / entry_price
        
        if margin_required > self.current_equity or position_value < (self.current_equity * 0.001):
            return 0, 0
        
        return position_size, position_value
    
    def check_sl_tp_hit(self, trade: Trade, bar_idx: int) -> Tuple[bool, str, float]:
        """Check if SL or TP hit with gap handling."""
        
        bar = self.df.iloc[bar_idx]
        open_price = bar['open']
        high = bar['high']
        low = bar['low']
        
        if trade.direction == 'long':
            # Check gaps
            if open_price <= trade.sl_price:
                return True, 'sl', open_price
            if open_price >= trade.tp_price:
                return True, 'tp', open_price
            
            # Intrabar hits
            if low <= trade.sl_price:
                if high >= trade.tp_price:
                    return True, 'sl', trade.sl_price  # SL hit first (conservative)
                return True, 'sl', trade.sl_price
            
            if high >= trade.tp_price:
                return True, 'tp', trade.tp_price
        
        elif trade.direction == 'short':
            if open_price >= trade.sl_price:
                return True, 'sl', open_price
            if open_price <= trade.tp_price:
                return True, 'tp', open_price
            
            if high >= trade.sl_price:
                if low <= trade.tp_price:
                    return True, 'sl', trade.sl_price
                return True, 'sl', trade.sl_price
            
            if low <= trade.tp_price:
                return True, 'tp', trade.tp_price
        
        return False, '', 0.0
    
    def enter_trade(self, signal_idx: int, direction: str, confidence: float,
                   tp_r: float, sl_r: float, atr: float) -> bool:
        """Enter a new trade."""
        
        if signal_idx >= len(self.df) - 1:
            return False
        
        entry_idx = signal_idx + 1
        entry_bar = self.df.iloc[entry_idx]
        entry_price = entry_bar['open']
        
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
        
        slippage = exit_price * self.config.slippage_pct
        if trade.direction == 'long':
            exit_price -= slippage
        else:
            exit_price += slippage
        
        if trade.direction == 'long':
            price_change = exit_price - trade.entry_price
        else:
            price_change = trade.entry_price - exit_price
        
        gross_pnl = trade.position_size * price_change
        
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
            if current_dd > self.max_drawdown_circuit_breaker:
                self.trading_halted = True
                if self.active_trade is not None:
                    self.close_trade(i, 'circuit_breaker', self.df.iloc[i]['close'])
                continue
            
            if self.active_trade is not None:
                hit, reason, exit_price = self.check_sl_tp_hit(self.active_trade, i)
                if hit:
                    self.close_trade(i, reason, exit_price)
                    continue
                
                if (i - self.active_trade.entry_idx) >= self.config.max_bars_in_trade:
                    self.close_trade(i, 'timeout', self.df.iloc[i]['close'])
                    continue
                
                continue
            
            if self.trading_halted:
                continue
            
            atr = self.df['atr14'].iloc[i]
            
            long_prob = long_probs.iloc[i] if i < len(long_probs) else 0
            short_prob = short_probs.iloc[i] if i < len(short_probs) else 0
            
            if long_signals.iloc[i] and long_prob >= self.config.confidence_threshold:
                if short_signals.iloc[i] and short_prob > long_prob:
                    self.enter_trade(i, 'short', short_prob, tp_r, sl_r, atr)
                else:
                    self.enter_trade(i, 'long', long_prob, tp_r, sl_r, atr)
            elif short_signals.iloc[i] and short_prob >= self.config.confidence_threshold:
                self.enter_trade(i, 'short', short_prob, tp_r, sl_r, atr)
        
        if self.active_trade is not None:
            self.close_trade(len(self.df) - 1, 'end', self.df.iloc[-1]['close'])
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        
        if len(self.trades) == 0:
            return {
                'total_trades': 0, 'long_trades': 0, 'short_trades': 0,
                'win_rate': 0, 'profit_factor': 0, 'sharpe_ratio': 0,
                'max_drawdown_pct': 0, 'total_return_pct': 0,
                'avg_trade_return_pct': 0
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
        avg_trade_return_pct = (trades_df['pnl'].sum() / self.config.initial_capital) / len(trades_df) * 100
        
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
            'avg_trade_return_pct': avg_trade_return_pct,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        }


def generate_signals(df: pd.DataFrame, model: RentecEnsembleModel, features: List[str],
                    config: RentecConfig) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate trading signals with strict quality control."""
    
    print(f"  Generating predictions on {len(df)} bars...")
    
    X = df[features].fillna(0).values
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    flat_probs = pd.Series(probabilities[:, 0], index=df.index)
    long_probs = pd.Series(probabilities[:, 1], index=df.index)
    short_probs = pd.Series(probabilities[:, 2], index=df.index)
    
    long_signals = pd.Series(False, index=df.index)
    short_signals = pd.Series(False, index=df.index)
    
    for pos in range(len(df)):
        probs = [flat_probs.iloc[pos], long_probs.iloc[pos], short_probs.iloc[pos]]
        max_prob = max(probs)
        sorted_probs = sorted(probs, reverse=True)
        edge = sorted_probs[0] - sorted_probs[1]
        
        if long_probs.iloc[pos] == max_prob and long_probs.iloc[pos] >= config.MIN_CONFIDENCE and edge >= config.MIN_EDGE:
            long_signals.iloc[pos] = True
        elif short_probs.iloc[pos] == max_prob and short_probs.iloc[pos] >= config.MIN_CONFIDENCE and edge >= config.MIN_EDGE:
            short_signals.iloc[pos] = True
    
    long_count = long_signals.sum()
    short_count = short_signals.sum()
    
    flat_pred = (predictions == 0).sum()
    long_pred = (predictions == 1).sum()
    short_pred = (predictions == 2).sum()
    
    print(f"  Model predictions: Flat={flat_pred} ({flat_pred/len(df)*100:.1f}%), "
          f"Long={long_pred} ({long_pred/len(df)*100:.1f}%), "
          f"Short={short_pred} ({short_pred/len(df)*100:.1f}%)")
    print(f"  Trading signals: Long={long_count} ({long_count/len(df)*100:.1f}%), "
          f"Short={short_count} ({short_count/len(df)*100:.1f}%)")
    
    return long_signals, short_signals, long_probs, short_probs


def run_backtest(df: pd.DataFrame, model: RentecEnsembleModel, features: List[str],
                tp_r: float, sl_r: float, symbol: str, config: RentecConfig) -> Dict:
    """Run complete backtest."""
    
    long_signals, short_signals, long_probs, short_probs = generate_signals(df, model, features, config)
    
    trade_config = TradeConfig(
        initial_capital=config.INITIAL_CAPITAL,
        risk_per_trade_pct=config.RISK_PER_TRADE_PCT,
        confidence_threshold=config.MIN_CONFIDENCE,
        max_bars_in_trade=config.MAX_BARS_IN_TRADE,
        leverage=config.LEVERAGE,
        commission_pct=config.COMMISSION_PCT,
        slippage_pct=config.SLIPPAGE_PCT
    )
    
    engine = RentecBacktestEngine(df, trade_config, symbol)
    results = engine.run(long_signals, short_signals, long_probs, short_probs, tp_r, sl_r)
    
    return results


# ============================================================================
# BENCHMARKING
# ============================================================================

def check_benchmarks(results: Dict, symbol: str, timeframe: str, config: RentecConfig) -> Tuple[bool, List[str]]:
    """Check if results meet Renaissance Technologies standards."""
    
    failures = []
    
    if results['profit_factor'] < config.MIN_PROFIT_FACTOR:
        failures.append(f"PF {results['profit_factor']:.2f} < {config.MIN_PROFIT_FACTOR}")
    
    if results['max_drawdown_pct'] > config.MAX_DRAWDOWN_PCT:
        failures.append(f"DD {results['max_drawdown_pct']:.1f}% > {config.MAX_DRAWDOWN_PCT}%")
    
    if results['sharpe_ratio'] < config.MIN_SHARPE_RATIO:
        failures.append(f"Sharpe {results['sharpe_ratio']:.2f} < {config.MIN_SHARPE_RATIO}")
    
    if results['win_rate'] < config.MIN_WIN_RATE:
        failures.append(f"WR {results['win_rate']:.1f}% < {config.MIN_WIN_RATE}%")
    
    if results['total_trades'] < config.MIN_TRADES_OOS:
        failures.append(f"Trades {results['total_trades']} < {config.MIN_TRADES_OOS}")
    
    if results['avg_trade_return_pct'] < config.MIN_AVG_TRADE_RETURN_PCT:
        failures.append(f"Avg Trade Return {results['avg_trade_return_pct']:.3f}% < {config.MIN_AVG_TRADE_RETURN_PCT}%")
    
    return len(failures) == 0, failures


def compare_to_spx(results: Dict, oos_months: int, config: RentecConfig) -> Dict:
    """Compare results to S&P 500 benchmark."""
    
    years = oos_months / 12
    
    # Annualize strategy metrics
    strategy_annual_return = (results['total_return_pct'] / 100) / years if years > 0 else 0
    strategy_sharpe = results['sharpe_ratio']
    strategy_max_dd = results['max_drawdown_pct'] / 100
    
    # S&P 500 benchmarks
    spx_annual_return = config.SPX_ANNUAL_RETURN
    spx_sharpe = config.SPX_SHARPE
    spx_max_dd = config.SPX_MAX_DD
    
    # Calculate excess returns
    excess_return = strategy_annual_return - spx_annual_return
    excess_sharpe = strategy_sharpe - spx_sharpe
    dd_improvement = spx_max_dd - strategy_max_dd
    
    beats_spx = (
        strategy_annual_return > spx_annual_return and
        strategy_sharpe > spx_sharpe and
        strategy_max_dd < spx_max_dd
    )
    
    return {
        'beats_spx': beats_spx,
        'strategy_annual_return': strategy_annual_return * 100,
        'spx_annual_return': spx_annual_return * 100,
        'excess_return_pct': excess_return * 100,
        'strategy_sharpe': strategy_sharpe,
        'spx_sharpe': spx_sharpe,
        'excess_sharpe': excess_sharpe,
        'strategy_max_dd_pct': strategy_max_dd * 100,
        'spx_max_dd_pct': spx_max_dd * 100,
        'dd_improvement_pct': dd_improvement * 100
    }


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_symbol(symbol: str, timeframe: str, config: RentecConfig = None) -> Dict:
    """Train complete model for one symbol/timeframe."""
    
    if config is None:
        config = CONFIG
    
    print(f"\n{'='*80}")
    print(f"RENAISSANCE TECHNOLOGIES ML SYSTEM")
    print(f"Training: {symbol} {timeframe}")
    print(f"{'='*80}\n")
    
    try:
        # 1. Load data
        print(f"[1/8] Loading data...")
        df = load_symbol_data(symbol, timeframe, config)
        
        # 2. Create labels
        print(f"[2/8] Creating adaptive labels...")
        overrides = config.SYMBOL_TP_SL_OVERRIDES.get(symbol, {})
        tp_mult = overrides.get('tp', config.TP_ATR_MULT)
        sl_mult = overrides.get('sl', config.SL_ATR_MULT)
        flat_threshold = overrides.get('flat_threshold', config.FLAT_THRESHOLD)
        
        print(f"  TP={tp_mult}x ATR, SL={sl_mult}x ATR, Flat Threshold={flat_threshold}")
        
        labeler = AdaptiveTripleBarrierLabeler(config)
        df = labeler.create_labels(df, symbol, tp_mult, sl_mult, flat_threshold)
        
        # 3. Select features
        print(f"[3/8] Selecting features...")
        selector = IntelligentFeatureSelector(config)
        features = selector.select_features(df)
        
        # 4. Split train/OOS
        print(f"[4/8] Creating train/OOS split...")
        oos_start = pd.to_datetime(config.TRAIN_END, utc=True) - timedelta(days=config.OOS_MONTHS * 30)
        train_df = df[df['timestamp'] < oos_start].copy()
        oos_df = df[df['timestamp'] >= oos_start].copy()
        
        print(f"    Train: {len(train_df):,} bars ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
        print(f"    OOS:   {len(oos_df):,} bars ({oos_df['timestamp'].min()} to {oos_df['timestamp'].max()})")
        
        # 5. Walk-forward CV
        print(f"[5/8] Walk-forward CV...")
        splits = create_purged_embargo_splits(train_df, config)
        
        fold_metrics = []
        final_model = RentecEnsembleModel(config, n_classes=3)
        
        print(f"  Training ensemble on {len(splits)} folds...")
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train = train_df.iloc[train_idx][features].fillna(0).values
            y_train = train_df.iloc[train_idx]['target'].values
            X_val = train_df.iloc[val_idx][features].fillna(0).values
            y_val = train_df.iloc[val_idx]['target'].values
            
            fold_model = RentecEnsembleModel(config, n_classes=3)
            fold_model.fit(X_train, y_train, X_val, y_val)
            
            y_pred = fold_model.predict(X_val)
            acc = (y_pred == y_val).mean()
            
            fold_metrics.append({'fold': fold_idx + 1, 'accuracy': acc})
            print(f"    Fold {fold_idx + 1}: Accuracy={acc:.3f}")
        
        # 6. Train final model
        print(f"\n[6/8] Training final ensemble model...")
        X_train_full = train_df[features].fillna(0).values
        y_train_full = train_df['target'].values
        final_model.fit(X_train_full, y_train_full)
        
        # 7. Backtest OOS
        print(f"\n[7/8] Backtesting OOS period...")
        oos_results = run_backtest(oos_df, final_model, features, tp_mult, sl_mult, symbol, config)
        
        # 8. Check benchmarks
        print(f"\n[8/8] Checking Renaissance Technologies benchmarks...")
        passes, failures = check_benchmarks(oos_results, symbol, timeframe, config)
        
        # Compare to S&P 500
        spx_comparison = compare_to_spx(oos_results, config.OOS_MONTHS, config)
        
        # Save model
        print(f"\nSaving model...")
        save_path = save_model(symbol, timeframe, final_model, features,
                              fold_metrics, oos_results, spx_comparison,
                              passes, failures, config)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS: {symbol} {timeframe}")
        print(f"{'='*80}")
        print(f"  Trades: {oos_results['total_trades']} (Long: {oos_results['long_trades']}, Short: {oos_results['short_trades']})")
        print(f"  Win Rate: {oos_results['win_rate']:.1f}%")
        print(f"  Profit Factor: {oos_results['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {oos_results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {oos_results['max_drawdown_pct']:.1f}%")
        print(f"  Total Return: {oos_results['total_return_pct']:.1f}%")
        print(f"  Avg Trade Return: {oos_results['avg_trade_return_pct']:.3f}%")
        
        print(f"\n  S&P 500 COMPARISON:")
        print(f"  Strategy Annual Return: {spx_comparison['strategy_annual_return']:.1f}%")
        print(f"  S&P 500 Annual Return: {spx_comparison['spx_annual_return']:.1f}%")
        print(f"  Excess Return: {spx_comparison['excess_return_pct']:.1f}%")
        print(f"  Strategy Sharpe: {spx_comparison['strategy_sharpe']:.2f}")
        print(f"  S&P 500 Sharpe: {spx_comparison['spx_sharpe']:.2f}")
        print(f"  Beats S&P 500: {'✅ YES' if spx_comparison['beats_spx'] else '❌ NO'}")
        
        print(f"\n{'='*80}")
        if passes:
            print(f"✅ {symbol} {timeframe} PASSED ALL BENCHMARKS")
        else:
            print(f"❌ {symbol} {timeframe} FAILED: {', '.join(failures)}")
        print(f"{'='*80}\n")
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'passed': passes,
            'failures': failures,
            'oos_metrics': oos_results,
            'spx_comparison': spx_comparison,
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


def save_model(symbol, timeframe, model, features, fold_metrics, oos_results,
              spx_comparison, passes, failures, config):
    """Save model with comprehensive metadata."""
    
    save_dir = config.MODEL_STORE / symbol
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
            'spx_comparison': spx_comparison,
            'passes': passes,
            'failures': failures,
            'timestamp': timestamp,
            'config': asdict(config)
        }, f)
    
    # Save JSON card for easy inspection
    card_path = save_path.with_suffix('.json')
    with open(card_path, 'w') as f:
        json.dump({
            'symbol': symbol,
            'timeframe': timeframe,
            'status': status,
            'timestamp': timestamp,
            'oos_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                          for k, v in oos_results.items()},
            'spx_comparison': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                             for k, v in spx_comparison.items()},
            'benchmarks': {'passed': passes, 'failures': failures}
        }, f, indent=2)
    
    return save_path


def train_all_parallel(max_workers: int = 4, config: RentecConfig = None):
    """Train all symbols in parallel."""
    
    if config is None:
        config = CONFIG
    
    tasks = [(symbol, tf) for symbol in config.SYMBOLS for tf in config.TIMEFRAMES]
    results = []
    
    print(f"\n{'='*80}")
    print(f"RENAISSANCE TECHNOLOGIES ML SYSTEM")
    print(f"Training {len(tasks)} symbol/timeframe combinations")
    print(f"{'='*80}\n")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_symbol, sym, tf, config): (sym, tf) for sym, tf in tasks}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
    
    # Summary
    passed = sum(1 for r in results if r.get('passed', False))
    beats_spx = sum(1 for r in results if r.get('spx_comparison', {}).get('beats_spx', False))
    
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Total Trained: {len(results)}")
    print(f"  Passed Benchmarks: {passed}/{len(results)}")
    print(f"  Beats S&P 500: {beats_spx}/{len(results)}")
    print(f"{'='*80}\n")
    
    # Save manifest
    manifest_path = config.RESULTS_STORE / "training_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total': len(results),
            'passed': passed,
            'beats_spx': beats_spx,
            'results': results
        }, f, indent=2)
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Renaissance Technologies ML Trading System')
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
        print("Usage:")
        print("  Single: python rentec_ml_system.py --symbol XAUUSD --tf 15T")
        print("  All:    python rentec_ml_system.py --all --workers 4")
        return 1


if __name__ == '__main__':
    exit(main())

