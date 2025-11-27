"""
═══════════════════════════════════════════════════════════════════════════════
CITADEL-GRADE ML TRADING SYSTEM V2.4 - FIXED SPREAD & DRAWDOWN
═══════════════════════════════════════════════════════════════════════════════

CRITICAL FIXES IN V2.4:
━━━━━━━━━━━━━━━━━━━━━━━━━
✓ REALISTIC spread costs (0.03-0.08R per timeframe)
✓ FIXED drawdown calculation (no more 800,000% DD)
✓ ROBUST model selection (min 300 trades, PF > 1.2, DD < 35%)
✓ NEW liquidity sweep features (stop-hunt patterns)
✓ All previous V2.3 leakage fixes maintained

Usage:
    python citadel_training_system_v2.py --symbol XAUUSD --timeframe 5T --full-system
    python citadel_training_system_v2.py --symbol XAUUSD --all-timeframes --walk-forward
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import argparse
import json
import joblib

# ML imports
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """System configuration with REALISTIC spread costs and guardrails."""
    
    # Paths
    FEATURE_STORE: Path = Path("ML_model/ML_model/feature_store")
    
    # Performance targets (STRICTER for DD control - V2.5)
    MIN_WIN_RATE: float = 0.55
    MIN_PROFIT_FACTOR: float = 1.3
    TARGET_PROFIT_FACTOR: float = 1.5
    MAX_DRAWDOWN: float = 0.06       # Max 6% DD
    TARGET_DRAWDOWN: float = 0.03    # Target ~3% DD
    MIN_SHARPE: float = 0.30  # STRICTER: Min Sharpe 0.30 (was 0.25)
    
    # REALISTIC SPREAD COSTS (in R units per trade) - FIXED V2.4
    # DEPRECATED: Use BASE_SPREAD_R + SLIPPAGE_R + COMMISSION_R instead
    SPREAD_R_BY_TIMEFRAME: Dict[str, float] = field(default_factory=lambda: {
        "5T": 0.08,   # ~0.08R per trade on 5T
        "15T": 0.06,  # ~0.06R per trade on 15T
        "30T": 0.05,  # ~0.05R per trade on 30T
        "1H": 0.04,   # ~0.04R per trade on 1H
        "4H": 0.03    # ~0.03R per trade on 4H
    })
    
    # TRANSACTION COSTS (per trade, in R units)
    BASE_SPREAD_R: Dict[str, float] = field(default_factory=lambda: {
        "5T": 0.10,   # assume ~0.10R spread on 5T (more realistic)
        "15T": 0.08,
        "30T": 0.06,
        "1H": 0.05,
        "4H": 0.04,
    })
    SLIPPAGE_R: float = 0.05       # extra R cost to approximate slippage
    COMMISSION_R: float = 0.02     # commission/fees in R
    
    # ATR NOISE FILTERS (to avoid ultra low/high vol bars)
    ATR_MIN_PCTL: float = 0.10     # ignore ATR < 10th percentile
    ATR_MAX_PCTL: float = 0.90     # ignore ATR > 90th percentile
    
    # Label noise filtering
    MIN_ABS_R_FOR_LABEL: float = 0.10  # ignore trades where |R| < 0.1 (too noisy)
    
    # GUARDRAILS: Minimum trades for model eligibility
    @staticmethod
    def get_min_trades_raw(timeframe: str) -> int:
        """Minimum trades for raw model evaluation."""
        min_trades = {
            '5T': 1000,
            '15T': 500,
            '30T': 300,
            '1H': 200,
            '4H': 100
        }
        return min_trades.get(timeframe, 500)
    
    @staticmethod
    def get_min_trades_filtered(timeframe: str) -> int:
        """Minimum trades for filtered (post-threshold) evaluation."""
        min_trades = {
            '5T': 300,
            '15T': 200,
            '30T': 150,
            '1H': 100,
            '4H': 50
        }
        return min_trades.get(timeframe, 150)
    
    # ROBUST MODEL SELECTION CRITERIA - V3.0 (FOCUS ON LOW DRAWDOWN)
    MIN_TRADES_TEST: int = 300  # Ignore models with fewer test trades
    MAX_ACCEPTABLE_DD: float = 6.0   # Max 6% DD
    TARGET_MAX_DD: float = 4.0       # Target <= 4% DD if possible
    MIN_ACCEPTABLE_PF: float = 1.3  # Min 1.3 PF
    TARGET_PF: float = 1.5  # Target PF for preferred models
    MIN_ANNUAL_RETURN: float = 0.15  # NEW: Minimum 15% annualized return
    TARGET_ANNUAL_RETURN: float = 0.20  # Target 20% annualized return
    
    # Evaluation risk per trade (for DD and annualized return calculations)
    RISK_PER_TRADE_EVAL: float = 0.003  # 0.3% risk per trade to reduce DD
    
    # Debug options
    DEBUG_LEAK_CHECK: bool = False  # Enable detailed leakage checks
    
    # Trade frequency targets
    MIN_TRADES_PER_DAY: float = 5.0
    MAX_TRADES_PER_DAY: float = 40.0
    
    # EXPANDED TP multiplier search space
    @staticmethod
    def get_tp_multipliers(timeframe: str) -> List[float]:
        """Expanded TP search space."""
        multipliers = {
            '5T':  [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6],
            '15T': [1.2, 1.5, 2.0, 2.5, 3.0],
            '30T': [1.5, 2.0, 2.5, 3.0, 3.5],
            '1H':  [2.0, 2.5, 3.0, 3.5, 4.0],
            '4H':  [2.5, 3.0, 4.0, 5.0]
        }
        return multipliers.get(timeframe, [1.5, 2.0, 2.5, 3.0])
    
    SL_MULTIPLIER: float = 1.0
    
    # Time barrier search space - REALISTIC CANDLE COUNTS
    @staticmethod
    def get_time_barriers(timeframe: str) -> List[int]:
        """
        Multiple time barriers to test based on realistic trade durations.
        
        Realistic ranges:
        - 5T:  5-20 candles (avg 10-15)  -> test 15, 20, 25 (covers up to max + buffer)
        - 15T: 10-35 candles (avg 20-25) -> test 30, 35, 40 (covers up to max + buffer)
        - 30T: 15-50 candles (avg 30-40, occasionally 60+) -> test 45, 50, 60 (covers occasional longer trades)
        """
        barriers = {
            '5T':  [15, 20, 25],      # 5-20 candles realistic, test up to 25
            '15T': [30, 35, 40],      # 10-35 candles realistic, test up to 40
            '30T': [45, 50, 60],      # 15-50 candles realistic, test up to 60 for slower sessions
            '1H':  [20, 30, 40],      # Keep existing
            '4H':  [10, 15, 20]       # Keep existing
        }
        return barriers.get(timeframe, [60])
    
    # Bad regime detection - STRICTER for DD control
    BAD_REGIME_WR_THRESHOLD: float = 0.50
    BAD_REGIME_PF_THRESHOLD: float = 1.0
    BAD_REGIME_DD_THRESHOLD: float = 0.08
    MIN_REGIME_TRADES: int = 50
    
    # Data splits
    TRAIN_RATIO: float = 0.60
    VAL_RATIO: float = 0.20
    
    # Walk-forward validation
    WF_N_SPLITS: int = 5
    
    # Confidence thresholds - EXPANDED for DD control (up to 0.90)
    CONFIDENCE_THRESHOLDS: List[float] = field(default_factory=lambda: 
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    )
    
    # Model hyperparameters
    LGBM_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1
    })
    
    XGB_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': 0
    })
    
    CATBOOST_PARAMS: Dict = field(default_factory=lambda: {
        'iterations': 200,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'verbose': False
    })
    
    RF_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 50,
        'min_samples_leaf': 20,
        'max_features': 'sqrt',
        'n_jobs': -1
    })


CONFIG = SystemConfig()


# ═══════════════════════════════════════════════════════════════════════════
# EQUITY & DRAWDOWN UTILITIES - NEW V2.4
# ═══════════════════════════════════════════════════════════════════════════

def compute_equity_and_dd(r_multiples: np.ndarray, risk_per_trade: float = CONFIG.RISK_PER_TRADE_EVAL) -> Tuple[np.ndarray, float, float]:
    """
    Compute equity curve and drawdown metrics correctly.
    
    Args:
        r_multiples: per-trade R results (1.0 = 1R profit, -1.0 = 1R loss)
        risk_per_trade: fraction of equity risked per trade (default 1%)
    
    Returns:
        equity_curve: array of equity values
        max_dd_pct: maximum drawdown as percentage
        max_dd_r: maximum drawdown in R units
    """
    if len(r_multiples) == 0:
        return np.array([1.0]), 0.0, 0.0
    
    # Build equity curve starting at 1.0 (100%)
    equity = np.zeros(len(r_multiples) + 1)
    equity[0] = 1.0
    
    for i, r in enumerate(r_multiples):
        # PnL as fraction of current equity
        pnl_fraction = r * risk_per_trade
        equity[i + 1] = equity[i] * (1.0 + pnl_fraction)
    
    # Compute running peak
    peaks = np.maximum.accumulate(equity)
    
    # Drawdown at each point
    dd_absolute = peaks - equity
    dd_pct = dd_absolute / peaks
    
    # Maximum drawdown
    max_dd_pct = dd_pct.max() * 100.0  # Convert to percentage
    
    # Max DD in R units (approximate)
    max_dd_r = dd_absolute.max() / risk_per_trade
    
    return equity, max_dd_pct, max_dd_r


def get_total_cost_r(timeframe: str) -> float:
    """
    Total transaction cost in R units per round-trip trade (spread + slippage + fees).
    """
    base_spread = CONFIG.BASE_SPREAD_R.get(timeframe, 0.05)
    total_cost = base_spread + CONFIG.SLIPPAGE_R + CONFIG.COMMISSION_R
    return total_cost


# ═══════════════════════════════════════════════════════════════════════════
# RISK METRICS CALCULATOR - UPDATED V2.4
# ═══════════════════════════════════════════════════════════════════════════

class RiskMetrics:
    """Calculate trading risk metrics from r_multiples."""
    
    @staticmethod
    def calculate_profit_factor(r_multiples: np.ndarray) -> float:
        """Calculate Profit Factor: sum(R+) / sum(|R-|)"""
        winners = r_multiples[r_multiples > 0]
        losers = r_multiples[r_multiples < 0]
        
        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_sharpe(r_multiples: np.ndarray) -> float:
        """Calculate Sharpe ratio: mean(R) / std(R) - per-trade, not annualized"""
        if len(r_multiples) < 2:
            return 0.0
        
        mean_r = r_multiples.mean()
        std_r = r_multiples.std()
        
        if std_r == 0 or np.isnan(std_r):
            return 0.0
        
        sharpe = mean_r / std_r
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown(r_multiples: np.ndarray, risk_per_trade: float = CONFIG.RISK_PER_TRADE_EVAL) -> Tuple[float, float]:
        """Calculate max drawdown using fixed equity curve - V2.4"""
        _, max_dd_pct, max_dd_r = compute_equity_and_dd(r_multiples, risk_per_trade)
        return max_dd_pct, max_dd_r
    
    @staticmethod
    def calculate_all_metrics(r_multiples: np.ndarray, y_true: np.ndarray = None, 
                             y_pred: np.ndarray = None, risk_per_trade: float = CONFIG.RISK_PER_TRADE_EVAL) -> Dict:
        """Calculate all risk metrics with FIXED drawdown calculation."""
        if len(r_multiples) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown_r': 0.0,
                'mean_r': 0.0,
                'median_r': 0.0,
                'total_r': 0.0
            }
        
        wins = (r_multiples > 0).sum()
        losses = (r_multiples < 0).sum()
        win_rate = wins / len(r_multiples) if len(r_multiples) > 0 else 0
        
        pf = RiskMetrics.calculate_profit_factor(r_multiples)
        sharpe = RiskMetrics.calculate_sharpe(r_multiples)
        max_dd_pct, max_dd_r = RiskMetrics.calculate_max_drawdown(r_multiples, risk_per_trade)
        
        return {
            'total_trades': len(r_multiples),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_factor': pf,
            'sharpe': sharpe,
            'max_drawdown_pct': max_dd_pct,
            'max_drawdown_r': max_dd_r,
            'mean_r': r_multiples.mean(),
            'median_r': np.median(r_multiples),
            'total_r': r_multiples.sum()
        }


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADER WITH LAG ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Load features with ENFORCED lag verification."""
    
    @staticmethod
    def load_timeframe_data(symbol: str, timeframe: str) -> Tuple[pd.DataFrame, Dict]:
        """Load features with MANDATORY lag verification."""
        print(f"\n{'='*80}")
        print(f"LOADING DATA: {symbol} {timeframe}")
        print(f"{'='*80}")
        
        file_path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
        metadata_path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}_metadata.json"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"❌ Feature file not found: {file_path}\n"
                f"\n💡 Run: python extract_features_from_s3_fixed.py --symbol {symbol}"
            )
        
        print(f"📂 Loading: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if not metadata.get('lag_verified', False):
                print(f"\n{'='*80}")
                print(f"⚠️  WARNING: LAG VERIFICATION NOT CONFIRMED")
                print(f"{'='*80}\n")
            else:
                print(f"📋 Metadata: ✅ LAG VERIFIED")
        else:
            metadata = {'lag_verified': False}
        
        # Verify structure
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"❌ Missing required columns: {missing}")
        
        if 'atr' not in df.columns:
            raise ValueError(f"❌ ATR column not found!")
        
        feature_cols = [c for c in df.columns if c not in required_cols]
        
        print(f"\n✅ Data loaded:")
        print(f"   Rows: {len(df):,}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        if not df['timestamp'].is_monotonic_increasing:
            print(f"   ⚠️  Sorting timestamps...")
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df, metadata


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING - WITH LIQUIDITY SWEEP FEATURES (NEW V2.4)
# ═══════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """Engineer strategy-specific features with PROPER LAGGING."""
    
    @staticmethod
    def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Regime detection features."""
        features = df.copy()
        
        # Volatility regime
        if 'atr' in features.columns:
            # Shift rolling window by 1 to exclude current bar (avoid lookahead)
            # Compare previous bar's ATR to historical distribution (exclude itself from comparison)
            features['regime_vol_percentile'] = features['atr'].shift(1).rolling(100).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5
            )
            
            features['regime_vol'] = pd.cut(
                features['regime_vol_percentile'],
                bins=[0, 0.33, 0.67, 1.0],
                labels=[0, 1, 2]
            ).astype(float)
        
        # Trend regimes - shift EMAs by 1 to use only past data (avoid lookahead)
        if 'ema_20' in features.columns and 'ema_50' in features.columns:
            features['regime_trend_20_50'] = (
                (features['ema_20'].shift(1) > features['ema_50'].shift(1)).astype(int) * 2 - 1
            )
        
        if 'ema_50' in features.columns and 'ema_200' in features.columns:
            features['regime_trend_50_200'] = (
                (features['ema_50'].shift(1) > features['ema_200'].shift(1)).astype(int) * 2 - 1
            )
        
        # Session flags
        if 'hour' in features.columns:
            features['regime_session_asian'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
            features['regime_session_london'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
            features['regime_session_ny'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
        
            # Range position - use previous bar's close to avoid lookahead
        if 'close' in features.columns:
            # Shift rolling windows by 1 to exclude current bar (avoid lookahead)
            high_20 = features['high'].shift(1).rolling(20).max()
            low_20 = features['low'].shift(1).rolling(20).min()
            range_20 = high_20 - low_20
            # Use previous bar's close, not current (avoid lookahead)
            features['regime_range_position'] = (
                (features['close'].shift(1) - low_20) / (range_20 + 1e-8)
            )
        
        return features
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators."""
        features = df.copy()
        
        for period in [5, 10, 20]:
            features[f'momentum_roc_{period}'] = (
                features['close'].pct_change(period)
            )
        
        if 'atr' in features.columns:
            features['momentum_strength_5'] = (
                features['close'].diff(5) / (features['atr'] + 1e-8)
            )
            
            features['momentum_strength_10'] = (
                features['close'].diff(10) / (features['atr'] + 1e-8)
            )
        
        if 'momentum_roc_5' in features.columns and 'momentum_roc_10' in features.columns:
            features['momentum_accel'] = (
                features['momentum_roc_5'] - features['momentum_roc_10']
            )
        
        return features
    
    @staticmethod
    def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
        """Mean reversion indicators."""
        features = df.copy()
        
        for ma in [20, 50, 100]:
            if f'sma_{ma}' in features.columns:
                features[f'mr_distance_sma_{ma}'] = (
                    (features['close'] - features[f'sma_{ma}']) / 
                    (features[f'sma_{ma}'] + 1e-8)
                )
        
        if 'bb_upper' in features.columns and 'bb_lower' in features.columns:
            features['mr_bb_position'] = (
                (features['close'] - features['bb_lower']) / 
                (features['bb_upper'] - features['bb_lower'] + 1e-8)
            )
            
            features['mr_bb_extreme'] = (
                (features['mr_bb_position'] > 0.95) | (features['mr_bb_position'] < 0.05)
            ).astype(int)
        
        if 'rsi' in features.columns:
            features['mr_rsi_oversold'] = (features['rsi'] < 30).astype(int)
            features['mr_rsi_overbought'] = (features['rsi'] > 70).astype(int)
            features['mr_rsi_neutral'] = (
                (features['rsi'] >= 40) & (features['rsi'] <= 60)
            ).astype(int)
        
        return features
    
    @staticmethod
    def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features."""
        features = df.copy()
        
        features['micro_body'] = abs(features['close'] - features['open'])
        features['micro_upper_wick'] = (
            features['high'] - np.maximum(features['open'], features['close'])
        )
        features['micro_lower_wick'] = (
            np.minimum(features['open'], features['close']) - features['low']
        )
        features['micro_total_range'] = (features['high'] - features['low'])
        
        features['micro_body_ratio'] = (
            features['micro_body'] / (features['micro_total_range'] + 1e-8)
        )
        features['micro_wick_ratio'] = (
            (features['micro_upper_wick'] + features['micro_lower_wick']) / 
            (features['micro_total_range'] + 1e-8)
        )
        
        if 'volume' in features.columns:
            # Shift by 1 to exclude current bar from rolling mean (avoid lookahead)
            vol_ma = features['volume'].shift(1).rolling(20).mean()
            features['micro_volume_surge'] = (
                features['volume'] / (vol_ma + 1)
            )
            
            features['micro_volume_anomaly'] = (
                features['micro_volume_surge'] > 2.0
            ).astype(int)
        
        features['micro_gap'] = (
            features['open'] - features['close'].shift(1)
        )
        
        features['micro_gap_pct'] = (
            features['micro_gap'] / (features['close'].shift(1) + 1e-8)
        )
        
        return features
    
    @staticmethod
    def add_liquidity_sweep_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Liquidity Sweep / Stop-Run Pattern Features
        
        Models "stop-hunt" style liquidity events using pure OHLCV & ATR.
        """
        features = df.copy()
        
        # 6.1.1 Basic candle components (if not already present)
        if 'body' not in features.columns:
            features['body'] = abs(features['close'] - features['open'])
        if 'upper_wick' not in features.columns:
            features['upper_wick'] = features['high'] - features[['open', 'close']].max(axis=1)
        if 'lower_wick' not in features.columns:
            features['lower_wick'] = features[['open', 'close']].min(axis=1) - features['low']
        if 'total_range' not in features.columns:
            features['total_range'] = features['high'] - features['low']
        
        # 6.1.2 Local swing highs/lows (3-bar pattern)
        # CRITICAL: shift(0) is redundant (no-op) - using current bar's high/low which is available at bar time
        # Pattern: previous bar is higher/lower than both current and 2-bars-ago
        swing_high = (
            (features['high'].shift(1) > features['high']) &  # Previous > current (current available at bar time)
            (features['high'].shift(1) > features['high'].shift(2))  # Previous > 2-bars-ago
        )
        
        swing_low = (
            (features['low'].shift(1) < features['low']) &  # Previous < current (current available at bar time)
            (features['low'].shift(1) < features['low'].shift(2))  # Previous < 2-bars-ago
        )
        
        # Rolling last swing levels (forward fill)
        features['last_swing_high'] = np.where(
            swing_high,
            features['high'].shift(1),
            np.nan
        )
        features['last_swing_high'] = features['last_swing_high'].ffill()
        
        features['last_swing_low'] = np.where(
            swing_low,
            features['low'].shift(1),
            np.nan
        )
        features['last_swing_low'] = features['last_swing_low'].ffill()
        
        # 6.1.3 Volume & ATR baselines
        # Shift by 1 to exclude current bar from rolling calculations (avoid lookahead)
        vol_ma_50 = features['volume'].shift(1).rolling(50).mean() if 'volume' in features.columns else pd.Series(1.0, index=features.index)
        atr_ma_50 = features['atr'].shift(1).rolling(50).mean() if 'atr' in features.columns else pd.Series(1.0, index=features.index)
        
        # 6.1.4 Bullish liquidity sweep (flush down, then reverse up)
        # Use previous bar's swing low (shift by 1) to avoid lookahead
        cond_break_low = features['low'] < features['last_swing_low'].shift(1)
        cond_long_lower_wick = features['lower_wick'] >= 2.0 * features['body']
        cond_volume_spike = features['volume'] >= 1.5 * vol_ma_50 if 'volume' in features.columns else pd.Series(False, index=features.index)
        cond_atr_spike = features['atr'] >= 1.2 * atr_ma_50 if 'atr' in features.columns else pd.Series(False, index=features.index)
        
        features['liq_sweep_bullish'] = (
            cond_break_low &
            cond_long_lower_wick &
            cond_volume_spike &
            cond_atr_spike
        ).astype(int)
        
        # 6.1.5 Bearish liquidity sweep (flush up, then reverse down)
        # Use previous bar's swing high (shift by 1) to avoid lookahead
        cond_break_high = features['high'] > features['last_swing_high'].shift(1)
        cond_long_upper_wick = features['upper_wick'] >= 2.0 * features['body']
        
        features['liq_sweep_bearish'] = (
            cond_break_high &
            cond_long_upper_wick &
            cond_volume_spike &
            cond_atr_spike
        ).astype(int)
        
        # Generic sweep flag
        features['liq_sweep_any'] = (
            (features['liq_sweep_bullish'] == 1) | (features['liq_sweep_bearish'] == 1)
        ).astype(int)
        
        return features
    
    @staticmethod
    def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering."""
        print(f"\n🔧 Engineering features...")
        
        initial_cols = len(df.columns)
        
        df = FeatureEngineer.add_regime_features(df)
        df = FeatureEngineer.add_momentum_features(df)
        df = FeatureEngineer.add_mean_reversion_features(df)
        df = FeatureEngineer.add_microstructure_features(df)
        df = FeatureEngineer.add_liquidity_sweep_features(df)  # NEW V2.4
        
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        final_cols = len(df.columns)
        added_cols = final_cols - initial_cols
        
        print(f"   ✅ Added {added_cols} features (including liquidity sweeps)")
        print(f"   🧹 Dropped {dropped_rows} rows with NaNs")
        print(f"   ✓ Final: {len(df):,} rows, {final_cols} columns")
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# TRIPLE BARRIER LABELING - FIXED SPREAD COSTS V2.4
# ═══════════════════════════════════════════════════════════════════════════

class TripleBarrierLabeler:
    """
    Multi-side triple barrier labeling with LONG / SHORT / FLAT.

    direction_label:
        +1  -> Long
        -1  -> Short
        0   -> Flat / No-trade (ambiguous / noisy)

    We compute hypothetical long and short outcomes separately, apply costs,
    then pick the side with the better R if it exceeds a minimum edge threshold.
    """
    
    @staticmethod
    def _label_single_side(df: pd.DataFrame,
                           tp_mult: float,
                           sl_mult: float,
                           time_barrier: int,
                           side: str) -> Tuple[pd.Series, pd.Series]:
        """
        Internal helper: label for ONE side (long or short) before costs.
        
        Returns:
            side_labels: 1=hit TP, 0=hit SL or time exit with loss, -1=unlabeled
            r_pre:       raw R-multiples (before costs)
        """
        assert side in ("long", "short")

        labels = pd.Series(-1, index=df.index)
        r_pre = pd.Series(0.0, index=df.index)
        
        if 'atr' not in df.columns:
            raise ValueError("ATR required for labeling")
        
        for i in range(len(df) - time_barrier):
            entry_price = df['close'].iloc[i]
            atr = df['atr'].iloc[i]
            
            if pd.isna(entry_price) or pd.isna(atr) or atr <= 0:
                continue
            
            if side == "long":
                tp_price = entry_price + tp_mult * atr
                sl_price = entry_price - sl_mult * atr
            else:  # short side
                tp_price = entry_price - tp_mult * atr  # profit if price goes down
                sl_price = entry_price + sl_mult * atr  # loss if price goes up

            hit = False
            for j in range(1, time_barrier + 1):
                if i + j >= len(df):
                    break
                high = df['high'].iloc[i + j]
                low = df['low'].iloc[i + j]
                
                if side == "long":
                    if high >= tp_price:
                        labels.iloc[i] = 1
                        r_pre.iloc[i] = tp_mult
                        hit = True
                        break
                    if low <= sl_price:
                        labels.iloc[i] = 0
                        r_pre.iloc[i] = -sl_mult
                        hit = True
                        break
                else:
                    # short
                    if low <= tp_price:
                        labels.iloc[i] = 1
                        r_pre.iloc[i] = tp_mult
                        hit = True
                        break
                    if high >= sl_price:
                        labels.iloc[i] = 0
                        r_pre.iloc[i] = -sl_mult
                        hit = True
                        break

            if not hit:
                # time-bar exit: compare entry vs exit
                exit_price = df['close'].iloc[i + time_barrier]
                pnl = (exit_price - entry_price) if side == "long" else (entry_price - exit_price)
                r_pre.iloc[i] = pnl / atr
                labels.iloc[i] = 1 if r_pre.iloc[i] > 0 else 0

        return labels, r_pre

    @staticmethod
    def label(df: pd.DataFrame,
              tp_mult: float,
              sl_mult: float,
              time_barrier: int,
              timeframe: str) -> Tuple[pd.Series, pd.Series]:
        """
        Main multi-side labeling function.

        Returns:
            direction_labels: -1 (short), 0 (flat), +1 (long)
            r_post:           realized R (after costs) for the chosen side
        """
        if 'atr' not in df.columns:
            raise ValueError("ATR required for labeling")

        total_cost_r = get_total_cost_r(timeframe)

        # ATR filter to avoid extreme noise regimes
        atr = df['atr']
        atr_valid_low = atr.quantile(CONFIG.ATR_MIN_PCTL)
        atr_valid_high = atr.quantile(CONFIG.ATR_MAX_PCTL)
        valid_atr_mask = (atr >= atr_valid_low) & (atr <= atr_valid_high)

        long_labels, long_r_pre = TripleBarrierLabeler._label_single_side(
            df, tp_mult, sl_mult, time_barrier, side="long"
        )
        short_labels, short_r_pre = TripleBarrierLabeler._label_single_side(
            df, tp_mult, sl_mult, time_barrier, side="short"
        )

        # Apply costs
        long_r_post = long_r_pre - total_cost_r
        short_r_post = short_r_pre - total_cost_r

        direction_labels = pd.Series(0, index=df.index)   # -1 short, 0 flat, +1 long
        r_post = pd.Series(0.0, index=df.index)
        
        # Track statistics for logging only (not for balancing)
        long_count = 0
        short_count = 0
        both_valid_count = 0

        for i in range(len(df)):
            if not valid_atr_mask.iloc[i]:
                direction_labels.iloc[i] = 0
                r_post.iloc[i] = 0.0
                continue

            # Only consider bars where at least one side has a valid label
            long_valid = long_labels.iloc[i] != -1
            short_valid = short_labels.iloc[i] != -1

            if not long_valid and not short_valid:
                direction_labels.iloc[i] = 0
                r_post.iloc[i] = 0.0
                continue

            # Label ALL trades (winners AND losers), not just profitable ones
            # Prefer the side with higher post-cost R (more profitable)
            direction_label = 0
            r_value = 0.0
            
            # Get R values for both sides
            rL_val = long_r_post.iloc[i] if long_valid else 0.0
            rS_val = short_r_post.iloc[i] if short_valid else 0.0
            
            # Enforce MIN_ABS_R_FOR_LABEL threshold for both sides
            if long_valid and abs(rL_val) < CONFIG.MIN_ABS_R_FOR_LABEL:
                long_valid = False
            if short_valid and abs(rS_val) < CONFIG.MIN_ABS_R_FOR_LABEL:
                short_valid = False
            
            # Selection rule: choose side with higher R (prefer more profitable side)
            if long_valid and short_valid:
                both_valid_count += 1
                # Choose the side with the higher R
                if rL_val >= rS_val:
                    direction_label = +1
                    r_value = rL_val
                    long_count += 1
                else:
                    direction_label = -1
                    r_value = rS_val
                    short_count += 1
            elif long_valid:
                direction_label = +1
                r_value = rL_val
                long_count += 1
            elif short_valid:
                direction_label = -1
                r_value = rS_val
                short_count += 1
            else:
                # No valid trade
                direction_label = 0
                r_value = 0.0

            direction_labels.iloc[i] = direction_label
            r_post.iloc[i] = r_value

        # Print balance statistics (for logging only)
        total_labeled = long_count + short_count
        if total_labeled > 0:
            print(f"   Label distribution: {long_count:,} longs ({long_count/total_labeled*100:.1f}%), "
                  f"{short_count:,} shorts ({short_count/total_labeled*100:.1f}%), "
                  f"{both_valid_count:,} bars had both sides valid")

        return direction_labels, r_post
    
    @staticmethod
    def find_best_config(df: pd.DataFrame, timeframe: str) -> Tuple[float, int]:
        """
        Optimize labeling config with robustness checks to prevent fake metrics.
        
        Adds filters for:
        - Too few losers (degenerate configs)
        - Suspiciously high WR/PF combinations (overfitting)
        - Internal robustness check across 3 chronological chunks
        """
        print("\n🔍 OPTIMIZING MULTI-SIDE LABELING CONFIGURATION (WITH ROBUSTNESS)")
        print("=" * 80)

        total_cost = get_total_cost_r(timeframe)
        print(f"   Total cost per trade: {total_cost:.3f}R (spread + slippage + fees)\n")
        
        tp_candidates = CONFIG.get_tp_multipliers(timeframe)
        time_barriers = CONFIG.get_time_barriers(timeframe)
        
        best_config = None
        best_pf = 0.0
        best_avg_pf = 0.0  # Average PF across chunks
        
        print(f"{'TP':>6} {'TB':>6} {'PF':>8} {'WR':>8} {'Trades':>10} {'Losers':>8} {'Status':<25}")
        print("-" * 90)
        
        # Split df into 3 chronological chunks for robustness check
        n = len(df)
        chunk_size = n // 3
        chunks = [
            df.iloc[:chunk_size],
            df.iloc[chunk_size:2*chunk_size],
            df.iloc[2*chunk_size:]
        ]
        
        for tp_mult in tp_candidates:
            for tb in time_barriers:
                direction_labels, r_post = TripleBarrierLabeler.label(
                    df, tp_mult, CONFIG.SL_MULTIPLIER, tb, timeframe
                )
                mask = direction_labels != 0
                if mask.sum() == 0:
                    continue
                
                r_sel = r_post[mask].values
                metrics = RiskMetrics.calculate_all_metrics(r_sel, risk_per_trade=CONFIG.RISK_PER_TRADE_EVAL)
                
                # Count losers
                losers = (r_sel < 0).sum()
                winners = (r_sel > 0).sum()
                total = len(r_sel)
                loser_ratio = losers / total if total > 0 else 0.0

                status = "✅ Viable"
                eligible = True
                
                # Hard filters
                if metrics['total_trades'] < CONFIG.get_min_trades_raw(timeframe):
                    status = "⚠️ Low trades"
                    eligible = False
                elif losers == 0 or loser_ratio < 0.05:
                    status = "❌ No/too few losers"
                    eligible = False
                elif metrics['win_rate'] > 0.95 and metrics['profit_factor'] > 3.0:
                    status = "❌ Suspicious (WR>95%, PF>3)"
                    eligible = False
                elif metrics['profit_factor'] < 1.1:
                    status = "❌ Low PF"
                    eligible = False
                elif metrics['win_rate'] < 0.50:
                    status = "⚠️ Low WR"
                    eligible = False
                
                # Internal robustness check: compute PF on each chunk
                chunk_pfs = []
                chunk_wrs = []
                for chunk_df in chunks:
                    if len(chunk_df) == 0:
                        continue
                    chunk_labels, chunk_r = TripleBarrierLabeler.label(
                        chunk_df, tp_mult, CONFIG.SL_MULTIPLIER, tb, timeframe
                    )
                    chunk_mask = chunk_labels != 0
                    if chunk_mask.sum() > 0:
                        chunk_r_sel = chunk_r[chunk_mask].values
                        chunk_metrics = RiskMetrics.calculate_all_metrics(
                            chunk_r_sel, risk_per_trade=CONFIG.RISK_PER_TRADE_EVAL
                        )
                        chunk_pfs.append(chunk_metrics['profit_factor'])
                        chunk_wrs.append(chunk_metrics['win_rate'])
                
                # Require PF > 1.0 in all chunks and WR doesn't collapse
                if len(chunk_pfs) == 3:
                    avg_pf_chunks = np.mean(chunk_pfs)
                    min_pf_chunks = np.min(chunk_pfs)
                    min_wr_chunks = np.min(chunk_wrs)
                    
                    if min_pf_chunks <= 1.0:
                        if eligible:
                            status = "❌ Fails chunk robustness"
                        eligible = False
                    elif min_wr_chunks < 0.50:
                        if eligible:
                            status = "⚠️ WR collapses in chunk"
                        eligible = False
                    else:
                        # Use average PF across chunks as ranking metric
                        if eligible and avg_pf_chunks > best_avg_pf:
                            best_avg_pf = avg_pf_chunks
                            best_pf = metrics['profit_factor']
                            best_config = (tp_mult, tb)
                else:
                    # Fallback if chunks don't work
                    if eligible and metrics['profit_factor'] > best_pf:
                        best_pf = metrics['profit_factor']
                        best_config = (tp_mult, tb)

                print(f"{tp_mult:>6.1f} {tb:>6} {metrics['profit_factor']:>8.2f} "
                      f"{metrics['win_rate']:>7.1%} {metrics['total_trades']:>10,} "
                      f"{losers:>8,} {status:<25}")
        
        if best_config is None:
            print("\n❌ NO PROFITABLE CONFIG FOUND, falling back to defaults")
            return tp_candidates[0], time_barriers[0]
        
        tp_best, tb_best = best_config
        print(f"\n✅ BEST CONFIG: TP={tp_best:.1f}x ATR, TB={tb_best} bars (PF={best_pf:.2f}, Avg Chunk PF={best_avg_pf:.2f})\n")
        return tp_best, tb_best


# ═══════════════════════════════════════════════════════════════════════════
# CHRONOLOGICAL SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

class DataSplitter:
    """Chronological train/val/test split (no shuffle, no overlap)."""
    
    @staticmethod
    def split_chronological(df: pd.DataFrame,
                            labels: pd.Series,
                           r_multiples: pd.Series) -> Dict:
        """Split data chronologically with period lengths."""
        print(f"\n✂️  CHRONOLOGICAL DATA SPLIT")
        print(f"{'='*80}")
        labeled_mask = labels != 0   # keep long (+1) and short (-1); flat (0) is no trade
        df_labeled = df[labeled_mask].copy()
        labels_filtered = labels[labeled_mask].copy()
        r_multiples_filtered = r_multiples[labeled_mask].copy()
        
        print(f"   Total samples: {len(df_labeled):,}")
        print(f"   Longs: {(labels_filtered == 1).sum():,}")
        print(f"   Shorts: {(labels_filtered == -1).sum():,}")
        
        n = len(df_labeled)
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        
        df_train = df_labeled.iloc[:train_end]
        df_val = df_labeled.iloc[train_end:val_end]
        df_test = df_labeled.iloc[val_end:]
        
        y_train = labels_filtered.iloc[:train_end]
        y_val = labels_filtered.iloc[train_end:val_end]
        y_test = labels_filtered.iloc[val_end:]
        
        r_train = r_multiples_filtered.iloc[:train_end]
        r_val = r_multiples_filtered.iloc[train_end:val_end]
        r_test = r_multiples_filtered.iloc[val_end:]
        
        train_min_ts = df_train['timestamp'].min()
        train_max_ts = df_train['timestamp'].max()
        val_min_ts = df_val['timestamp'].min()
        val_max_ts = df_val['timestamp'].max()
        test_min_ts = df_test['timestamp'].min()
        test_max_ts = df_test['timestamp'].max()
        
        print(f"\n   📅 Train: {train_min_ts} to {train_max_ts}")
        print(f"   📅 Val:   {val_min_ts} to {val_max_ts}")
        print(f"   📅 Test:  {test_min_ts} to {test_max_ts}")
        
        assert train_max_ts < val_min_ts, "❌ Train/Val overlap!"
        assert val_max_ts < test_min_ts, "❌ Val/Test overlap!"

        print(f"\n   ✅ No temporal overlap")
        
        for name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            longs = (y_split == 1).sum()
            shorts = (y_split == -1).sum()
            flats = (y_split == 0).sum()
            print(f"   {name}: {len(y_split):,} samples - Longs: {longs:,}, Shorts: {shorts:,}, Flats: {flats:,}")

            if len(np.unique(y_split[y_split != 0])) < 2:
                raise ValueError(f"❌ {name} split has only one non-flat class!")

        feature_cols = [
            c for c in df_labeled.columns
            if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        ]
        
        X_train = df_train[feature_cols].values
        X_val = df_val[feature_cols].values
        X_test = df_test[feature_cols].values

        # Period lengths (for annualized returns)
        val_days = (val_max_ts - val_min_ts).total_seconds() / 86400.0
        test_days = (test_max_ts - test_min_ts).total_seconds() / 86400.0
        
        print(f"\n   ✅ Split complete: {len(feature_cols)} features")
        print(f"   Val days:  {val_days:.1f}")
        print(f"   Test days: {test_days:.1f}")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train.values, 'y_val': y_val.values, 'y_test': y_test.values,
            'r_train': r_train.values, 'r_val': r_val.values, 'r_test': r_test.values,
            'feature_cols': feature_cols,
            'train_ts': df_train,
            'val_ts': df_val,
            'test_ts': df_test,
            'val_days': val_days,
            'test_days': test_days
        }
    
    @staticmethod
    def split_chronological_from_pre_split(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        labels_train: pd.Series,
        labels_val: pd.Series,
        labels_test: pd.Series,
        r_train: pd.Series,
        r_val: pd.Series,
        r_test: pd.Series
    ) -> Dict:
        """
        Reconstruct splits dictionary from pre-processed dataframes and labels.
        
        This is used when data has already been split and processed independently
        to avoid rolling window leakage.
        """
        # Filter to labeled only
        train_mask = labels_train != 0
        val_mask = labels_val != 0
        test_mask = labels_test != 0
        
        df_train_labeled = df_train[train_mask].copy()
        df_val_labeled = df_val[val_mask].copy()
        df_test_labeled = df_test[test_mask].copy()
        
        y_train = labels_train[train_mask].copy()
        y_val = labels_val[val_mask].copy()
        y_test = labels_test[test_mask].copy()
        
        r_train_filtered = r_train[train_mask].copy()
        r_val_filtered = r_val[val_mask].copy()
        r_test_filtered = r_test[test_mask].copy()
        
        # Extract feature columns
        feature_cols = [
            c for c in df_train_labeled.columns
            if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        ]
        
        X_train = df_train_labeled[feature_cols].values
        X_val = df_val_labeled[feature_cols].values
        X_test = df_test_labeled[feature_cols].values
        
        # Calculate period lengths
        train_days = (df_train_labeled['timestamp'].max() - df_train_labeled['timestamp'].min()).total_seconds() / 86400.0
        val_days = (df_val_labeled['timestamp'].max() - df_val_labeled['timestamp'].min()).total_seconds() / 86400.0
        test_days = (df_test_labeled['timestamp'].max() - df_test_labeled['timestamp'].min()).total_seconds() / 86400.0
        
        print(f"\n✂️  RECONSTRUCTING SPLITS FROM PRE-PROCESSED DATA")
        print(f"{'='*80}")
        print(f"   Train: {len(df_train_labeled):,} labeled samples")
        print(f"   Val:   {len(df_val_labeled):,} labeled samples")
        print(f"   Test:  {len(df_test_labeled):,} labeled samples")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train.values, 'y_val': y_val.values, 'y_test': y_test.values,
            'r_train': r_train_filtered.values, 'r_val': r_val_filtered.values, 'r_test': r_test_filtered.values,
            'feature_cols': feature_cols,
            'train_ts': df_train_labeled,
            'val_ts': df_val_labeled,
            'test_ts': df_test_labeled,
            'train_days': train_days,
            'val_days': val_days,
            'test_days': test_days
        }


# ═══════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def encode_direction_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """
    Map original direction labels {-1, +1} to {0, 1} for binary models.

    Returns:
        y_encoded
        forward_map: {orig -> enc}
        inverse_map: {enc -> orig}
    """
    unique = sorted(set(int(v) for v in np.unique(y) if v != 0))
    
    if len(unique) == 0:
        raise ValueError("No non-zero labels found for encoding")
    
    # We expect [-1, +1], map to [0, 1]
    forward = {}
    inverse = {}
    for enc, orig in enumerate(unique):
        forward[orig] = enc
        inverse[enc] = orig
    
    # Encode, handling missing labels gracefully
    y_enc = np.array([forward.get(int(v), 0) if v != 0 else -1 for v in y])
    
    return y_enc, forward, inverse


class ModelFactory:
    """Train multiple model families."""
    
    @staticmethod
    def prepare_sample_weights(y_train):
        """Compute balanced sample weights."""
        classes = np.unique(y_train)
        if len(classes) < 2:
            return np.ones(len(y_train)), 1.0
        
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[yi] for yi in y_train])
        scale_pos_weight = weight_dict[1] / weight_dict[0] if 0 in weight_dict and 1 in weight_dict else 1.0
        
        return sample_weights, scale_pos_weight
    
    @staticmethod
    def train_all_models(X_train, X_val, y_train, y_val):
        """Train all model families."""
        print(f"\n🤖 TRAINING MODELS")
        print(f"{'='*80}")
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        sample_weights, scale_pos_weight = ModelFactory.prepare_sample_weights(y_train)
        
        models = {}
        
        # LightGBM
        try:
            print(f"\n   Training LightGBM...")
            params = CONFIG.LGBM_PARAMS.copy()
            params['scale_pos_weight'] = scale_pos_weight
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights,
                     eval_set=[(X_val_scaled, y_val)],
                     callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            models['lightgbm'] = {'model': model, 'scaler': scaler}
            print(f"   ✅ LightGBM trained")
        except Exception as e:
            print(f"   ❌ LightGBM failed: {e}")
        
        # XGBoost
        try:
            print(f"\n   Training XGBoost...")
            params = CONFIG.XGB_PARAMS.copy()
            params['scale_pos_weight'] = scale_pos_weight
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights,
                     eval_set=[(X_val_scaled, y_val)], verbose=False)
            models['xgboost'] = {'model': model, 'scaler': scaler}
            print(f"   ✅ XGBoost trained")
        except Exception as e:
            print(f"   ❌ XGBoost failed: {e}")
        
        # CatBoost
        try:
            print(f"\n   Training CatBoost...")
            pos_weight = sample_weights[y_train == 1].mean() if (y_train == 1).sum() > 0 else 1.0
            neg_weight = sample_weights[y_train == 0].mean() if (y_train == 0).sum() > 0 else 1.0
            class_weights = {0: neg_weight, 1: pos_weight}
            params = CONFIG.CATBOOST_PARAMS.copy()
            params['class_weights'] = class_weights
            model = CatBoostClassifier(**params)
            model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val),
                     early_stopping_rounds=50, verbose=False)
            models['catboost'] = {'model': model, 'scaler': scaler}
            print(f"   ✅ CatBoost trained")
        except Exception as e:
            print(f"   ❌ CatBoost failed: {e}")
        
        # Random Forest
        try:
            print(f"\n   Training RandomForest...")
            model = RandomForestClassifier(**CONFIG.RF_PARAMS)
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            models['random_forest'] = {'model': model, 'scaler': scaler}
            print(f"   ✅ RandomForest trained")
        except Exception as e:
            print(f"   ❌ RandomForest failed: {e}")
        
        # Logistic Regression
        try:
            print(f"\n   Training LogisticRegression...")
            model = LogisticRegression(max_iter=1000, class_weight='balanced',
                                     C=0.1, solver='liblinear')
            model.fit(X_train_scaled, y_train)
            models['logistic'] = {'model': model, 'scaler': scaler}
            print(f"   ✅ LogisticRegression trained")
        except Exception as e:
            print(f"   ❌ LogisticRegression failed: {e}")
        
        print(f"\n   ✅ Trained {len(models)} models")
        
        return models


# ═══════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION - WITH ROBUST SELECTION V2.4
# ═══════════════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """Evaluate models with ROBUST trade count / DD / WR / PF guardrails."""
    
    @staticmethod
    def _compute_annual_return_from_r(r_multiples: np.ndarray,
                                      risk_per_trade: float,
                                      test_days: float) -> float:
        """
        Compute annualized % return from R-multiples.
        - risk_per_trade: fraction of equity per trade (e.g. 0.01 for 1%)
        - test_days: length of test period in days
        """
        if len(r_multiples) == 0 or test_days <= 0:
            return 0.0

        cumulative_return = np.prod(1.0 + r_multiples * risk_per_trade)

        if cumulative_return <= 0:
            return -100.0

        # Annualize
        annual_return = (cumulative_return ** (365.0 / test_days) - 1.0) * 100.0

        return float(annual_return)

    @staticmethod
    def _estimate_linear_annual_return(r_multiples: np.ndarray,
                                       risk_per_trade: float,
                                       test_days: float) -> float:
        """
        Estimate *linear* annual return (no crazy compounding).

        Approximation:
          - mean_r = mean R per trade
          - trades_per_day = num_trades / test_days
          - trades_per_year ≈ trades_per_day * 252
          - expected_annual_return ≈ mean_r * risk_per_trade * trades_per_year

        This gives a more realistic, order-of-magnitude estimate than compounding
        every trade.
        """
        if len(r_multiples) == 0 or test_days <= 0:
            return 0.0

        mean_r = float(np.mean(r_multiples))
        trades = len(r_multiples)
        trades_per_day = trades / max(test_days, 1e-6)
        trades_per_year = trades_per_day * 252.0

        expected_annual_return = mean_r * risk_per_trade * trades_per_year * 100.0
        return expected_annual_return

    @staticmethod
    def evaluate_all_models_on_split(models: Dict,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     r: np.ndarray,
                                     timeframe: str,
                                     days: float,
                                     risk_per_trade: float = CONFIG.RISK_PER_TRADE_EVAL) -> Dict:
        """
        Generic evaluation function for any split (val OR test).
        Does NOT decide eligibility based on annual return guardrails; it's purely scoring.
        """
        print(f"\n📊 EVALUATING MODELS ON SPLIT")
        print(f"{'='*80}")
        min_trades = CONFIG.get_min_trades_raw(timeframe)
        print(f"   Minimum trades: {min_trades}")
        print(f"   Min WR: {CONFIG.MIN_WIN_RATE:.0%}, "
              f"Min PF: {CONFIG.MIN_ACCEPTABLE_PF:.2f}, "
              f"Max DD: {CONFIG.MAX_ACCEPTABLE_DD:.1f}%\n")
        
        results = {}
        
        for model_name, model_dict in models.items():
            try:
                model = model_dict['model']
                scaler = model_dict['scaler']
                X_scaled = scaler.transform(X)

                y_pred = model.predict(X_scaled)
                
                acc = accuracy_score(y, y_pred)
                f1 = f1_score(y, y_pred, zero_division=0)
                
                trade_mask = (y_pred == 1)
                trades_taken = int(trade_mask.sum())
                
                if trades_taken == 0:
                    print(f"\n{model_name.upper()}: ⚠️  No trades predicted")
                    results[model_name] = {
                        'accuracy': acc,
                        'f1': f1,
                        'total_trades': 0,
                        'win_rate': 0.0,
                        'profit_factor': 0.0,
                        'sharpe': 0.0,
                        'max_drawdown_pct': 0.0,
                        'max_drawdown_r': 0.0,
                        'mean_r': 0.0,
                        'total_r': 0.0,
                        'period_return_pct': 0.0,
                        'annual_return_pct': 0.0,
                        'eligible': False,
                        'reason': 'No trades'
                    }
                    continue
                
                r_trades = r[trade_mask]
                metrics = RiskMetrics.calculate_all_metrics(r_trades, risk_per_trade=risk_per_trade)

                equity_curve, max_dd_pct, max_dd_r = compute_equity_and_dd(
                    r_trades, risk_per_trade=risk_per_trade
                )
                period_return_pct = (equity_curve[-1] - 1.0) * 100.0

                annual_return_pct = ModelEvaluator._estimate_linear_annual_return(
                    r_trades, risk_per_trade, days
                )

                # Eligibility criteria (softer for validation, stricter for test)
                eligible = True
                reason = "✅ Eligible"
                if metrics['total_trades'] < min_trades:
                    eligible = False
                    reason = f"❌ Low sample ({metrics['total_trades']} < {min_trades})"
                elif metrics['profit_factor'] < CONFIG.MIN_ACCEPTABLE_PF:
                    eligible = False
                    reason = f"❌ Low PF ({metrics['profit_factor']:.2f} < {CONFIG.MIN_ACCEPTABLE_PF:.2f})"
                elif metrics['max_drawdown_pct'] > CONFIG.MAX_ACCEPTABLE_DD:
                    eligible = False
                    reason = (f"❌ High DD ({metrics['max_drawdown_pct']:.1f}% "
                              f"> {CONFIG.MAX_ACCEPTABLE_DD:.1f}%)")
                elif metrics['win_rate'] < CONFIG.MIN_WIN_RATE:
                    eligible = False
                    reason = (f"❌ Win rate < {CONFIG.MIN_WIN_RATE:.0%} "
                              f"({metrics['win_rate']:.1%})")
                
                results[model_name] = {
                    'accuracy': acc,
                    'f1': f1,
                    'total_trades': metrics['total_trades'],
                    'win_rate': metrics['win_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'sharpe': metrics['sharpe'],
                    'max_drawdown_pct': metrics['max_drawdown_pct'],
                    'max_drawdown_r': metrics['max_drawdown_r'],
                    'mean_r': metrics['mean_r'],
                    'total_r': metrics['total_r'],
                    'period_return_pct': float(period_return_pct),
                    'annual_return_pct': float(annual_return_pct),
                    'eligible': eligible,
                    'reason': reason
                }
                
                print(f"\n{model_name.upper()}:")
                print(f"   Trades: {metrics['total_trades']:,} {reason}")
                print(f"   Period Return: {period_return_pct:.2f}%")
                print(f"   Expected Annual (linear): {annual_return_pct:.2f}%")
                print(f"   Win Rate: {metrics['win_rate']:.1%}")
                print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
                print(f"   Sharpe: {metrics['sharpe']:.2f}")
                print(f"   Max DD: {metrics['max_drawdown_pct']:.1f}% "
                      f"({metrics['max_drawdown_r']:.1f}R)")
                
            except Exception as e:
                print(f"\n{model_name.upper()}: ❌ Failed - {e}")
                results[model_name] = {
                    'accuracy': 0.0,
                    'f1': 0.0,
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe': 0.0,
                    'max_drawdown_pct': 0.0,
                    'max_drawdown_r': 0.0,
                    'mean_r': 0.0,
                    'total_r': 0.0,
                    'annual_return_pct': 0.0,
                    'eligible': False,
                    'reason': f'Exception: {e}'
                }
        
        return results
    
    @staticmethod
    def evaluate_all_models(models: Dict, X_test, y_test,
                            r_test: np.ndarray,
                            timeframe: str,
                            test_days: float,
                            risk_per_trade: float = CONFIG.RISK_PER_TRADE_EVAL) -> Dict:
        """
        Evaluate all models on the TEST set using raw predictions (p>=0.5)
        and apply strict guardrails.
        
        Models predict profit-based labels: class 1 = good trade (positive R),
        class 0 = bad/no-trade. We take trades when model predicts class 1.

        Guardrails for 'eligible':
          - total_trades >= MIN_TRADES_TEST
          - profit_factor >= MIN_ACCEPTABLE_PF
          - max_drawdown_pct <= MAX_ACCEPTABLE_DD
          - win_rate >= MIN_WIN_RATE
        """
        print(f"\n📊 EVALUATING MODELS ON TEST SET (RAW PREDICTIONS)")
        print(f"{'='*80}")
        min_trades = CONFIG.get_min_trades_raw(timeframe)
        print(f"   Minimum trades for eligibility: {min_trades}")
        print(f"   Min WR: {CONFIG.MIN_WIN_RATE:.0%}, "
              f"Min PF: {CONFIG.MIN_ACCEPTABLE_PF:.2f}, "
              f"Max DD: {CONFIG.MAX_ACCEPTABLE_DD:.1f}%\n")
        
        results = {}
        
        for model_name, model_dict in models.items():
            try:
                model = model_dict['model']
                scaler = model_dict['scaler']
                X_test_scaled = scaler.transform(X_test)

                y_pred = model.predict(X_test_scaled)
                
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # "Take trade" when model predicts class 1 (good trade = positive R)
                # Class 1 means model thinks this is a good trade based on profitability
                trade_mask = (y_pred == 1)

                trades_taken = int(trade_mask.sum())
                
                if trades_taken == 0:
                    print(f"\n{model_name.upper()}: ⚠️  No trades predicted on test set")
                    results[model_name] = {
                        'accuracy': acc,
                        'f1': f1,
                        'total_trades': 0,
                        'win_rate': 0.0,
                        'profit_factor': 0.0,
                        'sharpe': 0.0,
                        'max_drawdown_pct': 0.0,
                        'max_drawdown_r': 0.0,
                        'mean_r': 0.0,
                        'total_r': 0.0,
                        'period_return_pct': 0.0,
                        'annual_return_pct': 0.0,
                        'eligible': False,
                        'reason': 'No trades'
                    }
                    continue
                
                r_trades = r_test[trade_mask]
                metrics = RiskMetrics.calculate_all_metrics(r_trades, risk_per_trade=risk_per_trade)

                # Compute period return from equity (no insane compounding)
                equity_curve, max_dd_pct, max_dd_r = compute_equity_and_dd(
                    r_trades, risk_per_trade=risk_per_trade
                )
                period_return_pct = (equity_curve[-1] - 1.0) * 100.0

                # Use *linear* expected annual return for scoring
                annual_return_pct = ModelEvaluator._estimate_linear_annual_return(
                    r_trades, risk_per_trade, test_days
                )

                # ROBUST SELECTION CRITERIA
                eligible = True
                reason = "✅ Eligible"
                if metrics['total_trades'] < CONFIG.MIN_TRADES_TEST:
                    eligible = False
                    reason = f"❌ Low sample ({metrics['total_trades']} < {CONFIG.MIN_TRADES_TEST})"
                elif metrics['profit_factor'] < CONFIG.MIN_ACCEPTABLE_PF:
                    eligible = False
                    reason = f"❌ Low PF ({metrics['profit_factor']:.2f} < {CONFIG.MIN_ACCEPTABLE_PF:.2f})"
                elif metrics['max_drawdown_pct'] > CONFIG.MAX_ACCEPTABLE_DD:
                    eligible = False
                    reason = (f"❌ High DD ({metrics['max_drawdown_pct']:.1f}% "
                              f"> {CONFIG.MAX_ACCEPTABLE_DD:.1f}%)")
                elif metrics['win_rate'] < CONFIG.MIN_WIN_RATE:
                    eligible = False
                    reason = (f"❌ Win rate < {CONFIG.MIN_WIN_RATE:.0%} "
                              f"({metrics['win_rate']:.1%})")
                elif annual_return_pct < CONFIG.MIN_ANNUAL_RETURN * 100:
                    # We still record it, but mark as not production-viable
                    eligible = False
                    reason = (f"❌ Annual return < {CONFIG.MIN_ANNUAL_RETURN*100:.0f}% "
                              f"({annual_return_pct:.1f}%)")
                
                results[model_name] = {
                    'accuracy': acc,
                    'f1': f1,
                    'total_trades': metrics['total_trades'],
                    'win_rate': metrics['win_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'sharpe': metrics['sharpe'],
                    'max_drawdown_pct': metrics['max_drawdown_pct'],
                    'max_drawdown_r': metrics['max_drawdown_r'],
                    'mean_r': metrics['mean_r'],
                    'total_r': metrics['total_r'],
                    'period_return_pct': float(period_return_pct),
                    'annual_return_pct': float(annual_return_pct),
                    'eligible': eligible,
                    'reason': reason
                }
                
                print(f"\n{model_name.upper()}:")
                print(f"   Trades: {metrics['total_trades']:,} {reason}")
                print(f"   Period Return: {period_return_pct:.2f}%")
                print(f"   Expected Annual (linear): {annual_return_pct:.2f}%")
                print(f"   Win Rate: {metrics['win_rate']:.1%}")
                print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
                print(f"   Sharpe: {metrics['sharpe']:.2f}")
                print(f"   Max DD: {metrics['max_drawdown_pct']:.1f}% "
                      f"({metrics['max_drawdown_r']:.1f}R)")
                
            except Exception as e:
                print(f"\n{model_name.upper()}: ❌ Failed - {e}")
                results[model_name] = {
                    'accuracy': 0.0,
                    'f1': 0.0,
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe': 0.0,
                    'max_drawdown_pct': 0.0,
                    'max_drawdown_r': 0.0,
                    'mean_r': 0.0,
                    'total_r': 0.0,
                    'annual_return_pct': 0.0,
                    'eligible': False,
                    'reason': f'Exception: {e}'
                }
        
        return results
    
    @staticmethod
    def select_best_model(results: Dict) -> Optional[str]:
        """Select best model with V3 scoring (annual return + PF + WR / DD)."""
        print(f"\n🏆 SELECTING BEST MODEL")
        print(f"{'='*80}")
        eligible = {name: res for name, res in results.items() if res.get('eligible', False)}
        if not eligible:
            print(f"❌ NO ELIGIBLE MODELS UNDER GUARDRAILS!")
            if results:
                # Fallback: best PF (still marked mentally as 'research only')
                fallback_name = max(results.keys(),
                                    key=lambda k: results[k].get('profit_factor', 0.0))
                print(f"   ⚠️  Fallback (research): {fallback_name}")
                return fallback_name
            return None
        
        print(f"   Eligible models: {len(eligible)}/{len(results)}\n")
        
        best_name = None
        best_score = -np.inf
        
        print(f"{'Model':<20} {'AnnRet':>8} {'PF':>8} {'WR':>8} "
              f"{'DD':>8} {'Trades':>10} {'Score':>10}")
        print("-"*90)
        
        for name, res in eligible.items():
            pf = res['profit_factor']
            wr = res['win_rate']
            trades = res['total_trades']
            dd = res['max_drawdown_pct']
            annual_return = res.get('annual_return_pct', 0.0)

            # Score: heavily weight annual returns; aggressively penalize DD above target
            dd_penalty = dd * 50.0  # stronger penalty per percentage point of DD
            if dd > CONFIG.TARGET_MAX_DD:
                dd_penalty *= 3.0   # heavily penalize DD above target max (4%)
            annual_return_score = max(0.0, annual_return) ** 2  # square to emphasize

            score = (annual_return_score *
                     pf *
                     wr *
                     np.log(trades + 1.0) /
                     (1.0 + dd_penalty / 100.0))

            # Bonus if meeting targets
            if annual_return >= CONFIG.TARGET_ANNUAL_RETURN * 100:
                score *= 2.0
            elif annual_return >= CONFIG.MIN_ANNUAL_RETURN * 100:
                score *= 1.5
            elif annual_return < CONFIG.MIN_ANNUAL_RETURN * 100:
                score *= 0.1

            # Reward if DD is below our stricter TARGET_DRAWDOWN (3%)
            if dd < CONFIG.TARGET_DRAWDOWN * 100:
                score *= 1.3  # slightly higher reward for very low DD

            print(f"{name:<20} {annual_return:>7.1f}% {pf:>8.2f} {wr:>7.1%} "
                  f"{dd:>7.1f}% {trades:>10,} {score:>10.2f}")
            
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_name is not None:
            print(f"\n   ✅ Best model: {best_name} (Score: {best_score:.2f})")
        return best_name
    
    @staticmethod
    def evaluate_with_threshold(model_dict: Dict, X_test, y_test,
                                r_test: np.ndarray,
                                threshold: float,
                                test_days: float,
                                risk_per_trade: float = CONFIG.RISK_PER_TRADE_EVAL) -> Dict:
        """Evaluate with confidence threshold + annual return + DD."""
        model = model_dict['model']
        scaler = model_dict['scaler']
        X_test_scaled = scaler.transform(X_test)

        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        y_pred_filtered = (y_proba >= threshold).astype(int)
        
        trade_mask = (y_pred_filtered == 1)

        trades_taken = int(trade_mask.sum())
        
        if trades_taken == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown_r': 0.0,
                'annual_return_pct': 0.0,
                'period_return_pct': 0.0,
                'total_return_pct': 0.0
            }
        
        r_trades = r_test[trade_mask]

        metrics = RiskMetrics.calculate_all_metrics(r_trades, risk_per_trade=risk_per_trade)

        # Equity-based period return
        equity_curve, max_dd_pct, max_dd_r = compute_equity_and_dd(
            r_trades, risk_per_trade=risk_per_trade
        )
        period_return_pct = (equity_curve[-1] - 1.0) * 100.0

        # Linear expected annual return (no crazy compounding)
        annual_return_pct = ModelEvaluator._estimate_linear_annual_return(
            r_trades, risk_per_trade, test_days
        )

        metrics['max_drawdown_pct'] = max_dd_pct  # ensure aligned with equity calc
        metrics['max_drawdown_r'] = max_dd_r
        metrics['annual_return_pct'] = float(annual_return_pct)
        metrics['period_return_pct'] = float(period_return_pct)
        metrics['total_return_pct'] = float(period_return_pct)  # same as period return
        
        return metrics
    
    @staticmethod
    def print_comparison_table(results: Dict):
        """Print model comparison."""
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON (RAW PREDICTIONS)")
        print(f"{'='*80}")
        print(f"\n{'Model':<20} {'WinRate':>10} {'PF':>8} {'Sharpe':>8} "
              f"{'MaxDD':>10} {'Trades':>10} {'Status':<30}")
        print("-"*120)
        for model_name, metrics in sorted(
            results.items(),
            key=lambda x: x[1].get('profit_factor', 0.0),
            reverse=True
        ):
            print(f"{model_name:<20} "
                  f"{metrics.get('win_rate', 0.0):>9.1%} "
                  f"{metrics.get('profit_factor', 0.0):>8.2f} "
                  f"{metrics.get('sharpe', 0.0):>8.2f} "
                  f"{metrics.get('max_drawdown_pct', 0.0):>9.1f}% "
                  f"{metrics.get('total_trades', 0):>10,} "
                  f"{metrics.get('reason', 'N/A'):<30}")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIDENCE FILTERING
# ═══════════════════════════════════════════════════════════════════════════

class ConfidenceFilter:
    """Optimize confidence threshold on validation set."""
    
    @staticmethod
    def find_optimal_threshold(model_dict,
                               X_val, y_val,
                               r_val: np.ndarray,
                               timeframe: str,
                               val_days: float,
                               risk_per_trade: float = CONFIG.RISK_PER_TRADE_EVAL) -> Optional[float]:
        """
        Find optimal confidence threshold, enforcing WR / PF / DD / return.
        
        Threshold optimization is based on profit-based labels (good vs bad trades).
        Class 1 probability = P(good trade | features). We filter to high-confidence trades.
        """
        print(f"\n🎯 OPTIMIZING CONFIDENCE THRESHOLD")
        print(f"{'='*80}")
        min_trades = CONFIG.get_min_trades_filtered(timeframe)
        print(f"   Minimum trades: {min_trades}")
        print(f"   Max DD allowed: {CONFIG.MAX_DRAWDOWN:.1%}")
        print(f"   Min WR: {CONFIG.MIN_WIN_RATE:.0%}")
        print(f"   Min annual return: {CONFIG.MIN_ANNUAL_RETURN*100:.0f}%\n")
        
        model = model_dict['model']
        scaler = model_dict['scaler']
        X_val_scaled = scaler.transform(X_val)

        y_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        best_threshold = None
        best_score = -np.inf
        
        print(f"{'Threshold':>12} {'AnnRet':>8} {'WR':>8} {'PF':>8} "
              f"{'DD':>10} {'Trades':>10} {'Score':>10} {'Status':<20}")
        print("-"*120)
        
        for threshold in CONFIG.CONFIDENCE_THRESHOLDS:
            y_pred_filtered = (y_proba >= threshold).astype(int)
            
            trade_mask = (y_pred_filtered == 1)
            if trade_mask.sum() == 0:
                continue
            
            r_trades = r_val[trade_mask]

            metrics = RiskMetrics.calculate_all_metrics(r_trades, risk_per_trade=risk_per_trade)

            # Use linear expected annual return (no crazy compounding)
            annual_return_pct = ModelEvaluator._estimate_linear_annual_return(
                r_trades, risk_per_trade, val_days
            )
            metrics['annual_return_pct'] = annual_return_pct
            
            trades = metrics['total_trades']
            wr = metrics['win_rate']
            pf = metrics['profit_factor']
            dd = metrics['max_drawdown_pct']
            
            status = ""
            eligible = True
            
            if trades < min_trades:
                status = f"❌ Low trades (<{min_trades})"
                eligible = False
            elif annual_return_pct < CONFIG.MIN_ANNUAL_RETURN * 100:
                status = (f"❌ Low return ({annual_return_pct:.1f}% "
                          f"< {CONFIG.MIN_ANNUAL_RETURN*100:.0f}%)")
                eligible = False
            elif dd > CONFIG.MAX_DRAWDOWN * 100:
                status = f"❌ High DD (>{CONFIG.MAX_DRAWDOWN:.0%})"
                eligible = False
            elif pf < 1.0:
                status = "❌ Unprofitable (PF < 1.0)"
                eligible = False
            elif wr < CONFIG.MIN_WIN_RATE:
                status = (f"❌ WR < {CONFIG.MIN_WIN_RATE:.0%} "
                          f"({wr:.1%})")
                eligible = False
            else:
                status = "✅ Viable"
            
            # Score: heavily weight annual returns; HEAVILY penalize DD > 4%, discard DD > 6%
            if dd > CONFIG.MAX_DRAWDOWN * 100:
                # Outright discard thresholds with DD > 6%
                eligible = False
                status = f"❌ DD > {CONFIG.MAX_DRAWDOWN:.0%} (discarded)"
                dd_penalty = 1000.0  # Massive penalty
            elif dd > 4.0:
                # Heavily penalize DD > 4%
                dd_penalty = 100.0
            elif dd > CONFIG.TARGET_DRAWDOWN * 100:
                dd_penalty = 50.0
            else:
                # Lower penalty for DD < 3%
                dd_penalty = 10.0
            
            annual_return_score = max(0.0, annual_return_pct) ** 2

            score = (annual_return_score *
                     pf *
                     wr *
                     np.log(trades + 1.0) /
                     (1.0 + dd * dd_penalty / 100.0))
            
            # Apply DD bonus if applicable (reward low DD)
            if dd <= CONFIG.TARGET_DRAWDOWN * 100:
                score *= 1.3  # Reward for low DD

            if annual_return_pct >= CONFIG.MIN_ANNUAL_RETURN * 100:
                score *= 2.0

            print(f"{threshold:>12.2f} {annual_return_pct:>7.1f}% {wr:>7.1%} "
                  f"{pf:>8.2f} {dd:>9.1f}% {trades:>10,} {score:>10.2f} {status:<20}")
            
            if eligible and score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"\n{'='*80}")
        
        if best_threshold is None:
            print(f"❌ NO VALID THRESHOLD FOUND UNDER GUARDRAILS!")
            return None
        
        print(f"✅ Best threshold: {best_threshold:.2f} (Score: {best_score:.2f})")
        return best_threshold


# ═══════════════════════════════════════════════════════════════════════════
# REGIME ANALYSIS - WITH LIQUIDITY SWEEP BREAKDOWN (V2.4)
# ═══════════════════════════════════════════════════════════════════════════

class RegimeAnalyzer:
    """Analyze performance by regime with liquidity sweep breakdown."""
    
    @staticmethod
    def analyze_by_regime(df: pd.DataFrame, y_true, y_pred, r_test: np.ndarray) -> Dict:
        """Analyze by regime with LIQUIDITY SWEEP breakdown."""
        print(f"\n📊 REGIME-BASED PERFORMANCE ANALYSIS")
        print(f"{'='*80}")
        
        if 'regime_vol' in df.columns:
            regimes = df['regime_vol'].replace({0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'})
        elif 'atr' in df.columns:
            # Compare current ATR to historical distribution (exclude itself from comparison)
            atr_pct = df['atr'].shift(1).rolling(100).apply(lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5)
            regimes = pd.cut(atr_pct, bins=[0, 0.33, 0.67, 1.0], labels=['Low Vol', 'Med Vol', 'High Vol'])
        else:
            regimes = pd.Series('Unknown', index=df.index)
        
        regime_results = {}
        bad_regimes = []
        
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
            
            mask = regimes == regime
            regime_y_pred = y_pred[mask]
            
            trade_mask = regime_y_pred == 1
            trades_taken = trade_mask.sum()
            
            if trades_taken == 0:
                continue
            
            regime_r = r_test[mask][trade_mask]
            metrics = RiskMetrics.calculate_all_metrics(
                regime_r,
                risk_per_trade=CONFIG.RISK_PER_TRADE_EVAL
            )
            
            regime_results[str(regime)] = metrics
            
            print(f"\n{regime}:")
            print(f"   Win Rate: {metrics['win_rate']:.1%}")
            print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"   Sharpe: {metrics['sharpe']:.2f}")
            print(f"   Trades: {metrics['total_trades']:,}")
            
            # NEW V2.4: Liquidity sweep breakdown
            if 'liq_sweep_any' in df.columns:
                print(f"\n   Liquidity Sweep Breakdown:")
                for sweep_flag in [0, 1]:
                    sweep_label = "No Sweep" if sweep_flag == 0 else "Sweep Present"
                    mask2 = mask & (df['liq_sweep_any'] == sweep_flag)
                    regime_y_pred2 = y_pred[mask2]
                    trade_mask2 = regime_y_pred2 == 1
                    
                    if trade_mask2.sum() == 0:
                        continue
                    
                    regime_r2 = r_test[mask2][trade_mask2]
                    metrics2 = RiskMetrics.calculate_all_metrics(
                        regime_r2,
                        risk_per_trade=CONFIG.RISK_PER_TRADE_EVAL
                    )
                    
                    print(f"      {sweep_label}: "
                          f"WR={metrics2['win_rate']:.1%}, "
                          f"PF={metrics2['profit_factor']:.2f}, "
                          f"Trades={metrics2['total_trades']:,}")
            
            if (metrics['total_trades'] >= CONFIG.MIN_REGIME_TRADES and 
                (metrics['win_rate'] < CONFIG.BAD_REGIME_WR_THRESHOLD or 
                 metrics['profit_factor'] < CONFIG.BAD_REGIME_PF_THRESHOLD)):
                bad_regimes.append(str(regime))
                print(f"   ⚠️  BAD REGIME")
        
        if bad_regimes:
            print(f"\n🚫 BAD REGIMES: {bad_regimes}")
        else:
            print(f"\n✅ No bad regimes detected")
        
        return {
            'regime_stats': regime_results,
            'bad_regimes': bad_regimes
        }




# ═══════════════════════════════════════════════════════════════════════════
# MODEL SAVING FOR BACKTESTING (ADDED BY PATCH)
# ═══════════════════════════════════════════════════════════════════════════

def save_model_for_backtest(symbol: str, timeframe: str, best_model_name: str,
                            models: Dict, feature_cols: List[str], 
                            optimal_threshold: float,
                            base_path: Path = Path("ML_model/ML_model")):
    """
    Save trained model and metadata for backtesting.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        best_model_name: Name of best model
        models: Dict of trained models
        feature_cols: List of feature column names
        optimal_threshold: Optimal confidence threshold
        base_path: Base path for saving
    """
    # Create models directory
    models_dir = base_path / "models" / symbol
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best model
    model_path = models_dir / f"{symbol}_{timeframe}_best_model.pkl"
    best_model = models[best_model_name]
    
    joblib.dump(best_model, model_path)
    print(f"\n💾 Saved model to: {model_path}")
    
    # Save feature columns
    features_path = models_dir / f"{symbol}_{timeframe}_feature_cols.json"
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"💾 Saved feature columns to: {features_path}")
    
    # Save metadata
    metadata_path = models_dir / f"{symbol}_{timeframe}_metadata.json"
    metadata = {
        'symbol': symbol,
        'timeframe': timeframe,
        'best_model': best_model_name,
        'optimal_threshold': float(optimal_threshold),
        'n_features': len(feature_cols),
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata to: {metadata_path}")


# ═══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """Walk-forward validation - LEAK-FREE VERSION."""
    
    @staticmethod
    def run_walk_forward(raw_df: pd.DataFrame,
                        tp_mult: float,
                        time_barrier: int,
                        timeframe: str,
                        n_splits: int = 5) -> List[Dict]:
        """
        Execute leak-free walk-forward validation.
        
        Args:
            raw_df: Original OHLCV+ATR dataset BEFORE any feature engineering
            tp_mult: Best TP multiplier (optimized on train set only)
            time_barrier: Best time barrier (optimized on train set only)
            timeframe: Trading timeframe
            n_splits: Number of walk-forward folds
        
        Returns:
            List of fold results
        """
        print(f"\n🔄 WALK-FORWARD VALIDATION ({n_splits} folds) - LEAK-FREE")
        print(f"{'='*80}")
        print(f"   Using TP={tp_mult:.1f}x ATR, TB={time_barrier} bars")
        
        n_samples = len(raw_df)
        fold_size = n_samples // (n_splits + 1)
        
        results = []
        
        for i in range(n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = test_start + fold_size
            
            if test_end > n_samples:
                break
            
            # Slice raw data
            raw_df_train = raw_df.iloc[:train_end].copy()
            raw_df_test = raw_df.iloc[test_start:test_end].copy()
            
            # Engineer features independently on each fold
            df_train_fe = FeatureEngineer.engineer_all_features(raw_df_train)
            df_test_fe = FeatureEngineer.engineer_all_features(raw_df_test)
            
            # Label each fold independently
            labels_train, r_train = TripleBarrierLabeler.label(
                df_train_fe, tp_mult, CONFIG.SL_MULTIPLIER, time_barrier, timeframe
            )
            labels_test, r_test = TripleBarrierLabeler.label(
                df_test_fe, tp_mult, CONFIG.SL_MULTIPLIER, time_barrier, timeframe
            )
            
            # Filter to labeled trades
            train_mask = labels_train != 0
            test_mask = labels_test != 0
            
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                print(f"   ⚠️  Fold {i+1}: Skipped (no labeled trades)")
                continue
            
            # Convert to profit-based labels for training
            def profit_label_array(r_array: np.ndarray) -> np.ndarray:
                mask_good = (r_array > 0) & (np.abs(r_array) >= CONFIG.MIN_ABS_R_FOR_LABEL)
                return mask_good.astype(int)
            
            y_train_enc = profit_label_array(r_train[train_mask].values)
            y_test_enc = profit_label_array(r_test[test_mask].values)
            
            if len(np.unique(y_train_enc)) < 2 or len(np.unique(y_test_enc)) < 2:
                print(f"   ⚠️  Fold {i+1}: Skipped (imbalanced)")
                continue
            
            # Extract features
            feature_cols = [c for c in df_train_fe.columns 
                           if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            X_train = df_train_fe[train_mask][feature_cols].values
            X_test = df_test_fe[test_mask][feature_cols].values
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            classes = np.unique(y_train_enc)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train_enc)
            weight_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([weight_dict[yi] for yi in y_train_enc])
            scale_pos_weight = weight_dict[1] / weight_dict[0] if 0 in weight_dict and 1 in weight_dict else 1.0
            
            params = CONFIG.LGBM_PARAMS.copy()
            params['scale_pos_weight'] = scale_pos_weight
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled, y_train_enc, sample_weight=sample_weights)
            
            y_pred_enc = model.predict(X_test_scaled)
            
            trade_mask = y_pred_enc == 1
            if trade_mask.sum() == 0:
                print(f"   ⚠️  Fold {i+1}: No trades predicted")
                continue
            
            r_trades = r_test[test_mask].values[trade_mask]
            metrics = RiskMetrics.calculate_all_metrics(
                r_trades,
                risk_per_trade=CONFIG.RISK_PER_TRADE_EVAL
            )
            
            result = {
                'fold_num': i + 1,
                'train_dates': (raw_df_train['timestamp'].min(), raw_df_train['timestamp'].max()),
                'test_dates': (raw_df_test['timestamp'].min(), raw_df_test['timestamp'].max()),
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'sharpe': metrics['sharpe'],
                'total_trades': metrics['total_trades'],
                'max_dd': metrics['max_drawdown_pct'],
                'f1': f1_score(y_test_enc, y_pred_enc, zero_division=0)
            }
            
            results.append(result)
            
            print(f"\n   Fold {i+1}:")
            print(f"      WR: {result['win_rate']:.1%}, PF: {result['profit_factor']:.2f}, "
                  f"Sharpe: {result['sharpe']:.2f}, DD: {result['max_dd']:.1f}%")
        
        if results:
            avg_wr = np.mean([r['win_rate'] for r in results])
            std_wr = np.std([r['win_rate'] for r in results])
            avg_pf = np.mean([r['profit_factor'] for r in results])
            avg_dd = np.mean([r['max_dd'] for r in results])
            
            print(f"\n{'='*80}")
            print(f"WALK-FORWARD SUMMARY")
            print(f"{'='*80}")
            print(f"   Avg Win Rate: {avg_wr:.1%} ± {std_wr:.1%}")
            print(f"   Avg Profit Factor: {avg_pf:.2f}")
            print(f"   Avg Max DD: {avg_dd:.1f}%")
            
            if std_wr < 0.05:
                print(f"   ✅ Stable performance")
            elif std_wr < 0.10:
                print(f"   ⚠️  Moderate variance")
            else:
                print(f"   🚨 HIGH VARIANCE")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class TrainingPipeline:
    """Complete training pipeline V2.4 - FIXED SPREAD, DD, SELECTION."""
    
    def __init__(self, symbol: str, timeframe: str, 
                 enable_walk_forward: bool = False, 
                 enable_diagnostics: bool = False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.enable_walk_forward = enable_walk_forward
        self.enable_diagnostics = enable_diagnostics
        self.results = {}
    
    def run(self):
        """Execute complete pipeline."""
        print(f"\n{'#'*80}")
        print(f"# CITADEL ML TRAINING SYSTEM V2.4")
        print(f"# Symbol: {self.symbol} | Timeframe: {self.timeframe}")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")
        
        # Load data
        df, metadata = DataLoader.load_timeframe_data(self.symbol, self.timeframe)
        
        # CRITICAL FIX: Split data FIRST to prevent leakage
        # Do a preliminary chronological split to get train/val/test indices
        n = len(df)
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        
        df_train_raw = df.iloc[:train_end].copy()
        df_val_raw = df.iloc[train_end:val_end].copy()
        df_test_raw = df.iloc[val_end:].copy()
        
        print(f"\n✂️  PRELIMINARY CHRONOLOGICAL SPLIT (before feature engineering)")
        print(f"{'='*80}")
        print(f"   Train: {len(df_train_raw):,} bars ({df_train_raw['timestamp'].min()} to {df_train_raw['timestamp'].max()})")
        print(f"   Val:   {len(df_val_raw):,} bars ({df_val_raw['timestamp'].min()} to {df_val_raw['timestamp'].max()})")
        print(f"   Test:  {len(df_test_raw):,} bars ({df_test_raw['timestamp'].min()} to {df_test_raw['timestamp'].max()})")
        
        # Engineer features on each split independently (prevents rolling window leakage)
        print(f"\n🔧 Building features on each split independently...")
        df_train = FeatureEngineer.engineer_all_features(df_train_raw)
        df_val = FeatureEngineer.engineer_all_features(df_val_raw)
        df_test = FeatureEngineer.engineer_all_features(df_test_raw)
        
        # Find best labeling configuration on TRAIN set ONLY (no test set leakage)
        print(f"\n🔍 Optimizing labeling configuration on TRAIN set only...")
        best_tp, best_time_barrier = TripleBarrierLabeler.find_best_config(df_train, self.timeframe)
        
        # Label each split independently with the best config
        print(f"\n🏷️  Labeling each split independently...")
        labels_train, r_train = TripleBarrierLabeler.label(
            df_train, best_tp, CONFIG.SL_MULTIPLIER, best_time_barrier, self.timeframe
        )
        labels_val, r_val = TripleBarrierLabeler.label(
            df_val, best_tp, CONFIG.SL_MULTIPLIER, best_time_barrier, self.timeframe
        )
        labels_test, r_test = TripleBarrierLabeler.label(
            df_test, best_tp, CONFIG.SL_MULTIPLIER, best_time_barrier, self.timeframe
        )
        
        # Combine splits back for full dataset metrics (for reporting only)
        df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
        direction_labels = pd.concat([labels_train, labels_val, labels_test], ignore_index=True)
        r_post = pd.concat([r_train, r_val, r_test], ignore_index=True)
        
        # Print labeling stats
        labeled_mask = direction_labels != 0
        print(f"\n📊 FINAL LABELING STATISTICS")
        print(f"{'='*80}")
        print(f"   TP: {best_tp:.1f}x ATR")
        print(f"   Time Barrier: {best_time_barrier} bars")
        
        total_cost = get_total_cost_r(self.timeframe)
        print(f"   Total Cost: {total_cost:.3f}R (spread + slippage + fees)")
        
        r_sel = r_post[labeled_mask].values
        if len(r_sel) > 0:
            metrics = RiskMetrics.calculate_all_metrics(r_sel, risk_per_trade=CONFIG.RISK_PER_TRADE_EVAL)
            
            longs = (direction_labels[labeled_mask] == 1).sum()
            shorts = (direction_labels[labeled_mask] == -1).sum()
            
            print(f"   Post-cost PF: {metrics['profit_factor']:.2f}")
            print(f"   Win Rate: {metrics['win_rate']:.1%}")
            print(f"   Longs: {longs:,}, Shorts: {shorts:,}")
            
            if metrics['profit_factor'] < 1.0:
                print(f"\n   ⚠️  WARNING: Post-cost unprofitable!")
        else:
            print(f"   ⚠️  No labeled trades found")
        
        # Walk-forward validation (LEAK-FREE: use raw data, not engineered)
        if self.enable_walk_forward:
            # Use the original raw df from DataLoader (before any feature engineering)
            df_raw = pd.concat([df_train_raw, df_val_raw, df_test_raw], ignore_index=True)
            wf_results = WalkForwardValidator.run_walk_forward(
                raw_df=df_raw,
                tp_mult=best_tp,
                time_barrier=best_time_barrier,
                timeframe=self.timeframe,
                n_splits=CONFIG.WF_N_SPLITS
            )
        
        # Create splits dictionary using the independently processed data
        splits = DataSplitter.split_chronological_from_pre_split(
            df_train, df_val, df_test,
            labels_train, labels_val, labels_test,
            r_train, r_val, r_test
        )
        
        # SANITY CHECKS: Verify no temporal leakage
        print(f"\n🔍 SANITY CHECKS (LEAKAGE PREVENTION)")
        print(f"{'='*80}")
        df_train_labeled = splits['train_ts']
        df_val_labeled = splits['val_ts']
        df_test_labeled = splits['test_ts']
        
        assert df_train_labeled['timestamp'].max() < df_val_labeled['timestamp'].min(), \
            "❌ Train/Val temporal overlap detected!"
        assert df_val_labeled['timestamp'].max() < df_test_labeled['timestamp'].min(), \
            "❌ Val/Test temporal overlap detected!"
        print(f"   ✅ No temporal overlap between splits")
        
        # Verify feature engineering was done independently
        print(f"   Train date range: {df_train_labeled['timestamp'].min()} to {df_train_labeled['timestamp'].max()}")
        print(f"   Val date range:   {df_val_labeled['timestamp'].min()} to {df_val_labeled['timestamp'].max()}")
        print(f"   Test date range:  {df_test_labeled['timestamp'].min()} to {df_test_labeled['timestamp'].max()}")
        
        # Check class balance
        print(f"\n   Class balance (profit-based labels):")
        for name, y_split in [('Train', splits['y_train']), ('Val', splits['y_val']), ('Test', splits['y_test'])]:
            good_trades = (y_split == 1).sum()
            bad_trades = (y_split == 0).sum()
            total = len(y_split)
            print(f"   {name}: {good_trades:,} good ({good_trades/total*100:.1f}%), "
                  f"{bad_trades:,} bad ({bad_trades/total*100:.1f}%)")
            if good_trades == 0 or bad_trades == 0:
                print(f"      ⚠️  WARNING: {name} has only one class!")
        
        if CONFIG.DEBUG_LEAK_CHECK:
            print(f"\n   🔍 Running detailed leakage check...")
            # Sample a row from each split and verify rolling features
            for split_name, df_split in [('Train', df_train_labeled), ('Val', df_val_labeled), ('Test', df_test_labeled)]:
                if len(df_split) > 10:
                    test_idx = 10
                    test_row = df_split.iloc[test_idx]
                    test_ts = test_row['timestamp']
                    
                    # Check a rolling feature (e.g., ATR-based)
                    if 'atr' in df_split.columns:
                        # Recompute ATR rolling mean using only data up to this point
                        historical_data = df_split[df_split['timestamp'] <= test_ts]
                        if len(historical_data) > 20:
                            recomputed_atr_ma = historical_data['atr'].iloc[-20:].mean()
                            stored_atr_ma = test_row.get('atr_ma_20', None)
                            if stored_atr_ma is not None:
                                diff = abs(recomputed_atr_ma - stored_atr_ma)
                                if diff > 1e-6:
                                    print(f"      ⚠️  {split_name}: Possible rolling feature leakage detected "
                                          f"(diff={diff:.6f})")
                                else:
                                    print(f"      ✅ {split_name}: Rolling feature check passed")
        
        print(f"   ✅ Sanity checks complete")
        
        # Convert direction labels to profit-based binary labels for ML training
        # Profit labels: 1 = positive R (good trade), 0 = zero or negative R (bad/no-trade)
        def profit_label_array(r_array: np.ndarray) -> np.ndarray:
            """Create profit-based labels: 1 = good trade (R > 0 and |R| >= threshold), 0 = bad/no-trade."""
            mask_good = (r_array > 0) & (np.abs(r_array) >= CONFIG.MIN_ABS_R_FOR_LABEL)
            return mask_good.astype(int)
        
        # Override y_* with profit-based labels (replace direction labels)
        splits['y_train'] = profit_label_array(splits['r_train'])
        splits['y_val'] = profit_label_array(splits['r_val'])
        splits['y_test'] = profit_label_array(splits['r_test'])
        
        # For profit-based labels, y_* are already binary 0/1 (0 = bad/no-trade, 1 = good trade)
        y_train_enc = splits['y_train']
        y_val_enc = splits['y_val']
        y_test_enc = splits['y_test']
        
        # Check class distribution
        unique_train = np.unique(y_train_enc)
        print(f"\n📊 PROFIT-BASED LABEL DISTRIBUTION (TRAIN):")
        print(f"   Good trades (1): {(y_train_enc == 1).sum():,} samples ({(y_train_enc == 1).sum()/len(y_train_enc)*100:.1f}%)")
        print(f"   Bad/no-trade (0): {(y_train_enc == 0).sum():,} samples ({(y_train_enc == 0).sum()/len(y_train_enc)*100:.1f}%)")
        
        # Check if we have at least 2 classes
        if len(unique_train) < 2:
            print(f"\n❌ ERROR: Only one class found in training data!")
            print(f"   Cannot train binary classification models.")
            self.results = {
                'viable': False,
                'timeframe': self.timeframe,
                'reason': f'Only one class in training data: {unique_train}'
            }
            return self.results
        
        # Train models
        models = ModelFactory.train_all_models(
            splits['X_train'], splits['X_val'],
            y_train_enc, y_val_enc
        )
        
        # (a) Validation evaluation for model selection (NO TEST SET SNOOPING)
        print(f"\n{'='*80}")
        print(f"MODEL SELECTION ON VALIDATION SET")
        print(f"{'='*80}")
        val_results = ModelEvaluator.evaluate_all_models_on_split(
            models,
            splits['X_val'],
            y_val_enc,
            splits['r_val'],
            self.timeframe,
            days=splits.get('val_days', 365.0),
            risk_per_trade=CONFIG.RISK_PER_TRADE_EVAL
        )
        
        # Select best *family* based on VALIDATION results only
        best_model_name = ModelEvaluator.select_best_model(val_results)
        
        # (b) Test evaluation for reporting only (do NOT re-select model)
        print(f"\n{'='*80}")
        print(f"TEST SET EVALUATION (REPORTING ONLY)")
        print(f"{'='*80}")
        results_raw = ModelEvaluator.evaluate_all_models_on_split(
            {best_model_name: models[best_model_name]},
            splits['X_test'],
            y_test_enc,
            splits['r_test'],
            self.timeframe,
            days=splits.get('test_days', 365.0),
            risk_per_trade=CONFIG.RISK_PER_TRADE_EVAL
        )
        if best_model_name is None:
            print(f"\n❌ NO VIABLE MODEL FAMILY FOR {self.timeframe}")
            self.results = {
                'viable': False,
                'timeframe': self.timeframe,
                'reason': 'No model passed guardrails'
            }
            return self.results
        
        best_model = models[best_model_name]
        
        # Optimize threshold on VALIDATION set
        optimal_threshold = ConfidenceFilter.find_optimal_threshold(
            best_model,
            splits['X_val'],
            y_val_enc,  # Use encoded labels
            splits['r_val'],
            self.timeframe,
            splits.get('val_days', 365.0),
            risk_per_trade=CONFIG.RISK_PER_TRADE_EVAL
        )
        
        if optimal_threshold is None:
            print(f"\n❌ NO VALID THRESHOLD FOR {self.timeframe}")
            self.results = {
                'viable': False,
                'timeframe': self.timeframe,
                'best_model_name': best_model_name,
                'reason': 'No threshold meets WR/PF/DD/return on validation'
            }
            return self.results
        
        # Final post-threshold evaluation on TEST set
        print(f"\n{'='*80}")
        print(f"POST-THRESHOLD TEST SET EVALUATION")
        print(f"{'='*80}")
        filtered_metrics = ModelEvaluator.evaluate_with_threshold(
            best_model,
            splits['X_test'],
            y_test_enc,  # Use encoded labels
            splits['r_test'],
            optimal_threshold,
            test_days=splits.get('test_days', 365.0),
            risk_per_trade=CONFIG.RISK_PER_TRADE_EVAL
        )
        
        print(f"\n📊 FILTERED PERFORMANCE (TEST):")
        print(f"   Trades: {filtered_metrics['total_trades']}")
        if 'period_return_pct' in filtered_metrics:
            print(f"   Period Return: {filtered_metrics.get('period_return_pct', 0.0):.2f}%")
        print(f"   Expected Annual (linear): {filtered_metrics.get('annual_return_pct', 0.0):.2f}%")
        print(f"   Win Rate: {filtered_metrics['win_rate']:.1%}")
        print(f"   Profit Factor: {filtered_metrics['profit_factor']:.2f}")
        print(f"   Sharpe: {filtered_metrics['sharpe']:.2f}")
        print(f"   Max DD: {filtered_metrics['max_drawdown_pct']:.1f}%")
        
        annual_return = filtered_metrics.get('annual_return_pct', 0.0)
        if (annual_return >= CONFIG.MIN_ANNUAL_RETURN * 100 and
                filtered_metrics['win_rate'] >= CONFIG.MIN_WIN_RATE and
                filtered_metrics['profit_factor'] >= CONFIG.MIN_ACCEPTABLE_PF and
                filtered_metrics['max_drawdown_pct'] <= CONFIG.MAX_ACCEPTABLE_DD):
            print(f"   ✅ Meets all production guardrails "
                  f"(return / WR / PF / DD)")
            production_viable = True
        else:
            print(f"   ⚠️  Does NOT meet one or more production guardrails")
            production_viable = False
        
        # Regime analysis with liquidity sweep breakdown (TEST only)
        # Use the engineered test dataframe from splits (already has correct indexing)
        df_test_fe = splits['test_ts']  # This already has engineered features and correct indexing
        
        X_test_scaled = best_model['scaler'].transform(splits['X_test'])
        # Predictions are already 0/1 (profit-based labels)
        y_test_pred = best_model['model'].predict(X_test_scaled)
        
        # Ensure alignment: df_test_fe, y_test, y_test_pred, and r_test all refer to same rows
        # splits['test_ts'] is already filtered to labeled trades only, so indices should align
        regime_analysis = RegimeAnalyzer.analyze_by_regime(
            df_test_fe, splits['y_test'], y_test_pred, splits['r_test']
        )
        
        # Print raw comparison table across families
        ModelEvaluator.print_comparison_table(results_raw)
        
        # Store results
        self.results = {
            'viable': production_viable,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'best_tp_mult': best_tp,
            'time_barrier': best_time_barrier,
            'total_cost_r': get_total_cost_r(self.timeframe),
            'optimal_threshold': optimal_threshold,
            'best_model_name': best_model_name,
            'models': models,
            'results_raw': results_raw,
            'filtered_metrics': filtered_metrics,
            'regime_analysis': regime_analysis,
            'feature_cols': splits['feature_cols']
        }
        
        # SAVE MODEL ONLY IF PRODUCTION-VIABLE (live-trade candidate)
        if production_viable:
            try:
                save_model_for_backtest(
                    self.symbol,
                    self.timeframe,
                    best_model_name,
                    models,
                    splits['feature_cols'],
                    optimal_threshold
                )
                print(f"\n✅ Model saved - Meets production guardrails "
                      f"(AnnRet={annual_return:.2f}%, WR={filtered_metrics['win_rate']:.1%}, "
                      f"PF={filtered_metrics['profit_factor']:.2f}, "
                      f"DD={filtered_metrics['max_drawdown_pct']:.1f}%)")
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to save model for backtest: {e}")
        else:
            print(f"\n⚠️  Model NOT saved - Fails one or more guardrails")
        
        print(f"\n{'#'*80}")
        print(f"# TRAINING COMPLETE")
        print(f"# Best Model Family: {best_model_name}")
        print(f"# Expected Annual Return (linear): {annual_return:.2f}%")
        print(f"# Filtered WR: {filtered_metrics['win_rate']:.1%}")
        print(f"# Filtered PF: {filtered_metrics['profit_factor']:.2f}")
        print(f"# Max DD: {filtered_metrics['max_drawdown_pct']:.1f}%")
        if production_viable:
            print(f"# ✅ PRODUCTION-CANDIDATE MODEL SAVED")
        else:
            print(f"# ⚠️  RESEARCH-ONLY MODEL (NOT SAVED)")
        print(f"# Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}\n")
        
        return self.results


# ═══════════════════════════════════════════════════════════════════════════
# POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════

def position_size_from_confidence(confidence: float,
                                  account_equity: float,
                                  max_risk_pct: float = 0.01,
                                  min_risk_pct: float = 0.002) -> float:
    """
    Map model confidence [0.5..1.0] to a risk-per-trade % of equity.

    - Below 0.5 we don't trade.
    - Between 0.5 and 0.9, scale risk linearly from min_risk_pct to max_risk_pct.
    - Above 0.9, cap at max_risk_pct.

    Returns:
        risk_amount (cash) = equity * risk_pct
    """
    if confidence < 0.5:
        return 0.0

    conf_clipped = min(confidence, 0.9)
    alpha = (conf_clipped - 0.5) / (0.9 - 0.5 + 1e-8)
    risk_pct = min_risk_pct + alpha * (max_risk_pct - min_risk_pct)
    risk_pct = min(max_risk_pct, max(min_risk_pct, risk_pct))

    return account_equity * risk_pct


# ═══════════════════════════════════════════════════════════════════════════
# LIVE SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_live_signal(symbol: str,
                         timeframe: str,
                         latest_df: pd.DataFrame,
                         account_equity: float) -> Dict[str, Any]:
    """
    Live signal generator.

    Args:
        symbol: e.g. 'XAUUSD'
        timeframe: e.g. '5T'
        latest_df: DataFrame with the MOST RECENT bars + features
        account_equity: current account equity in base currency

    Returns:
        dict with keys:
            'side': 'long' / 'short' / 'flat'
            'confidence': float
            'entry_price': float
            'tp_price': float
            'sl_price': float
            'risk_amount': float
            'valid': bool
            'reason': str
    """
    models_dir = Path("ML_model/ML_model/models") / symbol
    model_path = models_dir / f"{symbol}_{timeframe}_best_model.pkl"
    meta_path = models_dir / f"{symbol}_{timeframe}_metadata.json"
    feature_cols_path = models_dir / f"{symbol}_{timeframe}_feature_cols.json"

    if not model_path.exists() or not meta_path.exists() or not feature_cols_path.exists():
        return {'valid': False, 'reason': 'Model or metadata not found', 'side': 'flat'}

    best_model = joblib.load(model_path)

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    with open(feature_cols_path, 'r') as f:
        feature_cols = json.load(f)

    threshold = float(meta.get('optimal_threshold', 0.5))

    # Use last row as the decision row
    last_row = latest_df.iloc[-1].copy()
    if not set(feature_cols).issubset(latest_df.columns):
        return {'valid': False, 'reason': 'Feature mismatch', 'side': 'flat'}

    X_live = last_row[feature_cols].values.reshape(1, -1)
    scaler = best_model['scaler']
    model = best_model['model']
    X_live_scaled = scaler.transform(X_live)

    # NOTE: The current model is trained on profit-based labels (good vs bad trade).
    # Class 1 probability = P(good trade | features). For now, generate_live_signal
    # interprets class 1 as "take trade" and uses a fixed 'long' side.
    # Directional modeling can be added in a later revision.
    proba = model.predict_proba(X_live_scaled)[0]
    # proba[1] = prob of class "1" (good trade)
    p1 = float(proba[1])

    if p1 < threshold:
        return {
            'valid': True,
            'side': 'flat',
            'confidence': p1,
            'entry_price': float(last_row['close']),
            'tp_price': float(last_row['close']),
            'sl_price': float(last_row['close']),
            'risk_amount': 0.0,
            'reason': 'Below threshold'
        }

    # For now, interpret class 1 as "take trade" and use fixed 'long' side
    # TODO: Add directional modeling in future revision
    side = 'long'
    entry_price = float(last_row['close'])

    # Use saved TP/SL config from metadata if you saved it, else fallback
    tp_mult = meta.get('best_tp_mult', CONFIG.get_tp_multipliers(timeframe)[0])
    sl_mult = meta.get('sl_mult', CONFIG.SL_MULTIPLIER)

    atr = float(last_row['atr'])
    if np.isnan(atr) or atr <= 0:
        return {'valid': False, 'reason': 'Invalid ATR', 'side': 'flat'}

    if side == 'long':
        tp_price = entry_price + tp_mult * atr
        sl_price = entry_price - sl_mult * atr
    else:
        tp_price = entry_price - tp_mult * atr
        sl_price = entry_price + sl_mult * atr

    risk_amount = position_size_from_confidence(
        p1, account_equity,
        max_risk_pct=CONFIG.RISK_PER_TRADE_EVAL,
        min_risk_pct=0.002
    )

    return {
        'valid': True,
        'side': side,
        'confidence': p1,
        'entry_price': entry_price,
        'tp_price': float(tp_price),
        'sl_price': float(sl_price),
        'risk_amount': float(risk_amount),
        'reason': 'Signal above threshold'
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Citadel ML Training System V2.4')
    parser.add_argument('--symbol', type=str, default='XAUUSD')
    parser.add_argument('--timeframe', type=str, help='Single timeframe')
    parser.add_argument('--all-timeframes', action='store_true')
    parser.add_argument('--walk-forward', action='store_true')
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--full-system', action='store_true')
    
    args = parser.parse_args()
    
    if args.full_system:
        args.walk_forward = True
        args.diagnose = True
    
    if args.all_timeframes:
        timeframes = ['5T', '15T', '30T', '1H']
    elif args.timeframe:
        timeframes = [args.timeframe]
    else:
        parser.print_help()
        return
    
    all_results = {}
    
    for timeframe in timeframes:
        try:
            pipeline = TrainingPipeline(
                args.symbol, timeframe,
                enable_walk_forward=args.walk_forward,
                enable_diagnostics=args.diagnose
            )
            results = pipeline.run()
            all_results[timeframe] = results
            
        except Exception as e:
            print(f"\n❌ ERROR in {timeframe}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY - {args.symbol} (POST-THRESHOLD)")
    print(f"{'='*80}")
    
    print(f"\n{'TF':<6} {'Model':<15} {'WR':>8} {'PF':>8} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8} {'Status':<15}")
    print("-"*110)
    
    for tf, result in all_results.items():
        if result.get('viable', False) and 'filtered_metrics' in result:
            fm = result['filtered_metrics']
            status = "✅ Production"
            if fm['profit_factor'] < CONFIG.MIN_ACCEPTABLE_PF:
                status = "⚠️  Marginal PF"
            if fm['max_drawdown_pct'] > CONFIG.MAX_ACCEPTABLE_DD:
                status = "❌ High DD"
            elif fm['max_drawdown_pct'] > CONFIG.TARGET_DRAWDOWN * 100:
                status = "⚠️  DD Warning"
            
            print(f"{tf:<6} {result['best_model_name']:<15} "
                  f"{fm['win_rate']:>7.1%} "
                  f"{fm['profit_factor']:>8.2f} "
                  f"{fm['sharpe']:>8.2f} "
                  f"{fm['max_drawdown_pct']:>9.1f}% "
                  f"{fm['total_trades']:>8,} "
                  f"{status:<15}")
        else:
            print(f"{tf:<6} {'N/A':<15} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'❌ Not viable':<15}")
    
    print(f"\n✅ ALL TRAINING COMPLETE\n")


if __name__ == '__main__':
    main()