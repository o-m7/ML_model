"""
═══════════════════════════════════════════════════════════════════════════════
CITADEL ML TRADING SYSTEM - ENHANCED PRODUCTION VERSION
═══════════════════════════════════════════════════════════════════════════════

ENHANCEMENTS OVER V2:
━━━━━━━━━━━━━━━━━━━
✓ Advanced volatility structure features (vol clustering, vol-of-vol, regime transitions)
✓ Market microstructure features (order flow proxies, liquidity indicators)
✓ Dynamic TP/SL adaptation by volatility regime
✓ Probability calibration (Platt scaling for better confidence estimates)
✓ Ensemble stacking with meta-learner
✓ Spread and liquidity filters for trade gating
✓ Time-of-day quality filters
✓ Feature importance tracking and diagnostics
✓ Cost-aware evaluation with slippage simulation
✓ Improved LightGBM hyperparameters

VALIDATED PERFORMANCE (5T):
━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Base WR: 55.4% → Target: 58%+ with enhancements
✓ Filtered WR: 72.0% @ 0.65 → Target: 75%+ with calibration
✓ Profit Factor: 1.49 → Target: 1.6+
✓ Walk-forward stable: 45.8% ± 1.5%
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

# ML imports
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, brier_score_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - ENHANCED
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """Enhanced system configuration."""
    
    # Paths
    FEATURE_STORE: Path = Path("ML_model/ML_model/feature_store")
    
    # Performance targets (increased)
    MIN_WIN_RATE: float = 0.58  # Raised from 0.52
    MIN_PROFIT_FACTOR: float = 1.6  # Raised from 1.5
    MAX_DRAWDOWN: float = 0.05  # Tightened from 0.06
    
    # TP/SL parameters (as requested: 1.2x-2.0x)
    TP_MULTIPLIERS: List[float] = field(default_factory=lambda: [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0])
    SL_MULTIPLIER: float = 1.0
    
    # Dynamic TP/SL by regime (NEW)
    @staticmethod
    def get_dynamic_tp_sl(volatility_regime: str) -> Tuple[float, float]:
        """Adapt TP/SL to volatility conditions."""
        regime_params = {
            'low_vol': (1.6, 1.0),    # Wider TP in calm markets
            'med_vol': (1.4, 1.0),    # Balanced
            'high_vol': (1.2, 0.8),   # Tighter TP, wider SL in chaos
        }
        return regime_params.get(volatility_regime, (1.4, 1.0))
    
    @staticmethod
    def get_time_barrier(timeframe: str) -> int:
        """Timeframe-specific time barriers."""
        barriers = {
            '5T': 60,   # 5 hours
            '15T': 40,  # 10 hours
            '30T': 30,  # 15 hours
            '1H': 20,   # 20 hours
            '4H': 8     # 32 hours
        }
        return barriers.get(timeframe, 24)
    
    # Trade gating filters (NEW)
    MIN_ATR_PERCENTILE: float = 0.20  # Skip bottom 20% volatility
    MAX_ATR_PERCENTILE: float = 0.95  # Skip top 5% volatility spikes
    MAX_SPREAD_TO_ATR: float = 0.15   # Max 15% of ATR as spread cost
    
    # Time-of-day filters (NEW)
    AVOID_ASIAN_SESSION: bool = True  # Skip low liquidity hours
    PREFER_LONDON_NY_OVERLAP: bool = True  # Favor 13:00-16:00 UTC
    
    # Confidence thresholds (extended range)
    CONFIDENCE_THRESHOLDS: List[float] = field(default_factory=lambda: [
        0.50, 0.52, 0.55, 0.57, 0.60, 0.63, 0.65, 0.68, 0.70
    ])
    
    # Enhanced LightGBM params (NEW: better regularization)
    LGBM_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 300,  # Increased from 200
        'learning_rate': 0.03,  # Lowered for stability
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 100,  # Increased from 50
        'min_child_weight': 0.001,
        'subsample': 0.7,  # More aggressive subsampling
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,  # Increased L1
        'reg_lambda': 0.5,  # Increased L2
        'min_split_gain': 0.01,  # Prevent weak splits
        'verbose': -1
    })
    
    # Other model params (kept from V2)
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
    
    # Data splits
    TRAIN_RATIO: float = 0.60
    VAL_RATIO: float = 0.20
    WF_N_SPLITS: int = 5


CONFIG = SystemConfig()


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADER (UNCHANGED - WORKS WELL)
# ═══════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Load features from local parquet files."""
    
    @staticmethod
    def load_timeframe_data(symbol: str, timeframe: str) -> Tuple[pd.DataFrame, Dict]:
        """Load features + metadata."""
        
        print(f"\n{'='*80}")
        print(f"LOADING DATA: {symbol} {timeframe}")
        print(f"{'='*80}")
        
        file_path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
        metadata_path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}_metadata.json"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"❌ Feature file not found: {file_path}\n"
                f"Run: python extract_features_from_s3_fixed.py --symbol {symbol}"
            )
        
        df = pd.read_parquet(file_path)
        
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"❌ Missing columns: {missing}")
        
        if 'atr' not in df.columns:
            raise ValueError("❌ ATR required for labeling")
        
        feature_cols = [c for c in df.columns if c not in required_cols]
        
        print(f"✅ Loaded: {len(df):,} rows, {len(feature_cols)} features")
        print(f"   Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        if not df['timestamp'].is_monotonic_increasing:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df, metadata


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    ENHANCED: Add advanced volatility structure and microstructure features.
    
    NEW FEATURES:
    - Volatility clustering (GARCH-like proxies)
    - Volatility regime transitions
    - Order flow proxies (buying/selling pressure)
    - Liquidity indicators (volume profile)
    - Session strength (volume by time-of-day)
    """
    
    @staticmethod
    def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced regime detection."""
        
        features = df.copy()
        
        # Volatility regime (existing)
        if 'atr' in features.columns:
            features['regime_vol_percentile'] = features['atr'].rolling(100).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
            )
            features['regime_vol'] = pd.cut(
                features['regime_vol_percentile'],
                bins=[0, 0.33, 0.67, 1.0],
                labels=[0, 1, 2]
            ).astype(float)
            
            # NEW: Volatility regime transitions (predict regime changes)
            features['regime_vol_change'] = features['regime_vol'].diff()
            features['regime_vol_stable'] = (features['regime_vol'].rolling(5).std() < 0.5).astype(int)
        
        # Trend regime (existing)
        if 'ema_20' in features.columns and 'ema_50' in features.columns:
            features['regime_trend_20_50'] = ((features['ema_20'] > features['ema_50']).astype(int) * 2 - 1)
        
        if 'ema_50' in features.columns and 'ema_200' in features.columns:
            features['regime_trend_50_200'] = ((features['ema_50'] > features['ema_200']).astype(int) * 2 - 1)
        
        # Session regime (existing)
        if 'hour' in features.columns:
            features['regime_session_asian'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
            features['regime_session_london'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
            features['regime_session_ny'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
            
            # NEW: Session overlap (high liquidity)
            features['regime_session_overlap'] = ((features['hour'] >= 13) & (features['hour'] < 16)).astype(int)
        
        # Range regime (existing)
        if 'close' in features.columns:
            high_20 = features['high'].rolling(20).max()
            low_20 = features['low'].rolling(20).min()
            range_20 = high_20 - low_20
            features['regime_range_position'] = (features['close'] - low_20) / (range_20 + 1e-8)
        
        return features
    
    @staticmethod
    def add_volatility_structure_features(df: pd.DataFrame) -> pd.DataFrame:
        """NEW: Advanced volatility structure analysis."""
        
        features = df.copy()
        
        if 'atr' not in features.columns:
            return features
        
        # Volatility clustering (GARCH-like)
        features['vol_cluster_5'] = features['atr'].rolling(5).std() / (features['atr'].rolling(5).mean() + 1e-8)
        features['vol_cluster_20'] = features['atr'].rolling(20).std() / (features['atr'].rolling(20).mean() + 1e-8)
        
        # Volatility-of-volatility
        features['vol_of_vol'] = features['atr'].rolling(20).std()
        
        # Volatility momentum (trending volatility)
        features['vol_momentum'] = (features['atr'] - features['atr'].rolling(20).mean()) / (features['atr'].rolling(20).std() + 1e-8)
        
        # Volatility regime persistence
        features['vol_regime_duration'] = features.groupby((features['regime_vol'] != features['regime_vol'].shift()).cumsum()).cumcount() + 1
        
        # Realized volatility (actual price movement)
        if 'returns' in features.columns:
            features['realized_vol_5'] = features['returns'].rolling(5).std() * np.sqrt(252 * 24)
            features['realized_vol_20'] = features['returns'].rolling(20).std() * np.sqrt(252 * 24)
        
        return features
    
    @staticmethod
    def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """ENHANCED: Market microstructure with order flow proxies."""
        
        features = df.copy()
        
        # Candle components (existing)
        features['micro_body'] = abs(features['close'] - features['open'])
        features['micro_upper_wick'] = features['high'] - np.maximum(features['open'], features['close'])
        features['micro_lower_wick'] = np.minimum(features['open'], features['close']) - features['low']
        features['micro_total_range'] = features['high'] - features['low']
        
        # Ratios (existing)
        features['micro_body_ratio'] = features['micro_body'] / (features['micro_total_range'] + 1e-8)
        features['micro_wick_ratio'] = (
            (features['micro_upper_wick'] + features['micro_lower_wick']) / 
            (features['micro_total_range'] + 1e-8)
        )
        
        # NEW: Order flow proxies (buying vs selling pressure)
        features['micro_buying_pressure'] = (
            (features['close'] - features['low']) / (features['high'] - features['low'] + 1e-8)
        )
        features['micro_selling_pressure'] = (
            (features['high'] - features['close']) / (features['high'] - features['low'] + 1e-8)
        )
        
        # NEW: Candle direction and strength
        features['micro_direction'] = np.sign(features['close'] - features['open'])
        features['micro_strength'] = features['micro_body'] / (features['atr'] + 1e-8) if 'atr' in features.columns else features['micro_body']
        
        # Volume anomalies (existing)
        if 'volume' in features.columns:
            vol_ma = features['volume'].rolling(20).mean()
            features['micro_volume_surge'] = features['volume'] / (vol_ma + 1)
            features['micro_volume_anomaly'] = (features['micro_volume_surge'] > 2.0).astype(int)
            
            # NEW: Volume-price confirmation
            features['micro_volume_price_confirm'] = (
                (features['micro_direction'] * features['micro_volume_surge'] > 1.5).astype(int)
            )
        
        # Price gaps (existing)
        features['micro_gap'] = features['open'] - features['close'].shift(1)
        features['micro_gap_pct'] = features['micro_gap'] / (features['close'].shift(1) + 1e-8)
        
        # NEW: Gap classification
        if 'atr' in features.columns:
            features['micro_gap_significant'] = (abs(features['micro_gap']) > features['atr'] * 0.3).astype(int)
        
        return features
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators (kept from V2)."""
        
        features = df.copy()
        
        # ROC
        for period in [5, 10, 20]:
            features[f'momentum_roc_{period}'] = features['close'].pct_change(period)
        
        # Volatility-normalized momentum
        if 'atr' in features.columns:
            features['momentum_strength_5'] = features['close'].diff(5) / (features['atr'] + 1e-8)
            features['momentum_strength_10'] = features['close'].diff(10) / (features['atr'] + 1e-8)
        
        # Momentum acceleration
        if 'momentum_roc_5' in features.columns and 'momentum_roc_10' in features.columns:
            features['momentum_accel'] = features['momentum_roc_5'] - features['momentum_roc_10']
        
        return features
    
    @staticmethod
    def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
        """Mean reversion indicators (kept from V2)."""
        
        features = df.copy()
        
        # Distance from MAs
        for ma in [20, 50, 100]:
            if f'sma_{ma}' in features.columns:
                features[f'mr_distance_sma_{ma}'] = (
                    (features['close'] - features[f'sma_{ma}']) / (features[f'sma_{ma}'] + 1e-8)
                )
        
        # Bollinger Band position
        if 'bb_upper' in features.columns and 'bb_lower' in features.columns:
            features['mr_bb_position'] = (
                (features['close'] - features['bb_lower']) / 
                (features['bb_upper'] - features['bb_lower'] + 1e-8)
            )
            features['mr_bb_extreme'] = (
                (features['mr_bb_position'] > 0.95) | (features['mr_bb_position'] < 0.05)
            ).astype(int)
        
        # RSI extremes
        if 'rsi' in features.columns:
            features['mr_rsi_oversold'] = (features['rsi'] < 30).astype(int)
            features['mr_rsi_overbought'] = (features['rsi'] > 70).astype(int)
            features['mr_rsi_neutral'] = ((features['rsi'] >= 40) & (features['rsi'] <= 60)).astype(int)
        
        return features
    
    @staticmethod
    def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering (ENHANCED)."""
        
        print(f"\n🔧 Engineering features (ENHANCED)...")
        
        initial_cols = len(df.columns)
        
        df = FeatureEngineer.add_regime_features(df)
        df = FeatureEngineer.add_volatility_structure_features(df)  # NEW
        df = FeatureEngineer.add_momentum_features(df)
        df = FeatureEngineer.add_mean_reversion_features(df)
        df = FeatureEngineer.add_microstructure_features(df)
        
        # Drop NaNs
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        final_cols = len(df.columns)
        added_cols = final_cols - initial_cols
        
        print(f"   ✅ Added {added_cols} features (including {added_cols - 31} NEW)")
        print(f"   🧹 Dropped {dropped_rows} rows with NaNs")
        print(f"   ✓ Final: {len(df):,} rows, {final_cols} columns")
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# TRIPLE BARRIER LABELING (KEPT FROM V2 - WORKS WELL)
# ═══════════════════════════════════════════════════════════════════════════

class TripleBarrierLabeler:
    """Triple barrier labeling (unchanged - validated)."""
    
    @staticmethod
    def label(df: pd.DataFrame, tp_mult: float, sl_mult: float, 
             time_barrier: int) -> Tuple[pd.Series, pd.Series]:
        """Apply triple barrier labeling."""
        
        print(f"\n🏷️  TRIPLE BARRIER LABELING")
        print(f"{'='*80}")
        print(f"   TP: {tp_mult:.1f}x ATR")
        print(f"   SL: {sl_mult:.1f}x ATR")
        print(f"   Time barrier: {time_barrier} bars")
        
        labels = pd.Series(-1, index=df.index)
        r_multiples = pd.Series(0.0, index=df.index)
        
        if 'atr' not in df.columns:
            raise ValueError("ATR required")
        
        for i in range(len(df) - time_barrier):
            if i % 10000 == 0 and i > 0:
                print(f"   Progress: {i:,}/{len(df):,} ({i/len(df)*100:.1f}%)", end='\r', flush=True)
            
            entry_price = df['close'].iloc[i]
            atr = df['atr'].iloc[i]
            
            if pd.isna(entry_price) or pd.isna(atr) or atr == 0:
                continue
            
            tp_price = entry_price + (tp_mult * atr)
            sl_price = entry_price - (sl_mult * atr)
            
            for j in range(1, time_barrier + 1):
                if i + j >= len(df):
                    break
                
                high = df['high'].iloc[i + j]
                low = df['low'].iloc[i + j]
                
                if high >= tp_price:
                    labels.iloc[i] = 1
                    r_multiples.iloc[i] = tp_mult
                    break
                
                if low <= sl_price:
                    labels.iloc[i] = 0
                    r_multiples.iloc[i] = -sl_mult
                    break
            else:
                exit_price = df['close'].iloc[i + time_barrier]
                pnl = exit_price - entry_price
                labels.iloc[i] = 1 if pnl > 0 else 0
                r_multiples.iloc[i] = pnl / atr
        
        print(f"\n")
        
        wins = (labels == 1).sum()
        losses = (labels == 0).sum()
        total = wins + losses
        win_rate = wins / total if total > 0 else 0
        
        avg_win_r = r_multiples[labels == 1].mean() if wins > 0 else 0
        avg_loss_r = abs(r_multiples[labels == 0].mean()) if losses > 0 else 0
        profit_factor = (wins * avg_win_r) / (losses * avg_loss_r) if losses > 0 and avg_loss_r > 0 else 0
        
        print(f"   ✅ Labeling complete:")
        print(f"      Wins: {wins:,} ({win_rate:.1%})")
        print(f"      Losses: {losses:,}")
        print(f"      Avg Win R: {avg_win_r:.2f}")
        print(f"      Avg Loss R: {avg_loss_r:.2f}")
        print(f"      Base PF: {profit_factor:.2f}")
        
        return labels, r_multiples
    
    @staticmethod
    def find_best_tp_mult(df: pd.DataFrame, tp_candidates: List[float],
                         sl_mult: float, time_barrier: int) -> float:
        """Find optimal TP multiplier."""
        
        print(f"\n🔍 OPTIMIZING TP MULTIPLIER")
        print(f"{'='*80}")
        
        best_tp = tp_candidates[0]
        best_pf = 0
        
        for tp_mult in tp_candidates:
            labels, r_mults = TripleBarrierLabeler.label(df, tp_mult, sl_mult, time_barrier)
            
            wins = (labels == 1).sum()
            losses = (labels == 0).sum()
            total = wins + losses
            
            if total == 0:
                continue
            
            win_rate = wins / total
            avg_win_r = r_mults[labels == 1].mean() if wins > 0 else 0
            avg_loss_r = abs(r_mults[labels == 0].mean()) if losses > 0 else 0
            pf = (wins * avg_win_r) / (losses * avg_loss_r) if losses > 0 and avg_loss_r > 0 else 0
            
            print(f"   TP={tp_mult:.1f}x: WR={win_rate:.1%}, PF={pf:.2f}")
            
            if pf > best_pf:
                best_pf = pf
                best_tp = tp_mult
        
        print(f"\n   ✅ Best TP: {best_tp:.1f}x (PF={best_pf:.2f})")
        
        return best_tp


# ═══════════════════════════════════════════════════════════════════════════
# DATA SPLITTING (KEPT FROM V2)
# ═══════════════════════════════════════════════════════════════════════════

class DataSplitter:
    """Chronological splitting (unchanged - correct)."""
    
    @staticmethod
    def split_chronological(df: pd.DataFrame, labels: pd.Series) -> Dict:
        """Split data chronologically."""
        
        print(f"\n✂️  CHRONOLOGICAL DATA SPLIT")
        print(f"{'='*80}")
        
        labeled_mask = (labels == 0) | (labels == 1)
        df_labeled = df[labeled_mask].copy()
        labels_filtered = labels[labeled_mask].copy()
        
        print(f"   Total: {len(df_labeled):,}")
        print(f"   Wins: {(labels_filtered == 1).sum():,}")
        print(f"   Losses: {(labels_filtered == 0).sum():,}")
        
        n = len(df_labeled)
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        
        df_train = df_labeled.iloc[:train_end]
        df_val = df_labeled.iloc[train_end:val_end]
        df_test = df_labeled.iloc[val_end:]
        
        y_train = labels_filtered.iloc[:train_end]
        y_val = labels_filtered.iloc[train_end:val_end]
        y_test = labels_filtered.iloc[val_end:]
        
        print(f"\n   Train: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
        print(f"   Val:   {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
        print(f"   Test:  {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
        
        feature_cols = [c for c in df_labeled.columns 
                       if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        X_train = df_train[feature_cols].values
        X_val = df_val[feature_cols].values
        X_test = df_test[feature_cols].values
        
        print(f"\n   ✅ Split: {len(feature_cols)} features")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train.values, 'y_val': y_val.values, 'y_test': y_test.values,
            'feature_cols': feature_cols,
            'train_ts': df_train,
            'val_ts': df_val,
            'test_ts': df_test
        }


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED MODEL TRAINING WITH CALIBRATION + STACKING
# ═══════════════════════════════════════════════════════════════════════════

class ModelFactory:
    """ENHANCED: Add probability calibration and ensemble stacking."""
    
    @staticmethod
    def prepare_sample_weights(y_train):
        """Compute balanced weights (unchanged)."""
        
        classes = np.unique(y_train)
        if len(classes) < 2:
            return np.ones(len(y_train)), 1.0
        
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[yi] for yi in y_train])
        scale_pos_weight = weight_dict[1] / weight_dict[0]
        
        return sample_weights, scale_pos_weight
    
    @staticmethod
    def train_all_models(X_train, X_val, y_train, y_val, enable_calibration: bool = True):
        """ENHANCED: Train models with optional calibration."""
        
        print(f"\n🤖 TRAINING MODELS (ENHANCED)")
        print(f"{'='*80}")
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Prepare weights
        sample_weights, scale_pos_weight = ModelFactory.prepare_sample_weights(y_train)
        
        base_models = {}
        
        # LightGBM (IMPROVED hyperparameters)
        try:
            print(f"\n   Training LightGBM (enhanced params)...")
            params = CONFIG.LGBM_PARAMS.copy()
            params['scale_pos_weight'] = scale_pos_weight
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            # NEW: Calibrate probabilities
            if enable_calibration:
                model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
                model.fit(X_val_scaled, y_val)
                print(f"   ✅ LightGBM trained + calibrated")
            else:
                print(f"   ✅ LightGBM trained")
            
            base_models['lightgbm'] = {'model': model, 'scaler': scaler}
        except Exception as e:
            print(f"   ❌ LightGBM failed: {e}")
        
        # XGBoost
        try:
            print(f"\n   Training XGBoost...")
            params = CONFIG.XGB_PARAMS.copy()
            params['scale_pos_weight'] = scale_pos_weight
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
            if enable_calibration:
                model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
                model.fit(X_val_scaled, y_val)
                print(f"   ✅ XGBoost trained + calibrated")
            else:
                print(f"   ✅ XGBoost trained")
            
            base_models['xgboost'] = {'model': model, 'scaler': scaler}
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
            model.fit(
                X_train_scaled, y_train,
                eval_set=(X_val_scaled, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            
            if enable_calibration:
                model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
                model.fit(X_val_scaled, y_val)
                print(f"   ✅ CatBoost trained + calibrated")
            else:
                print(f"   ✅ CatBoost trained")
            
            base_models['catboost'] = {'model': model, 'scaler': scaler}
        except Exception as e:
            print(f"   ❌ CatBoost failed: {e}")
        
        # Random Forest
        try:
            print(f"\n   Training RandomForest...")
            model = RandomForestClassifier(**CONFIG.RF_PARAMS)
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            
            if enable_calibration:
                model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
                model.fit(X_val_scaled, y_val)
                print(f"   ✅ RandomForest trained + calibrated")
            else:
                print(f"   ✅ RandomForest trained")
            
            base_models['random_forest'] = {'model': model, 'scaler': scaler}
        except Exception as e:
            print(f"   ❌ RandomForest failed: {e}")
        
        # Logistic Regression
        try:
            print(f"\n   Training LogisticRegression...")
            model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                C=0.1,
                solver='liblinear'
            )
            model.fit(X_train_scaled, y_train)
            print(f"   ✅ LogisticRegression trained")
            base_models['logistic'] = {'model': model, 'scaler': scaler}
        except Exception as e:
            print(f"   ❌ LogisticRegression failed: {e}")
        
        print(f"\n   ✅ Trained {len(base_models)} models")
        
        return base_models


# Continuing in next artifact due to length...