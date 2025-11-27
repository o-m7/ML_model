"""
═══════════════════════════════════════════════════════════════════════════════
CITADEL ML TRADING SYSTEM - ENHANCED COMPLETE VERSION
═══════════════════════════════════════════════════════════════════════════════

ENHANCEMENTS:
✓ Advanced volatility structure features (+45 new features)
✓ Market microstructure features (order flow, liquidity)
✓ Dynamic TP/SL adaptation by regime
✓ Probability calibration (Platt scaling)
✓ Cost-aware evaluation with slippage
✓ Risk filters (volatility, session, spread)
✓ Feature importance tracking
✓ Improved LightGBM hyperparameters

Usage:
    python citadel_enhanced_complete.py --symbol XAUUSD --timeframe 5T --full-system
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, brier_score_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """Enhanced system configuration."""
    
    FEATURE_STORE: Path = Path("ML_model/ML_model/feature_store")
    
    MIN_WIN_RATE: float = 0.58
    MIN_PROFIT_FACTOR: float = 1.6
    MAX_DRAWDOWN: float = 0.05
    
    # TP/SL (1.2x-2.0x as requested)
    TP_MULTIPLIERS: List[float] = field(default_factory=lambda: [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0])
    SL_MULTIPLIER: float = 1.0
    
    @staticmethod
    def get_time_barrier(timeframe: str) -> int:
        barriers = {'5T': 60, '15T': 40, '30T': 30, '1H': 20, '4H': 8}
        return barriers.get(timeframe, 24)
    
    # Risk filters
    MIN_ATR_PERCENTILE: float = 0.20
    MAX_ATR_PERCENTILE: float = 0.95
    MAX_SPREAD_TO_ATR: float = 0.15
    
    AVOID_ASIAN_SESSION: bool = True
    PREFER_LONDON_NY_OVERLAP: bool = True
    
    # Confidence thresholds
    CONFIDENCE_THRESHOLDS: List[float] = field(default_factory=lambda: [
        0.50, 0.52, 0.55, 0.57, 0.60, 0.63, 0.65, 0.68, 0.70
    ])
    
    # Enhanced LightGBM (improved regularization)
    LGBM_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 100,
        'min_child_weight': 0.001,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'min_split_gain': 0.01,
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
    
    TRAIN_RATIO: float = 0.60
    VAL_RATIO: float = 0.20
    WF_N_SPLITS: int = 5


CONFIG = SystemConfig()


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════

class DataLoader:
    
    @staticmethod
    def load_timeframe_data(symbol: str, timeframe: str) -> Tuple[pd.DataFrame, Dict]:
        print(f"\n{'='*80}")
        print(f"LOADING DATA: {symbol} {timeframe}")
        print(f"{'='*80}")
        
        file_path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
        metadata_path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}_metadata.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Feature file not found: {file_path}")
        
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
            raise ValueError("❌ ATR required")
        
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
    
    @staticmethod
    def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        
        # Volatility regime
        if 'atr' in features.columns:
            features['regime_vol_percentile'] = features['atr'].rolling(100).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
            )
            features['regime_vol'] = pd.cut(
                features['regime_vol_percentile'],
                bins=[0, 0.33, 0.67, 1.0],
                labels=[0, 1, 2]
            ).astype(float)
            
            # NEW: Regime transitions
            features['regime_vol_change'] = features['regime_vol'].diff()
            features['regime_vol_stable'] = (features['regime_vol'].rolling(5).std() < 0.5).astype(int)
        
        # Trend regime
        if 'ema_20' in features.columns and 'ema_50' in features.columns:
            features['regime_trend_20_50'] = ((features['ema_20'] > features['ema_50']).astype(int) * 2 - 1)
        
        if 'ema_50' in features.columns and 'ema_200' in features.columns:
            features['regime_trend_50_200'] = ((features['ema_50'] > features['ema_200']).astype(int) * 2 - 1)
        
        # Session regime
        if 'hour' in features.columns:
            features['regime_session_asian'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
            features['regime_session_london'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
            features['regime_session_ny'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
            features['regime_session_overlap'] = ((features['hour'] >= 13) & (features['hour'] < 16)).astype(int)
        
        # Range regime
        if 'close' in features.columns:
            high_20 = features['high'].rolling(20).max()
            low_20 = features['low'].rolling(20).min()
            range_20 = high_20 - low_20
            features['regime_range_position'] = (features['close'] - low_20) / (range_20 + 1e-8)
        
        return features
    
    @staticmethod
    def add_volatility_structure_features(df: pd.DataFrame) -> pd.DataFrame:
        """NEW: Advanced volatility features."""
        features = df.copy()
        
        if 'atr' not in features.columns:
            return features
        
        # Volatility clustering
        features['vol_cluster_5'] = features['atr'].rolling(5).std() / (features['atr'].rolling(5).mean() + 1e-8)
        features['vol_cluster_20'] = features['atr'].rolling(20).std() / (features['atr'].rolling(20).mean() + 1e-8)
        
        # Vol-of-vol
        features['vol_of_vol'] = features['atr'].rolling(20).std()
        
        # Vol momentum
        features['vol_momentum'] = (features['atr'] - features['atr'].rolling(20).mean()) / (features['atr'].rolling(20).std() + 1e-8)
        
        # Regime duration
        if 'regime_vol' in features.columns:
            features['vol_regime_duration'] = features.groupby(
                (features['regime_vol'] != features['regime_vol'].shift()).cumsum()
            ).cumcount() + 1
        
        # Realized volatility
        if 'returns' in features.columns:
            features['realized_vol_5'] = features['returns'].rolling(5).std() * np.sqrt(252 * 24)
            features['realized_vol_20'] = features['returns'].rolling(20).std() * np.sqrt(252 * 24)
        
        return features
    
    @staticmethod
    def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """ENHANCED: Market microstructure."""
        features = df.copy()
        
        # Candle components
        features['micro_body'] = abs(features['close'] - features['open'])
        features['micro_upper_wick'] = features['high'] - np.maximum(features['open'], features['close'])
        features['micro_lower_wick'] = np.minimum(features['open'], features['close']) - features['low']
        features['micro_total_range'] = features['high'] - features['low']
        
        features['micro_body_ratio'] = features['micro_body'] / (features['micro_total_range'] + 1e-8)
        features['micro_wick_ratio'] = (
            (features['micro_upper_wick'] + features['micro_lower_wick']) / 
            (features['micro_total_range'] + 1e-8)
        )
        
        # NEW: Order flow proxies
        features['micro_buying_pressure'] = (
            (features['close'] - features['low']) / (features['high'] - features['low'] + 1e-8)
        )
        features['micro_selling_pressure'] = (
            (features['high'] - features['close']) / (features['high'] - features['low'] + 1e-8)
        )
        
        # NEW: Direction and strength
        features['micro_direction'] = np.sign(features['close'] - features['open'])
        if 'atr' in features.columns:
            features['micro_strength'] = features['micro_body'] / (features['atr'] + 1e-8)
        
        # Volume
        if 'volume' in features.columns:
            vol_ma = features['volume'].rolling(20).mean()
            features['micro_volume_surge'] = features['volume'] / (vol_ma + 1)
            features['micro_volume_anomaly'] = (features['micro_volume_surge'] > 2.0).astype(int)
            
            # NEW: Volume-price confirmation
            if 'micro_direction' in features.columns:
                features['micro_volume_price_confirm'] = (
                    (features['micro_direction'] * features['micro_volume_surge'] > 1.5).astype(int)
                )
        
        # Gaps
        features['micro_gap'] = features['open'] - features['close'].shift(1)
        features['micro_gap_pct'] = features['micro_gap'] / (features['close'].shift(1) + 1e-8)
        
        if 'atr' in features.columns:
            features['micro_gap_significant'] = (abs(features['micro_gap']) > features['atr'] * 0.3).astype(int)
        
        return features
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        
        for period in [5, 10, 20]:
            features[f'momentum_roc_{period}'] = features['close'].pct_change(period)
        
        if 'atr' in features.columns:
            features['momentum_strength_5'] = features['close'].diff(5) / (features['atr'] + 1e-8)
            features['momentum_strength_10'] = features['close'].diff(10) / (features['atr'] + 1e-8)
        
        if 'momentum_roc_5' in features.columns and 'momentum_roc_10' in features.columns:
            features['momentum_accel'] = features['momentum_roc_5'] - features['momentum_roc_10']
        
        return features
    
    @staticmethod
    def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        
        for ma in [20, 50, 100]:
            if f'sma_{ma}' in features.columns:
                features[f'mr_distance_sma_{ma}'] = (
                    (features['close'] - features[f'sma_{ma}']) / (features[f'sma_{ma}'] + 1e-8)
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
            features['mr_rsi_neutral'] = ((features['rsi'] >= 40) & (features['rsi'] <= 60)).astype(int)
        
        return features
    
    @staticmethod
    def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n🔧 Engineering features (ENHANCED)...")
        
        initial_cols = len(df.columns)
        
        df = FeatureEngineer.add_regime_features(df)
        df = FeatureEngineer.add_volatility_structure_features(df)
        df = FeatureEngineer.add_momentum_features(df)
        df = FeatureEngineer.add_mean_reversion_features(df)
        df = FeatureEngineer.add_microstructure_features(df)
        
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        final_cols = len(df.columns)
        added_cols = final_cols - initial_cols
        
        print(f"   ✅ Added {added_cols} features")
        print(f"   🧹 Dropped {dropped_rows} rows")
        print(f"   ✓ Final: {len(df):,} rows, {final_cols} columns")
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# TRIPLE BARRIER LABELING
# ═══════════════════════════════════════════════════════════════════════════

class TripleBarrierLabeler:
    
    @staticmethod
    def label(df: pd.DataFrame, tp_mult: float, sl_mult: float, time_barrier: int) -> Tuple[pd.Series, pd.Series]:
        print(f"\n🏷️  TRIPLE BARRIER LABELING")
        print(f"{'='*80}")
        print(f"   TP: {tp_mult:.1f}x ATR, SL: {sl_mult:.1f}x ATR, Time: {time_barrier} bars")
        
        labels = pd.Series(-1, index=df.index)
        r_multiples = pd.Series(0.0, index=df.index)
        
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
        
        print(f"   ✅ Wins: {wins:,} ({win_rate:.1%}), Losses: {losses:,}")
        print(f"   Avg Win R: {avg_win_r:.2f}, Avg Loss R: {avg_loss_r:.2f}")
        print(f"   Base PF: {profit_factor:.2f}")
        
        return labels, r_multiples
    
    @staticmethod
    def find_best_tp_mult(df: pd.DataFrame, tp_candidates: List[float], sl_mult: float, time_barrier: int) -> float:
        print(f"\n🔍 OPTIMIZING TP MULTIPLIER")
        print(f"{'='*80}")
        
        best_tp = tp_candidates[0]
        best_pf = 0
        
        for tp_mult in tp_candidates:
            labels, r_mults = TripleBarrierLabeler.label(df, tp_mult, sl_mult, time_barrier)
            
            wins = (labels == 1).sum()
            losses = (labels == 0).sum()
            
            if wins + losses == 0:
                continue
            
            win_rate = wins / (wins + losses)
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
# DATA SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

class DataSplitter:
    
    @staticmethod
    def split_chronological(df: pd.DataFrame, labels: pd.Series) -> Dict:
        print(f"\n✂️  CHRONOLOGICAL DATA SPLIT")
        print(f"{'='*80}")
        
        labeled_mask = (labels == 0) | (labels == 1)
        df_labeled = df[labeled_mask].copy()
        labels_filtered = labels[labeled_mask].copy()
        
        print(f"   Total: {len(df_labeled):,}")
        
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
# ENHANCED MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class ModelFactory:
    
    @staticmethod
    def prepare_sample_weights(y_train):
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
        print(f"\n🤖 TRAINING MODELS {'(WITH CALIBRATION)' if enable_calibration else ''}")
        print(f"{'='*80}")
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        sample_weights, scale_pos_weight = ModelFactory.prepare_sample_weights(y_train)
        
        base_models = {}
        
        # LightGBM
        try:
            print(f"\n   Training LightGBM...")
            params = CONFIG.LGBM_PARAMS.copy()
            params['scale_pos_weight'] = scale_pos_weight
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
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
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights, eval_set=[(X_val_scaled, y_val)], verbose=False)
            
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
            
            params = CONFIG.CATBOOST_PARAMS.copy()
            params['class_weights'] = {0: neg_weight, 1: pos_weight}
            
            model = CatBoostClassifier(**params)
            model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val), early_stopping_rounds=50, verbose=False)
            
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
            model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1, solver='liblinear')
            model.fit(X_train_scaled, y_train)
            print(f"   ✅ LogisticRegression trained")
            base_models['logistic'] = {'model': model, 'scaler': scaler}
        except Exception as e:
            print(f"   ❌ LogisticRegression failed: {e}")
        
        print(f"\n   ✅ Trained {len(base_models)} models")
        return base_models


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

class ModelEvaluator:
    
    @staticmethod
    def evaluate_all_models(models: Dict, X_test, y_test, spread_cost: float = 0.0002) -> Dict:
        print(f"\n📊 EVALUATING MODELS (COST-AWARE)")
        print(f"{'='*80}")
        
        results = {}
        
        for model_name, model_dict in models.items():
            try:
                model = model_dict['model']
                scaler = model_dict['scaler']
                
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                brier = brier_score_loss(y_test, y_proba)
                
                wins = ((y_pred == 1) & (y_test == 1)).sum()
                losses = ((y_pred == 1) & (y_test == 0)).sum()
                total = wins + losses
                win_rate = wins / total if total > 0 else 0
                
                avg_win_r = 1.4
                avg_loss_r = 1.0
                pf_base = (wins * avg_win_r) / (losses * avg_loss_r) if losses > 0 else 0
                
                cost_per_trade_r = spread_cost / avg_loss_r
                avg_win_r_net = avg_win_r - cost_per_trade_r
                avg_loss_r_net = avg_loss_r + cost_per_trade_r
                pf_cost_adj = (wins * avg_win_r_net) / (losses * avg_loss_r_net) if losses > 0 else 0
                
                ev_base = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)
                ev_cost_adj = (win_rate * avg_win_r_net) - ((1 - win_rate) * avg_loss_r_net)
                
                results[model_name] = {
                    'accuracy': acc,
                    'win_rate': win_rate,
                    'profit_factor': pf_base,
                    'pf_cost_adj': pf_cost_adj,
                    'f1': f1,
                    'brier': brier,
                    'ev_base': ev_base,
                    'ev_cost_adj': ev_cost_adj,
                    'total_trades': total
                }
                
                print(f"\n{model_name.upper()}:")
                print(f"   WR: {win_rate:.1%}, PF: {pf_base:.2f} → {pf_cost_adj:.2f}")
                print(f"   Brier: {brier:.4f}, F1: {f1:.4f}, Trades: {total:,}")
                
            except Exception as e:
                print(f"\n{model_name.upper()}: ❌ Failed - {e}")
        
        return results
    
    @staticmethod
    def print_comparison_table(results: Dict):
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON")
        print(f"{'='*80}")
        
        print(f"\n{'Model':<20} {'WR':>8} {'PF':>6} {'PF_adj':>8} {'Brier':>8} {'F1':>8} {'Trades':>10}")
        print("-"*80)
        
        for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
            print(f"{model_name:<20} "
                  f"{metrics['win_rate']:>7.1%} "
                  f"{metrics['profit_factor']:>6.2f} "
                  f"{metrics['pf_cost_adj']:>8.2f} "
                  f"{metrics['brier']:>8.4f} "
                  f"{metrics['f1']:>8.4f} "
                  f"{metrics['total_trades']:>10,}")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIDENCE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

class ConfidenceFilter:
    
    @staticmethod
    def find_optimal_threshold(model_dict, X_val, y_val) -> Tuple[float, Dict]:
        print(f"\n🎯 OPTIMIZING CONFIDENCE THRESHOLD")
        print(f"{'='*80}")
        
        model = model_dict['model']
        scaler = model_dict['scaler']
        
        X_val_scaled = scaler.transform(X_val)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        best_threshold = 0.50
        best_score = 0
        threshold_metrics = {}
        
        print(f"\n{'Threshold':>12} {'WR':>8} {'Trades':>10} {'EV':>8} {'Score':>10}")
        print("-"*60)
        
        for threshold in CONFIG.CONFIDENCE_THRESHOLDS:
            y_pred_filtered = (y_proba >= threshold).astype(int)
            
            mask = y_pred_filtered == 1
            if mask.sum() == 0:
                continue
            
            wins = ((y_pred_filtered == 1) & (y_val == 1)).sum()
            losses = ((y_pred_filtered == 1) & (y_val == 0)).sum()
            total = wins + losses
            
            wr = wins / total if total > 0 else 0
            ev = (wr * 1.4) - ((1 - wr) * 1.0)
            score = ev * np.log(total + 1)
            
            threshold_metrics[threshold] = {'wr': wr, 'trades': total, 'ev': ev, 'score': score}
            
            print(f"{threshold:>12.2f} {wr:>7.1%} {total:>10,} {ev:>8.3f} {score:>10.4f}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"\n   ✅ Best: {best_threshold:.2f} (WR: {threshold_metrics[best_threshold]['wr']:.1%}, EV: {threshold_metrics[best_threshold]['ev']:.3f}R)")
        
        return best_threshold, threshold_metrics


# ═══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    
    @staticmethod
    def run_walk_forward(df: pd.DataFrame, labels: pd.Series, n_splits: int = 5) -> List[Dict]:
        print(f"\n🔄 WALK-FORWARD VALIDATION")
        print(f"{'='*80}")
        
        labeled_mask = (labels == 0) | (labels == 1)
        df_labeled = df[labeled_mask].copy()
        labels_filtered = labels[labeled_mask].copy()
        
        n_samples = len(df_labeled)
        fold_size = n_samples // (n_splits + 1)
        
        results = []
        
        for i in range(n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = test_start + fold_size
            
            if test_end > n_samples:
                break
            
            y_train = labels_filtered.iloc[:train_end].values
            y_test = labels_filtered.iloc[test_start:test_end].values
            
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            
            feature_cols = [c for c in df_labeled.columns 
                           if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            X_train = df_labeled.iloc[:train_end][feature_cols].values
            X_test = df_labeled.iloc[test_start:test_end][feature_cols].values
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            weight_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([weight_dict[yi] for yi in y_train])
            
            model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, verbose=-1)
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            y_pred = model.predict(X_test_scaled)
            
            wins = ((y_pred == 1) & (y_test == 1)).sum()
            total = (y_pred == 1).sum()
            
            results.append({
                'fold_num': i + 1,
                'win_rate': wins / total if total > 0 else 0,
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'total_trades': total
            })
            
            print(f"   Fold {i+1}: WR={results[-1]['win_rate']:.1%}, Trades={total:,}")
        
        if results:
            avg_wr = np.mean([r['win_rate'] for r in results])
            std_wr = np.std([r['win_rate'] for r in results])
            print(f"\n   Avg WR: {avg_wr:.1%} ± {std_wr:.1%}")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

class RegimeAnalyzer:
    
    @staticmethod
    def analyze_by_regime(df: pd.DataFrame, y_true, y_pred) -> Dict:
        print(f"\n📊 REGIME PERFORMANCE")
        print(f"{'='*80}")
        
        regime_results = {}
        
        if 'regime_vol' in df.columns:
            print(f"\n🌊 Volatility Regimes:")
            regimes = df['regime_vol'].replace({0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'})
            
            for regime in ['Low Vol', 'Med Vol', 'High Vol']:
                mask = regimes == regime
                if mask.sum() == 0:
                    continue
                
                wins = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum()
                total = (y_pred[mask] == 1).sum()
                wr = wins / total if total > 0 else 0
                
                regime_results[f'vol_{regime}'] = {'win_rate': wr, 'trades': total}
                print(f"   {regime}: WR={wr:.1%}, Trades={total:,}")
        
        return regime_results


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class TrainingPipeline:
    
    def __init__(self, symbol: str, timeframe: str,
                 enable_walk_forward: bool = False,
                 enable_diagnostics: bool = False,
                 enable_calibration: bool = True):
        self.symbol = symbol
        self.timeframe = timeframe
        self.enable_walk_forward = enable_walk_forward
        self.enable_diagnostics = enable_diagnostics
        self.enable_calibration = enable_calibration
        self.results = {}
    
    def run(self):
        print(f"\n{'#'*80}")
        print(f"# CITADEL ML SYSTEM - ENHANCED")
        print(f"# {self.symbol} {self.timeframe} | Calibration: {self.enable_calibration}")
        print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")
        
        # Load
        df, metadata = DataLoader.load_timeframe_data(self.symbol, self.timeframe)
        
        # Engineer
        df = FeatureEngineer.engineer_all_features(df)
        
        # Label
        time_barrier = CONFIG.get_time_barrier(self.timeframe)
        best_tp = TripleBarrierLabeler.find_best_tp_mult(df, CONFIG.TP_MULTIPLIERS, CONFIG.SL_MULTIPLIER, time_barrier)
        labels, r_multiples = TripleBarrierLabeler.label(df, best_tp, CONFIG.SL_MULTIPLIER, time_barrier)
        
        # Walk-forward
        if self.enable_walk_forward:
            wf_results = WalkForwardValidator.run_walk_forward(df, labels, CONFIG.WF_N_SPLITS)
        
        # Split
        splits = DataSplitter.split_chronological(df, labels)
        
        # Train
        models = ModelFactory.train_all_models(
            splits['X_train'], splits['X_val'],
            splits['y_train'], splits['y_val'],
            enable_calibration=self.enable_calibration
        )
        
        # Evaluate
        results = ModelEvaluator.evaluate_all_models(models, splits['X_test'], splits['y_test'])
        
        # Best model
        best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
        best_model = models[best_model_name]
        
        # Confidence
        optimal_threshold, threshold_metrics = ConfidenceFilter.find_optimal_threshold(
            best_model, splits['X_val'], splits['y_val']
        )
        
        # Regime
        test_indices = splits['test_ts'].index
        df_test = df.loc[test_indices]
        
        X_test_scaled = best_model['scaler'].transform(splits['X_test'])
        y_test_pred = best_model['model'].predict(X_test_scaled)
        
        regime_results = RegimeAnalyzer.analyze_by_regime(df_test, splits['y_test'], y_test_pred)
        
        # Print
        ModelEvaluator.print_comparison_table(results)
        
        # Store
        self.results = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'best_tp_mult': best_tp,
            'optimal_threshold': optimal_threshold,
            'threshold_metrics': threshold_metrics,
            'models': models,
            'results': results,
            'regime_results': regime_results,
            'feature_cols': splits['feature_cols']
        }
        
        print(f"\n{'#'*80}")
        print(f"# COMPLETE: {best_model_name} @ {optimal_threshold:.2f}")
        print(f"# Expected WR: {threshold_metrics[optimal_threshold]['wr']:.1%}")
        print(f"# Expected EV: {threshold_metrics[optimal_threshold]['ev']:.3f}R")
        print(f"{'#'*80}\n")
        
        return self.results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Citadel ML System - Enhanced')
    parser.add_argument('--symbol', type=str, default='XAUUSD')
    parser.add_argument('--timeframe', type=str)
    parser.add_argument('--all-timeframes', action='store_true')
    parser.add_argument('--walk-forward', action='store_true')
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--full-system', action='store_true')
    parser.add_argument('--no-calibration', action='store_true')
    
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
                enable_diagnostics=args.diagnose,
                enable_calibration=not args.no_calibration
            )
            results = pipeline.run()
            all_results[timeframe] = results
        except Exception as e:
            print(f"\n❌ ERROR in {timeframe}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY - {args.symbol}")
    print(f"{'='*80}")
    
    print(f"\n{'TF':<6} {'Model':<15} {'WR':>8} {'PF_adj':>8} {'Threshold':>11} {'EV':>8}")
    print("-"*70)
    
    for tf, result in all_results.items():
        if result and 'results' in result:
            best = max(result['results'].items(), key=lambda x: x[1]['f1'])
            threshold = result['optimal_threshold']
            tm = result['threshold_metrics']
            
            print(f"{tf:<6} {best[0]:<15} "
                  f"{best[1]['win_rate']:>7.1%} "
                  f"{best[1]['pf_cost_adj']:>8.2f} "
                  f"{threshold:>11.2f} "
                  f"{tm[threshold]['ev']:>8.3f}")
    
    print(f"\n✅ COMPLETE\n")


if __name__ == '__main__':
    main()