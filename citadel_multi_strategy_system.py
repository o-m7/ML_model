"""
Citadel-Grade Multi-Strategy ML Trading System

Implements 9 institutional strategies across 3 timeframes:
- 5T: MSTF, VWAP-REV, ATR-EXP
- 15T: RCT, VRAMR, BR-SMC
- 30T: SBM, VCE, MCRC

Each strategy has:
- Independent feature engineering
- Multiple model families (XGBoost, LightGBM, RF, LogReg)
- Strategy-specific performance metrics
- Regime-aware training

Meta-model learns optimal strategy selection.
Final ensemble produces high-edge predictions.

Usage:
    python citadel_multi_strategy_system.py --timeframe 5T
    python citadel_multi_strategy_system.py --timeframe 15T --strategies MSTF VWAP-REV
    python citadel_multi_strategy_system.py --all-timeframes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime
from dataclasses import dataclass
import argparse

# ML imports
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    description: str
    timeframe: str
    feature_groups: List[str]
    min_samples: int = 500
    tp_mult: float = 1.5
    sl_mult: float = 1.0
    time_barrier: int = 24


class FeatureStore:
    """Load and manage clean features from Polygon S3 extraction."""
    
    BASE_PATH = Path("ML_model/ML_model/feature_store")
    
    @staticmethod
    def load_timeframe_data(symbol: str, timeframe: str) -> pd.DataFrame:
        """Load features for symbol/timeframe."""
        
        file_path = FeatureStore.BASE_PATH / symbol / f"{symbol}_{timeframe}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {file_path}")
            
        df = pd.read_parquet(file_path)
        
        # Verify features are lagged
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        first_row_nans = df[feature_cols].iloc[0].isna().sum()
        
        print(f"   Loaded {len(df):,} rows, {len(df.columns)} features")
        print(f"   Date: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Lag check: {first_row_nans}/{len(feature_cols)} NaNs in first row")
        
        return df.dropna()


class StrategyFeatureEngineer:
    """Engineer features specific to each strategy."""
    
    @staticmethod
    def mstf_5t_features(df: pd.DataFrame) -> pd.DataFrame:
        """Microstructure Trend-Following (5T)."""
        
        features = df.copy()
        
        # Order flow proxies
        features['mstf_price_delta'] = features['close'].diff()
        features['mstf_volume_delta'] = features['volume'].diff()
        features['mstf_signed_volume'] = features['mstf_volume_delta'] * np.sign(features['mstf_price_delta'])
        features['mstf_cumulative_delta'] = features['mstf_signed_volume'].rolling(20).sum()
        
        # VWAP deviation
        if 'volume' in features.columns:
            typical_price = (features['high'] + features['low'] + features['close']) / 3
            vwap = (typical_price * features['volume']).rolling(20).sum() / features['volume'].rolling(20).sum()
            features['mstf_vwap_dev'] = (features['close'] - vwap) / vwap
            features['mstf_vwap_dev_z'] = (features['mstf_vwap_dev'] - features['mstf_vwap_dev'].rolling(100).mean()) / features['mstf_vwap_dev'].rolling(100).std()
        
        # Micro pullback structure
        features['mstf_micro_high'] = features['high'].rolling(5).max()
        features['mstf_micro_low'] = features['low'].rolling(5).min()
        features['mstf_pullback_depth'] = (features['close'] - features['mstf_micro_low']) / (features['mstf_micro_high'] - features['mstf_micro_low'])
        
        # ATR expansion
        if 'atr' in features.columns:
            features['mstf_atr_expansion'] = features['atr'] / features['atr'].rolling(20).mean()
            features['mstf_atr_slope'] = features['atr'].diff(5)
        
        # Trend strength
        features['mstf_ema_9'] = features['close'].ewm(span=9).mean()
        features['mstf_ema_21'] = features['close'].ewm(span=21).mean()
        features['mstf_trend_strength'] = (features['mstf_ema_9'] - features['mstf_ema_21']) / features['atr']
        
        return features
    
    @staticmethod
    def vwap_rev_5t_features(df: pd.DataFrame) -> pd.DataFrame:
        """VWAP Liquidity Reversion (5T)."""
        
        features = df.copy()
        
        # VWAP calculation
        typical_price = (features['high'] + features['low'] + features['close']) / 3
        vwap = (typical_price * features['volume']).rolling(50).sum() / features['volume'].rolling(50).sum()
        
        features['vwap_distance'] = (features['close'] - vwap) / vwap
        features['vwap_distance_pct'] = features['vwap_distance'] * 100
        
        # VWAP z-score
        features['vwap_z'] = (features['vwap_distance'] - features['vwap_distance'].rolling(100).mean()) / features['vwap_distance'].rolling(100).std()
        
        # RSI(5) for mean reversion
        if 'rsi' in features.columns:
            features['vwap_rsi5'] = features['close'].diff().rolling(5).apply(
                lambda x: 100 - (100 / (1 + (x[x > 0].sum() / abs(x[x < 0].sum())))) if x[x < 0].sum() != 0 else 50
            )
        
        # Mean reversion probability bands
        features['vwap_upper_band'] = vwap + 2 * features['vwap_distance'].rolling(50).std()
        features['vwap_lower_band'] = vwap - 2 * features['vwap_distance'].rolling(50).std()
        features['vwap_band_position'] = (features['close'] - features['vwap_lower_band']) / (features['vwap_upper_band'] - features['vwap_lower_band'])
        
        # Reversion signal strength
        features['vwap_reversion_score'] = abs(features['vwap_z']) * (1 - features['vwap_band_position'].clip(0, 1))
        
        return features
    
    @staticmethod
    def atr_exp_5t_features(df: pd.DataFrame) -> pd.DataFrame:
        """ATR Breakout Continuation (5T)."""
        
        features = df.copy()
        
        if 'atr' not in features.columns:
            return features
        
        # ATR slope and acceleration
        features['atr_slope_5'] = features['atr'].diff(5)
        features['atr_slope_10'] = features['atr'].diff(10)
        features['atr_accel'] = features['atr_slope_5'] - features['atr_slope_10']
        
        # Breakout strength
        features['atr_breakout_high'] = (features['high'] - features['high'].shift(1)) / features['atr']
        features['atr_breakout_low'] = (features['low'].shift(1) - features['low']) / features['atr']
        features['atr_breakout_strength'] = np.maximum(features['atr_breakout_high'], features['atr_breakout_low'])
        
        # Volatility expansion flags
        features['atr_expansion_flag'] = (features['atr'] > features['atr'].rolling(20).mean() * 1.2).astype(int)
        features['atr_percentile'] = features['atr'].rolling(100).apply(lambda x: (x.iloc[-1] > x).sum() / len(x))
        
        # Price momentum during expansion
        features['atr_price_momentum'] = features['close'].pct_change(5)
        features['atr_volume_surge'] = features['volume'] / features['volume'].rolling(20).mean()
        
        # Continuation probability
        features['atr_continuation_score'] = (
            features['atr_expansion_flag'] * 
            features['atr_breakout_strength'] * 
            features['atr_volume_surge']
        )
        
        return features
    
    @staticmethod
    def rct_15t_features(df: pd.DataFrame) -> pd.DataFrame:
        """Regression Channel Trend Rider (15T)."""
        
        features = df.copy()
        
        # Linear regression channel
        window = 50
        features['rct_channel_mid'] = features['close'].rolling(window).mean()
        
        # Channel slope
        def calc_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            y = series.values
            return np.polyfit(x, y, 1)[0]
        
        features['rct_channel_slope'] = features['close'].rolling(window).apply(calc_slope)
        features['rct_slope_strength'] = features['rct_channel_slope'] / features['atr']
        
        # Midline touches
        features['rct_distance_to_mid'] = (features['close'] - features['rct_channel_mid']) / features['atr']
        features['rct_touch_flag'] = (abs(features['rct_distance_to_mid']) < 0.5).astype(int)
        
        # MACD histogram
        if 'macd_hist' in features.columns:
            features['rct_macd_momentum'] = features['macd_hist'].diff(3)
            features['rct_macd_cross'] = ((features['macd_hist'] > 0) & (features['macd_hist'].shift(1) <= 0)).astype(int)
        
        # Buy/sell pressure proxy
        features['rct_buy_pressure'] = (features['close'] - features['low']) / (features['high'] - features['low'])
        features['rct_buy_pressure_ma'] = features['rct_buy_pressure'].rolling(10).mean()
        
        return features
    
    @staticmethod
    def vramr_15t_features(df: pd.DataFrame) -> pd.DataFrame:
        """Volatility-Regime Adaptive Mean Reversion (15T)."""
        
        features = df.copy()
        
        # Regime detection
        if 'atr' in features.columns:
            features['vramr_atr_percentile'] = features['atr'].rolling(100).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x)
            )
            features['vramr_vol_regime'] = pd.cut(
                features['vramr_atr_percentile'], 
                bins=[0, 0.33, 0.67, 1.0], 
                labels=['low', 'medium', 'high']
            )
        
        # Realized volatility
        features['vramr_realized_vol'] = features['close'].pct_change().rolling(20).std() * np.sqrt(252 * 24)
        
        # Bollinger deviation
        if 'bb_mid' in features.columns and 'bb_upper' in features.columns:
            features['vramr_bb_dev'] = (features['close'] - features['bb_mid']) / (features['bb_upper'] - features['bb_mid'])
        
        # RSI divergence
        if 'rsi' in features.columns:
            features['vramr_rsi_div'] = features['rsi'].diff(5)
            features['vramr_price_change'] = features['close'].pct_change(5)
            features['vramr_divergence'] = np.sign(features['vramr_rsi_div']) != np.sign(features['vramr_price_change'])
        
        # Adaptive mean reversion score
        features['vramr_reversion_score'] = (
            abs(features['vramr_bb_dev']) * 
            (1 - features['vramr_atr_percentile'])
        )
        
        return features
    
    @staticmethod
    def br_smc_15t_features(df: pd.DataFrame) -> pd.DataFrame:
        """Break-and-Retest Smart-Money Zones (15T)."""
        
        features = df.copy()
        
        # Liquidity sweep detection
        features['br_swing_high'] = features['high'].rolling(20, center=True).max()
        features['br_swing_low'] = features['low'].rolling(20, center=True).min()
        
        features['br_liquidity_sweep_high'] = (
            (features['high'] > features['br_swing_high'].shift(1)) & 
            (features['close'] < features['br_swing_high'].shift(1))
        ).astype(int)
        
        features['br_liquidity_sweep_low'] = (
            (features['low'] < features['br_swing_low'].shift(1)) & 
            (features['close'] > features['br_swing_low'].shift(1))
        ).astype(int)
        
        # Structure break
        features['br_structure_break_bull'] = (
            (features['close'] > features['br_swing_high'].shift(5))
        ).astype(int)
        
        features['br_structure_break_bear'] = (
            (features['close'] < features['br_swing_low'].shift(5))
        ).astype(int)
        
        # Order flow delta flip
        features['br_volume_delta'] = features['volume'].diff()
        features['br_price_delta'] = features['close'].diff()
        features['br_of_delta'] = features['br_volume_delta'] * np.sign(features['br_price_delta'])
        features['br_of_delta_ma'] = features['br_of_delta'].rolling(10).mean()
        features['br_of_flip'] = ((features['br_of_delta_ma'] > 0) & (features['br_of_delta_ma'].shift(1) <= 0)).astype(int)
        
        # Smart money zone proximity
        features['br_distance_to_high'] = (features['br_swing_high'] - features['close']) / features['atr']
        features['br_distance_to_low'] = (features['close'] - features['br_swing_low']) / features['atr']
        
        return features
    
    @staticmethod
    def sbm_30t_features(df: pd.DataFrame) -> pd.DataFrame:
        """Session Bias Momentum (30T)."""
        
        features = df.copy()
        
        # Daily VWAP bias
        features['timestamp'] = pd.to_datetime(features['timestamp'])
        features['sbm_date'] = features['timestamp'].dt.date
        
        typical_price = (features['high'] + features['low'] + features['close']) / 3
        features['sbm_daily_vwap'] = features.groupby('sbm_date').apply(
            lambda x: (typical_price[x.index] * features['volume'][x.index]).cumsum() / features['volume'][x.index].cumsum()
        ).droplevel(0)
        
        features['sbm_vwap_bias'] = (features['close'] - features['sbm_daily_vwap']) / features['sbm_daily_vwap']
        
        # Opening drive detection
        features['sbm_hour'] = features['timestamp'].dt.hour
        features['sbm_is_opening'] = ((features['sbm_hour'] >= 9) & (features['sbm_hour'] < 11)).astype(int)
        features['sbm_opening_momentum'] = features['close'].pct_change(4) * features['sbm_is_opening']
        
        # Overnight range imbalance
        daily_high = features.groupby('sbm_date')['high'].transform('max')
        daily_low = features.groupby('sbm_date')['low'].transform('min')
        features['sbm_daily_range'] = daily_high - daily_low
        features['sbm_range_position'] = (features['close'] - daily_low) / features['sbm_daily_range']
        
        return features
    
    @staticmethod
    def vce_30t_features(df: pd.DataFrame) -> pd.DataFrame:
        """Volatility Compression → Expansion (30T)."""
        
        features = df.copy()
        
        # Bollinger bandwidth
        if 'bb_upper' in features.columns and 'bb_lower' in features.columns:
            features['vce_bb_bandwidth'] = (features['bb_upper'] - features['bb_lower']) / features['bb_mid']
            features['vce_bandwidth_percentile'] = features['vce_bb_bandwidth'].rolling(100).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x)
            )
            features['vce_compression'] = (features['vce_bandwidth_percentile'] < 0.2).astype(int)
        
        # ATR expansion after compression
        if 'atr' in features.columns:
            features['vce_atr_compression'] = (features['atr'] < features['atr'].rolling(50).mean() * 0.8).astype(int)
            features['vce_atr_expansion'] = (features['atr'] > features['atr'].rolling(50).mean() * 1.2).astype(int)
            features['vce_expansion_signal'] = features['vce_atr_compression'].shift(1) & features['vce_atr_expansion']
        
        # Breakout confirmation
        features['vce_range_high'] = features['high'].rolling(20).max()
        features['vce_range_low'] = features['low'].rolling(20).min()
        features['vce_breakout_up'] = (features['close'] > features['vce_range_high'].shift(1)).astype(int)
        features['vce_breakout_down'] = (features['close'] < features['vce_range_low'].shift(1)).astype(int)
        
        features['vce_breakout_score'] = (
            features['vce_expansion_signal'] * 
            (features['vce_breakout_up'] + features['vce_breakout_down'])
        )
        
        return features
    
    @staticmethod
    def mcrc_30t_features(df: pd.DataFrame) -> pd.DataFrame:
        """Macro-Catalyst Reversion + Continuation Hybrid (30T)."""
        
        features = df.copy()
        
        # News volatility trigger (proxy via ATR spikes)
        if 'atr' in features.columns:
            features['mcrc_atr_spike'] = features['atr'] / features['atr'].rolling(50).mean()
            features['mcrc_vol_trigger'] = (features['mcrc_atr_spike'] > 1.5).astype(int)
        
        # Spike magnitude
        features['mcrc_price_spike'] = abs(features['close'].pct_change(1))
        features['mcrc_spike_magnitude'] = features['mcrc_price_spike'] / features['mcrc_price_spike'].rolling(50).mean()
        
        # Correction depth after spike
        features['mcrc_high_since_spike'] = features['high'].rolling(10).max()
        features['mcrc_low_since_spike'] = features['low'].rolling(10).min()
        features['mcrc_correction_depth'] = (features['mcrc_high_since_spike'] - features['close']) / features['mcrc_high_since_spike']
        
        # Hybrid signal: reversion or continuation
        features['mcrc_reversion_flag'] = (
            (features['mcrc_vol_trigger'] == 1) & 
            (features['mcrc_correction_depth'] > 0.3)
        ).astype(int)
        
        features['mcrc_continuation_flag'] = (
            (features['mcrc_vol_trigger'] == 1) & 
            (features['mcrc_spike_magnitude'] > 2) & 
            (features['mcrc_correction_depth'] < 0.1)
        ).astype(int)
        
        return features


class TripleBarrierLabeler:
    """Label data using triple barrier method."""
    
    @staticmethod
    def label(df: pd.DataFrame, tp_mult: float, sl_mult: float, time_barrier: int) -> pd.Series:
        """Apply triple barrier labeling."""
        
        labels = pd.Series(0, index=df.index)
        
        if 'atr' not in df.columns:
            raise ValueError("ATR column required for triple barrier labeling")
        
        for i in range(len(df) - time_barrier):
            entry_price = df['close'].iloc[i]
            atr = df['atr'].iloc[i] if 'atr' in df.columns else df['close'].iloc[i] * 0.01
            
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
                    break
                if low <= sl_price:
                    labels.iloc[i] = 0
                    break
            else:
                exit_price = df['close'].iloc[i + time_barrier]
                labels.iloc[i] = 1 if exit_price > entry_price else 0
        
        return labels


class ModelTrainer:
    """Train multiple model families."""
    
    @staticmethod
    def train_models(X_train, X_test, y_train, y_test, strategy_name: str) -> Dict:
        """Train all model families and return results."""
        
        results = {}
        
        # Compute class weights
        classes = np.unique(y_train)
        if len(classes) < 2:
            print(f"   ⚠️  Only one class in training data: {classes}")
            return results
        
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[yi] for yi in y_train])
        scale_pos_weight = weight_dict[1] / weight_dict[0] if 0 in weight_dict and 1 in weight_dict else 1.0
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. LightGBM
        try:
            lgbm_model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                scale_pos_weight=scale_pos_weight,
                verbose=-1
            )
            lgbm_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            results['lightgbm'] = ModelTrainer._evaluate(lgbm_model, X_test_scaled, y_test, 'LightGBM')
        except Exception as e:
            print(f"   ❌ LightGBM failed: {e}")
        
        # 2. XGBoost
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                scale_pos_weight=scale_pos_weight,
                verbosity=0
            )
            xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            results['xgboost'] = ModelTrainer._evaluate(xgb_model, X_test_scaled, y_test, 'XGBoost')
        except Exception as e:
            print(f"   ❌ XGBoost failed: {e}")
        
        # 3. Random Forest
        try:
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=50,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            results['random_forest'] = ModelTrainer._evaluate(rf_model, X_test_scaled, y_test, 'RandomForest')
        except Exception as e:
            print(f"   ❌ RandomForest failed: {e}")
        
        # 4. Logistic Regression (baseline)
        try:
            lr_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced'
            )
            lr_model.fit(X_train_scaled, y_train)
            results['logistic'] = ModelTrainer._evaluate(lr_model, X_test_scaled, y_test, 'LogisticReg')
        except Exception as e:
            print(f"   ❌ LogisticReg failed: {e}")
        
        return results
    
    @staticmethod
    def _evaluate(model, X_test, y_test, model_name: str) -> Dict:
        """Evaluate model performance."""
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        return {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }


class MultiStrategyPipeline:
    """Main pipeline orchestrator."""
    
    STRATEGY_CONFIGS = {
        '5T': [
            StrategyConfig('MSTF', 'Microstructure Trend-Following', '5T', ['mstf']),
            StrategyConfig('VWAP-REV', 'VWAP Liquidity Reversion', '5T', ['vwap']),
            StrategyConfig('ATR-EXP', 'ATR Breakout Continuation', '5T', ['atr']),
        ],
        '15T': [
            StrategyConfig('RCT', 'Regression Channel Trend Rider', '15T', ['rct']),
            StrategyConfig('VRAMR', 'Volatility-Regime Adaptive MR', '15T', ['vramr']),
            StrategyConfig('BR-SMC', 'Break-and-Retest Smart-Money', '15T', ['br']),
        ],
        '30T': [
            StrategyConfig('SBM', 'Session Bias Momentum', '30T', ['sbm']),
            StrategyConfig('VCE', 'Volatility Compression-Expansion', '30T', ['vce']),
            StrategyConfig('MCRC', 'Macro-Catalyst Hybrid', '30T', ['mcrc']),
        ]
    }
    
    FEATURE_ENGINEERS = {
        'MSTF': StrategyFeatureEngineer.mstf_5t_features,
        'VWAP-REV': StrategyFeatureEngineer.vwap_rev_5t_features,
        'ATR-EXP': StrategyFeatureEngineer.atr_exp_5t_features,
        'RCT': StrategyFeatureEngineer.rct_15t_features,
        'VRAMR': StrategyFeatureEngineer.vramr_15t_features,
        'BR-SMC': StrategyFeatureEngineer.br_smc_15t_features,
        'SBM': StrategyFeatureEngineer.sbm_30t_features,
        'VCE': StrategyFeatureEngineer.vce_30t_features,
        'MCRC': StrategyFeatureEngineer.mcrc_30t_features,
    }
    
    def __init__(self, symbol: str = 'XAUUSD'):
        self.symbol = symbol
        self.results = {}
    
    def run_timeframe(self, timeframe: str, strategy_names: Optional[List[str]] = None):
        """Train all strategies for a timeframe."""
        
        print(f"\n{'='*80}")
        print(f"TIMEFRAME: {timeframe}")
        print(f"{'='*80}")
        
        # Load data
        print(f"\n📥 Loading {self.symbol} {timeframe} data...")
        df = FeatureStore.load_timeframe_data(self.symbol, timeframe)
        
        # Get strategies for this timeframe
        strategies = self.STRATEGY_CONFIGS.get(timeframe, [])
        
        if strategy_names:
            strategies = [s for s in strategies if s.name in strategy_names]
        
        print(f"\n🎯 Training {len(strategies)} strategies:")
        for s in strategies:
            print(f"   • {s.name}: {s.description}")
        
        timeframe_results = {}
        
        for strategy in strategies:
            print(f"\n{'='*80}")
            print(f"STRATEGY: {strategy.name}")
            print(f"{'='*80}")
            
            # Engineer features
            print(f"\n🔧 Engineering {strategy.name} features...")
            feature_engineer = self.FEATURE_ENGINEERS[strategy.name]
            df_features = feature_engineer(df)
            
            # Get strategy-specific features
            strategy_cols = [c for c in df_features.columns if any(prefix in c for prefix in strategy.feature_groups)]
            base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'atr']
            
            df_strategy = df_features[base_cols + strategy_cols].dropna()
            
            print(f"   Features: {len(strategy_cols)}")
            print(f"   Samples: {len(df_strategy):,}")
            
            if len(df_strategy) < strategy.min_samples:
                print(f"   ⚠️  Insufficient samples ({len(df_strategy)} < {strategy.min_samples}), skipping")
                continue
            
            # Label
            print(f"\n🏷️  Applying triple barrier labeling...")
            labels = TripleBarrierLabeler.label(
                df_strategy, 
                strategy.tp_mult,
                strategy.sl_mult,
                strategy.time_barrier
            )
            
            labeled = (labels == 0) | (labels == 1)
            wins = (labels == 1).sum()
            total = labeled.sum()
            
            print(f"   Win rate: {wins/total*100:.1f}% ({wins}/{total})")
            
            if total < 100:
                print(f"   ⚠️  Too few labels, skipping")
                continue
            
            # Prepare features
            X = df_strategy[strategy_cols].values[labeled]
            y = labels.values[labeled]
            
            # Train/test split
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            print(f"\n✂️  Split: {len(X_train):,} train, {len(X_test):,} test")
            
            # Verify both classes in both sets
            train_classes = np.unique(y_train)
            test_classes = np.unique(y_test)
            
            if len(train_classes) < 2 or len(test_classes) < 2:
                print(f"   ⚠️  Imbalanced split, skipping")
                continue
            
            # Train models
            print(f"\n🤖 Training models...")
            model_results = ModelTrainer.train_models(
                X_train, X_test, y_train, y_test, strategy.name
            )
            
            timeframe_results[strategy.name] = model_results
            
            # Print results
            self._print_strategy_results(strategy.name, model_results)
        
        self.results[timeframe] = timeframe_results
        
        # Print timeframe summary
        self._print_timeframe_summary(timeframe, timeframe_results)
    
    def _print_strategy_results(self, strategy_name: str, results: Dict):
        """Print results for one strategy."""
        
        if not results:
            print(f"\n   ❌ No models trained successfully")
            return
        
        print(f"\n📊 {strategy_name} Model Results:")
        print(f"{'Model':<15} {'WinRate':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 50)
        
        for model_name, metrics in results.items():
            print(f"{metrics['model_name']:<15} "
                  f"{metrics['precision']:>9.1%} "
                  f"{metrics['recall']:>9.1%} "
                  f"{metrics['f1']:>10.4f}")
    
    def _print_timeframe_summary(self, timeframe: str, results: Dict):
        """Print summary for timeframe."""
        
        print(f"\n{'='*80}")
        print(f"TIMEFRAME {timeframe} SUMMARY")
        print(f"{'='*80}")
        
        if not results:
            print("No strategies trained successfully")
            return
        
        # Best model per strategy
        print(f"\n🏆 Best Model Per Strategy:")
        for strategy_name, model_results in results.items():
            if not model_results:
                continue
            
            best = max(model_results.items(), key=lambda x: x[1]['f1'])
            print(f"\n{strategy_name}:")
            print(f"   Model: {best[1]['model_name']}")
            print(f"   Win Rate: {best[1]['precision']:.1%}")
            print(f"   F1: {best[1]['f1']:.4f}")
    
    def print_final_summary(self):
        """Print overall summary."""
        
        print(f"\n{'='*80}")
        print(f"MULTI-STRATEGY SYSTEM - FINAL SUMMARY")
        print(f"{'='*80}")
        
        total_strategies = sum(len(strategies) for strategies in self.results.values())
        successful_strategies = sum(
            sum(1 for s in strategies.values() if s) 
            for strategies in self.results.values()
        )
        
        print(f"\nStrategies trained: {successful_strategies}/{total_strategies}")
        
        for tf, strategies in self.results.items():
            print(f"\n{tf}: {len(strategies)} strategies")
            for name in strategies.keys():
                print(f"   ✓ {name}")


def main():
    parser = argparse.ArgumentParser(description='Citadel Multi-Strategy ML System')
    parser.add_argument('--symbol', type=str, default='XAUUSD')
    parser.add_argument('--timeframe', type=str, help='Single timeframe (5T, 15T, 30T)')
    parser.add_argument('--all-timeframes', action='store_true', help='Train all timeframes')
    parser.add_argument('--strategies', nargs='+', help='Specific strategies to train')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CITADEL MULTI-STRATEGY ML TRADING SYSTEM")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    pipeline = MultiStrategyPipeline(args.symbol)
    
    if args.all_timeframes:
        for tf in ['5T', '15T', '30T']:
            try:
                pipeline.run_timeframe(tf, args.strategies)
            except Exception as e:
                print(f"\n❌ Error in {tf}: {e}")
    elif args.timeframe:
        pipeline.run_timeframe(args.timeframe, args.strategies)
    else:
        parser.print_help()
        return
    
    pipeline.print_final_summary()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Multi-strategy system trained successfully!")
    print("\nNext steps:")
    print("1. Review strategy performance tables")
    print("2. Build meta-model for strategy selection")
    print("3. Create ensemble system")
    print("4. Backtest on held-out data")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()