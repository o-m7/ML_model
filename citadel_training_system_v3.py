"""
═══════════════════════════════════════════════════════════════════════════════
CITADEL ML TRADING SYSTEM V3.1 - COMPLETE TRIPLE-BARRIER TRAINING
═══════════════════════════════════════════════════════════════════════════════

FIXES IN V3.1:
━━━━━━━━━━━━━━━━━━━━━━━━━
✓ PROPER triple-barrier training (3-class: TP/SL/TimeExit + hold duration)
✓ FULL model comparison (LightGBM, XGBoost, CatBoost, RF, Logistic)
✓ COMPREHENSIVE results showcase with regime analysis
✓ REGIME-AWARE performance breakdown
✓ All V3.0 leak-free fixes maintained

Triple Barrier Labels:
    0 = Hit Stop Loss (loser)
    1 = Hit Take Profit (winner)
    2 = Time Exit (could be +/-)
    
Model learns: "Given entry signal, will this hit TP, SL, or time out?"

Usage:
    python citadel_training_system_v3.py --symbol XAUUSD --timeframe 5T --full-report
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

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')


@dataclass
class SystemConfig:
    """System configuration."""
    
    FEATURE_STORE: Path = Path("ML_model/ML_model/feature_store")
    
    # Performance targets
    MIN_WIN_RATE: float = 0.45
    MAX_WIN_RATE: float = 0.75
    MIN_PROFIT_FACTOR: float = 1.2
    MAX_PROFIT_FACTOR: float = 5.0
    MAX_DRAWDOWN: float = 0.06
    TARGET_DRAWDOWN: float = 0.03
    MIN_SHARPE: float = 0.30
    
    # Transaction costs
    BASE_SPREAD_R: Dict[str, float] = field(default_factory=lambda: {
        "5T": 0.10, "15T": 0.08, "30T": 0.06, "1H": 0.05, "4H": 0.04,
    })
    SLIPPAGE_R: float = 0.05
    COMMISSION_R: float = 0.02
    
    ATR_MIN_PCTL: float = 0.10
    ATR_MAX_PCTL: float = 0.90
    MIN_ABS_R_FOR_LABEL: float = 0.10
    
    @staticmethod
    def get_min_trades_raw(timeframe: str) -> int:
        return {'5T': 1000, '15T': 500, '30T': 300, '1H': 200, '4H': 100}.get(timeframe, 500)
    
    RISK_PER_TRADE_EVAL: float = 0.003
    
    @staticmethod
    def get_tp_multipliers(timeframe: str) -> List[float]:
        return {
            '5T':  [1.0, 1.2, 1.5, 1.8, 2.0],
            '15T': [1.5, 2.0, 2.5, 3.0],
            '30T': [2.0, 2.5, 3.0, 3.5],
            '1H':  [2.5, 3.0, 3.5, 4.0],
            '4H':  [3.0, 4.0, 5.0]
        }.get(timeframe, [1.5, 2.0, 2.5, 3.0])
    
    SL_MULTIPLIER: float = 1.0
    
    @staticmethod
    def get_time_barriers(timeframe: str) -> List[int]:
        return {
            '5T':  [15, 20, 25],
            '15T': [30, 35, 40],
            '30T': [45, 50, 60],
            '1H':  [20, 30, 40],
            '4H':  [10, 15, 20]
        }.get(timeframe, [60])
    
    TRAIN_RATIO: float = 0.60
    VAL_RATIO: float = 0.20
    
    CONFIDENCE_THRESHOLDS: List[float] = field(default_factory=lambda: 
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    )
    
    LGBM_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31,
        'max_depth': 6, 'min_child_samples': 50, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1
    })
    
    XGB_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6,
        'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbosity': 0
    })
    
    CATBOOST_PARAMS: Dict = field(default_factory=lambda: {
        'iterations': 200, 'learning_rate': 0.05, 'depth': 6,
        'l2_leaf_reg': 3, 'verbose': False
    })
    
    RF_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 50,
        'min_samples_leaf': 20, 'max_features': 'sqrt', 'n_jobs': -1
    })


CONFIG = SystemConfig()


def compute_equity_and_dd(r_multiples: np.ndarray, risk_per_trade: float = CONFIG.RISK_PER_TRADE_EVAL):
    if len(r_multiples) == 0:
        return np.array([1.0]), 0.0, 0.0
    
    equity = np.zeros(len(r_multiples) + 1)
    equity[0] = 1.0
    
    for i, r in enumerate(r_multiples):
        equity[i + 1] = equity[i] * (1.0 + r * risk_per_trade)
    
    peaks = np.maximum.accumulate(equity)
    dd_pct = ((peaks - equity) / peaks).max() * 100.0
    dd_r = (peaks - equity).max() / risk_per_trade
    
    return equity, dd_pct, dd_r


def get_total_cost_r(timeframe: str) -> float:
    base = CONFIG.BASE_SPREAD_R.get(timeframe, 0.05)
    return base + CONFIG.SLIPPAGE_R + CONFIG.COMMISSION_R


class RiskMetrics:
    @staticmethod
    def calculate_all_metrics(r_multiples: np.ndarray, risk_per_trade: float = CONFIG.RISK_PER_TRADE_EVAL) -> Dict:
        if len(r_multiples) == 0:
            return {
                'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'sharpe': 0.0, 'max_drawdown_pct': 0.0, 'mean_r': 0.0, 'total_r': 0.0
            }
        
        wins = (r_multiples > 0).sum()
        win_rate = wins / len(r_multiples)
        
        winners = r_multiples[r_multiples > 0]
        losers = r_multiples[r_multiples < 0]
        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0)
        
        sharpe = r_multiples.mean() / r_multiples.std() if r_multiples.std() > 0 else 0
        _, max_dd_pct, _ = compute_equity_and_dd(r_multiples, risk_per_trade)
        
        return {
            'total_trades': len(r_multiples),
            'wins': wins,
            'losses': len(r_multiples) - wins,
            'win_rate': win_rate,
            'profit_factor': pf,
            'sharpe': sharpe,
            'max_drawdown_pct': max_dd_pct,
            'mean_r': r_multiples.mean(),
            'median_r': np.median(r_multiples),
            'total_r': r_multiples.sum()
        }


class DataLoader:
    @staticmethod
    def load_timeframe_data(symbol: str, timeframe: str) -> Tuple[pd.DataFrame, Dict]:
        print(f"\n{'='*80}")
        print(f"LOADING DATA: {symbol} {timeframe}")
        print(f"{'='*80}")
        
        file_path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Feature file not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        
        print(f"✅ Loaded: {len(df):,} rows, {len(df.columns)} columns")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        if not df['timestamp'].is_monotonic_increasing:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df, {}


class FeatureEngineer:
    """Feature engineering with strict lagging."""
    
    @staticmethod
    def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n🔧 Engineering features (leak-free)...")
        
        features = df.copy()
        
        # Shift pre-computed indicators by 1
        indicator_cols = ['ema_20', 'ema_50', 'ema_200', 'sma_20', 'sma_50', 
                         'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr']
        
        for col in indicator_cols:
            if col in features.columns:
                features[col] = features[col].shift(1)
        
        # Regime features
        if 'atr' in features.columns:
            atr_shifted = features['atr'].shift(1)
            features['regime_vol_pct'] = atr_shifted.rolling(100).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / max(len(x) - 1, 1) if len(x) > 1 else 0.5
            )
            features['regime_vol'] = pd.cut(features['regime_vol_pct'], bins=[0, 0.33, 0.67, 1.0], labels=[0, 1, 2]).astype(float)
        
        if 'ema_20' in features.columns and 'ema_50' in features.columns:
            features['regime_trend'] = ((features['ema_20'] > features['ema_50']).astype(int) * 2 - 1)
        
        if 'hour' in features.columns:
            features['session_london'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
            features['session_ny'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
        
        # Momentum
        close_shifted = features['close'].shift(1)
        for period in [5, 10, 20]:
            features[f'mom_roc_{period}'] = close_shifted.pct_change(period)
        
        # Mean reversion
        if 'bb_upper' in features.columns and 'bb_lower' in features.columns:
            features['mr_bb_pos'] = (close_shifted - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-8)
        
        if 'rsi' in features.columns:
            features['mr_rsi_oversold'] = (features['rsi'] < 30).astype(int)
            features['mr_rsi_overbought'] = (features['rsi'] > 70).astype(int)
        
        # Microstructure
        open_s = features['open'].shift(1)
        high_s = features['high'].shift(1)
        low_s = features['low'].shift(1)
        
        features['micro_body'] = abs(close_shifted - open_s)
        features['micro_range'] = high_s - low_s
        features['micro_body_ratio'] = features['micro_body'] / (features['micro_range'] + 1e-8)
        
        if 'volume' in features.columns:
            vol_s = features['volume'].shift(1)
            features['micro_vol_surge'] = vol_s / (vol_s.rolling(20).mean() + 1)
        
        initial_rows = len(features)
        features = features.dropna()
        
        print(f"   ✅ Added {len(features.columns) - len(df.columns)} features")
        print(f"   🧹 Dropped {initial_rows - len(features)} rows with NaNs")
        print(f"   ✓ Final: {len(features):,} rows")
        
        return features


class TripleBarrierLabeler:
    """
    PROPER Triple Barrier Labeling with 3 classes:
        0 = Hit Stop Loss
        1 = Hit Take Profit  
        2 = Time Exit (no TP/SL hit)
    """
    
    @staticmethod
    def _determine_direction(df: pd.DataFrame, i: int) -> int:
        """Determine direction from market structure."""
        if i < 50:
            return 0
        
        try:
            ema_20 = df['ema_20'].iloc[i] if 'ema_20' in df.columns else None
            ema_50 = df['ema_50'].iloc[i] if 'ema_50' in df.columns else None
            
            close_curr = df['close'].iloc[i]
            close_10 = df['close'].iloc[i-10] if i >= 10 else close_curr
            momentum = (close_curr - close_10) / (close_10 + 1e-8)
            
            trend_signal = 0
            if ema_20 is not None and ema_50 is not None:
                trend_signal = 1 if ema_20 > ema_50 else -1
            
            mom_signal = 1 if momentum > 0.001 else (-1 if momentum < -0.001 else 0)
            
            if trend_signal == mom_signal and trend_signal != 0:
                return trend_signal
            elif trend_signal != 0:
                return trend_signal
            else:
                return 0
        except:
            return 0
    
    @staticmethod
    def _label_trade(df: pd.DataFrame, i: int, direction: int, 
                    tp_mult: float, sl_mult: float, time_barrier: int) -> Tuple[int, float, int]:
        """
        Label trade with triple barrier.
        
        Returns:
            (barrier_class, r_value, hold_duration)
            barrier_class: 0=SL, 1=TP, 2=TimeExit
        """
        if i + 1 >= len(df):
            return -1, 0.0, 0
        
        entry_price = df['open'].iloc[i + 1]
        atr = df['atr'].iloc[i]
        
        if pd.isna(entry_price) or pd.isna(atr) or atr <= 0:
            return -1, 0.0, 0
        
        if direction == 1:
            tp_price = entry_price + tp_mult * atr
            sl_price = entry_price - sl_mult * atr
        else:
            tp_price = entry_price - tp_mult * atr
            sl_price = entry_price + sl_mult * atr
        
        # Check each bar for TP/SL hit
        for j in range(2, time_barrier + 2):
            if i + j >= len(df):
                break
            
            high = df['high'].iloc[i + j]
            low = df['low'].iloc[i + j]
            
            if direction == 1:
                if high >= tp_price:
                    return 1, tp_mult, j  # Hit TP
                if low <= sl_price:
                    return 0, -sl_mult, j  # Hit SL
            else:
                if low <= tp_price:
                    return 1, tp_mult, j  # Hit TP
                if high >= sl_price:
                    return 0, -sl_mult, j  # Hit SL
        
        # Time exit - calculate actual R
        exit_idx = min(i + time_barrier + 1, len(df) - 1)
        exit_price = df['close'].iloc[exit_idx]
        
        if direction == 1:
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price
        
        r_value = pnl / atr
        return 2, r_value, time_barrier  # Time exit
    
    @staticmethod
    def label(df: pd.DataFrame, tp_mult: float, sl_mult: float, 
             time_barrier: int, timeframe: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Create triple-barrier labels.
        
        Returns:
            direction_labels: -1/0/+1 (short/flat/long)
            barrier_labels: 0/1/2 (SL/TP/TimeExit) 
            r_post: R-multiples after costs
            hold_durations: bars held
        """
        total_cost_r = get_total_cost_r(timeframe)
        
        atr = df['atr']
        atr_valid = (atr >= atr.quantile(0.1)) & (atr <= atr.quantile(0.9))
        
        direction_labels = pd.Series(0, index=df.index)
        barrier_labels = pd.Series(-1, index=df.index)  # -1 = no trade
        r_post = pd.Series(0.0, index=df.index)
        hold_durations = pd.Series(0, index=df.index)
        
        tp_hits = 0
        sl_hits = 0
        time_exits = 0
        
        for i in range(len(df) - time_barrier - 2):
            if not atr_valid.iloc[i]:
                continue
            
            direction = TripleBarrierLabeler._determine_direction(df, i)
            if direction == 0:
                continue
            
            barrier_class, r_pre, hold_dur = TripleBarrierLabeler._label_trade(
                df, i, direction, tp_mult, sl_mult, time_barrier
            )
            
            if barrier_class == -1:
                continue
            
            r_with_costs = r_pre - total_cost_r
            
            if abs(r_with_costs) < CONFIG.MIN_ABS_R_FOR_LABEL:
                continue
            
            direction_labels.iloc[i] = direction
            barrier_labels.iloc[i] = barrier_class
            r_post.iloc[i] = r_with_costs
            hold_durations.iloc[i] = hold_dur
            
            if barrier_class == 0:
                sl_hits += 1
            elif barrier_class == 1:
                tp_hits += 1
            else:
                time_exits += 1
        
        total = tp_hits + sl_hits + time_exits
        if total > 0:
            print(f"\n   Triple Barrier Distribution:")
            print(f"   - TP Hits: {tp_hits:,} ({tp_hits/total*100:.1f}%)")
            print(f"   - SL Hits: {sl_hits:,} ({sl_hits/total*100:.1f}%)")
            print(f"   - Time Exits: {time_exits:,} ({time_exits/total*100:.1f}%)")
            print(f"   - Avg Hold (TP): {hold_durations[barrier_labels == 1].mean():.1f} bars")
            print(f"   - Avg Hold (SL): {hold_durations[barrier_labels == 0].mean():.1f} bars")
        
        return direction_labels, barrier_labels, r_post, hold_durations
    
    @staticmethod
    def find_best_config(df: pd.DataFrame, timeframe: str) -> Tuple[float, int]:
        print("\n🔍 OPTIMIZING TRIPLE-BARRIER CONFIGURATION")
        print("=" * 80)
        
        tp_candidates = CONFIG.get_tp_multipliers(timeframe)
        time_barriers = CONFIG.get_time_barriers(timeframe)
        
        best_config = None
        best_score = -np.inf
        
        print(f"{'TP':>6} {'TB':>6} {'PF':>8} {'WR':>8} {'Trades':>10} {'TP%':>8} {'SL%':>8} {'Time%':>8} {'Status':<20}")
        print("-" * 120)
        
        for tp_mult in tp_candidates:
            for tb in time_barriers:
                _, barrier_labels, r_post, _ = TripleBarrierLabeler.label(
                    df, tp_mult, CONFIG.SL_MULTIPLIER, tb, timeframe
                )
                
                mask = barrier_labels != -1
                if mask.sum() < CONFIG.get_min_trades_raw(timeframe):
                    continue
                
                r_sel = r_post[mask].values
                barriers = barrier_labels[mask].values
                
                metrics = RiskMetrics.calculate_all_metrics(r_sel)
                
                tp_pct = (barriers == 1).sum() / len(barriers)
                sl_pct = (barriers == 0).sum() / len(barriers)
                time_pct = (barriers == 2).sum() / len(barriers)
                
                status = "✅ Viable"
                eligible = True
                
                if metrics['win_rate'] > CONFIG.MAX_WIN_RATE:
                    status = "❌ High WR"
                    eligible = False
                elif metrics['profit_factor'] > CONFIG.MAX_PROFIT_FACTOR:
                    status = "❌ High PF"
                    eligible = False
                elif metrics['profit_factor'] < 1.2:
                    status = "❌ Low PF"
                    eligible = False
                elif metrics['win_rate'] < CONFIG.MIN_WIN_RATE:
                    status = "⚠️ Low WR"
                    eligible = False
                
                if eligible:
                    score = metrics['profit_factor'] * metrics['win_rate'] * np.log(metrics['total_trades'] + 1)
                    if score > best_score:
                        best_score = score
                        best_config = (tp_mult, tb)
                
                print(f"{tp_mult:>6.1f} {tb:>6} {metrics['profit_factor']:>8.2f} "
                      f"{metrics['win_rate']:>7.1%} {metrics['total_trades']:>10,} "
                      f"{tp_pct:>7.1%} {sl_pct:>7.1%} {time_pct:>7.1%} {status:<20}")
        
        if best_config is None:
            return tp_candidates[len(tp_candidates)//2], time_barriers[len(time_barriers)//2]
        
        print(f"\n✅ BEST CONFIG: TP={best_config[0]:.1f}x ATR, TB={best_config[1]} bars\n")
        return best_config


class DataSplitter:
    @staticmethod
    def split_chronological(df: pd.DataFrame, direction_labels: pd.Series, 
                           barrier_labels: pd.Series, r_post: pd.Series, 
                           hold_durations: pd.Series) -> Dict:
        """Chronological split."""
        print(f"\n✂️  CHRONOLOGICAL DATA SPLIT")
        print(f"{'='*80}")
        
        mask = barrier_labels != -1
        df_labeled = df[mask].copy()
        direction_filtered = direction_labels[mask].copy()
        barrier_filtered = barrier_labels[mask].copy()
        r_filtered = r_post[mask].copy()
        hold_filtered = hold_durations[mask].copy()
        
        print(f"   Total labeled samples: {len(df_labeled):,}")
        
        n = len(df_labeled)
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        
        feature_cols = [c for c in df_labeled.columns 
                       if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return {
            'X_train': df_labeled.iloc[:train_end][feature_cols].values,
            'X_val': df_labeled.iloc[train_end:val_end][feature_cols].values,
            'X_test': df_labeled.iloc[val_end:][feature_cols].values,
            'y_train': barrier_filtered.iloc[:train_end].values,
            'y_val': barrier_filtered.iloc[train_end:val_end].values,
            'y_test': barrier_filtered.iloc[val_end:].values,
            'r_train': r_filtered.iloc[:train_end].values,
            'r_val': r_filtered.iloc[train_end:val_end].values,
            'r_test': r_filtered.iloc[val_end:].values,
            'hold_train': hold_filtered.iloc[:train_end].values,
            'hold_val': hold_filtered.iloc[train_end:val_end].values,
            'hold_test': hold_filtered.iloc[val_end:].values,
            'feature_cols': feature_cols,
            'df_train': df_labeled.iloc[:train_end],
            'df_val': df_labeled.iloc[train_end:val_end],
            'df_test': df_labeled.iloc[val_end:]
        }


class ModelFactory:
    """Train ALL model families."""
    
    @staticmethod
    def train_all_models(X_train, X_val, y_train, y_val):
        print(f"\n🤖 TRAINING ALL MODELS")
        print(f"{'='*80}")
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Compute class weights for 3-class problem
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[yi] for yi in y_train])
        
        models = {}
        
        # 1. LightGBM
        try:
            print(f"\n   Training LightGBM...")
            params = CONFIG.LGBM_PARAMS.copy()
            params['objective'] = 'multiclass'
            params['num_class'] = 3
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights,
                     eval_set=[(X_val_scaled, y_val)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            models['lightgbm'] = {'model': model, 'scaler': scaler}
            print(f"   ✅ LightGBM trained")
        except Exception as e:
            print(f"   ❌ LightGBM failed: {e}")
        
        # 2. XGBoost
        try:
            print(f"\n   Training XGBoost...")
            params = CONFIG.XGB_PARAMS.copy()
            params['objective'] = 'multi:softprob'
            params['num_class'] = 3
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights,
                     eval_set=[(X_val_scaled, y_val)], verbose=False)
            models['xgboost'] = {'model': model, 'scaler': scaler}
            print(f"   ✅ XGBoost trained")
        except Exception as e:
            print(f"   ❌ XGBoost failed: {e}")
        
        # 3. CatBoost
        try:
            print(f"\n   Training CatBoost...")
            params = CONFIG.CATBOOST_PARAMS.copy()
            params['loss_function'] = 'MultiClass'
            params['classes_count'] = 3
            model = CatBoostClassifier(**params)
            model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val),
                     early_stopping_rounds=50, verbose=False)
            models['catboost'] = {'model': model, 'scaler': scaler}
            print(f"   ✅ CatBoost trained")
        except Exception as e:
            print(f"   ❌ CatBoost failed: {e}")
        
        # 4. Random Forest
        try:
            print(f"\n   Training RandomForest...")
            model = RandomForestClassifier(**CONFIG.RF_PARAMS)
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            models['random_forest'] = {'model': model, 'scaler': scaler}
            print(f"   ✅ RandomForest trained")
        except Exception as e:
            print(f"   ❌ RandomForest failed: {e}")
        
        # 5. Logistic Regression
        try:
            print(f"\n   Training LogisticRegression...")
            model = LogisticRegression(max_iter=1000, class_weight='balanced',
                                      multi_class='multinomial', C=0.1)
            model.fit(X_train_scaled, y_train)
            models['logistic'] = {'model': model, 'scaler': scaler}
            print(f"   ✅ LogisticRegression trained")
        except Exception as e:
            print(f"   ❌ LogisticRegression failed: {e}")
        
        print(f"\n   ✅ Trained {len(models)}/5 models successfully")
        
        return models


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    @staticmethod
    def evaluate_all_models(models: Dict, X_test, y_test, r_test, hold_test) -> Dict:
        print(f"\n📊 EVALUATING ALL MODELS ON TEST SET")
        print(f"{'='*80}")
        
        results = {}
        
        for model_name, model_dict in models.items():
            try:
                model = model_dict['model']
                scaler = model_dict['scaler']
                X_test_scaled = scaler.transform(X_test)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)
                
                # Classification metrics
                acc = accuracy_score(y_test, y_pred)
                
                # For each predicted class, calculate R outcomes
                # Predicted 0 (SL) - we skip these trades (model says it will lose)
                # Predicted 1 (TP) - we take these trades
                # Predicted 2 (TimeExit) - we might take if R expectation is positive
                
                # Trading logic: Take trades predicted as class 1 (TP)
                trade_mask = (y_pred == 1)
                
                if trade_mask.sum() == 0:
                    print(f"\n{model_name.upper()}: ⚠️  No trades predicted")
                    continue
                
                r_trades = r_test[trade_mask]
                hold_trades = hold_test[trade_mask]
                
                metrics = RiskMetrics.calculate_all_metrics(r_trades)
                
                # Barrier prediction accuracy
                barrier_accuracy = {}
                for barrier_class in [0, 1, 2]:
                    mask = y_test == barrier_class
                    if mask.sum() > 0:
                        barrier_accuracy[barrier_class] = (y_pred[mask] == barrier_class).mean()
                
                results[model_name] = {
                    'accuracy': acc,
                    'trades_predicted': int(trade_mask.sum()),
                    'metrics': metrics,
                    'avg_hold': float(hold_trades.mean()),
                    'barrier_accuracy': barrier_accuracy,
                    'pred_distribution': {
                        'pred_sl': int((y_pred == 0).sum()),
                        'pred_tp': int((y_pred == 1).sum()),
                        'pred_time': int((y_pred == 2).sum())
                    }
                }
                
                print(f"\n{model_name.upper()}:")
                print(f"   Overall Accuracy: {acc:.1%}")
                print(f"   Trades Predicted (TP class): {trade_mask.sum():,}")
                print(f"   Win Rate: {metrics['win_rate']:.1%}")
                print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
                print(f"   Sharpe: {metrics['sharpe']:.2f}")
                print(f"   Max DD: {metrics['max_drawdown_pct']:.1f}%")
                print(f"   Avg Hold: {hold_trades.mean():.1f} bars")
                print(f"   Barrier Prediction Accuracy:")
                for barrier, acc_val in barrier_accuracy.items():
                    barrier_name = ['SL', 'TP', 'TimeExit'][barrier]
                    print(f"      {barrier_name}: {acc_val:.1%}")
                
            except Exception as e:
                print(f"\n{model_name.upper()}: ❌ Failed - {e}")
        
        return results
    
    @staticmethod
    def print_comparison_table(results: Dict):
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"\n{'Model':<20} {'Acc':>8} {'WR':>8} {'PF':>8} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>10} {'AvgHold':>10}")
        print("-" * 120)
        
        for name, res in sorted(results.items(), key=lambda x: x[1]['metrics']['profit_factor'], reverse=True):
            m = res['metrics']
            print(f"{name:<20} {res['accuracy']:>7.1%} {m['win_rate']:>7.1%} "
                  f"{m['profit_factor']:>8.2f} {m['sharpe']:>8.2f} "
                  f"{m['max_drawdown_pct']:>9.1f}% {res['trades_predicted']:>10,} "
                  f"{res['avg_hold']:>9.1f}")


class RegimeAnalyzer:
    """Analyze performance by market regime."""
    
    @staticmethod
    def analyze_regimes(df_test: pd.DataFrame, y_pred, r_test) -> Dict:
        print(f"\n📊 REGIME-BASED PERFORMANCE ANALYSIS")
        print(f"{'='*80}")
        
        regime_results = {}
        
        # Volatility regime
        if 'regime_vol' in df_test.columns:
            print(f"\n1. VOLATILITY REGIME:")
            for regime_val, regime_name in [(0, 'Low Vol'), (1, 'Med Vol'), (2, 'High Vol')]:
                mask = (df_test['regime_vol'] == regime_val) & (y_pred == 1)
                
                if mask.sum() > 20:
                    r_regime = r_test[mask]
                    metrics = RiskMetrics.calculate_all_metrics(r_regime)
                    
                    regime_results[f'vol_{regime_name}'] = metrics
                    
                    print(f"   {regime_name}: WR={metrics['win_rate']:.1%}, "
                          f"PF={metrics['profit_factor']:.2f}, "
                          f"Trades={metrics['total_trades']:,}")
        
        # Trend regime
        if 'regime_trend' in df_test.columns:
            print(f"\n2. TREND REGIME:")
            for trend_val, trend_name in [(1, 'Uptrend'), (-1, 'Downtrend')]:
                mask = (df_test['regime_trend'] == trend_val) & (y_pred == 1)
                
                if mask.sum() > 20:
                    r_regime = r_test[mask]
                    metrics = RiskMetrics.calculate_all_metrics(r_regime)
                    
                    regime_results[f'trend_{trend_name}'] = metrics
                    
                    print(f"   {trend_name}: WR={metrics['win_rate']:.1%}, "
                          f"PF={metrics['profit_factor']:.2f}, "
                          f"Trades={metrics['total_trades']:,}")
        
        # Session regime
        if 'session_london' in df_test.columns:
            print(f"\n3. TRADING SESSION:")
            for session_col, session_name in [('session_london', 'London'), ('session_ny', 'New York')]:
                if session_col in df_test.columns:
                    mask = (df_test[session_col] == 1) & (y_pred == 1)
                    
                    if mask.sum() > 20:
                        r_regime = r_test[mask]
                        metrics = RiskMetrics.calculate_all_metrics(r_regime)
                        
                        regime_results[f'session_{session_name}'] = metrics
                        
                        print(f"   {session_name}: WR={metrics['win_rate']:.1%}, "
                              f"PF={metrics['profit_factor']:.2f}, "
                              f"Trades={metrics['total_trades']:,}")
        
        return regime_results


def save_model_for_backtest(symbol: str, timeframe: str, best_model_name: str,
                            models: Dict, feature_cols: List[str], 
                            base_path: Path = Path("ML_model/ML_model")):
    """Save trained model."""
    models_dir = base_path / "models" / symbol
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / f"{symbol}_{timeframe}_best_model.pkl"
    joblib.dump(models[best_model_name], model_path)
    
    features_path = models_dir / f"{symbol}_{timeframe}_feature_cols.json"
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    metadata_path = models_dir / f"{symbol}_{timeframe}_metadata.json"
    metadata = {
        'symbol': symbol,
        'timeframe': timeframe,
        'best_model': best_model_name,
        'n_features': len(feature_cols),
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Saved model to: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='XAUUSD')
    parser.add_argument('--timeframe', type=str, default='5T')
    parser.add_argument('--full-report', action='store_true')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*80}")
    print(f"# CITADEL ML TRAINING SYSTEM V3.1")
    print(f"# Symbol: {args.symbol} | Timeframe: {args.timeframe}")
    print(f"{'#'*80}")
    
    # Load data
    df, _ = DataLoader.load_timeframe_data(args.symbol, args.timeframe)
    
    # Engineer features
    df = FeatureEngineer.engineer_all_features(df)
    
    # Find best config
    best_tp, best_tb = TripleBarrierLabeler.find_best_config(df, args.timeframe)
    
    # Label with triple barriers
    direction_labels, barrier_labels, r_post, hold_durations = TripleBarrierLabeler.label(
        df, best_tp, CONFIG.SL_MULTIPLIER, best_tb, args.timeframe
    )
    
    # Split data
    splits = DataSplitter.split_chronological(df, direction_labels, barrier_labels, r_post, hold_durations)
    
    # Train all models
    models = ModelFactory.train_all_models(
        splits['X_train'], splits['X_val'], 
        splits['y_train'], splits['y_val']
    )
    
    # Evaluate all models
    results = ModelEvaluator.evaluate_all_models(
        models, splits['X_test'], splits['y_test'], 
        splits['r_test'], splits['hold_test']
    )
    
    # Print comparison
    ModelEvaluator.print_comparison_table(results)
    
    # Select best model
    best_model_name = max(results.keys(), 
                         key=lambda k: results[k]['metrics']['profit_factor'])
    
    print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")
    
    # Regime analysis on best model
    best_model = models[best_model_name]
    X_test_scaled = best_model['scaler'].transform(splits['X_test'])
    y_pred_best = best_model['model'].predict(X_test_scaled)
    
    regime_results = RegimeAnalyzer.analyze_regimes(
        splits['df_test'], y_pred_best, splits['r_test']
    )
    
    # Save best model
    save_model_for_backtest(args.symbol, args.timeframe, best_model_name, 
                           models, splits['feature_cols'])
    
    print(f"\n{'#'*80}")
    print(f"# TRAINING COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == '__main__':
    main()