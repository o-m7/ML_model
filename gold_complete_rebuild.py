#!/usr/bin/env python3
"""
GOLD COMPLETE REBUILD - From Scratch
=====================================

Problem: Existing features are NOT predictive for Gold
Solution: Build Gold-specific features from raw OHLCV only

Gold trends strongly → Need trend-following features ONLY
- Momentum (multiple timeframes)
- Trend strength (ADX, MA slopes)
- Volatility breakouts
- Price action (HH/LL patterns)

NO mean-reversion, NO oscillators, NO SMC garbage
"""

import pickle
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")


class GoldConfig:
    FEATURE_STORE = Path("feature_store")
    MODEL_STORE = Path("models_production")
    
    TRAIN_START = "2019-01-01"
    TRAIN_END = "2025-10-22"
    OOS_MONTHS = 12
    
    # Gold-specific (AGGRESSIVE for trends)
    GOLD_PARAMS = {
        '15T': {
            'tp': 2.2,           # LONG TP for trends
            'sl': 1.0,
            'min_conf': 0.30,    # Less strict
            'min_edge': 0.05,
            'pos_size': 0.7,
            'forecast_horizon': 60,  # Look further ahead
            'min_move_atr': 1.8,     # Only label clear 1.8+ ATR moves
        },
        '1H': {
            'tp': 2.5,           # Even LONGER
            'sl': 1.0,
            'min_conf': 0.28,
            'min_edge': 0.04,
            'pos_size': 0.8,
            'forecast_horizon': 80,
            'min_move_atr': 2.0,     # Only label strong 2+ ATR moves
        },
    }
    
    INITIAL_CAPITAL = 100000
    RISK_PER_TRADE = 0.006  # 0.6%
    LEVERAGE = 20.0
    COMMISSION = 0.00005
    SLIPPAGE = 0.00002
    MAX_DD_BREAKER = 0.08
    
    # Higher expectations for Gold
    MIN_PROFIT_FACTOR = 1.35
    MAX_DRAWDOWN_PCT = 7.5
    MIN_SHARPE = 0.22
    MIN_WIN_RATE = 47.0
    MIN_TRADES_BY_TF = {'15T': 80, '1H': 50}


CONFIG = GoldConfig()
CONFIG.MODEL_STORE.mkdir(parents=True, exist_ok=True)


def build_gold_features_from_scratch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build POWERFUL trend-following features from raw OHLCV.
    These are designed specifically for Gold's trending nature.
    """
    
    print("  Building Gold-specific features from raw OHLCV...")
    
    df = df.copy()
    
    # === 1. MOMENTUM (Most important for Gold) ===
    print("    → Momentum features")
    for period in [3, 5, 10, 20, 40, 60]:
        df[f'mom_{period}'] = df['close'].pct_change(period)
    
    # Momentum acceleration (trend strengthening/weakening)
    df['mom_accel_short'] = df['mom_10'] - df['mom_5']
    df['mom_accel_long'] = df['mom_40'] - df['mom_20']
    
    # Momentum consistency (all pointing same direction = strong trend)
    df['mom_consistency'] = (
        np.sign(df['mom_5']) + 
        np.sign(df['mom_10']) + 
        np.sign(df['mom_20'])
    ) / 3  # -1 to +1
    
    # === 2. ATR (Volatility) ===
    print("    → Volatility features")
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_28'] = true_range.rolling(28).mean()
    df['atr_ratio'] = df['atr_14'] / (df['atr_28'] + 1e-10)  # Volatility expanding
    
    # Volatility breakout (high vol = trend starting)
    df['vol_pct'] = df['close'].pct_change().rolling(10).std()
    df['vol_breakout'] = (df['vol_pct'] > df['vol_pct'].rolling(50).mean() * 1.5).astype(int)
    
    # === 3. MOVING AVERAGES (Trend identification) ===
    print("    → Moving average features")
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_55'] = df['close'].ewm(span=55, adjust=False).mean()
    df['ema_144'] = df['close'].ewm(span=144, adjust=False).mean()
    
    # Distance from EMAs (normalized by ATR)
    atr = df['atr_14'].copy()
    atr = atr.where(atr > 0, df['close'] * 0.02)
    df['dist_ema_8'] = (df['close'] - df['ema_8']) / atr
    df['dist_ema_21'] = (df['close'] - df['ema_21']) / atr
    df['dist_ema_55'] = (df['close'] - df['ema_55']) / atr
    df['dist_ema_144'] = (df['close'] - df['ema_144']) / atr
    
    # EMA slopes (trend direction and strength)
    df['ema_8_slope'] = df['ema_8'].pct_change(5)
    df['ema_21_slope'] = df['ema_21'].pct_change(10)
    df['ema_55_slope'] = df['ema_55'].pct_change(20)
    
    # EMA alignment (all aligned = strong trend)
    df['ema_bullish'] = ((df['ema_8'] > df['ema_21']) & 
                          (df['ema_21'] > df['ema_55']) & 
                          (df['ema_55'] > df['ema_144'])).astype(int)
    df['ema_bearish'] = ((df['ema_8'] < df['ema_21']) & 
                          (df['ema_21'] < df['ema_55']) & 
                          (df['ema_55'] < df['ema_144'])).astype(int)
    
    # === 4. PRICE ACTION (HH/LL patterns) ===
    print("    → Price action features")
    df['hh'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['ll'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['hh_count_10'] = df['hh'].rolling(10).sum()
    df['ll_count_10'] = df['ll'].rolling(10).sum()
    
    # Range expansion/contraction
    df['range'] = (df['high'] - df['low']) / df['close']
    df['range_avg'] = df['range'].rolling(20).mean()
    df['range_expansion'] = df['range'] / (df['range_avg'] + 1e-10)
    
    # === 5. ADX (Trend strength) - Calculate manually ===
    print("    → ADX (trend strength)")
    
    # Directional movement
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = -df['low'].diff()
    
    df['plus_dm'] = np.where((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0), df['high_diff'], 0)
    df['minus_dm'] = np.where((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0), df['low_diff'], 0)
    
    # Smooth DM
    period = 14
    df['plus_dm_smooth'] = df['plus_dm'].ewm(span=period, adjust=False).mean()
    df['minus_dm_smooth'] = df['minus_dm'].ewm(span=period, adjust=False).mean()
    df['atr_smooth'] = true_range.ewm(span=period, adjust=False).mean()
    
    # Directional indicators
    df['plus_di'] = 100 * df['plus_dm_smooth'] / (df['atr_smooth'] + 1e-10)
    df['minus_di'] = 100 * df['minus_dm_smooth'] / (df['atr_smooth'] + 1e-10)
    
    # ADX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)
    df['adx'] = df['dx'].ewm(span=period, adjust=False).mean()
    
    df['adx_strong'] = (df['adx'] > 25).astype(int)
    df['adx_very_strong'] = (df['adx'] > 40).astype(int)
    df['adx_rising'] = (df['adx'] > df['adx'].shift(5)).astype(int)
    
    # === 6. DONCHIAN CHANNELS (Breakouts) ===
    print("    → Breakout features")
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['dc_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-10)
    df['breakout_high'] = (df['close'] >= df['high_20'].shift(1)).astype(int)
    df['breakout_low'] = (df['close'] <= df['low_20'].shift(1)).astype(int)
    
    # === 7. HIGHER-TIMEFRAME CONTEXT ===
    print("    → Multi-timeframe features")
    df['mom_120'] = df['close'].pct_change(120)  # ~2 days for 15T, ~5 days for 1H
    df['ema_233'] = df['close'].ewm(span=233, adjust=False).mean()
    atr_safe = df['atr_14'].copy()
    atr_safe = atr_safe.where(atr_safe > 0, df['close'] * 0.02)
    df['dist_ema_233'] = (df['close'] - df['ema_233']) / atr_safe
    
    # Clean up intermediates
    df = df.drop(['high_diff', 'low_diff', 'plus_dm', 'minus_dm', 
                  'plus_dm_smooth', 'minus_dm_smooth', 'atr_smooth', 
                  'dx', 'high_20', 'low_20'], axis=1, errors='ignore')
    
    num_features = len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])
    print(f"    ✅ Created {num_features} powerful trend-following features\n")
    
    return df


def create_strict_gold_labels(df: pd.DataFrame, tp_mult: float, sl_mult: float, 
                               horizon: int, min_move_atr: float) -> pd.DataFrame:
    """
    STRICT labeling: Only label moves that are CLEARLY directional.
    This prevents labeling noise as signal.
    """
    
    print(f"  Creating STRICT Gold labels...")
    print(f"    TP/SL: {tp_mult}:{sl_mult}")
    print(f"    Horizon: {horizon} bars")
    print(f"    Min move: {min_move_atr} ATR (only label strong moves)")
    
    df = df.copy()
    n = len(df)
    
    atr = df['atr_14'].fillna(df['close'] * 0.02).values
    entry = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    labels = np.zeros(n, dtype=int)
    
    for i in range(n - horizon):
        end = min(i + 1 + horizon, n)
        
        # Look ahead
        future_highs = highs[i+1:end]
        future_lows = lows[i+1:end]
        
        if len(future_highs) == 0:
            continue
        
        # Calculate max move in each direction
        max_up_move = (future_highs.max() - entry[i]) / atr[i]
        max_down_move = (entry[i] - future_lows.min()) / atr[i]
        
        # STRICT: Only label if move is >= min_move_atr AND hits TP before SL
        if max_up_move >= min_move_atr:
            tp = entry[i] + (atr[i] * tp_mult)
            sl = entry[i] - (atr[i] * sl_mult)
            
            tp_hit = np.where(future_highs >= tp)[0]
            sl_hit = np.where(future_lows <= sl)[0]
            
            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]):
                labels[i] = 1  # Clear UP trend
        
        elif max_down_move >= min_move_atr:
            tp = entry[i] - (atr[i] * tp_mult)
            sl = entry[i] + (atr[i] * sl_mult)
            
            tp_hit = np.where(future_lows <= tp)[0]
            sl_hit = np.where(future_highs >= sl)[0]
            
            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]):
                labels[i] = 2  # Clear DOWN trend
        
        # Everything else is Flat (no clear opportunity)
    
    df['target'] = labels
    df = df.iloc[:-horizon]
    
    counts = df['target'].value_counts()
    total = len(df)
    flat_pct = counts.get(0, 0) / total * 100
    up_pct = counts.get(1, 0) / total * 100
    down_pct = counts.get(2, 0) / total * 100
    
    print(f"    Flat: {counts.get(0, 0):,} ({flat_pct:.1f}%)")
    print(f"    Up:   {counts.get(1, 0):,} ({up_pct:.1f}%)")
    print(f"    Down: {counts.get(2, 0):,} ({down_pct:.1f}%)")
    
    # For Gold, we WANT some Flat (10-20%) but mostly directional
    if flat_pct < 10:
        print(f"    ⚠️  Very low Flat - might be overfitting")
    elif 10 <= flat_pct <= 25:
        print(f"    ✅ Good balance for trending Gold")
    else:
        print(f"    ⚠️  High Flat - may miss trends")
    
    print()
    return df


def select_best_features(df: pd.DataFrame, target_col: str = 'target') -> list:
    """Select only features that are actually predictive."""
    
    print("  Selecting most predictive features...")
    
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    candidates = [col for col in df.columns 
                  if col not in exclude 
                  and pd.api.types.is_numeric_dtype(df[col])]
    
    # Calculate predictive power
    df_copy = df.copy()
    df_copy['future_ret'] = df_copy['close'].pct_change(20).shift(-20)
    
    feature_scores = []
    for feat in candidates:
        corr = abs(df_copy[feat].corr(df_copy['future_ret']))
        if not np.isnan(corr):
            feature_scores.append((feat, corr))
    
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top features
    selected = [f[0] for f in feature_scores[:40]]
    
    print(f"    Top 10 most predictive:")
    for i, (feat, score) in enumerate(feature_scores[:10], 1):
        print(f"      {i:2d}. {feat:30s} | {score:.4f}")
    
    avg_score = np.mean([s[1] for s in feature_scores[:40]])
    print(f"\n    ✅ Selected top 40 features (avg correlation: {avg_score:.4f})")
    
    if avg_score < 0.015:
        print(f"    ⚠️  WARNING: Features still not very predictive")
    else:
        print(f"    ✅ Features are significantly predictive!")
    
    print()
    return selected


class PowerfulGoldModel:
    """Gold-specific model with heavy trend emphasis."""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        
        # BALANCED weights (don't over-boost Flat)
        counts = np.bincount(y)
        weights = len(y) / (len(counts) * counts)
        
        # Equal weight for Up/Down, moderate Flat
        weights[0] *= 1.2  # Flat
        weights[1] *= 1.5  # Up (boost trends)
        if len(weights) > 2:
            weights[2] *= 1.5  # Down (boost trends)
        
        sample_weight = weights[y]
        
        print(f"    Class weights: Flat={weights[0]:.2f}, Up={weights[1]:.2f}, Down={weights[2]:.2f}")
        
        # Less regularization (Gold has clear patterns)
        self.model = lgb.LGBMClassifier(
            n_estimators=300,      # More trees
            max_depth=6,           # Deeper trees
            learning_rate=0.02,    # Slower = more careful
            num_leaves=25,         # More leaves
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.5,         # Less regularization
            reg_lambda=2.0,
            min_child_samples=25,
            random_state=42,
            verbosity=-1,
            force_row_wise=True
        )
        
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        print(f"    ✅ Model trained\n")
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class GoldBacktest:
    """Backtest engine for Gold."""
    
    def __init__(self, df, params):
        self.df = df.copy()
        self.params = params
        self.trades = []
        self.equity = [CONFIG.INITIAL_CAPITAL]
        self.current_equity = CONFIG.INITIAL_CAPITAL
        self.peak = CONFIG.INITIAL_CAPITAL
        self.halted = False
        
    def run(self, signals_long, signals_short):
        active_trade = None
        
        for i in range(len(self.df)):
            self.equity.append(self.current_equity)
            
            # Circuit breaker
            dd = (self.peak - self.current_equity) / self.peak
            if dd > CONFIG.MAX_DD_BREAKER and not self.halted:
                self.halted = True
                if active_trade:
                    self._close(active_trade, i, self.df.iloc[i]['close'], 'breaker')
                    active_trade = None
            
            if self.halted:
                continue
            
            # Manage trade
            if active_trade:
                bar = self.df.iloc[i]
                
                if active_trade['dir'] == 'long':
                    if bar['low'] <= active_trade['sl']:
                        self._close(active_trade, i, active_trade['sl'], 'sl')
                        active_trade = None
                        continue
                    if bar['high'] >= active_trade['tp']:
                        self._close(active_trade, i, active_trade['tp'], 'tp')
                        active_trade = None
                        continue
                else:
                    if bar['high'] >= active_trade['sl']:
                        self._close(active_trade, i, active_trade['sl'], 'sl')
                        active_trade = None
                        continue
                    if bar['low'] <= active_trade['tp']:
                        self._close(active_trade, i, active_trade['tp'], 'tp')
                        active_trade = None
                        continue
                
                if (i - active_trade['entry_idx']) >= 100:
                    self._close(active_trade, i, bar['close'], 'timeout')
                    active_trade = None
                
                continue
            
            # New entry
            if i >= len(self.df) - 1:
                continue
            
            atr = self.df['atr_14'].iloc[i]
            
            if signals_long.iloc[i]:
                active_trade = self._enter(i, 'long', atr)
            elif signals_short.iloc[i]:
                active_trade = self._enter(i, 'short', atr)
        
        if active_trade:
            self._close(active_trade, len(self.df)-1, self.df.iloc[-1]['close'], 'end')
        
        return self._metrics()
    
    def _enter(self, idx, direction, atr):
        entry_bar = self.df.iloc[idx + 1]
        entry_price = entry_bar['open']
        entry_price = entry_price * (1 + CONFIG.SLIPPAGE) if direction == 'long' else entry_price * (1 - CONFIG.SLIPPAGE)
        
        tp_mult = self.params['tp']
        sl_mult = self.params['sl']
        
        if direction == 'long':
            sl = entry_price - (atr * sl_mult)
            tp = entry_price + (atr * tp_mult)
        else:
            sl = entry_price + (atr * sl_mult)
            tp = entry_price - (atr * tp_mult)
        
        risk = self.current_equity * CONFIG.RISK_PER_TRADE * self.params['pos_size']
        size = risk / abs(entry_price - sl)
        max_size = (self.current_equity * 0.20 * CONFIG.LEVERAGE) / entry_price
        size = min(size, max_size)
        
        return {
            'entry_idx': idx + 1,
            'entry': entry_price,
            'dir': direction,
            'size': size,
            'sl': sl,
            'tp': tp
        }
    
    def _close(self, trade, idx, exit_price, reason):
        exit_price = exit_price * (1 - CONFIG.SLIPPAGE) if trade['dir'] == 'long' else exit_price * (1 + CONFIG.SLIPPAGE)
        
        if trade['dir'] == 'long':
            pnl = trade['size'] * (exit_price - trade['entry'])
        else:
            pnl = trade['size'] * (trade['entry'] - exit_price)
        
        comm = trade['size'] * (trade['entry'] + exit_price) * CONFIG.COMMISSION
        pnl -= comm
        
        self.current_equity += pnl
        if self.current_equity > self.peak:
            self.peak = self.current_equity
        
        self.trades.append({'pnl': pnl, 'dir': trade['dir'], 'reason': reason})
    
    def _metrics(self):
        if not self.trades:
            return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                    'sharpe_ratio': 0, 'max_drawdown_pct': 0, 'total_return_pct': 0,
                    'long_trades': 0, 'short_trades': 0}
        
        tdf = pd.DataFrame(self.trades)
        wins = tdf[tdf['pnl'] > 0]
        losses = tdf[tdf['pnl'] <= 0]
        
        wr = len(wins) / len(tdf) * 100
        tw = wins['pnl'].sum() if len(wins) > 0 else 0
        tl = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
        pf = tw / tl if tl > 0 else 0
        
        eq = pd.Series(self.equity)
        rmax = eq.expanding().max()
        dd = (eq - rmax) / rmax
        mdd = abs(dd.min()) * 100
        
        rets = eq.pct_change().dropna()
        sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
        
        ret = (self.current_equity / CONFIG.INITIAL_CAPITAL - 1) * 100
        
        return {
            'total_trades': len(tdf),
            'long_trades': len(tdf[tdf['dir'] == 'long']),
            'short_trades': len(tdf[tdf['dir'] == 'short']),
            'win_rate': wr,
            'profit_factor': pf,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': mdd,
            'total_return_pct': ret
        }


def rebuild_gold(timeframe: str):
    """Complete rebuild for Gold."""
    
    print(f"\n{'='*80}")
    print(f"GOLD COMPLETE REBUILD: XAUUSD {timeframe}")
    print(f"{'='*80}\n")
    
    try:
        params = CONFIG.GOLD_PARAMS.get(timeframe)
        if not params:
            return {'symbol': 'XAUUSD', 'timeframe': timeframe, 'passed': False}
        
        print(f"Parameters: TP={params['tp']}, Horizon={params['forecast_horizon']}, Min Move={params['min_move_atr']} ATR\n")
        
        # Load RAW data
        print("[1/5] Loading RAW OHLCV data...")
        path = CONFIG.FEATURE_STORE / 'XAUUSD' / f"XAUUSD_{timeframe}.parquet"
        df = pd.read_parquet(path)
        
        if 'timestamp' not in df.columns:
            df = df.reset_index()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Keep only OHLCV
        keep_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in keep_cols if c in df.columns]]
        
        train_start = pd.to_datetime(CONFIG.TRAIN_START, utc=True)
        train_end = pd.to_datetime(CONFIG.TRAIN_END, utc=True)
        df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)]
        
        print(f"  Loaded {len(df):,} bars of raw OHLCV\n")
        
        # Build features FROM SCRATCH
        print("[2/5] Building Gold-specific features...")
        df = build_gold_features_from_scratch(df)
        
        # Create labels
        print("[3/5] Creating strict labels...")
        df = create_strict_gold_labels(df, params['tp'], params['sl'], 
                                        params['forecast_horizon'], params['min_move_atr'])
        
        # Select best features
        features = select_best_features(df)
        
        # Split
        print("[4/5] Training model...")
        oos_start = pd.to_datetime(CONFIG.TRAIN_END, utc=True) - timedelta(days=CONFIG.OOS_MONTHS * 30)
        train_df = df[df['timestamp'] < oos_start].copy()
        test_df = df[df['timestamp'] >= oos_start].copy()
        
        print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}\n")
        
        X_train = train_df[features].fillna(0).values
        y_train = train_df['target'].values
        
        model = PowerfulGoldModel()
        model.fit(X_train, y_train)
        
        # Backtest
        print("[5/5] Backtesting...")
        X_test = test_df[features].fillna(0).values
        probs = model.predict_proba(X_test)
        
        flat_p = pd.Series(probs[:, 0], index=test_df.index)
        long_p = pd.Series(probs[:, 1], index=test_df.index)
        short_p = pd.Series(probs[:, 2], index=test_df.index)
        
        signals_long = pd.Series(False, index=test_df.index)
        signals_short = pd.Series(False, index=test_df.index)
        
        for pos in range(len(test_df)):
            probs_i = [flat_p.iloc[pos], long_p.iloc[pos], short_p.iloc[pos]]
            max_prob = max(probs_i)
            sorted_p = sorted(probs_i, reverse=True)
            edge = sorted_p[0] - sorted_p[1]
            
            if long_p.iloc[pos] == max_prob and long_p.iloc[pos] >= params['min_conf'] and edge >= params['min_edge']:
                signals_long.iloc[pos] = True
            elif short_p.iloc[pos] == max_prob and short_p.iloc[pos] >= params['min_conf'] and edge >= params['min_edge']:
                signals_short.iloc[pos] = True
        
        print(f"  Signals: Long={signals_long.sum()}, Short={signals_short.sum()}\n")
        
        engine = GoldBacktest(test_df, params)
        results = engine.run(signals_long, signals_short)
        
        # Check
        min_trades = CONFIG.MIN_TRADES_BY_TF.get(timeframe, 60)
        
        failures = []
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
        
        passed = len(failures) == 0
        
        # Save
        save_dir = CONFIG.MODEL_STORE / 'XAUUSD'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        status = "PRODUCTION_READY" if passed else "FAILED"
        save_path = save_dir / f"XAUUSD_{timeframe}_{status}.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump({'model': model, 'features': features, 'results': results, 'params': params}, f)
        
        # Results
        print(f"{'='*80}")
        print(f"GOLD RESULTS: XAUUSD {timeframe}")
        print(f"{'='*80}")
        print(f"Trades:        {results['total_trades']} (L:{results['long_trades']}, S:{results['short_trades']})")
        print(f"Win Rate:      {results['win_rate']:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Sharpe:        {results['sharpe_ratio']:.2f}")
        print(f"Max DD:        {results['max_drawdown_pct']:.1f}%")
        print(f"Return:        {results['total_return_pct']:.1f}%")
        print(f"\n{'✅ GOLD IS NOW PROFITABLE!' if passed else '❌ FAILED: ' + ', '.join(failures)}")
        print(f"{'='*80}\n")
        
        return {'symbol': 'XAUUSD', 'timeframe': timeframe, 'passed': passed, 'results': results}
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'symbol': 'XAUUSD', 'timeframe': timeframe, 'passed': False, 'error': str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', type=str, help='15T or 1H')
    parser.add_argument('--all', action='store_true')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"GOLD COMPLETE REBUILD - Building from Raw OHLCV")
    print(f"{'='*80}\n")
    
    timeframes = ['15T', '1H'] if args.all else [args.tf] if args.tf else ['15T', '1H']
    
    results = []
    for tf in timeframes:
        result = rebuild_gold(tf)
        results.append(result)
    
    passed = sum(1 for r in results if r.get('passed', False))
    
    print(f"\n{'='*80}")
    print(f"REBUILD SUMMARY")
    print(f"{'='*80}")
    print(f"Tested: {len(results)}")
    print(f"Profitable: {passed}/{len(results)}")
    
    if passed > 0:
        print(f"\n✅ FIXED GOLD MODELS:")
        for r in results:
            if r.get('passed'):
                res = r['results']
                print(f"  {r['symbol']} {r['timeframe']}: PF={res['profit_factor']:.2f}, WR={res['win_rate']:.1f}%, Return={res['total_return_pct']:.1f}%")
    
    print(f"{'='*80}\n")
    
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    exit(main())

