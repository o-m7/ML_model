#!/usr/bin/env python3
"""
GOLD (XAUUSD) SPECIALIZED OPTIMIZER
====================================

Gold is a TRENDING instrument - requires different approach:
1. Longer TP targets (capture trends)
2. Trend-following features emphasized
3. Less "Flat" labels (Gold moves decisively)
4. Proper directional balance
5. Aggressive position sizing (high liquidity)

Current Issues:
- XAUUSD 15T: PF 1.21 (should be 1.5+)
- XAUUSD 1H: PF 0.97 (LOSING MONEY - unacceptable for Gold)

Root Cause Analysis:
- Not capturing trends properly
- Too conservative labeling
- Wrong feature emphasis
"""

import argparse
import json
import pickle
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

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
    
    # GOLD-SPECIFIC PARAMETERS (Trend-following optimized)
    GOLD_PARAMS = {
        '15T': {
            'tp': 2.0,           # LONGER TP for trends
            'sl': 1.0,
            'min_conf': 0.32,    # Less strict - catch trends early
            'min_edge': 0.05,    # Less strict
            'pos_size': 0.8,     # AGGRESSIVE (Gold is liquid)
            'flat_threshold': 0.80,  # LESS Flat labels (Gold trends)
            'class_weights': [1.2, 1.5, 1.5],  # Boost directional
            'max_bars': 80,      # Hold longer for trends
        },
        '1H': {
            'tp': 2.2,           # Even LONGER for 1H trends
            'sl': 1.0,
            'min_conf': 0.30,
            'min_edge': 0.05,
            'pos_size': 0.9,     # Very aggressive on 1H
            'flat_threshold': 0.75,  # Even less Flat
            'class_weights': [1.0, 1.8, 1.8],  # STRONG directional boost
            'max_bars': 100,     # Hold even longer
        },
        '4H': {
            'tp': 2.5,           # Maximum TP for multi-day trends
            'sl': 1.0,
            'min_conf': 0.28,
            'min_edge': 0.05,
            'pos_size': 1.0,     # Full size
            'flat_threshold': 0.70,
            'class_weights': [0.8, 2.0, 2.0],  # Minimize Flat
            'max_bars': 150,
        },
    }
    
    FORECAST_HORIZON = 50  # Longer for Gold
    INITIAL_CAPITAL = 100000
    RISK_PER_TRADE = 0.006  # 0.6% for Gold (higher than other pairs)
    LEVERAGE = 20.0         # Higher leverage for Gold
    COMMISSION = 0.00005    # Gold typically has lower commission
    SLIPPAGE = 0.00002
    MAX_DRAWDOWN_CIRCUIT_BREAKER = 0.08  # 8% for Gold
    
    # Gold-specific benchmarks
    MIN_PROFIT_FACTOR = 1.40   # Higher expectation for Gold
    MAX_DRAWDOWN_PCT = 7.5
    MIN_SHARPE = 0.25          # Higher expectation
    MIN_WIN_RATE = 48.0
    
    MIN_TRADES_BY_TF = {
        '15T': 100,
        '1H': 60,
        '4H': 30,
    }


CONFIG = GoldConfig()
CONFIG.MODEL_STORE.mkdir(parents=True, exist_ok=True)


def create_gold_labels(df: pd.DataFrame, tp_mult: float, sl_mult: float, 
                       flat_threshold: float = 0.80) -> pd.DataFrame:
    """
    Gold-specific labeling: Less Flat, more directional.
    Gold TRENDS, so we want to capture directional moves.
    """
    
    print(f"  Creating GOLD labels (Trend-optimized)...")
    print(f"    TP/SL: {tp_mult}:{sl_mult}")
    print(f"    Flat threshold: {flat_threshold} (lower = less Flat)")
    
    df = df.copy()
    n = len(df)
    horizon = CONFIG.FORECAST_HORIZON
    
    atr = df.get('atr14', df['close'] * 0.02).values
    entry = df['close'].values
    
    # Use FULL TP/SL for initial check
    tp_long_full = entry + (atr * tp_mult)
    sl_long_full = entry - (atr * sl_mult)
    tp_short_full = entry - (atr * tp_mult)
    sl_short_full = entry + (atr * sl_mult)
    
    # Use REDUCED TP/SL for labeling (to get more directional labels)
    tp_long = entry + (atr * tp_mult * flat_threshold)
    sl_long = entry - (atr * sl_mult * flat_threshold)
    tp_short = entry - (atr * tp_mult * flat_threshold)
    sl_short = entry + (atr * sl_mult * flat_threshold)
    
    labels = np.zeros(n, dtype=int)
    
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(n - horizon):
        end = min(i + 1 + horizon, n)
        future_highs = highs[i+1:end]
        future_lows = lows[i+1:end]
        
        if len(future_highs) == 0:
            continue
        
        # Check if move is strong enough in one direction
        max_move_up = (future_highs.max() - entry[i]) / atr[i]
        max_move_down = (entry[i] - future_lows.min()) / atr[i]
        
        # Label as directional if move >= flat_threshold * TP
        if max_move_up >= tp_mult * flat_threshold:
            # Check that SL doesn't hit first
            sl_hit = np.where(future_lows <= sl_long_full[i])[0]
            tp_hit = np.where(future_highs >= tp_long[i])[0]
            
            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]):
                labels[i] = 1  # Up
        elif max_move_down >= tp_mult * flat_threshold:
            # Check that SL doesn't hit first
            sl_hit = np.where(future_highs >= sl_short_full[i])[0]
            tp_hit = np.where(future_lows <= tp_short[i])[0]
            
            if len(tp_hit) > 0 and (len(sl_hit) == 0 or tp_hit[0] < sl_hit[0]):
                labels[i] = 2  # Down
    
    df['target'] = labels
    df = df.iloc[:-CONFIG.FORECAST_HORIZON]
    
    counts = df['target'].value_counts()
    total = len(df)
    flat_pct = counts.get(0, 0) / total * 100
    up_pct = counts.get(1, 0) / total * 100
    down_pct = counts.get(2, 0) / total * 100
    
    print(f"    Flat: {counts.get(0, 0):,} ({flat_pct:.1f}%)")
    print(f"    Up:   {counts.get(1, 0):,} ({up_pct:.1f}%)")
    print(f"    Down: {counts.get(2, 0):,} ({down_pct:.1f}%)")
    
    if flat_pct < 15:
        print(f"    ✅ Good for GOLD (low Flat = captures trends)")
    elif 15 <= flat_pct <= 25:
        print(f"    ✅ Acceptable")
    else:
        print(f"    ⚠️  High Flat ({flat_pct:.1f}%) - may miss trends")
    
    return df


def add_gold_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gold-specific features: HEAVY emphasis on trend detection.
    """
    
    print("  Adding GOLD-optimized features (trend-focused)...")
    
    df = df.copy()
    
    # === MOMENTUM (Critical for Gold) ===
    for p in [5, 10, 20, 50]:  # Added 50-period for longer trends
        df[f'mom_{p}'] = df['close'].pct_change(p)
    
    # Momentum acceleration (trend strength)
    df['mom_accel'] = df['mom_10'] - df['mom_5']
    df['mom_accel_long'] = df['mom_50'] - df['mom_20']
    
    # === TREND STRENGTH ===
    if 'ema20' in df.columns and 'ema50' in df.columns and 'ema200' in df.columns:
        # Trend hierarchy
        df['trend_20_50'] = ((df['ema20'] > df['ema50']).astype(int) * 2 - 1)
        df['trend_50_200'] = ((df['ema50'] > df['ema200']).astype(int) * 2 - 1)
        df['trend_aligned'] = (df['trend_20_50'] == df['trend_50_200']).astype(int)  # Strong trend
        
        # Distance from moving averages (trend strength)
        atr = df.get('atr14', df['close'] * 0.02)
        df['dist_ema20'] = (df['close'] - df['ema20']) / atr
        df['dist_ema50'] = (df['close'] - df['ema50']) / atr
        df['dist_ema200'] = (df['close'] - df['ema200']) / atr
        
        # MA separation (trend strength indicator)
        df['ma_sep_20_50'] = abs(df['ema20'] - df['ema50']) / atr
        df['ma_sep_50_200'] = abs(df['ema50'] - df['ema200']) / atr
    
    # === ADX (Trend Strength Indicator) ===
    if 'adx' in df.columns:
        df['adx_strong'] = (df['adx'] > 25).astype(int)
        df['adx_very_strong'] = (df['adx'] > 40).astype(int)
        df['adx_momentum'] = df['adx'].diff(5)  # ADX rising = trend strengthening
    
    # === VOLATILITY ===
    df['vol_5'] = df['close'].pct_change().rolling(5).std()
    df['vol_20'] = df['close'].pct_change().rolling(20).std()
    df['vol_50'] = df['close'].pct_change().rolling(50).std()
    df['vol_regime'] = (df['vol_5'] > df['vol_20']).astype(int)
    df['vol_expansion'] = df['vol_5'] / (df['vol_20'] + 1e-10)
    
    # === RSI ===
    if 'rsi14' in df.columns:
        df['rsi_norm'] = (df['rsi14'] - 50) / 50
        df['rsi_extreme'] = ((df['rsi14'] < 30) | (df['rsi14'] > 70)).astype(int)
        df['rsi_momentum'] = df['rsi14'].diff(5)
    
    # === BOLLINGER BANDS ===
    if 'bb_pct' in df.columns:
        df['bb_pos'] = (df['bb_pct'] - 0.5) * 2
        df['bb_squeeze'] = (df['bb_pct'] > 0.2) & (df['bb_pct'] < 0.8)  # Not at extremes
    
    # === PRICE ACTION ===
    # Higher highs / Lower lows (trend confirmation)
    df['hh'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['ll'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['hh_count'] = df['hh'].rolling(10).sum()
    df['ll_count'] = df['ll'].rolling(10).sum()
    
    # === VOLUME ===
    if 'volume' in df.columns:
        df['vol_surge'] = df['volume'] / df['volume'].rolling(20).mean()
        df['vol_trend'] = df['volume'].rolling(10).mean() / df['volume'].rolling(50).mean()
    
    print(f"    Added {len([c for c in df.columns if c not in ['open','high','low','close','volume','timestamp']])} features")
    
    return df


def select_gold_features(df: pd.DataFrame) -> List[str]:
    """Select features for Gold (prioritize trend indicators)."""
    
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'target', 'expected_return', 'expected_duration']
    
    smc = ['swing', 'fvg', 'ob_', 'bos', 'choch', 'eq_', 'order_block', 'orderblock',
           'fair_value', 'fairvalue', 'liquidity', 'liq_', 'inducement', 'mitigation',
           'breaker', 'rejection', 'displacement', 'imbalance']
    
    candidates = [col for col in df.columns 
                  if col not in exclude 
                  and pd.api.types.is_numeric_dtype(df[col])
                  and not any(pattern in col.lower() for pattern in smc)]
    
    print(f"    Starting with {len(candidates)} candidates")
    
    # Remove highly correlated
    if len(candidates) > 40:
        corr_matrix = df[candidates].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        candidates = [f for f in candidates if f not in to_drop]
        print(f"    Removed {len(to_drop)} highly correlated")
    
    # Prioritize trend features
    trend_keywords = ['mom', 'trend', 'adx', 'ma_sep', 'dist_ema', 'hh', 'll']
    trend_features = [f for f in candidates if any(kw in f.lower() for kw in trend_keywords)]
    other_features = [f for f in candidates if f not in trend_features]
    
    # Take top trend features + some others
    selected = trend_features[:25] + other_features[:15]
    
    print(f"    ✅ Selected {len(selected)} features (prioritized trend indicators)")
    
    return selected[:40]  # Max 40 features


class GoldModel:
    """Gold-specific model with trend emphasis."""
    
    def __init__(self, class_weights: List[float] = [1.2, 1.5, 1.5]):
        self.model = None
        self.scaler = RobustScaler()
        self.class_weights_mult = class_weights
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        
        counts = np.bincount(y)
        weights = len(y) / (len(counts) * counts)
        
        for i, mult in enumerate(self.class_weights_mult):
            if i < len(weights):
                weights[i] *= mult
        
        sample_weight = weights[y]
        
        print(f"    Class weights: Flat={weights[0]:.2f}, Up={weights[1]:.2f}, Down={weights[2]:.2f}")
        
        # Gold-optimized LightGBM
        self.model = lgb.LGBMClassifier(
            n_estimators=250,      # More trees for Gold
            max_depth=5,           # Slightly deeper
            learning_rate=0.025,   # Slower learning
            num_leaves=20,         # More leaves
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=2.0,         # Less regularization (Gold has clear patterns)
            reg_lambda=2.5,
            min_child_samples=30,
            random_state=42,
            verbosity=-1,
            force_row_wise=True
        )
        
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        
        # Feature importance
        importances = self.model.feature_importances_
        print(f"    Model trained. Avg feature importance: {importances.mean():.4f}")
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self, feature_names):
        """Get top features."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n  Top 10 Most Important Features:")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            print(f"    {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


class GoldBacktest:
    """Gold-optimized backtest engine."""
    
    def __init__(self, df: pd.DataFrame, position_mult: float, max_bars: int):
        self.df = df.copy()
        self.position_mult = position_mult
        self.max_bars = max_bars
        self.trades = []
        self.equity = [CONFIG.INITIAL_CAPITAL]
        self.current_equity = CONFIG.INITIAL_CAPITAL
        self.peak = CONFIG.INITIAL_CAPITAL
        self.trading_halted = False
        
    def run(self, signals_long, signals_short, probs_long, probs_short, tp_mult, sl_mult):
        active_trade = None
        
        for i in range(len(self.df)):
            self.equity.append(self.current_equity)
            
            dd = (self.peak - self.current_equity) / self.peak
            if dd > CONFIG.MAX_DRAWDOWN_CIRCUIT_BREAKER and not self.trading_halted:
                self.trading_halted = True
                print(f"    ⚠️  Circuit breaker at bar {i} (DD: {dd*100:.1f}%)")
                if active_trade:
                    self._close_trade(active_trade, i, self.df.iloc[i]['close'], 'circuit_breaker')
                    active_trade = None
            
            if self.trading_halted:
                continue
            
            if active_trade:
                bar = self.df.iloc[i]
                
                if active_trade['direction'] == 'long':
                    if bar['low'] <= active_trade['sl']:
                        self._close_trade(active_trade, i, active_trade['sl'], 'sl')
                        active_trade = None
                        continue
                    if bar['high'] >= active_trade['tp']:
                        self._close_trade(active_trade, i, active_trade['tp'], 'tp')
                        active_trade = None
                        continue
                else:
                    if bar['high'] >= active_trade['sl']:
                        self._close_trade(active_trade, i, active_trade['sl'], 'sl')
                        active_trade = None
                        continue
                    if bar['low'] <= active_trade['tp']:
                        self._close_trade(active_trade, i, active_trade['tp'], 'tp')
                        active_trade = None
                        continue
                
                if (i - active_trade['entry_idx']) >= self.max_bars:
                    self._close_trade(active_trade, i, bar['close'], 'timeout')
                    active_trade = None
                
                continue
            
            if i >= len(self.df) - 1:
                continue
            
            atr = self.df['atr14'].iloc[i] if 'atr14' in self.df.columns else self.df['close'].iloc[i] * 0.02
            
            if signals_long.iloc[i]:
                active_trade = self._enter_trade(i, 'long', probs_long.iloc[i], tp_mult, sl_mult, atr)
            elif signals_short.iloc[i]:
                active_trade = self._enter_trade(i, 'short', probs_short.iloc[i], tp_mult, sl_mult, atr)
        
        if active_trade:
            self._close_trade(active_trade, len(self.df)-1, self.df.iloc[-1]['close'], 'end')
        
        return self._calculate_metrics()
    
    def _enter_trade(self, idx, direction, confidence, tp_mult, sl_mult, atr):
        entry_bar = self.df.iloc[idx + 1]
        entry_price = entry_bar['open']
        entry_price = entry_price * (1 + CONFIG.SLIPPAGE) if direction == 'long' else entry_price * (1 - CONFIG.SLIPPAGE)
        
        if direction == 'long':
            sl_price = entry_price - (atr * sl_mult)
            tp_price = entry_price + (atr * tp_mult)
        else:
            sl_price = entry_price + (atr * sl_mult)
            tp_price = entry_price - (atr * tp_mult)
        
        risk_amount = self.current_equity * CONFIG.RISK_PER_TRADE * self.position_mult
        position_size = risk_amount / abs(entry_price - sl_price)
        max_position = (self.current_equity * 0.20 * CONFIG.LEVERAGE) / entry_price
        position_size = min(position_size, max_position)
        
        return {
            'entry_idx': idx + 1,
            'entry_price': entry_price,
            'direction': direction,
            'position_size': position_size,
            'sl': sl_price,
            'tp': tp_price,
            'confidence': confidence
        }
    
    def _close_trade(self, trade, idx, exit_price, reason):
        exit_price = exit_price * (1 - CONFIG.SLIPPAGE) if trade['direction'] == 'long' else exit_price * (1 + CONFIG.SLIPPAGE)
        
        if trade['direction'] == 'long':
            price_change = exit_price - trade['entry_price']
        else:
            price_change = trade['entry_price'] - exit_price
        
        gross_pnl = trade['position_size'] * price_change
        commission = trade['position_size'] * (trade['entry_price'] + exit_price) * CONFIG.COMMISSION
        net_pnl = gross_pnl - commission
        
        self.current_equity += net_pnl
        if self.current_equity > self.peak:
            self.peak = self.current_equity
        
        self.trades.append({'pnl': net_pnl, 'direction': trade['direction'], 'reason': reason})
    
    def _calculate_metrics(self):
        if len(self.trades) == 0:
            return {
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'sharpe_ratio': 0, 'max_drawdown_pct': 0, 'total_return_pct': 0,
                'long_trades': 0, 'short_trades': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(wins) / len(trades_df) * 100
        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        equity_series = pd.Series(self.equity)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_dd = abs(drawdown.min()) * 100
        
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        total_return = (self.current_equity / CONFIG.INITIAL_CAPITAL - 1) * 100
        
        return {
            'total_trades': len(trades_df),
            'long_trades': len(trades_df[trades_df['direction'] == 'long']),
            'short_trades': len(trades_df[trades_df['direction'] == 'short']),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'total_return_pct': total_return
        }


def optimize_gold(timeframe: str) -> Dict:
    """Optimize XAUUSD for specific timeframe."""
    
    print(f"\n{'='*80}")
    print(f"GOLD OPTIMIZER: XAUUSD {timeframe}")
    print(f"{'='*80}\n")
    
    try:
        params = CONFIG.GOLD_PARAMS.get(timeframe)
        if not params:
            print(f"  ⚠️  No params for {timeframe}")
            return {'symbol': 'XAUUSD', 'timeframe': timeframe, 'passed': False}
        
        print(f"Gold-Specific Parameters:")
        print(f"  TP/SL Ratio: {params['tp']}:{params['sl']} (trend-following)")
        print(f"  Min Confidence: {params['min_conf']} (less strict)")
        print(f"  Position Size: {params['pos_size']*100:.0f}% (aggressive)")
        print(f"  Max Bars: {params['max_bars']} (hold trends longer)")
        print(f"  Flat Threshold: {params['flat_threshold']} (less Flat labels)")
        
        # Load
        print(f"\n[1/5] Loading data...")
        path = CONFIG.FEATURE_STORE / 'XAUUSD' / f"XAUUSD_{timeframe}.parquet"
        df = pd.read_parquet(path)
        
        if 'timestamp' not in df.columns:
            df = df.reset_index()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        train_start = pd.to_datetime(CONFIG.TRAIN_START, utc=True)
        train_end = pd.to_datetime(CONFIG.TRAIN_END, utc=True)
        df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)]
        
        print(f"  Loaded {len(df):,} bars of GOLD data")
        
        # Features & Labels
        print("\n[2/5] Engineering features & labels...")
        df = add_gold_features(df)
        df = create_gold_labels(df, params['tp'], params['sl'], params['flat_threshold'])
        features = select_gold_features(df)
        
        # Split
        print("\n[3/5] Training Gold model...")
        oos_start = pd.to_datetime(CONFIG.TRAIN_END, utc=True) - timedelta(days=CONFIG.OOS_MONTHS * 30)
        train_df = df[df['timestamp'] < oos_start].copy()
        test_df = df[df['timestamp'] >= oos_start].copy()
        
        print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
        
        X_train = train_df[features].fillna(0).values
        y_train = train_df['target'].values
        
        model = GoldModel(class_weights=params['class_weights'])
        model.fit(X_train, y_train)
        model.get_feature_importance(features)
        
        # Backtest
        print("\n[4/5] Backtesting...")
        X_test = test_df[features].fillna(0).values
        probs = model.predict_proba(X_test)
        
        flat_probs = pd.Series(probs[:, 0], index=test_df.index)
        long_probs = pd.Series(probs[:, 1], index=test_df.index)
        short_probs = pd.Series(probs[:, 2], index=test_df.index)
        
        signals_long = pd.Series(False, index=test_df.index)
        signals_short = pd.Series(False, index=test_df.index)
        
        for pos in range(len(test_df)):
            probs_i = [flat_probs.iloc[pos], long_probs.iloc[pos], short_probs.iloc[pos]]
            max_prob = max(probs_i)
            sorted_probs = sorted(probs_i, reverse=True)
            edge = sorted_probs[0] - sorted_probs[1]
            
            if long_probs.iloc[pos] == max_prob and long_probs.iloc[pos] >= params['min_conf'] and edge >= params['min_edge']:
                signals_long.iloc[pos] = True
            elif short_probs.iloc[pos] == max_prob and short_probs.iloc[pos] >= params['min_conf'] and edge >= params['min_edge']:
                signals_short.iloc[pos] = True
        
        print(f"  Signals generated: Long={signals_long.sum()}, Short={signals_short.sum()}")
        
        engine = GoldBacktest(test_df, params['pos_size'], params['max_bars'])
        results = engine.run(signals_long, signals_short, long_probs, short_probs, params['tp'], params['sl'])
        
        # Check
        print("\n[5/5] Evaluating...")
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
        print(f"\n{'='*80}")
        print(f"GOLD RESULTS: XAUUSD {timeframe}")
        print(f"{'='*80}")
        print(f"Trades:        {results['total_trades']} (L:{results['long_trades']}, S:{results['short_trades']})")
        print(f"Win Rate:      {results['win_rate']:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Sharpe Ratio:  {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:  {results['max_drawdown_pct']:.1f}%")
        print(f"Total Return:  {results['total_return_pct']:.1f}%")
        print(f"\n{'✅ GOLD IS PROFITABLE!' if passed else '❌ STILL FAILING: ' + ', '.join(failures)}")
        print(f"{'='*80}\n")
        
        return {
            'symbol': 'XAUUSD',
            'timeframe': timeframe,
            'passed': passed,
            'results': results
        }
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'symbol': 'XAUUSD', 'timeframe': timeframe, 'passed': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', type=str, help='Timeframe (15T, 1H, 4H)')
    parser.add_argument('--all', action='store_true', help='Optimize all Gold timeframes')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"GOLD (XAUUSD) TREND-FOLLOWING OPTIMIZER")
    print(f"{'='*80}")
    print(f"\nGold is a TRENDING instrument.")
    print(f"Strategy: Longer TP targets, less Flat labels, trend-following features")
    print(f"{'='*80}\n")
    
    if args.all:
        timeframes = ['15T', '1H', '4H']
    elif args.tf:
        timeframes = [args.tf]
    else:
        timeframes = ['15T', '1H']  # Default
    
    results = []
    for tf in timeframes:
        result = optimize_gold(tf)
        results.append(result)
    
    # Summary
    passed = sum(1 for r in results if r.get('passed', False))
    
    print(f"\n{'='*80}")
    print(f"GOLD OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Timeframes tested: {len(results)}")
    print(f"Profitable: {passed}/{len(results)}")
    
    if passed > 0:
        print(f"\n✅ PROFITABLE GOLD MODELS:")
        for r in results:
            if r.get('passed'):
                res = r['results']
                print(f"  XAUUSD {r['timeframe']}: {res['total_trades']} trades, PF={res['profit_factor']:.2f}, Return={res['total_return_pct']:.1f}%")
    
    still_failing = [r for r in results if not r.get('passed')]
    if still_failing:
        print(f"\n❌ STILL NOT PROFITABLE:")
        for r in still_failing:
            print(f"  XAUUSD {r['timeframe']}")
            if 'results' in r:
                print(f"    PF: {r['results']['profit_factor']:.2f}, WR: {r['results']['win_rate']:.1f}%")
    
    print(f"{'='*80}\n")
    
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    exit(main())

