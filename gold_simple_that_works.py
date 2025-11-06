#!/usr/bin/env python3
"""
GOLD - SIMPLE APPROACH THAT ACTUALLY WORKS
===========================================

Problem: Over-engineered system is failing (WR < 30%)
Solution: Go back to BASICS that work

Simple approach:
1. Use EXISTING features from feature store (they work for other pairs)
2. REALISTIC TP/SL (1.5:1, not 2.5:1)
3. BALANCED labels (30% Flat, not 55%)
4. CLEAR signal logic (no complex filtering)
5. Proper position sizing

No tricks, no complexity - just solid ML trading.
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


class SimpleConfig:
    FEATURE_STORE = Path("feature_store")
    MODEL_STORE = Path("models_production")
    
    TRAIN_START = "2019-01-01"
    TRAIN_END = "2025-10-22"
    OOS_MONTHS = 12
    
    # REALISTIC Gold parameters
    GOLD_PARAMS = {
        '15T': {
            'tp': 1.6,           # REALISTIC (not 2.2)
            'sl': 1.0,
            'min_conf': 0.32,
            'min_edge': 0.06,
            'pos_size': 0.6,
            'horizon': 40,       # Standard
            'flat_mult': 0.85,   # Only 85% of move needed = more directional labels
        },
        '1H': {
            'tp': 1.7,           # REALISTIC (not 2.5)
            'sl': 1.0,
            'min_conf': 0.30,
            'min_edge': 0.05,
            'pos_size': 0.7,
            'horizon': 40,
            'flat_mult': 0.80,   # Even less strict
        },
    }
    
    INITIAL_CAPITAL = 100000
    RISK_PER_TRADE = 0.006
    LEVERAGE = 20.0
    COMMISSION = 0.00005
    SLIPPAGE = 0.00002
    MAX_DD = 0.08
    
    MIN_PF = 1.30
    MAX_DD_PCT = 7.5
    MIN_SHARPE = 0.20
    MIN_WR = 46.0
    MIN_TRADES = {'15T': 80, '1H': 50}


CONFIG = SimpleConfig()
CONFIG.MODEL_STORE.mkdir(parents=True, exist_ok=True)


def simple_labels(df: pd.DataFrame, tp_mult: float, sl_mult: float, 
                  horizon: int, flat_mult: float) -> pd.DataFrame:
    """
    Simple, proven labeling method.
    Not too strict, not too loose.
    """
    
    print(f"  Creating labels (TP:{tp_mult}, SL:{sl_mult}, Flat mult:{flat_mult})...")
    
    df = df.copy()
    n = len(df)
    
    atr = df.get('atr14', df['close'] * 0.02).values
    entry = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Full TP/SL for checking
    tp_long_full = entry + (atr * tp_mult)
    sl_long_full = entry - (atr * sl_mult)
    tp_short_full = entry - (atr * tp_mult)
    sl_short_full = entry + (atr * sl_mult)
    
    # Reduced for labeling (more lenient)
    tp_long = entry + (atr * tp_mult * flat_mult)
    sl_long = entry - (atr * sl_mult * flat_mult)
    tp_short = entry - (atr * tp_mult * flat_mult)
    sl_short = entry + (atr * sl_mult * flat_mult)
    
    labels = np.zeros(n, dtype=int)
    
    for i in range(n - horizon):
        end = min(i + 1 + horizon, n)
        fut_h = highs[i+1:end]
        fut_l = lows[i+1:end]
        
        if len(fut_h) == 0:
            continue
        
        # Check reduced TP hits
        tp_long_hit = np.where(fut_h >= tp_long[i])[0]
        sl_long_hit = np.where(fut_l <= sl_long_full[i])[0]
        tp_short_hit = np.where(fut_l <= tp_short[i])[0]
        sl_short_hit = np.where(fut_h >= sl_short_full[i])[0]
        
        # Label if TP hits before SL
        if len(tp_long_hit) > 0 and (len(sl_long_hit) == 0 or tp_long_hit[0] < sl_long_hit[0]):
            labels[i] = 1
        elif len(tp_short_hit) > 0 and (len(sl_short_hit) == 0 or tp_short_hit[0] < sl_short_hit[0]):
            labels[i] = 2
    
    df['target'] = labels
    df = df.iloc[:-horizon]
    
    counts = df['target'].value_counts()
    total = len(df)
    print(f"    Flat: {counts.get(0,0):,} ({counts.get(0,0)/total*100:.1f}%)")
    print(f"    Up:   {counts.get(1,0):,} ({counts.get(1,0)/total*100:.1f}%)")
    print(f"    Down: {counts.get(2,0):,} ({counts.get(2,0)/total*100:.1f}%)")
    
    return df


def select_features(df: pd.DataFrame) -> list:
    """Select good features from feature store."""
    
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    smc = ['swing', 'fvg', 'ob_', 'bos', 'choch', 'eq_', 'order', 'liquidity', 
           'fair_value', 'inducement', 'mitigation', 'breaker', 'displacement', 'imbalance']
    
    candidates = [col for col in df.columns 
                  if col not in exclude 
                  and pd.api.types.is_numeric_dtype(df[col])
                  and not any(s in col.lower() for s in smc)]
    
    # Remove high correlation
    if len(candidates) > 35:
        corr = df[candidates].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
        candidates = [c for c in candidates if c not in to_drop]
    
    return candidates[:30]


class SimpleModel:
    """Simple, reliable LightGBM."""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        
        # BALANCED weights (don't over-do it)
        counts = np.bincount(y)
        weights = len(y) / (len(counts) * counts)
        weights[0] *= 1.3  # Moderate Flat boost
        weights[1] *= 1.4  # Slight Up boost
        if len(weights) > 2:
            weights[2] *= 1.4  # Slight Down boost
        
        sample_weight = weights[y]
        
        print(f"    Weights: Flat={weights[0]:.2f}, Up={weights[1]:.2f}, Down={weights[2]:.2f}")
        
        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=16,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=2.5,
            reg_lambda=3.5,
            min_child_samples=35,
            random_state=42,
            verbosity=-1,
            force_row_wise=True
        )
        
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        
    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


class SimpleBacktest:
    """Simple, clean backtest."""
    
    def __init__(self, df, params):
        self.df = df.copy()
        self.params = params
        self.trades = []
        self.equity = [CONFIG.INITIAL_CAPITAL]
        self.curr = CONFIG.INITIAL_CAPITAL
        self.peak = CONFIG.INITIAL_CAPITAL
        self.halted = False
        
    def run(self, probs):
        """Run backtest with SIMPLE signal logic."""
        
        active = None
        
        for i in range(len(self.df)):
            self.equity.append(self.curr)
            
            # Circuit breaker
            dd = (self.peak - self.curr) / self.peak
            if dd > CONFIG.MAX_DD:
                self.halted = True
                if active:
                    self._close(active, i, self.df.iloc[i]['close'], 'breaker')
                    active = None
            
            if self.halted:
                continue
            
            # Manage position
            if active:
                bar = self.df.iloc[i]
                
                if active['dir'] == 'long':
                    if bar['low'] <= active['sl']:
                        self._close(active, i, active['sl'], 'sl')
                        active = None
                        continue
                    if bar['high'] >= active['tp']:
                        self._close(active, i, active['tp'], 'tp')
                        active = None
                        continue
                else:
                    if bar['high'] >= active['sl']:
                        self._close(active, i, active['sl'], 'sl')
                        active = None
                        continue
                    if bar['low'] <= active['tp']:
                        self._close(active, i, active['tp'], 'tp')
                        active = None
                        continue
                
                if (i - active['entry_idx']) >= 80:
                    self._close(active, i, bar['close'], 'timeout')
                    active = None
                
                continue
            
            # New signal - SIMPLE logic
            if i >= len(self.df) - 1:
                continue
            
            prob_flat = probs[i, 0]
            prob_long = probs[i, 1]
            prob_short = probs[i, 2]
            
            max_prob = max(prob_flat, prob_long, prob_short)
            probs_sorted = sorted([prob_flat, prob_long, prob_short], reverse=True)
            edge = probs_sorted[0] - probs_sorted[1]
            
            # SIMPLE: If model is confident AND has edge, take the trade
            if max_prob >= self.params['min_conf'] and edge >= self.params['min_edge']:
                atr = self.df['atr14'].iloc[i] if 'atr14' in self.df.columns else self.df['close'].iloc[i] * 0.02
                
                if prob_long == max_prob:
                    active = self._enter(i, 'long', atr, prob_long)
                elif prob_short == max_prob:
                    active = self._enter(i, 'short', atr, prob_short)
        
        if active:
            self._close(active, len(self.df)-1, self.df.iloc[-1]['close'], 'end')
        
        return self._metrics()
    
    def _enter(self, idx, direction, atr, conf):
        entry = self.df.iloc[idx + 1]['open']
        entry = entry * (1 + CONFIG.SLIPPAGE) if direction == 'long' else entry * (1 - CONFIG.SLIPPAGE)
        
        if direction == 'long':
            sl = entry - (atr * self.params['sl'])
            tp = entry + (atr * self.params['tp'])
        else:
            sl = entry + (atr * self.params['sl'])
            tp = entry - (atr * self.params['tp'])
        
        risk = self.curr * CONFIG.RISK_PER_TRADE * self.params['pos_size']
        size = risk / abs(entry - sl)
        max_size = (self.curr * 0.20 * CONFIG.LEVERAGE) / entry
        size = min(size, max_size)
        
        return {
            'entry_idx': idx + 1,
            'entry': entry,
            'dir': direction,
            'size': size,
            'sl': sl,
            'tp': tp,
            'conf': conf
        }
    
    def _close(self, trade, idx, exit_price, reason):
        exit_price = exit_price * (1 - CONFIG.SLIPPAGE) if trade['dir'] == 'long' else exit_price * (1 + CONFIG.SLIPPAGE)
        
        if trade['dir'] == 'long':
            pnl = trade['size'] * (exit_price - trade['entry'])
        else:
            pnl = trade['size'] * (trade['entry'] - exit_price)
        
        comm = trade['size'] * (trade['entry'] + exit_price) * CONFIG.COMMISSION
        pnl -= comm
        
        self.curr += pnl
        if self.curr > self.peak:
            self.peak = self.curr
        
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
        ret = (self.curr / CONFIG.INITIAL_CAPITAL - 1) * 100
        
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


def train_gold_simple(timeframe: str):
    """Simple approach that actually works."""
    
    print(f"\n{'='*80}")
    print(f"GOLD SIMPLE: XAUUSD {timeframe}")
    print(f"{'='*80}\n")
    
    try:
        params = CONFIG.GOLD_PARAMS.get(timeframe)
        if not params:
            return {'symbol': 'XAUUSD', 'timeframe': timeframe, 'passed': False}
        
        # Load data
        print("[1/5] Loading data...")
        path = CONFIG.FEATURE_STORE / 'XAUUSD' / f"XAUUSD_{timeframe}.parquet"
        df = pd.read_parquet(path)
        
        if 'timestamp' not in df.columns:
            df = df.reset_index()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        train_start = pd.to_datetime(CONFIG.TRAIN_START, utc=True)
        train_end = pd.to_datetime(CONFIG.TRAIN_END, utc=True)
        df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)]
        
        print(f"  {len(df):,} bars\n")
        
        # Labels
        print("[2/5] Creating labels...")
        df = simple_labels(df, params['tp'], params['sl'], params['horizon'], params['flat_mult'])
        
        # Features
        print("\n[3/5] Selecting features...")
        features = select_features(df)
        print(f"  Using {len(features)} features\n")
        
        # Split
        print("[4/5] Training...")
        oos_start = pd.to_datetime(CONFIG.TRAIN_END, utc=True) - timedelta(days=CONFIG.OOS_MONTHS * 30)
        train_df = df[df['timestamp'] < oos_start].copy()
        test_df = df[df['timestamp'] >= oos_start].copy()
        
        print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}\n")
        
        X_train = train_df[features].fillna(0).values
        y_train = train_df['target'].values
        X_test = test_df[features].fillna(0).values
        
        model = SimpleModel()
        model.fit(X_train, y_train)
        
        # Backtest
        print("\n[5/5] Backtesting...")
        probs = model.predict_proba(X_test)
        
        engine = SimpleBacktest(test_df, params)
        results = engine.run(probs)
        
        print(f"  Trades: {results['total_trades']}")
        
        # Check
        min_trades = CONFIG.MIN_TRADES.get(timeframe, 60)
        
        failures = []
        if results['profit_factor'] < CONFIG.MIN_PF:
            failures.append(f"PF {results['profit_factor']:.2f}")
        if results['max_drawdown_pct'] > CONFIG.MAX_DD_PCT:
            failures.append(f"DD {results['max_drawdown_pct']:.1f}%")
        if results['sharpe_ratio'] < CONFIG.MIN_SHARPE:
            failures.append(f"Sharpe {results['sharpe_ratio']:.2f}")
        if results['win_rate'] < CONFIG.MIN_WR:
            failures.append(f"WR {results['win_rate']:.1f}%")
        if results['total_trades'] < min_trades:
            failures.append(f"Trades {results['total_trades']}")
        
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
        print(f"RESULTS: XAUUSD {timeframe}")
        print(f"{'='*80}")
        print(f"Trades:  {results['total_trades']} (L:{results['long_trades']}, S:{results['short_trades']})")
        print(f"WR:      {results['win_rate']:.1f}%")
        print(f"PF:      {results['profit_factor']:.2f}")
        print(f"Sharpe:  {results['sharpe_ratio']:.2f}")
        print(f"DD:      {results['max_drawdown_pct']:.1f}%")
        print(f"Return:  {results['total_return_pct']:.1f}%")
        print(f"\n{'✅ PASS' if passed else '❌ FAIL: ' + ', '.join(failures)}")
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
    parser.add_argument('--tf', type=str)
    parser.add_argument('--all', action='store_true')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GOLD - SIMPLE APPROACH (No Over-Engineering)")
    print("="*80 + "\n")
    
    tfs = ['15T', '1H'] if args.all else [args.tf] if args.tf else ['15T', '1H']
    
    results = []
    for tf in tfs:
        result = train_gold_simple(tf)
        results.append(result)
    
    passed = sum(1 for r in results if r.get('passed', False))
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Pass rate: {passed}/{len(results)}")
    
    if passed > 0:
        print(f"\n✅ PROFITABLE:")
        for r in results:
            if r.get('passed'):
                res = r['results']
                print(f"  {r['symbol']} {r['timeframe']}: PF={res['profit_factor']:.2f}, WR={res['win_rate']:.1f}%")
    
    print(f"{'='*80}\n")
    
    return 0 if passed > 0 else 1


if __name__ == '__main__':
    exit(main())

