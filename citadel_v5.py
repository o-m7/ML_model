"""
═══════════════════════════════════════════════════════════════════════════════
CITADEL ML TRADING SYSTEM V5.3 - INSTITUTIONAL GRADE
═══════════════════════════════════════════════════════════════════════════════

CRITICAL FEATURES:
━━━━━━━━━━━━━━━━━━
✅ Centralized equity/DD calculation (realistic, never >100%)
✅ Long AND short trade labeling
✅ Grid search for TP/SL/time-barrier optimization
✅ Raw edge analysis BEFORE any ML training
✅ Strict model-level gates for saving
✅ HTF features hard-filtered out
✅ No saving of losing strategies

Usage:
    python citadel_v5.py --symbol XAUUSD --timeframe 15T
    python citadel_v5.py --symbol XAUUSD --all-timeframes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
import argparse
import json
import joblib
import logging
from itertools import product

import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

from citadel_features import (
    SHARED_FEATURES,
    build_feature_cols_for_strategy,
    StrategyFeatures,
)

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("CitadelML")
    logger.setLevel(getattr(logging, level))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logging()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

# HTF prefixes to filter out
HTF_PREFIXES = ("1H_", "4H_", "D1_", "W1_", "D_", "HTF_", "1h_", "4h_", "d1_", "w1_", "d_", "htf_")

# Spread costs per timeframe (in R-units, based on typical gold spread)
SPREAD_R = {'5T': 0.15, '15T': 0.12, '30T': 0.10, '1H': 0.08}

# Risk per trade for equity calculation
RISK_PER_TRADE = 0.01

# ─────────────────────────────────────────────────────────────────────────────
# RAW EDGE THRESHOLDS (must pass BEFORE any ML training)
# ─────────────────────────────────────────────────────────────────────────────
MIN_RAW_PF = 1.05
MIN_RAW_AVG_R = 0.01
MAX_RAW_DD = 50.0
MIN_RAW_TRADES = 500

# ─────────────────────────────────────────────────────────────────────────────
# MODEL-LEVEL THRESHOLDS (must pass to SAVE model)
# ─────────────────────────────────────────────────────────────────────────────
MIN_MODEL_PF = 1.15
MIN_MODEL_AVG_R = 0.02
MAX_MODEL_DD = 35.0
MIN_MODEL_SHARPE = 0.15
MIN_MODEL_TRADES = 200

# ─────────────────────────────────────────────────────────────────────────────
# GRID SEARCH PARAMETERS BY TIMEFRAME
# ─────────────────────────────────────────────────────────────────────────────
GRID_PARAMS = {
    '5T': {
        'tp_list': [0.6, 0.8, 1.0, 1.2, 1.5],
        'sl_list': [0.6, 0.8, 1.0, 1.2],
        'tb_list': [12, 24, 36, 48],  # 1-4 hours
    },
    '15T': {
        'tp_list': [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
        'sl_list': [0.6, 0.8, 1.0, 1.2],
        'tb_list': [8, 16, 24, 32, 48],  # 2-12 hours
    },
    '30T': {
        'tp_list': [0.8, 1.0, 1.2, 1.5, 2.0],
        'sl_list': [0.6, 0.8, 1.0, 1.2],
        'tb_list': [8, 16, 24, 32],  # 4-16 hours
    },
    '1H': {
        'tp_list': [1.0, 1.5, 2.0, 2.5],
        'sl_list': [0.8, 1.0, 1.2],
        'tb_list': [8, 12, 16, 24],  # 8-24 hours
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# CENTRALIZED EQUITY & DRAWDOWN CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_equity_and_drawdown(
    r_multiples: np.ndarray, 
    risk_per_trade: float = RISK_PER_TRADE
) -> Tuple[float, float, np.ndarray]:
    """
    Compute equity curve and drawdown from R-multiples.
    
    Args:
        r_multiples: Array of R-multiples per trade (AFTER costs)
        risk_per_trade: Fraction of equity risked per trade (e.g. 0.01 = 1%)
    
    Returns:
        total_return_pct: Total return as percentage
        max_dd_pct: Maximum drawdown as percentage (0-100, never exceeds 100)
        equity: Equity curve array starting at 1.0
    """
    if len(r_multiples) == 0:
        return 0.0, 0.0, np.array([1.0])
    
    equity = np.zeros(len(r_multiples) + 1, dtype=np.float64)
    equity[0] = 1.0
    
    for i, r in enumerate(r_multiples):
        pnl_fraction = r * risk_per_trade
        # Clamp to prevent going below zero (can't lose more than 100%)
        pnl_fraction = max(pnl_fraction, -0.99)
        equity[i + 1] = equity[i] * (1.0 + pnl_fraction)
        # Floor at small positive to prevent numerical issues
        equity[i + 1] = max(equity[i + 1], 1e-10)
    
    # Compute drawdown
    peaks = np.maximum.accumulate(equity)
    dd_abs = peaks - equity
    dd_pct = np.where(peaks > 0, dd_abs / peaks, 0)
    max_dd_pct = float(dd_pct.max() * 100.0)
    
    # Cap at 100% (can't lose more than everything)
    max_dd_pct = min(max_dd_pct, 100.0)
    
    total_return_pct = float((equity[-1] / equity[0] - 1.0) * 100.0)
    
    return total_return_pct, max_dd_pct, equity


def compute_full_metrics(r_multiples: np.ndarray) -> Dict:
    """Compute all performance metrics from R-multiples (AFTER costs)."""
    if len(r_multiples) == 0:
        return {
            'n_trades': 0, 'n_wins': 0, 'n_losses': 0,
            'wr': 0, 'pf': 0, 'avg_r': 0, 'med_r': 0,
            'avg_win': 0, 'avg_loss': 0,
            'total_return_pct': 0, 'max_dd_pct': 100,
            'sharpe': 0, 'has_edge': False
        }
    
    wins = r_multiples[r_multiples > 0]
    losses = r_multiples[r_multiples < 0]
    
    n_wins = len(wins)
    n_losses = len(losses)
    n_trades = len(r_multiples)
    
    wr = n_wins / n_trades if n_trades > 0 else 0
    
    gross_profit = wins.sum() if n_wins > 0 else 0
    gross_loss = abs(losses.sum()) if n_losses > 0 else 0
    pf = gross_profit / (gross_loss + 1e-10)
    
    avg_r = r_multiples.mean()
    med_r = np.median(r_multiples)
    avg_win = wins.mean() if n_wins > 0 else 0
    avg_loss = losses.mean() if n_losses > 0 else 0
    
    # Sharpe ratio (per-trade)
    if r_multiples.std() > 0:
        sharpe = avg_r / r_multiples.std()
    else:
        sharpe = 0
    
    # Equity and DD
    total_return_pct, max_dd_pct, _ = compute_equity_and_drawdown(r_multiples)
    
    return {
        'n_trades': n_trades,
        'n_wins': n_wins,
        'n_losses': n_losses,
        'wr': wr,
        'pf': pf,
        'avg_r': avg_r,
        'med_r': med_r,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_return_pct': total_return_pct,
        'max_dd_pct': max_dd_pct,
        'sharpe': sharpe,
    }


# ═══════════════════════════════════════════════════════════════════════════
# RAW EDGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_raw_edge(
    r_post: np.ndarray, 
    strategy_name: str, 
    timeframe: str,
    symbol: str = "XAUUSD"
) -> Dict:
    """
    Analyze if there's a raw statistical edge BEFORE any ML.
    
    Args:
        r_post: R-multiples AFTER spread/costs
        strategy_name: Name of strategy
        timeframe: Timeframe string
        symbol: Trading symbol
    
    Returns:
        Dict with all stats and 'has_raw_edge' boolean
    """
    metrics = compute_full_metrics(r_post)
    
    # Check edge conditions
    has_edge = (
        metrics['n_trades'] >= MIN_RAW_TRADES and
        metrics['pf'] >= MIN_RAW_PF and
        metrics['avg_r'] >= MIN_RAW_AVG_R and
        metrics['max_dd_pct'] <= MAX_RAW_DD
    )
    
    metrics['has_raw_edge'] = has_edge
    
    # Log results
    edge_str = "✅ RAW EDGE" if has_edge else "❌ NO RAW EDGE"
    
    logger.info(f"\n[{edge_str}] {symbol} {timeframe} {strategy_name}:")
    logger.info(f"   Trades={metrics['n_trades']}, WR={metrics['wr']:.1%}, "
               f"PF={metrics['pf']:.2f}, AvgR={metrics['avg_r']:+.3f}R")
    logger.info(f"   AvgWin={metrics['avg_win']:.2f}R, AvgLoss={metrics['avg_loss']:.2f}R")
    logger.info(f"   TotalRet={metrics['total_return_pct']:+.1f}%, MaxDD={metrics['max_dd_pct']:.1f}%")
    
    if not has_edge:
        reasons = []
        if metrics['n_trades'] < MIN_RAW_TRADES:
            reasons.append(f"Trades {metrics['n_trades']} < {MIN_RAW_TRADES}")
        if metrics['pf'] < MIN_RAW_PF:
            reasons.append(f"PF {metrics['pf']:.2f} < {MIN_RAW_PF}")
        if metrics['avg_r'] < MIN_RAW_AVG_R:
            reasons.append(f"AvgR {metrics['avg_r']:.3f} < {MIN_RAW_AVG_R}")
        if metrics['max_dd_pct'] > MAX_RAW_DD:
            reasons.append(f"DD {metrics['max_dd_pct']:.1f}% > {MAX_RAW_DD}%")
        logger.info(f"   Fail reasons: {', '.join(reasons)}")
    
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# BIDIRECTIONAL TRIPLE-BARRIER LABELER (LONG + SHORT)
# ═══════════════════════════════════════════════════════════════════════════

class BidirectionalLabeler:
    """
    Triple-barrier labeler that computes BOTH long and short trade outcomes.
    Returns R-multiples for both directions.
    """
    
    @staticmethod
    def label(
        df: pd.DataFrame, 
        tp_mult: float, 
        sl_mult: float, 
        max_hold: int,
        spread_r: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Label both long and short trades.
        
        Returns:
            r_long_post: R-multiples for long trades (AFTER spread)
            r_short_post: R-multiples for short trades (AFTER spread)
            labels_long: Binary labels for long trades (1=win, 0=loss)
            labels_short: Binary labels for short trades (1=win, 0=loss)
        """
        n = len(df)
        
        r_long_pre = np.full(n, np.nan, dtype=np.float32)
        r_short_pre = np.full(n, np.nan, dtype=np.float32)
        labels_long = np.full(n, -1, dtype=np.int8)
        labels_short = np.full(n, -1, dtype=np.int8)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        atr = df['atr'].values
        
        for i in range(n - max_hold - 1):
            if atr[i] <= 0 or np.isnan(atr[i]):
                continue
            
            entry = close[i]
            unit_risk = atr[i]
            
            # ─────────────────────────────────────────────────────────
            # LONG TRADE
            # ─────────────────────────────────────────────────────────
            tp_long = entry + (tp_mult * unit_risk)
            sl_long = entry - (sl_mult * unit_risk)
            
            fut_high = high[i+1:i+1+max_hold]
            fut_low = low[i+1:i+1+max_hold]
            fut_close = close[i+1:i+1+max_hold]
            
            # Find first TP/SL hit
            tp_long_hits = np.where(fut_high >= tp_long)[0]
            sl_long_hits = np.where(fut_low <= sl_long)[0]
            
            tp_long_bar = tp_long_hits[0] if len(tp_long_hits) > 0 else max_hold + 1
            sl_long_bar = sl_long_hits[0] if len(sl_long_hits) > 0 else max_hold + 1
            
            if tp_long_bar < sl_long_bar and tp_long_bar < max_hold:
                # TP hit first
                r_long_pre[i] = tp_mult
                labels_long[i] = 1
            elif sl_long_bar < tp_long_bar and sl_long_bar < max_hold:
                # SL hit first
                r_long_pre[i] = -sl_mult
                labels_long[i] = 0
            elif len(fut_close) > 0:
                # Time barrier - exit at last close
                pnl = (fut_close[-1] - entry) / unit_risk
                pnl = np.clip(pnl, -sl_mult * 1.5, tp_mult)  # Cap at 1.5x SL
                r_long_pre[i] = pnl
                labels_long[i] = 1 if pnl > 0 else 0
            
            # ─────────────────────────────────────────────────────────
            # SHORT TRADE
            # ─────────────────────────────────────────────────────────
            tp_short = entry - (tp_mult * unit_risk)
            sl_short = entry + (sl_mult * unit_risk)
            
            # For shorts: TP hit when LOW goes below target, SL hit when HIGH goes above
            tp_short_hits = np.where(fut_low <= tp_short)[0]
            sl_short_hits = np.where(fut_high >= sl_short)[0]
            
            tp_short_bar = tp_short_hits[0] if len(tp_short_hits) > 0 else max_hold + 1
            sl_short_bar = sl_short_hits[0] if len(sl_short_hits) > 0 else max_hold + 1
            
            if tp_short_bar < sl_short_bar and tp_short_bar < max_hold:
                # TP hit first (price went down to target)
                r_short_pre[i] = tp_mult
                labels_short[i] = 1
            elif sl_short_bar < tp_short_bar and sl_short_bar < max_hold:
                # SL hit first (price went up to stop)
                r_short_pre[i] = -sl_mult
                labels_short[i] = 0
            elif len(fut_close) > 0:
                # Time barrier - exit at last close
                pnl = (entry - fut_close[-1]) / unit_risk  # Inverted for short
                pnl = np.clip(pnl, -sl_mult * 1.5, tp_mult)
                r_short_pre[i] = pnl
                labels_short[i] = 1 if pnl > 0 else 0
        
        # Apply spread costs
        r_long_post = r_long_pre - spread_r
        r_short_post = r_short_pre - spread_r
        
        return r_long_post, r_short_post, labels_long, labels_short


# ═══════════════════════════════════════════════════════════════════════════
# GRID SEARCH FOR OPTIMAL TP/SL/TB
# ═══════════════════════════════════════════════════════════════════════════

def grid_search_tp_sl_tb(
    df: pd.DataFrame,
    strategy_name: str,
    timeframe: str,
    strategy_mask: np.ndarray,
    symbol: str = "XAUUSD"
) -> Optional[Tuple[float, float, int, Dict]]:
    """
    Grid search to find optimal TP/SL/time-barrier combination.
    
    Args:
        df: DataFrame with OHLCV + ATR
        strategy_name: Name of strategy
        timeframe: Timeframe string
        strategy_mask: Boolean mask for strategy's trade universe
        symbol: Trading symbol
    
    Returns:
        (best_tp, best_sl, best_tb, best_metrics) or None if no edge found
    """
    params = GRID_PARAMS.get(timeframe, GRID_PARAMS['15T'])
    spread_r = SPREAD_R.get(timeframe, 0.12)
    
    tp_list = params['tp_list']
    sl_list = params['sl_list']
    tb_list = params['tb_list']
    
    # Apply mask to get strategy universe
    df_universe = df[strategy_mask].reset_index(drop=True)
    
    if len(df_universe) < MIN_RAW_TRADES:
        logger.warning(f"   Grid search: Not enough data ({len(df_universe)} < {MIN_RAW_TRADES})")
        return None
    
    logger.info(f"\n🔍 GRID SEARCH: {symbol} {timeframe} {strategy_name}")
    logger.info(f"   Universe: {len(df_universe):,} bars")
    logger.info(f"   TP: {tp_list}, SL: {sl_list}, TB: {tb_list}")
    logger.info(f"   Spread cost: {spread_r}R")
    
    best_result = None
    best_score = -np.inf
    
    results_log = []
    
    for tp, sl, tb in product(tp_list, sl_list, tb_list):
        # Label with current parameters
        r_long, r_short, labels_long, labels_short = BidirectionalLabeler.label(
            df_universe, tp, sl, tb, spread_r
        )
        
        # Combine long and short (taking the better direction for each bar)
        # For now: use symmetric approach - average of both directions
        valid_long = ~np.isnan(r_long)
        valid_short = ~np.isnan(r_short)
        
        # Simple approach: take long trades where r_long > r_short, else short
        r_combined = []
        for i in range(len(r_long)):
            if valid_long[i] and valid_short[i]:
                # Take the better trade
                r_combined.append(max(r_long[i], r_short[i]))
            elif valid_long[i]:
                r_combined.append(r_long[i])
            elif valid_short[i]:
                r_combined.append(r_short[i])
        
        r_combined = np.array(r_combined)
        
        if len(r_combined) < 100:
            continue
        
        metrics = compute_full_metrics(r_combined)
        
        # Score: optimize for PF * AvgR, penalize DD
        if metrics['pf'] >= 1.0 and metrics['avg_r'] > 0:
            score = metrics['pf'] * metrics['avg_r'] * (1 - metrics['max_dd_pct']/100)
        else:
            score = -1
        
        results_log.append({
            'tp': tp, 'sl': sl, 'tb': tb,
            'pf': metrics['pf'], 'avg_r': metrics['avg_r'],
            'trades': metrics['n_trades'], 'dd': metrics['max_dd_pct'],
            'score': score
        })
        
        # Check if this is the best so far
        if score > best_score:
            passes_gates = (
                metrics['n_trades'] >= MIN_RAW_TRADES and
                metrics['pf'] >= MIN_RAW_PF and
                metrics['avg_r'] >= MIN_RAW_AVG_R and
                metrics['max_dd_pct'] <= MAX_RAW_DD
            )
            
            if passes_gates:
                best_score = score
                best_result = (tp, sl, tb, metrics)
    
    # Log top 5 results
    results_log.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"\n   Top 5 parameter combinations:")
    for i, r in enumerate(results_log[:5]):
        status = "✓" if r['pf'] >= MIN_RAW_PF and r['avg_r'] >= MIN_RAW_AVG_R else "✗"
        logger.info(f"   {status} TP={r['tp']}, SL={r['sl']}, TB={r['tb']} | "
                   f"PF={r['pf']:.2f}, AvgR={r['avg_r']:+.3f}R, "
                   f"Trades={r['trades']}, DD={r['dd']:.1f}%")
    
    if best_result is None:
        logger.warning(f"\n❌ [GRID SEARCH FAIL] {symbol} {timeframe} {strategy_name}:")
        logger.warning(f"   No TP/SL/TB combination yields positive edge after spread.")
        return None
    
    tp, sl, tb, metrics = best_result
    logger.info(f"\n✅ [GRID SEARCH BEST] {symbol} {timeframe} {strategy_name}:")
    logger.info(f"   TP={tp}, SL={sl}, TB={tb} | PF={metrics['pf']:.2f}, "
               f"AvgR={metrics['avg_r']:+.3f}R, Trades={metrics['n_trades']}, DD={metrics['max_dd_pct']:.1f}%")
    
    return best_result


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyConfig:
    name: str
    required_features: List[str]
    session_filter: Optional[List[str]]
    description: str
    prediction_threshold: float = 0.50


STRATEGIES = {
    '5T': [
        StrategyConfig(
            name='mean_reversion_rsi',
            required_features=['rsi', 'atr', 'ema_20', 'body_pct'],
            session_filter=['london', 'ny'],
            description='RSI mean reversion with ATR confirmation',
        ),
        StrategyConfig(
            name='momentum_breakout',
            required_features=['higher_high', 'lower_low', 'volume', 'atr', 'momentum_5'],
            session_filter=['london', 'ny'],
            description='Simple breakout on higher highs/lows with volume',
        ),
        StrategyConfig(
            name='ema_crossover',
            required_features=['ema_10', 'ema_20', 'atr', 'volume', 'close'],
            session_filter=['london', 'ny'],
            description='EMA crossover with momentum filter',
        ),
    ],
    '15T': [
        StrategyConfig(
            name='vwap_deviation_reversion',
            required_features=[
                'vwap_deviation', 'vwap_deviation_20', 'vwap_deviation_50',
                'vwap_zscore', 'vwap_band_position',
                'vwap_oversold', 'vwap_overbought', 'vwap_extreme',
                'mean_reversion_signal', 'mr_signal_long', 'mr_signal_short',
                'rsi', 'rsi_divergence',
                'delta_proxy', 'wick_rejection', 'body_pct',
            ],
            session_filter=['london', 'ny'],
            description='Mean revert to VWAP from extremes',
        ),
        StrategyConfig(
            name='session_volatility_burst',
            required_features=[
                'session_open_flag', 'london_open', 'ny_open',
                'volatility_expansion', 'vol_ratio', 'atr_expansion',
                'opening_range', 'range_breakout',
                'breakout_direction', 'volume_spike',
                'momentum_5', 'price_velocity',
            ],
            session_filter=['london', 'ny'],
            description='Trade session open volatility bursts',
        ),
        StrategyConfig(
            name='breakout_failure_fade',
            required_features=[
                'failed_breakout', 'failed_breakout_high', 'failed_breakout_low',
                'trap_signal', 'bull_trap', 'bear_trap',
                'sweep_bullish', 'sweep_bearish',
                'wick_rejection', 'swing_high_20', 'swing_low_20',
                'delta_proxy', 'absorption',
            ],
            session_filter=['london', 'ny'],
            description='Fade failed breakouts and fakeouts',
        ),
    ],
    '30T': [
        StrategyConfig(
            name='vwap_reversion',
            required_features=[
                'vwap_deviation', 'vwap_zscore', 'vwap_band_position',
                'vwap_oversold', 'vwap_overbought', 'vwap_extreme',
                'mean_reversion_signal',
                'rsi', 'rsi_divergence', 'macd_hist',
            ],
            session_filter=['london', 'ny'],
            description='Institutional VWAP mean reversion',
        ),
    ],
    '1H': [
        StrategyConfig(
            name='trend_continuation',
            required_features=[
                'regime_trending', 'regime_trend_up', 'regime_trend_down',
                'trend_strength', 'di_cross',
                'momentum_5', 'momentum_10',
                'pullback_entry', 'pullback_depth',
                'ema_slope_20',
            ],
            session_filter=None,
            description='Trend continuation with pullback entries',
        ),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY-SPECIFIC TRADE UNIVERSE MASKS
# ═══════════════════════════════════════════════════════════════════════════

class StrategyMasks:
    """Create strategy-specific trade universe masks."""
    
    @staticmethod
    def get_mask(df: pd.DataFrame, strategy_name: str, session_filter: Optional[List[str]]) -> np.ndarray:
        """Get boolean mask for rows where strategy should be considered."""
        n = len(df)
        mask = np.ones(n, dtype=bool)
        
        # ─────────────────────────────────────────────────────────────
        # VWAP REVERSION - only at extremes
        # ─────────────────────────────────────────────────────────────
        if 'vwap' in strategy_name.lower() and 'reversion' in strategy_name.lower():
            extreme_mask = np.zeros(n, dtype=bool)
            
            if 'vwap_zscore' in df.columns:
                extreme_mask |= np.abs(df['vwap_zscore'].values) > 1.5
            if 'vwap_oversold' in df.columns:
                extreme_mask |= df['vwap_oversold'].values == 1
            if 'vwap_overbought' in df.columns:
                extreme_mask |= df['vwap_overbought'].values == 1
            if 'vwap_extreme' in df.columns:
                extreme_mask |= df['vwap_extreme'].values == 1
            
            mask &= extreme_mask
        
        # ─────────────────────────────────────────────────────────────
        # SESSION BURST - near opens or vol expansion
        # ─────────────────────────────────────────────────────────────
        elif 'session' in strategy_name.lower() and 'burst' in strategy_name.lower():
            burst_mask = np.zeros(n, dtype=bool)
            
            if 'session_open_flag' in df.columns:
                burst_mask |= df['session_open_flag'].values == 1
            if 'london_open' in df.columns:
                burst_mask |= df['london_open'].values == 1
            if 'ny_open' in df.columns:
                burst_mask |= df['ny_open'].values == 1
            if 'volatility_expansion' in df.columns:
                burst_mask |= df['volatility_expansion'].values == 1
            if 'atr_expansion' in df.columns:
                burst_mask |= df['atr_expansion'].values == 1
            
            # First 2 hours of sessions
            if 'hour' in df.columns:
                hour = df['hour'].values
                burst_mask |= ((hour >= 8) & (hour < 10)) | ((hour >= 13) & (hour < 15))
            
            mask &= burst_mask
        
        # ─────────────────────────────────────────────────────────────
        # BREAKOUT FAILURE - around failures/traps/sweeps
        # ─────────────────────────────────────────────────────────────
        elif 'breakout' in strategy_name.lower() and 'failure' in strategy_name.lower():
            failure_mask = np.zeros(n, dtype=bool)
            
            for col in ['failed_breakout', 'trap_signal', 'bull_trap', 'bear_trap',
                       'sweep_bullish', 'sweep_bearish', 'wick_rejection']:
                if col in df.columns:
                    failure_mask |= df[col].values == 1
            
            mask &= failure_mask
        
        # ─────────────────────────────────────────────────────────────
        # LIQUIDITY SWEEP - after sweeps
        # ─────────────────────────────────────────────────────────────
        elif 'liquidity' in strategy_name.lower() and 'sweep' in strategy_name.lower():
            sweep_mask = np.zeros(n, dtype=bool)
            
            for col in ['sweep_bullish', 'sweep_bearish', 'quality_sweep_long', 'quality_sweep_short']:
                if col in df.columns:
                    sweep_mask |= df[col].values == 1
            
            mask &= sweep_mask
        
        # ─────────────────────────────────────────────────────────────
        # RSI MEAN REVERSION - oversold/overbought
        # ─────────────────────────────────────────────────────────────
        if 'mean_reversion' in strategy_name.lower() and 'rsi' in strategy_name.lower():
            rsi_mask = np.zeros(n, dtype=bool)
            
            if 'rsi' in df.columns:
                rsi = df['rsi'].values
                rsi_mask |= (rsi < 30) | (rsi > 70)  # Oversold/overbought
            
            mask &= rsi_mask
        
        # ─────────────────────────────────────────────────────────────
        # MOMENTUM BREAKOUT - higher highs or higher lows
        # ─────────────────────────────────────────────────────────────
        elif 'momentum' in strategy_name.lower() and 'breakout' in strategy_name.lower():
            breakout_mask = np.zeros(n, dtype=bool)
            
            if 'higher_high' in df.columns:
                breakout_mask |= df['higher_high'].values == 1
            if 'lower_low' in df.columns:
                breakout_mask |= df['lower_low'].values == 1
            if 'momentum_5' in df.columns:
                breakout_mask |= np.abs(df['momentum_5'].values) > 0.5
            
            mask &= breakout_mask
        
        # ─────────────────────────────────────────────────────────────
        # EMA CROSSOVER - EMA alignment
        # ─────────────────────────────────────────────────────────────
        elif 'ema' in strategy_name.lower() and 'crossover' in strategy_name.lower():
            ema_mask = np.zeros(n, dtype=bool)
            
            if 'ema_cross_10_20' in df.columns:
                ema_mask = df['ema_cross_10_20'].values == 1
            
            mask &= ema_mask
        
        # ─────────────────────────────────────────────────────────────
        # TREND CONTINUATION - in trending regime
        # ─────────────────────────────────────────────────────────────
        elif 'trend' in strategy_name.lower() and 'continuation' in strategy_name.lower():
            trend_mask = np.zeros(n, dtype=bool)
            
            if 'regime_trending' in df.columns:
                trend_mask |= df['regime_trending'].values == 1
            if 'pullback_entry' in df.columns:
                trend_mask |= df['pullback_entry'].values == 1
            if 'momentum_quality' in df.columns:
                trend_mask |= df['momentum_quality'].values == 1
            
            mask &= trend_mask
        
        # ─────────────────────────────────────────────────────────────
        # SESSION FILTER
        # ─────────────────────────────────────────────────────────────
        if session_filter:
            session_mask = np.zeros(n, dtype=bool)
            
            session_map = {
                'london': 'session_london',
                'ny': 'session_ny',
                'overlap': 'session_overlap',
                'asian': 'session_asian',
            }
            
            for session in session_filter:
                col = session_map.get(session, session)
                if col in df.columns:
                    session_mask |= df[col].values == 1
            
            if session_mask.any():
                mask &= session_mask
        
        return mask


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    FEATURE_STORE: Path = Path("feature_store")
    MODELS_DIR: Path = Path("production_models")
    TRAIN_RATIO: float = 0.60
    VAL_RATIO: float = 0.20

CONFIG = Config()


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Load data and HARD FILTER HTF features."""
    
    @staticmethod
    def load(symbol: str, timeframe: str) -> pd.DataFrame:
        path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")
        
        df = pd.read_parquet(path)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # HARD DROP ALL HTF COLUMNS
        htf_cols = [c for c in df.columns if any(c.startswith(p) for p in HTF_PREFIXES)]
        if htf_cols:
            logger.warning(f"⚠️ DROPPING {len(htf_cols)} HTF columns: {htf_cols[:5]}...")
            df = df.drop(columns=htf_cols)
        
        # Ensure ATR exists
        if 'atr' not in df.columns:
            prev_close = df['close'].shift(1)
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs()
            ], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
        
        logger.info(f"Loaded {len(df):,} rows from {path}")
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY TRAINER
# ═══════════════════════════════════════════════════════════════════════════

class StrategyTrainer:
    """Train strategy-specific models with rigorous validation."""
    
    @staticmethod
    def train_strategy(
        df: pd.DataFrame, 
        strategy: StrategyConfig, 
        timeframe: str,
        symbol: str = "XAUUSD"
    ) -> Optional[Dict]:
        """
        Full pipeline for one strategy:
        1. Get strategy mask
        2. Grid search for optimal TP/SL/TB
        3. Analyze raw edge
        4. Train ML model
        5. Evaluate and gate
        """
        logger.info(f"\n{'═'*70}")
        logger.info(f"STRATEGY: {strategy.name}")
        logger.info(f"Description: {strategy.description}")
        logger.info(f"{'═'*70}")
        
        spread_r = SPREAD_R.get(timeframe, 0.12)
        
        # ─────────────────────────────────────────────────────────────
        # STEP 1: Get strategy-specific mask
        # ─────────────────────────────────────────────────────────────
        strategy_mask = StrategyMasks.get_mask(df, strategy.name, strategy.session_filter)
        n_universe = strategy_mask.sum()
        
        logger.info(f"Strategy universe: {n_universe:,} / {len(df):,} bars ({100*n_universe/len(df):.1f}%)")
        
        if n_universe < MIN_RAW_TRADES:
            logger.warning(f"❌ Not enough bars in universe: {n_universe} < {MIN_RAW_TRADES}")
            return None
        
        # ─────────────────────────────────────────────────────────────
        # STEP 2: Grid search for optimal TP/SL/TB
        # IMPORTANT: Run grid search ONLY on the chronological training
        # partition to avoid look-ahead/data leakage. Previously the grid
        # search ran on the full universe which leaks future information
        # into parameter selection.
        # ─────────────────────────────────────────────────────────────
        df_universe = df[strategy_mask].reset_index(drop=True)

        n = len(df_universe)
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))

        if train_end < MIN_RAW_TRADES:
            logger.warning(f"   Grid search: Not enough training data ({train_end} < {MIN_RAW_TRADES})")
            return None

        # Run grid search on training partition only
        df_universe_train = df_universe.iloc[:train_end].reset_index(drop=True)
        grid_result = grid_search_tp_sl_tb(
            df_universe_train, strategy.name, timeframe, np.ones(len(df_universe_train), dtype=bool), symbol
        )

        if grid_result is None:
            return None

        best_tp, best_sl, best_tb, grid_metrics = grid_result
        
        # ─────────────────────────────────────────────────────────────
        # STEP 3: Re-label with best params and analyze raw edge
        # ─────────────────────────────────────────────────────────────
        # Re-label the full universe with the chosen params (params were
        # chosen using the training partition only). We will compute raw
        # metrics on the training-labeled subset to avoid leakage.
        r_long, r_short, labels_long, labels_short = BidirectionalLabeler.label(
            df_universe, best_tp, best_sl, best_tb, spread_r
        )
        
        # Combine: for each bar, take the better direction
        valid_long = ~np.isnan(r_long)
        valid_short = ~np.isnan(r_short)
        
        r_best = []
        direction = []  # 1 = long, -1 = short
        indices = []
        
        for i in range(len(r_long)):
            if valid_long[i] and valid_short[i]:
                if r_long[i] >= r_short[i]:
                    r_best.append(r_long[i])
                    direction.append(1)
                else:
                    r_best.append(r_short[i])
                    direction.append(-1)
                indices.append(i)
            elif valid_long[i]:
                r_best.append(r_long[i])
                direction.append(1)
                indices.append(i)
            elif valid_short[i]:
                r_best.append(r_short[i])
                direction.append(-1)
                indices.append(i)
        
        r_best = np.array(r_best)
        direction = np.array(direction)
        indices = np.array(indices)
        
        # Analyze raw edge on the TRAINING portion only (no future info).
        # This prevents the pipeline from declaring an edge because of
        # parameters that were tuned on future data.
        train_mask_on_indices = indices < train_end
        if train_mask_on_indices.sum() == 0:
            logger.warning("❌ No labeled trades in training partition after relabeling")
            return None

        r_best_train = r_best[train_mask_on_indices]
        raw_metrics = analyze_raw_edge(r_best_train, strategy.name, timeframe, symbol)

        if not raw_metrics['has_raw_edge']:
            return None
        
        # ─────────────────────────────────────────────────────────────
        # STEP 4: Prepare features and train ML model
        # ─────────────────────────────────────────────────────────────
        df_labeled = df_universe.iloc[indices].reset_index(drop=True)
        y = (direction == 1).astype(int)  # 1 = long preferred, 0 = short preferred
        r = r_best
        
        # Build features (with HTF filtering)
        feature_cols = build_feature_cols_for_strategy(strategy.required_features, df_labeled)
        
        # HARD FILTER: Remove any HTF features that might have slipped through
        feature_cols = [c for c in feature_cols if not any(c.startswith(p) for p in HTF_PREFIXES)]
        
        if len(feature_cols) < 5:
            logger.warning(f"❌ Too few features: {len(feature_cols)}")
            return None
        
        logger.info(f"\n🔧 ML Training: {len(feature_cols)} features, {len(y)} samples")
        logger.info(f"   HTF features filtered: ✓")
        
        X = df_labeled[feature_cols].values
        
        n = len(X)
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        
        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
        r_test = r[val_end:]
        dir_test = direction[val_end:]
        
        if len(X_test) < MIN_MODEL_TRADES:
            logger.warning(f"❌ Not enough test samples: {len(X_test)} < {MIN_MODEL_TRADES}")
            return None
        
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        
        classes = np.unique(y_train)
        if len(classes) < 2:
            logger.warning("❌ Single class in training data")
            return None
        
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        sample_weights = np.array([weights[int(yi)] for yi in y_train])
        
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbose=-1
        )
        
        model.fit(
            X_train_s, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val_s, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # ─────────────────────────────────────────────────────────────
        # STEP 5: Evaluate on test set
        # ─────────────────────────────────────────────────────────────
        y_proba = model.predict_proba(X_test_s)[:, 1]
        threshold = strategy.prediction_threshold
        
        # Trade selection: long if p > 0.5 + margin, short if p < 0.5 - margin
        margin = 0.1
        trade_long = y_proba > (0.5 + margin)
        trade_short = y_proba < (0.5 - margin)
        trade_mask = trade_long | trade_short
        
        if trade_mask.sum() < MIN_MODEL_TRADES:
            logger.warning(f"❌ ML predicted too few trades: {trade_mask.sum()} < {MIN_MODEL_TRADES}")
            return None
        
        # Calculate R for selected trades
        r_selected = []
        for i in range(len(r_test)):
            if trade_long[i]:
                # Model says go long - use long R
                r_selected.append(r_test[i] if dir_test[i] == 1 else -r_test[i])
            elif trade_short[i]:
                # Model says go short - use short R
                r_selected.append(r_test[i] if dir_test[i] == -1 else -r_test[i])
        
        r_selected = np.array(r_selected)
        ml_metrics = compute_full_metrics(r_selected)
        
        logger.info(f"\n📈 POST-ML TEST METRICS ({ml_metrics['n_trades']} trades):")
        logger.info(f"   WR={ml_metrics['wr']:.1%}, PF={ml_metrics['pf']:.2f}, "
                   f"AvgR={ml_metrics['avg_r']:+.3f}R, Sharpe={ml_metrics['sharpe']:.2f}")
        logger.info(f"   TotalRet={ml_metrics['total_return_pct']:+.1f}%, MaxDD={ml_metrics['max_dd_pct']:.1f}%")
        
        # Compare to raw
        logger.info(f"   Δ vs Raw: PF {ml_metrics['pf'] - raw_metrics['pf']:+.2f}, "
                   f"AvgR {ml_metrics['avg_r'] - raw_metrics['avg_r']:+.3f}")
        
        # ─────────────────────────────────────────────────────────────
        # STEP 6: Model-level gates
        # ─────────────────────────────────────────────────────────────
        save_worthy = True
        fail_reasons = []
        
        if ml_metrics['pf'] < MIN_MODEL_PF:
            fail_reasons.append(f"PF {ml_metrics['pf']:.2f} < {MIN_MODEL_PF}")
            save_worthy = False
        
        if ml_metrics['avg_r'] < MIN_MODEL_AVG_R:
            fail_reasons.append(f"AvgR {ml_metrics['avg_r']:.3f} < {MIN_MODEL_AVG_R}")
            save_worthy = False
        
        if ml_metrics['max_dd_pct'] > MAX_MODEL_DD:
            fail_reasons.append(f"DD {ml_metrics['max_dd_pct']:.1f}% > {MAX_MODEL_DD}%")
            save_worthy = False
        
        if ml_metrics['sharpe'] < MIN_MODEL_SHARPE:
            fail_reasons.append(f"Sharpe {ml_metrics['sharpe']:.2f} < {MIN_MODEL_SHARPE}")
            save_worthy = False
        
        if ml_metrics['n_trades'] < MIN_MODEL_TRADES:
            fail_reasons.append(f"Trades {ml_metrics['n_trades']} < {MIN_MODEL_TRADES}")
            save_worthy = False
        
        if not save_worthy:
            logger.warning(f"\n⚠️ [MODEL NOT SAVED] {symbol} {timeframe} {strategy.name}:")
            logger.warning(f"   Fail reasons: {', '.join(fail_reasons)}")
        else:
            logger.info(f"\n✅ [MODEL PASSES GATES] {symbol} {timeframe} {strategy.name}")
        
        # Log top features
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:5]
        logger.info("\nTop features: " + ", ".join(
            [f"{feature_cols[i]}:{imp[i]:.0f}" for i in top_idx]
        ))
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'save_worthy': save_worthy,
            'best_params': {'tp': best_tp, 'sl': best_sl, 'tb': best_tb},
            'strategy_config': {
                'name': strategy.name,
                'tp_mult': best_tp,
                'sl_mult': best_sl,
                'time_barrier': best_tb,
                'session_filter': strategy.session_filter,
                'prediction_threshold': threshold,
            },
            'raw_metrics': raw_metrics,
            'ml_metrics': ml_metrics,
        }


# ═══════════════════════════════════════════════════════════════════════════
# MODEL SAVER
# ═══════════════════════════════════════════════════════════════════════════

class ModelSaver:
    """Save only models that pass all gates."""
    
    @staticmethod
    def save(symbol: str, timeframe: str, results: List[Dict]) -> int:
        save_dir = CONFIG.MODELS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        
        for result in results:
            if result is None:
                continue
            
            if not result.get('save_worthy', False):
                continue
            
            strategy_name = result['strategy_config']['name']
            model_path = save_dir / f"{symbol}_{timeframe}_{strategy_name}.pkl"
            
            joblib.dump({
                'model': result['model'],
                'scaler': result['scaler'],
                'feature_cols': result['feature_cols'],
                'timeframe': timeframe,
                'model_name': strategy_name,
                'threshold': result['strategy_config'].get('prediction_threshold', 0.5),
                'symbol': symbol,
                'best_params': result['best_params'],
                'strategy_config': result['strategy_config'],
                'raw_metrics': result['raw_metrics'],
                'ml_metrics': result['ml_metrics'],
                'saved_at': datetime.now().isoformat(),
            }, model_path)
            
            # JSON meta
            meta = {
                'symbol': symbol,
                'timeframe': timeframe,
                'model_name': strategy_name,
                'best_params': result['best_params'],
                'feature_count': len(result['feature_cols']),
                'feature_cols': result['feature_cols'],
                'raw_metrics': result['raw_metrics'],
                'ml_metrics': result['ml_metrics'],
                'saved_at': datetime.now().isoformat(),
            }
            
            meta_path = save_dir / f"{symbol}_{timeframe}_{strategy_name}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2, default=str)
            
            ml = result['ml_metrics']
            logger.info(f"✅ SAVED: {model_path.name}")
            logger.info(f"   PF={ml['pf']:.2f}, AvgR={ml['avg_r']:+.3f}R, DD={ml['max_dd_pct']:.1f}%")
            saved_count += 1
        
        return saved_count


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class Pipeline:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
    
    def run(self) -> Dict:
        logger.info(f"\n{'═'*70}")
        logger.info(f"CITADEL ML V5.3 - INSTITUTIONAL GRADE")
        logger.info(f"{self.symbol} {self.timeframe}")
        logger.info(f"{'═'*70}")
        logger.info(f"Raw edge gates: PF≥{MIN_RAW_PF}, AvgR≥{MIN_RAW_AVG_R}, DD≤{MAX_RAW_DD}%, Trades≥{MIN_RAW_TRADES}")
        logger.info(f"Model gates: PF≥{MIN_MODEL_PF}, AvgR≥{MIN_MODEL_AVG_R}, DD≤{MAX_MODEL_DD}%, Sharpe≥{MIN_MODEL_SHARPE}")
        
        strategies = STRATEGIES.get(self.timeframe, [])
        if not strategies:
            logger.error(f"No strategies defined for {self.timeframe}")
            return {'viable': False}
        
        logger.info(f"Strategies: {[s.name for s in strategies]}")
        
        # Load data
        df = DataLoader.load(self.symbol, self.timeframe)
        
        # Engineer features
        all_required = set()
        for s in strategies:
            all_required.update(s.required_features)
        
        df = StrategyFeatures.add_all_features(df, self.timeframe, list(all_required))
        logger.info(f"DataFrame after engineering: {df.shape}")
        
        # Train each strategy
        results = []
        for strategy in strategies:
            try:
                result = StrategyTrainer.train_strategy(df, strategy, self.timeframe, self.symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")
                import traceback
                traceback.print_exc()
                results.append(None)
        
        # Save passing models
        n_saved = ModelSaver.save(self.symbol, self.timeframe, results)
        
        # Summary
        logger.info(f"\n{'═'*70}")
        logger.info(f"FINAL SUMMARY - {self.symbol} {self.timeframe}")
        logger.info(f"{'═'*70}")
        
        for strategy, result in zip(strategies, results):
            if result is None:
                logger.info(f"❌ {strategy.name}: No edge or failed")
            elif not result.get('save_worthy'):
                ml = result['ml_metrics']
                logger.info(f"⚠️ {strategy.name}: Edge found but model not saved")
                logger.info(f"   ML: PF={ml['pf']:.2f}, AvgR={ml['avg_r']:+.3f}R, DD={ml['max_dd_pct']:.1f}%")
            else:
                ml = result['ml_metrics']
                bp = result['best_params']
                logger.info(f"✅ {strategy.name}: SAVED")
                logger.info(f"   Params: TP={bp['tp']}, SL={bp['sl']}, TB={bp['tb']}")
                logger.info(f"   ML: PF={ml['pf']:.2f}, AvgR={ml['avg_r']:+.3f}R, DD={ml['max_dd_pct']:.1f}%")
        
        logger.info(f"\n{n_saved} models saved to {CONFIG.MODELS_DIR}/")
        
        if n_saved == 0:
            logger.warning("\n⚠️ NO MODELS PASSED ALL GATES")
            logger.warning("This means no strategy has a tradable edge after costs.")
        
        return {'viable': n_saved > 0, 'n_saved': n_saved}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Citadel ML V5.3 - Institutional Grade')
    parser.add_argument('--symbol', type=str, default='XAUUSD')
    parser.add_argument('--timeframe', type=str)
    parser.add_argument('--all-timeframes', action='store_true')
    
    args = parser.parse_args()
    
    if args.all_timeframes:
        timeframes = ['5T', '15T', '30T', '1H']
    elif args.timeframe:
        timeframes = [args.timeframe]
    else:
        parser.print_help()
        return
    
    total_saved = 0
    
    for tf in timeframes:
        try:
            pipeline = Pipeline(args.symbol, tf)
            result = pipeline.run()
            total_saved += result.get('n_saved', 0)
        except Exception as e:
            logger.error(f"{tf} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'═'*70}")
    print(f"GRAND TOTAL: {total_saved} models saved")
    print(f"{'═'*70}")
    
    if total_saved == 0:
        print("\n⚠️ NO TRADABLE EDGE FOUND IN ANY STRATEGY/TIMEFRAME")
        print("The honest answer: these strategies have no statistical edge after costs.")


if __name__ == '__main__':
    main()