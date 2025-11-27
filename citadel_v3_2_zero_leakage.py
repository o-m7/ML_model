"""
CITADEL V3.2 - ABSOLUTE ZERO LEAKAGE
=====================================

This version:
1. Uses ONLY raw OHLCV (no pre-computed indicators)
2. Computes ALL features with strict causality
3. Adds extensive diagnostics to verify no leakage
4. Simplified direction detection
5. Label quality checks

Run: python citadel_v3_2_zero_leakage.py --symbol XAUUSD --timeframe 5T --diagnose
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import argparse
import json
import joblib

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
DEFAULT_FEATURE_STORE = SCRIPT_DIR / "feature_store"


@dataclass
class SystemConfig:
    FEATURE_STORE: Path = DEFAULT_FEATURE_STORE
    
    MIN_WIN_RATE: float = 0.45
    MAX_WIN_RATE: float = 0.75
    MIN_PROFIT_FACTOR: float = 1.2
    MAX_PROFIT_FACTOR: float = 5.0
    MIN_ACCEPTABLE_PF: float = 1.3
    
    # Transaction costs (REDUCED - your previous costs were too high)
    BASE_SPREAD_R: Dict[str, float] = field(default_factory=lambda: {
        "5T": 0.04,   # ~4 pips on gold = ~0.04R (was 0.10R)
        "15T": 0.03,  # ~3 pips
        "30T": 0.025, # ~2.5 pips
        "1H": 0.02,   # ~2 pips
    })
    SLIPPAGE_R: float = 0.02  # ~2 pips (was 0.05R)
    COMMISSION_R: float = 0.01  # ~1 pip (was 0.02R)
    
    RISK_PER_TRADE_EVAL: float = 0.003
    
    @staticmethod
    def get_tp_multipliers(timeframe: str) -> List[float]:
        return {
            '5T': [1.2, 1.5, 2.0, 2.5, 3.0],  # Wider targets to overcome costs
            '15T': [2.0, 2.5, 3.0, 3.5],
        }.get(timeframe, [2.0, 2.5, 3.0])
    
    SL_MULTIPLIER: float = 1.0
    
    @staticmethod
    def get_time_barriers(timeframe: str) -> List[int]:
        return {
            '5T': [15, 20, 25],
            '15T': [30, 35, 40],
        }.get(timeframe, [20, 30])
    
    @staticmethod
    def get_min_trades_raw(timeframe: str) -> int:
        return {'5T': 1000, '15T': 500}.get(timeframe, 500)
    
    TRAIN_RATIO: float = 0.60
    VAL_RATIO: float = 0.20
    
    LGBM_PARAMS: Dict = field(default_factory=lambda: {
        'n_estimators': 150, 'learning_rate': 0.05, 'num_leaves': 31,
        'max_depth': 5, 'min_child_samples': 100, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_alpha': 0.3, 'reg_lambda': 0.3, 'verbose': -1
    })


CONFIG = SystemConfig()


def get_total_cost_r(timeframe: str) -> float:
    return CONFIG.BASE_SPREAD_R.get(timeframe, 0.05) + CONFIG.SLIPPAGE_R + CONFIG.COMMISSION_R


def compute_equity_and_dd(r_multiples: np.ndarray, risk_per_trade: float = CONFIG.RISK_PER_TRADE_EVAL):
    if len(r_multiples) == 0:
        return np.array([1.0]), 0.0, 0.0
    equity = np.zeros(len(r_multiples) + 1)
    equity[0] = 1.0
    for i, r in enumerate(r_multiples):
        equity[i + 1] = equity[i] * (1.0 + r * risk_per_trade)
    peaks = np.maximum.accumulate(equity)
    dd_pct = ((peaks - equity) / peaks).max() * 100.0
    return equity, dd_pct, 0.0


class RiskMetrics:
    @staticmethod
    def calculate_all_metrics(r_multiples: np.ndarray, risk_per_trade: float = CONFIG.RISK_PER_TRADE_EVAL) -> Dict:
        if len(r_multiples) == 0:
            return {'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                    'sharpe': 0.0, 'max_drawdown_pct': 0.0, 'mean_r': 0.0}
        
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
            'total_trades': len(r_multiples), 'wins': wins, 'losses': len(r_multiples) - wins,
            'win_rate': win_rate, 'profit_factor': pf, 'sharpe': sharpe,
            'max_drawdown_pct': max_dd_pct, 'mean_r': r_multiples.mean()
        }


class DataLoader:
    @staticmethod
    def load_timeframe_data(symbol: str, timeframe: str) -> pd.DataFrame:
        """Load ONLY raw OHLCV data - discard all pre-computed indicators."""
        file_path = CONFIG.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        
        # Handle timestamp
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                if 'index' in df.columns:
                    df = df.rename(columns={'index': 'timestamp'})
        
        if 'timestamp' not in df.columns:
            raise KeyError("No timestamp column found")
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Keep ONLY raw OHLCV - discard everything else
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[required].copy()
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\n✅ Loaded RAW OHLCV: {len(df):,} rows")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Columns: {list(df.columns)}")
        
        return df


class ZeroLeakageFeatures:
    """
    Build ALL features from scratch using ONLY historical data.
    
    Key principle: At bar i, we use data from bars 0 to i-1 ONLY.
    """
    
    @staticmethod
    def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Compute ATR using PREVIOUS bar's close."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.shift(1)  # Shift so bar i sees bar i-1's ATR
    
    @staticmethod
    def compute_ema(close: pd.Series, period: int) -> pd.Series:
        """Compute EMA using only past data."""
        ema = close.ewm(span=period, adjust=False).mean()
        return ema.shift(1)  # Shift so bar i sees bar i-1's EMA
    
    @staticmethod
    def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI using only past data."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.shift(1)  # Shift so bar i sees bar i-1's RSI
    
    @staticmethod
    def engineer_all(df: pd.DataFrame, diagnose: bool = False) -> pd.DataFrame:
        """Build all features with ZERO leakage."""
        print(f"\n🔧 ZERO-LEAKAGE FEATURE ENGINEERING")
        print(f"{'='*80}")
        
        features = df.copy()
        
        # 1. ATR (critical for labeling)
        features['atr'] = ZeroLeakageFeatures.compute_atr(
            features['high'], features['low'], features['close'], 14
        )
        
        # 2. Price-based features (using PREVIOUS bar)
        close_prev = features['close'].shift(1)
        open_prev = features['open'].shift(1)
        high_prev = features['high'].shift(1)
        low_prev = features['low'].shift(1)
        
        # Returns (bar i-1 to i-2)
        features['returns_1'] = close_prev.pct_change(1)
        features['returns_5'] = close_prev.pct_change(5)
        features['returns_10'] = close_prev.pct_change(10)
        
        # Trend indicators
        features['ema_10'] = ZeroLeakageFeatures.compute_ema(features['close'], 10)
        features['ema_20'] = ZeroLeakageFeatures.compute_ema(features['close'], 20)
        features['ema_50'] = ZeroLeakageFeatures.compute_ema(features['close'], 50)
        
        # Trend strength
        features['trend_10_20'] = (features['ema_10'] > features['ema_20']).astype(int) * 2 - 1
        features['trend_20_50'] = (features['ema_20'] > features['ema_50']).astype(int) * 2 - 1
        
        # RSI
        features['rsi'] = ZeroLeakageFeatures.compute_rsi(features['close'], 14)
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
        
        # Volatility (using previous bars)
        features['volatility_10'] = close_prev.rolling(10).std() / (close_prev.rolling(10).mean() + 1e-8)
        features['volatility_20'] = close_prev.rolling(20).std() / (close_prev.rolling(20).mean() + 1e-8)
        
        # Price position in range
        high_20 = high_prev.rolling(20).max()
        low_20 = low_prev.rolling(20).min()
        features['range_position'] = (close_prev - low_20) / (high_20 - low_20 + 1e-8)
        
        # Candle patterns (previous bar)
        features['body_size'] = abs(close_prev - open_prev) / (close_prev + 1e-8)
        features['upper_wick'] = (high_prev - np.maximum(close_prev, open_prev)) / (high_prev - low_prev + 1e-8)
        features['lower_wick'] = (np.minimum(close_prev, open_prev) - low_prev) / (high_prev - low_prev + 1e-8)
        
        # Volume features
        if 'volume' in features.columns:
            vol_prev = features['volume'].shift(1)
            features['volume_ratio'] = vol_prev / (vol_prev.rolling(20).mean() + 1)
        
        # Time features
        features['hour'] = features['timestamp'].dt.hour
        features['day_of_week'] = features['timestamp'].dt.dayofweek
        features['is_london'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
        features['is_newyork'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
        
        # Drop NaNs
        initial_rows = len(features)
        features = features.dropna()
        
        print(f"   ✅ Built {len(features.columns) - len(df.columns)} features")
        print(f"   🧹 Dropped {initial_rows - len(features)} rows with NaNs")
        print(f"   ✓ Final: {len(features):,} rows")
        
        # DIAGNOSTIC: Verify no leakage
        if diagnose:
            print(f"\n🔍 LEAKAGE DIAGNOSTIC")
            print(f"   Checking if features use ONLY past data...")
            
            # Test: Compute EMA manually for bar 100, verify it matches
            test_idx = 100
            if test_idx < len(features):
                # Manual EMA using only bars 0 to 99
                manual_ema = features['close'].iloc[:test_idx].ewm(span=20, adjust=False).mean().iloc[-1]
                stored_ema = features['ema_20'].iloc[test_idx]
                
                if pd.notna(stored_ema):
                    diff = abs(manual_ema - stored_ema)
                    if diff < 1e-6:
                        print(f"      ✅ EMA verification passed (diff={diff:.10f})")
                    else:
                        print(f"      ⚠️  EMA mismatch: manual={manual_ema:.4f}, stored={stored_ema:.4f}")
        
        return features


class SimpleDirectionDetector:
    """
    Ultra-simple direction detection using only trend + momentum.
    """
    
    @staticmethod
    def detect(df: pd.DataFrame, i: int) -> int:
        """
        Determine direction from simple rules.
        
        Returns: +1 (long), -1 (short), 0 (no trade)
        """
        if i < 50:
            return 0
        
        try:
            # Get previous bar's indicators
            ema_10 = df['ema_10'].iloc[i]
            ema_20 = df['ema_20'].iloc[i]
            ema_50 = df['ema_50'].iloc[i]
            rsi = df['rsi'].iloc[i]
            
            if pd.isna(ema_10) or pd.isna(ema_20) or pd.isna(ema_50):
                return 0
            
            # Strong trend signals
            strong_uptrend = (ema_10 > ema_20) and (ema_20 > ema_50)
            strong_downtrend = (ema_10 < ema_20) and (ema_20 < ema_50)
            
            # RSI confirmation
            rsi_bullish = rsi < 60  # Not overbought
            rsi_bearish = rsi > 40  # Not oversold
            
            if strong_uptrend and rsi_bullish:
                return 1
            elif strong_downtrend and rsi_bearish:
                return -1
            else:
                return 0
        except:
            return 0
    
    @staticmethod
    def analyze_signals(df: pd.DataFrame) -> Dict:
        """Analyze quality of direction signals."""
        print(f"\n📊 DIRECTION SIGNAL ANALYSIS")
        print(f"{'='*80}")
        
        signals = []
        for i in range(len(df)):
            sig = SimpleDirectionDetector.detect(df, i)
            signals.append(sig)
        
        signals = np.array(signals)
        
        longs = (signals == 1).sum()
        shorts = (signals == -1).sum()
        flats = (signals == 0).sum()
        total = len(signals)
        
        print(f"   Long signals: {longs:,} ({longs/total*100:.1f}%)")
        print(f"   Short signals: {shorts:,} ({shorts/total*100:.1f}%)")
        print(f"   No-trade: {flats:,} ({flats/total*100:.1f}%)")
        
        if longs + shorts < total * 0.10:
            print(f"   ⚠️  WARNING: Only {(longs+shorts)/total*100:.1f}% trade signals!")
            print(f"      Direction detection may be too conservative")
        
        return {'longs': longs, 'shorts': shorts, 'flats': flats}


class TripleBarrierLabeler:
    """Triple-barrier labeling with entry at NEXT bar open."""
    
    @staticmethod
    def label(df: pd.DataFrame, tp_mult: float, sl_mult: float, 
             time_barrier: int, timeframe: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Label trades with triple barriers.
        
        Returns: (direction_labels, barrier_labels, r_post)
        """
        total_cost_r = get_total_cost_r(timeframe)
        
        direction_labels = pd.Series(0, index=df.index)
        barrier_labels = pd.Series(-1, index=df.index)
        r_post = pd.Series(0.0, index=df.index)
        
        tp_hits, sl_hits, time_exits = 0, 0, 0
        
        for i in range(len(df) - time_barrier - 2):
            # Determine direction using ONLY data up to bar i
            direction = SimpleDirectionDetector.detect(df, i)
            if direction == 0:
                continue
            
            # Entry at NEXT bar open
            if i + 1 >= len(df):
                continue
            
            entry_price = df['open'].iloc[i + 1]
            atr = df['atr'].iloc[i]
            
            if pd.isna(entry_price) or pd.isna(atr) or atr <= 0:
                continue
            
            # Set barriers
            if direction == 1:
                tp_price = entry_price + tp_mult * atr
                sl_price = entry_price - sl_mult * atr
            else:
                tp_price = entry_price - tp_mult * atr
                sl_price = entry_price + sl_mult * atr
            
            # Check barriers (starting from bar i+2)
            hit_barrier = False
            for j in range(2, time_barrier + 2):
                if i + j >= len(df):
                    break
                
                high = df['high'].iloc[i + j]
                low = df['low'].iloc[i + j]
                
                if direction == 1:
                    if high >= tp_price:
                        barrier_labels.iloc[i] = 1  # TP
                        r_post.iloc[i] = tp_mult - total_cost_r
                        tp_hits += 1
                        hit_barrier = True
                        break
                    if low <= sl_price:
                        barrier_labels.iloc[i] = 0  # SL
                        r_post.iloc[i] = -sl_mult - total_cost_r
                        sl_hits += 1
                        hit_barrier = True
                        break
                else:
                    if low <= tp_price:
                        barrier_labels.iloc[i] = 1  # TP
                        r_post.iloc[i] = tp_mult - total_cost_r
                        tp_hits += 1
                        hit_barrier = True
                        break
                    if high >= sl_price:
                        barrier_labels.iloc[i] = 0  # SL
                        r_post.iloc[i] = -sl_mult - total_cost_r
                        sl_hits += 1
                        hit_barrier = True
                        break
            
            if not hit_barrier:
                # Time exit
                exit_idx = min(i + time_barrier + 1, len(df) - 1)
                exit_price = df['close'].iloc[exit_idx]
                pnl = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
                r_value = pnl / atr
                barrier_labels.iloc[i] = 2  # Time
                r_post.iloc[i] = r_value - total_cost_r
                time_exits += 1
            
            direction_labels.iloc[i] = direction
        
        total = tp_hits + sl_hits + time_exits
        if total > 0:
            print(f"\n   Triple Barrier Distribution:")
            print(f"   - TP: {tp_hits:,} ({tp_hits/total*100:.1f}%)")
            print(f"   - SL: {sl_hits:,} ({sl_hits/total*100:.1f}%)")
            print(f"   - Time: {time_exits:,} ({time_exits/total*100:.1f}%)")
        
        return direction_labels, barrier_labels, r_post


def find_best_config(df: pd.DataFrame, timeframe: str, verbose: bool = True) -> Tuple[float, int]:
    """Find best TP/TB config with detailed diagnostics."""
    print(f"\n🔍 OPTIMIZING CONFIGURATION (VERBOSE)")
    print(f"{'='*80}")
    
    tp_candidates = CONFIG.get_tp_multipliers(timeframe)
    tb_candidates = CONFIG.get_time_barriers(timeframe)
    
    best_config = None
    best_score = -np.inf
    
    print(f"{'TP':>6} {'TB':>6} {'Trades':>10} {'Winners':>10} {'Losers':>10} {'WR':>8} {'PF':>8} {'MeanR':>8} {'Status':<15}")
    print("-" * 120)
    
    for tp in tp_candidates:
        for tb in tb_candidates:
            direction_labels, barrier_labels, r_post = TripleBarrierLabeler.label(
                df, tp, CONFIG.SL_MULTIPLIER, tb, timeframe
            )
            
            mask = barrier_labels != -1
            total_trades = mask.sum()
            
            if total_trades == 0:
                print(f"{tp:>6.1f} {tb:>6} {total_trades:>10,} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'N/A':>8} ❌ No trades")
                continue
            
            if total_trades < CONFIG.get_min_trades_raw(timeframe):
                print(f"{tp:>6.1f} {tb:>6} {total_trades:>10,} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'N/A':>8} ⚠️  Too few")
                continue
            
            r_sel = r_post[mask].values
            metrics = RiskMetrics.calculate_all_metrics(r_sel)
            
            winners = (r_sel > 0).sum()
            losers = (r_sel < 0).sum()
            
            # Detailed status
            status_parts = []
            if metrics['profit_factor'] < 1.0:
                status_parts.append("❌ PF<1")
            elif metrics['profit_factor'] < 1.2:
                status_parts.append("⚠️ Low PF")
            else:
                status_parts.append("✅ Good PF")
            
            if metrics['win_rate'] < 0.45:
                status_parts.append("Low WR")
            
            status = " ".join(status_parts)
            
            print(f"{tp:>6.1f} {tb:>6} {total_trades:>10,} {winners:>10,} {losers:>10,} "
                  f"{metrics['win_rate']:>7.1%} {metrics['profit_factor']:>8.2f} "
                  f"{metrics['mean_r']:>8.3f} {status:<15}")
            
            # Show a sample of R values for first config
            if verbose and tp == tp_candidates[0] and tb == tb_candidates[0]:
                print(f"\n   📊 Sample R-values for TP={tp}, TB={tb}:")
                sample_r = r_sel[:50]
                print(f"      First 50 trades: {sample_r}")
                print(f"      Min R: {r_sel.min():.3f}, Max R: {r_sel.max():.3f}")
                print(f"      Positive: {(r_sel > 0).sum()}, Negative: {(r_sel < 0).sum()}, Zero: {(r_sel == 0).sum()}")
                print()
            
            if metrics['profit_factor'] > 1.2 and metrics['win_rate'] > 0.45:
                score = metrics['profit_factor'] * metrics['win_rate']
                if score > best_score:
                    best_score = score
                    best_config = (tp, tb)
    
    if best_config is None:
        print(f"\n❌ NO PROFITABLE CONFIG FOUND")
        print(f"\n🔍 DIAGNOSTIC SUGGESTIONS:")
        print(f"   1. Check if direction signals are being generated")
        print(f"   2. Verify entry price (should be NEXT bar open)")
        print(f"   3. Check ATR values (shouldn't be 0 or NaN)")
        print(f"   4. Review transaction costs ({get_total_cost_r(timeframe):.3f}R per trade)")
        print(f"\n   Using default config for further analysis...")
        return tp_candidates[1], tb_candidates[1]
    
    print(f"\n✅ BEST: TP={best_config[0]:.1f}, TB={best_config[1]}")
    return best_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='XAUUSD')
    parser.add_argument('--timeframe', type=str, default='5T')
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--sample-size', type=int, default=None, help='Use only first N rows for testing')
    parser.add_argument('--debug', action='store_true', help='Ultra-verbose debugging')
    args = parser.parse_args()
    
    print(f"\n{'#'*80}")
    print(f"# CITADEL V3.2 - ZERO LEAKAGE DIAGNOSTIC SYSTEM")
    print(f"# {args.symbol} {args.timeframe}")
    if args.debug:
        print(f"# DEBUG MODE: ON")
    print(f"{'#'*80}")
    
    # Load RAW data only
    df = DataLoader.load_timeframe_data(args.symbol, args.timeframe)
    
    # Use sample if specified (for faster testing)
    if args.sample_size:
        print(f"\n⚠️  Using only first {args.sample_size:,} rows for testing")
        df = df.head(args.sample_size)
    
    # Build features from scratch
    df = ZeroLeakageFeatures.engineer_all(df, diagnose=args.diagnose)
    
    # Verify we have data
    print(f"\n📊 DATA SUMMARY AFTER FEATURE ENGINEERING")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Column names: {list(df.columns)}")
    print(f"   Has ATR: {'atr' in df.columns}")
    if 'atr' in df.columns:
        print(f"   ATR range: {df['atr'].min():.4f} to {df['atr'].max():.4f}")
        print(f"   ATR mean: {df['atr'].mean():.4f}")
        print(f"   ATR nulls: {df['atr'].isna().sum()}")
    
    # Check EMAs
    for col in ['ema_10', 'ema_20', 'ema_50']:
        if col in df.columns:
            print(f"   {col}: mean={df[col].mean():.2f}, nulls={df[col].isna().sum()}")
    
    # Analyze direction signals
    signal_stats = SimpleDirectionDetector.analyze_signals(df)
    
    if signal_stats['longs'] + signal_stats['shorts'] == 0:
        print(f"\n❌ FATAL ERROR: No directional signals generated!")
        print(f"   This means SimpleDirectionDetector.detect() always returns 0")
        print(f"   Check that EMA columns exist and have valid values")
        print(f"\n   Sample of EMA/RSI values (rows 50-55):")
        if len(df) > 55:
            print(df.iloc[50:55][['ema_10', 'ema_20', 'ema_50', 'rsi', 'atr']])
        return
    
    # Manual test: Check one specific bar
    print(f"\n🧪 MANUAL LABELING TEST (Bar 100)")
    print(f"{'='*80}")
    if len(df) > 150:
        test_idx = 100
        direction = SimpleDirectionDetector.detect(df, test_idx)
        
        print(f"   Bar {test_idx} direction: {direction} ({'Long' if direction == 1 else 'Short' if direction == -1 else 'Flat'})")
        
        if args.debug:
            print(f"\n   DEBUG - Bar {test_idx} details:")
            print(f"      ema_10: {df['ema_10'].iloc[test_idx]:.2f}")
            print(f"      ema_20: {df['ema_20'].iloc[test_idx]:.2f}")
            print(f"      ema_50: {df['ema_50'].iloc[test_idx]:.2f}")
            print(f"      rsi: {df['rsi'].iloc[test_idx]:.2f}")
            print(f"      Strong uptrend: {(df['ema_10'].iloc[test_idx] > df['ema_20'].iloc[test_idx]) and (df['ema_20'].iloc[test_idx] > df['ema_50'].iloc[test_idx])}")
            print(f"      Strong downtrend: {(df['ema_10'].iloc[test_idx] < df['ema_20'].iloc[test_idx]) and (df['ema_20'].iloc[test_idx] < df['ema_50'].iloc[test_idx])}")
        
        if direction != 0:
            entry_price = df['open'].iloc[test_idx + 1]
            atr = df['atr'].iloc[test_idx]
            
            print(f"\n   Entry price (bar {test_idx+1} open): {entry_price:.2f}")
            print(f"   ATR (bar {test_idx}): {atr:.4f}")
            
            tp_test = 2.0  # Test with 2.0 multiplier
            sl_test = 1.0
            
            if direction == 1:
                tp_price = entry_price + tp_test * atr
                sl_price = entry_price - sl_test * atr
                print(f"   TP target: {tp_price:.2f} (entry + {tp_test} ATR)")
                print(f"   SL target: {sl_price:.2f} (entry - {sl_test} ATR)")
                
                # Check next 20 bars
                print(f"\n   Checking next 20 bars:")
                hit_something = False
                for j in range(2, min(22, len(df) - test_idx)):
                    high = df['high'].iloc[test_idx + j]
                    low = df['low'].iloc[test_idx + j]
                    
                    if high >= tp_price:
                        print(f"      Bar {test_idx + j}: ✅ HIT TP at high={high:.2f}")
                        hit_something = True
                        break
                    elif low <= sl_price:
                        print(f"      Bar {test_idx + j}: ❌ HIT SL at low={low:.2f}")
                        hit_something = True
                        break
                    else:
                        if j <= 5 or args.debug:  # Show first 5 bars or all if debug
                            print(f"      Bar {test_idx + j}: No hit (high={high:.2f}, low={low:.2f})")
                
                if not hit_something:
                    print(f"      ⏱️ Timed out - no TP/SL hit in 20 bars")
                    final_price = df['close'].iloc[test_idx + 20]
                    pnl = final_price - entry_price
                    print(f"      Final close: {final_price:.2f}, PnL: {pnl:.2f}, R: {pnl/atr:.3f}")
            else:
                # Similar for shorts
                tp_price = entry_price - tp_test * atr
                sl_price = entry_price + sl_test * atr
                print(f"   TP target: {tp_price:.2f} (entry - {tp_test} ATR)")
                print(f"   SL target: {sl_price:.2f} (entry + {sl_test} ATR)")
        else:
            print(f"   ⚠️  No trade signal at bar {test_idx}")
    
    # Find best config with verbose output
    print(f"\n" + "="*80)
    best_tp, best_tb = find_best_config(df, args.timeframe, verbose=True)
    
    print(f"\n{'='*80}")
    print(f"✅ DIAGNOSTIC COMPLETE")
    print(f"{'='*80}")
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Review direction signal percentages above")
    print(f"   2. Check manual labeling test - did it hit TP/SL?")
    print(f"   3. Look at config table - what were the actual PF/WR values?")
    print(f"   4. If still no profit, share FULL output for analysis")
    print(f"\n   For ultra-verbose: add --debug flag")
    print(f"   For faster testing: --sample-size 20000")


if __name__ == '__main__':
    main()