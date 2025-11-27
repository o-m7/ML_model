"""
═══════════════════════════════════════════════════════════════════════════════
CITADEL SHARED FEATURE ENGINEERING MODULE V2.0
═══════════════════════════════════════════════════════════════════════════════

MAJOR ENHANCEMENTS:
- Expanded feature sets with more predictive power
- Proper orderflow/liquidity features from OHLCV
- Bid/ask spread analysis when quote data available
- Multi-level liquidity sweep detection
- Institutional order block identification
- Volume profile approximation

NO LOOK-AHEAD BIAS: All rolling calculations use .shift(1) where needed.
"""

import numpy as np
import pandas as pd
from typing import List, Set, Optional
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════
# SHARED FEATURES - CORE SET AVAILABLE TO ALL STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

SHARED_FEATURES = [
    # Core volatility
    'atr', 'atr_pct', 'realized_vol', 'vol_ratio',
    
    # Regime detection
    'regime_trending', 'regime_ranging',
    'regime_high_vol', 'regime_low_vol', 'regime_volatile',
    'trend_strength', 'price_position',
    'adx', 'vol_zscore',
    
    # Session detection
    'session_asian', 'session_london', 'session_ny', 'session_overlap',
    'london_open', 'ny_open', 'comex_open', 'session_open_flag',
    'time_of_day_vol',
    'hour', 'minute', 'dayofweek', 'friday_afternoon',
    
    # Basic momentum
    'rsi', 'rsi_divergence',
    'macd_hist', 'macd_cross',
    
    # Price structure
    'higher_high', 'lower_low', 'inside_bar',
    'body_pct', 'upper_wick_pct', 'lower_wick_pct',
]

EXCLUDE_COLS = frozenset([
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'bid_open', 'bid_high', 'bid_low', 'bid_close',
    'ask_open', 'ask_high', 'ask_low', 'ask_close',
])

# Higher timeframe prefixes to exclude from intraday models
HTF_PREFIXES = ("1H_", "4H_", "D1_", "W1_", "HTF_")


def build_feature_cols_for_strategy(required_features: List[str], df: pd.DataFrame) -> List[str]:
    """Build controlled feature list for a strategy."""
    candidate_features = list(required_features) + SHARED_FEATURES
    
    seen: Set[str] = set()
    feature_cols: List[str] = []
    for f in candidate_features:
        if f in seen:
            continue
        if f in EXCLUDE_COLS:
            continue
        if f not in df.columns:
            continue
        if any(f.startswith(p) for p in HTF_PREFIXES):
            continue
        seen.add(f)
        feature_cols.append(f)
    
    return feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# CORE PRICE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

class CoreFeatures:
    """Core price and volatility features needed by all strategies."""
    
    @staticmethod
    def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add fundamental price features."""
        df = df.copy()
        
        # ATR
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        
        df['true_range'] = tr
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / (df['close'] + 1e-10)
        
        # Volatility measures
        df['returns'] = df['close'].pct_change()
        df['realized_vol'] = df['returns'].rolling(20).std()
        df['vol_ratio'] = df['atr'] / (df['atr'].rolling(50).mean().shift(1) + 1e-10)
        
        # Candle structure
        range_ = df['high'] - df['low'] + 1e-10
        body = abs(df['close'] - df['open'])
        upper_wick = df['high'] - np.maximum(df['close'], df['open'])
        lower_wick = np.minimum(df['close'], df['open']) - df['low']
        
        df['body_pct'] = body / range_
        df['upper_wick_pct'] = upper_wick / range_
        df['lower_wick_pct'] = lower_wick / range_
        
        # Price structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(np.int8)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(np.int8)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(np.int8)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(np.int8)
        df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & 
                           (df['low'] > df['low'].shift(1))).astype(np.int8)
        
        # Basic momentum indicators
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_ma'] = df['rsi'].rolling(5).mean()
        df['rsi_divergence'] = (df['rsi'] - df['rsi'].shift(5)) * np.sign(df['close'].diff(5))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = (np.sign(df['macd_hist']) != np.sign(df['macd_hist'].shift(1))).astype(np.int8)
        
        # Moving averages
        for period in [10, 20, 50]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / (df['atr'] + 1e-10)
        
        df['ema_slope_20'] = df['ema_20'].diff(5) / (df['atr'] + 1e-10)
        df['ema_cross_10_20'] = (df['ema_10'] > df['ema_20']).astype(np.int8)
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# SESSION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class SessionDetector:
    """Detect trading sessions and key times."""
    
    @staticmethod
    def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive session features."""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            df['hour'] = ts.dt.hour
            df['minute'] = ts.dt.minute
            df['dayofweek'] = ts.dt.dayofweek
        else:
            df['hour'] = df.get('hour', 12)
            df['minute'] = df.get('minute', 0)
            df['dayofweek'] = df.get('dayofweek', 2)
        
        df['session_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(np.int8)
        df['session_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(np.int8)
        df['session_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(np.int8)
        df['session_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(np.int8)
        
        df['london_open'] = ((df['hour'] == 8) & (df['minute'] < 30)).astype(np.int8)
        df['ny_open'] = ((df['hour'] == 14) & (df['minute'] >= 30) | 
                        (df['hour'] == 15) & (df['minute'] < 30)).astype(np.int8)
        df['comex_open'] = ((df['hour'] == 13) & (df['minute'] >= 15) & 
                           (df['minute'] < 45)).astype(np.int8)
        
        df['session_open_flag'] = (
            df['london_open'] | df['ny_open'] | df['comex_open']
        ).astype(np.int8)
        
        df['time_of_day_vol'] = (
            df['session_asian'] * 0.5 +
            df['session_london'] * 0.8 +
            df['session_ny'] * 0.9 +
            df['session_overlap'] * 1.0
        )
        
        df['friday_afternoon'] = (
            (df['dayofweek'] == 4) & (df['hour'] >= 18)
        ).astype(np.int8)
        
        # Session-relative position (how far into session)
        df['session_progress'] = df['minute'] / 60 + (df['hour'] % 8) / 8
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """Detect market regimes with proper time-shifting."""
    
    @staticmethod
    def add_regime_features(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """Add regime detection features - ALL HISTORICAL ONLY."""
        df = df.copy()
        
        atr = df['atr'] if 'atr' in df.columns else df['close'].rolling(14).std()
        
        # Volatility regime
        atr_ma = atr.rolling(lookback).mean().shift(1)
        atr_std = atr.rolling(lookback).std().shift(1)
        df['vol_zscore'] = (atr - atr_ma) / (atr_std + 1e-10)
        
        df['regime_high_vol'] = (df['vol_zscore'] > 1.0).astype(np.int8)
        df['regime_low_vol'] = (df['vol_zscore'] < -0.5).astype(np.int8)
        df['regime_volatile'] = (df['vol_zscore'].abs() > 1.5).astype(np.int8)
        
        # Price position in range (using prior lookback window only)
        high_roll = df['high'].shift(1).rolling(lookback).max()
        low_roll = df['low'].shift(1).rolling(lookback).min()
        range_roll = high_roll - low_roll + 1e-10
        df['price_position'] = (df['close'] - low_roll) / range_roll
        
        # ADX calculation
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        atr_14 = tr.rolling(14).mean().shift(1)
        plus_di = 100 * (plus_dm.rolling(14).mean().shift(1) / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean().shift(1) / (atr_14 + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean().shift(1)
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['di_cross'] = (plus_di > minus_di).astype(np.int8)
        
        df['trend_strength'] = df['adx'] / 100
        
        df['regime_trending'] = (df['adx'] > 25).astype(np.int8)
        df['regime_ranging'] = (df['adx'] <= 20).astype(np.int8)
        
        df['regime_trend_up'] = (
            (df['regime_trending'] == 1) & (df['price_position'] > 0.6)
        ).astype(np.int8)
        df['regime_trend_down'] = (
            (df['regime_trending'] == 1) & (df['price_position'] < 0.4)
        ).astype(np.int8)
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# ORDERFLOW & LIQUIDITY FEATURES
# ═══════════════════════════════════════════════════════════════════════════

class OrderflowFeatures:
    """
    Orderflow and liquidity features derived from OHLCV.
    
    These approximate institutional activity patterns:
    - Delta proxy from candle structure
    - Volume-weighted moves
    - Absorption patterns
    - Liquidity sweep detection
    """
    
    @staticmethod
    def add_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add orderflow-derived features from OHLCV."""
        df = df.copy()
        
        atr = df['atr'] if 'atr' in df.columns else df['close'].rolling(14).std()
        
        # Delta proxy: where price closes relative to range
        range_ = df['high'] - df['low'] + 1e-10
        df['delta_proxy'] = (df['close'] - df['low']) / range_ * 2 - 1  # -1 to 1
        
        # Cumulative delta
        df['delta_cumsum_5'] = df['delta_proxy'].rolling(5).sum()
        df['delta_cumsum_10'] = df['delta_proxy'].rolling(10).sum()
        df['delta_cumsum_20'] = df['delta_proxy'].rolling(20).sum()
        
        # Delta momentum/acceleration
        df['delta_momentum'] = df['delta_proxy'].diff(3)
        df['delta_acceleration'] = df['delta_momentum'].diff(2)
        
        # Aggressor imbalance: sustained buying or selling
        df['aggressor_imbalance'] = (
            (df['delta_cumsum_5'].abs() > 3) |
            (df['delta_proxy'].rolling(3).sum().abs() > 2)
        ).astype(np.int8)
        
        # Volume analysis
        if 'volume' in df.columns and df['volume'].sum() > 0:
            vol = df['volume']
        else:
            vol = pd.Series(1, index=df.index)
        
        vol_ma = vol.rolling(20).mean().shift(1)
        vol_std = vol.rolling(20).std().shift(1)
        
        df['volume_zscore'] = (vol - vol_ma) / (vol_std + 1e-10)
        df['volume_spike'] = (vol > vol_ma * 1.5).astype(np.int8)
        df['volume_climax'] = (vol > vol_ma * 2.5).astype(np.int8)
        df['volume_dry'] = (vol < vol_ma * 0.5).astype(np.int8)
        
        # Volume-weighted move
        df['volume_delta'] = vol * df['delta_proxy']
        df['cumulative_volume_delta'] = df['volume_delta'].rolling(20).sum()
        
        # Absorption: high volume but small range (orders being absorbed)
        df['absorption'] = (
            (df['volume_spike'] == 1) & 
            (range_ < atr * 0.5)
        ).astype(np.int8)
        
        # Effort vs Result divergence
        move = df['close'].diff().abs()
        df['effort_result_ratio'] = move / (vol + 1e-10)
        effort_ma = df['effort_result_ratio'].rolling(20).mean().shift(1)
        df['effort_divergence'] = (df['effort_result_ratio'] - effort_ma) / (effort_ma + 1e-10)
        
        # Quote velocity proxy (rate of price change)
        df['price_velocity'] = df['close'].diff(3) / (atr + 1e-10)
        df['price_acceleration'] = df['price_velocity'].diff(2)
        
        return df


class LiquidityFeatures:
    """
    Liquidity analysis features.
    
    Identifies:
    - Liquidity sweeps (stop hunts)
    - High liquidity zones (order blocks)
    - Failed breakouts (trapped traders)
    """
    
    @staticmethod
    def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity and sweep detection features."""
        df = df.copy()
        
        atr = df['atr'] if 'atr' in df.columns else df['close'].rolling(14).std()
        
        # Multi-level swing detection (using prior bars only)
        for lb in [5, 10, 20, 50]:
            df[f'swing_high_{lb}'] = df['high'].shift(1).rolling(lb).max()
            df[f'swing_low_{lb}'] = df['low'].shift(1).rolling(lb).min()
        
        # Liquidity sweep: price exceeds level then reverses
        # Bullish sweep: current bar takes out prior lows then closes above prior low
        df['sweep_bullish_5'] = (
            (df['low'] < df['swing_low_5'].shift(1)) &
            (df['close'] > df['swing_low_5'].shift(1))
        ).astype(np.int8)
        
        df['sweep_bullish_20'] = (
            (df['low'] < df['swing_low_20'].shift(1)) &
            (df['close'] > df['swing_low_20'].shift(1))
        ).astype(np.int8)
        
        df['sweep_bullish_50'] = (
            (df['low'] < df['swing_low_50'].shift(1)) &
            (df['close'] > df['swing_low_50'].shift(1))
        ).astype(np.int8)
        
        # Bearish sweep: takes out highs then closes below prior swing highs
        df['sweep_bearish_5'] = (
            (df['high'] > df['swing_high_5'].shift(1)) &
            (df['close'] < df['swing_high_5'].shift(1))
        ).astype(np.int8)
        
        df['sweep_bearish_20'] = (
            (df['high'] > df['swing_high_20'].shift(1)) &
            (df['close'] < df['swing_high_20'].shift(1))
        ).astype(np.int8)
        
        df['sweep_bearish_50'] = (
            (df['high'] > df['swing_high_50'].shift(1)) &
            (df['close'] < df['swing_high_50'].shift(1))
        ).astype(np.int8)
        
        # Combined sweep signals
        df['sweep_bullish'] = (
            df['sweep_bullish_5'] | df['sweep_bullish_20'] | df['sweep_bullish_50']
        ).astype(np.int8)
        
        df['sweep_bearish'] = (
            df['sweep_bearish_5'] | df['sweep_bearish_20'] | df['sweep_bearish_50']
        ).astype(np.int8)
        
        # Sweep magnitude (how far past the prior swing level)
        df['sweep_magnitude'] = np.where(
            df['sweep_bullish'] == 1,
            (df['swing_low_20'].shift(1) - df['low']) / (atr + 1e-10),
            np.where(
                df['sweep_bearish'] == 1,
                (df['high'] - df['swing_high_20'].shift(1)) / (atr + 1e-10),
                0
            )
        )
        
        # Wick rejection after sweep
        range_ = df['high'] - df['low'] + 1e-10
        lower_wick = np.minimum(df['close'], df['open']) - df['low']
        upper_wick = df['high'] - np.maximum(df['close'], df['open'])
        
        df['wick_rejection_low'] = (lower_wick / range_ > 0.6).astype(np.int8)
        df['wick_rejection_high'] = (upper_wick / range_ > 0.6).astype(np.int8)
        df['wick_rejection'] = (df['wick_rejection_low'] | df['wick_rejection_high']).astype(np.int8)
        
        # Quality sweep: sweep + rejection + magnitude
        df['quality_sweep_long'] = (
            (df['sweep_bullish'] == 1) &
            (df['wick_rejection_low'] == 1) &
            (df['sweep_magnitude'] > 0.3)
        ).astype(np.int8)
        
        df['quality_sweep_short'] = (
            (df['sweep_bearish'] == 1) &
            (df['wick_rejection_high'] == 1) &
            (df['sweep_magnitude'] > 0.3)
        ).astype(np.int8)
        
        # Failed breakout detection (compare current bar to prior swing levels)
        df['failed_breakout_high'] = (
            (df['high'] > df['swing_high_20'].shift(1)) &
            (df['close'] < df['swing_high_20'].shift(1))
        ).astype(np.int8)
        
        df['failed_breakout_low'] = (
            (df['low'] < df['swing_low_20'].shift(1)) &
            (df['close'] > df['swing_low_20'].shift(1))
        ).astype(np.int8)
        
        df['failed_breakout'] = (
            df['failed_breakout_high'] | df['failed_breakout_low']
        ).astype(np.int8)
        
        # Trap signal: failed breakout + rejection wick
        df['bull_trap'] = (
            (df['failed_breakout_high'] == 1) &
            (df['wick_rejection_high'] == 1)
        ).astype(np.int8)
        
        df['bear_trap'] = (
            (df['failed_breakout_low'] == 1) &
            (df['wick_rejection_low'] == 1)
        ).astype(np.int8)
        
        df['trap_signal'] = (df['bull_trap'] | df['bear_trap']).astype(np.int8)
        
        # No follow-through (price fails to continue after breakout)
        df['no_followthrough'] = (
            df['failed_breakout'].rolling(3).sum() > 0
        ).astype(np.int8)
        
        return df


class SpreadFeatures:
    """
    Spread and quote-based features when bid/ask data available.
    """
    
    @staticmethod
    def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add spread features if bid/ask columns exist."""
        df = df.copy()
        
        has_quotes = 'bid_close' in df.columns and 'ask_close' in df.columns
        
        if has_quotes:
            df['mid_price'] = (df['bid_close'] + df['ask_close']) / 2
            df['spread'] = df['ask_close'] - df['bid_close']
            df['spread_pct'] = df['spread'] / (df['mid_price'] + 1e-10)
            
            atr = df['atr'] if 'atr' in df.columns else df['close'].rolling(14).std()
            df['spread_vs_atr'] = df['spread'] / (atr + 1e-10)
            
            # Spread expansion/contraction
            spread_ma = df['spread'].rolling(20).mean().shift(1)
            df['spread_expansion'] = (df['spread'] > spread_ma * 1.5).astype(np.int8)
            df['spread_contraction'] = (df['spread'] < spread_ma * 0.5).astype(np.int8)
            
            # Quote imbalance: where close is vs mid
            df['quote_imbalance'] = (df['close'] - df['mid_price']) / (df['spread'] + 1e-10)
            
            # Bid/ask velocity
            df['bid_velocity'] = df['bid_close'].diff(3) / (atr + 1e-10)
            df['ask_velocity'] = df['ask_close'].diff(3) / (atr + 1e-10)
            df['quote_velocity_diff'] = df['bid_velocity'] - df['ask_velocity']
        else:
            # Proxy features when no quote data
            atr = df['atr'] if 'atr' in df.columns else df['close'].rolling(14).std()
            df['spread'] = atr * 0.001  # Typical gold spread proxy
            df['spread_pct'] = df['spread'] / (df['close'] + 1e-10)
            df['spread_vs_atr'] = 0.001
            df['spread_expansion'] = 0
            df['spread_contraction'] = 0
            df['quote_imbalance'] = 0
            df['quote_velocity_diff'] = df['close'].diff(3) / (atr + 1e-10)
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# VWAP & INSTITUTIONAL LEVELS
# ═══════════════════════════════════════════════════════════════════════════

class InstitutionalFeatures:
    """
    VWAP, volume profile, and institutional level features.
    """
    
    @staticmethod
    def add_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP and related features."""
        df = df.copy()
        
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            df['volume'] = 1
        
        atr = df['atr'] if 'atr' in df.columns else df['close'].rolling(14).std()
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        pv = typical_price * df['volume']
        
        # VWAP over different periods
        for period in [20, 50, 100]:
            df[f'vwap_{period}'] = (
                pv.rolling(period).sum() / 
                (df['volume'].rolling(period).sum() + 1e-10)
            )
            df[f'vwap_deviation_{period}'] = (
                (df['close'] - df[f'vwap_{period}']) / (atr + 1e-10)
            )
        
        df['vwap_deviation'] = df['vwap_deviation_20']
        
        # VWAP Z-score (using prior-bar statistics only to avoid look-ahead)
        dev_ma = df['vwap_deviation'].shift(1).rolling(50).mean()
        dev_std = df['vwap_deviation'].shift(1).rolling(50).std()
        df['vwap_zscore'] = (df['vwap_deviation'] - dev_ma) / (dev_std + 1e-10)
        
        # VWAP bands (using prior-bar VWAP calculations to avoid look-ahead)
        price_sq_vol = (df['close'] ** 2) * df['volume']
        vwap_sq = price_sq_vol.shift(1).rolling(20).sum() / (df['volume'].shift(1).rolling(20).sum() + 1e-10)
        vwap_var = vwap_sq - (df['vwap_20'].shift(1) ** 2)
        df['vwap_std'] = np.sqrt(np.maximum(0, vwap_var))
        
        df['vwap_upper'] = df['vwap_20'].shift(1) + 2 * df['vwap_std']
        df['vwap_lower'] = df['vwap_20'].shift(1) - 2 * df['vwap_std']
        
        band_range = df['vwap_upper'] - df['vwap_lower'] + 1e-10
        df['vwap_band_position'] = (df['close'] - df['vwap_lower']) / band_range
        
        # Extreme deviations
        df['vwap_oversold'] = (df['vwap_zscore'] < -2.0).astype(np.int8)
        df['vwap_overbought'] = (df['vwap_zscore'] > 2.0).astype(np.int8)
        df['vwap_extreme'] = (df['vwap_zscore'].abs() > 2.0).astype(np.int8)
        
        # Mean reversion signal: at extreme + showing reversal
        df['mr_signal_long'] = (
            (df['vwap_zscore'] < -1.5) &
            (df['close'] > df['open'])
        ).astype(np.int8)
        
        df['mr_signal_short'] = (
            (df['vwap_zscore'] > 1.5) &
            (df['close'] < df['open'])
        ).astype(np.int8)
        
        df['mean_reversion_signal'] = (
            df['mr_signal_long'] | df['mr_signal_short']
        ).astype(np.int8)
        
        df['institutional_level'] = df['vwap_20']
        
        # VWAP slope (trend of fair value)
        df['vwap_slope'] = df['vwap_20'].diff(5) / (atr + 1e-10)
        
        # Price vs multiple VWAPs (confluence)
        df['above_vwap_20'] = (df['close'] > df['vwap_20']).astype(np.int8)
        df['above_vwap_50'] = (df['close'] > df['vwap_50']).astype(np.int8)
        df['vwap_confluence'] = df['above_vwap_20'] + df['above_vwap_50']
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# VOLATILITY & COMPRESSION FEATURES
# ═══════════════════════════════════════════════════════════════════════════

class VolatilityFeatures:
    """Volatility compression/expansion features."""
    
    @staticmethod
    def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility compression/expansion features."""
        df = df.copy()
        
        atr = df['atr'] if 'atr' in df.columns else df['close'].rolling(14).std()
        
        # ATR ratio (short vs long)
        atr_short = atr.rolling(5).mean().shift(1)
        atr_long = atr.rolling(20).mean().shift(1)
        df['atr_ratio'] = atr_short / (atr_long + 1e-10)
        
        df['atr_compression'] = (df['atr_ratio'] < 0.7).astype(np.int8)
        df['atr_expansion'] = (df['atr_ratio'] > 1.3).astype(np.int8)
        
        # Range squeeze
        tr = df['high'] - df['low']
        tr_min = tr.rolling(10).min().shift(1)
        df['range_squeeze'] = (tr <= tr_min * 1.1).astype(np.int8)
        
        # Bollinger squeeze
        close_ma = df['close'].rolling(20).mean().shift(1)
        close_std = df['close'].rolling(20).std().shift(1)
        df['bb_width'] = (close_std * 4) / (close_ma + 1e-10)
        bb_width_ma = df['bb_width'].rolling(50).mean().shift(1)
        df['bb_squeeze'] = (df['bb_width'] < bb_width_ma * 0.5).astype(np.int8)
        
        # Combined squeeze
        df['squeeze_setup'] = (
            (df['atr_compression'] == 1) |
            (df['range_squeeze'] == 1) |
            (df['bb_squeeze'] == 1)
        ).astype(np.int8)
        
        df['squeeze_bars'] = df['squeeze_setup'].rolling(10).sum()
        
        # Breakout direction after squeeze
        df['breakout_direction'] = np.sign(df['close'].diff(5))
        
        # Volatility expansion (for session burst strategies)
        df['volatility_expansion'] = (atr_short > atr_long * 1.3).astype(np.int8)
        
        # Opening range (using prior 6-bar window, not including current bar)
        df['opening_range_high'] = df['high'].shift(1).rolling(6).max()
        df['opening_range_low'] = df['low'].shift(1).rolling(6).min()
        df['opening_range'] = (df['opening_range_high'] - df['opening_range_low']) / (atr + 1e-10)
        
        # Volume with compression
        if 'volume' in df.columns and df['volume'].sum() > 0:
            vol_ma = df['volume'].rolling(20).mean().shift(1)
            df['volume_expansion'] = (df['volume'] > vol_ma * 1.3).astype(np.int8)
        else:
            df['volume_expansion'] = 0
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# MOMENTUM & TREND FEATURES
# ═══════════════════════════════════════════════════════════════════════════

class MomentumFeatures:
    """Momentum and trend following features."""
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum quality and trend features."""
        df = df.copy()
        
        atr = df['atr'] if 'atr' in df.columns else df['close'].rolling(14).std()
        
        # Multi-period returns
        for period in [3, 5, 10, 20]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
            df[f'momentum_{period}'] = df['close'].diff(period) / (atr + 1e-10)
        
        # Momentum quality: aligned across timeframes
        df['momentum_quality'] = (
            (np.sign(df['momentum_5']) == np.sign(df['momentum_10'])) &
            (np.sign(df['momentum_10']) == np.sign(df['momentum_20']))
        ).astype(np.int8)
        
        # Momentum acceleration
        df['momentum_accel'] = df['momentum_5'].diff(3)
        
        # Pullback detection
        df['pullback_depth'] = df['close'].diff(3) / (atr + 1e-10)
        
        regime_trend_up = df.get('regime_trend_up', pd.Series(0, index=df.index))
        regime_trend_down = df.get('regime_trend_down', pd.Series(0, index=df.index))
        
        df['pullback_long'] = (
            (regime_trend_up == 1) &
            (df['pullback_depth'] < -0.3) &
            (df['pullback_depth'] > -1.5)
        ).astype(np.int8)
        
        df['pullback_short'] = (
            (regime_trend_down == 1) &
            (df['pullback_depth'] > 0.3) &
            (df['pullback_depth'] < 1.5)
        ).astype(np.int8)
        
        df['pullback_entry'] = (df['pullback_long'] | df['pullback_short']).astype(np.int8)
        
        # Trend continuation signals
        df['trend_continuation_long'] = (
            (df['momentum_quality'] == 1) &
            (df['momentum_5'] > 0) &
            (df['pullback_long'] == 1)
        ).astype(np.int8)
        
        df['trend_continuation_short'] = (
            (df['momentum_quality'] == 1) &
            (df['momentum_5'] < 0) &
            (df['pullback_short'] == 1)
        ).astype(np.int8)
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# SESSION RANGE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

class SessionRangeFeatures:
    """Session-based range and breakout features."""
    
    @staticmethod
    def add_session_range_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add session range features."""
        df = df.copy()
        
        atr = df['atr'] if 'atr' in df.columns else df['close'].rolling(14).std()
        
        # Session highs/lows (16 bars ≈ 4 hours on 15T)
        df['session_high'] = df['high'].rolling(16).max().shift(1)
        df['session_low'] = df['low'].rolling(16).min().shift(1)
        df['session_range'] = (df['session_high'] - df['session_low']) / (atr + 1e-10)
        
        # Breakout detection (using prior session levels)
        df['range_breakout_up'] = (df['close'] > df['session_high'].shift(1)).astype(np.int8)
        df['range_breakout_down'] = (df['close'] < df['session_low'].shift(1)).astype(np.int8)
        df['range_breakout'] = (df['range_breakout_up'] | df['range_breakout_down']).astype(np.int8)
        
        # Volatility-confirmed breakout
        regime_high_vol = df.get('regime_high_vol', pd.Series(0, index=df.index))
        df['volatility_filter'] = (regime_high_vol == 1).astype(np.int8)
        
        df['valid_range_breakout'] = (
            (df['range_breakout'] == 1) & (df['volatility_filter'] == 1)
        ).astype(np.int8)
        
        return df


class VolumeExhaustionFeatures:
    """Volume exhaustion detection."""
    
    @staticmethod
    def add_volume_exhaustion_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume exhaustion features."""
        df = df.copy()
        
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            df['volume'] = 1
        
        vol_ma = df['volume'].rolling(20).mean().shift(1)
        vol_std = df['volume'].rolling(20).std().shift(1)
        df['vol_exhaustion_zscore'] = (df['volume'] - vol_ma) / (vol_std + 1e-10)
        
        # Exhaustion: high volume reversal candle
        df['volume_exhaustion_up'] = (
            (df['vol_exhaustion_zscore'] > 2.0) &
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1))
        ).astype(np.int8)
        
        df['volume_exhaustion_down'] = (
            (df['vol_exhaustion_zscore'] > 2.0) &
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1))
        ).astype(np.int8)
        
        df['volume_exhaustion'] = (
            df['volume_exhaustion_up'] | df['volume_exhaustion_down']
        ).astype(np.int8)
        
        # Overextension
        atr = df['atr'] if 'atr' in df.columns else df['close'].rolling(14).std()
        momentum_5 = df['close'].pct_change(5)
        df['overextension'] = (momentum_5.abs() > atr / df['close'] * 2).astype(np.int8)
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

class StrategyFeatures:
    """Main feature engineering orchestrator."""
    
    @staticmethod
    def add_all_features(df: pd.DataFrame, timeframe: str, 
                         required_features_union: List[str] = None) -> pd.DataFrame:
        """Add all features for the given timeframe."""
        initial_rows = len(df)
        
        # Core universal features
        df = CoreFeatures.add_core_features(df)
        df = SessionDetector.add_session_features(df)
        df = RegimeDetector.add_regime_features(df)
        
        # Orderflow and liquidity
        df = OrderflowFeatures.add_orderflow_features(df)
        df = LiquidityFeatures.add_liquidity_features(df)
        df = SpreadFeatures.add_spread_features(df)
        
        # One VWAP block only - available for all timeframes
        df = InstitutionalFeatures.add_vwap_features(df)
        
        # Timeframe-specific blocks
        if timeframe in ['5T', '15T']:
            df = VolatilityFeatures.add_volatility_features(df)
            df = MomentumFeatures.add_momentum_features(df)
            
        if timeframe in ['15T', '30T', '1H']:
            df = SessionRangeFeatures.add_session_range_features(df)
            df = VolumeExhaustionFeatures.add_volume_exhaustion_features(df)
            
        if timeframe in ['30T', '1H']:
            df = MomentumFeatures.add_momentum_features(df)
        
        # Clean
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop NaNs on essential columns only
        core_cols = ['open', 'high', 'low', 'close', 'atr']
        essential_cols = core_cols + SHARED_FEATURES
        
        if required_features_union:
            essential_cols = essential_cols + list(required_features_union)
        
        essential_cols = [c for c in essential_cols if c in df.columns]
        df = df.dropna(subset=essential_cols)
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# LIVE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def build_live_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Build features for live inference using SAME logic as training."""
    df = StrategyFeatures.add_all_features(df, timeframe, required_features_union=None)
    df = df.fillna(0)
    return df


def validate_feature_availability(df: pd.DataFrame, feature_cols: List[str]) -> tuple:
    """Validate that required features are available."""
    available = [f for f in feature_cols if f in df.columns]
    missing = [f for f in feature_cols if f not in df.columns]
    return available, missing