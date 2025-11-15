#!/usr/bin/env python3
"""
Utility functions for building live feature sets that match the production
training pipeline. Shared by the live trading engine and the standalone
GitHub Actions signal generator.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


ESSENTIAL_FEATURES = [
    'atr14', 'trange', 'vol_percentile', 'bb_mid_20', 'bb_bbp_20', 'bb_bw_20',
    'kelt_bw_20', 'squeeze_on', 'ema20_slope', 'ema50_slope', 'ema200_slope',
    'trend_strength', 'pullback_depth', 'pullback_time', 'adx14', 'aroon_up',
    'aroon_dn', 'rsi14', 'macd', 'macdh', 'stoch_k', 'obv', 'adosc',
    'dist_nearest_sr0', 'upper_wick_ratio', 'lower_wick_ratio',
    'wick_pressure', 'minute_of_day', 'dow', 'session', 'sess_asia'
]


SESSION_WINDOWS = [
    (0, 7 * 60, 0),    # Asia
    (7 * 60, 13 * 60, 1),  # Europe
    (13 * 60, 21 * 60, 2),  # US
    (21 * 60, 24 * 60, 3),  # Late/Overnight
]


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    ranges = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1)
    return ranges.max(axis=1)


def _bars_since_high(series: pd.Series) -> float:
    arr = series.to_numpy()
    if arr.size == 0 or np.isnan(arr).all():
        return np.nan
    return float(len(arr) - 1 - np.nanargmax(arr))


def _session_from_minute(minute_of_day: float) -> int:
    for start, end, label in SESSION_WINDOWS:
        if start <= minute_of_day < end:
            return label
    return SESSION_WINDOWS[-1][2]


def _session_pos(minute_of_day: float, session_label: int) -> float:
    session_label = int(session_label)
    start, end = SESSION_WINDOWS[session_label][:2]
    span = end - start
    if span <= 0:
        return 0.0
    pos = (minute_of_day - start) / span
    return float(np.clip(pos, 0.0, 1.0))


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature-enriched dataframe aligned with production models."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy().sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df['trange'] = _true_range(df)
    df['atr14'] = df['trange'].rolling(14).mean()
    df['atr20'] = df['trange'].rolling(20).mean()

    # Volume features (needed by some models)
    df['volume_sma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma20'] + 1e-10)
    df['volume_ratio'].fillna(1.0, inplace=True)  # Handle edge cases
    df['volume_sma20'].fillna(df['volume'].mean(), inplace=True)

    # Basic returns features (needed by XAUUSD models)
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    returns = df['returns']
    df['vol_10'] = returns.rolling(10).std()
    df['vol_20'] = returns.rolling(20).std()
    df['vol_ratio'] = df['vol_10'] / (df['vol_20'] + 1e-10)
    df['vol_percentile'] = df['vol_20'].rolling(120).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if pd.notna(x).all() else np.nan,
        raw=False
    )

    df['rsi14'] = ta.rsi(df['close'], length=14)

    # SMA features (needed by XAUUSD models)
    df['sma10'] = df['close'].rolling(10).mean()
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['close_vs_sma10'] = (df['close'] - df['sma10']) / (df['sma10'] + 1e-10)
    df['close_vs_sma20'] = (df['close'] - df['sma20']) / (df['sma20'] + 1e-10)
    df['close_vs_sma50'] = (df['close'] - df['sma50']) / (df['sma50'] + 1e-10)

    # Bollinger Bands
    sma20 = df['sma20']
    std20 = df['close'].rolling(20).std()
    df['bb_mid_20'] = sma20
    df['bb_up_20'] = sma20 + (2 * std20)
    df['bb_lo_20'] = sma20 - (2 * std20)

    # Bollinger Band features with both naming conventions
    df['bb_upper'] = df['bb_up_20']  # Alias for XAUUSD models
    df['bb_lower'] = df['bb_lo_20']  # Alias for XAUUSD models
    df['bb_position'] = (df['close'] - df['bb_lo_20']) / (df['bb_up_20'] - df['bb_lo_20'] + 1e-10)

    df['bb_bbp_20'] = df['bb_position']  # Alias
    df['bb_bw_20'] = (df['bb_up_20'] - df['bb_lo_20']) / (df['bb_mid_20'] + 1e-10)

    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    ema200 = df['close'].ewm(span=200, adjust=False).mean()
    df['ema20'] = ema20
    df['ema50'] = ema50
    df['ema200'] = ema200
    df['ema20_slope'] = (ema20 - ema20.shift(5)) / (ema20.shift(5).abs() + 1e-10)
    df['ema50_slope'] = (ema50 - ema50.shift(5)) / (ema50.shift(5).abs() + 1e-10)
    df['ema200_slope'] = (ema200 - ema200.shift(5)) / (ema200.shift(5).abs() + 1e-10)

    df['trend_strength'] = (ema20 - ema50).abs() / (df['atr14'] + 1e-10)

    atr20 = df['atr20'].fillna(df['atr14'])
    df['kelt_mid_20'] = ema20
    df['kelt_up_20'] = ema20 + (2 * atr20)
    df['kelt_lo_20'] = ema20 - (2 * atr20)
    df['kelt_bw_20'] = (df['kelt_up_20'] - df['kelt_lo_20']) / (df['kelt_mid_20'] + 1e-10)
    df['squeeze_on'] = (df['bb_bw_20'] < df['kelt_bw_20']).astype(int)

    lookback = 50
    rolling_high = df['close'].rolling(lookback).max()
    df['pullback_depth'] = (rolling_high - df['close']) / (df['atr14'] + 1e-10)
    df['pullback_time'] = df['close'].rolling(lookback).apply(_bars_since_high, raw=False)
    df['pullback_time'] = df['pullback_time'].fillna(0.0)

    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx is not None and not adx.empty:
        df['adx14'] = adx['ADX_14']
    else:
        df['adx14'] = 20.0

    aroon = ta.aroon(df['high'], df['low'], length=14)
    if aroon is not None and not aroon.empty:
        df['aroon_up'] = aroon['AROONU_14']
        df['aroon_dn'] = aroon['AROOND_14']
    else:
        df['aroon_up'] = 50.0
        df['aroon_dn'] = 50.0

    macd = ta.macd(df['close'])
    if macd is not None and not macd.empty:
        df['macd'] = macd['MACD_12_26_9']
        df['macds'] = macd['MACDs_12_26_9']
        df['macdh'] = macd['MACDh_12_26_9']
        # Aliases for XAUUSD models
        df['macd_signal'] = df['macds']
        df['macd_hist'] = df['macdh']
    else:
        df['macd'] = 0.0
        df['macds'] = 0.0
        df['macdh'] = 0.0
        df['macd_signal'] = 0.0
        df['macd_hist'] = 0.0

    stoch = ta.stoch(df['high'], df['low'], df['close'])
    if stoch is not None and not stoch.empty:
        df['stoch_k'] = stoch.iloc[:, 0]
    else:
        df['stoch_k'] = 50.0

    df['obv'] = ta.obv(df['close'], df['volume'])
    adosc = ta.adosc(df['high'], df['low'], df['close'], df['volume'])
    df['adosc'] = adosc if adosc is not None else 0.0

    rolling_low = df['low'].rolling(lookback).min()
    dist_high = (rolling_high - df['close']).abs()
    dist_low = (df['close'] - rolling_low).abs()
    df['dist_nearest_sr0'] = np.minimum(dist_high, dist_low) / (df['atr14'] + 1e-10)

    candle_range = (df['high'] - df['low']).replace(0, np.nan)
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_ratio'] = (upper_wick / (candle_range + 1e-10)).fillna(0.0)
    df['lower_wick_ratio'] = (lower_wick / (candle_range + 1e-10)).fillna(0.0)
    df['wick_pressure'] = (df['upper_wick_ratio'] - df['lower_wick_ratio'])

    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['dow'] = df.index.dayofweek
    df['session'] = df['minute_of_day'].apply(_session_from_minute).astype(int)
    df['session_pos'] = df.apply(lambda row: _session_pos(row['minute_of_day'], row['session']), axis=1)
    df['sess_asia'] = (df['session'] == 0).astype(int)
    df['sess_eu'] = (df['session'] == 1).astype(int)
    df['sess_us'] = (df['session'] == 2).astype(int)

    feature_cols = ESSENTIAL_FEATURES + [
        'bb_up_20', 'bb_lo_20', 'ema20', 'ema50', 'ema200', 'macds',
        'pullback_time', 'vol_10', 'vol_20', 'vol_ratio', 'session_pos',
        'sess_asia', 'sess_eu', 'sess_us', 'volume_ratio', 'volume_sma20'
    ]

    df = df.dropna(subset=ESSENTIAL_FEATURES, how='any')

    return df[sorted(set(feature_cols) | set(df.columns))]

