"""
Feature engineering module for trading system.

Implements technical indicators, regime detection, and target construction
with strict no-lookahead guarantees.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


def compute_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
    """Compute log returns over multiple periods."""
    for period in periods:
        df[f'return_{period}'] = np.log(df['close'] / df['close'].shift(period))
    return df


def compute_moving_averages(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """Compute EMAs and SMAs."""
    for period in periods:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

        # MA slopes (normalized by ATR)
        if f'atr_14' in df.columns:
            df[f'ema_{period}_slope'] = (df[f'ema_{period}'] - df[f'ema_{period}'].shift(5)) / df['atr_14']

    return df


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD indicator."""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()

    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df


def compute_rsi(df: pd.DataFrame, periods: List[int] = [14]) -> pd.DataFrame:
    """Compute RSI indicator."""
    for period in periods:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    return df


def compute_atr(df: pd.DataFrame, periods: List[int] = [14]) -> pd.DataFrame:
    """Compute Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    for period in periods:
        df[f'atr_{period}'] = tr.rolling(window=period).mean()

        # Relative ATR (ATR / price)
        df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close'] * 100

    return df


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    bb_std = df['close'].rolling(window=period).std()

    df['bb_upper'] = df['bb_middle'] + (std_dev * bb_std)
    df['bb_lower'] = df['bb_middle'] - (std_dev * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # Position within bands
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    return df


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute ADX (Average Directional Index)."""
    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed TR, +DM, -DM
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(window=period).mean()

    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    return df


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Compute Stochastic Oscillator."""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()

    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

    return df


def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Compute Commodity Channel Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)

    df['cci'] = (tp - sma_tp) / (0.015 * mad)

    return df


def compute_volatility_features(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Compute volatility-based features."""
    # Realized volatility
    df['realized_vol'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    # ATR percentile
    if 'atr_14' in df.columns:
        df['atr_percentile'] = df['atr_14'].rolling(window=window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

    # High-low range
    df['hl_range'] = (df['high'] - df['low']) / df['close']

    return df


def compute_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price structure features."""
    # Distance to recent highs/lows
    df['dist_to_20h'] = (df['close'] - df['high'].rolling(20).max()) / df['close']
    df['dist_to_20l'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
    df['dist_to_50h'] = (df['close'] - df['high'].rolling(50).max()) / df['close']
    df['dist_to_50l'] = (df['close'] - df['low'].rolling(50).min()) / df['close']

    # Distance to MAs
    if 'ema_20' in df.columns:
        df['dist_to_ema20'] = (df['close'] - df['ema_20']) / df['close']
    if 'ema_50' in df.columns:
        df['dist_to_ema50'] = (df['close'] - df['ema_50']) / df['close']

    # Price momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    return df


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-based features."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # Session flags (UTC times, adjust for your broker's timezone)
    # London: 8-17 UTC, NY: 13-22 UTC
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
    df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] < 17)).astype(int)

    # Weekend gap flag
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)

    return df


def detect_regime(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Detect market regime: trend_up, trend_down, range, event_vol.

    Uses ADX, MA alignment, and volatility percentiles.
    Strictly backward-looking.
    """
    regime = pd.Series('unknown', index=df.index)

    # Ensure required columns exist
    if 'adx' not in df.columns or 'atr_percentile' not in df.columns:
        df['regime'] = regime
        return df

    adx = df['adx']
    atr_pct = df['atr_percentile']

    # MA alignment for trend direction
    ma_periods = config.regime.ma_alignment_periods
    if all(f'ema_{p}' in df.columns for p in ma_periods):
        trend_up_ma = (df[f'ema_{ma_periods[0]}'] > df[f'ema_{ma_periods[1]}']) & \
                      (df[f'ema_{ma_periods[1]}'] > df[f'ema_{ma_periods[2]}'])
        trend_down_ma = (df[f'ema_{ma_periods[0]}'] < df[f'ema_{ma_periods[1]}']) & \
                        (df[f'ema_{ma_periods[1]}'] < df[f'ema_{ma_periods[2]}'])
    else:
        trend_up_ma = df['close'] > df['close'].rolling(50).mean()
        trend_down_ma = df['close'] < df['close'].rolling(50).mean()

    # Event / high volatility regime
    event_vol_mask = atr_pct > config.regime.vol_high_threshold

    # Trending regimes
    trending_mask = adx > config.regime.adx_trend_threshold
    trend_up_mask = trending_mask & trend_up_ma & ~event_vol_mask
    trend_down_mask = trending_mask & trend_down_ma & ~event_vol_mask

    # Range regime
    range_mask = (adx < config.regime.adx_trend_threshold) & ~event_vol_mask

    # Assign regimes
    regime[event_vol_mask] = 'event_vol'
    regime[trend_up_mask] = 'trend_up'
    regime[trend_down_mask] = 'trend_down'
    regime[range_mask] = 'range'

    df['regime'] = regime

    # One-hot encode regimes for ML
    regime_dummies = pd.get_dummies(df['regime'], prefix='regime')
    df = pd.concat([df, regime_dummies], axis=1)

    return df


def build_news_features(df: pd.DataFrame, news_data_path: str) -> pd.DataFrame:
    """
    Build news and event-based features.

    Assumes news_data has columns: ['timestamp', 'impact_level', 'surprise']
    impact_level: 'low', 'medium', 'high'
    surprise: numeric value representing deviation from forecast
    """
    try:
        news_df = pd.read_csv(news_data_path, parse_dates=['timestamp'])
        news_df = news_df.sort_values('timestamp')

        # Binary flags
        df['high_impact_event_next_30m'] = 0
        df['high_impact_event_last_30m'] = 0
        df['time_to_next_event'] = np.nan
        df['time_since_last_event'] = np.nan

        # For each bar, find nearest events
        for idx, row in df.iterrows():
            current_time = idx

            # Next high-impact event
            future_events = news_df[(news_df['timestamp'] > current_time) &
                                   (news_df['impact_level'] == 'high')]
            if len(future_events) > 0:
                next_event = future_events.iloc[0]
                minutes_to_event = (next_event['timestamp'] - current_time).total_seconds() / 60
                df.loc[idx, 'time_to_next_event'] = minutes_to_event
                if minutes_to_event <= 30:
                    df.loc[idx, 'high_impact_event_next_30m'] = 1

            # Last high-impact event
            past_events = news_df[(news_df['timestamp'] <= current_time) &
                                 (news_df['impact_level'] == 'high')]
            if len(past_events) > 0:
                last_event = past_events.iloc[-1]
                minutes_since_event = (current_time - last_event['timestamp']).total_seconds() / 60
                df.loc[idx, 'time_since_last_event'] = minutes_since_event
                if minutes_since_event <= 30:
                    df.loc[idx, 'high_impact_event_last_30m'] = 1

        # Fill NaNs
        df['time_to_next_event'].fillna(1440, inplace=True)  # 1 day if no event
        df['time_since_last_event'].fillna(1440, inplace=True)

    except Exception as e:
        print(f"News features not available: {e}")
        df['high_impact_event_next_30m'] = 0
        df['high_impact_event_last_30m'] = 0
        df['time_to_next_event'] = 1440
        df['time_since_last_event'] = 1440

    return df


def build_targets(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Build trade targets: binary label indicating if TP hit before SL.

    For each bar, simulate a hypothetical trade based on regime:
    - trend_up: long entry
    - trend_down: short entry
    - range: fade extremes (context-dependent)

    Then check forward if TP or SL is hit first.
    """
    df['target'] = np.nan
    df['entry_price'] = df['close']

    sl_atr_mult = config.strategy.sl_atr_multiplier
    tp_atr_mult = config.strategy.tp_atr_multiplier

    if 'atr_14' not in df.columns:
        print("Warning: ATR not found, cannot build targets")
        return df

    atr = df['atr_14']

    for i in range(len(df) - 1):
        regime = df['regime'].iloc[i]
        entry = df['close'].iloc[i]
        atr_val = atr.iloc[i]

        if pd.isna(atr_val) or atr_val == 0:
            continue

        # Determine trade direction based on regime
        if regime == 'trend_up':
            direction = 'long'
        elif regime == 'trend_down':
            direction = 'short'
        elif regime == 'range':
            # Fade extremes: if price at upper BB, short; if at lower BB, long
            if 'bb_position' in df.columns:
                bb_pos = df['bb_position'].iloc[i]
                if bb_pos > 0.9:
                    direction = 'short'
                elif bb_pos < 0.1:
                    direction = 'long'
                else:
                    continue  # No trade in middle of range
            else:
                continue
        else:
            continue  # No trade in event_vol or unknown regimes

        # Set SL and TP
        if direction == 'long':
            sl = entry - sl_atr_mult * atr_val
            tp = entry + tp_atr_mult * atr_val
        else:  # short
            sl = entry + sl_atr_mult * atr_val
            tp = entry - tp_atr_mult * atr_val

        # Check forward bars to see if TP or SL hit first
        # Limit lookforward to reasonable window (e.g., 100 bars)
        lookforward = min(100, len(df) - i - 1)

        tp_hit = False
        sl_hit = False

        for j in range(1, lookforward + 1):
            future_high = df['high'].iloc[i + j]
            future_low = df['low'].iloc[i + j]

            if direction == 'long':
                if future_low <= sl:
                    sl_hit = True
                    break
                if future_high >= tp:
                    tp_hit = True
                    break
            else:  # short
                if future_high >= sl:
                    sl_hit = True
                    break
                if future_low <= tp:
                    tp_hit = True
                    break

        # Label: 1 if TP hit first, 0 if SL hit first
        if tp_hit:
            df.loc[df.index[i], 'target'] = 1
        elif sl_hit:
            df.loc[df.index[i], 'target'] = 0
        # else: neither hit within lookforward window, leave as NaN

    return df


def build_features(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Master feature engineering function.

    Computes all technical indicators, regime labels, and targets.
    Ensures no lookahead bias.
    """
    df = df.copy()

    # Sort by time
    df = df.sort_index()

    # 1. Basic returns
    df = compute_returns(df)

    # 2. ATR (needed for many features)
    df = compute_atr(df, periods=config.features.atr_periods)

    # 3. Moving averages
    df = compute_moving_averages(df, periods=config.features.ma_periods)

    # 4. MACD
    df = compute_macd(df,
                      fast=config.features.macd_fast,
                      slow=config.features.macd_slow,
                      signal=config.features.macd_signal)

    # 5. RSI
    df = compute_rsi(df, periods=config.features.rsi_periods)

    # 6. Bollinger Bands
    df = compute_bollinger_bands(df,
                                 period=config.features.bb_period,
                                 std_dev=config.features.bb_std)

    # 7. ADX
    df = compute_adx(df, period=config.features.adx_period)

    # 8. Stochastic
    df = compute_stochastic(df)

    # 9. CCI
    df = compute_cci(df)

    # 10. Volatility features
    df = compute_volatility_features(df, window=config.features.vol_percentile_window)

    # 11. Structure features
    df = compute_structure_features(df)

    # 12. Time features
    df = compute_time_features(df)

    # 13. Regime detection
    df = detect_regime(df, config)

    # 14. News features (optional)
    if config.features.use_news_features:
        df = build_news_features(df, str(config.features.news_data_path))

    # 15. Build targets
    df = build_targets(df, config)

    # Drop rows with NaN in target or critical features
    df = df.dropna(subset=['target'])

    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """
    Get list of feature columns for ML, excluding OHLCV, target, and metadata.
    """
    if exclude_cols is None:
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                       'target', 'entry_price', 'regime']

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols


if __name__ == "__main__":
    # Test feature engineering
    from config import get_default_config

    # Create dummy data
    dates = pd.date_range('2020-01-01', periods=5000, freq='5min')
    np.random.seed(42)

    df = pd.DataFrame({
        'open': 1800 + np.random.randn(5000).cumsum(),
        'high': 1800 + np.random.randn(5000).cumsum() + 5,
        'low': 1800 + np.random.randn(5000).cumsum() - 5,
        'close': 1800 + np.random.randn(5000).cumsum(),
        'volume': np.random.randint(1000, 10000, 5000)
    }, index=dates)

    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    config = get_default_config()

    print("Building features...")
    df_features = build_features(df, config)

    print(f"Original shape: {df.shape}")
    print(f"After features: {df_features.shape}")
    print(f"\nRegime distribution:\n{df_features['regime'].value_counts()}")
    print(f"\nTarget distribution:\n{df_features['target'].value_counts()}")
    print(f"\nSample features:\n{df_features.head()}")
