#!/usr/bin/env python3
"""
FIXED MULTI-TIMEFRAME TRAINING SYSTEM
======================================

Fixes for poor model performance:
1. Balanced long/short label creation
2. Improved feature engineering
3. Better model parameters to prevent directional bias
4. Comprehensive testing on all timeframes

Trains: 5T, 15T, 30T, 1H, 4H for XAUUSD
"""

import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import classification_report
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

# Import unified cost model
from market_costs import get_tp_sl, get_costs

load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    print("❌ ERROR: POLYGON_API_KEY not set in .env")
    sys.exit(1)

SYMBOL = 'XAUUSD'
TICKER = 'C:XAUUSD'
TIMEFRAMES = ['5T', '15T', '30T', '1H', '4H']

print("\n" + "="*80)
print(f"FIXED MULTI-TIMEFRAME TRAINING - {SYMBOL}")
print("="*80 + "\n")


def fetch_ohlcv_from_polygon(timeframe: str, days_back=730):
    """Fetch OHLCV data from Polygon API."""
    print(f"📡 Fetching {days_back} days of {timeframe} data from Polygon...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)

    # Map timeframe to Polygon multiplier
    tf_map = {'5T': 5, '15T': 15, '30T': 30, '1H': 60, '4H': 240}
    multiplier = tf_map.get(timeframe, 15)

    url = f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/{multiplier}/minute"
    url += f"/{start_time.strftime('%Y-%m-%d')}/{end_time.strftime('%Y-%m-%d')}"

    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': POLYGON_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get('status') != 'OK' or not data.get('results'):
            print(f"❌ No data from Polygon: {data.get('message')}")
            return None

        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms', utc=True)
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"✅ Fetched {len(df):,} bars ({df['timestamp'].min()} to {df['timestamp'].max()})")

        return df

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None


def calculate_features(df):
    """Calculate comprehensive technical indicators."""
    print("\n📊 Calculating features...")

    df = df.copy()

    # ATR (critical for labeling)
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(window=14).mean()
    df['atr20'] = tr.rolling(window=20).mean()

    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving averages (multiple timeframes)
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema{period}'] = df['close'].ewm(span=period).mean()
        df[f'close_vs_sma{period}'] = (df['close'] - df[f'sma{period}']) / df[f'sma{period}']
        df[f'close_vs_ema{period}'] = (df['close'] - df[f'ema{period}']) / df[f'ema{period}']

    # Momentum indicators
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
        df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)

    # RSI (multiple periods)
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi{period}'] = 100 - (100 / (1 + rs))

    # Bollinger Bands (multiple periods)
    for period in [10, 20, 30]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'bb_upper_{period}'] = sma + (std * 2)
        df[f'bb_lower_{period}'] = sma - (std * 2)
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / \
                                       (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Volume features
    df['volume_sma10'] = df['volume'].rolling(window=10).mean()
    df['volume_sma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma20'] + 1e-10)
    df['volume_std'] = df['volume'].rolling(window=20).std()

    # Price action features
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_open_diff'] = (df['close'] - df['open']) / df['open']

    # Candle patterns
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    df['body_size'] = body / df['close']
    df['upper_shadow_ratio'] = upper_shadow / (body + 1e-10)
    df['lower_shadow_ratio'] = lower_shadow / (body + 1e-10)

    # Trend strength
    df['adx'] = calculate_adx(df, period=14)

    # Volatility features
    for period in [10, 20, 50]:
        df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
        df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / \
                                             (df[f'volatility_{period}'].rolling(50).mean() + 1e-10)

    # Support/Resistance levels
    df['highest_high_20'] = df['high'].rolling(window=20).max()
    df['lowest_low_20'] = df['low'].rolling(window=20).min()
    df['dist_from_high'] = (df['highest_high_20'] - df['close']) / df['atr14']
    df['dist_from_low'] = (df['close'] - df['lowest_low_20']) / df['atr14']

    # Drop NaN rows
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = initial_len - len(df)

    print(f"✅ Calculated features, {len(df):,} bars remaining (dropped {dropped:,} NaN rows)")

    return df


def calculate_adx(df, period=14):
    """Calculate Average Directional Index."""
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()

    return adx


def create_balanced_labels(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Create BALANCED labels using proper long/short logic.

    Key fixes:
    1. Use NEXT bar open for entry (realistic)
    2. Check BOTH long and short opportunities for each bar
    3. Ensure balanced distribution
    """
    print(f"\n🏷️  Creating BALANCED labels for {symbol} {timeframe}...")

    df = df.copy()
    n = len(df)
    horizon = 50  # Look ahead 50 bars

    tp_sl_params = get_tp_sl(symbol, timeframe)
    tp_mult = tp_sl_params.tp_atr_mult
    sl_mult = tp_sl_params.sl_atr_mult

    print(f"   TP: {tp_mult:.1f}x ATR, SL: {sl_mult:.1f}x ATR (R:R = {tp_mult/sl_mult:.2f})")

    atr = df['atr14'].values

    # CRITICAL FIX: Entry at NEXT bar open!
    next_bar_opens = df['open'].shift(-1).values
    entry_prices = next_bar_opens

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # Initialize labels: 0=Flat, 1=Long, 2=Short
    labels = np.zeros(n, dtype=int)

    # Track statistics
    long_tp_wins = 0
    short_tp_wins = 0
    both_viable = 0

    for i in range(n - horizon - 1):
        if np.isnan(entry_prices[i]) or np.isnan(atr[i]):
            continue

        entry = entry_prices[i]

        # Define TP/SL for LONG
        tp_long = entry + (atr[i] * tp_mult)
        sl_long = entry - (atr[i] * sl_mult)

        # Define TP/SL for SHORT
        tp_short = entry - (atr[i] * tp_mult)
        sl_short = entry + (atr[i] * sl_mult)

        # Look ahead
        future_highs = highs[i+1:i+1+horizon]
        future_lows = lows[i+1:i+1+horizon]

        if len(future_highs) == 0:
            continue

        # Check LONG trade outcome
        tp_long_hits = np.where(future_highs >= tp_long)[0]
        sl_long_hits = np.where(future_lows <= sl_long)[0]

        long_wins = len(tp_long_hits) > 0 and (len(sl_long_hits) == 0 or tp_long_hits[0] < sl_long_hits[0])

        # Check SHORT trade outcome
        tp_short_hits = np.where(future_lows <= tp_short)[0]
        sl_short_hits = np.where(future_highs >= sl_short)[0]

        short_wins = len(tp_short_hits) > 0 and (len(sl_short_hits) == 0 or tp_short_hits[0] < sl_short_hits[0])

        # Label logic: Only label if ONE direction clearly wins
        if long_wins and not short_wins:
            labels[i] = 1  # Long
            long_tp_wins += 1
        elif short_wins and not long_wins:
            labels[i] = 2  # Short
            short_tp_wins += 1
        elif long_wins and short_wins:
            # Both directions win - choose the faster one
            long_bars = tp_long_hits[0]
            short_bars = tp_short_hits[0]
            if long_bars < short_bars:
                labels[i] = 1
                long_tp_wins += 1
            else:
                labels[i] = 2
                short_tp_wins += 1
            both_viable += 1
        # else: labels[i] = 0 (Flat) - no clear winner

    df['target'] = labels
    df = df.iloc[:-(horizon + 1)]

    # Show distribution
    counts = df['target'].value_counts().sort_index()
    total = len(df)
    flat_pct = counts.get(0, 0) / total * 100
    long_pct = counts.get(1, 0) / total * 100
    short_pct = counts.get(2, 0) / total * 100

    print(f"\n   Label Distribution:")
    print(f"   Flat:  {counts.get(0, 0):,} ({flat_pct:.1f}%)")
    print(f"   Long:  {counts.get(1, 0):,} ({long_pct:.1f}%)")
    print(f"   Short: {counts.get(2, 0):,} ({short_pct:.1f}%)")
    print(f"\n   Balance Check:")
    print(f"   Long TP wins: {long_tp_wins}")
    print(f"   Short TP wins: {short_tp_wins}")
    print(f"   Both viable (chose faster): {both_viable}")

    # Check for severe imbalance
    if long_pct < 5 or short_pct < 5:
        print(f"\n   ⚠️  WARNING: Severe imbalance detected!")
        print(f"   This may indicate a strong trending period in the data.")

    return df


class BalancedModel:
    """LightGBM model with balanced training to prevent directional bias."""

    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()

    def fit(self, X, y):
        """Train with carefully balanced class weights."""

        X_scaled = self.scaler.fit_transform(X)

        # Calculate balanced weights
        counts = np.bincount(y)
        print(f"\n   Class distribution in training set:")
        print(f"   Class 0 (Flat): {counts[0]:,}")
        print(f"   Class 1 (Long): {counts[1]:,}")
        print(f"   Class 2 (Short): {counts[2]:,}")

        # Balanced weights (inverse of frequency)
        weights = len(y) / (len(counts) * counts + 1e-10)

        # CRITICAL: Don't over-boost any class
        # Apply gentle boosting to minority classes only
        weights[0] *= 1.2  # Slight Flat boost (encourage selectivity)
        # Do NOT boost Long or Short differently to avoid bias

        sample_weight = weights[y]

        print(f"\n   Class weights: Flat={weights[0]:.2f}, Long={weights[1]:.2f}, Short={weights[2]:.2f}")

        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=200,  # Increased from 150
            max_depth=5,       # Slightly deeper
            learning_rate=0.03,  # Lower learning rate for stability
            num_leaves=16,     # More leaves for complexity
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=2.0,     # L1 regularization
            reg_lambda=3.0,    # L2 regularization
            min_child_samples=50,  # Increased to prevent overfitting
            random_state=42,
            verbosity=-1,
            force_row_wise=True,
            importance_type='gain'
        )

        self.model.fit(X_scaled, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def train_model(df, symbol, timeframe):
    """Train model with comprehensive evaluation."""
    print(f"\n🤖 Training model for {symbol} {timeframe}...")

    # Remove non-feature columns
    exclude_cols = ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].fillna(0).values
    y = df['target'].values

    # Check unique classes
    unique_classes = np.unique(y)
    print(f"\n   Unique classes in data: {unique_classes}")

    if len(unique_classes) < 3:
        print(f"   ⚠️  WARNING: Only {len(unique_classes)} classes found!")
        print(f"   Expected 3 classes (Flat, Long, Short)")

    # Split: 70% train, 30% test (more test data for validation)
    split_idx = int(len(X) * 0.70)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")

    # Train
    model = BalancedModel()
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"\n📊 Test Set Performance:")
    print("="*80)
    print(classification_report(y_test, y_pred, target_names=['Flat', 'Long', 'Short'], digits=3))

    # Check prediction distribution
    pred_counts = np.bincount(y_pred)
    print(f"\n   Prediction Distribution on Test Set:")
    print(f"   Flat:  {pred_counts[0]:,} ({pred_counts[0]/len(y_pred)*100:.1f}%)")
    print(f"   Long:  {pred_counts[1]:,} ({pred_counts[1]/len(y_pred)*100:.1f}%)")
    print(f"   Short: {pred_counts[2]:,} ({pred_counts[2]/len(y_pred)*100:.1f}%)")

    # Check for prediction bias
    long_short_ratio = pred_counts[1] / (pred_counts[2] + 1e-10)
    if long_short_ratio < 0.5 or long_short_ratio > 2.0:
        print(f"\n   ⚠️  WARNING: Prediction bias detected!")
        print(f"   Long/Short ratio: {long_short_ratio:.2f} (should be ~1.0)")
    else:
        print(f"\n   ✅ Predictions well-balanced (Long/Short ratio: {long_short_ratio:.2f})")

    # Feature importance
    if hasattr(model.model, 'feature_importances_'):
        importances = model.model.feature_importances_
        top_indices = np.argsort(importances)[-10:]
        print(f"\n   Top 10 Features:")
        for idx in reversed(top_indices):
            print(f"   {feature_cols[idx]:30s}: {importances[idx]:.4f}")

    # Save model
    model_dir = Path("models_institutional") / symbol
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    model_path = model_dir / f"{symbol}_{timeframe}_FIXED_{timestamp}.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': feature_cols,
            'class_names': ['Flat', 'Long', 'Short'],
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'trained_at': datetime.now(timezone.utc).isoformat(),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'num_classes': 3,
                'entry_method': 'next_bar_open_FIXED',
                'long_short_ratio': float(long_short_ratio)
            }
        }, f)

    print(f"\n✅ Model saved to: {model_path}")

    return model, feature_cols, model_path


def simple_backtest(df, model, features, symbol, timeframe):
    """
    Simple backtest to validate model generates BOTH long and short signals.
    """
    print(f"\n🔄 Running simple validation backtest...")

    X = df[features].fillna(0).values
    y_proba = model.predict_proba(X)

    # Get predictions with confidence threshold
    confidence_threshold = 0.50  # Lower threshold to generate more signals

    flat_probs = y_proba[:, 0]
    long_probs = y_proba[:, 1]
    short_probs = y_proba[:, 2]

    # Generate signals
    signals_long = []
    signals_short = []

    for i in range(len(df)):
        probs = [flat_probs[i], long_probs[i], short_probs[i]]
        max_prob = max(probs)
        pred_class = np.argmax(probs)

        if max_prob >= confidence_threshold:
            if pred_class == 1:  # Long
                signals_long.append(i)
            elif pred_class == 2:  # Short
                signals_short.append(i)

    print(f"\n📊 Signal Generation Test:")
    print(f"   Long signals:  {len(signals_long)}")
    print(f"   Short signals: {len(signals_short)}")
    print(f"   Total signals: {len(signals_long) + len(signals_short)}")

    if len(signals_long) == 0 or len(signals_short) == 0:
        print(f"\n   ❌ CRITICAL: Model is still biased!")
        print(f"   Model only generates {'LONG' if len(signals_short) == 0 else 'SHORT'} signals")
        return False
    else:
        ratio = len(signals_long) / (len(signals_short) + 1e-10)
        print(f"   ✅ Model generates both directions (L/S ratio: {ratio:.2f})")
        return True


def main():
    """Train models for all timeframes."""

    results = []

    for timeframe in TIMEFRAMES:
        print("\n" + "="*80)
        print(f"TRAINING: {SYMBOL} {timeframe}")
        print("="*80)

        try:
            # Fetch data
            df = fetch_ohlcv_from_polygon(timeframe, days_back=730)
            if df is None or len(df) < 1000:
                print(f"❌ Insufficient data for {timeframe}")
                continue

            # Calculate features
            df = calculate_features(df)

            # Create balanced labels
            df = create_balanced_labels(df, SYMBOL, timeframe)

            # Train model
            model, features, model_path = train_model(df, SYMBOL, timeframe)

            # Validate signal generation
            is_balanced = simple_backtest(df, model, features, SYMBOL, timeframe)

            results.append({
                'timeframe': timeframe,
                'success': True,
                'balanced': is_balanced,
                'model_path': model_path
            })

            print(f"\n{'='*80}")
            print(f"✅ {timeframe} TRAINING COMPLETE")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n❌ ERROR training {timeframe}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'timeframe': timeframe,
                'success': False,
                'error': str(e)
            })

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    for r in results:
        tf = r['timeframe']
        if r['success']:
            status = "✅ SUCCESS" if r.get('balanced', False) else "⚠️  TRAINED (check bias)"
            print(f"{tf:6s}: {status}")
        else:
            print(f"{tf:6s}: ❌ FAILED - {r.get('error', 'Unknown error')}")

    print("="*80)
    print("\nNext steps:")
    print("  1. Run validation: python validate_backtest_with_costs.py --symbol XAUUSD --tf 15T")
    print("  2. Test signal generation: python signal_generator.py")
    print("\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
