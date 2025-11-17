#!/usr/bin/env python3
"""
MINIMAL DEBUG SCRIPT - Run ONE segment and show EVERYTHING

This will show EXACTLY what's happening:
1. Load ONE segment of data
2. Create labels EXACTLY like main pipeline
3. Train model EXACTLY like main pipeline
4. Generate predictions
5. Backtest those predictions with EXACT position sizing
6. Show every trade with entry/exit/PnL

If this fails, we'll see WHERE and WHY.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# Top 20 features from diagnostic
TOP_FEATURES = [
    'roc_3', 'price_vs_vwma_10', 'stoch_k', 'macd', 'correlation_20',
    'macd_signal', 'price_vs_vwma_50', 'price_vs_vwma_20', 'roc_10',
    'vwma_20', 'bb_width_20', 'distance_from_ma_100', 'zscore_100',
    'bb_width_50', 'volume_ratio_20', 'bb_position_20', 'mfi',
    'volume_ratio_10', 'price_ratio', 'vwap'
]

def load_data():
    """Load data from parquet."""
    feature_store = Path("feature_store")
    data_path = feature_store / "XAUUSD" / "XAUUSD_15T.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_parquet(data_path)
    print(f"Loaded: {len(df):,} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

def create_tpsl_labels(df, tp_atr_mult=2.0, sl_atr_mult=1.0, max_bars=8):
    """Create TP/SL labels EXACTLY like main pipeline."""
    df = df.copy()

    # Calculate ATR if not present
    if 'atr_14' not in df.columns:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()

    # Forward returns for both long and short
    df['forward_return_long'] = np.nan
    df['forward_return_short'] = np.nan

    for i in range(len(df) - max_bars):
        entry_price = df.iloc[i]['close']
        atr = df.iloc[i]['atr_14']

        if pd.isna(atr) or atr <= 0:
            continue

        # Long position
        tp_long = entry_price + (tp_atr_mult * atr)
        sl_long = entry_price - (sl_atr_mult * atr)

        # Short position
        tp_short = entry_price - (tp_atr_mult * atr)
        sl_short = entry_price + (sl_atr_mult * atr)

        # Check next max_bars
        for j in range(1, max_bars + 1):
            if i + j >= len(df):
                break

            future_high = df.iloc[i + j]['high']
            future_low = df.iloc[i + j]['low']

            # Long TP/SL check
            if pd.isna(df.iloc[i]['forward_return_long']):
                if future_high >= tp_long:
                    df.iloc[i, df.columns.get_loc('forward_return_long')] = (tp_long - entry_price) / entry_price
                elif future_low <= sl_long:
                    df.iloc[i, df.columns.get_loc('forward_return_long')] = (sl_long - entry_price) / entry_price

            # Short TP/SL check
            if pd.isna(df.iloc[i]['forward_return_short']):
                if future_low <= tp_short:
                    df.iloc[i, df.columns.get_loc('forward_return_short')] = (entry_price - tp_short) / entry_price
                elif future_high >= sl_short:
                    df.iloc[i, df.columns.get_loc('forward_return_short')] = (entry_price - sl_short) / entry_price

    # Target: 1 if EITHER long or short TP hit (profitable setup)
    df['target'] = 0
    df.loc[(df['forward_return_long'] > 0) | (df['forward_return_short'] > 0), 'target'] = 1

    return df

def prepare_features(df, feature_list):
    """Prepare features - use only features that exist."""
    available_features = [f for f in feature_list if f in df.columns]
    print(f"\nFeatures: {len(available_features)}/{len(feature_list)} available")

    if len(available_features) < 10:
        print(f"WARNING: Only {len(available_features)} features found!")
        print(f"Missing: {set(feature_list) - set(available_features)}")

    X = df[available_features].copy()

    # Fill NaN
    for col in X.columns:
        X[col] = X[col].ffill().bfill().fillna(0)

    y = df['target'].values

    return X.values, y, available_features

def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost EXACTLY like main pipeline."""
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        random_state=42
    )

    model.fit(X_train, y_train, verbose=False)

    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_pred) if len(np.unique(y_train)) > 1 else 0.5
    test_auc = roc_auc_score(y_test, test_pred) if len(np.unique(y_test)) > 1 else 0.5

    print(f"\nModel Performance:")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Test AUC:  {test_auc:.4f}")
    print(f"  Label distribution: {(y_test == 1).sum()}/{len(y_test)} positive ({100*(y_test == 1).sum()/len(y_test):.1f}%)")

    return model, test_pred

def backtest_trades(test_df, predictions, threshold=0.6):
    """Backtest trades with EXACT position sizing."""
    test_df = test_df.copy()
    test_df['prediction'] = predictions

    # Filter by threshold (top 40%)
    threshold_value = np.quantile(predictions, threshold)
    test_df['signal'] = test_df['prediction'] >= threshold_value

    trade_rows = test_df[test_df['signal']].copy()
    print(f"\nTrade Signals: {len(trade_rows)} (top {100*(1-threshold):.0f}% of predictions)")

    if len(trade_rows) == 0:
        print("❌ NO TRADES - threshold too strict!")
        return None

    # Simulate trades
    trades = []
    capital = 25000.0
    risk_pct = 0.015  # 1.5%

    for idx, row in trade_rows.iterrows():
        atr = row.get('atr_14', 0)
        if pd.isna(atr) or atr <= 0:
            continue

        entry_price = row['close']

        # Determine direction (choose better of long/short)
        long_return = row.get('forward_return_long', np.nan)
        short_return = row.get('forward_return_short', np.nan)

        if pd.isna(long_return) and pd.isna(short_return):
            continue

        # Pick the direction that hit TP (or lesser loss if both hit SL)
        if not pd.isna(long_return) and not pd.isna(short_return):
            direction = 'LONG' if long_return > short_return else 'SHORT'
            pnl_pct = long_return if direction == 'LONG' else short_return
        elif not pd.isna(long_return):
            direction = 'LONG'
            pnl_pct = long_return
        else:
            direction = 'SHORT'
            pnl_pct = short_return

        # Position sizing
        sl_distance = 1.0 * atr  # SL = 1R
        risk_amount = capital * risk_pct
        position_size_lots = risk_amount / (sl_distance * 100)  # 1 lot = 100oz, $1 move = $100
        position_size_lots = min(position_size_lots, 1.0)  # Max 1 lot

        # Calculate P&L
        dollar_move = entry_price * pnl_pct
        pnl_dollars = dollar_move * 100 * position_size_lots

        # Update capital
        capital += pnl_dollars

        trades.append({
            'timestamp': row['timestamp'],
            'direction': direction,
            'entry': entry_price,
            'position_lots': position_size_lots,
            'pnl_pct': pnl_pct * 100,
            'pnl_dollars': pnl_dollars,
            'capital': capital,
            'prediction': row['prediction']
        })

    if not trades:
        print("❌ NO VALID TRADES")
        return None

    # Print trades
    trades_df = pd.DataFrame(trades)
    print(f"\n{'='*80}")
    print(f"BACKTEST RESULTS - {len(trades)} TRADES")
    print(f"{'='*80}")

    print(f"\nFirst 10 trades:")
    print(trades_df.head(10)[['timestamp', 'direction', 'entry', 'position_lots', 'pnl_dollars', 'capital']].to_string(index=False))

    # Summary
    winning_trades = trades_df[trades_df['pnl_dollars'] > 0]
    losing_trades = trades_df[trades_df['pnl_dollars'] < 0]

    total_pnl = trades_df['pnl_dollars'].sum()
    win_rate = len(winning_trades) / len(trades) * 100

    avg_win = winning_trades['pnl_dollars'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_dollars'].mean() if len(losing_trades) > 0 else 0

    profit_factor = abs(winning_trades['pnl_dollars'].sum() / losing_trades['pnl_dollars'].sum()) if len(losing_trades) > 0 and losing_trades['pnl_dollars'].sum() != 0 else np.inf

    final_capital = 25000 + total_pnl
    total_return_pct = (final_capital / 25000 - 1) * 100

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total Trades:    {len(trades)}")
    print(f"Winning Trades:  {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"Losing Trades:   {len(losing_trades)}")
    print(f"")
    print(f"Avg Win:         ${avg_win:,.2f}")
    print(f"Avg Loss:        ${avg_loss:,.2f}")
    print(f"Profit Factor:   {profit_factor:.2f}")
    print(f"")
    print(f"Total P&L:       ${total_pnl:,.2f}")
    print(f"Final Capital:   ${final_capital:,.2f}")
    print(f"Total Return:    {total_return_pct:+.2f}%")
    print(f"{'='*80}")

    if total_return_pct < -20:
        print(f"\n❌ MAJOR LOSS: {total_return_pct:.1f}%")
        print(f"Something is BROKEN. Investigate:")
        print(f"  1. Are labels correct? (TP/SL logic)")
        print(f"  2. Is position sizing correct?")
        print(f"  3. Are predictions inversely correlated with outcomes?")

    return trades_df

def main():
    print("="*80)
    print("DEBUG ONE SEGMENT - SHOW EVERYTHING")
    print("="*80)

    # 1. Load data
    print("\n[1] Loading data...")
    df = load_data()

    # Use middle segment (has quote features and is stable)
    start_idx = 60000
    end_idx = 90000
    segment = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    print(f"Using segment: rows {start_idx}-{end_idx} ({len(segment):,} bars)")
    print(f"Date range: {segment['timestamp'].min()} to {segment['timestamp'].max()}")

    # 2. Create labels
    print("\n[2] Creating TP/SL labels...")
    segment = create_tpsl_labels(segment, tp_atr_mult=2.0, sl_atr_mult=1.0, max_bars=8)

    valid_labels = segment['target'].notna()
    print(f"Valid labels: {valid_labels.sum():,}/{len(segment):,}")
    print(f"Positive (TP hit): {(segment['target'] == 1).sum():,} ({100*(segment['target'] == 1).sum()/valid_labels.sum():.1f}%)")

    segment = segment[valid_labels].copy()

    # 3. Split train/test
    split = int(len(segment) * 0.7)
    train_df = segment.iloc[:split].copy()
    test_df = segment.iloc[split:].copy()
    print(f"\nTrain: {len(train_df):,} bars")
    print(f"Test:  {len(test_df):,} bars")

    # 4. Prepare features
    print("\n[3] Preparing features...")
    X_train, y_train, features = prepare_features(train_df, TOP_FEATURES)
    X_test, y_test, _ = prepare_features(test_df, TOP_FEATURES)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    # 5. Train model
    print("\n[4] Training model...")
    model, predictions = train_model(X_train, y_train, X_test, y_test)

    # 6. Backtest
    print("\n[5] Backtesting trades...")
    trades = backtest_trades(test_df, predictions, threshold=0.60)

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
