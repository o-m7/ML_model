import pandas as pd
import numpy as np

# Load your data
df = pd.read_parquet('ML_model/ML_model/feature_store/XAUUSD/XAUUSD_1H.parquet')

print("AVAILABLE COLUMNS:")
print("="*60)
for i, col in enumerate(df.columns):
    print(f"{col}", end="  ")
    if (i + 1) % 5 == 0:
        print()
print(f"\n\nTotal: {len(df.columns)} columns\n")

print("="*60)
print("LEAKAGE DETECTION TEST")
print("="*60)

# Test RSI (your column is 'rsi')
if 'rsi' in df.columns:
    df['future_return_1'] = df['close'].shift(-1) / df['close'] - 1
    df['same_bar_return'] = df['close'] / df['open'] - 1
    df['past_return_1'] = df['close'] / df['close'].shift(1) - 1
    
    corr_same = df['rsi'].corr(df['same_bar_return'])
    corr_past = df['rsi'].corr(df['past_return_1'])
    corr_future = df['rsi'].corr(df['future_return_1'])
    
    print(f"\nRSI TIMING:")
    print(f"  vs SAME bar return:   {corr_same:>8.4f}  {'❌ LEAKAGE!' if abs(corr_same) > 0.05 else '✅ OK'}")
    print(f"  vs PAST bar return:   {corr_past:>8.4f}")
    print(f"  vs FUTURE bar return: {corr_future:>8.4f}")

# Test ATR
if 'atr' in df.columns:
    df['realized_vol_same'] = (df['high'] - df['low']) / df['close']
    df['realized_vol_past'] = ((df['high'].shift(1) - df['low'].shift(1)) / df['close'].shift(1))
    
    corr_same = df['atr'].corr(df['realized_vol_same'])
    corr_past = df['atr'].corr(df['realized_vol_past'])
    
    print(f"\nATR TIMING:")
    print(f"  vs SAME bar range:    {corr_same:>8.4f}  {'❌ LEAKAGE!' if corr_same > 0.7 else '✅ OK'}")
    print(f"  vs PAST bar range:    {corr_past:>8.4f}")

# Test Bollinger Bands
if 'bb_position' in df.columns:
    # BB position should be based on past SMA, not current close
    corr_same = df['bb_position'].corr(df['same_bar_return'])
    corr_future = df['bb_position'].corr(df['future_return_1'])
    
    print(f"\nBB_POSITION TIMING:")
    print(f"  vs SAME bar return:   {corr_same:>8.4f}  {'❌ LEAKAGE!' if abs(corr_same) > 0.1 else '✅ OK'}")
    print(f"  vs FUTURE bar return: {corr_future:>8.4f}")

# Test MACD
if 'macd' in df.columns:
    corr_same = df['macd'].corr(df['same_bar_return'])
    
    print(f"\nMACD TIMING:")
    print(f"  vs SAME bar return:   {corr_same:>8.4f}  {'❌ LEAKAGE!' if abs(corr_same) > 0.1 else '✅ OK'}")

print(f"\n{'='*60}")
print("INTERPRETATION:")
print("="*60)
print("If indicators show HIGH correlation with SAME bar returns:")
print("  → They INCLUDE current bar data (LEAKAGE)")
print("  → Solution: DON'T shift them again")
print("\nIf indicators show LOW correlation with SAME bar returns:")
print("  → They use PAST data only (CORRECT)")
print("  → Solution: Use as-is, don't shift")
print("\n❌ Your 34% WR suggests indicators are already correct")
print("   and we're OVER-SHIFTING them (making them use i-2 data)")