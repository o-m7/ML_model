#!/usr/bin/env python3
"""Quick data diagnostic script."""

import pandas as pd
from pathlib import Path

data_path = Path("feature_store/XAUUSD/XAUUSD_15T.parquet")

if data_path.exists():
    df = pd.read_parquet(data_path)

    print("📊 Data File Analysis")
    print("=" * 60)
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"File size: {data_path.stat().st_size / 1024:.1f} KB")

    # Check for timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        total_days = (max_date - min_date).days

        print(f"\n📅 Date Range:")
        print(f"   Start: {min_date}")
        print(f"   End:   {max_date}")
        print(f"   Days:  {total_days}")
        print(f"   Months: {total_days / 30:.1f}")

        print(f"\n📈 Data Density:")
        print(f"   Bars per day: {len(df) / max(total_days, 1):.0f}")
        print(f"   Expected (15min): ~96 bars/day")

        print(f"\n💡 Walk-Forward Validation:")
        print(f"   Default config needs: 6 months train + 1 month test = 7 months")
        print(f"   You have: {total_days / 30:.1f} months")

        if total_days < 210:  # 7 months
            print(f"\n   ⚠️  INSUFFICIENT DATA for default config")
            print(f"\n   Recommended adjustments:")
            print(f"   1. train_months=2, test_months=1 (need 3 months)")
            print(f"   2. train_months=1, test_months=1 (need 2 months)")
            print(f"   3. Download more historical data")
        else:
            expected_segments = (total_days - 180) // 30
            print(f"\n   ✅ Sufficient data")
            print(f"   Expected segments: ~{expected_segments}")
    else:
        print("\n⚠️  No 'timestamp' column found")
        print(f"Available columns: {list(df.columns)[:10]}")

        if isinstance(df.index, pd.DatetimeIndex):
            print("\n   Timestamp is in index (this is OK)")
            min_date = df.index.min()
            max_date = df.index.max()
            total_days = (max_date - min_date).days
            print(f"   Date range: {min_date} to {max_date}")
            print(f"   Total days: {total_days} ({total_days/30:.1f} months)")
else:
    print(f"❌ File not found: {data_path}")
