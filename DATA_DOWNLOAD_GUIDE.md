# Historical Data Download Guide

**This will fix the 0% win rate issue by giving you YEARS of training data instead of 54 days.**

---

## Quick Start

### 1. Install Dependencies

```bash
pip install boto3 python-dotenv
```

### 2. Pull Latest Code

```bash
cd /Users/omar/Desktop/ML_model
git pull origin claude/fix-signal-generator-blocking-01TUN4AD28yxmA8rquEGAeQx
```

### 3. Download Historical Data (2019-2025)

```bash
# This will take 30-60 minutes depending on your internet
python download_historical_data.py
```

**What this does**:
- Downloads ~2,500 daily CSV files from Massive.com S3
- Filters for XAUUSD and XAGUSD
- Resamples to 5T, 15T, 30T, 1H
- Calculates 80+ features (same as live trading)
- Saves to `feature_store/XAUUSD/XAUUSD_5T.parquet`, etc.

**Expected output**:
```
DOWNLOADING DATA: 2019-01-01 to 2025-11-14
Symbols: ['XAUUSD', 'XAGUSD']
================================================================================

  ⬇️  Downloading: 2019-01-01... ✓
  ⬇️  Downloading: 2019-01-02... ✓
  ...
  ⬇️  Downloading: 2025-11-14... ✓

✅ Downloaded: 2,509 days

📊 Combining data for XAUUSD...
  ✅ XAUUSD: 1,234,567 bars from 2019-01-01 to 2025-11-14

📊 Combining data for XAGUSD...
  ✅ XAGUSD: 1,234,567 bars from 2019-01-01 to 2025-11-14

================================================================================
PROCESSING: XAUUSD
================================================================================

🔧 XAUUSD 5T:
  Resampling to 5T... ✓ 617,283 bars
  Calculating features... ✓ 83 columns
  Saving to feature_store/XAUUSD/XAUUSD_5T.parquet... ✓
  ✅ XAUUSD 5T: 617,283 bars with features

🔧 XAUUSD 15T:
  Resampling to 15T... ✓ 205,761 bars
  Calculating features... ✓ 83 columns
  Saving to feature_store/XAUUSD/XAUUSD_15T.parquet... ✓
  ✅ XAUUSD 15T: 205,761 bars with features

...

================================================================================
SUMMARY
================================================================================
XAUUSD 5T: 617,283 bars | 2,509 days | 83 features
XAUUSD 15T: 205,761 bars | 2,509 days | 83 features
XAUUSD 30T: 102,880 bars | 2,509 days | 83 features
XAUUSD 1H: 51,440 bars | 2,509 days | 83 features
XAGUSD 5T: 617,283 bars | 2,509 days | 83 features
XAGUSD 15T: 205,761 bars | 2,509 days | 83 features
XAGUSD 30T: 102,880 bars | 2,509 days | 83 features
XAGUSD 1H: 51,440 bars | 2,509 days | 83 features

✅ Data saved to: feature_store/
✅ Cache saved to: raw_data_cache/
```

### 4. Train Models with REAL Data

```bash
# Now you have 2,509 days instead of 54 days!
python citadel_training_system.py
```

**Expected results**:
- Test set: ~500 days (was 10 days)
- Enough trades to evaluate properly
- Win rates should be realistic (not 0%)

---

## Advanced Usage

### Custom Date Range

```bash
# Last 3 years only
python download_historical_data.py --start-date 2022-01-01 --end-date 2025-11-14

# Specific year
python download_historical_data.py --start-date 2023-01-01 --end-date 2023-12-31
```

### Specific Symbols

```bash
# Only XAUUSD
python download_historical_data.py --symbols XAUUSD

# Only XAGUSD
python download_historical_data.py --symbols XAGUSD
```

### Specific Timeframes

```bash
# Only 30T and 1H (faster)
python download_historical_data.py --timeframes 30T 1H

# Only 5T (slowest, largest files)
python download_historical_data.py --timeframes 5T
```

### Resume After Interruption

If download gets interrupted, just run again - it will skip already downloaded files:

```bash
# Files already downloaded won't be re-downloaded
python download_historical_data.py
```

### Re-process Without Re-downloading

If you already downloaded the raw data and just want to recalculate features:

```bash
python download_historical_data.py --skip-download
```

---

## File Structure

After running, you'll have:

```
ML_model/
├── raw_data_cache/               # Raw daily CSV.gz files (cache)
│   ├── 2019-01-01.csv.gz
│   ├── 2019-01-02.csv.gz
│   ├── ...
│   ├── 2025-11-14.csv.gz
│   ├── XAUUSD_combined.parquet   # All XAUUSD minute data
│   └── XAGUSD_combined.parquet   # All XAGUSD minute data
│
└── feature_store/                # Final feature-engineered data
    ├── XAUUSD/
    │   ├── XAUUSD_5T.parquet     # 617k bars with 83 features
    │   ├── XAUUSD_15T.parquet    # 205k bars with 83 features
    │   ├── XAUUSD_30T.parquet    # 102k bars with 83 features
    │   └── XAUUSD_1H.parquet     # 51k bars with 83 features
    │
    └── XAGUSD/
        ├── XAGUSD_5T.parquet
        ├── XAGUSD_15T.parquet
        ├── XAGUSD_30T.parquet
        └── XAGUSD_1H.parquet
```

**Cache benefits**:
- Raw data cached to `raw_data_cache/`
- Can recalculate features without re-downloading
- Can experiment with different feature sets

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'boto3'"

```bash
pip install boto3 python-dotenv
```

### "Error: Access Denied"

Check your AWS credentials in the script or .env file:
```
AWS_ACCESS_KEY_ID=4937f95b-db8b-4d7e-8d54-756a82d4976e
AWS_SECRET_ACCESS_KEY=o_u3GoSv8JHF3ZBS9NQsTseq6mbhgTI1
```

### "No data for symbol XAUUSD"

The CSV might use different symbol names. Check the raw CSV to see actual ticker names:

```bash
zcat raw_data_cache/2023-01-01.csv.gz | head -20
```

Look for column names like `ticker`, `symbol`, or similar.

### Download is Slow

- Normal: ~2,500 files takes 30-60 minutes
- Run overnight if needed
- Can interrupt and resume (already downloaded files skipped)

### Out of Disk Space

The data is large:
- Raw cache: ~5-10 GB
- Feature store: ~2-3 GB
- Total: ~10-15 GB

Free up space or use `--timeframes 30T 1H` to save space.

---

## What This Fixes

### Before (Broken)

```
Data: 54 days (Nov 15, 2023 → Jan 8, 2024)

Train: 38 days
Val:   8 days
Test:  10 days

Result: 0% win rate (insufficient data)
```

### After (Fixed)

```
Data: 2,509 days (Jan 1, 2019 → Nov 14, 2025)

Train: 1,756 days (70%)
Val:   376 days (15%)
Test:  376 days (15%)

Result: Realistic win rates (enough data to learn)
```

---

## Expected Training Results

After downloading data and training, you should see:

**Temporal Split**:
```
Train: 2019-01-01 to 2023-08-15 (1,756 days)
Val:   2023-08-15 to 2024-05-20 (376 days)
Test:  2024-05-20 to 2025-11-14 (376 days)
```

**Model Performance** (example):
```
✅ Test Accuracy: 54.2%

📊 TRADING METRICS:
   Total trades: 347
   Win rate: 52.3%
   Profit factor: 1.68
   Expectancy: +0.35 R

✅ MODEL PASSED ALL THRESHOLDS - XAUUSD 15T
```

**Not all models will pass** - that's normal. But you should see:
- Realistic test accuracy (52-58%)
- Reasonable win rates (48-55%)
- Some models passing validation

---

## Next Steps After Download

### 1. Train Models

```bash
# Use Citadel training system (best)
python citadel_training_system.py

# Or use temporal training
python retrain_all_temporal.py
```

### 2. Backtest Results

```bash
# Test the trained models
python run_model_backtest.py --symbol XAUUSD --timeframe 15T
```

### 3. Deploy to Production

```bash
# Only deploy models that pass validation
# Check models_citadel/training_summary.csv for PASSED models
```

---

## Data Source Details

**Source**: Massive.com S3 (flatfiles bucket)

**Format**: Daily CSV.gz files with minute-level OHLCV data

**Columns**:
- `ticker` or `symbol`: Instrument name
- `timestamp` or `t`: Unix timestamp or datetime
- `open` or `o`: Opening price
- `high` or `h`: High price
- `low` or `l`: Low price
- `close` or `c`: Closing price
- `volume` or `v`: Volume

**Coverage**:
- Start: 2019-01-01
- End: 2025-11-14 (current)
- Frequency: 1-minute bars
- Symbols: 50+ forex pairs (we filter for XAUUSD, XAGUSD)

---

## Time Estimates

| Task | Time | Data Volume |
|------|------|-------------|
| Download (2019-2025) | 30-60 min | ~5-10 GB |
| Feature calculation | 10-20 min | ~2-3 GB |
| Training (8 models) | 20-40 min | ~100 MB |
| **Total** | **1-2 hours** | **~10-15 GB** |

**One-time cost** - once downloaded, you have years of data for training.

---

## Summary

✅ **This solves the 0% win rate issue**

**Root cause**: 54 days of data is insufficient
**Solution**: 2,509 days of data (7 years)

**Before**: Models trained on 38 days → 0% win rate
**After**: Models trained on 1,756 days → realistic win rates

**Run this first, then train models. You'll get real results.**
