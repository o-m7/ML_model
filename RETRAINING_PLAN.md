# Model Retraining Plan
**Date:** 2025-11-15
**Status:** Ready for Local Execution

---

## 📊 Performance Analysis Summary

### 🟢 BEST PERFORMERS (Keep As-Is)

**XAGUSD Models:**
- **15T LONG:** 90.0% win rate, 1.25 avg R (9W-1L) ⭐⭐⭐
- **30T LONG:** 88.9% win rate, 1.67 avg R (8W-1L) ⭐⭐⭐
- **5T LONG:** 76.9% win rate, 0.85 avg R (10W-3L) ⭐⭐
- **1H:** Good performance ⭐

**XAUUSD Models:**
- **15T LONG:** 75.9% win rate, 0.97 avg R (22W-7L) ⭐⭐
- **15T SHORT:** 54.3% win rate, 0.41 avg R (19W-16L) ⭐

### 🔴 POOR PERFORMERS (Need Retraining)

**XAUUSD Models:**
- **5T LONG:** 37.2% win rate, -0.11 avg R (16W-27L) ❌ **CRITICAL**
- **5T SHORT:** 36.8% win rate, -0.12 avg R (7W-12L) ❌ **CRITICAL**
- **30T SHORT:** 35.0% win rate, 0.05 avg R (21W-39L) ❌

### 🟡 MARGINAL (Monitor)

- XAUUSD 30T LONG: 50% win rate (small sample)
- XAGUSD 5T SHORT: 49% win rate (breakeven)
- XAGUSD 15T/30T SHORT: 39-41% win rate

---

## 🎯 Action Plan

### ✅ Completed

1. **Removed old currency pairs:**
   - ❌ EURUSD
   - ❌ NZDUSD
   - ❌ GBPUSD
   - ❌ AUDUSD

2. **Removed 4H timeframe** (no 4H model exists)

3. **Consolidated XAGUSD models:**
   - Copied from `models_production/` to `models_rentec/`
   - Active: 5T, 15T, 30T, 1H

### 🔄 To Execute Locally

**Models requiring retraining:**
1. **XAUUSD 5T** - Both directions failing (37% win rate)
2. **XAUUSD 30T** - SHORT direction failing (35% win rate)

**Optionally retrain:**
- XAUUSD 1H - For completeness and consistency

---

## 🚀 Execution Instructions

### Prerequisites

Ensure you have:
- Python environment activated
- All dependencies installed (`pip install -r requirements.txt`)
- Training data exists in `feature_store/XAUUSD/`

### Option 1: Automated Script (Recommended)

```bash
./retrain_poor_performers.sh
```

This script will:
1. ✅ Backup existing models
2. ✅ Retrain XAUUSD 5T
3. ✅ Retrain XAUUSD 30T
4. ✅ Optionally retrain XAUUSD 1H

### Option 2: Manual Training

```bash
# Retrain XAUUSD 5T (worst performer - 37% win rate)
python train_model.py --symbol XAUUSD --tf 5T

# Retrain XAUUSD 30T (poor SHORT - 35% win rate)
python train_model.py --symbol XAUUSD --tf 30T

# Optional: Retrain 1H for completeness
python train_model.py --symbol XAUUSD --tf 1H
```

### Option 3: Train All Models from Scratch

```bash
python train_all_models.py
```

This will train all 8 models (XAUUSD + XAGUSD, 4 timeframes each).

---

## 📁 Model Locations

**Active Production Models:**
```
models_rentec/
├── XAGUSD/
│   ├── XAGUSD_5T.pkl   ✅ (76.9% win rate)
│   ├── XAGUSD_15T.pkl  ✅ (90.0% win rate)
│   ├── XAGUSD_30T.pkl  ✅ (88.9% win rate)
│   └── XAGUSD_1H.pkl   ✅
└── XAUUSD/
    ├── XAUUSD_5T.pkl   ❌ (37.2% - RETRAIN!)
    ├── XAUUSD_15T.pkl  ✅ (75.9% win rate)
    ├── XAUUSD_30T.pkl  ⚠️  (50% LONG, 35% SHORT - RETRAIN!)
    └── XAUUSD_1H.pkl   ✅
```

**Backup Location:**
```
models_rentec/XAUUSD/backup_YYYYMMDD_HHMMSS/
```

---

## 🧪 Testing After Retraining

### 1. Verify Models Loaded

```bash
python -c "from ensemble_predictor import EnsemblePredictor; \
           e = EnsemblePredictor('XAUUSD'); \
           print('XAUUSD models:', list(e.models.keys()))"
```

### 2. Test Signal Generation

```bash
python signal_generator.py
```

Look for:
- ✅ All models load successfully
- ✅ No missing feature warnings
- ✅ Signals generated for both XAUUSD and XAGUSD
- ✅ Confidence scores reasonable (40-80%)

### 3. Monitor Production Performance

Track these metrics for XAUUSD 5T and 30T:
- **Win rate:** Target 55%+ (vs 37% before)
- **Avg R:** Target >0.3 (vs -0.11 before)
- **Signal frequency:** Should remain similar
- **Confidence distribution:** 45-75% range

---

## 📈 Expected Improvements

### XAUUSD 5T
- **Current:** 37.2% win rate, -0.11 avg R (losing money)
- **Target:** 55%+ win rate, 0.3+ avg R
- **Impact:** Most critical fix - 5T has high frequency

### XAUUSD 30T
- **Current:** 50% LONG, 35% SHORT
- **Target:** 55%+ for both directions
- **Impact:** Balance SHORT performance with LONG

### Overall Portfolio
- **Current:** Mixed results, 5T dragging down performance
- **Target:** Consistent 55-70% across all timeframes
- **Benefit:** Higher confidence in automated trading

---

## ⚠️ Important Notes

1. **Backup existing models** before retraining
2. **Training takes 2-5 minutes** per model
3. **Feature data must exist** in `feature_store/`
4. **Test thoroughly** before deploying to production
5. **Monitor first 50 signals** closely after deployment

---

## 🎯 Success Criteria

**Minimum acceptable performance:**
- Win rate: ≥50%
- Avg R: ≥0.2
- Test accuracy: ≥52%

**Target performance:**
- Win rate: ≥55%
- Avg R: ≥0.4
- Test accuracy: ≥55%

**Excellent performance:**
- Win rate: ≥60%
- Avg R: ≥0.6
- Test accuracy: ≥58%

---

## 📚 Related Documents

- `FINAL_IMPLEMENTATION_SUMMARY.md` - Overall system status
- `train_model.py` - Training script
- `retrain_poor_performers.sh` - Automated retraining
- `signal_generator.py` - Production signal generation

---

**Status:** ✅ Ready to execute locally
**Next Step:** Run `./retrain_poor_performers.sh` on your local machine
**Estimated Time:** 5-10 minutes total
