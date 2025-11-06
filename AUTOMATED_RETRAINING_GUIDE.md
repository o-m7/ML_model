# 🤖 Automated Weekly Retraining System

## 🎯 Overview

Your ML models now **automatically retrain every week** with the latest market data, ensuring they stay adaptive to current market conditions.

---

## 🔄 What Happens Every Sunday at 2 AM UTC

### **Complete Automated Pipeline:**

```
Sunday 2 AM UTC
    ↓
📊 Fetch Latest Data (365 days history from Polygon)
    ↓
💾 Save Raw OHLCV Data (Parquet format)
    ↓
🔧 Calculate 30+ Technical Features
    ↓
🏷️  Create Labels (Triple-Barrier Method)
    ↓
🎯 Select Best Features (Top 50)
    ↓
🤖 Train New Model (LightGBM + Balanced Classes)
    ↓
📊 Backtest on Recent Data
    ↓
⚖️  Compare with Current Model
    ↓
✅ Deploy if 5%+ Better
    ↓
📤 Update Supabase Metadata
    ↓
🔄 Repeat for All 25+ Models
```

---

## 📊 What Gets Retrained

### **Symbols:**
- XAUUSD (Gold)
- XAGUSD (Silver)
- EURUSD (Euro)
- GBPUSD (Pound)
- AUDUSD (Aussie Dollar)
- NZDUSD (Kiwi Dollar)

### **Timeframes:**
- 5T (5 minutes)
- 15T (15 minutes)
- 30T (30 minutes)
- 1H (1 hour)
- 4H (4 hours)

### **Total:** 30 models retrained every week!

---

## 🧠 Intelligent Deployment Logic

### **Model Comparison Score:**

```python
score = (
    profit_factor * 0.4 +      # 40% weight
    win_rate * 0.3 +            # 30% weight
    sharpe_ratio * 0.2 -        # 20% weight
    max_drawdown * 0.1          # 10% weight (negative)
)
```

### **Deployment Decision:**

✅ **Deploy new model** if:
- New score is 5% better than old score
- Ensures only meaningful improvements are deployed

⏭️ **Skip deployment** if:
- Improvement is less than 5%
- Keeps stable, proven models in production

---

## 📁 Data Storage

### **Raw Data:**
```
feature_store/
├── XAUUSD/
│   ├── XAUUSD_5T_raw.parquet
│   ├── XAUUSD_15T_raw.parquet
│   └── ...
├── EURUSD/
│   ├── EURUSD_5T_raw.parquet
│   └── ...
```

### **Models:**
```
models_production/
├── XAUUSD/
│   ├── XAUUSD_5T_PRODUCTION_READY.pkl
│   ├── XAUUSD_15T_PRODUCTION_READY.pkl
│   └── ...
├── EURUSD/
│   └── ...
```

---

## 🎛️ How to Monitor Retraining

### **1. GitHub Actions Dashboard:**

Visit: **https://github.com/o-m7/ML_model/actions/workflows/weekly_retraining.yml**

**You'll see:**
- ✅ Retraining status (success/failure)
- ⏱️ Duration (typically 1-2 hours)
- 📊 Logs for each model
- 📦 Artifacts (trained models, logs)

### **2. Check Logs:**

Click on any retraining run to see:
```
📊 Fetching XAUUSD 5T data...
  ✅ Fetched 105,120 bars (2024-01-01 to 2025-01-01)
  💾 Saved to feature_store/XAUUSD/XAUUSD_5T_raw.parquet
  🔧 Calculating features...
  ✅ Calculated 35 features, 105,000 valid bars

🤖 Retraining XAUUSD 5T...
  📊 Performance:
     Win Rate: 58.3%
     Profit Factor: 1.82
     Sharpe: 0.45
     Max DD: 4.2%
     Total Trades: 234

📊 Model Comparison:
   Old Score: 8.45
   New Score: 9.12
   Improvement: +7.9%
   
  ✅ Deployed to models_production/XAUUSD/XAUUSD_5T_PRODUCTION_READY.pkl
  ✅ Updated Supabase metadata
```

### **3. Supabase Metadata:**

Check `ml_models` table for:
- Last update timestamp
- Performance metrics
- Model version history

---

## 🔧 Manual Retraining

### **Trigger Manually (Anytime):**

1. Go to: **https://github.com/o-m7/ML_model/actions/workflows/weekly_retraining.yml**
2. Click **"Run workflow"**
3. Click **"Run workflow"** again
4. Wait 1-2 hours for completion

### **Run Locally:**

```bash
cd /Users/omar/Desktop/ML_Trading
source .venv312/bin/activate

# Retrain all models
python3 automated_retraining.py

# Check results
ls -lh models_production/*/
```

---

## 📊 Expected Performance

### **Training Time:**

| Models | Time | GitHub Actions Cost |
|--------|------|---------------------|
| 1 model | ~3 min | ~3 minutes |
| 6 models (1 symbol) | ~20 min | ~20 minutes |
| 30 models (all) | ~90 min | ~90 minutes |

**GitHub Actions free tier:** 2,000 minutes/month
**Your usage:** ~360 minutes/month (4 weeks × 90 min)
**Remaining:** ~1,640 minutes for other workflows

---

## 🎯 Success Metrics

### **Good Retraining:**
```
Total Models: 30
✅ Retrained: 30
🚀 Deployed: 18
⏭️  Skipped: 12
❌ Failed: 0
```

**Explanation:**
- All 30 models trained successfully
- 18 showed 5%+ improvement and were deployed
- 12 kept existing models (no improvement needed)
- 0 failures

### **Problem Retraining:**
```
Total Models: 30
✅ Retrained: 15
🚀 Deployed: 10
⏭️  Skipped: 5
❌ Failed: 15
```

**Explanation:**
- 15 models failed (check logs for Polygon API errors, data issues)
- Need to investigate and fix

---

## 🛠️ Configuration

### **Change Retraining Schedule:**

Edit `.github/workflows/weekly_retraining.yml`:

```yaml
# Current: Every Sunday at 2 AM
- cron: '0 2 * * 0'

# Options:
- cron: '0 2 * * 1'  # Every Monday at 2 AM
- cron: '0 2 1 * *'  # First day of month at 2 AM
- cron: '0 2 */3 * *'  # Every 3 days at 2 AM
```

### **Change Data History:**

Edit `automated_retraining.py`:

```python
# Current: 365 days (1 year)
df = fetch_historical_data(symbol, timeframe, days_back=365)

# Options:
days_back=180  # 6 months
days_back=730  # 2 years
```

### **Change Deployment Threshold:**

Edit `automated_retraining.py`:

```python
# Current: 5% improvement required
return improvement >= 5

# Options:
return improvement >= 10  # More conservative
return improvement >= 2   # More aggressive
```

---

## 🚨 Troubleshooting

### **Issue: Retraining fails with "No data"**

**Cause:** Polygon API rate limits or data not available

**Fix:**
- Check Polygon API key is valid
- Verify symbol/timeframe is supported
- Wait and retry later

### **Issue: "No improvement" for all models**

**Cause:** Models are already optimal, or market conditions haven't changed

**Action:** 
- ✅ This is actually good! Your models are stable
- Only deploy when there's real improvement

### **Issue: Training takes too long**

**Cause:** Too much historical data or too many models

**Fix:**
- Reduce `days_back` from 365 to 180
- Split into multiple workflows (by symbol)
- Use faster features (remove heavy calculations)

---

## 📈 Benefits of Automated Retraining

### **1. Adaptive to Market Changes**
- ✅ Models learn new patterns weekly
- ✅ Adjust to volatility shifts
- ✅ Capture regime changes

### **2. No Manual Work**
- ✅ Fully automated
- ✅ Runs while you sleep
- ✅ Zero maintenance

### **3. Always Fresh Data**
- ✅ Latest 365 days of history
- ✅ Removes outdated patterns
- ✅ Focuses on recent behavior

### **4. Quality Control**
- ✅ Only deploy if 5%+ better
- ✅ Backtest before deployment
- ✅ Keeps proven models

### **5. Version Control**
- ✅ Models versioned in git
- ✅ Can rollback if needed
- ✅ Full audit trail

---

## 🎉 Your Complete System

**Now you have:**

### **1. Signal Generation** (Every 3 minutes)
```
generate_signals_standalone.py → Supabase → Lovable
```

### **2. Model Retraining** (Every Sunday)
```
automated_retraining.py → Better Models → Production
```

### **3. Full Automation** (Zero manual work!)
```
Data → Features → Training → Testing → Deployment → Signals
```

---

## 🚀 Result

**Your ML trading system is now:**
- ✅ Fully automated signal generation
- ✅ Fully automated model retraining
- ✅ Adaptive to market changes
- ✅ Self-improving every week
- ✅ Production-grade infrastructure

**This is a Renaissance-level automated ML system!** 🎊

---

## 📞 Quick Reference

**Monitor retraining:**
```
https://github.com/o-m7/ML_model/actions/workflows/weekly_retraining.yml
```

**Trigger manual retrain:**
```
Actions → Weekly Model Retraining → Run workflow
```

**Check model updates:**
```
Supabase → ml_models table → Sort by updated_at
```

**Your system is now COMPLETE and AUTONOMOUS!** 🤖✨

