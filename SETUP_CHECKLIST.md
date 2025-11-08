# 🔧 SETUP CHECKLIST - Complete These Steps

## ✅ **STEP 1: Supabase Tables** (5 minutes)

### Action:
1. Go to your Supabase project: https://supabase.com/dashboard
2. Click on **SQL Editor** (left sidebar)
3. Click **New Query**
4. Copy the entire contents of `supabase_setup_simple.sql`
5. Paste into the query editor
6. Click **Run** (or press Cmd/Ctrl + Enter)

### Expected Result:
```
Success. No rows returned
```

### Verify Tables Created:
Go to **Table Editor** and confirm these 10 tables exist:
- ✅ ml_models
- ✅ live_signals
- ✅ trades
- ✅ performance_metrics
- ✅ ensemble_metadata (Phase 4)
- ✅ economic_events (Phase 4)
- ✅ sentiment_data (Phase 4)

---

## ✅ **STEP 2: Install Python Dependencies** (2 minutes)

### Action:
```bash
cd /Users/omar/Desktop/ML_Trading
source .venv312/bin/activate  # Or your virtual environment

pip install newsapi-python praw vaderSentiment optuna
```

### Verify Installation:
```bash
python3 -c "import newsapi; import praw; import vaderSentiment; import optuna; print('✅ All packages installed')"
```

---

## ✅ **STEP 3: Get API Keys** (10 minutes)

### 3.1 NewsAPI (Free - for news sentiment)
1. Go to: https://newsapi.org/register
2. Sign up (free tier: 100 requests/day)
3. Copy your API key
4. Add to `.env`:
   ```bash
   NEWSAPI_KEY=your_key_here
   ```

### 3.2 Reddit API (Free - for Reddit sentiment)
1. Go to: https://www.reddit.com/prefs/apps
2. Scroll to bottom, click **"create another app"**
3. Fill in:
   - Name: `ML Trading Bot`
   - Type: Select **script**
   - Redirect URI: `http://localhost:8080`
   - Description: (optional)
4. Click **Create app**
5. Copy:
   - **Client ID** (under the app name)
   - **Client Secret** (shown as "secret")
6. Add to `.env`:
   ```bash
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT="ML Trading Bot v1.0"
   ```

---

## ✅ **STEP 4: Add GitHub Secrets** (5 minutes)

### Action:
1. Go to: https://github.com/o-m7/ML_model/settings/secrets/actions
2. Click **New repository secret** for each:

| Name | Value | Source |
|------|-------|--------|
| `NEWSAPI_KEY` | Your NewsAPI key | Step 3.1 |
| `REDDIT_CLIENT_ID` | Your Reddit client ID | Step 3.2 |
| `REDDIT_CLIENT_SECRET` | Your Reddit secret | Step 3.2 |
| `REDDIT_USER_AGENT` | `ML Trading Bot v1.0` | Manual |

**Already Set (verify):**
- ✅ POLYGON_API_KEY
- ✅ SUPABASE_URL
- ✅ SUPABASE_KEY

---

## ✅ **STEP 5: Test Ensemble System** (5 minutes)

### Action:
```bash
cd /Users/omar/Desktop/ML_Trading
python3 -c "
from ensemble_predictor import EnsemblePredictor
import numpy as np

# Test loading ensemble for XAUUSD
ensemble = EnsemblePredictor('XAUUSD')
print(f'✅ Loaded {len(ensemble.models)} models for XAUUSD')

# Test prediction
features = np.random.randn(30)
result = ensemble.ensemble_predict(features, strategy='confidence_weighted')
print(f'✅ Prediction: {result[\"prediction\"]}, Confidence: {result[\"confidence\"]:.3f}')
print('✅ Ensemble system working!')
"
```

**Expected Output:**
```
✅ Loaded 5 models for XAUUSD
✅ Prediction: 1, Confidence: 0.XXX
✅ Ensemble system working!
```

---

## ✅ **STEP 6: Test Sentiment Collection** (5 minutes)

### Action:
```bash
cd /Users/omar/Desktop/ML_Trading
python3 sentiment_data_collector.py
```

**Expected Output:**
```
📰 COLLECTING SENTIMENT DATA
Processing XAUUSD...
  News articles: X
  Reddit posts: Y
  Aggregate sentiment: Z
✅ Stored sentiment for XAUUSD
...
✅ Sentiment collection complete!
```

---

## ✅ **STEP 7: Verify GitHub Actions** (2 minutes)

### Action:
1. Go to: https://github.com/o-m7/ML_model/actions
2. Check these workflows are enabled:
   - ✅ `generate_signals.yml` (every 3 minutes)
   - ✅ `weekly_retraining.yml` (every Sunday)
   - ✅ `calendar_update.yml` (daily)
   - ✅ `sentiment_collection.yml` (hourly)

3. Manually trigger sentiment collection:
   - Click on **Sentiment Collection**
   - Click **Run workflow** → **Run workflow**
   - Wait 1-2 minutes
   - Should show ✅ Success

---

## ✅ **STEP 8: Test Trade Learning System** (5 minutes)

### Action:
```bash
cd /Users/omar/Desktop/ML_Trading

# Test trade collection (will work once you have trades in Supabase)
python3 trade_collector.py

# Test dashboard generation (requires matplotlib)
pip install matplotlib seaborn
python3 trade_learning_dashboard.py
```

**Expected Output:**
```
📊 Fetching trades from last 30 days...
✅ Fetched X trades
📉 ANALYZING X LOSING TRADES
✅ Live trades saved to live_trades/
📊 Report saved: trade_analysis/...
```

---

## 📋 **COMPLETION CHECKLIST**

Mark each as you complete:

- [ ] **Step 1:** Supabase tables created
- [ ] **Step 2:** Python packages installed (including matplotlib, seaborn)
- [ ] **Step 3:** API keys obtained (NewsAPI + Reddit)
- [ ] **Step 4:** GitHub secrets added
- [ ] **Step 5:** Ensemble system tested locally
- [ ] **Step 6:** Sentiment collection tested locally
- [ ] **Step 7:** GitHub Actions verified
- [ ] **Step 8:** Trade learning system tested

---

## 🎯 **AFTER COMPLETION**

Once all steps are ✅, you'll have:
- ✅ All database tables ready
- ✅ Ensemble voting operational
- ✅ News blackout filtering active
- ✅ Sentiment analysis running hourly
- ✅ **Trade learning system** - Models improve from every trade
- ✅ Full automation via GitHub Actions

**The system will now:**
1. Generate signals every 3 minutes
2. Collect sentiment data hourly
3. Update economic calendar daily
4. **Learn from live trades daily** (NEW!)
5. Retrain models weekly

**Next:** Monitor performance and optionally add Parameter Optimization + RL Agent

---

## ❓ **TROUBLESHOOTING**

### Issue: Supabase SQL fails
- **Fix:** Tables may already exist. That's OK! Just verify they exist in Table Editor.

### Issue: `pip install` fails
- **Fix:** Upgrade pip: `pip install --upgrade pip`

### Issue: Ensemble test fails with "No models found"
- **Fix:** Ensure `.pkl` files exist in `models_production/XAUUSD/`

### Issue: Sentiment collection fails
- **Fix:** Check API keys in `.env` are correct

### Issue: GitHub Actions fails
- **Fix:** Verify secrets are set correctly in GitHub settings

---

**Current Status:** Ready to begin setup!

**Time Required:** ~30 minutes total

