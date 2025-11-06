# 🚂 Deploy to Railway - 15 Minute Guide

## Why Railway?
- ✅ Free tier with 500 hours/month
- ✅ Auto-detects Python/FastAPI
- ✅ Easy environment variables
- ✅ GitHub integration
- ✅ Free PostgreSQL if needed
- ✅ Automatic HTTPS

---

## 📋 Pre-Deployment Checklist

Make sure you have:
- ✅ GitHub account
- ✅ Railway account (sign up at railway.app)
- ✅ Your `.env` file with API keys
- ✅ Tested locally (`python3 api_server.py` works)

---

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Repository (5 minutes)

**A. Create necessary files:**

**1. `Procfile` (tells Railway what to run)**
```bash
cat > Procfile << 'EOF'
web: python3 api_server.py
worker: python3 live_trading_engine.py continuous
EOF
```

**2. `railway.toml` (Railway configuration)**
```bash
cat > railway.toml << 'EOF'
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "python3 api_server.py"
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
EOF
```

**3. Verify `requirements_api.txt` exists**
```bash
cat requirements_api.txt
# Should include: fastapi, uvicorn, supabase, etc.
```

**B. Push to GitHub:**
```bash
# Initialize git if not already
git init

# Create .gitignore
cat > .gitignore << 'EOF'
.env
.venv/
.venv312/
__pycache__/
*.pyc
*.pyo
*.egg-info/
.DS_Store
*.pkl
*.joblib
feature_store/
models_production/
*.log
EOF

# Commit everything
git add .
git commit -m "Deploy ML Trading System to Railway"

# Create repo on GitHub, then:
git remote add origin https://github.com/yourusername/ml-trading.git
git branch -M main
git push -u origin main
```

---

### Step 2: Deploy on Railway (5 minutes)

**A. Create New Project:**
1. Go to [railway.app](https://railway.app)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Connect your GitHub account if not already
5. Select your `ml-trading` repository

**B. Configure Service:**
Railway will auto-detect Python and FastAPI. It will:
- ✅ Install from `requirements_api.txt`
- ✅ Run the command from `Procfile` or `railway.toml`
- ✅ Assign a public URL

**C. Add Environment Variables:**
1. Click on your service
2. Go to **"Variables"** tab
3. Add each variable:

```
POLYGON_API_KEY=your_polygon_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_service_role_key_here
PORT=8000
```

**D. Deploy:**
- Railway will automatically deploy
- Wait 2-3 minutes for build
- Check logs for any errors

---

### Step 3: Test Deployment (2 minutes)

**A. Get your Railway URL:**
```
https://your-app-name.up.railway.app
```

**B. Test endpoints:**
```bash
# Health check
curl https://your-app-name.up.railway.app/health

# List models
curl https://your-app-name.up.railway.app/models

# Test prediction
curl -X POST https://your-app-name.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "XAUUSD",
    "timeframe": "5T",
    "features": [0.001, -0.002, 0.0015, 0.003, -0.001, 0.002, 0.0005, -0.0008, 0.0012, 0.0018, 0.0007, -0.0003, 0.0009, 0.0011, -0.0006, 0.0014, 0.0004, -0.0002, 0.0016, 0.0013, 0.0019, -0.0004, 0.0008, 0.0021, -0.0009, 0.0017, 0.0006, -0.0005, 0.0022, 0.0015]
  }'
```

---

### Step 4: Deploy Live Engine (3 minutes)

**Option A: Add as separate service (Recommended)**

1. In Railway project, click **"+ New"**
2. Select **"Empty Service"**
3. Connect same GitHub repo
4. Go to **"Settings"**
5. Set **Start Command**: `python3 live_trading_engine.py continuous`
6. Add same environment variables
7. Deploy

**Option B: Use Procfile with multiple processes**
```
web: python3 api_server.py
worker: python3 live_trading_engine.py continuous
```
(Note: Railway free tier might only run one process)

**Option C: Keep it local for now**
```bash
# Run on your machine
python3 live_trading_engine.py continuous
```

---

## 🌐 Update Your Lovable Frontend

### Use Railway URL in Lovable:

**Option 1: Direct API calls**
```typescript
const API_URL = 'https://your-app-name.up.railway.app';

const response = await fetch(`${API_URL}/models`);
const models = await response.json();
```

**Option 2: Just use Supabase (Recommended)**
```typescript
// You don't need to call the API directly from frontend
// The live_trading_engine is already storing signals in Supabase
const { data } = await supabase
  .from('live_signals')
  .select('*')
  .order('timestamp', { ascending: false });
```

---

## 🔍 Monitoring Your Deployment

### Railway Dashboard:
- **Metrics**: CPU, Memory, Network usage
- **Logs**: Real-time application logs
- **Deployments**: History and rollback

### Check API Health:
```bash
curl https://your-app.railway.app/health
```

### Check Supabase:
```bash
# In Supabase SQL Editor
SELECT COUNT(*) FROM live_signals WHERE status = 'active';
```

---

## 🐛 Troubleshooting

### "Build Failed"
**Check:**
- `requirements_api.txt` has all dependencies
- Python version compatible (Railway uses 3.11 by default)
- No syntax errors in code

**Fix:**
```bash
# Test locally first
python3 api_server.py
```

### "API Not Responding"
**Check:**
- Environment variables are set correctly
- Port is 8000 (or $PORT from Railway)
- Health endpoint returns 200

**Fix:**
```python
# In api_server.py, ensure:
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### "No Signals Generated"
**Check:**
- Live engine is running
- Polygon API key is valid
- Supabase connection works

**Fix:**
```bash
# Test manually
python3 live_trading_engine.py once
```

### "ONNX Models Not Found"
**Issue:** ONNX files might be too large for Git

**Fix:** Models are in Supabase Storage, API loads from there
```python
# api_server.py already handles this
# Just ensure SUPABASE_URL and SUPABASE_KEY are set
```

---

## 💰 Cost Breakdown

### Railway Free Tier:
- ✅ 500 hours/month execution time
- ✅ 1GB RAM
- ✅ 1GB Disk
- ✅ Shared CPU

**Your Usage:**
- API Server: ~720 hours/month (if 24/7)
- Worker: Use local or GitHub Actions

**Recommendation:**
- API on Railway (24/7)
- Live engine locally or GitHub Actions cron

### If You Exceed Free Tier:
- $5/month for Hobby plan (unlimited hours)

---

## 🔄 Alternative: Run Everything Locally

If you prefer not to deploy yet:

```bash
# Terminal 1 - API
python3 api_server.py

# Terminal 2 - Live Engine
python3 live_trading_engine.py continuous

# Terminal 3 - Expose API (using ngrok)
ngrok http 8000
# Use the ngrok URL in Lovable
```

---

## ✅ Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Railway project created
- [ ] Environment variables added
- [ ] API deployed and tested
- [ ] Live engine running (Railway or local)
- [ ] Lovable connected to Supabase
- [ ] Signals appearing in database
- [ ] Frontend displaying signals

---

## 🎉 You're Live!

Once deployed:
1. **API serves predictions** at your Railway URL
2. **Live engine** generates signals every 5 minutes
3. **Supabase stores** all signals with TP/SL
4. **Lovable UI** displays beautiful trading dashboard

**Your Renaissance-grade trading system is LIVE! 🚀📈**

