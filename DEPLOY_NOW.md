# 🚀 Deploy to Railway - Quick Reference

## ✅ Status: READY TO DEPLOY

Everything is pushed to GitHub and Railway-ready!

---

## 🚂 Deploy in 5 Minutes

### 1️⃣ Go to Railway
👉 https://railway.app

### 2️⃣ Create Project
- Click **"New Project"**
- Select **"Deploy from GitHub repo"**
- Choose: **`o-m7/ML_model`**
- Click **"Deploy Now"**

### 3️⃣ Add Environment Variables
Click **Variables** → **+ New Variable**:

```
POLYGON_API_KEY=your_polygon_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_service_key_here
PORT=8000
```

### 4️⃣ Configure Services
Railway detects 2 services from Procfile:

**Service 1: Web (API)**
- ✅ Enable Public Domain
- Port: 8000
- Start: `bash railway_start.sh`

**Service 2: Worker**
- Add variable: `RAILWAY_SERVICE_NAME=worker`
- ❌ Disable Public Domain
- Start: `bash railway_start.sh`

### 5️⃣ Wait & Verify (3-5 min)
Build completes → Test API:

```bash
curl https://your-app.railway.app/health
```

Expected:
```json
{"status":"healthy","models_available":25}
```

---

## ✅ What Was Fixed

| Issue | Solution |
|-------|----------|
| ❌ pandas-ta==0.3.14b not found | ✅ Changed to `pandas-ta>=0.3.14b` |
| ❌ No web service in Procfile | ✅ Added `web: bash railway_start.sh` |
| ❌ Models not in repo | ✅ Auto-download from Supabase |
| ❌ No startup script | ✅ Created `railway_start.sh` |

---

## 📊 What Happens

```
Railway Build:
  ├─ Install Python 3.12
  ├─ Install packages (requirements.txt)
  └─ Build container (2-3 min)

Startup:
  ├─ Run railway_start.sh
  ├─ Download models from Supabase
  └─ Start service (API or Worker)

Running:
  ├─ API: https://your-app.railway.app
  ├─ Worker: Generates signals every 3 min
  └─ Signals: Uploaded to Supabase
```

---

## 💰 Cost

- **Railway**: $10/month (2 services)
- **Supabase**: Free tier OK
- **Total**: ~$10/month

---

## 🔍 Verify Deployment

### Check API:
```bash
curl https://your-app.railway.app/health
curl https://your-app.railway.app/models
```

### Check Supabase Signals:
```sql
SELECT symbol, timeframe, signal_type, timestamp 
FROM live_signals 
ORDER BY timestamp DESC 
LIMIT 10;
```

Should see new signals every 3 minutes!

---

## 📚 Full Docs

- **RAILWAY_DEPLOY_COMPLETE.md** - Detailed guide
- **AUTOMATIC_LEARNING_SYSTEM.md** - Learning system
- **SETUP_AUTOMATIC_LEARNING.md** - Quick setup

---

## 🎉 You're Ready!

1. Go to Railway: https://railway.app
2. Deploy from GitHub: `o-m7/ML_model`
3. Add 4 environment variables
4. Wait 5 minutes
5. Done! 🚀

