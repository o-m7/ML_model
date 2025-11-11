# 🚂 Railway Deployment - Complete Guide

## What's Fixed

✅ **Requirements.txt**: Updated to use flexible versions (pandas-ta issue fixed)  
✅ **Procfile**: Both web and worker services configured  
✅ **Model Download**: Automatic download from Supabase on startup  
✅ **Startup Script**: Handles both API and worker processes  
✅ **.railwayignore**: Properly configured  

---

## 🚀 Deploy to Railway (Step-by-Step)

### **Step 1: Push Everything to GitHub**

```bash
cd /Users/omar/Desktop/ML_Trading

# Add all files
git add -A

# Commit
git commit -m "🚂 Railway deployment ready - fixed requirements and added model download"

# Push
git push origin main
```

### **Step 2: Create Railway Project**

1. Go to: https://railway.app
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose: `o-m7/ML_model`
5. Click **"Deploy Now"**

### **Step 3: Add Environment Variables**

In Railway dashboard → **Variables** tab:

```
POLYGON_API_KEY=your_polygon_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
PORT=8000
```

### **Step 4: Configure Services**

Railway will detect both services from Procfile:

#### **Web Service** (API):
- **Start Command**: `bash railway_start.sh`
- **Port**: 8000
- **Public Domain**: Enable (for API access)

#### **Worker Service** (Signal Generator):
- **Start Command**: `bash railway_start.sh`
- **Environment Variable**: `RAILWAY_SERVICE_NAME=worker`
- **Public Domain**: Disable (internal only)

### **Step 5: Deploy**

1. Railway will automatically build and deploy
2. Wait for build to complete (~3-5 minutes)
3. Check logs for "✅ System operational"

---

## 📊 What Gets Deployed

### **Web Service (API)**:
```
railway_start.sh
  ↓
download_models.py (downloads from Supabase)
  ↓
api_server.py (starts FastAPI on port 8000)
  ↓
Serves predictions at: https://your-app.railway.app
```

### **Worker Service (Signal Generator)**:
```
railway_start.sh
  ↓
download_models.py (downloads from Supabase)
  ↓
worker_continuous.py
  ↓
signal_generator.py (runs every 3 minutes)
  ↓
Uploads signals to Supabase
```

---

## 🔍 Verify Deployment

### Check API Health:
```bash
curl https://your-app.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_available": 25,
  "supabase": "connected"
}
```

### Check Signals in Supabase:
```bash
# In Supabase SQL Editor:
SELECT symbol, timeframe, signal_type, timestamp 
FROM live_signals 
ORDER BY timestamp DESC 
LIMIT 10;
```

Should see new signals every 3 minutes.

---

## 🐛 Troubleshooting

### **Build Failed: pandas-ta not found**
✅ **Fixed**: Updated `requirements.txt` to use `pandas-ta>=0.3.14b`

### **Models not loading**
✅ **Fixed**: `download_models.py` downloads from Supabase on startup

### **Web service not starting**
- Check logs: Railway → Deployments → View Logs
- Ensure `PORT=8000` in environment variables
- Verify `api_server.py` uses `PORT` env var

### **Worker not generating signals**
- Check: Railway → Worker service → Logs
- Should see: "🔄 ITERATION #1" every 3 minutes
- Verify Polygon API key is valid

### **No signals in Supabase**
- Check worker logs for errors
- Verify Supabase credentials
- Test manually: `python3 signal_generator.py`

---

## 💰 Railway Pricing

### **Starter Plan** (Recommended):
- **$5/month** per service
- **Total**: $10/month (web + worker)
- **Includes**: 500 hours, $5 credit

### **Resource Usage**:
- **API**: ~0.5GB RAM, minimal CPU
- **Worker**: ~1GB RAM, runs every 3 minutes

### **Cost Estimate**:
- **Railway**: $10/month
- **Supabase**: Free (Pro: $25/month if needed)
- **Polygon API**: $0-200/month (depends on plan)
- **Total**: ~$10-235/month

---

## 🔄 Continuous Deployment

### **Auto-Deploy on Git Push**:

Railway automatically redeploys when you push to GitHub:

```bash
# Make changes
git add .
git commit -m "Update models"
git push

# Railway automatically:
# 1. Pulls latest code
# 2. Rebuilds containers
# 3. Redeploys both services
# 4. Zero downtime
```

### **Manual Redeploy**:
Railway Dashboard → Service → **Redeploy**

---

## 📈 Monitoring

### **Railway Dashboard**:
- **Metrics**: CPU, RAM, Network usage
- **Logs**: Real-time application logs
- **Deployments**: History of all deploys

### **View Logs**:
```bash
# In Railway dashboard
Web Service → Logs → View real-time
Worker Service → Logs → View real-time
```

### **Supabase Monitoring**:
```sql
-- Signal generation rate
SELECT 
    date_trunc('hour', timestamp) as hour,
    COUNT(*) as signals
FROM live_signals
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;

-- Model performance
SELECT 
    symbol, 
    timeframe,
    COUNT(*) as total_signals,
    AVG(confidence) as avg_confidence
FROM live_signals
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY symbol, timeframe
ORDER BY total_signals DESC;
```

---

## 🔒 Security

### **Environment Variables**:
- ✅ Never commit `.env` to GitHub
- ✅ Set in Railway dashboard only
- ✅ Use Supabase service_role key

### **API Access**:
- Public API endpoint (anyone can call)
- Consider adding API key authentication later
- Rate limiting via Railway

### **Database**:
- Supabase RLS policies enabled
- Service role key bypasses RLS (needed)

---

## 🎯 What Happens After Deploy

### **Immediate (0-5 minutes)**:
1. ✅ Railway builds containers
2. ✅ Models download from Supabase
3. ✅ API starts on port 8000
4. ✅ Worker starts generating signals

### **First Hour**:
1. ✅ ~20 signals generated
2. ✅ Stored in Supabase
3. ✅ API serving predictions

### **First Day**:
1. ✅ ~480 signals generated (20/hour × 24)
2. ✅ Some signals close (hit TP/SL)
3. ✅ Trades logged

### **First Week**:
1. ✅ ~3,360 signals generated
2. ✅ 50-100 trades closed
3. ✅ First retraining cycle begins
4. ✅ Models improve

---

## 📋 Deployment Checklist

- [ ] Push all code to GitHub
- [ ] Create Railway project
- [ ] Link GitHub repo
- [ ] Add environment variables (POLYGON_API_KEY, SUPABASE_URL, SUPABASE_KEY, PORT)
- [ ] Configure web service (enable public domain)
- [ ] Configure worker service (set RAILWAY_SERVICE_NAME=worker)
- [ ] Deploy both services
- [ ] Verify API health endpoint
- [ ] Check Supabase for signals
- [ ] Monitor logs for errors
- [ ] Enable GitHub Actions (optional)

---

## 🚨 Important Notes

### **Models**:
- Downloaded from Supabase on startup
- Not stored in GitHub (too large)
- Cached after first download

### **Data**:
- Feature store rebuilt on-demand
- Historical data fetched from Polygon
- Cached in Railway persistent storage

### **Retraining**:
- Happens via GitHub Actions (not Railway)
- New models uploaded to Supabase
- Railway downloads new models on restart

---

## 🎉 Success Indicators

### **Deployment Successful**:
```
✅ Build completed
✅ Web service: Healthy (200 OK)
✅ Worker service: Running
✅ Logs show: "🔄 ITERATION #1"
✅ Supabase: New signals appearing
```

### **System Operational**:
```
✅ API responding: /health returns 200
✅ Models loaded: 25+ models available
✅ Signals generating: Every 3 minutes
✅ Trades logging: When TP/SL hit
✅ Learning active: When 10+ trades
```

---

## 🔗 Useful Links

- **Railway Dashboard**: https://railway.app/dashboard
- **Railway Docs**: https://docs.railway.app
- **Supabase Dashboard**: https://supabase.com/dashboard
- **GitHub Repo**: https://github.com/o-m7/ML_model

---

## 📞 Support

If deployment fails:

1. Check Railway logs
2. Verify environment variables
3. Test locally: `python3 api_server.py`
4. Check Supabase connection
5. Verify Polygon API key

---

**Your system is ready for Railway deployment! 🚀**

Run the commands in Step 1, then follow Steps 2-5 in Railway dashboard.

