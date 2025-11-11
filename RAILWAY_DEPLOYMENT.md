# Deploy to Railway - Complete Guide

Railway will run your signal generator continuously 24/7 in the cloud.

---

## Step 1: Push to GitHub (Already Done ✅)

Your code is already on GitHub at: `https://github.com/o-m7/ML_model`

---

## Step 2: Deploy to Railway

### A. Create Railway Account

1. Go to: https://railway.app
2. Click "Login" → Sign in with GitHub
3. Authorize Railway to access your repositories

### B. Create New Project

1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose `o-m7/ML_model`
4. Railway will auto-detect your `Procfile`

### C. Add Environment Variables

Click on your project → "Variables" tab → Add these:

```
POLYGON_API_KEY=jVLDXLylHzIpygLbXc0oYuuMGKnNOqpx
SUPABASE_URL=https://ifetofkhyblyijghuwzs.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlmZXRvZmtoeWJseWlqZ2h1d3pzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1MjI1MSwiZXhwIjoyMDY5ODI4MjUxfQ.RV-Z912bds7JsS1PeDcv1YvjenyvnfdpvUo9fZwaK7E
```

### D. Deploy

1. Railway will automatically build and deploy
2. Check "Deployments" tab to see progress
3. Click "View Logs" to see your worker running

---

## Step 3: Verify It's Working

### Check Railway Logs:

You should see:
```
🚀 Starting continuous signal generator (interval: 3 minutes)
⏰ Started at: 2025-11-10 23:50:00

================================================================================
🔄 ITERATION #1 - 2025-11-10 23:50:00
================================================================================

STANDALONE SIGNAL GENERATOR - 2025-11-10 23:50:00
================================================================================

  ✅ EURUSD 30T: LONG @ 1.15570 (TP: 1.15738, SL: 1.15486)
  ✅ EURUSD 5T: LONG @ 1.15570 (TP: 1.15609, SL: 1.15538)
  ...
✅ Processed 22/22 models
```

### Check Supabase:

1. Go to your Supabase dashboard
2. Click "Table Editor" → `live_signals`
3. You should see new signals every 3 minutes with fresh timestamps

---

## Cost

Railway Free Tier:
- ✅ **500 hours/month FREE** (20.8 days)
- ✅ Your worker uses ~730 hours/month
- 💰 After free tier: **~$5/month**

---

## Managing Your Deployment

### View Logs:
```
Railway Dashboard → Your Project → Deployments → View Logs
```

### Restart Worker:
```
Railway Dashboard → Your Project → Settings → Restart
```

### Update Code:
```bash
# Just push to GitHub - Railway auto-deploys:
git add .
git commit -m "Update signal generator"
git push
```

Railway automatically rebuilds and redeploys!

---

## Troubleshooting

### Worker Not Starting?
- Check "Deployments" tab for build errors
- Verify all 3 environment variables are set
- Check logs for error messages

### No Signals Generated?
- Check Railway logs for error messages
- Verify Supabase credentials are correct
- Ensure Polygon API key is valid

### Want to Change Frequency?
Edit `worker_continuous.py`:
```python
INTERVAL_MINUTES = 1  # Run every 1 minute instead of 3
```
Then push to GitHub.

---

## ✅ What Happens After Deployment

1. **Railway automatically starts** `worker_continuous.py`
2. **Generates signals every 3 minutes**
3. **Uploads to Supabase** automatically
4. **Runs 24/7** without your computer
5. **Auto-restarts** if it crashes
6. **Auto-updates** when you push to GitHub

---

## Next Steps

After deploying:
1. ✅ Check Railway logs to confirm it's running
2. ✅ Check Supabase to see fresh signals
3. ✅ Connect your Lovable frontend to Supabase
4. ✅ You're done! Signals will generate continuously.

**Your trading system is now fully automated and running in the cloud!** 🚀

