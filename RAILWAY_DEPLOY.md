# Deploy to Railway - 1 Minute Setup

## Step 1: Push to GitHub (Already Done ✅)

Your code is already on GitHub at: `https://github.com/o-m7/ML_model`

## Step 2: Deploy to Railway

1. **Go to Railway**: https://railway.app/new

2. **Click "Deploy from GitHub repo"**

3. **Select your repository**: `o-m7/ML_model`

4. **Railway will auto-detect**:
   - ✅ `Procfile` (will run `worker_continuous.py`)
   - ✅ `requirements.txt` (will install dependencies)
   - ✅ Python runtime

## Step 3: Add Environment Variables

In Railway dashboard, go to **Variables** tab and add:

```
POLYGON_API_KEY=jVLDXLylHzIpygLbXc0oYuuMGKnNOqpx
SUPABASE_URL=https://ifetofkhyblyijghuwzs.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlmZXRvZmtoeWJseWlqZ2h1d3pzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI1MjI1MSwiZXhwIjoyMDY5ODI4MjUxfQ.RV-Z912bds7JsS1PeDcv1YvjenyvnfdpvUo9fZwaK7E
```

## Step 4: Deploy!

Click **"Deploy"** - Railway will:
- ✅ Clone your repo
- ✅ Install dependencies
- ✅ Start `worker_continuous.py`
- ✅ Run continuously, generating signals every 3 minutes

## Step 5: Monitor

In Railway dashboard, you'll see:
- **Logs**: Real-time output from `worker_continuous.py`
- **Metrics**: CPU, Memory usage
- **Status**: Running/Stopped

## Cost

- **Free Tier**: $5 credit/month (~500 hours)
- **Paid**: $5/month for unlimited hours
- Your worker uses ~720 hours/month = **$5/month**

## That's It!

Your signals will now generate automatically every 3 minutes, 24/7, and upload to Supabase!

---

## Troubleshooting

### If deployment fails:

1. Check logs in Railway dashboard
2. Verify all 3 environment variables are set
3. Re-deploy by clicking "Restart"

### To change interval:

Edit `worker_continuous.py` line 11:
```python
INTERVAL_MINUTES = 3  # Change to 1, 5, etc.
```

Then commit and push - Railway will auto-redeploy.

### To stop:

In Railway dashboard, click **"Stop"** or delete the service.

