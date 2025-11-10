# How to Run Signal Generator on Schedule

GitHub Actions cron is **unreliable** for frequent intervals (every 1-3 minutes). Here are **3 production-ready alternatives**:

---

## **✅ OPTION 1: Railway/Render (RECOMMENDED - Easiest)**

### Why Choose This:
- ✅ Free tier available
- ✅ Runs 24/7 automatically
- ✅ No server management
- ✅ Easy deployment

### Steps:

1. **Push to GitHub** (already done):
```bash
cd /Users/omar/Desktop/ML_Trading
git add worker_continuous.py Procfile
git commit -m "Add continuous worker for cloud deployment"
git push
```

2. **Deploy to Railway**:
   - Go to https://railway.app
   - Click "New Project" → "Deploy from GitHub"
   - Select your `ML_model` repository
   - Railway will auto-detect `Procfile`
   
3. **Add Environment Variables** in Railway:
   ```
   POLYGON_API_KEY=your_key
   SUPABASE_URL=your_url
   SUPABASE_KEY=your_key
   ```

4. **Start the Worker**:
   - Railway will show two services: `web` and `worker`
   - Make sure `worker` is running
   - Check logs to verify it's generating signals every 3 minutes

**Cost**: Free for 500 hours/month, then $5/month

---

## **✅ OPTION 2: Local Machine Cron (FREE - Best for Testing)**

### Why Choose This:
- ✅ Completely free
- ✅ Runs on your Mac 24/7
- ✅ Full control
- ❌ Requires your computer to stay on

### Steps:

1. **Run the setup script**:
```bash
cd /Users/omar/Desktop/ML_Trading
./setup_local_cron.sh
```

2. **Verify it's working**:
```bash
# Wait 3 minutes, then check logs:
tail -f /tmp/ml_trading_signals.log
```

3. **Check Supabase**:
   - Open Supabase dashboard
   - Check `live_signals` table
   - You should see new signals every 3 minutes

**To remove** (if you want to stop):
```bash
crontab -e
# Delete the line with signal_generator.py
```

---

## **✅ OPTION 3: AWS Lambda (Production - Scalable)**

### Why Choose This:
- ✅ Industry standard
- ✅ Highly scalable
- ✅ Pay only for executions
- ❌ More complex setup

### Steps:

1. **Install AWS CLI**:
```bash
brew install awscli
aws configure
```

2. **Create deployment package**:
```bash
cd /Users/omar/Desktop/ML_Trading
pip install -r requirements_api.txt -t ./package
cp *.py ./package/
cd package
zip -r ../lambda_function.zip .
```

3. **Deploy to AWS**:
   - Go to AWS Lambda console
   - Create new function (Python 3.12)
   - Upload `lambda_function.zip`
   - Set handler: `lambda_handler.lambda_handler`
   - Set environment variables
   - Set memory: 512MB, timeout: 5 minutes

4. **Create EventBridge trigger**:
   - In Lambda, click "Add trigger"
   - Select "EventBridge (CloudWatch Events)"
   - Create new rule: `rate(3 minutes)`

**Cost**: Free tier = 1M requests/month, then $0.20 per 1M requests

---

## **🎯 RECOMMENDATION**

For your use case (live trading signals every 3 minutes):

- **Testing/Personal Use**: → **Option 2 (Local Cron)** - Free, easy, reliable
- **Production/Team Use**: → **Option 1 (Railway)** - Professional, reliable, low cost
- **Enterprise/Scale**: → **Option 3 (AWS Lambda)** - Most robust, scalable

---

## **Current Status**

✅ All files created and ready to deploy
✅ GitHub Actions still works for manual triggers (`workflow_dispatch`)
✅ All 3 options are production-ready

**Next Step**: Choose an option and I'll help you deploy it.

