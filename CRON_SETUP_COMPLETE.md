# ✅ Local Cron Job Installed Successfully!

## What's Running:

Your Mac will now automatically run `signal_generator.py` **every 3 minutes**.

### Cron Schedule:
```
*/3 * * * * - Runs at: 00:00, 00:03, 00:06, 00:09, 00:12, 00:15, etc.
```

---

## 📊 How to Monitor:

### Check if it's working:
```bash
# Watch live logs (press Ctrl+C to exit):
tail -f /tmp/ml_trading_signals.log

# View last 50 lines:
tail -50 /tmp/ml_trading_signals.log

# Check if signals are being uploaded to Supabase:
# Go to: https://supabase.com → Your project → Table Editor → live_signals
```

### Verify cron is active:
```bash
crontab -l
```

You should see:
```
*/3 * * * * cd /Users/omar/Desktop/ML_Trading && /Users/omar/.../python3 signal_generator.py >> /tmp/ml_trading_signals.log 2>&1
```

---

## 🛑 How to Stop (if needed):

```bash
crontab -e
# Delete the line with signal_generator.py, then save and exit
```

Or remove completely:
```bash
crontab -r
```

---

## ⚠️ Important Notes:

1. **Your Mac must stay on** for this to work
   - If you close your laptop, it won't run
   - Consider disabling sleep: System Settings → Energy → Prevent automatic sleeping

2. **Python environment** is already configured
   - Uses your `.venv312` environment automatically
   - All packages are already installed

3. **First run** will happen in the next 0-3 minutes
   - Watch the logs to confirm: `tail -f /tmp/ml_trading_signals.log`

---

## 🚀 Alternative: Railway (if you want cloud deployment)

If you want it to run 24/7 without keeping your Mac on:

### Railway.app (Free tier: 500 hours/month = ~$0/month for this use case):

1. Go to https://railway.app
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `ML_model` repository
5. Add environment variables:
   - `POLYGON_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
6. Click "Deploy"

Railway will automatically:
- Read your `Procfile`
- Start the `worker` process
- Run `worker_continuous.py` forever (every 3 minutes)

**Cost**: Free for first 500 hours/month, then $5/month

---

## ✅ Current Status:

- ✅ Local cron job installed
- ✅ Runs every 3 minutes automatically
- ✅ Logs to `/tmp/ml_trading_signals.log`
- ✅ Uploads signals to Supabase
- ✅ GitHub Actions workflow still available for manual triggers

**Next**: Wait 3 minutes and check logs to confirm it's working!

