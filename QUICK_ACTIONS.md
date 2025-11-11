# ⚡ Quick Actions - Do This Now!

## ✅ Code Deployed to Railway

Railway is auto-deploying your web service right now (3-5 minutes).

---

## 🚀 4 Quick Steps to Complete Setup

### **1️⃣ Add Worker Service** (2 min)

**Railway Dashboard** → **New Service**:

```
1. Choose: "Deploy from GitHub repo"
2. Select: o-m7/ML_model
3. Add these environment variables:
   - POLYGON_API_KEY=your_key
   - SUPABASE_URL=your_url
   - SUPABASE_KEY=your_key
   - RAILWAY_SERVICE_NAME=worker  ⚠️ IMPORTANT!
4. Disable "Public Domain"
5. Click "Deploy"
```

---

### **2️⃣ Add Edge Column** (30 sec)

**Supabase** → **SQL Editor** → Paste & Run:

```sql
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='trades' AND column_name='edge') THEN
        ALTER TABLE trades ADD COLUMN edge DECIMAL(5,4);
    END IF;
END $$;
```

---

### **3️⃣ Enable GitHub Actions** (30 sec)

**GitHub** → **Actions** → Enable:

1. Go to: https://github.com/o-m7/ML_model/actions
2. Find: **"Session-Based Learning"** → Enable
3. Find: **"Monitor Signals & Retrain Models"** → Enable

---

### **4️⃣ Verify Deployment** (1 min)

**Test API**:
```bash
curl https://your-railway-url.railway.app/health
```

**Check Signals** (after 10 min):
```sql
-- In Supabase SQL Editor
SELECT * FROM live_signals ORDER BY timestamp DESC LIMIT 10;
```

---

## 🎯 That's It!

After these 4 steps:

✅ **API**: Serving predictions  
✅ **Worker**: Generating signals every 3 minutes  
✅ **Learning**: Improving every 4 hours  
✅ **Benchmarks**: Only elite models (Sharpe > 1.0, PF > 1.6)  

**Your institutional-grade trading system is live! 🏆**

