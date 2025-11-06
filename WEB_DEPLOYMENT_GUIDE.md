# 🚀 Web Deployment Guide - ML Trading System

Complete guide to deploying your ML models to Supabase and integrating with Lovable webapp.

---

## 📋 Prerequisites

- ✅ Python environment activated
- ✅ `.env` file configured with Supabase credentials
- ✅ 25 production-ready models trained
- ✅ Supabase account created

---

## 🎯 STEP-BY-STEP DEPLOYMENT

### **STEP 1: Install Dependencies** ⚙️

```bash
cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate
pip install -r requirements_api.txt
```

**Expected time:** 2-3 minutes

---

### **STEP 2: Convert Models to ONNX** 🔄

Convert all 25 production-ready models to ONNX format (web-compatible):

```bash
python3 convert_models_to_onnx.py
```

**What this does:**
- Converts each LightGBM model to ONNX format
- Creates metadata JSON for each model
- Verifies each conversion works
- Saves to `models_onnx/` directory

**Expected output:**
```
✅ XAUUSD 5T  - 30 features, WR: 55.8%, PF: 1.79
✅ XAUUSD 15T - 30 features, WR: 52.3%, PF: 1.58
...
CONVERSION COMPLETE: 25 models converted
```

**Expected time:** 2-5 minutes

---

### **STEP 3: Set Up Supabase Database** 🗄️

1. **Go to your Supabase project:** https://app.supabase.com/project/YOUR_PROJECT

2. **Open SQL Editor** (left sidebar)

3. **Copy and paste** the entire contents of `supabase_schema.sql`

4. **Run the SQL script**

This creates:
- `ml_models` table (stores model metadata)
- `live_signals` table (stores trading signals)
- `trades` table (stores executed trades)
- `performance_metrics` table (aggregated stats)

5. **Create Storage Bucket:**
   - Go to Storage → Buckets
   - Click "New bucket"
   - Name: `ml_models`
   - Public: ✅ Yes
   - Click "Create bucket"

**Expected time:** 5 minutes

---

### **STEP 4: Sync Models to Supabase** ☁️

Upload all ONNX models and metadata to Supabase:

```bash
python3 supabase_sync.py
```

**What this does:**
- Uploads each ONNX model to Supabase Storage
- Inserts model metadata into `ml_models` table
- Makes models accessible via API

**Expected output:**
```
✅ Uploaded models/XAUUSD/XAUUSD_5T.onnx
✅ Inserted XAUUSD 5T into database
...
SYNC COMPLETE: 25 models synced
```

**Expected time:** 5-10 minutes

---

### **STEP 5: Start API Server** 🌐

Launch the FastAPI server:

```bash
python3 api_server.py
```

**Expected output:**
```
Starting ML Trading API Server
✅ Connected to Supabase
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Server will be running at:**
- Local: http://localhost:8000
- Network: http://YOUR_IP:8000

**Keep this terminal open!**

---

### **STEP 6: Test API Endpoints** 🧪

Open a **new terminal** and test:

```bash
# Health check
curl http://localhost:8000/health

# List all models
curl http://localhost:8000/models

# Get model info
curl http://localhost:8000/models/XAUUSD/5T

# Get performance summary
curl http://localhost:8000/performance/summary
```

**Expected response:**
```json
{
  "status": "healthy",
  "models_available": 25,
  "models_cached": 0,
  "available_models": ["XAUUSD_5T", "XAUUSD_15T", ...]
}
```

---

### **STEP 7: Test Predictions** 🎯

Create a test script to verify predictions work:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "XAUUSD",
    "timeframe": "5T",
    "features": [0.001, -0.002, 0.0015, 0.003, -0.001, 0.002, 0.0005, -0.0008, 0.0012, 0.0018,
                 0.0007, -0.0003, 0.0009, 0.0011, -0.0006, 0.0014, 0.0004, -0.0002, 0.0016, 0.0013,
                 0.0019, -0.0004, 0.0008, 0.0021, -0.0009, 0.0017, 0.0006, -0.0005, 0.0022, 0.0015]
  }'
```

**Expected response:**
```json
{
  "symbol": "XAUUSD",
  "timeframe": "5T",
  "signal": "long",
  "confidence": 0.5234,
  "probabilities": {
    "flat": 0.2145,
    "long": 0.5234,
    "short": 0.2621
  },
  "edge": 0.2613,
  "should_trade": true,
  "backtest_metrics": {
    "win_rate": 55.8,
    "profit_factor": 1.79,
    "sharpe_ratio": 0.45,
    "max_drawdown": 3.2
  }
}
```

---

## 🌐 STEP 8: Integrate with Lovable Webapp

### **Frontend Integration (React/Next.js)**

```javascript
// Example: Fetch available models
const getModels = async () => {
  const response = await fetch('http://YOUR_API_URL:8000/models');
  const models = await response.json();
  return models;
};

// Example: Get prediction
const getPrediction = async (symbol, timeframe, features) => {
  const response = await fetch('http://YOUR_API_URL:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      symbol,
      timeframe,
      features
    })
  });
  
  const prediction = await response.json();
  
  if (prediction.should_trade) {
    console.log(`🎯 ${prediction.signal.toUpperCase()} signal!`);
    console.log(`   Confidence: ${(prediction.confidence * 100).toFixed(1)}%`);
    console.log(`   Edge: ${(prediction.edge * 100).toFixed(1)}%`);
  }
  
  return prediction;
};

// Example: Get performance summary
const getPerformance = async () => {
  const response = await fetch('http://YOUR_API_URL:8000/performance/summary');
  const summary = await response.json();
  return summary;
};

// Example: Display model info
const ModelCard = ({ symbol, timeframe }) => {
  const [modelInfo, setModelInfo] = useState(null);
  
  useEffect(() => {
    fetch(`http://YOUR_API_URL:8000/models/${symbol}/${timeframe}`)
      .then(r => r.json())
      .then(setModelInfo);
  }, [symbol, timeframe]);
  
  if (!modelInfo) return <div>Loading...</div>;
  
  return (
    <div className="model-card">
      <h3>{symbol} - {timeframe}</h3>
      <div className="metrics">
        <p>Win Rate: {modelInfo.backtest_results.win_rate.toFixed(1)}%</p>
        <p>Profit Factor: {modelInfo.backtest_results.profit_factor.toFixed(2)}</p>
        <p>Sharpe: {modelInfo.backtest_results.sharpe_ratio.toFixed(2)}</p>
        <p>Max DD: {modelInfo.backtest_results.max_drawdown_pct.toFixed(1)}%</p>
      </div>
    </div>
  );
};
```

### **Connect to Supabase from Frontend**

```javascript
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
);

// Get all models from database
const { data: models, error } = await supabase
  .from('ml_models')
  .select('*')
  .eq('status', 'production_ready');

// Get active signals
const { data: signals, error } = await supabase
  .from('live_signals')
  .select('*')
  .eq('status', 'active')
  .order('timestamp', { ascending: false });

// Get recent trades
const { data: trades, error } = await supabase
  .from('trades')
  .select('*')
  .order('entry_time', { ascending: false })
  .limit(50);

// Real-time updates (live signals)
supabase
  .channel('live_signals')
  .on('postgres_changes', 
      { event: 'INSERT', schema: 'public', table: 'live_signals' },
      (payload) => {
        console.log('🔔 New signal:', payload.new);
        // Update UI with new signal
      }
  )
  .subscribe();
```

---

## 📊 API Endpoints Reference

### **Core Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/models` | GET | List all available models |
| `/models/{symbol}/{timeframe}` | GET | Get model metadata |
| `/predict` | POST | Get prediction from model |
| `/performance/summary` | GET | Overall performance stats |
| `/signals/active` | GET | Active trading signals |
| `/trades/recent` | GET | Recent trades (limit=50) |

---

## 🚀 Production Deployment Options

### **Option 1: Deploy to Railway/Render**

1. Push code to GitHub
2. Connect Railway/Render to repo
3. Set environment variables
4. Deploy!

### **Option 2: Deploy to Vercel (Serverless)**

1. Convert to serverless functions
2. Deploy via `vercel` CLI
3. Set environment variables

### **Option 3: Deploy to DigitalOcean/AWS**

1. Create VM/EC2 instance
2. Install dependencies
3. Run with systemd/PM2
4. Configure nginx reverse proxy

---

## 🔒 Security Checklist

- [ ] Update CORS origins in `api_server.py` with your Lovable domain
- [ ] Set up Supabase Row Level Security (RLS) policies
- [ ] Use environment variables for all secrets
- [ ] Enable rate limiting on API
- [ ] Set up SSL/HTTPS for production
- [ ] Monitor API usage and errors

---

## 📈 Monitoring

### **Check API Status**

```bash
curl http://localhost:8000/health
```

### **View Supabase Logs**
- Go to Supabase Dashboard → Logs
- Check API requests, database queries, storage access

### **Monitor Model Performance**
- Query `performance_metrics` table
- Track live signals vs closed trades
- Calculate rolling win rates

---

## 🐛 Troubleshooting

### **Models not loading?**
```bash
# Check if ONNX files exist
ls -la models_onnx/
```

### **Supabase connection failed?**
```bash
# Verify environment variables
echo $SUPABASE_URL
echo $SUPABASE_KEY
```

### **CORS errors?**
- Update `allow_origins` in `api_server.py` with your webapp domain

### **Predictions failing?**
- Verify you're sending exactly 30 features
- Check feature order matches model training

---

## ✅ Final Checklist

- [x] Dependencies installed
- [x] Models converted to ONNX
- [x] Supabase database set up
- [x] Models synced to Supabase
- [x] API server running
- [x] Endpoints tested
- [ ] Integrated with Lovable webapp
- [ ] Deployed to production
- [ ] Monitoring enabled

---

## 🎉 You're Ready!

Your ML trading system is now:
- ✅ **Web-compatible** (ONNX models)
- ✅ **Cloud-hosted** (Supabase)
- ✅ **API-accessible** (FastAPI)
- ✅ **Ready for Lovable** (CORS enabled)

**Need help?** Check the API docs at: http://localhost:8000/docs

