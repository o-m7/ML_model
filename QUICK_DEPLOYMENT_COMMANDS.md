# ⚡ Quick Deployment Commands

## 🚀 AUTOMATED DEPLOYMENT (Recommended)

```bash
cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate
./deploy_to_web.sh
```

---

## 📝 MANUAL STEP-BY-STEP

### 1️⃣ Install Dependencies

```bash
cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate
pip install -r requirements_api.txt
```

### 2️⃣ Convert Models to ONNX

```bash
python3 convert_models_to_onnx.py
```

### 3️⃣ Set Up Supabase Database

1. Go to: https://app.supabase.com/project/YOUR_PROJECT
2. Open **SQL Editor**
3. Copy/paste **supabase_schema.sql**
4. Click **Run**
5. Go to **Storage** → Create bucket: `ml_models` (public)

### 4️⃣ Sync Models to Supabase

```bash
python3 supabase_sync.py
```

### 5️⃣ Start API Server

```bash
python3 api_server.py
```

### 6️⃣ Test API

**New terminal:**

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Performance summary
curl http://localhost:8000/performance/summary
```

---

## 🧪 TEST PREDICTION

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

---

## 🌐 LOVABLE INTEGRATION

### JavaScript Example

```javascript
// Get all models
const models = await fetch('http://YOUR_API:8000/models').then(r => r.json());

// Get prediction
const prediction = await fetch('http://YOUR_API:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    symbol: 'XAUUSD',
    timeframe: '5T',
    features: [/* 30 features */]
  })
}).then(r => r.json());

if (prediction.should_trade) {
  alert(`${prediction.signal.toUpperCase()} - ${prediction.confidence}% confident`);
}
```

---

## 📊 USEFUL ENDPOINTS

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Health check |
| `http://localhost:8000/docs` | Interactive API docs |
| `http://localhost:8000/models` | List all models |
| `http://localhost:8000/models/XAUUSD/5T` | Model info |
| `http://localhost:8000/performance/summary` | Performance stats |

---

## 🔧 TROUBLESHOOTING

### API won't start?
```bash
# Check if port 8000 is in use
lsof -i :8000
# Kill if needed
kill -9 <PID>
```

### Models not found?
```bash
# Verify ONNX files exist
ls -la models_onnx/*/
```

### Supabase errors?
```bash
# Check environment variables
echo $SUPABASE_URL
echo $SUPABASE_KEY
```

---

## 📚 Full Guide

See **WEB_DEPLOYMENT_GUIDE.md** for complete documentation.

