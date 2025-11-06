# 🚀 Backend API for Lovable Integration

Complete API documentation for connecting your Lovable webapp to the ML Trading backend.

---

## 🌐 Base URLs

- **Local Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com` (after deployment)

---

## 📡 Available Endpoints

### 1. **Health Check**
Check if API is running

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_available": 25,
  "models_cached": 5,
  "available_models": ["XAUUSD_5T", "EURUSD_5T", ...],
  "supabase": "connected"
}
```

---

### 2. **List All Models**
Get all available trading models

```http
GET /models
```

**Response:**
```json
["XAUUSD_5T", "XAUUSD_15T", ..., "NZDUSD_4H"]
```

---

### 3. **Get Model Details**
Get detailed information about a specific model

```http
GET /models/{symbol}/{timeframe}
```

**Example:**
```http
GET /models/XAUUSD/5T
```

**Response:**
```json
{
  "symbol": "XAUUSD",
  "timeframe": "5T",
  "num_features": 30,
  "features": ["mom_5", "mom_10", "rsi14", ...],
  "backtest_results": {
    "win_rate": 70.4,
    "profit_factor": 2.39,
    "sharpe_ratio": 0.85,
    "max_drawdown_pct": 1.14,
    "total_trades": 1318
  },
  "parameters": {
    "tp": 1.4,
    "sl": 1.0,
    "min_conf": 0.40,
    "min_edge": 0.12,
    "pos_size": 0.4
  }
}
```

---

### 4. **Get Prediction** (Manual Features)
Get trading signal for manual feature input

```http
POST /predict
```

**Request Body:**
```json
{
  "symbol": "XAUUSD",
  "timeframe": "5T",
  "features": [0.001, -0.002, ..., 0.0015]  // 30 features
}
```

**Response:**
```json
{
  "symbol": "XAUUSD",
  "timeframe": "5T",
  "signal": "flat",
  "directional_signal": "long",  // ⭐ Always 'long' or 'short'
  "confidence": 0.523,
  "probabilities": {
    "flat": 0.382,
    "long": 0.523,
    "short": 0.095
  },
  "edge": 0.141,
  "should_trade": true,
  "signal_quality": "high",  // 'high', 'medium', or 'low'
  "backtest_metrics": {
    "win_rate": 70.4,
    "profit_factor": 2.39,
    "sharpe_ratio": 0.85,
    "max_drawdown": 1.14
  }
}
```

---

### 5. **Performance Summary**
Get aggregate performance across all models

```http
GET /performance/summary
```

**Response:**
```json
{
  "total_models": 25,
  "symbols": {
    "XAUUSD": {
      "count": 5,
      "timeframes": ["5T", "15T", "30T", "1H", "4H"]
    },
    ...
  },
  "timeframes": {
    "5T": 6,
    "15T": 5,
    ...
  },
  "aggregate_metrics": {
    "avg_win_rate": 56.3,
    "avg_profit_factor": 1.67,
    "avg_sharpe": 0.49,
    "avg_max_drawdown": 2.4
  }
}
```

---

## 🗄️ Supabase Database

### Connection Details

```javascript
const supabaseUrl = 'https://ifetofkhyblyijghuwzs.supabase.co';
const supabaseAnonKey = 'YOUR_ANON_KEY';  // Get from Supabase dashboard
```

---

### Tables

#### 1. **ml_models** - Model Metadata

```sql
SELECT * FROM ml_models ORDER BY profit_factor DESC;
```

**Columns:**
- `id` (UUID)
- `symbol` (TEXT)
- `timeframe` (TEXT)
- `win_rate` (DECIMAL)
- `profit_factor` (DECIMAL)
- `sharpe_ratio` (DECIMAL)
- `max_drawdown` (DECIMAL)
- `total_trades` (INTEGER)
- `features` (JSONB)
- `parameters` (JSONB)
- `backtest_results` (JSONB)
- `status` (TEXT)

**Example Query:**
```javascript
const { data } = await supabase
  .from('ml_models')
  .select('*')
  .order('profit_factor', { ascending: false });
```

---

#### 2. **live_signals** - Real-Time Trading Signals

```sql
SELECT * FROM live_signals WHERE status = 'active' ORDER BY timestamp DESC;
```

**Columns:**
- `id` (UUID)
- `symbol` (TEXT)
- `timeframe` (TEXT)
- `signal_type` (TEXT) - 'long', 'short', or 'flat'
- `confidence` (DECIMAL)
- `edge` (DECIMAL)
- `entry_price` (DECIMAL)
- `timestamp` (TIMESTAMPTZ)
- `status` (TEXT) - 'active', 'closed', 'expired'

**Example Query:**
```javascript
const { data } = await supabase
  .from('live_signals')
  .select('*')
  .eq('status', 'active')
  .order('timestamp', { ascending: false })
  .limit(50);
```

**Real-Time Subscription:**
```javascript
supabase
  .channel('signals')
  .on('postgres_changes', {
    event: 'INSERT',
    schema: 'public',
    table: 'live_signals'
  }, (payload) => {
    console.log('New signal:', payload.new);
    // Update your UI
  })
  .subscribe();
```

---

#### 3. **trades** - Executed Trades

```sql
SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time DESC;
```

**Columns:**
- `id` (UUID)
- `symbol` (TEXT)
- `timeframe` (TEXT)
- `direction` (TEXT) - 'long' or 'short'
- `entry_time` (TIMESTAMPTZ)
- `entry_price` (DECIMAL)
- `exit_time` (TIMESTAMPTZ)
- `exit_price` (DECIMAL)
- `pnl` (DECIMAL)
- `pnl_pct` (DECIMAL)
- `status` (TEXT) - 'open' or 'closed'

**Example Query:**
```javascript
const { data } = await supabase
  .from('trades')
  .select('*')
  .eq('status', 'closed')
  .order('exit_time', { ascending: false })
  .limit(100);
```

---

## 🔄 Live Data Flow

```
┌─────────────┐
│   Polygon   │ ← Fetches live OHLCV every 5 min
│     API     │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│live_trading_│ ← Calculates 30 features
│  engine.py  │ ← Gets predictions from models
└──────┬──────┘ ← Stores signals in Supabase
       │
       ↓
┌─────────────┐
│  Supabase   │ ← Real-time database
│  live_signals│
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Lovable   │ ← Your webapp subscribes
│   Webapp    │ ← Displays signals in real-time
└─────────────┘
```

---

## 💻 Lovable Integration Code

### 1. **Setup Supabase Client**

```javascript
// lib/supabase.js
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
```

---

### 2. **Fetch All Models**

```javascript
// Get all models from database
const fetchModels = async () => {
  const { data, error } = await supabase
    .from('ml_models')
    .select('*')
    .order('profit_factor', { ascending: false });
  
  if (error) {
    console.error('Error:', error);
    return [];
  }
  
  return data;
};
```

---

### 3. **Fetch Live Signals**

```javascript
// Get latest signals
const fetchSignals = async () => {
  const { data, error } = await supabase
    .from('live_signals')
    .select('*')
    .eq('status', 'active')
    .order('timestamp', { ascending: false })
    .limit(50);
  
  return data || [];
};
```

---

### 4. **Subscribe to Real-Time Updates**

```javascript
// Subscribe to new signals
useEffect(() => {
  const channel = supabase
    .channel('live_signals_channel')
    .on('postgres_changes', {
      event: 'INSERT',
      schema: 'public',
      table: 'live_signals'
    }, (payload) => {
      console.log('🆕 New signal:', payload.new);
      
      // Update state
      setSignals(prev => [payload.new, ...prev]);
      
      // Show notification
      if (payload.new.confidence > 0.5) {
        showNotification(payload.new);
      }
    })
    .subscribe();
  
  return () => {
    channel.unsubscribe();
  };
}, []);
```

---

### 5. **Display Signal Card**

```javascript
const SignalCard = ({ signal }) => {
  const getQualityColor = () => {
    if (signal.confidence >= 0.5 && signal.edge >= 0.1) return 'green';
    if (signal.confidence >= 0.4 && signal.edge >= 0.05) return 'yellow';
    return 'gray';
  };
  
  return (
    <div className={`signal-card ${signal.signal_type}`}>
      <div className="header">
        <h3>{signal.symbol} - {signal.timeframe}</h3>
        <span className={`quality ${getQualityColor()}`}>
          {(signal.confidence * 100).toFixed(1)}%
        </span>
      </div>
      
      <div className={`signal ${signal.signal_type}`}>
        {signal.signal_type.toUpperCase()}
      </div>
      
      <div className="metrics">
        <div>Confidence: {(signal.confidence * 100).toFixed(1)}%</div>
        <div>Edge: {(signal.edge * 100).toFixed(1)}%</div>
        <div>Entry: {signal.entry_price?.toFixed(5)}</div>
      </div>
      
      <div className="timestamp">
        {new Date(signal.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
};
```

---

## 🚀 Starting the Backend

### 1. **Start API Server**

```bash
cd /Users/omar/Desktop/ML_Trading
source .venv312/bin/activate
python3 api_server.py
```

### 2. **Start Live Trading Engine**

```bash
# Terminal 2
cd /Users/omar/Desktop/ML_Trading
source .venv312/bin/activate
python3 live_trading_engine.py
```

Or run once:
```bash
python3 live_trading_engine.py once
```

---

## 🔑 Environment Variables for Lovable

```env
VITE_SUPABASE_URL=https://ifetofkhyblyijghuwzs.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGc...
VITE_API_URL=http://localhost:8000
```

---

## 📊 Example: Complete Dashboard

```javascript
import { useState, useEffect } from 'react';
import { supabase } from './lib/supabase';

function Dashboard() {
  const [signals, setSignals] = useState([]);
  const [models, setModels] = useState([]);
  
  useEffect(() => {
    // Load models
    loadModels();
    
    // Load signals
    loadSignals();
    
    // Subscribe to updates
    const channel = supabase
      .channel('dashboard')
      .on('postgres_changes', {
        event: 'INSERT',
        schema: 'public',
        table: 'live_signals'
      }, handleNewSignal)
      .subscribe();
    
    return () => channel.unsubscribe();
  }, []);
  
  const loadModels = async () => {
    const { data } = await supabase
      .from('ml_models')
      .select('*');
    setModels(data || []);
  };
  
  const loadSignals = async () => {
    const { data } = await supabase
      .from('live_signals')
      .select('*')
      .eq('status', 'active')
      .order('timestamp', { ascending: false });
    setSignals(data || []);
  };
  
  const handleNewSignal = (payload) => {
    setSignals(prev => [payload.new, ...prev]);
  };
  
  return (
    <div>
      <h1>Live Trading Signals</h1>
      <div className="stats">
        <div>Models: {models.length}</div>
        <div>Active Signals: {signals.length}</div>
      </div>
      
      <div className="signals-grid">
        {signals.map(signal => (
          <SignalCard key={signal.id} signal={signal} />
        ))}
      </div>
    </div>
  );
}
```

---

## 🎯 Summary

Your Lovable frontend needs to:

1. ✅ **Connect to Supabase** using the anon key
2. ✅ **Query `ml_models`** table for model metadata
3. ✅ **Query `live_signals`** table for real-time signals
4. ✅ **Subscribe to INSERT events** on `live_signals` for updates
5. ✅ **Display signals** with quality indicators

**Backend handles:**
- Fetching data from Polygon
- Calculating features
- Getting predictions
- Storing in Supabase

**Frontend (Lovable) handles:**
- Displaying data
- Real-time updates
- User interactions

---

That's it! Your backend is ready for Lovable integration! 🚀

