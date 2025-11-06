# 🌐 Lovable Webapp Integration

Complete integration code for your ML Trading System in Lovable.

## 📋 Setup

1. **Install Dependencies**

```bash
npm install @supabase/supabase-js axios recharts lucide-react
```

2. **Environment Variables**

Create `.env.local` in your Lovable project:

```env
VITE_SUPABASE_URL=https://ifetofkhyblyijghuwzs.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key_here
VITE_API_URL=http://localhost:8000
```

3. **Copy Components**

Copy all files from this directory to your Lovable `src/components/` folder.

---

## 🎯 Components Included

### 1. **SignalDashboard.tsx** - Main Dashboard
Real-time signals from all 25 models

### 2. **ModelCard.tsx** - Individual Model Display
Shows signal, quality, confidence for each model

### 3. **TradeList.tsx** - Recent Trades
Display recent trades with P&L

### 4. **PerformanceChart.tsx** - Performance Visualization
Charts showing model performance over time

### 5. **supabaseClient.ts** - Supabase Setup
Database connection configuration

---

## 🚀 Quick Start

### Minimal Implementation

```tsx
import { SignalDashboard } from './components/SignalDashboard';

function App() {
  return (
    <div className="app">
      <h1>ML Trading System</h1>
      <SignalDashboard />
    </div>
  );
}
```

---

## 📊 Full Dashboard Example

```tsx
import { useState, useEffect } from 'react';
import { supabase } from './lib/supabaseClient';
import { SignalDashboard } from './components/SignalDashboard';
import { TradeList } from './components/TradeList';
import { PerformanceChart } from './components/PerformanceChart';

function TradingDashboard() {
  const [models, setModels] = useState([]);
  const [signals, setSignals] = useState([]);
  
  useEffect(() => {
    // Load models
    loadModels();
    
    // Subscribe to real-time signals
    const signalSubscription = supabase
      .channel('live_signals')
      .on('postgres_changes', {
        event: 'INSERT',
        schema: 'public',
        table: 'live_signals'
      }, (payload) => {
        console.log('New signal:', payload.new);
        setSignals(prev => [payload.new, ...prev]);
      })
      .subscribe();
    
    return () => {
      signalSubscription.unsubscribe();
    };
  }, []);
  
  const loadModels = async () => {
    const { data } = await supabase
      .from('ml_models')
      .select('*')
      .order('profit_factor', { ascending: false });
    
    setModels(data);
  };
  
  return (
    <div className="dashboard">
      <header>
        <h1>ML Trading Dashboard</h1>
        <div className="stats">
          <div>Models: {models.length}</div>
          <div>Active Signals: {signals.length}</div>
        </div>
      </header>
      
      <SignalDashboard />
      
      <div className="grid-2">
        <TradeList />
        <PerformanceChart />
      </div>
    </div>
  );
}
```

---

## 📱 Mobile Responsive

All components are mobile-responsive and will adapt to screen size.

---

## 🎨 Customization

### Colors

```css
:root {
  --signal-high: #10b981;   /* Green */
  --signal-medium: #f59e0b; /* Orange */
  --signal-low: #6b7280;    /* Gray */
  --long: #10b981;
  --short: #ef4444;
}
```

### Themes

Components support light and dark mode out of the box.

---

## 🔄 Real-Time Updates

### Auto-refresh Signals

```tsx
useEffect(() => {
  const interval = setInterval(() => {
    loadSignals();
  }, 30000); // Every 30 seconds
  
  return () => clearInterval(interval);
}, []);
```

### Push Notifications

```tsx
if (signal.signal_quality === 'high') {
  new Notification('High Quality Signal!', {
    body: `${signal.symbol} ${signal.timeframe}: ${signal.directional_signal.toUpperCase()}`,
    icon: '/logo.png'
  });
}
```

---

## 📚 API Endpoints Used

- `GET /health` - Check API status
- `GET /models` - List all models
- `GET /models/{symbol}/{timeframe}` - Get model details
- `POST /predict` - Get prediction (not used in dashboard)
- `GET /performance/summary` - Overall stats

---

## 🗄️ Supabase Tables Used

- `ml_models` - Model metadata and backtest results
- `live_signals` - Real-time trading signals
- `trades` - Executed trades history
- `performance_metrics` - Performance tracking

---

## 🎯 Features

✅ Real-time signal updates
✅ 25 models displayed
✅ Color-coded by quality (high/medium/low)
✅ Automatic refresh
✅ Trade history
✅ Performance charts
✅ Mobile responsive
✅ Dark mode support
✅ WebSocket updates

---

## 🐛 Troubleshooting

### Signals not updating?

Check if live trading engine is running:
```bash
python3 live_trading_engine.py
```

### API connection failed?

Verify API is running:
```bash
curl http://localhost:8000/health
```

### Supabase errors?

Check environment variables are set correctly.

---

## 📖 Next Steps

1. Deploy live trading engine to cloud
2. Set up automated alerts
3. Add trade execution
4. Implement portfolio tracking
5. Add risk management controls

Enjoy your live ML trading dashboard! 🚀

