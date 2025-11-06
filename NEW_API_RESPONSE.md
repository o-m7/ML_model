# 🎯 Updated API - Always Returns Directional Signals

## ✅ What Changed

The API now **ALWAYS returns a directional signal** (LONG or SHORT), even when the model predicts FLAT or confidence is low.

---

## 📊 New Response Format

```json
{
  "symbol": "XAUUSD",
  "timeframe": "5T",
  
  // What the model actually predicts (can be 'flat', 'long', or 'short')
  "signal": "flat",
  
  // ✨ NEW: Always directional - never 'flat'
  "directional_signal": "short",  // ALWAYS 'long' or 'short'
  
  "confidence": 0.382,
  "probabilities": {
    "flat": 0.382,  // 38.2%
    "long": 0.299,  // 29.9%
    "short": 0.319  // 31.9% <- higher than long, so directional_signal = 'short'
  },
  "edge": 0.064,
  
  // Strict quality check (original logic)
  "should_trade": false,
  
  // ✨ NEW: Signal quality rating
  "signal_quality": "low",  // 'high', 'medium', or 'low'
  
  "backtest_metrics": {
    "win_rate": 70.4,
    "profit_factor": 2.39,
    "sharpe_ratio": 0.85,
    "max_drawdown": 1.1
  }
}
```

---

## 🎯 Signal Quality Levels

### 🟢 **HIGH** Quality
- Model predicts directional (not flat)
- Confidence > threshold (38-40%)
- Edge > threshold (8-12%)
- **should_trade = true**
- ✅ **Safe to trade**

### 🟡 **MEDIUM** Quality
- Directional confidence ≥ 40%
- Edge ≥ 5%
- But doesn't meet strict thresholds
- **should_trade = false**
- ⚠️ **Consider with caution**

### 🔴 **LOW** Quality
- Below medium thresholds
- Weak conviction
- **should_trade = false**
- ❌ **Not recommended**

---

## 💻 Usage in Your Lovable Webapp

### Example 1: Always Show Directional Signal

```javascript
const signal = await fetch('http://YOUR_API:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    symbol: 'XAUUSD',
    timeframe: '5T',
    features: realMarketFeatures
  })
}).then(r => r.json());

// ✨ Now you ALWAYS get a directional signal
console.log(`Direction: ${signal.directional_signal.toUpperCase()}`);
console.log(`Quality: ${signal.signal_quality.toUpperCase()}`);

// Display in UI with quality indicator
if (signal.signal_quality === 'high') {
  showSignal(`🟢 ${signal.directional_signal.toUpperCase()}`, 'success');
} else if (signal.signal_quality === 'medium') {
  showSignal(`🟡 ${signal.directional_signal.toUpperCase()}`, 'warning');
} else {
  showSignal(`🔴 ${signal.directional_signal.toUpperCase()}`, 'info');
}
```

### Example 2: Color-Coded UI

```jsx
function SignalDisplay({ signal }) {
  const getColor = () => {
    switch(signal.signal_quality) {
      case 'high': return 'green';
      case 'medium': return 'orange';
      case 'low': return 'gray';
    }
  };
  
  const getIcon = () => {
    switch(signal.signal_quality) {
      case 'high': return '🟢';
      case 'medium': return '🟡';
      case 'low': return '🔴';
    }
  };
  
  return (
    <div className={`signal-card ${getColor()}`}>
      <h2>{signal.symbol} - {signal.timeframe}</h2>
      
      {/* Always show directional signal */}
      <div className="signal-direction">
        {getIcon()} {signal.directional_signal.toUpperCase()}
      </div>
      
      <div className="confidence">
        {(signal.confidence * 100).toFixed(1)}% confident
      </div>
      
      <div className="quality">
        Quality: {signal.signal_quality.toUpperCase()}
      </div>
      
      {/* Show if it's a tradeable signal */}
      {signal.should_trade && (
        <button className="trade-btn">
          Execute Trade 🚀
        </button>
      )}
      
      {/* Show probabilities */}
      <div className="probabilities">
        <div>Long: {(signal.probabilities.long * 100).toFixed(1)}%</div>
        <div>Short: {(signal.probabilities.short * 100).toFixed(1)}%</div>
        <div>Flat: {(signal.probabilities.flat * 100).toFixed(1)}%</div>
      </div>
    </div>
  );
}
```

### Example 3: Auto-Refresh Dashboard

```javascript
// Poll API every 5 seconds for all models
async function startSignalStream() {
  const models = await fetch('http://YOUR_API:8000/models').then(r => r.json());
  
  setInterval(async () => {
    for (const modelKey of models) {
      const [symbol, timeframe] = modelKey.split('_');
      
      const features = await getLatestMarketFeatures(symbol, timeframe);
      
      const signal = await fetch('http://YOUR_API:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, timeframe, features })
      }).then(r => r.json());
      
      // Update UI with new signal
      updateDashboard(signal);
      
      // Alert on high-quality signals
      if (signal.signal_quality === 'high') {
        notify(`🚨 ${symbol} ${timeframe}: ${signal.directional_signal.toUpperCase()}`);
      }
    }
  }, 5000); // Every 5 seconds
}
```

---

## 🎨 UI Design Suggestions

### Signal Card Layout

```
┌─────────────────────────────────┐
│ XAUUSD - 5T                     │
│                                 │
│     🟢 LONG                     │
│     Quality: HIGH               │
│     Confidence: 52.3%           │
│                                 │
│ Probabilities:                  │
│ Long:  52.3% ████████████       │
│ Short: 28.1% ██████             │
│ Flat:  19.6% ████               │
│                                 │
│ [EXECUTE TRADE] 🚀              │
└─────────────────────────────────┘
```

### Dashboard Grid

```
┌──────────┬──────────┬──────────┐
│ 🟢 XAUUSD │ 🟡 EURUSD│ 🔴 GBPUSD│
│   5T     │   5T     │   15T    │
│   LONG   │   SHORT  │   LONG   │
│   HIGH   │  MEDIUM  │   LOW    │
└──────────┴──────────┴──────────┘
```

---

## ⚠️ Important Notes

1. **directional_signal** - ALWAYS long or short (based on highest directional probability)
2. **signal** - What model actually predicts (can include 'flat')
3. **should_trade** - Strict quality filter (use this for live trading)
4. **signal_quality** - Visual indicator of signal strength

---

## 🔧 Technical Details

### How directional_signal is Determined

```python
# Compare long vs short probabilities
if probabilities['long'] > probabilities['short']:
    directional_signal = 'long'
else:
    directional_signal = 'short'
```

### How signal_quality is Determined

```python
if should_trade:
    signal_quality = 'high'
elif directional_confidence >= 0.40 and edge >= 0.05:
    signal_quality = 'medium'
else:
    signal_quality = 'low'
```

---

## 🚀 Ready to Use

The API is now perfect for a live dashboard that needs to **always show a signal**, with clear visual indicators of quality!

**Test it:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "XAUUSD",
    "timeframe": "5T",
    "features": [/* your 30 features */]
  }'
```

You'll always get a `directional_signal` field that's either "long" or "short"! 🎯

