# POLYGON API INTEGRATION GUIDE

Complete guide for using Polygon API to fetch live data and generate trading signals.

---

## 🔑 Setup

### 1. Get Polygon API Key

1. Sign up at [polygon.io](https://polygon.io)
2. Go to Dashboard → API Keys
3. Copy your API key

### 2. Create .env File

Create a `.env` file in `/Users/omar/Desktop/ML_Trading/`:

```bash
POLYGON_API_KEY=your_actual_api_key_here
```

### 3. Install Dependencies

```bash
pip install python-dotenv requests
```

---

## 🚀 Quick Start

### Generate a Live Signal

```python
from intraday_system.live import generate_live_signal

# Generate signal with live Polygon data
signal = generate_live_signal(
    symbol='XAUUSD',
    timeframe='15T',
    account_equity=100000
)

# Check signal
if signal['signal'] == 'BUY':
    print(f"🟢 BUY at {signal['entry_ref']}")
    print(f"Stop Loss: {signal['stop_loss']}")
    print(f"Take Profit: {signal['take_profit']}")
    print(f"Position Size: {signal['position_sizing']['position_size']}")
```

---

## 📊 Fetch Live Data Only

### Basic Usage

```python
from intraday_system.live import get_live_data

# Fetch latest 200 bars
df = get_live_data(
    symbol='XAUUSD',
    timeframe='15T',
    n_bars=200
)

print(df.tail())
```

### Advanced Usage

```python
from intraday_system.live import PolygonDataFetcher

# Initialize fetcher
fetcher = PolygonDataFetcher(api_key='your_key')  # Or uses .env

# Fetch data for multiple symbols
symbols = ['XAUUSD', 'EURUSD', 'GBPUSD']
for symbol in symbols:
    df = fetcher.fetch_latest_bars(symbol, '15T', n_bars=100)
    print(f"{symbol}: {len(df)} bars")
```

---

## 🔄 Complete Pipeline

### End-to-End Signal Generation

```python
from intraday_system.live import LiveSignalGenerator

# Initialize generator
generator = LiveSignalGenerator(
    models_dir='models_intraday',
    config_path='intraday_system/config/settings.yaml'
)

# Generate signal with all filters
signal = generator.generate_signal(
    symbol='XAUUSD',
    timeframe='15T',
    account_equity=100000,
    apply_post_filters=True
)

# Signal includes:
# - ML prediction
# - Position sizing
# - Cooldown filter
# - Spread filter
# - Risk management

print(signal)
```

### Signal Structure

```python
{
    'signal': 'BUY',  # BUY, SELL, or HOLD
    'confidence': 0.73,
    'entry_ref': 2651.50,
    'stop_loss': 2648.30,
    'take_profit': 2656.10,
    'expected_R': 1.44,
    'horizon_bars': 8,
    'atr': 3.20,
    'probabilities': {
        'flat': 0.15,
        'up': 0.73,
        'down': 0.12
    },
    'position_sizing': {
        'position_size': 31.25,
        'position_value': 82843.75,
        'risk_amount': 1000.00,
        'stop_distance': 3.20,
        'risk_reward_ratio': 1.44
    },
    'filtered': False,
    'generated_at': '2025-11-03T14:30:00Z',
    'data_source': 'Polygon API',
    'latest_price': 2651.50
}
```

---

## 🎯 Supported Symbols

### Forex Pairs
- EURUSD
- GBPUSD
- USDJPY
- USDCAD
- AUDUSD
- NZDUSD
- USDCHF

### Metals
- XAUUSD (Gold)
- XAGUSD (Silver)

### Stocks
- Any US stock ticker (AAPL, TSLA, etc.)

---

## ⏱️ Supported Timeframes

- **5T** - 5 minutes
- **15T** - 15 minutes
- **30T** - 30 minutes
- **1H** - 1 hour
- **2H** - 2 hours
- **4H** - 4 hours

---

## 🔒 API Limits

### Free Tier
- 5 API calls per minute
- 2 years historical data
- Real-time US stocks only

### Starter ($29/month)
- 100 calls per minute
- Forex & Crypto data
- 5+ years historical

### Developer ($99/month)
- Unlimited calls
- All asset classes
- Full historical data

---

## 🛡️ Error Handling

```python
from intraday_system.live import generate_live_signal

try:
    signal = generate_live_signal('XAUUSD', '15T')
    
    if 'error' in signal:
        print(f"Error: {signal['error']}")
    elif signal['signal'] == 'HOLD':
        print("No trade opportunity")
    else:
        # Execute trade
        print(f"Signal: {signal['signal']}")
        
except Exception as e:
    print(f"Failed to generate signal: {e}")
```

---

## 📝 Example: Live Trading Loop

```python
import time
from intraday_system.live import LiveSignalGenerator

# Initialize
generator = LiveSignalGenerator()

# Track state
account_equity = 100000
symbols = ['XAUUSD', 'EURUSD']
timeframe = '15T'

while True:
    for symbol in symbols:
        try:
            # Generate signal
            signal = generator.generate_signal(
                symbol=symbol,
                timeframe=timeframe,
                account_equity=account_equity
            )
            
            # Act on signal
            if signal['signal'] in ['BUY', 'SELL'] and not signal.get('filtered'):
                print(f"\n🚨 {signal['signal']} SIGNAL: {symbol}")
                # Send to broker API here...
                
                # Update state after trade
                current_bar = signal['bars_used'] - 1
                generator.update_trade_state(symbol, timeframe, current_bar)
            
        except Exception as e:
            print(f"Error with {symbol}: {e}")
    
    # Wait for next bar (15 minutes)
    time.sleep(900)
```

---

## 🔍 Debugging

### Check Data Quality

```python
from intraday_system.live import get_live_data

df = get_live_data('XAUUSD', '15T', n_bars=100)

print("Data Quality Check:")
print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Null values: {df.isnull().sum().sum()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nSample data:")
print(df.tail())
```

### Test API Connection

```python
from intraday_system.live import PolygonDataFetcher

fetcher = PolygonDataFetcher()
print("API Key loaded:", "Yes" if fetcher.api_key else "No")

# Test fetch
try:
    df = fetcher.fetch_latest_bars('XAUUSD', '15T', n_bars=10)
    print("✓ API connection successful")
except Exception as e:
    print(f"✗ API connection failed: {e}")
```

---

## ⚙️ Configuration

### Override Default Settings

```python
# In your .env file:
POLYGON_API_KEY=your_key
POLYGON_BASE_URL=https://api.polygon.io/v2
POLYGON_TIMEOUT=30
```

### Custom Fetcher

```python
from intraday_system.live import PolygonDataFetcher

fetcher = PolygonDataFetcher(api_key='custom_key')

# Fetch with custom parameters
df = fetcher.fetch_latest_bars(
    symbol='EURUSD',
    timeframe='5T',
    n_bars=500,
    adjusted=True  # Use adjusted prices
)
```

---

## 📞 Support

**Polygon Documentation**: [polygon.io/docs](https://polygon.io/docs)

**Common Issues**:

1. **API Key Error**: Check `.env` file exists and key is correct
2. **No Data Returned**: Symbol may not be supported or market closed
3. **Rate Limit**: Upgrade plan or reduce request frequency
4. **Timeframe Error**: Use supported formats (5T, 15T, 1H, etc.)

---

## ✅ Quick Test

```bash
cd /Users/omar/Desktop/ML_Trading

# Test data fetching
python -c "from intraday_system.live import get_live_data; \
           df = get_live_data('XAUUSD', '15T', 50); \
           print(f'Success! Got {len(df)} bars')"

# Test signal generation
python intraday_system/live/live_signal_generator.py
```

---

**Your live trading system with Polygon API is ready!** 🚀

Use `generate_live_signal()` for complete end-to-end signal generation with live data.

