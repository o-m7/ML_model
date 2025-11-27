# Model-Adaptive TP/SL System

## Overview

The trading signals now use **model confidence and market analysis** to dynamically calculate TP/SL levels, not fixed multipliers.

## How It Works

### 1. Confidence-Based Adjustment

**High Confidence (≥75%)**
- TP: 2.0x → 3.0x ATR (go for bigger wins)
- SL: 1.0x → 0.8x ATR (tighter stop)
- Order: MARKET execution
- Logic: Model is very confident, maximize profit potential

**Medium Confidence (55-75%)**
- TP: 2.0x ATR (standard)
- SL: 1.0x ATR (standard)
- Order: LIMIT execution
- Logic: Balanced risk/reward

**Low Confidence (<55%)**
- TP: 2.0x → 1.4x ATR (take smaller wins)
- SL: 1.0x → 1.2x ATR (wider safety net)
- Order: LIMIT execution
- Logic: Model uncertain, be conservative

### 2. Support/Resistance Integration

When 50+ bars available:
- **BUY signals**: TP near resistance, SL below support
- **SELL signals**: TP near support, SL above resistance

Constraints:
- TP distance: 0.5x to 5x ATR (prevents extreme levels)
- SL distance: 0.3x to 2x ATR (sensible protection)

### 3. Volatility Adjustment

Recent volatility factor = recent_range / (ATR × 20)

- **High volatility (>1.5x)**: SL × 1.2 (wider stops)
- **Low volatility (<0.7x)**: SL × 0.8 (tighter stops)

## Real Example (from last run)

```
30T Quote Model: SELL at $4151.46
- Confidence: 84.4% (HIGH)
- TP: $4146.40 (5.06 distance = ~2.97 ATR adjusted)
- SL: $4153.16 (1.70 distance = ~1.0 ATR tight)
- R:R: 2.97 (excellent for high confidence)
- Order: SELL_MARKET (immediate execution)

vs.

1T OHLCV Model: SELL at $4151.46
- Confidence: 54.3% (LOW)
- TP: $4149.65 (1.81 distance = ~1.4 ATR conservative)
- SL: $4152.41 (0.95 distance = ~1.0 ATR)
- R:R: 1.9 (conservative for low confidence)
- Order: SELL_LIMIT (wait for better price)
```

## Benefits

1. **Model-Driven**: Each model's confidence directly impacts trade parameters
2. **Market-Aware**: Uses actual S/R levels, not arbitrary multiples
3. **Adaptive**: Adjusts to current volatility conditions
4. **Risk-Managed**: Low confidence = conservative, High confidence = aggressive

## Implementation

```python
def calculate_dynamic_tp_sl(
    signal_type: str,
    confidence: float,  # Model's prediction confidence
    entry_price: float,
    atr: float,
    bars_df: pd.DataFrame  # For S/R analysis
) -> tuple[tp_distance, sl_distance]
```

**Factors considered:**
1. Model confidence → Multiplier adjustment
2. Recent price range → Volatility factor
3. Support/Resistance → Intelligent level placement
4. ATR → Base measurement unit

## Model Performance Impact

**Expected Improvements:**
- High confidence trades: Better win rate with wider TPs
- Low confidence trades: Better preservation with tighter TPs
- Overall: Improved risk-adjusted returns

**Validation Needed:**
- Track actual TP/SL hit rates by confidence level
- Monitor average R:R by timeframe
- Adjust multipliers based on backtest results

## Configuration

Current settings in `live_signals_production.py`:

```python
BASE_TP_MULTIPLIER = 2.0
BASE_SL_MULTIPLIER = 1.0
HIGH_CONFIDENCE_THRESHOLD = 0.75
LOW_CONFIDENCE_THRESHOLD = 0.55
SPREAD_BUFFER = 0.5
```

## Future Enhancements

1. **ML-Predicted TP/SL**: Train separate models to predict optimal distances
2. **Time-Based Adjustment**: Adjust for session (Asian/London/NY)
3. **Multi-Timeframe Confluence**: Use higher TF S/R for lower TF trades
4. **Trailing Stops**: Dynamic SL adjustment as trade moves in profit

---

**Status**: ✅ Active in production signal generator
**File**: `live_signals_production.py`
**Method**: `calculate_dynamic_tp_sl()`
