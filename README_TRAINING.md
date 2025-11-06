# ML Trading Signal Engine - Training Guide

## REALISTIC RESULTS ACHIEVED ✅

Your system now has **PROPER** bar-by-bar trade simulation. No more 99% win rate fantasy.

### Current Results (XAUUSD_1H):
```
Win Rate:       80.77%   ✅ (Target: >50%)
Profit Factor:  0.44     ❌ (Target: >1.6)
Max DD:         3.44%    ✅ (Target: <6%)
Expectancy:     -0.088R  ❌ (Target: >0.08R)
Avg Win:        0.09R    ⚠️  (Too small!)
Avg Loss:       -0.82R   ⚠️  (Full stop hit)
```

**Status**: PAPER_ONLY (not profitable yet)

---

## The Problem

The model catches 80% winners but they're tiny (0.09R avg) while losses hit the full stop (-0.82R).

**Why?**  
- Target TP = 2.5R (middle of your 1:2 to 1:3 range)
- But price rarely reaches 2.5R before reversing
- So wins close early for small gains, losses hit full stop

---

## How to Fix - Adjust R:R Targets

### Option 1: Lower TP Target (Recommended)
Edit `config.yaml`:
```yaml
risk:
  min_risk_reward: 1.2    # Lower from 2.0
  max_risk_reward: 1.8    # Lower from 3.0
```

This targets 1.2R to 1.8R (more realistic for gold).

### Option 2: Tighter Stop Loss
```yaml
symbols:
  XAUUSD:
    min_atr_multiple: 1.0  # Tighter stop (from 1.5)
```

Smaller stops mean better R:R on same price moves.

### Option 3: Train Separate Models
Train one model for LONG, one for SHORT with different parameters.

---

## Training Files Location

All files are in: `/home/claude/ML_Trading/`

### Main Training Script:
```bash
cd /home/claude/ML_Trading
python3 train.py --symbol XAUUSD --timeframe 1H
```

### Configuration:
```bash
nano config.yaml
```

### View Results:
```bash
cat outputs/metrics/XAUUSD_1H_metrics.json
```

---

## Key Files You Need:

1. **train.py** - Main training orchestrator
2. **config.yaml** - All parameters (edit this!)
3. **models/target_constructor.py** - Creates realistic labels with bar-by-bar checking
4. **models/threshold_optimizer.py** - Simulates trades realistically
5. **models/trainer.py** - Walk-forward CV training
6. **models/feature_engineer.py** - Feature selection
7. **models/utils.py** - Utilities

---

## Quick Start - Retrain with Better Parameters

```bash
cd /home/claude/ML_Trading

# Edit config
nano config.yaml
# Change min_risk_reward to 1.2 and max_risk_reward to 1.8

# Retrain
python3 train.py --symbol XAUUSD --timeframe 1H

# Check results
cat outputs/metrics/XAUUSD_1H_metrics.json | grep -A5 threshold_metrics
```

---

## Target Metrics You're Aiming For:

```
Win Rate:       50-60%
Profit Factor:  1.6-2.5
Max DD:         2-5%
Expectancy:     0.08-0.20R
Avg Win:        1.2-1.8R
Avg Loss:       -0.8 to -1.0R
```

---

## What's FIXED:

✅ Bar-by-bar trade simulation (no more cheating)  
✅ Realistic win rates (not 99%)  
✅ Proper stop loss checking  
✅ Proper take profit checking  
✅ Fixed position sizing (no exponential growth)  
✅ Walk-forward cross-validation  
✅ No data leakage  

---

## Next Steps:

1. **Adjust R:R targets in config.yaml** (set to 1.2-1.8)
2. **Retrain the model**
3. **Check if PF > 1.6 and Expectancy > 0.08R**
4. **If yes**: Move to Step 2 (Backtesting)
5. **If no**: Try tighter stops or different features

---

## Files You Can Download:

All training output files are in:
```
/home/claude/ML_Trading/outputs/
```

Your trained model:
```
/home/claude/ML_Trading/outputs/models/XAUUSD_1H_model.pkl
```

Metrics JSON:
```
/home/claude/ML_Trading/outputs/metrics/XAUUSD_1H_metrics.json
```

---

**NO MORE 99% WIN RATE BULLSHIT. THIS IS REAL.**
