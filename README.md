# Institutional-Grade XAUUSD/XAGUSD Trading System

## Overview

This is a comprehensive, production-quality systematic trading system for precious metals (Gold and Silver) designed with institutional quant research practices in mind. The system combines regime-based rule logic with machine learning to achieve robust, out-of-sample performance.

### Key Features

- **Multi-timeframe strategy**: 5-minute, 15-minute, 30-minute, and 1-hour
- **Regime-adaptive**: Different strategies for trending, ranging, and high-volatility regimes
- **ML-enhanced filtering**: XGBoost/LightGBM models predict trade success probability
- **Realistic cost modeling**: Spread, commission, slippage explicitly modeled
- **Walk-forward validation**: Prevents overfitting through proper time-series CV
- **Prop firm evaluation**: Built-in checks for funded account rules
- **Performance monitoring**: Automated drift detection and retraining triggers

### Target Metrics

The system is designed to achieve (on out-of-sample tests):

- **Win Rate**: ≥ 60%
- **Profit Factor**: ≥ 1.5
- **Max Drawdown**: ≤ 6%

---

## Architecture & Design Philosophy

### Core Trading Philosophy

This is a **regime-based hybrid strategy**:

1. **Trending Regimes** (Trend-Up / Trend-Down)
   - Trade WITH the trend using structured pullback entries
   - Entry: Pullbacks to 20 EMA with confirmation
   - Edge: Capturing momentum continuation while avoiding extended moves

2. **Ranging/Mean-Reverting Regimes**
   - Fade extremes (Bollinger Band edges, RSI overbought/oversold)
   - Entry: 2+ standard deviation moves with exhaustion confirmation
   - Edge: Exploiting mean-reversion in choppy, low-ADX environments

3. **High-Volatility / Event-Driven Regimes**
   - Scale down or stand aside around major news events
   - Edge preservation: Protecting capital during high-uncertainty periods

### Machine Learning Role

**ML is NOT a black box.** It serves one specific purpose:

- **Task**: Predict probability that a candidate trade (identified by regime rules) will reach TP before SL
- **Output**: `P(TP before SL | features)` — a probability between 0 and 1
- **Decision rule**: Only execute trades where `P(success) ≥ threshold` (e.g., 0.60)

This creates **conditional edge** — we selectively trade only when statistical setup favors us.

### Edge Definition

A trade has **positive edge** if and only if:

```
Expected R = P(win) × R_win - P(loss) × R_loss - C_transaction > min_threshold
```

Where:
- `R_win` = TP distance in ATR units (e.g., +2.0 ATR)
- `R_loss` = SL distance in ATR units (e.g., -1.0 ATR)
- `C_transaction` = Total transaction cost in R units

**Enforcement**: A trade is only taken if the ML-predicted P(success) exceeds a threshold calibrated to ensure Expected R > minimum.

---

## Project Structure

```
ML_model/
│
├── config.py                 # Configuration management
├── data_loader.py            # Data loading and preprocessing
├── features.py               # Feature engineering and regime detection
├── model_training.py         # ML model training with walk-forward CV
├── backtest.py              # Backtesting engine with regime logic
├── metrics.py               # Performance metrics calculation
├── prop_eval.py             # Funded account / prop firm evaluation
├── walk_forward.py          # Walk-forward validation
├── monitoring.py            # Performance monitoring and retraining
├── main.py                  # Main orchestration script
│
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── data/                   # Data directory (OHLCV CSVs)
├── models/                 # Saved models
├── results/                # Backtest and validation results
└── logs/                   # Performance logs
```

---

## Installation

### Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

### Setup

```bash
# Clone repository (if applicable)
cd ML_model

# Install dependencies
pip install -r requirements.txt

# Create necessary directories (done automatically by main.py)
mkdir -p data models results logs
```

---

## Usage

### Quick Start

```python
# Run the complete pipeline
python main.py
```

This will:
1. Load (or generate sample) data
2. Engineer features and detect regimes
3. Train ML models with walk-forward CV
4. Run backtests on each timeframe
5. Perform final out-of-sample tests
6. Generate performance reports

### Configuration

Edit `config.py` or override in `main.py`:

```python
from config import get_default_config

config = get_default_config()

# Customize
config.data.symbols = ["XAUUSD"]  # Trade only gold
config.data.timeframes = [5, 15]  # Only 5m and 15m
config.strategy.ml_prob_threshold = 0.65  # Higher threshold
config.risk.risk_per_trade_pct = 0.25  # Lower risk per trade

# Run
from main import run_full_pipeline
run_full_pipeline(config, mode='full')
```

### Data Format

Place OHLCV data in `data/` directory with naming convention:

```
data/XAUUSD_5.csv    # Gold, 5-minute
data/XAUUSD_15.csv   # Gold, 15-minute
data/XAGUSD_5.csv    # Silver, 5-minute
...
```

CSV format:
```csv
timestamp,open,high,low,close,volume
2020-01-01 00:00:00,1800.5,1801.2,1799.8,1800.0,1000
2020-01-01 00:05:00,1800.0,1800.5,1799.5,1799.8,1200
...
```

**Note**: If no data files exist, the system will generate synthetic data for testing.

---

## Modules

### 1. `config.py` - Configuration Management

Centralized configuration using dataclasses:

- `DataConfig`: Data paths, symbols, timeframes
- `FeatureConfig`: Technical indicator parameters
- `RegimeConfig`: Regime detection thresholds
- `StrategyConfig`: Trading rules, SL/TP, ML threshold
- `RiskConfig`: Position sizing, max positions
- `CostConfig`: Spreads, commissions, slippage
- `ModelConfig`: ML hyperparameters
- `PropEvalConfig`: Funded account rules
- `MonitoringConfig`: Retraining frequency, thresholds

### 2. `data_loader.py` - Data Loading

Functions:
- `load_data_from_csv()`: Load OHLCV from CSV
- `resample_data()`: Resample to target timeframe
- `split_data()`: Chronological train/val/test split
- `load_all_data()`: Load all symbol-timeframe pairs
- `create_sample_data()`: Generate synthetic data for testing

### 3. `features.py` - Feature Engineering

**Technical Indicators**:
- Moving averages (EMA, SMA) with slopes
- MACD, RSI, Stochastic, CCI
- ATR and volatility metrics
- Bollinger Bands
- ADX for trend strength

**Regime Detection**:
- `detect_regime()`: Classify each bar into:
  - `trend_up`, `trend_down`, `range`, `event_vol`
- Uses ADX, MA alignment, volatility percentiles
- Strictly backward-looking (no lookahead)

**Target Construction**:
- `build_targets()`: For each bar, simulate hypothetical trade
- Label: 1 if TP hit before SL, 0 otherwise
- Forward-looking but isolated (no contamination to features)

### 4. `model_training.py` - ML Training

**TradingModel Class**:
- Wrapper for XGBoost/LightGBM/Logistic/RandomForest
- `train()`: Train with optional validation set
- `predict_proba()`: Get success probability
- `save()` / `load()`: Model persistence

**Walk-Forward Training**:
- `train_model_with_walk_forward()`: TimeSeriesSplit CV
- Out-of-fold predictions for validation
- Final model trained on full dataset

### 5. `backtest.py` - Backtesting Engine

**Backtest Class**:
- Regime-aware signal generation
- ML probability filter
- ATR-based position sizing
- Realistic cost modeling (spread, commission, slippage)
- Trade management (SL, TP, time-based exits)

**Process**:
1. For each bar, detect regime
2. Generate candidate trade based on regime rules
3. Query ML model for success probability
4. If `prob >= threshold`, execute trade
5. Manage open positions, check SL/TP

### 6. `metrics.py` - Performance Metrics

**PerformanceMetrics Class**:
- Win rate, profit factor
- Sharpe, Sortino, Calmar ratios
- Max drawdown, drawdown duration
- Average R-multiple, expectancy
- Trading frequency

**Functions**:
- `calculate_metrics()`: Compute all metrics from equity curve and trades
- `calculate_drawdown_stats()`: Detailed drawdown analysis
- `check_target_metrics()`: Verify if metrics meet targets

### 7. `prop_eval.py` - Prop Firm Evaluation

**PropEvalResults Class**:
- Check daily drawdown rule
- Check overall drawdown rule
- Check minimum trading days and trades
- Verify target metrics

**Functions**:
- `calculate_daily_drawdown()`: Daily DD from equity curve
- `check_prop_firm_rules()`: Comprehensive evaluation

### 8. `walk_forward.py` - Walk-Forward Validation

**Functions**:
- `create_walk_forward_splits()`: Generate rolling train/test windows
- `run_walk_forward_validation()`: Train and test on each window
- `run_final_oos_test()`: Final hold-out test on completely unseen data

**Purpose**: Prevent overfitting, assess true out-of-sample performance

### 9. `monitoring.py` - Performance Monitoring

**PerformanceMonitor Class**:
- Log performance over time
- Detect performance degradation
- Trigger retraining when:
  - Time-based: X days since last training
  - Performance-based: Sharpe/PF/DD below thresholds

**Functions**:
- `schedule_retraining()`: Automated retraining check (for cron jobs)
- `backtest_on_new_data()`: Test existing model on new data

### 10. `main.py` - Orchestration

**Main Pipeline**:
1. Setup directories
2. Load data
3. Build features
4. Train models
5. Walk-forward validation
6. Final OOS testing
7. Generate reports
8. Save results

**Modes**:
- `full`: Complete pipeline for all symbols/timeframes
- `quick`: Single symbol/timeframe for testing
- `validation_only`: Skip training, run validation only

---

## Walk-Forward Validation & Out-of-Sample Testing

### Why This Matters

Overfitting is the #1 killer of trading strategies. To combat this:

1. **Walk-Forward Validation**:
   - Split data into multiple rolling windows
   - For each window: train on past, test on future
   - Aggregate results to estimate true performance

2. **Final Hold-Out Test**:
   - Train on train + validation sets
   - Test on completely unseen final test set
   - This simulates deploying a model trained on historical data

3. **No Data Leakage**:
   - Features use only past information
   - Regime detection is backward-looking
   - Models never see future data

### Alignment with Institutional Practices

This approach mirrors what you'd see at:

- **Systematic hedge funds** (e.g., Renaissance, Two Sigma):
  - Clear separation of signal generation vs. filtering
  - Explicit cost modeling
  - Walk-forward validation

- **QuantConnect / Quantopian-style research**:
  - Modular pipeline: data → features → model → backtest → evaluation
  - Strict time-based splits
  - Performance degradation monitoring

---

## Retraining & Model Lifecycle

### Retraining Plan

1. **Frequency**: Weekly or monthly (configurable)

2. **Training Window**: Rolling 1–2 year lookback
   - Example: On 2025-11-16, use 2023-11-16 to 2025-10-16 for training

3. **Process**:
   - Load most recent data
   - Re-engineer features and labels
   - Train new model using TimeSeriesSplit
   - Validate on hold-out period
   - If performance meets thresholds: deploy
   - Else: flag for manual review, keep existing model

4. **Performance Monitoring**:
   - Maintain model performance log (JSON/CSV)
   - Track: Sharpe, PF, Win Rate, Max DD on rolling windows
   - **Drift detection**:
     - If rolling 30-day Sharpe < 0.5 or PF < 1.2 or DD > 8%:
       - Trigger alert
       - Reduce position sizes or halt trading until review

5. **Version Control**:
   - Each model saved with timestamp and hyperparameters
   - Ability to roll back to previous model if new model underperforms

### Integration with Live Environment

In production (e.g., on a VPS with live data feed):

- **Scheduled cron job** (weekly) runs retraining script
- **Automated backtests** on new data before deployment
- **A/B testing**: run new model in paper-trading mode alongside live model
- **Manual approval** gate before switching to new model in live trading

---

## Performance Targets & Evaluation

### Target Metrics

The system aims for (on out-of-sample tests):

- **Win Rate**: ≥ 60%
- **Profit Factor**: ≥ 1.5
- **Max Drawdown**: ≤ 6%

**Note**: These are not guarantees, but design targets. The system measures, enforces, and uses these metrics to accept/reject models and configurations.

### Prop Firm / Funded Account Rules

The system checks compliance with typical prop firm rules:

- **Max Daily Drawdown**: ≤ 5% (configurable)
- **Max Overall Drawdown**: ≤ 6% (configurable)
- **Minimum Trading Days**: ≥ 10
- **Minimum Trades**: ≥ 20

Only models that pass these checks (on OOS tests) are considered deployment-ready.

---

## Example Workflow

### 1. Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Place your OHLCV data in data/ directory
# Or let the system generate synthetic data for testing
```

### 2. Run Full Pipeline

```python
python main.py
```

Output:
```
============================================================
INSTITUTIONAL-GRADE TRADING SYSTEM
XAUUSD / XAGUSD Multi-Timeframe Strategy
============================================================

============================================================
STEP 1: Loading Data
============================================================
Loaded 20000 bars for XAUUSD 5min: 2020-01-01 to 2021-04-15
...

============================================================
STEP 2: Feature Engineering
============================================================
Processing XAUUSD 5min...
  Train: 8500 bars
  Val:   2800 bars
  Test:  2700 bars
  Regime distribution: {'trend_up': 3200, 'range': 2400, ...}
...

============================================================
STEP 3: Model Training
============================================================
Training model: XAUUSD 5min
...

============================================================
STEP 4: Walk-Forward Validation
============================================================
...

============================================================
STEP 5: Final Out-of-Sample Testing
============================================================
OOS Test Results:
  Total Trades:     145
  Win Rate:         62.1% (✓)
  Profit Factor:    1.68 (✓)
  Max Drawdown:     -5.2% (✓)
  
  Meets All Targets: ✓ YES
  Passes Prop Eval:  ✓ YES
...

============================================================
FINAL SUMMARY REPORT
============================================================
Out-of-Sample Test Results:
Symbol   TF  Trades    WR      PF   DD    Sharpe    R  Targets Prop
XAUUSD   5m     145  62.1%  1.68  -5.2%   1.45  1.12      ✓    ✓
XAUUSD  15m      98  61.2%  1.55  -5.8%   1.32  1.05      ✓    ✓
...
```

### 3. Review Results

Results are saved in `results/` directory:

- `walk_forward_YYYYMMDD_HHMMSS.json`: Walk-forward validation results
- `oos_YYYYMMDD_HHMMSS.json`: Out-of-sample test results
- `summary_YYYYMMDD_HHMMSS.csv`: Summary table

### 4. Monitor Performance

```python
from monitoring import PerformanceMonitor

monitor = PerformanceMonitor(config)

# Log new performance
monitor.log_performance(
    "XAUUSD", 5, "model_v1", 
    {"win_rate": 0.62, "profit_factor": 1.68, ...},
    "2025-11-01", "2025-11-30"
)

# Check if retraining needed
decision = monitor.should_retrain("XAUUSD", 5)
if decision['should_retrain']:
    print(f"Retraining recommended: {decision['reasons']}")
```

### 5. Deploy (Conceptual)

Once a model passes all checks on OOS tests:

1. Save model to production directory
2. Set up live data feed
3. Run in paper-trading mode for validation
4. After successful paper trading, enable live trading
5. Monitor performance continuously
6. Retrain periodically or when performance degrades

---

## Important Notes & Disclaimers

### This is NOT Financial Advice

This system is for **educational and research purposes only**. It demonstrates professional quant research practices and systematic trading methodology. 

**Do NOT deploy real capital without**:
- Extensive additional testing
- Risk management review
- Understanding of all code and logic
- Regulatory compliance (if applicable)
- Professional supervision

### Performance Variability

- Past performance does NOT guarantee future results
- Market conditions change
- Regime distributions shift
- Transaction costs and slippage vary
- The target metrics (60% WR, 1.5 PF, 6% DD) are **design goals**, not promises

### Recommended Next Steps for Production

If you wish to deploy this system in a live environment:

1. **Extended backtesting**: 5+ years of data across different market regimes
2. **Monte Carlo simulation**: Test robustness to parameter variations
3. **Regime analysis**: Ensure model performs across bull/bear/sideways markets
4. **Cost sensitivity**: Verify performance with realistic (or pessimistic) cost assumptions
5. **Slippage modeling**: Use tick data for more accurate fills
6. **Live paper trading**: Run in real-time without capital for 3–6 months
7. **Risk controls**: Implement additional kill switches and safeguards
8. **Professional review**: Have experienced traders/quants review the logic

---

## Contributing

If you'd like to extend or improve this system:

- **Add new features**: More indicators, alternative data, sentiment
- **Improve models**: Try neural networks, ensemble methods, online learning
- **Enhance regime detection**: Use hidden Markov models, clustering
- **Optimize execution**: Implement limit orders, smart order routing
- **Add asset classes**: Extend to FX, indices, crypto

---

## License

This project is provided as-is for educational purposes. Use at your own risk.

---

## Support & Contact

For questions or issues, please refer to the code documentation and docstrings. Each module is extensively commented to facilitate understanding.

---

## Acknowledgments

This system synthesizes best practices from:

- Quantitative finance literature
- Professional systematic trading firms
- Open-source quant frameworks (QuantConnect, Backtrader, etc.)
- Academic research on regime detection and ML in trading

It represents a serious attempt to build a trading system using **institutional-grade** methodology, transparency, and rigor.

---

**Happy Trading (Responsibly)!** 📈
