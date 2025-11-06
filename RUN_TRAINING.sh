#!/bin/bash
#
# COMPLETE TRAINING RUN SCRIPT
# =============================
#
# Usage:
#   bash RUN_TRAINING.sh          # Test on 1 symbol
#   bash RUN_TRAINING.sh --all    # Train all symbols
#

set -e  # Exit on error

echo "=================================================="
echo "COMPLETE ML TRADING SYSTEM - PRODUCTION RUN"
echo "=================================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Install Python 3.8+"
    exit 1
fi

echo "✓ Python3: $(python3 --version)"
echo ""

# Check required packages
echo "Checking packages..."
python3 -c "import pandas, numpy, xgboost, lightgbm, sklearn" 2>/dev/null || {
    echo "❌ Missing packages. Installing..."
    pip install pandas numpy xgboost lightgbm scikit-learn
}
echo "✓ All packages installed"
echo ""

# Check data
if [ ! -d "feature_store" ]; then
    echo "❌ feature_store/ directory not found"
    echo "   Please ensure your data is in feature_store/SYMBOL/SYMBOL_TF.parquet"
    exit 1
fi

echo "✓ feature_store/ found"
echo ""

# Create output directories
mkdir -p models_production
mkdir -p logs

# Determine mode
if [ "$1" == "--all" ]; then
    echo "=================================================="
    echo "MODE: TRAIN ALL SYMBOLS IN PARALLEL"
    echo "=================================================="
    echo ""
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOGFILE="logs/training_all_${TIMESTAMP}.log"
    
    echo "Starting parallel training..."
    echo "Log file: $LOGFILE"
    echo ""
    
    python3 production_training_system.py --all --workers 4 2>&1 | tee "$LOGFILE"
    
    echo ""
    echo "=================================================="
    echo "TRAINING COMPLETE"
    echo "=================================================="
    echo ""
    echo "Results saved to: models_production/"
    echo "Check manifest: models_production/manifest.json"
    echo ""
    
    # Show summary
    if [ -f "models_production/manifest.json" ]; then
        echo "SUMMARY:"
        python3 -c "
import json
m = json.load(open('models_production/manifest.json'))
print(f\"  Total trained: {m['total']}\")
print(f\"  Passed: {m['passed']}\")
print(f\"  Failed: {m['total'] - m['passed']}\")
if m['passed'] > 0:
    print(f\"\n  ✅ Ready for production:\")
    for r in m['results']:
        if r['passed']:
            print(f\"     {r['symbol']} {r['timeframe']}\")
"
    fi
    
else
    # Test mode: single symbol
    echo "=================================================="
    echo "MODE: TEST ON SINGLE SYMBOL"
    echo "=================================================="
    echo ""
    
    SYMBOL="${2:-XAUUSD}"
    TF="${3:-15T}"
    
    echo "Testing: $SYMBOL $TF"
    echo ""
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOGFILE="logs/training_${SYMBOL}_${TF}_${TIMESTAMP}.log"
    
    python3 production_training_system.py --symbol "$SYMBOL" --tf "$TF" 2>&1 | tee "$LOGFILE"
    
    echo ""
    echo "=================================================="
    echo "TEST COMPLETE"
    echo "=================================================="
    echo ""
    
    # Check result
    if [ -d "models_production/$SYMBOL" ]; then
        LATEST=$(ls -t models_production/$SYMBOL/*.json 2>/dev/null | head -1)
        if [ -f "$LATEST" ]; then
            echo "Model card: $LATEST"
            echo ""
            python3 -c "
import json
m = json.load(open('$LATEST'))
print(f\"Status: {m['status']}\")
print(f\"Metrics:\")
print(f\"  Trades: {m['oos_metrics']['total_trades']}\")
print(f\"  Long: {m['oos_metrics']['long_trades']}\")
print(f\"  Short: {m['oos_metrics']['short_trades']}\")
print(f\"  Win Rate: {m['oos_metrics']['win_rate']:.1f}%\")
print(f\"  Profit Factor: {m['oos_metrics']['profit_factor']:.2f}\")
print(f\"  Max DD: {m['oos_metrics']['max_drawdown_pct']:.1f}%\")
print(f\"  Sharpe: {m['oos_metrics']['sharpe_ratio']:.2f}\")
if m['status'] == 'READY':
    print(f\"\n✅ PASSED - Ready for production\")
else:
    print(f\"\n❌ FAILED - {', '.join(m['benchmarks']['failures'])}\")
"
        fi
    fi
    
    echo ""
    echo "To train all symbols:"
    echo "  bash RUN_TRAINING.sh --all"
fi

echo ""
echo "Done!"