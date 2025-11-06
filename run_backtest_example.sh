#!/bin/bash
# Example script to run realistic backtest with different configurations

echo "=================================="
echo "REALISTIC BACKTEST EXAMPLES"
echo "=================================="
echo ""

# Find the most recent model
MODEL=$(ls -t models/XAUUSD/*.pkl 2>/dev/null | head -1)

if [ -z "$MODEL" ]; then
    echo "❌ No model found in models/XAUUSD/"
    echo "Please train a model first using:"
    echo "  python3 jpm_production_system.py --symbol XAUUSD --tf 15T"
    exit 1
fi

echo "📦 Using model: $MODEL"
echo ""

# Run backtest with default settings
echo "=================================="
echo "1. Running with DEFAULT settings"
echo "   - Confidence: 75%"
echo "   - Risk: 0.5%"
echo "=================================="
python3 realistic_backtest.py --model "$MODEL"

echo ""
echo ""
echo "=================================="
echo "2. Running with CONSERVATIVE settings"
echo "   - Confidence: 80%"
echo "   - Risk: 0.3%"
echo "=================================="
python3 realistic_backtest.py --model "$MODEL" --conf 0.80 --risk 0.003

echo ""
echo ""
echo "=================================="
echo "3. Running with AGGRESSIVE settings"
echo "   - Confidence: 65%"
echo "   - Risk: 1.0%"
echo "=================================="
python3 realistic_backtest.py --model "$MODEL" --conf 0.65 --risk 0.01

echo ""
echo "=================================="
echo "✅ BACKTEST COMPLETE"
echo "=================================="
echo ""
echo "Check the models/XAUUSD/ directory for:"
echo "  - *_backtest_charts.png (visualizations)"
echo "  - *_trade_log.csv (detailed trades)"
echo "  - *_summary.txt (metrics summary)"

