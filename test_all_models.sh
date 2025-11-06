#!/bin/bash
# Test all symbols at 1H with regime filtering

cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate

symbols=(XAUUSD XAGUSD EURUSD GBPUSD AUDUSD NZDUSD USDJPY USDCAD)

for sym in "${symbols[@]}"; do
    echo ""
    echo "========================================="
    echo "Training: $sym 1H"
    echo "========================================="
    
    python production_training_system.py --symbol $sym --tf 1H 2>&1 | tail -50
    
    echo ""
done

echo ""
echo "========================================="
echo "SUMMARY: Checking Results"
echo "========================================="

for sym in "${symbols[@]}"; do
    latest=$(ls -t models_production/$sym/*.json 2>/dev/null | head -1)
    if [ -f "$latest" ]; then
        status=$(grep -o '"status": "[^"]*"' "$latest" | cut -d'"' -f4)
        pf=$(grep -o '"profit_factor": [0-9.]*' "$latest" | cut -d' ' -f2)
        dd=$(grep -o '"max_drawdown_pct": [0-9.]*' "$latest" | cut -d' ' -f2)
        wr=$(grep -o '"win_rate": [0-9.]*' "$latest" | cut -d' ' -f2)
        trades=$(grep -o '"total_trades": [0-9]*' "$latest" | cut -d' ' -f2)
        
        printf "%-8s %s | PF: %-6s | DD: %-6s%% | WR: %-6s%% | Trades: %s\n" \
            "$sym" "$status" "$pf" "$dd" "$wr" "$trades"
    else
        echo "$sym: No model found"
    fi
done

