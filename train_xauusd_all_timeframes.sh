#!/bin/bash
# Train all XAUUSD timeframes systematically

cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate

echo "========================================"
echo "TRAINING ALL XAUUSD TIMEFRAMES"
echo "========================================"
echo ""

timeframes=(5T 15T 30T 1H 2H 4H)

for tf in "${timeframes[@]}"; do
    echo ""
    echo "========================================"
    echo "Training: XAUUSD $tf"
    echo "========================================"
    echo ""
    
    python production_training_system.py --symbol XAUUSD --tf $tf
    
    echo ""
    echo "----------------------------------------"
    echo "Completed: XAUUSD $tf"
    echo "----------------------------------------"
    echo ""
    
    # Small delay between trainings
    sleep 2
done

echo ""
echo "========================================"
echo "SUMMARY: XAUUSD Training Results"
echo "========================================"
echo ""

for tf in "${timeframes[@]}"; do
    latest=$(ls -t models_production/XAUUSD/XAUUSD_${tf}_*.json 2>/dev/null | head -1)
    if [ -f "$latest" ]; then
        status=$(grep -o '"status": "[^"]*"' "$latest" | cut -d'"' -f4)
        pf=$(grep -o '"profit_factor": [0-9.]*' "$latest" | cut -d' ' -f2)
        dd=$(grep -o '"max_drawdown_pct": [0-9.]*' "$latest" | cut -d' ' -f2)
        wr=$(grep -o '"win_rate": [0-9.]*' "$latest" | cut -d' ' -f2)
        trades=$(grep -o '"total_trades": [0-9]*' "$latest" | cut -d' ' -f2)
        
        printf "XAUUSD %-4s %s | PF: %-6s | DD: %-6s%% | WR: %-6s%% | Trades: %s\n" \
            "$tf" "$status" "$pf" "$dd" "$wr" "$trades"
    else
        echo "XAUUSD $tf: No model found"
    fi
done

echo ""
echo "========================================"
echo "Models saved in: models_production/XAUUSD/"
echo "========================================"
ls -lh models_production/XAUUSD/*.pkl 2>/dev/null | tail -10

