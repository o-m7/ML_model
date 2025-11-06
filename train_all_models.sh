#!/bin/bash
################################################################################
# TRAIN ALL MODELS - ALL SYMBOLS & TIMEFRAMES
################################################################################

echo "================================================================================"
echo "TRAINING ALL MODELS WITH TRUE BACKTEST"
echo "================================================================================"
echo ""

SYMBOLS=(
    "XAUUSD"
    "XAGUSD"
    "EURUSD"
    "GBPUSD"
    "USDJPY"
    "USDCAD"
    "NZDUSD"
    "AUDUSD"
)

TIMEFRAMES=(
    "1T"
    "5T"
    "15T"
    "30T"
    "1H"
    "4H"
)

SUCCESS=0
FAILED=0
TOTAL=0

for symbol in "${SYMBOLS[@]}"; do
    for tf in "${TIMEFRAMES[@]}"; do
        ((TOTAL++))
        
        echo ""
        echo "################################################################################"
        echo "# TRAINING: $symbol $tf ($TOTAL/48)"
        echo "################################################################################"
        echo ""
        
        if python3 jpm_production_system.py --symbol "$symbol" --tf "$tf"; then
            echo "✓ $symbol $tf trained successfully"
            ((SUCCESS++))
        else
            echo "✗ $symbol $tf failed"
            ((FAILED++))
        fi
        
        echo ""
        echo "Progress: $SUCCESS success, $FAILED failed, $((TOTAL-SUCCESS-FAILED)) remaining"
        echo ""
        
        # Small delay to avoid overwhelming system
        sleep 2
    done
done

echo ""
echo "================================================================================"
echo "TRAINING COMPLETE"
echo "================================================================================"
echo "Total: $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo "Success Rate: $(( SUCCESS * 100 / TOTAL ))%"
echo "================================================================================"
echo ""

if [ $SUCCESS -gt 0 ]; then
    echo "✅ Trained $SUCCESS models!"
    echo ""
    echo "Next steps:"
    echo "  1. Run walk-forward validation on best models"
    echo "  2. Compare performance across symbols/timeframes"
    echo "  3. Select top performers for production"
fi

