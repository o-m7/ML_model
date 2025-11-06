#!/bin/bash
################################################################################
# ENRICH ALL SYMBOLS WITH TA-LIB FEATURES
################################################################################

echo "================================================================================"
echo "ENRICHING ALL SYMBOLS WITH TA-LIB FEATURES"
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

SUCCESS=0
FAILED=0

for symbol in "${SYMBOLS[@]}"; do
    echo ""
    echo "################################################################################"
    echo "# ENRICHING: $symbol (ALL TIMEFRAMES)"
    echo "################################################################################"
    echo ""
    
    if python3 add_talib_features.py --symbol "$symbol"; then
        echo "✓ $symbol enriched successfully"
        ((SUCCESS++))
    else
        echo "✗ $symbol failed"
        ((FAILED++))
    fi
done

echo ""
echo "================================================================================"
echo "ENRICHMENT COMPLETE"
echo "================================================================================"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo "================================================================================"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL SYMBOLS ENRICHED!"
    echo ""
    echo "Next step: Train all models"
    echo "  bash train_all_models.sh"
else
    echo "⚠️  Some symbols failed - check errors above"
fi

