#!/bin/bash
################################################################################
# TEST DIFFERENT RR RATIOS TO FIND OPTIMAL
################################################################################

echo "================================================================================"
echo "TESTING DIFFERENT RISK:REWARD RATIOS"
echo "================================================================================"
echo ""

SYMBOL="XAUUSD"
TF="15T"

# Test different TP:SL ratios
configs=(
    "1.5 1.0"  # 1.5:1 RR
    "1.8 1.0"  # 1.8:1 RR
    "2.0 1.0"  # 2.0:1 RR (current)
    "2.5 1.0"  # 2.5:1 RR
)

echo "Testing $SYMBOL $TF with different TP:SL ratios..."
echo ""

for config in "${configs[@]}"; do
    TP=$(echo $config | awk '{print $1}')
    SL=$(echo $config | awk '{print $2}')
    RR=$(echo "scale=1; $TP / $SL" | bc)
    
    echo "################################################################################"
    echo "# TESTING: ${RR}:1 RR (TP=${TP}, SL=${SL})"
    echo "################################################################################"
    echo ""
    
    python3 jpm_production_system.py --symbol "$SYMBOL" --tf "$TF" --tp "$TP" --sl "$SL"
    
    echo ""
    echo "================================================================================"
    echo ""
    sleep 2
done

echo ""
echo "================================================================================"
echo "TESTING COMPLETE"
echo "================================================================================"
echo ""
echo "Review results above and choose the best RR ratio that achieves:"
echo "  • Win Rate: 48-55%"
echo "  • Return: 15-30%"
echo "  • Profit Factor: 1.5-2.5"
echo ""
echo "Then use that configuration for all symbols/timeframes"
echo "================================================================================"

