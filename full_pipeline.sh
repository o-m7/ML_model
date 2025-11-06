#!/bin/bash
################################################################################
# COMPLETE PIPELINE: ENRICH FEATURES + TRAIN ALL MODELS
################################################################################

echo ""
echo "================================================================================"
echo "FULL ML TRADING PIPELINE"
echo "================================================================================"
echo "This will:"
echo "  1. Enrich ALL symbols with 180+ TA-Lib features"
echo "  2. Train models for ALL symbols & timeframes (8 x 6 = 48 models)"
echo ""
echo "Expected time: 3-6 hours"
echo "================================================================================"
echo ""

read -p "Press ENTER to start or CTRL+C to cancel..."

START_TIME=$(date +%s)

# Step 1: Enrich all symbols
echo ""
echo "STEP 1/2: ENRICHING FEATURES"
echo "================================================================================"
bash enrich_all_symbols.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Feature enrichment failed. Stopping."
    exit 1
fi

# Step 2: Train all models
echo ""
echo "STEP 2/2: TRAINING MODELS"
echo "================================================================================"
bash train_all_models.sh

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "================================================================================"
echo "FULL PIPELINE COMPLETE"
echo "================================================================================"
echo "Time elapsed: ${HOURS}h ${MINUTES}m"
echo ""
echo "Check results in:"
echo "  models/SYMBOL/SYMBOL_TF_*.pkl"
echo ""
echo "Next steps:"
echo "  1. Review model performance metrics"
echo "  2. Run walk-forward validation on best models"
echo "  3. Compare to S&P 500 benchmark"
echo "================================================================================"

