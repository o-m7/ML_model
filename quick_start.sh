#!/bin/bash
################################################################################
# PRODUCTION SYSTEM - QUICK START
################################################################################

echo ""
echo "================================================================================"
echo "PRODUCTION ML TRADING SYSTEM - QUICK START"
echo "================================================================================"
echo ""
echo "This will:"
echo "  1. Enrich 48 files with 180+ TA-Lib features (1-2 hours)"
echo "  2. Train 48 models with walk-forward CV (3-6 hours)"
echo "  3. Generate production-ready models with GO-LIVE benchmarks"
echo ""
echo "Total time: 4-8 hours"
echo "================================================================================"
echo ""

read -p "Press ENTER to start or CTRL+C to cancel..."

START_TIME=$(date +%s)

# Step 1: Enrich
echo ""
echo "STEP 1/2: ENRICHING DATA WITH TA-LIB FEATURES"
echo "================================================================================"
bash enrich_all_symbols.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Feature enrichment failed!"
    exit 1
fi

# Step 2: Train
echo ""
echo "STEP 2/2: TRAINING PRODUCTION MODELS"
echo "================================================================================"
python3 production_training_system.py --all --workers 4

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Training failed!"
    exit 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "================================================================================"
echo "✅ PRODUCTION SYSTEM COMPLETE"
echo "================================================================================"
echo "Time elapsed: ${HOURS}h ${MINUTES}m"
echo ""
echo "Next steps:"
echo "  1. Review manifest: cat models_production/manifest.json"
echo "  2. Check model cards: cat models_production/XAUUSD/*.json"
echo "  3. Start live inference: python3 live_runner.py --mode live"
echo ""
echo "See PRODUCTION_SYSTEM_README.md for complete documentation"
echo "================================================================================"
