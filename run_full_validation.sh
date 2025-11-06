#!/bin/bash
###############################################################################
# FULL VALIDATION PIPELINE
###############################################################################
# Trains models, runs walk-forward tests, compares configs, and generates
# final production recommendation.
#
# Usage:
#   ./run_full_validation.sh               # Run all steps
#   ./run_full_validation.sh --skip-train  # Skip model training
###############################################################################

set -e  # Exit on error

SYMBOL="XAUUSD"
TIMEFRAME="15T"
SKIP_TRAIN=false

# Parse arguments
if [ "$1" == "--skip-train" ]; then
    SKIP_TRAIN=true
fi

echo "================================================================================"
echo "FULL VALIDATION PIPELINE FOR BEATING S&P 500"
echo "================================================================================"
echo "Symbol: $SYMBOL | Timeframe: $TIMEFRAME"
echo "================================================================================\n"

# Step 1: Train models (if not skipping)
if [ "$SKIP_TRAIN" = false ]; then
    echo "\n[1/6] Training Models..."
    echo "--------------------------------------------------------------------------------"
    
    echo "Training Config A model (1.8:1 RR)..."
    python3 jpm_production_system.py --symbol $SYMBOL --tf $TIMEFRAME --tp 1.8 --sl 1.0
    
    echo "\nTraining Config B model (2:1 RR)..."
    python3 jpm_production_system.py --symbol $SYMBOL --tf $TIMEFRAME --tp 2.0 --sl 1.0
    
    echo "\nTraining Config C model (2.5:1 RR)..."
    python3 jpm_production_system.py --symbol $SYMBOL --tf $TIMEFRAME --tp 2.5 --sl 1.0
    
    echo "\nTraining Config D model (3:1 RR)..."
    python3 jpm_production_system.py --symbol $SYMBOL --tf $TIMEFRAME --tp 3.0 --sl 1.0
    
    echo "\n✓ Model training complete!\n"
else
    echo "\n[1/6] Skipping model training (using existing models)...\n"
fi

# Step 2: Walk-forward validation
echo "\n[2/6] Running Walk-Forward Validation..."
echo "--------------------------------------------------------------------------------"

echo "Testing Config A (1.8:1 RR, 0.65 conf, 2% risk)..."
python3 walk_forward_validator.py --symbol $SYMBOL --tf $TIMEFRAME --config A --save

echo "\nTesting Config B (2:1 RR, 0.60 conf, 2% risk)..."
python3 walk_forward_validator.py --symbol $SYMBOL --tf $TIMEFRAME --config B --save

echo "\nTesting Config C (2.5:1 RR, 0.60 conf, 1.5% risk)..."
python3 walk_forward_validator.py --symbol $SYMBOL --tf $TIMEFRAME --config C --save

echo "\nTesting Config D (3:1 RR, 0.55 conf, 1.5% risk)..."
python3 walk_forward_validator.py --symbol $SYMBOL --tf $TIMEFRAME --config D --save

echo "\n✓ Walk-forward validation complete!\n"

# Step 3: Compare configs
echo "\n[3/6] Comparing Configurations..."
echo "--------------------------------------------------------------------------------"
python3 compare_configs.py --save-report
echo "\n✓ Comparison complete!\n"

# Step 4: Production stress testing (for viable configs)
echo "\n[4/6] Running Production Stress Tests..."
echo "--------------------------------------------------------------------------------"

# Test each config (will warn if not viable)
for CONFIG in A B C D; do
    echo "\nStress testing Config $CONFIG..."
    python3 production_validator.py --symbol $SYMBOL --tf $TIMEFRAME --config $CONFIG || true
done

echo "\n✓ Stress testing complete!\n"

# Step 5: Test ensemble strategy
echo "\n[5/6] Testing Ensemble Strategy..."
echo "--------------------------------------------------------------------------------"
echo "Testing ensemble of best configs..."
python3 ensemble_strategy.py --symbol $SYMBOL --tf $TIMEFRAME --configs A B C || true
echo "\n✓ Ensemble testing complete!\n"

# Step 6: Generate final summary
echo "\n[6/6] Generating Final Summary..."
echo "--------------------------------------------------------------------------------"

cat << 'EOF'

================================================================================
VALIDATION COMPLETE
================================================================================

Results saved in:
  - walk_forward_results/     Walk-forward test results per config
  - stress_test_results/      Production stress test results
  - ensemble_results/         Ensemble strategy results
  - config_comparison_report.txt   Comprehensive comparison report
  - config_comparison_data.csv     Raw data for further analysis

Next Steps:
  1. Review: config_comparison_report.txt
  2. Check recommendation at the end of the report
  3. If single config is viable, deploy that
  4. If multiple configs viable, consider ensemble
  5. If no configs viable, adjust parameters and retrain

Production Deployment:
  - Use the recommended configuration
  - Monitor daily for first 2 weeks
  - Implement 5% drawdown circuit breaker
  - Consider paper trading for 1 month before live

================================================================================
EOF

echo "\n✓ All validation complete!"
echo "\nReview the config_comparison_report.txt for final recommendation.\n"

