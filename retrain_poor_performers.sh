#!/bin/bash
#
# RETRAIN POOR PERFORMING MODELS
# ================================
# Based on performance analysis:
# - XAUUSD 5T: 37.2% win rate (BAD - needs retraining)
# - XAUUSD 30T SHORT: 35.0% win rate (BAD - needs retraining)
#
# Good performers (no retraining needed):
# - XAGUSD 15T LONG: 90.0% win rate ⭐⭐⭐
# - XAGUSD 30T LONG: 88.9% win rate ⭐⭐⭐
# - XAGUSD 5T LONG: 76.9% win rate ⭐⭐
# - XAUUSD 15T: 75.9% LONG, 54.3% SHORT ⭐⭐
#

set -e

echo "========================================"
echo "RETRAINING POOR PERFORMING MODELS"
echo "========================================"
echo ""

# Check if data exists
if [ ! -d "feature_store/XAUUSD" ]; then
    echo "❌ ERROR: feature_store/XAUUSD not found"
    echo "   Please ensure training data exists"
    exit 1
fi

# Backup old models
echo "📦 Backing up old models..."
mkdir -p models_rentec/XAUUSD/backup_$(date +%Y%m%d_%H%M%S)
cp models_rentec/XAUUSD/XAUUSD_5T.pkl models_rentec/XAUUSD/backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
cp models_rentec/XAUUSD/XAUUSD_30T.pkl models_rentec/XAUUSD/backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
echo "✅ Backup complete"
echo ""

# Retrain XAUUSD 5T (worst performer)
echo "========================================"
echo "RETRAINING XAUUSD 5T (37.2% → target: 60%+)"
echo "========================================"
python train_model.py --symbol XAUUSD --tf 5T
echo ""

# Retrain XAUUSD 30T
echo "========================================"
echo "RETRAINING XAUUSD 30T (35% SHORT → target: 55%+)"
echo "========================================"
python train_model.py --symbol XAUUSD --tf 30T
echo ""

# Optionally retrain 1H for completeness
read -p "Retrain XAUUSD 1H as well? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "========================================"
    echo "RETRAINING XAUUSD 1H (optional)"
    echo "========================================"
    python train_model.py --symbol XAUUSD --tf 1H
    echo ""
fi

echo "========================================"
echo "RETRAINING COMPLETE"
echo "========================================"
echo ""
echo "✅ Models retrained and saved to models_rentec/XAUUSD/"
echo ""
echo "Next steps:"
echo "  1. Test models: python signal_generator.py"
echo "  2. Monitor win rates in production"
echo "  3. Compare with backup models if needed"
echo ""
echo "Backup location: models_rentec/XAUUSD/backup_*"
echo "========================================"
