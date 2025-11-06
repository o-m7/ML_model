#!/bin/bash
# Regenerate ALL features with look-ahead bias fix
# This will take 4-8 hours for all symbols!

set -e

echo "=============================================="
echo "REGENERATING ALL FEATURES (LOOK-AHEAD FIX)"
echo "=============================================="
echo ""
echo "⚠️  WARNING: This will:"
echo "  1. Delete all existing feature parquet files"
echo "  2. Regenerate features with proper shifting"
echo "  3. Take 30-60 minutes PER SYMBOL"
echo ""
echo "Symbols to process: XAUUSD XAGUSD EURUSD AUDUSD USDCAD NZDUSD USDJPY GBPUSD"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled"
    exit 0
fi

# Backup existing features
echo ""
echo "Creating backup..."
cd /Users/omar/Desktop/ML_Trading/feature_store
tar -czf ../feature_store_backup_$(date +%Y%m%d_%H%M%S).tar.gz . 2>/dev/null || true
echo "✅ Backup created (if features existed)"

# Delete old features
echo ""
echo "Deleting old features..."
for symbol in XAUUSD XAGUSD EURUSD AUDUSD USDCAD NZDUSD USDJPY GBPUSD; do
    if [ -d "$symbol" ]; then
        rm -rf $symbol/*.parquet
        echo "  Deleted $symbol/*.parquet"
    fi
done

# Regenerate features
cd /Users/omar/Desktop/Polygon-ML-data

SYMBOLS=("XAUUSD" "XAGUSD" "EURUSD" "AUDUSD" "USDCAD" "NZDUSD" "USDJPY" "GBPUSD")
SUCCESS=0
FAILED=0

for i in "${!SYMBOLS[@]}"; do
    symbol="${SYMBOLS[$i]}"
    num=$((i+1))
    total=${#SYMBOLS[@]}
    
    echo ""
    echo "========================================"
    echo "[$num/$total] Processing: $symbol"
    echo "========================================"
    
    if [ -f "raw_data/${symbol}_minute.csv" ]; then
        python3 feature_engineering.py \
            --input raw_data/${symbol}_minute.csv \
            --output /Users/omar/Desktop/ML_Trading/feature_store/$symbol \
            --symbol $symbol
        
        if [ $? -eq 0 ]; then
            echo "✅ $symbol complete"
            ((SUCCESS++))
        else
            echo "❌ $symbol failed"
            ((FAILED++))
        fi
    else
        echo "⚠️  $symbol: raw data not found (raw_data/${symbol}_minute.csv)"
        echo "   Run: python3 polygon_forex_pipeline.py --mode backfill --symbol $symbol"
        ((FAILED++))
    fi
done

# Summary
echo ""
echo "=============================================="
echo "REGENERATION COMPLETE"
echo "=============================================="
echo "✅ Successful: $SUCCESS"
echo "❌ Failed: $FAILED"
echo ""

# Verify
echo "Verifying features..."
cd /Users/omar/Desktop/ML_Trading
source .venv/bin/activate 2>/dev/null || true

for symbol in "${SYMBOLS[@]}"; do
    if [ -f "feature_store/$symbol/${symbol}_15T.parquet" ]; then
        echo "✅ $symbol: Features exist"
    else
        echo "❌ $symbol: Missing features"
    fi
done

echo ""
echo "=============================================="
echo "NEXT STEPS"
echo "=============================================="
echo "1. Run diagnostic:"
echo "   cd /Users/omar/Desktop/ML_Trading"
echo "   python diagnose_leakage.py"
echo ""
echo "2. Delete old models:"
echo "   rm -rf models/XAUUSD/*.pkl models/XAUUSD/*_meta.json"
echo ""
echo "3. Retrain with clean features:"
echo "   python3 jpm_production_system.py --symbol XAUUSD --tf 15T"
echo ""
echo "4. Expect REALISTIC results:"
echo "   - Win Rate: 52-58% (not 73%!)"
echo "   - Max DD: 3-8% (not 0.14%!)"
echo "   - Profit Factor: 1.5-2.5 (not 3.11!)"
echo ""

