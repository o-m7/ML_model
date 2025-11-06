#!/bin/bash

# ============================================================================
# ML Trading System - Web Deployment Script
# ============================================================================

set -e  # Exit on error

echo ""
echo "================================================================================"
echo "🚀 ML TRADING SYSTEM - WEB DEPLOYMENT"
echo "================================================================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please create .env with SUPABASE_URL and SUPABASE_KEY"
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Verify Supabase credentials
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_KEY" ]; then
    echo "❌ SUPABASE_URL or SUPABASE_KEY not set in .env"
    exit 1
fi

echo "✅ Environment variables loaded"
echo ""

# ============================================================================
# STEP 1: Install Dependencies
# ============================================================================

echo "================================================================================"
echo "STEP 1: Installing Dependencies"
echo "================================================================================"
echo ""

pip install -r requirements_api.txt

echo ""
echo "✅ Dependencies installed"
echo ""

# ============================================================================
# STEP 2: Convert Models to ONNX
# ============================================================================

echo "================================================================================"
echo "STEP 2: Converting Models to ONNX"
echo "================================================================================"
echo ""

python3 convert_models_to_onnx.py

echo ""
echo "✅ Models converted to ONNX"
echo ""

# ============================================================================
# STEP 3: Sync to Supabase
# ============================================================================

echo "================================================================================"
echo "STEP 3: Syncing Models to Supabase"
echo "================================================================================"
echo ""

python3 supabase_sync.py

echo ""
echo "✅ Models synced to Supabase"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "================================================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Set up Supabase tables:"
echo "   → Go to Supabase SQL Editor"
echo "   → Run: supabase_schema.sql"
echo ""
echo "2. Create storage bucket:"
echo "   → Go to Supabase Storage"
echo "   → Create bucket: 'ml_models' (public)"
echo ""
echo "3. Start API server:"
echo "   → python3 api_server.py"
echo ""
echo "4. Test API:"
echo "   → curl http://localhost:8000/health"
echo ""
echo "5. Integrate with Lovable:"
echo "   → See WEB_DEPLOYMENT_GUIDE.md for code examples"
echo ""
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "================================================================================"

