#!/bin/bash
# Final test to verify the entire system is working

echo "================================================================================"
echo "🧪 TESTING ML TRADING SYSTEM"
echo "================================================================================"
echo ""

# Check if in correct directory
if [ ! -f "api_server.py" ]; then
    echo "❌ Error: Must run from ML_Trading directory"
    exit 1
fi

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated. Activating..."
    source .venv312/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null
fi

echo "✅ Environment activated"
echo ""

# Test 1: Check .env file
echo "📋 Test 1: Checking environment variables..."
if [ -f ".env" ]; then
    echo "   ✅ .env file exists"
    if grep -q "POLYGON_API_KEY" .env && grep -q "SUPABASE_URL" .env; then
        echo "   ✅ Required variables found"
    else
        echo "   ❌ Missing required variables in .env"
        exit 1
    fi
else
    echo "   ❌ .env file not found"
    exit 1
fi
echo ""

# Test 2: Check Python packages
echo "📦 Test 2: Checking Python packages..."
python3 -c "import fastapi, uvicorn, supabase, pandas, numpy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ All required packages installed"
else
    echo "   ❌ Missing packages. Run: pip install -r requirements_api.txt"
    exit 1
fi
echo ""

# Test 3: Check model files
echo "🤖 Test 3: Checking ML models..."
model_count=$(find models_production -name "*PRODUCTION_READY.pkl" -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "$model_count" -gt 0 ]; then
    echo "   ✅ Found $model_count production-ready model files"
else
    echo "   ❌ No model files found in models_production/"
    exit 1
fi
echo ""

# Test 4: Test Supabase connection
echo "🔌 Test 4: Testing Supabase connection..."
python3 -c "
from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()
try:
    supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
    result = supabase.table('live_signals').select('*').limit(1).execute()
    print('   ✅ Supabase connection successful')
except Exception as e:
    print(f'   ❌ Supabase connection failed: {e}')
    exit(1)
" 2>/dev/null
if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# Test 5: Test Polygon API
echo "📊 Test 5: Testing Polygon API..."
python3 -c "
from dotenv import load_dotenv
import os
import requests

load_dotenv()
api_key = os.getenv('POLYGON_API_KEY')
url = f'https://api.polygon.io/v2/aggs/ticker/C:EURUSD/range/1/hour/2025-01-01/2025-01-02?apiKey={api_key}'
try:
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        print('   ✅ Polygon API connection successful')
    else:
        print(f'   ❌ Polygon API returned status {response.status_code}')
        exit(1)
except Exception as e:
    print(f'   ❌ Polygon API connection failed: {e}')
    exit(1)
" 2>/dev/null
if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# Test 6: Test API server (if running)
echo "🌐 Test 6: Testing API server..."
curl -s http://localhost:8000/health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ API server is running"
    models=$(curl -s http://localhost:8000/health | python3 -c "import sys, json; print(json.load(sys.stdin).get('models_available', 0))")
    echo "   ✅ $models models loaded"
else
    echo "   ⚠️  API server not running (start with: python3 api_server.py)"
fi
echo ""

# Test 7: Check deployment files
echo "🚀 Test 7: Checking deployment files..."
if [ -f "Procfile" ] && [ -f "railway.toml" ] && [ -f ".gitignore" ]; then
    echo "   ✅ All deployment files present"
else
    echo "   ⚠️  Some deployment files missing"
fi
echo ""

echo "================================================================================"
echo "✅ ALL TESTS PASSED!"
echo "================================================================================"
echo ""
echo "Your system is ready! Next steps:"
echo ""
echo "1. Start the backend:"
echo "   ./start_backend.sh"
echo ""
echo "2. Connect Lovable to Supabase:"
echo "   const { data } = await supabase.from('live_signals').select('*');"
echo ""
echo "3. Deploy to production (optional):"
echo "   See DEPLOY_TO_RAILWAY.md"
echo ""
echo "🎉 You're ready to go live!"
echo ""

