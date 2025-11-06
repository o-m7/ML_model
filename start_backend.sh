#!/bin/bash

# ============================================================================
# START BACKEND FOR LOVABLE
# ============================================================================

set -e

echo ""
echo "================================================================================"
echo "🚀 STARTING ML TRADING BACKEND FOR LOVABLE"
echo "================================================================================"
echo ""

# Check if Python 3.12 venv exists
if [ ! -d ".venv312" ]; then
    echo "❌ Python 3.12 virtual environment not found"
    echo "Please run: python3.12 -m venv .venv312"
    exit 1
fi

# Activate venv
source .venv312/bin/activate

# Check environment variables
if [ -z "$SUPABASE_URL" ]; then
    echo "⚠️  Loading .env file..."
    export $(grep -v '^#' .env | xargs)
fi

echo "✅ Environment configured"
echo "   Supabase URL: $SUPABASE_URL"
echo "   Polygon API: ${POLYGON_API_KEY:0:10}..."
echo ""

# Kill any existing processes
echo "🧹 Cleaning up old processes..."
pkill -f "api_server.py" 2>/dev/null || true
pkill -f "live_trading_engine.py" 2>/dev/null || true
sleep 2

# Start API server in background
echo "🌐 Starting API Server..."
nohup python3 api_server.py > logs/api_server.log 2>&1 &
API_PID=$!
echo "   PID: $API_PID"

# Wait for API to be ready
echo "⏳ Waiting for API to start..."
sleep 5

# Test API
echo "🧪 Testing API..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "   ✅ API is running!"
else
    echo "   ❌ API failed to start"
    cat logs/api_server.log
    exit 1
fi

echo ""
echo "================================================================================"
echo "✅ BACKEND READY FOR LOVABLE!"
echo "================================================================================"
echo ""
echo "📡 API Server: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "🗄️  Supabase: https://app.supabase.com/project/ifetofkhyblyijghuwzs"
echo ""
echo "To generate live signals, run in a new terminal:"
echo "   source .venv312/bin/activate"
echo "   python3 live_trading_engine.py once"
echo ""
echo "Or for continuous monitoring:"
echo "   python3 live_trading_engine.py"
echo ""
echo "To stop:"
echo "   pkill -f api_server.py"
echo "   pkill -f live_trading_engine.py"
echo ""
echo "================================================================================"

