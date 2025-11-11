#!/bin/bash
# Railway startup script

echo "🚀 Starting ML Trading System on Railway..."
echo "Environment variables check:"
echo "  POLYGON_API_KEY: ${POLYGON_API_KEY:0:10}..."
echo "  SUPABASE_URL: ${SUPABASE_URL:0:30}..."
echo "  SUPABASE_KEY: ${SUPABASE_KEY:0:20}..."
echo "  PORT: ${PORT:-8000}"
echo "  RAILWAY_SERVICE_NAME: ${RAILWAY_SERVICE_NAME:-web}"

# Download models from Supabase (don't fail if it doesn't work)
echo ""
echo "📦 Downloading models from Supabase..."
python3 download_models.py || {
    echo "⚠️  Model download failed or skipped"
    echo "Service will use fallback predictions"
}

echo ""
# Start the appropriate service based on environment
if [ "$RAILWAY_SERVICE_NAME" = "worker" ]; then
    echo "🔧 Starting WORKER service..."
    exec python3 worker_continuous.py
else
    echo "🌐 Starting API SERVER on port ${PORT:-8000}..."
    exec python3 api_server.py
fi

