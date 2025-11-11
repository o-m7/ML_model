#!/bin/bash
# Railway startup script

set -e

echo "🚀 Starting ML Trading System on Railway..."

# Download models from Supabase
echo "📦 Downloading models..."
python3 download_models.py || echo "⚠️  Model download failed, continuing anyway..."

# Start the appropriate service based on environment
if [ "$RAILWAY_SERVICE_NAME" = "worker" ]; then
    echo "🔧 Starting worker service..."
    exec python3 worker_continuous.py
else
    echo "🌐 Starting API server..."
    exec python3 api_server.py
fi

