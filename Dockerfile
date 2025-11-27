FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_live_signals.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_live_signals.txt

# Copy application code
COPY live_signal_engine.py .
COPY polygon_connector.py .
COPY supabase_store.py .
COPY model_converter.py .
COPY run_live_signals.py .

# Copy artifacts (models, scalers, features)
COPY artifacts/ ./artifacts/

# Copy environment
COPY .env .env

# Create logs directory
RUN mkdir -p logs

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from live_signal_engine import SignalGenerator; print('ok')" || exit 1

# Run signal generator
CMD ["python", "-u", "run_live_signals.py"]
