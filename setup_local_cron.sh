#!/bin/bash
# Setup cron job to run signal generator every 3 minutes

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH=$(which python3)
LOG_FILE="/tmp/ml_trading_signals.log"

echo "Setting up cron job..."
echo "Script directory: $SCRIPT_DIR"
echo "Python path: $PYTHON_PATH"
echo "Log file: $LOG_FILE"

# Create cron entry (runs every 3 minutes)
CRON_COMMAND="*/3 * * * * cd $SCRIPT_DIR && $PYTHON_PATH signal_generator.py >> $LOG_FILE 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "signal_generator.py"; then
    echo "⚠️  Cron job already exists. Removing old entry..."
    crontab -l | grep -v "signal_generator.py" | crontab -
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_COMMAND") | crontab -

echo ""
echo "✅ Cron job installed successfully!"
echo ""
echo "Current crontab:"
crontab -l | grep signal_generator
echo ""
echo "📊 Monitor logs with: tail -f $LOG_FILE"
echo "🗑️  Remove cron job with: crontab -e (then delete the line)"

