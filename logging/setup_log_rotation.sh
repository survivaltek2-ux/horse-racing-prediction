#!/bin/bash
"""
Log Rotation Setup Script
Sets up automated log rotation using cron jobs and logrotate.
"""

set -e

# Configuration
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/logs"
SCRIPT_DIR="$APP_DIR/logging"

echo "🔧 Setting up log rotation for Horse Racing Prediction Application"
echo "📁 Application Directory: $APP_DIR"
echo "📁 Log Directory: $LOG_DIR"

# Create log directories if they don't exist
mkdir -p "$LOG_DIR"/{errors,audit,deployment,performance,security}

# Create logrotate configuration
cat > "$APP_DIR/logrotate.conf" << 'EOF'
# Logrotate configuration for Horse Racing Prediction Application

/Users/richardsiebert/HorseRacingPrediction/APP/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(id -gn)
    postrotate
        # Restart application if needed (optional)
        # systemctl reload horse-racing-app || true
    endscript
}

/Users/richardsiebert/HorseRacingPrediction/APP/logs/*/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(id -gn)
    copytruncate
}

# Error logs - keep longer
/Users/richardsiebert/HorseRacingPrediction/APP/logs/errors/*.log {
    daily
    missingok
    rotate 90
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(id -gn)
    copytruncate
}

# Audit logs - keep much longer for compliance
/Users/richardsiebert/HorseRacingPrediction/APP/logs/audit/*.log {
    daily
    missingok
    rotate 365
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(id -gn)
    copytruncate
}
EOF

echo "✅ Created logrotate configuration"

# Create log rotation script
cat > "$SCRIPT_DIR/rotate_logs.sh" << 'EOF'
#!/bin/bash
# Log rotation script for Horse Racing Prediction Application

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SCRIPT="$APP_DIR/logging/log_rotation.py"

echo "$(date): Starting log rotation..."

# Run Python log rotation script
if [ -f "$PYTHON_SCRIPT" ]; then
    cd "$APP_DIR"
    python3 "$PYTHON_SCRIPT"
    echo "$(date): Python log rotation completed"
else
    echo "$(date): Python log rotation script not found: $PYTHON_SCRIPT"
fi

# Run system logrotate
if [ -f "$APP_DIR/logrotate.conf" ]; then
    /usr/sbin/logrotate -f "$APP_DIR/logrotate.conf"
    echo "$(date): System logrotate completed"
else
    echo "$(date): Logrotate config not found: $APP_DIR/logrotate.conf"
fi

echo "$(date): Log rotation finished"
EOF

chmod +x "$SCRIPT_DIR/rotate_logs.sh"
echo "✅ Created log rotation script"

# Create log cleanup script
cat > "$SCRIPT_DIR/cleanup_logs.sh" << 'EOF'
#!/bin/bash
# Log cleanup script - removes very old logs

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/logs"

echo "$(date): Starting log cleanup..."

# Remove compressed logs older than 6 months
find "$LOG_DIR" -name "*.gz" -type f -mtime +180 -delete 2>/dev/null || true

# Remove empty log directories
find "$LOG_DIR" -type d -empty -delete 2>/dev/null || true

echo "$(date): Log cleanup completed"
EOF

chmod +x "$SCRIPT_DIR/cleanup_logs.sh"
echo "✅ Created log cleanup script"

# Create monitoring script
cat > "$SCRIPT_DIR/monitor_logs.sh" << 'EOF'
#!/bin/bash
# Log monitoring script - checks for disk usage and errors

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/logs"

# Check disk usage
DISK_USAGE=$(du -sm "$LOG_DIR" 2>/dev/null | cut -f1)
MAX_SIZE_MB=1000  # 1GB limit

echo "$(date): Log directory size: ${DISK_USAGE}MB"

if [ "$DISK_USAGE" -gt "$MAX_SIZE_MB" ]; then
    echo "$(date): WARNING: Log directory size exceeds ${MAX_SIZE_MB}MB"
    # Could send alert here
fi

# Check for recent errors
ERROR_COUNT=$(find "$LOG_DIR/errors" -name "*.log" -mtime -1 -exec grep -c "ERROR" {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
echo "$(date): Recent errors in last 24h: $ERROR_COUNT"

if [ "$ERROR_COUNT" -gt 100 ]; then
    echo "$(date): WARNING: High error count detected: $ERROR_COUNT"
    # Could send alert here
fi
EOF

chmod +x "$SCRIPT_DIR/monitor_logs.sh"
echo "✅ Created log monitoring script"

# Create cron job entries
CRON_FILE="/tmp/horse_racing_cron"
cat > "$CRON_FILE" << EOF
# Horse Racing Prediction Application Log Management
# Rotate logs daily at 2 AM
0 2 * * * $SCRIPT_DIR/rotate_logs.sh >> $LOG_DIR/rotation.log 2>&1

# Clean up old logs weekly on Sunday at 3 AM
0 3 * * 0 $SCRIPT_DIR/cleanup_logs.sh >> $LOG_DIR/cleanup.log 2>&1

# Monitor logs every 6 hours
0 */6 * * * $SCRIPT_DIR/monitor_logs.sh >> $LOG_DIR/monitoring.log 2>&1
EOF

echo "📋 Cron job configuration created at: $CRON_FILE"
echo ""
echo "To install the cron jobs, run:"
echo "  crontab $CRON_FILE"
echo ""
echo "To view current cron jobs:"
echo "  crontab -l"
echo ""
echo "To edit cron jobs manually:"
echo "  crontab -e"

# Test the rotation script
echo "🧪 Testing log rotation script..."
if "$SCRIPT_DIR/rotate_logs.sh"; then
    echo "✅ Log rotation test successful"
else
    echo "❌ Log rotation test failed"
fi

echo ""
echo "🎉 Log rotation setup completed!"
echo ""
echo "📁 Files created:"
echo "  - $APP_DIR/logrotate.conf"
echo "  - $SCRIPT_DIR/rotate_logs.sh"
echo "  - $SCRIPT_DIR/cleanup_logs.sh"
echo "  - $SCRIPT_DIR/monitor_logs.sh"
echo "  - $CRON_FILE"
echo ""
echo "📝 Next steps:"
echo "  1. Install cron jobs: crontab $CRON_FILE"
echo "  2. Monitor logs: tail -f $LOG_DIR/rotation.log"
echo "  3. Check log statistics: python3 $SCRIPT_DIR/log_rotation.py"