#!/bin/bash

# Health check for backup service
BACKUP_LOG="/backups/logs/backup.log"
HEALTH_FILE="/backups/.health"

# Check if backup log exists and has recent entries
if [ -f "$BACKUP_LOG" ]; then
    # Check if there's a log entry from the last 25 hours (allowing for daily backup schedule)
    if find "$BACKUP_LOG" -mtime -1 -exec grep -q "backup process completed" {} \; 2>/dev/null; then
        echo "healthy" > "$HEALTH_FILE"
        echo "Backup service is healthy"
        exit 0
    fi
fi

# Check if backup directories exist
if [ ! -d "/backups/postgres" ] || [ ! -d "/backups/redis" ] || [ ! -d "/backups/app" ]; then
    echo "unhealthy" > "$HEALTH_FILE"
    echo "Backup directories missing"
    exit 1
fi

# If no recent backup found, check if it's the first run
if [ ! -f "$BACKUP_LOG" ]; then
    echo "initializing" > "$HEALTH_FILE"
    echo "Backup service initializing"
    exit 0
fi

echo "unhealthy" > "$HEALTH_FILE"
echo "No recent backup found"
exit 1