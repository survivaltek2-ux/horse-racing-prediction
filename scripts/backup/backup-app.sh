#!/bin/bash
set -e

# Configuration
BACKUP_DIR="/backups/app"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-7}

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /backups/logs/backup.log
}

# Create backup directory
mkdir -p "$BACKUP_DIR" /backups/logs

log "Starting application data backup..."

# Create backup filename
APP_BACKUP_FILE="$BACKUP_DIR/app_data_${TIMESTAMP}.tar.gz"

# Backup application data
log "Creating application data backup: $APP_BACKUP_FILE"

# Create temporary directory for staging
TEMP_DIR="/tmp/app_backup_$$"
mkdir -p "$TEMP_DIR"

# Copy application data
if [ -d "/app/data" ]; then
    cp -r /app/data "$TEMP_DIR/"
    log "Application data copied"
fi

# Copy shared storage
if [ -d "/app/shared" ]; then
    cp -r /app/shared "$TEMP_DIR/"
    log "Shared storage copied"
fi

# Copy configuration files
if [ -d "/app/config" ]; then
    cp -r /app/config "$TEMP_DIR/"
    log "Configuration files copied"
fi

# Copy logs (last 7 days only)
if [ -d "/app/logs" ]; then
    mkdir -p "$TEMP_DIR/logs"
    find /app/logs -name "*.log" -mtime -7 -exec cp {} "$TEMP_DIR/logs/" \;
    log "Recent logs copied"
fi

# Create metadata file
cat > "$TEMP_DIR/backup_metadata.json" << EOF
{
    "backup_timestamp": "$TIMESTAMP",
    "backup_type": "application_data",
    "hostname": "$(hostname)",
    "backup_size": "$(du -sh $TEMP_DIR | cut -f1)",
    "files_count": "$(find $TEMP_DIR -type f | wc -l)"
}
EOF

# Create compressed archive
if tar -czf "$APP_BACKUP_FILE" -C "$TEMP_DIR" .; then
    log "Application backup completed successfully: $APP_BACKUP_FILE"
    
    # Verify backup
    if [ -s "$APP_BACKUP_FILE" ]; then
        BACKUP_SIZE=$(du -h "$APP_BACKUP_FILE" | cut -f1)
        log "Application backup size: $BACKUP_SIZE"
    else
        log "ERROR: Application backup file is empty"
        rm -f "$APP_BACKUP_FILE"
        exit 1
    fi
else
    log "ERROR: Application backup failed"
    rm -f "$APP_BACKUP_FILE"
    exit 1
fi

# Cleanup temporary directory
rm -rf "$TEMP_DIR"

# Cleanup old backups
log "Cleaning up old application backups (retention: $RETENTION_DAYS days)"
find "$BACKUP_DIR" -name "app_data_*.tar.gz" -mtime +$RETENTION_DAYS -delete

# Upload to cloud storage (if configured)
if [ -n "$AWS_S3_BUCKET" ]; then
    log "Uploading application backup to S3: s3://$AWS_S3_BUCKET/app/"
    aws s3 cp "$APP_BACKUP_FILE" "s3://$AWS_S3_BUCKET/app/" || log "WARNING: S3 upload failed"
fi

log "Application data backup process completed"