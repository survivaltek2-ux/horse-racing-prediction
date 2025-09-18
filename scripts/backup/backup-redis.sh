#!/bin/bash
set -e

# Configuration
BACKUP_DIR="/backups/redis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-7}

# Redis connection
REDIS_HOST=${REDIS_HOST:-redis-primary}
REDIS_PORT=${REDIS_PORT:-6379}
REDIS_PASSWORD=${REDIS_PASSWORD:-redis_password}

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /backups/logs/backup.log
}

# Create backup directory
mkdir -p "$BACKUP_DIR" /backups/logs

log "Starting Redis backup..."

# Check Redis connectivity
if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" ping > /dev/null 2>&1; then
    log "ERROR: Cannot connect to Redis server"
    exit 1
fi

# Create backup filename
RDB_BACKUP_FILE="$BACKUP_DIR/redis_dump_${TIMESTAMP}.rdb.gz"
AOF_BACKUP_FILE="$BACKUP_DIR/redis_appendonly_${TIMESTAMP}.aof.gz"

# Perform RDB backup
log "Creating RDB backup: $RDB_BACKUP_FILE"
if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --rdb - | gzip > "$RDB_BACKUP_FILE"; then
    log "RDB backup completed successfully"
    
    # Verify backup
    if [ -s "$RDB_BACKUP_FILE" ]; then
        BACKUP_SIZE=$(du -h "$RDB_BACKUP_FILE" | cut -f1)
        log "RDB backup size: $BACKUP_SIZE"
    else
        log "ERROR: RDB backup file is empty"
        rm -f "$RDB_BACKUP_FILE"
        exit 1
    fi
else
    log "ERROR: RDB backup failed"
    rm -f "$RDB_BACKUP_FILE"
    exit 1
fi

# Backup Redis configuration and info
CONFIG_FILE="$BACKUP_DIR/redis_config_${TIMESTAMP}.txt"
INFO_FILE="$BACKUP_DIR/redis_info_${TIMESTAMP}.txt"

redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" CONFIG GET "*" > "$CONFIG_FILE"
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" INFO > "$INFO_FILE"

log "Redis configuration and info saved"

# Cleanup old backups
log "Cleaning up old Redis backups (retention: $RETENTION_DAYS days)"
find "$BACKUP_DIR" -name "redis_*" -mtime +$RETENTION_DAYS -delete

# Upload to cloud storage (if configured)
if [ -n "$AWS_S3_BUCKET" ]; then
    log "Uploading Redis backup to S3: s3://$AWS_S3_BUCKET/redis/"
    aws s3 cp "$RDB_BACKUP_FILE" "s3://$AWS_S3_BUCKET/redis/" || log "WARNING: S3 upload failed"
    aws s3 cp "$CONFIG_FILE" "s3://$AWS_S3_BUCKET/redis/" || log "WARNING: Config upload failed"
    aws s3 cp "$INFO_FILE" "s3://$AWS_S3_BUCKET/redis/" || log "WARNING: Info upload failed"
fi

log "Redis backup process completed"