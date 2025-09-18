#!/bin/bash
# PostgreSQL Backup Script for Production Environment

set -euo pipefail

# Configuration
BACKUP_DIR="/backups/postgres"
RETENTION_DAYS=30
POSTGRES_HOST="${POSTGRES_HOST:-postgres-master}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-horse_racing_prediction}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/postgres_backup_${TIMESTAMP}.sql"
COMPRESSED_BACKUP="${BACKUP_FILE}.gz"

# Logging
LOG_FILE="/var/log/backup/postgres_backup.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Create backup directory
mkdir -p "$BACKUP_DIR" /backups/logs

log "Starting PostgreSQL backup..."

# Check database connectivity
if ! pg_isready -h "$PGHOST" -p "$PGPORT" -U "$PGUSER"; then
    log "ERROR: Cannot connect to PostgreSQL database"
    exit 1
fi

# Create backup filename
BACKUP_FILE="$BACKUP_DIR/postgres_backup_${TIMESTAMP}.sql.gz"

# Perform backup
log "Creating backup: $BACKUP_FILE"
if pg_dump -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" \
    --verbose --clean --if-exists --create \
    --format=custom --compress=9 | gzip > "$BACKUP_FILE"; then
    
    log "Backup completed successfully: $BACKUP_FILE"
    
    # Verify backup
    if [ -s "$BACKUP_FILE" ]; then
        BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
        log "Backup size: $BACKUP_SIZE"
    else
        log "ERROR: Backup file is empty"
        rm -f "$BACKUP_FILE"
        exit 1
    fi
else
    log "ERROR: Backup failed"
    rm -f "$BACKUP_FILE"
    exit 1
fi

# Cleanup old backups
log "Cleaning up old backups (retention: $RETENTION_DAYS days, max: $MAX_BACKUPS)"

# Remove backups older than retention period
find "$BACKUP_DIR" -name "postgres_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete

# Keep only the latest MAX_BACKUPS files
ls -t "$BACKUP_DIR"/postgres_backup_*.sql.gz | tail -n +$((MAX_BACKUPS + 1)) | xargs -r rm -f

# Upload to cloud storage (if configured)
if [ -n "$AWS_S3_BUCKET" ]; then
    log "Uploading backup to S3: s3://$AWS_S3_BUCKET/postgres/"
    aws s3 cp "$BACKUP_FILE" "s3://$AWS_S3_BUCKET/postgres/" || log "WARNING: S3 upload failed"
fi

log "PostgreSQL backup process completed"