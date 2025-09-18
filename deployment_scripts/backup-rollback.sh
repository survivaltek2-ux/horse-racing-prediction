#!/bin/bash

# =============================================================================
# Backup and Rollback Script
# Horse Racing Prediction Application
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="$PROJECT_ROOT/backups"
LOG_DIR="$PROJECT_ROOT/logs/backup"

# Create directories
mkdir -p "$BACKUP_DIR" "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default configuration
ENVIRONMENT="production"
ACTION="backup"
BACKUP_TYPE="full"
RETENTION_DAYS=30
COMPRESS=true
REMOTE_BACKUP=false
S3_BUCKET=""
BACKUP_NAME=""
ROLLBACK_TARGET=""

# =============================================================================
# Utility Functions
# =============================================================================

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# =============================================================================
# Backup Functions
# =============================================================================

create_backup_metadata() {
    local backup_path="$1"
    local backup_type="$2"
    
    local metadata_file="$backup_path/backup_metadata.json"
    
    cat > "$metadata_file" << EOF
{
    "backup_id": "$(basename "$backup_path")",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$ENVIRONMENT",
    "backup_type": "$backup_type",
    "application_version": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "database_version": "$(docker-compose exec -T postgres psql -U ${POSTGRES_USER:-app_user} -d ${POSTGRES_DB:-horse_racing_db} -t -c 'SELECT version();' 2>/dev/null | head -1 | xargs || echo 'unknown')",
    "created_by": "$(whoami)",
    "hostname": "$(hostname)",
    "size_bytes": 0,
    "files": []
}
EOF
    
    log "Backup metadata created: $metadata_file"
}

backup_database() {
    local backup_path="$1"
    
    log "Creating database backup..."
    
    local db_backup_file="$backup_path/database.sql"
    local db_user="${POSTGRES_USER:-app_user}"
    local db_name="${POSTGRES_DB:-horse_racing_db}"
    
    # Create database dump
    if docker-compose exec -T postgres pg_dump -U "$db_user" -d "$db_name" --verbose --clean --no-owner --no-privileges > "$db_backup_file" 2>/dev/null; then
        log "✓ Database backup completed: $(du -h "$db_backup_file" | cut -f1)"
        
        # Create compressed version if requested
        if [[ "$COMPRESS" == "true" ]]; then
            gzip "$db_backup_file"
            log "✓ Database backup compressed: ${db_backup_file}.gz"
        fi
        
        return 0
    else
        error "✗ Database backup failed"
        return 1
    fi
}

backup_application_files() {
    local backup_path="$1"
    
    log "Creating application files backup..."
    
    local app_backup_dir="$backup_path/application"
    mkdir -p "$app_backup_dir"
    
    # Files and directories to backup
    local backup_items=(
        "app.py"
        "config/"
        "models/"
        "routes/"
        "utils/"
        "static/"
        "templates/"
        "requirements.txt"
        "docker-compose.yml"
        "Dockerfile"
        ".env.$ENVIRONMENT"
        "nginx/"
        "ssl/"
    )
    
    # Copy application files
    for item in "${backup_items[@]}"; do
        if [[ -e "$PROJECT_ROOT/$item" ]]; then
            cp -r "$PROJECT_ROOT/$item" "$app_backup_dir/"
            log "✓ Backed up: $item"
        else
            warn "⚠ Item not found, skipping: $item"
        fi
    done
    
    # Create tar archive if compression is enabled
    if [[ "$COMPRESS" == "true" ]]; then
        local tar_file="$backup_path/application.tar.gz"
        tar -czf "$tar_file" -C "$backup_path" application/
        rm -rf "$app_backup_dir"
        log "✓ Application files compressed: $(du -h "$tar_file" | cut -f1)"
    else
        log "✓ Application files backup completed: $(du -sh "$app_backup_dir" | cut -f1)"
    fi
}

backup_logs() {
    local backup_path="$1"
    
    log "Creating logs backup..."
    
    local logs_backup_dir="$backup_path/logs"
    mkdir -p "$logs_backup_dir"
    
    # Copy log files
    if [[ -d "$PROJECT_ROOT/logs" ]]; then
        cp -r "$PROJECT_ROOT/logs"/* "$logs_backup_dir/" 2>/dev/null || true
        
        # Compress logs
        if [[ "$COMPRESS" == "true" ]]; then
            local tar_file="$backup_path/logs.tar.gz"
            tar -czf "$tar_file" -C "$backup_path" logs/
            rm -rf "$logs_backup_dir"
            log "✓ Logs compressed: $(du -h "$tar_file" | cut -f1)"
        else
            log "✓ Logs backup completed: $(du -sh "$logs_backup_dir" | cut -f1)"
        fi
    else
        warn "⚠ No logs directory found"
    fi
}

backup_docker_volumes() {
    local backup_path="$1"
    
    log "Creating Docker volumes backup..."
    
    local volumes_backup_dir="$backup_path/volumes"
    mkdir -p "$volumes_backup_dir"
    
    # Get list of Docker volumes for this project
    local project_name=$(basename "$PROJECT_ROOT" | tr '[:upper:]' '[:lower:]')
    local volumes=$(docker volume ls --filter "name=${project_name}" --format "{{.Name}}" 2>/dev/null || echo "")
    
    if [[ -n "$volumes" ]]; then
        for volume in $volumes; do
            log "Backing up volume: $volume"
            
            # Create volume backup using a temporary container
            docker run --rm \
                -v "$volume:/source:ro" \
                -v "$volumes_backup_dir:/backup" \
                alpine:latest \
                tar -czf "/backup/${volume}.tar.gz" -C /source . 2>/dev/null || warn "Failed to backup volume: $volume"
        done
        
        log "✓ Docker volumes backup completed"
    else
        info "No Docker volumes found for project: $project_name"
    fi
}

backup_ssl_certificates() {
    local backup_path="$1"
    
    log "Creating SSL certificates backup..."
    
    local ssl_backup_dir="$backup_path/ssl"
    mkdir -p "$ssl_backup_dir"
    
    # Backup SSL certificates
    if [[ -d "$PROJECT_ROOT/ssl" ]]; then
        cp -r "$PROJECT_ROOT/ssl"/* "$ssl_backup_dir/" 2>/dev/null || true
        log "✓ SSL certificates backup completed"
    fi
    
    # Backup Let's Encrypt certificates if they exist
    if [[ -d "/etc/letsencrypt" ]]; then
        local letsencrypt_backup="$ssl_backup_dir/letsencrypt.tar.gz"
        tar -czf "$letsencrypt_backup" -C /etc letsencrypt/ 2>/dev/null || warn "Failed to backup Let's Encrypt certificates"
        log "✓ Let's Encrypt certificates backup completed"
    fi
}

create_full_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="${BACKUP_NAME:-full_backup_${timestamp}}"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    log "Creating full backup: $backup_name"
    mkdir -p "$backup_path"
    
    # Create backup metadata
    create_backup_metadata "$backup_path" "full"
    
    # Backup components
    backup_database "$backup_path"
    backup_application_files "$backup_path"
    backup_logs "$backup_path"
    backup_docker_volumes "$backup_path"
    backup_ssl_certificates "$backup_path"
    
    # Calculate total backup size
    local backup_size=$(du -sh "$backup_path" | cut -f1)
    log "✓ Full backup completed: $backup_size"
    
    # Update metadata with final size
    local size_bytes=$(du -sb "$backup_path" | cut -f1)
    sed -i "s/\"size_bytes\": 0/\"size_bytes\": $size_bytes/" "$backup_path/backup_metadata.json"
    
    # Upload to remote storage if configured
    if [[ "$REMOTE_BACKUP" == "true" ]]; then
        upload_to_remote "$backup_path"
    fi
    
    echo "$backup_path"
}

create_database_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="${BACKUP_NAME:-db_backup_${timestamp}}"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    log "Creating database backup: $backup_name"
    mkdir -p "$backup_path"
    
    # Create backup metadata
    create_backup_metadata "$backup_path" "database"
    
    # Backup database only
    backup_database "$backup_path"
    
    # Calculate backup size
    local backup_size=$(du -sh "$backup_path" | cut -f1)
    log "✓ Database backup completed: $backup_size"
    
    # Update metadata with final size
    local size_bytes=$(du -sb "$backup_path" | cut -f1)
    sed -i "s/\"size_bytes\": 0/\"size_bytes\": $size_bytes/" "$backup_path/backup_metadata.json"
    
    # Upload to remote storage if configured
    if [[ "$REMOTE_BACKUP" == "true" ]]; then
        upload_to_remote "$backup_path"
    fi
    
    echo "$backup_path"
}

create_application_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="${BACKUP_NAME:-app_backup_${timestamp}}"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    log "Creating application backup: $backup_name"
    mkdir -p "$backup_path"
    
    # Create backup metadata
    create_backup_metadata "$backup_path" "application"
    
    # Backup application files only
    backup_application_files "$backup_path"
    
    # Calculate backup size
    local backup_size=$(du -sh "$backup_path" | cut -f1)
    log "✓ Application backup completed: $backup_size"
    
    # Update metadata with final size
    local size_bytes=$(du -sb "$backup_path" | cut -f1)
    sed -i "s/\"size_bytes\": 0/\"size_bytes\": $size_bytes/" "$backup_path/backup_metadata.json"
    
    echo "$backup_path"
}

# =============================================================================
# Remote Backup Functions
# =============================================================================

upload_to_s3() {
    local backup_path="$1"
    local backup_name=$(basename "$backup_path")
    
    if [[ -z "$S3_BUCKET" ]]; then
        warn "S3 bucket not configured"
        return 1
    fi
    
    log "Uploading backup to S3: s3://$S3_BUCKET/$backup_name"
    
    # Create tar archive for upload
    local tar_file="$backup_path.tar.gz"
    tar -czf "$tar_file" -C "$BACKUP_DIR" "$backup_name"
    
    # Upload to S3
    if command -v aws &> /dev/null; then
        if aws s3 cp "$tar_file" "s3://$S3_BUCKET/$backup_name.tar.gz"; then
            log "✓ Backup uploaded to S3 successfully"
            rm "$tar_file"
            return 0
        else
            error "✗ Failed to upload backup to S3"
            rm "$tar_file"
            return 1
        fi
    else
        error "AWS CLI not installed"
        rm "$tar_file"
        return 1
    fi
}

upload_to_remote() {
    local backup_path="$1"
    
    log "Uploading backup to remote storage..."
    
    # Currently only S3 is supported
    upload_to_s3 "$backup_path"
}

# =============================================================================
# Rollback Functions
# =============================================================================

list_backups() {
    log "Available backups:"
    
    if [[ ! -d "$BACKUP_DIR" || -z "$(ls -A "$BACKUP_DIR" 2>/dev/null)" ]]; then
        warn "No backups found in $BACKUP_DIR"
        return 1
    fi
    
    # List backups with metadata
    for backup_dir in "$BACKUP_DIR"/*; do
        if [[ -d "$backup_dir" ]]; then
            local backup_name=$(basename "$backup_dir")
            local metadata_file="$backup_dir/backup_metadata.json"
            
            if [[ -f "$metadata_file" ]]; then
                local timestamp=$(jq -r '.timestamp' "$metadata_file" 2>/dev/null || echo "unknown")
                local backup_type=$(jq -r '.backup_type' "$metadata_file" 2>/dev/null || echo "unknown")
                local size=$(du -sh "$backup_dir" | cut -f1)
                
                log "  $backup_name ($backup_type, $timestamp, $size)"
            else
                local size=$(du -sh "$backup_dir" | cut -f1)
                log "  $backup_name (no metadata, $size)"
            fi
        fi
    done
}

validate_backup() {
    local backup_path="$1"
    
    if [[ ! -d "$backup_path" ]]; then
        error "Backup directory not found: $backup_path"
        return 1
    fi
    
    log "Validating backup: $(basename "$backup_path")"
    
    # Check metadata
    local metadata_file="$backup_path/backup_metadata.json"
    if [[ -f "$metadata_file" ]]; then
        if jq . "$metadata_file" &> /dev/null; then
            log "✓ Backup metadata is valid"
        else
            warn "⚠ Backup metadata is invalid"
        fi
    else
        warn "⚠ No backup metadata found"
    fi
    
    # Check database backup
    if [[ -f "$backup_path/database.sql" || -f "$backup_path/database.sql.gz" ]]; then
        log "✓ Database backup found"
    else
        warn "⚠ No database backup found"
    fi
    
    # Check application backup
    if [[ -d "$backup_path/application" || -f "$backup_path/application.tar.gz" ]]; then
        log "✓ Application backup found"
    else
        warn "⚠ No application backup found"
    fi
    
    log "Backup validation completed"
    return 0
}

rollback_database() {
    local backup_path="$1"
    
    log "Rolling back database..."
    
    local db_backup_file=""
    if [[ -f "$backup_path/database.sql.gz" ]]; then
        db_backup_file="$backup_path/database.sql.gz"
        log "Using compressed database backup"
    elif [[ -f "$backup_path/database.sql" ]]; then
        db_backup_file="$backup_path/database.sql"
        log "Using uncompressed database backup"
    else
        error "No database backup found in $backup_path"
        return 1
    fi
    
    local db_user="${POSTGRES_USER:-app_user}"
    local db_name="${POSTGRES_DB:-horse_racing_db}"
    
    # Create database backup before rollback
    log "Creating safety backup before rollback..."
    local safety_backup_path="$BACKUP_DIR/safety_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$safety_backup_path"
    backup_database "$safety_backup_path"
    
    # Stop application
    log "Stopping application..."
    docker-compose stop app
    
    # Restore database
    if [[ "$db_backup_file" == *.gz ]]; then
        if gunzip -c "$db_backup_file" | docker-compose exec -T postgres psql -U "$db_user" -d "$db_name"; then
            log "✓ Database rollback completed"
        else
            error "✗ Database rollback failed"
            return 1
        fi
    else
        if docker-compose exec -T postgres psql -U "$db_user" -d "$db_name" < "$db_backup_file"; then
            log "✓ Database rollback completed"
        else
            error "✗ Database rollback failed"
            return 1
        fi
    fi
    
    # Start application
    log "Starting application..."
    docker-compose start app
    
    return 0
}

rollback_application() {
    local backup_path="$1"
    
    log "Rolling back application files..."
    
    # Create safety backup of current application
    log "Creating safety backup of current application..."
    local safety_backup_path="$BACKUP_DIR/safety_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$safety_backup_path"
    backup_application_files "$safety_backup_path"
    
    # Stop application
    log "Stopping application..."
    docker-compose down
    
    # Restore application files
    if [[ -f "$backup_path/application.tar.gz" ]]; then
        log "Extracting compressed application backup..."
        tar -xzf "$backup_path/application.tar.gz" -C "$backup_path"
    fi
    
    if [[ -d "$backup_path/application" ]]; then
        log "Restoring application files..."
        
        # Backup items to restore
        local restore_items=(
            "app.py"
            "config/"
            "models/"
            "routes/"
            "utils/"
            "static/"
            "templates/"
            "requirements.txt"
            "docker-compose.yml"
            "Dockerfile"
            "nginx/"
        )
        
        for item in "${restore_items[@]}"; do
            if [[ -e "$backup_path/application/$item" ]]; then
                rm -rf "$PROJECT_ROOT/$item"
                cp -r "$backup_path/application/$item" "$PROJECT_ROOT/"
                log "✓ Restored: $item"
            fi
        done
        
        log "✓ Application files rollback completed"
    else
        error "No application backup found in $backup_path"
        return 1
    fi
    
    # Rebuild and start application
    log "Rebuilding and starting application..."
    docker-compose build
    docker-compose up -d
    
    return 0
}

rollback_ssl() {
    local backup_path="$1"
    
    if [[ -d "$backup_path/ssl" ]]; then
        log "Rolling back SSL certificates..."
        
        # Backup current SSL certificates
        if [[ -d "$PROJECT_ROOT/ssl" ]]; then
            mv "$PROJECT_ROOT/ssl" "$PROJECT_ROOT/ssl.backup.$(date +%Y%m%d_%H%M%S)"
        fi
        
        # Restore SSL certificates
        cp -r "$backup_path/ssl" "$PROJECT_ROOT/"
        log "✓ SSL certificates rollback completed"
        
        # Restart nginx
        if docker-compose ps nginx 2>/dev/null | grep -q "Up"; then
            docker-compose restart nginx
            log "✓ Nginx restarted"
        fi
    else
        info "No SSL backup found, skipping SSL rollback"
    fi
}

perform_full_rollback() {
    local backup_path="$1"
    
    log "Performing full rollback from: $(basename "$backup_path")"
    
    # Validate backup
    validate_backup "$backup_path" || return 1
    
    # Confirm rollback
    warn "This will replace current application and database with backup data!"
    read -p "Are you sure you want to continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log "Rollback cancelled"
        return 1
    fi
    
    # Perform rollback
    rollback_database "$backup_path"
    rollback_application "$backup_path"
    rollback_ssl "$backup_path"
    
    log "✓ Full rollback completed successfully"
}

# =============================================================================
# Cleanup Functions
# =============================================================================

cleanup_old_backups() {
    log "Cleaning up old backups (retention: $RETENTION_DAYS days)..."
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        info "No backup directory found"
        return 0
    fi
    
    local deleted_count=0
    
    # Find and delete old backups
    while IFS= read -r -d '' backup_dir; do
        local backup_name=$(basename "$backup_dir")
        local backup_age=$(find "$backup_dir" -maxdepth 0 -mtime +$RETENTION_DAYS 2>/dev/null | wc -l)
        
        if [[ $backup_age -gt 0 ]]; then
            log "Deleting old backup: $backup_name"
            rm -rf "$backup_dir"
            ((deleted_count++))
        fi
    done < <(find "$BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
    
    log "✓ Cleanup completed: $deleted_count old backups deleted"
}

# =============================================================================
# Scheduled Backup Functions
# =============================================================================

setup_scheduled_backups() {
    log "Setting up scheduled backups..."
    
    # Create backup script for cron
    local backup_script="$SCRIPT_DIR/scheduled-backup.sh"
    
    cat > "$backup_script" << EOF
#!/bin/bash
# Scheduled backup script

cd "$PROJECT_ROOT"
"$SCRIPT_DIR/backup-rollback.sh" $ENVIRONMENT backup --type full --compress --cleanup
EOF

    chmod +x "$backup_script"
    
    # Add cron job for daily backups
    local cron_job="0 2 * * * $backup_script >> $LOG_DIR/scheduled-backup.log 2>&1"
    
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "$backup_script"; then
        log "Scheduled backup cron job already exists"
    else
        log "Adding cron job for daily backups..."
        (crontab -l 2>/dev/null; echo "$cron_job") | crontab -
        log "✓ Cron job added: Daily backup at 2 AM"
    fi
    
    log "✓ Scheduled backups setup completed"
}

# =============================================================================
# Usage and Help
# =============================================================================

show_usage() {
    cat << EOF
Backup and Rollback Script

Usage: $0 [environment] [action] [options]

Environments:
    development     Development environment
    staging         Staging environment
    production      Production environment (default)

Actions:
    backup          Create backup (default)
    rollback        Restore from backup
    list            List available backups
    cleanup         Clean up old backups
    validate        Validate backup integrity
    schedule        Setup scheduled backups

Backup Types:
    full            Complete backup (database + application + logs + volumes)
    database        Database only
    application     Application files only

Options:
    --type TYPE             Backup type (full, database, application)
    --name NAME             Custom backup name
    --target BACKUP_NAME    Target backup for rollback
    --compress              Compress backup files
    --no-compress           Don't compress backup files
    --remote                Upload to remote storage
    --s3-bucket BUCKET      S3 bucket for remote backup
    --retention DAYS        Retention period in days (default: 30)
    --cleanup               Clean up old backups after creating new one
    --help                  Show this help message

Examples:
    $0 production backup --type full --compress
    $0 production backup --type database --name pre_migration_backup
    $0 production rollback --target full_backup_20240115_143022
    $0 production list
    $0 production cleanup --retention 7
    $0 production schedule

EOF
}

# =============================================================================
# Main Function
# =============================================================================

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            development|staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            backup|rollback|list|cleanup|validate|schedule)
                ACTION="$1"
                shift
                ;;
            --type)
                BACKUP_TYPE="$2"
                shift 2
                ;;
            --name)
                BACKUP_NAME="$2"
                shift 2
                ;;
            --target)
                ROLLBACK_TARGET="$2"
                shift 2
                ;;
            --compress)
                COMPRESS=true
                shift
                ;;
            --no-compress)
                COMPRESS=false
                shift
                ;;
            --remote)
                REMOTE_BACKUP=true
                shift
                ;;
            --s3-bucket)
                S3_BUCKET="$2"
                REMOTE_BACKUP=true
                shift 2
                ;;
            --retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            --cleanup)
                CLEANUP_AFTER=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use --help for usage information."
                exit 1
                ;;
        esac
    done
    
    log "Backup/Rollback for environment: $ENVIRONMENT"
    log "Action: $ACTION"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Load environment configuration
    local env_file="$PROJECT_ROOT/.env.$ENVIRONMENT"
    if [[ -f "$env_file" ]]; then
        set -a
        source "$env_file"
        set +a
    fi
    
    # Execute action
    case $ACTION in
        backup)
            case $BACKUP_TYPE in
                full)
                    backup_path=$(create_full_backup)
                    ;;
                database)
                    backup_path=$(create_database_backup)
                    ;;
                application)
                    backup_path=$(create_application_backup)
                    ;;
                *)
                    error "Unknown backup type: $BACKUP_TYPE"
                    exit 1
                    ;;
            esac
            
            log "Backup created: $backup_path"
            
            # Cleanup old backups if requested
            if [[ "${CLEANUP_AFTER:-false}" == "true" ]]; then
                cleanup_old_backups
            fi
            ;;
        rollback)
            if [[ -z "$ROLLBACK_TARGET" ]]; then
                error "Rollback target is required. Use --target option."
                list_backups
                exit 1
            fi
            
            local backup_path="$BACKUP_DIR/$ROLLBACK_TARGET"
            perform_full_rollback "$backup_path"
            ;;
        list)
            list_backups
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        validate)
            if [[ -n "$ROLLBACK_TARGET" ]]; then
                validate_backup "$BACKUP_DIR/$ROLLBACK_TARGET"
            else
                # Validate all backups
                for backup_dir in "$BACKUP_DIR"/*; do
                    if [[ -d "$backup_dir" ]]; then
                        validate_backup "$backup_dir"
                    fi
                done
            fi
            ;;
        schedule)
            setup_scheduled_backups
            ;;
        *)
            error "Unknown action: $ACTION"
            exit 1
            ;;
    esac
    
    log "Backup/Rollback script completed!"
}

# Run main function with all arguments
main "$@"