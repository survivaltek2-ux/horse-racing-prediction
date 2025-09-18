#!/bin/bash
set -e

# Storage Synchronization Script
# This script synchronizes data between different storage systems

# Configuration
SYNC_INTERVAL=${SYNC_INTERVAL:-300}  # 5 minutes default
GLUSTER_MOUNT=${GLUSTER_MOUNT:-/mnt/gluster}
MINIO_ENDPOINT=${MINIO_ENDPOINT:-http://minio-1:9000}
MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-hrp_admin}
MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-your_secure_minio_password}
MINIO_BUCKET=${MINIO_BUCKET:-hrp-data}

# Logging
LOG_FILE="/var/log/storage-sync.log"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Install required packages
install_dependencies() {
    log "Installing dependencies..."
    apk update
    apk add --no-cache \
        rsync \
        curl \
        jq \
        python3 \
        py3-pip \
        glusterfs-client
    
    # Install MinIO client
    curl -o /usr/local/bin/mc https://dl.min.io/client/mc/release/linux-amd64/mc
    chmod +x /usr/local/bin/mc
    
    # Install AWS CLI for S3 compatibility
    pip3 install awscli
    
    log "Dependencies installed successfully"
}

# Configure MinIO client
configure_minio() {
    log "Configuring MinIO client..."
    
    mc alias set hrp-minio "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
    
    # Create bucket if it doesn't exist
    if ! mc ls hrp-minio/"$MINIO_BUCKET" >/dev/null 2>&1; then
        log "Creating MinIO bucket: $MINIO_BUCKET"
        mc mb hrp-minio/"$MINIO_BUCKET"
    fi
    
    log "MinIO client configured successfully"
}

# Mount GlusterFS volume
mount_gluster() {
    log "Mounting GlusterFS volume..."
    
    # Wait for GlusterFS nodes to be ready
    local timeout=300
    local count=0
    
    while [ $count -lt $timeout ]; do
        if gluster volume status hrp-data >/dev/null 2>&1; then
            log "GlusterFS volume is ready"
            break
        fi
        log "Waiting for GlusterFS volume to be ready..."
        sleep 10
        count=$((count + 10))
    done
    
    if [ $count -ge $timeout ]; then
        log "ERROR: GlusterFS volume not ready after $timeout seconds"
        return 1
    fi
    
    # Mount the volume
    mkdir -p "$GLUSTER_MOUNT"
    if ! mountpoint -q "$GLUSTER_MOUNT"; then
        mount -t glusterfs storage-node-1:/hrp-data "$GLUSTER_MOUNT"
        log "GlusterFS volume mounted at $GLUSTER_MOUNT"
    else
        log "GlusterFS volume already mounted"
    fi
}

# Sync data from GlusterFS to MinIO
sync_gluster_to_minio() {
    log "Syncing data from GlusterFS to MinIO..."
    
    if [ ! -d "$GLUSTER_MOUNT" ]; then
        log "ERROR: GlusterFS mount point not found"
        return 1
    fi
    
    # Sync using mc mirror command
    if mc mirror --overwrite --remove "$GLUSTER_MOUNT"/ hrp-minio/"$MINIO_BUCKET"/gluster/; then
        log "Successfully synced GlusterFS to MinIO"
        return 0
    else
        log "ERROR: Failed to sync GlusterFS to MinIO"
        return 1
    fi
}

# Sync data from MinIO to GlusterFS (for disaster recovery)
sync_minio_to_gluster() {
    log "Syncing data from MinIO to GlusterFS..."
    
    if [ ! -d "$GLUSTER_MOUNT" ]; then
        log "ERROR: GlusterFS mount point not found"
        return 1
    fi
    
    # Create temporary directory for MinIO data
    local temp_dir="/tmp/minio-sync"
    mkdir -p "$temp_dir"
    
    # Download from MinIO
    if mc mirror hrp-minio/"$MINIO_BUCKET"/gluster/ "$temp_dir"/; then
        # Sync to GlusterFS
        if rsync -av --delete "$temp_dir"/ "$GLUSTER_MOUNT"/; then
            log "Successfully synced MinIO to GlusterFS"
            rm -rf "$temp_dir"
            return 0
        else
            log "ERROR: Failed to rsync to GlusterFS"
            rm -rf "$temp_dir"
            return 1
        fi
    else
        log "ERROR: Failed to download from MinIO"
        rm -rf "$temp_dir"
        return 1
    fi
}

# Check storage health
check_storage_health() {
    log "Checking storage health..."
    
    local gluster_healthy=false
    local minio_healthy=false
    
    # Check GlusterFS
    if gluster volume status hrp-data >/dev/null 2>&1; then
        log "✓ GlusterFS is healthy"
        gluster_healthy=true
    else
        log "✗ GlusterFS has issues"
    fi
    
    # Check MinIO
    if mc admin info hrp-minio >/dev/null 2>&1; then
        log "✓ MinIO is healthy"
        minio_healthy=true
    else
        log "✗ MinIO has issues"
    fi
    
    if $gluster_healthy && $minio_healthy; then
        log "✓ All storage systems are healthy"
        return 0
    else
        log "✗ Some storage systems have issues"
        return 1
    fi
}

# Verify data consistency
verify_data_consistency() {
    log "Verifying data consistency..."
    
    local temp_dir="/tmp/consistency-check"
    mkdir -p "$temp_dir"
    
    # Get checksums from GlusterFS
    find "$GLUSTER_MOUNT" -type f -exec md5sum {} \; | sort > "$temp_dir/gluster-checksums.txt"
    
    # Download from MinIO and get checksums
    mc mirror hrp-minio/"$MINIO_BUCKET"/gluster/ "$temp_dir/minio-data/"
    find "$temp_dir/minio-data" -type f -exec md5sum {} \; | sed "s|$temp_dir/minio-data|$GLUSTER_MOUNT|g" | sort > "$temp_dir/minio-checksums.txt"
    
    # Compare checksums
    if diff "$temp_dir/gluster-checksums.txt" "$temp_dir/minio-checksums.txt" >/dev/null; then
        log "✓ Data consistency verified"
        rm -rf "$temp_dir"
        return 0
    else
        log "✗ Data inconsistency detected"
        log "Checksum differences:"
        diff "$temp_dir/gluster-checksums.txt" "$temp_dir/minio-checksums.txt" | head -20 | tee -a "$LOG_FILE"
        rm -rf "$temp_dir"
        return 1
    fi
}

# Send notification
send_notification() {
    local message="$1"
    local severity="${2:-info}"
    
    log "$message"
    
    # Send to monitoring system
    if command -v curl >/dev/null 2>&1; then
        curl -X POST -H "Content-Type: application/json" \
             -d "{\"text\":\"$message\",\"severity\":\"$severity\"}" \
             "${WEBHOOK_URL:-http://monitoring:9093/api/v1/alerts}" >/dev/null 2>&1 || true
    fi
}

# Cleanup old sync logs
cleanup_logs() {
    # Keep only last 7 days of logs
    find /var/log -name "storage-sync.log*" -mtime +7 -delete 2>/dev/null || true
}

# Main synchronization loop
main_sync_loop() {
    log "Starting storage synchronization service..."
    
    # Initial setup
    install_dependencies
    configure_minio
    mount_gluster
    
    # Main loop
    while true; do
        log "Starting sync cycle..."
        
        # Check storage health
        if check_storage_health; then
            # Perform synchronization
            if sync_gluster_to_minio; then
                log "Sync cycle completed successfully"
                
                # Verify consistency periodically (every 6th cycle)
                if [ $(($(date +%s) / SYNC_INTERVAL % 6)) -eq 0 ]; then
                    if ! verify_data_consistency; then
                        send_notification "Storage data inconsistency detected" "warning"
                    fi
                fi
            else
                log "Sync cycle failed"
                send_notification "Storage synchronization failed" "error"
            fi
        else
            log "Storage health check failed, skipping sync"
            send_notification "Storage health check failed" "warning"
        fi
        
        # Cleanup old logs
        cleanup_logs
        
        # Wait for next cycle
        log "Waiting $SYNC_INTERVAL seconds for next sync cycle..."
        sleep "$SYNC_INTERVAL"
    done
}

# Handle different commands
case "${1:-sync}" in
    "sync")
        main_sync_loop
        ;;
    "check")
        install_dependencies
        configure_minio
        mount_gluster
        check_storage_health
        ;;
    "verify")
        install_dependencies
        configure_minio
        mount_gluster
        verify_data_consistency
        ;;
    "restore")
        install_dependencies
        configure_minio
        mount_gluster
        sync_minio_to_gluster
        ;;
    *)
        echo "Usage: $0 {sync|check|verify|restore}"
        echo ""
        echo "Commands:"
        echo "  sync     - Start continuous synchronization (default)"
        echo "  check    - Check storage health"
        echo "  verify   - Verify data consistency"
        echo "  restore  - Restore data from MinIO to GlusterFS"
        exit 1
        ;;
esac