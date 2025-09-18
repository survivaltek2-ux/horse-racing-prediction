#!/bin/bash
set -e

# Storage Monitoring Script
# This script monitors storage systems health and performance

# Configuration
GLUSTER_NODES=("storage-node-1" "storage-node-2" "storage-node-3")
MINIO_NODES=("minio-1" "minio-2" "minio-3" "minio-4")
ALERT_THRESHOLD_DISK_USAGE=85
ALERT_THRESHOLD_INODE_USAGE=90
ALERT_THRESHOLD_RESPONSE_TIME=5000  # milliseconds

# Logging
LOG_FILE="/var/log/storage-monitor.log"
METRICS_FILE="/var/log/storage-metrics.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_metric() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$METRICS_FILE"
}

# Check GlusterFS cluster health
check_gluster_health() {
    log "Checking GlusterFS cluster health..."
    
    local healthy_nodes=0
    local total_nodes=${#GLUSTER_NODES[@]}
    local volume_status="unknown"
    
    # Check each node
    for node in "${GLUSTER_NODES[@]}"; do
        if gluster peer status | grep -q "$node.*Connected"; then
            log "✓ GlusterFS node $node is connected"
            healthy_nodes=$((healthy_nodes + 1))
        else
            log "✗ GlusterFS node $node is disconnected"
        fi
    done
    
    # Check volume status
    if gluster volume status hrp-data >/dev/null 2>&1; then
        volume_status="online"
        log "✓ GlusterFS volume hrp-data is online"
    else
        volume_status="offline"
        log "✗ GlusterFS volume hrp-data is offline"
    fi
    
    # Log metrics
    log_metric "gluster_healthy_nodes $healthy_nodes"
    log_metric "gluster_total_nodes $total_nodes"
    log_metric "gluster_volume_status $([ "$volume_status" = "online" ] && echo "1" || echo "0")"
    
    # Health assessment
    local health_percentage=$((healthy_nodes * 100 / total_nodes))
    if [ "$health_percentage" -ge 67 ] && [ "$volume_status" = "online" ]; then
        log "✓ GlusterFS cluster is healthy ($health_percentage% nodes healthy)"
        return 0
    else
        log "✗ GlusterFS cluster is unhealthy ($health_percentage% nodes healthy, volume: $volume_status)"
        return 1
    fi
}

# Check MinIO cluster health
check_minio_health() {
    log "Checking MinIO cluster health..."
    
    local healthy_nodes=0
    local total_nodes=${#MINIO_NODES[@]}
    
    # Check each node
    for node in "${MINIO_NODES[@]}"; do
        if curl -f -s "http://$node:9000/minio/health/live" >/dev/null 2>&1; then
            log "✓ MinIO node $node is healthy"
            healthy_nodes=$((healthy_nodes + 1))
        else
            log "✗ MinIO node $node is unhealthy"
        fi
    done
    
    # Check cluster status via admin API
    local cluster_status="unknown"
    if mc admin info hrp-minio >/dev/null 2>&1; then
        cluster_status="online"
        log "✓ MinIO cluster is accessible"
    else
        cluster_status="offline"
        log "✗ MinIO cluster is not accessible"
    fi
    
    # Log metrics
    log_metric "minio_healthy_nodes $healthy_nodes"
    log_metric "minio_total_nodes $total_nodes"
    log_metric "minio_cluster_status $([ "$cluster_status" = "online" ] && echo "1" || echo "0")"
    
    # Health assessment
    local health_percentage=$((healthy_nodes * 100 / total_nodes))
    if [ "$health_percentage" -ge 75 ] && [ "$cluster_status" = "online" ]; then
        log "✓ MinIO cluster is healthy ($health_percentage% nodes healthy)"
        return 0
    else
        log "✗ MinIO cluster is unhealthy ($health_percentage% nodes healthy, cluster: $cluster_status)"
        return 1
    fi
}

# Check disk usage
check_disk_usage() {
    log "Checking disk usage..."
    
    local critical_disks=()
    
    # Check GlusterFS mount points
    for node in "${GLUSTER_NODES[@]}"; do
        local usage=$(docker exec "hrp-gluster-${node##*-}" df /data/glusterfs/hrp-data 2>/dev/null | awk 'NR==2 {print $5}' | sed 's/%//' || echo "0")
        
        if [ "$usage" -gt "$ALERT_THRESHOLD_DISK_USAGE" ]; then
            critical_disks+=("$node:$usage%")
            log "✗ High disk usage on $node: $usage%"
        else
            log "✓ Disk usage on $node: $usage%"
        fi
        
        log_metric "disk_usage_${node//-/_} $usage"
    done
    
    # Check MinIO nodes
    for node in "${MINIO_NODES[@]}"; do
        local usage=$(docker exec "hrp-$node" df /data 2>/dev/null | awk 'NR==2 {print $5}' | sed 's/%//' || echo "0")
        
        if [ "$usage" -gt "$ALERT_THRESHOLD_DISK_USAGE" ]; then
            critical_disks+=("$node:$usage%")
            log "✗ High disk usage on $node: $usage%"
        else
            log "✓ Disk usage on $node: $usage%"
        fi
        
        log_metric "disk_usage_${node//-/_} $usage"
    done
    
    if [ ${#critical_disks[@]} -gt 0 ]; then
        send_notification "High disk usage detected: ${critical_disks[*]}" "warning"
        return 1
    else
        log "✓ All disks have acceptable usage levels"
        return 0
    fi
}

# Check storage performance
check_storage_performance() {
    log "Checking storage performance..."
    
    local test_file="/tmp/storage-perf-test"
    local gluster_mount="/mnt/gluster"
    
    # Test GlusterFS write performance
    if mountpoint -q "$gluster_mount"; then
        local start_time=$(date +%s%3N)
        dd if=/dev/zero of="$gluster_mount/perf-test" bs=1M count=10 >/dev/null 2>&1
        local end_time=$(date +%s%3N)
        local write_time=$((end_time - start_time))
        
        log_metric "gluster_write_time_ms $write_time"
        
        # Test read performance
        start_time=$(date +%s%3N)
        dd if="$gluster_mount/perf-test" of=/dev/null bs=1M >/dev/null 2>&1
        end_time=$(date +%s%3N)
        local read_time=$((end_time - start_time))
        
        log_metric "gluster_read_time_ms $read_time"
        
        # Cleanup
        rm -f "$gluster_mount/perf-test"
        
        log "GlusterFS performance: Write=${write_time}ms, Read=${read_time}ms"
        
        if [ "$write_time" -gt "$ALERT_THRESHOLD_RESPONSE_TIME" ] || [ "$read_time" -gt "$ALERT_THRESHOLD_RESPONSE_TIME" ]; then
            send_notification "GlusterFS performance degraded: Write=${write_time}ms, Read=${read_time}ms" "warning"
        fi
    else
        log "✗ GlusterFS not mounted, skipping performance test"
    fi
    
    # Test MinIO performance
    local minio_write_time=$(mc cp /dev/zero hrp-minio/hrp-data/perf-test --quiet 2>&1 | grep -o '[0-9]*ms' | sed 's/ms//' || echo "0")
    local minio_read_time=$(mc cp hrp-minio/hrp-data/perf-test /dev/null --quiet 2>&1 | grep -o '[0-9]*ms' | sed 's/ms//' || echo "0")
    
    log_metric "minio_write_time_ms $minio_write_time"
    log_metric "minio_read_time_ms $minio_read_time"
    
    # Cleanup
    mc rm hrp-minio/hrp-data/perf-test >/dev/null 2>&1 || true
    
    log "MinIO performance: Write=${minio_write_time}ms, Read=${minio_read_time}ms"
}

# Check replication status
check_replication_status() {
    log "Checking replication status..."
    
    # Check GlusterFS heal status
    local heal_info=$(gluster volume heal hrp-data info 2>/dev/null || echo "error")
    if echo "$heal_info" | grep -q "Number of entries: 0"; then
        log "✓ GlusterFS replication is healthy (no pending heals)"
        log_metric "gluster_pending_heals 0"
    else
        local pending_heals=$(echo "$heal_info" | grep -c "Number of entries:" || echo "unknown")
        log "✗ GlusterFS has pending heal operations: $pending_heals"
        log_metric "gluster_pending_heals $pending_heals"
        send_notification "GlusterFS has pending heal operations: $pending_heals" "warning"
    fi
    
    # Check MinIO replication (if configured)
    # This would depend on your MinIO replication setup
    log "MinIO replication check completed"
}

# Generate storage report
generate_storage_report() {
    local report_file="/tmp/storage-report-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "=== Storage Health Report ==="
        echo "Generated: $(date)"
        echo ""
        
        echo "=== GlusterFS Status ==="
        gluster volume status hrp-data 2>/dev/null || echo "Volume status unavailable"
        echo ""
        
        echo "=== GlusterFS Peer Status ==="
        gluster peer status 2>/dev/null || echo "Peer status unavailable"
        echo ""
        
        echo "=== MinIO Admin Info ==="
        mc admin info hrp-minio 2>/dev/null || echo "MinIO info unavailable"
        echo ""
        
        echo "=== Disk Usage ==="
        for node in "${GLUSTER_NODES[@]}"; do
            echo "GlusterFS $node:"
            docker exec "hrp-gluster-${node##*-}" df -h /data/glusterfs/hrp-data 2>/dev/null || echo "  Unavailable"
        done
        
        for node in "${MINIO_NODES[@]}"; do
            echo "MinIO $node:"
            docker exec "hrp-$node" df -h /data 2>/dev/null || echo "  Unavailable"
        done
        
    } > "$report_file"
    
    log "Storage report generated: $report_file"
    echo "$report_file"
}

# Send notification
send_notification() {
    local message="$1"
    local severity="${2:-info}"
    
    log "$message"
    
    # Send to monitoring system
    if command -v curl >/dev/null 2>&1; then
        curl -X POST -H "Content-Type: application/json" \
             -d "{\"text\":\"Storage Alert: $message\",\"severity\":\"$severity\"}" \
             "${WEBHOOK_URL:-http://monitoring:9093/api/v1/alerts}" >/dev/null 2>&1 || true
    fi
}

# Main monitoring function
main_monitor() {
    log "Starting storage monitoring..."
    
    local gluster_healthy=false
    local minio_healthy=false
    local disk_healthy=false
    
    # Run health checks
    if check_gluster_health; then
        gluster_healthy=true
    fi
    
    if check_minio_health; then
        minio_healthy=true
    fi
    
    if check_disk_usage; then
        disk_healthy=true
    fi
    
    # Additional checks
    check_replication_status
    check_storage_performance
    
    # Overall health assessment
    if $gluster_healthy && $minio_healthy && $disk_healthy; then
        log "✓ All storage systems are healthy"
        log_metric "storage_overall_health 1"
        return 0
    else
        log "✗ Some storage systems have issues"
        log_metric "storage_overall_health 0"
        return 1
    fi
}

# Main function
case "${1:-monitor}" in
    "monitor")
        main_monitor
        ;;
    "report")
        generate_storage_report
        ;;
    "gluster")
        check_gluster_health
        ;;
    "minio")
        check_minio_health
        ;;
    "disk")
        check_disk_usage
        ;;
    "performance")
        check_storage_performance
        ;;
    "replication")
        check_replication_status
        ;;
    *)
        echo "Usage: $0 {monitor|report|gluster|minio|disk|performance|replication}"
        echo ""
        echo "Commands:"
        echo "  monitor      - Run complete storage monitoring (default)"
        echo "  report       - Generate detailed storage report"
        echo "  gluster      - Check GlusterFS health only"
        echo "  minio        - Check MinIO health only"
        echo "  disk         - Check disk usage only"
        echo "  performance  - Check storage performance only"
        echo "  replication  - Check replication status only"
        exit 1
        ;;
esac