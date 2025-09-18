#!/bin/bash
set -e

# Redis Failover Script
# This script handles Redis failover using Sentinel

# Configuration
REDIS_PRIMARY_HOST=${REDIS_PRIMARY_HOST:-redis-primary}
REDIS_REPLICA_HOST=${REDIS_REPLICA_HOST:-redis-replica}
SENTINEL_HOST=${REDIS_SENTINEL_HOST:-redis-sentinel}
SENTINEL_PORT=${REDIS_SENTINEL_PORT:-26379}
REDIS_MASTER_NAME=${REDIS_MASTER_NAME:-mymaster}
REDIS_PASSWORD=${REDIS_PASSWORD:-your_redis_password}

# Logging
LOG_FILE="/var/log/redis-failover.log"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check Redis master health
check_redis_master() {
    if redis-cli -h "$REDIS_PRIMARY_HOST" -p 6379 -a "$REDIS_PASSWORD" ping >/dev/null 2>&1; then
        return 0  # Master is healthy
    else
        return 1  # Master is down
    fi
}

# Check Redis replica health
check_redis_replica() {
    if redis-cli -h "$REDIS_REPLICA_HOST" -p 6379 -a "$REDIS_PASSWORD" ping >/dev/null 2>&1; then
        return 0  # Replica is healthy
    else
        return 1  # Replica is down
    fi
}

# Check Sentinel health
check_sentinel() {
    if redis-cli -h "$SENTINEL_HOST" -p "$SENTINEL_PORT" ping >/dev/null 2>&1; then
        return 0  # Sentinel is healthy
    else
        return 1  # Sentinel is down
    fi
}

# Get current master from Sentinel
get_current_master() {
    redis-cli -h "$SENTINEL_HOST" -p "$SENTINEL_PORT" \
        SENTINEL get-master-addr-by-name "$REDIS_MASTER_NAME" 2>/dev/null | head -1
}

# Force failover through Sentinel
force_failover() {
    log "Forcing Redis failover through Sentinel..."
    
    if redis-cli -h "$SENTINEL_HOST" -p "$SENTINEL_PORT" \
        SENTINEL failover "$REDIS_MASTER_NAME" >/dev/null 2>&1; then
        log "Failover command sent to Sentinel"
        
        # Wait for failover to complete
        local timeout=60
        local count=0
        
        while [ $count -lt $timeout ]; do
            local new_master=$(get_current_master)
            if [ "$new_master" != "$REDIS_PRIMARY_HOST" ] && [ -n "$new_master" ]; then
                log "Failover completed. New master: $new_master"
                
                # Update application configuration
                update_app_redis_config "$new_master"
                
                # Send notification
                send_notification "Redis failover completed. New master: $new_master"
                
                return 0
            fi
            
            sleep 2
            count=$((count + 2))
        done
        
        log "ERROR: Failover timeout"
        return 1
    else
        log "ERROR: Failed to send failover command to Sentinel"
        return 1
    fi
}

# Update application Redis configuration
update_app_redis_config() {
    local new_master="$1"
    log "Updating application Redis configuration to use master: $new_master"
    
    # Update environment variables for all app instances
    for container in hrp-app1 hrp-app2 hrp-app3; do
        if docker ps --format "table {{.Names}}" | grep -q "$container"; then
            log "Updating Redis config for $container"
            # In a real scenario, you might update config files or restart with new env vars
            # For now, applications should be using Sentinel for Redis discovery
        fi
    done
}

# Check Redis cluster health
check_cluster_health() {
    log "Checking Redis cluster health..."
    
    # Check Sentinel
    if ! check_sentinel; then
        log "WARNING: Sentinel is down"
        return 1
    fi
    
    # Get master info from Sentinel
    local current_master=$(get_current_master)
    if [ -z "$current_master" ]; then
        log "ERROR: Cannot get master info from Sentinel"
        return 1
    fi
    
    log "Current Redis master according to Sentinel: $current_master"
    
    # Check if current master is responding
    if redis-cli -h "$current_master" -p 6379 -a "$REDIS_PASSWORD" ping >/dev/null 2>&1; then
        log "Current Redis master is healthy"
        return 0
    else
        log "Current Redis master is not responding"
        return 1
    fi
}

# Monitor Redis replication lag
check_replication_lag() {
    local master_host="$1"
    
    # Get replication info from master
    local master_offset=$(redis-cli -h "$master_host" -p 6379 -a "$REDIS_PASSWORD" \
        INFO replication | grep "master_repl_offset" | cut -d: -f2 | tr -d '\r')
    
    # Get replication info from replica
    local replica_offset=$(redis-cli -h "$REDIS_REPLICA_HOST" -p 6379 -a "$REDIS_PASSWORD" \
        INFO replication | grep "slave_repl_offset" | cut -d: -f2 | tr -d '\r')
    
    if [ -n "$master_offset" ] && [ -n "$replica_offset" ]; then
        local lag=$((master_offset - replica_offset))
        log "Replication lag: $lag bytes"
        
        # Alert if lag is too high (>1MB)
        if [ "$lag" -gt 1048576 ]; then
            log "WARNING: High replication lag detected: $lag bytes"
            send_notification "WARNING: Redis replication lag is high: $lag bytes"
        fi
    fi
}

# Send notification
send_notification() {
    local message="$1"
    log "$message"
    
    # Send to monitoring system
    if command -v curl >/dev/null 2>&1; then
        curl -X POST -H "Content-Type: application/json" \
             -d "{\"text\":\"$message\",\"severity\":\"warning\"}" \
             "${WEBHOOK_URL:-http://monitoring:9093/api/v1/alerts}" >/dev/null 2>&1 || true
    fi
    
    # Send email if configured
    if [ -n "$ALERT_EMAIL" ] && command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "Redis Failover Alert" "$ALERT_EMAIL" || true
    fi
}

# Main function
main() {
    local action="${1:-check}"
    
    case "$action" in
        "check")
            log "Checking Redis cluster health..."
            if check_cluster_health; then
                # Check replication lag
                local current_master=$(get_current_master)
                if [ -n "$current_master" ]; then
                    check_replication_lag "$current_master"
                fi
                log "Redis cluster is healthy"
            else
                log "Redis cluster health check failed"
                exit 1
            fi
            ;;
        "failover")
            log "Manual Redis failover requested..."
            if force_failover; then
                log "Manual failover completed successfully"
            else
                log "Manual failover failed"
                exit 1
            fi
            ;;
        "auto-failover")
            log "Checking if automatic failover is needed..."
            if ! check_cluster_health; then
                log "Cluster unhealthy, attempting automatic failover..."
                if force_failover; then
                    log "Automatic failover completed successfully"
                else
                    log "Automatic failover failed"
                    exit 1
                fi
            else
                log "Cluster is healthy, no failover needed"
            fi
            ;;
        *)
            echo "Usage: $0 {check|failover|auto-failover}"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"