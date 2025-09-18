#!/bin/bash
set -e

# PostgreSQL Failover Script
# This script promotes a replica to primary when the primary fails

# Configuration
PRIMARY_HOST=${POSTGRES_PRIMARY_HOST:-postgres-primary}
REPLICA_HOST=${POSTGRES_REPLICA_HOST:-postgres-replica}
POSTGRES_USER=${POSTGRES_USER:-hrp_user}
POSTGRES_DB=${POSTGRES_DB:-hrp_database}
REPLICATION_USER=${POSTGRES_REPLICATION_USER:-replicator}

# Logging
LOG_FILE="/var/log/postgres-failover.log"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if primary is down
check_primary_health() {
    if pg_isready -h "$PRIMARY_HOST" -p 5432 -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
        return 0  # Primary is healthy
    else
        return 1  # Primary is down
    fi
}

# Check if replica is healthy
check_replica_health() {
    if pg_isready -h "$REPLICA_HOST" -p 5432 -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
        return 0  # Replica is healthy
    else
        return 1  # Replica is down
    fi
}

# Promote replica to primary
promote_replica() {
    log "Promoting replica $REPLICA_HOST to primary..."
    
    # Connect to replica and promote it
    if docker exec hrp-postgres-replica pg_ctl promote -D /var/lib/postgresql/data; then
        log "Replica promoted successfully"
        
        # Update application configuration to point to new primary
        update_app_config
        
        # Restart application instances to pick up new config
        restart_applications
        
        # Send notification
        send_notification "PostgreSQL failover completed. Replica $REPLICA_HOST is now primary."
        
        return 0
    else
        log "ERROR: Failed to promote replica"
        send_notification "CRITICAL: PostgreSQL failover failed!"
        return 1
    fi
}

# Update application configuration
update_app_config() {
    log "Updating application configuration..."
    
    # Update environment variables for all app instances
    for container in hrp-app1 hrp-app2 hrp-app3; do
        if docker ps --format "table {{.Names}}" | grep -q "$container"; then
            log "Updating $container configuration"
            # This would typically involve updating environment variables or config files
            # For now, we'll restart the containers to pick up the new database host
        fi
    done
}

# Restart application instances
restart_applications() {
    log "Restarting application instances..."
    
    # Graceful restart of application containers
    for container in hrp-app1 hrp-app2 hrp-app3; do
        if docker ps --format "table {{.Names}}" | grep -q "$container"; then
            log "Restarting $container"
            docker restart "$container" || log "WARNING: Failed to restart $container"
        fi
    done
    
    # Wait for applications to come back online
    sleep 30
    
    # Verify applications are healthy
    for i in {1..3}; do
        if curl -f "http://app$i:8000/health" >/dev/null 2>&1; then
            log "App$i is healthy after restart"
        else
            log "WARNING: App$i health check failed"
        fi
    done
}

# Send notification
send_notification() {
    local message="$1"
    log "$message"
    
    # Send to monitoring system
    if command -v curl >/dev/null 2>&1; then
        # Send to webhook or alerting system
        curl -X POST -H "Content-Type: application/json" \
             -d "{\"text\":\"$message\",\"severity\":\"critical\"}" \
             "${WEBHOOK_URL:-http://monitoring:9093/api/v1/alerts}" >/dev/null 2>&1 || true
    fi
    
    # Send email if configured
    if [ -n "$ALERT_EMAIL" ] && command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "PostgreSQL Failover Alert" "$ALERT_EMAIL" || true
    fi
}

# Main failover logic
main() {
    log "Starting PostgreSQL failover check..."
    
    # Check if primary is down
    if check_primary_health; then
        log "Primary database is healthy, no failover needed"
        exit 0
    fi
    
    log "Primary database is down, checking replica..."
    
    # Check if replica is healthy
    if ! check_replica_health; then
        log "ERROR: Both primary and replica are down!"
        send_notification "CRITICAL: Both PostgreSQL primary and replica are down!"
        exit 1
    fi
    
    log "Replica is healthy, initiating failover..."
    
    # Wait a bit to ensure primary is really down (avoid split-brain)
    sleep 10
    
    # Double-check primary is still down
    if check_primary_health; then
        log "Primary came back online, aborting failover"
        exit 0
    fi
    
    # Perform failover
    if promote_replica; then
        log "Failover completed successfully"
        exit 0
    else
        log "Failover failed"
        exit 1
    fi
}

# Run main function
main "$@"