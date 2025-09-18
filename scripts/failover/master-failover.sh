#!/bin/bash
set -e

# Master Failover Orchestration Script
# This script coordinates failover operations across all services

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSTGRES_FAILOVER_SCRIPT="$SCRIPT_DIR/postgres-failover.sh"
REDIS_FAILOVER_SCRIPT="$SCRIPT_DIR/redis-failover.sh"
APP_FAILOVER_SCRIPT="$SCRIPT_DIR/app-failover.sh"

# Health check intervals (seconds)
HEALTH_CHECK_INTERVAL=30
POSTGRES_CHECK_INTERVAL=60
REDIS_CHECK_INTERVAL=45
APP_CHECK_INTERVAL=30

# Logging
LOG_FILE="/var/log/master-failover.log"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if all required scripts exist
check_dependencies() {
    local missing_scripts=()
    
    for script in "$POSTGRES_FAILOVER_SCRIPT" "$REDIS_FAILOVER_SCRIPT" "$APP_FAILOVER_SCRIPT"; do
        if [ ! -f "$script" ]; then
            missing_scripts+=("$script")
        elif [ ! -x "$script" ]; then
            log "Making $script executable"
            chmod +x "$script"
        fi
    done
    
    if [ ${#missing_scripts[@]} -gt 0 ]; then
        log "ERROR: Missing required scripts: ${missing_scripts[*]}"
        exit 1
    fi
    
    log "All failover scripts are available"
}

# Check PostgreSQL health
check_postgres_health() {
    if "$POSTGRES_FAILOVER_SCRIPT" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check Redis health
check_redis_health() {
    if "$REDIS_FAILOVER_SCRIPT" check >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check application health
check_app_health() {
    if "$APP_FAILOVER_SCRIPT" check >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Perform comprehensive health check
comprehensive_health_check() {
    log "Performing comprehensive health check..."
    
    local postgres_healthy=false
    local redis_healthy=false
    local app_healthy=false
    
    # Check PostgreSQL
    if check_postgres_health; then
        log "✓ PostgreSQL cluster is healthy"
        postgres_healthy=true
    else
        log "✗ PostgreSQL cluster has issues"
    fi
    
    # Check Redis
    if check_redis_health; then
        log "✓ Redis cluster is healthy"
        redis_healthy=true
    else
        log "✗ Redis cluster has issues"
    fi
    
    # Check Applications
    if check_app_health; then
        log "✓ Application cluster is healthy"
        app_healthy=true
    else
        log "✗ Application cluster has issues"
    fi
    
    # Overall health assessment
    if $postgres_healthy && $redis_healthy && $app_healthy; then
        log "✓ All systems are healthy"
        return 0
    else
        log "✗ Some systems require attention"
        return 1
    fi
}

# Handle PostgreSQL failover
handle_postgres_failover() {
    log "Initiating PostgreSQL failover..."
    
    if "$POSTGRES_FAILOVER_SCRIPT"; then
        log "PostgreSQL failover completed successfully"
        
        # Wait for database to stabilize
        sleep 30
        
        # Restart applications to pick up new database connection
        log "Restarting applications after database failover..."
        "$APP_FAILOVER_SCRIPT" restart all
        
        return 0
    else
        log "PostgreSQL failover failed"
        return 1
    fi
}

# Handle Redis failover
handle_redis_failover() {
    log "Initiating Redis failover..."
    
    if "$REDIS_FAILOVER_SCRIPT" auto-failover; then
        log "Redis failover completed successfully"
        
        # Applications should automatically reconnect to new Redis master
        # through Sentinel, but we can restart them to be sure
        sleep 10
        log "Restarting applications after Redis failover..."
        "$APP_FAILOVER_SCRIPT" restart all
        
        return 0
    else
        log "Redis failover failed"
        return 1
    fi
}

# Handle application failover
handle_app_failover() {
    log "Initiating application failover..."
    
    if "$APP_FAILOVER_SCRIPT" failover; then
        log "Application failover completed successfully"
        return 0
    else
        log "Application failover failed"
        return 1
    fi
}

# Automated monitoring and failover
automated_monitoring() {
    log "Starting automated monitoring and failover..."
    
    local postgres_last_check=0
    local redis_last_check=0
    local app_last_check=0
    
    while true; do
        local current_time=$(date +%s)
        
        # Check PostgreSQL
        if [ $((current_time - postgres_last_check)) -ge $POSTGRES_CHECK_INTERVAL ]; then
            if ! check_postgres_health; then
                log "PostgreSQL health check failed, initiating failover..."
                handle_postgres_failover
            fi
            postgres_last_check=$current_time
        fi
        
        # Check Redis
        if [ $((current_time - redis_last_check)) -ge $REDIS_CHECK_INTERVAL ]; then
            if ! check_redis_health; then
                log "Redis health check failed, initiating failover..."
                handle_redis_failover
            fi
            redis_last_check=$current_time
        fi
        
        # Check Applications
        if [ $((current_time - app_last_check)) -ge $APP_CHECK_INTERVAL ]; then
            if ! check_app_health; then
                log "Application health check failed, initiating failover..."
                handle_app_failover
            fi
            app_last_check=$current_time
        fi
        
        # Wait before next check
        sleep $HEALTH_CHECK_INTERVAL
    done
}

# Emergency failover (all services)
emergency_failover() {
    log "EMERGENCY: Initiating full system failover..."
    
    local postgres_success=false
    local redis_success=false
    local app_success=false
    
    # Try PostgreSQL failover
    if handle_postgres_failover; then
        postgres_success=true
    fi
    
    # Try Redis failover
    if handle_redis_failover; then
        redis_success=true
    fi
    
    # Try application failover
    if handle_app_failover; then
        app_success=true
    fi
    
    # Report results
    log "Emergency failover results:"
    log "  PostgreSQL: $([ $postgres_success = true ] && echo "SUCCESS" || echo "FAILED")"
    log "  Redis: $([ $redis_success = true ] && echo "SUCCESS" || echo "FAILED")"
    log "  Applications: $([ $app_success = true ] && echo "SUCCESS" || echo "FAILED")"
    
    if $postgres_success && $redis_success && $app_success; then
        log "Emergency failover completed successfully"
        send_notification "Emergency failover completed successfully"
        return 0
    else
        log "Emergency failover partially failed"
        send_notification "CRITICAL: Emergency failover partially failed!"
        return 1
    fi
}

# Send notification
send_notification() {
    local message="$1"
    log "$message"
    
    # Send to monitoring system
    if command -v curl >/dev/null 2>&1; then
        curl -X POST -H "Content-Type: application/json" \
             -d "{\"text\":\"$message\",\"severity\":\"critical\"}" \
             "${WEBHOOK_URL:-http://monitoring:9093/api/v1/alerts}" >/dev/null 2>&1 || true
    fi
    
    # Send email if configured
    if [ -n "$ALERT_EMAIL" ] && command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "Master Failover Alert" "$ALERT_EMAIL" || true
    fi
}

# Show system status
show_status() {
    log "=== System Status ==="
    
    echo "PostgreSQL Status:"
    if check_postgres_health; then
        echo "  ✓ Healthy"
    else
        echo "  ✗ Unhealthy"
    fi
    
    echo "Redis Status:"
    if check_redis_health; then
        echo "  ✓ Healthy"
    else
        echo "  ✗ Unhealthy"
    fi
    
    echo "Application Status:"
    if check_app_health; then
        echo "  ✓ Healthy"
    else
        echo "  ✗ Unhealthy"
    fi
    
    echo ""
    comprehensive_health_check
}

# Main function
main() {
    local action="${1:-status}"
    
    # Check dependencies first
    check_dependencies
    
    case "$action" in
        "status")
            show_status
            ;;
        "check")
            comprehensive_health_check
            ;;
        "monitor")
            automated_monitoring
            ;;
        "postgres-failover")
            handle_postgres_failover
            ;;
        "redis-failover")
            handle_redis_failover
            ;;
        "app-failover")
            handle_app_failover
            ;;
        "emergency")
            emergency_failover
            ;;
        *)
            echo "Usage: $0 {status|check|monitor|postgres-failover|redis-failover|app-failover|emergency}"
            echo ""
            echo "Commands:"
            echo "  status           - Show current system status"
            echo "  check            - Perform comprehensive health check"
            echo "  monitor          - Start automated monitoring (runs continuously)"
            echo "  postgres-failover - Force PostgreSQL failover"
            echo "  redis-failover   - Force Redis failover"
            echo "  app-failover     - Force application failover"
            echo "  emergency        - Emergency failover of all services"
            exit 1
            ;;
    esac
}

# Handle signals for graceful shutdown
trap 'log "Received shutdown signal, exiting..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"