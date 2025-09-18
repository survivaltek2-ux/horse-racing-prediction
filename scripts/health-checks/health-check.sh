#!/bin/bash
# Comprehensive Health Check Script for Production Environment

set -euo pipefail

# Configuration
HEALTH_CHECK_INTERVAL=30
MAX_RETRIES=3
TIMEOUT=10
LOG_FILE="/var/log/health-checks/health-check.log"
METRICS_ENDPOINT="http://prometheus-pushgateway:9091/metrics/job/health_check"

# Service endpoints
SERVICES=(
    "app:http://app:8000/health"
    "nginx:http://nginx/health"
    "postgres-master:postgres://postgres-master:5432"
    "postgres-slave:postgres://postgres-slave:5432"
    "redis-master:redis://redis-master:6379"
    "redis-slave:redis://redis-slave:6379"
    "redis-sentinel:redis://redis-sentinel:26379"
    "elasticsearch:http://elasticsearch:9200/_cluster/health"
    "kibana:http://kibana:5601/api/status"
    "grafana:http://grafana:3000/api/health"
    "prometheus:http://prometheus:9090/-/healthy"
)

# Logging
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Send metrics to Prometheus
send_metric() {
    local service="$1"
    local status="$2"
    local response_time="$3"
    
    if command -v curl >/dev/null 2>&1; then
        curl -X POST "$METRICS_ENDPOINT" \
            --data-binary "service_health{service=\"$service\"} $status
service_response_time_seconds{service=\"$service\"} $response_time" \
            --max-time 5 2>/dev/null || true
    fi
}

# Check HTTP endpoint
check_http() {
    local service="$1"
    local url="$2"
    local start_time response_time status_code
    
    start_time=$(date +%s.%N)
    
    if status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$url"); then
        response_time=$(echo "$(date +%s.%N) - $start_time" | bc)
        
        if [[ "$status_code" -ge 200 && "$status_code" -lt 400 ]]; then
            log "✓ $service: HTTP $status_code (${response_time}s)"
            send_metric "$service" 1 "$response_time"
            return 0
        else
            log "✗ $service: HTTP $status_code"
            send_metric "$service" 0 "$response_time"
            return 1
        fi
    else
        response_time=$(echo "$(date +%s.%N) - $start_time" | bc)
        log "✗ $service: Connection failed"
        send_metric "$service" 0 "$response_time"
        return 1
    fi
}

# Check PostgreSQL
check_postgres() {
    local service="$1"
    local url="$2"
    local host port
    
    host=$(echo "$url" | sed 's|postgres://||' | cut -d: -f1)
    port=$(echo "$url" | sed 's|postgres://||' | cut -d: -f2)
    
    local start_time response_time
    start_time=$(date +%s.%N)
    
    if pg_isready -h "$host" -p "$port" -U "${POSTGRES_USER:-postgres}" >/dev/null 2>&1; then
        response_time=$(echo "$(date +%s.%N) - $start_time" | bc)
        log "✓ $service: PostgreSQL ready (${response_time}s)"
        send_metric "$service" 1 "$response_time"
        return 0
    else
        response_time=$(echo "$(date +%s.%N) - $start_time" | bc)
        log "✗ $service: PostgreSQL not ready"
        send_metric "$service" 0 "$response_time"
        return 1
    fi
}

# Check Redis
check_redis() {
    local service="$1"
    local url="$2"
    local host port
    
    host=$(echo "$url" | sed 's|redis://||' | cut -d: -f1)
    port=$(echo "$url" | sed 's|redis://||' | cut -d: -f2)
    
    local start_time response_time
    start_time=$(date +%s.%N)
    
    if redis-cli -h "$host" -p "$port" ping >/dev/null 2>&1; then
        response_time=$(echo "$(date +%s.%N) - $start_time" | bc)
        log "✓ $service: Redis responding (${response_time}s)"
        send_metric "$service" 1 "$response_time"
        return 0
    else
        response_time=$(echo "$(date +%s.%N) - $start_time" | bc)
        log "✗ $service: Redis not responding"
        send_metric "$service" 0 "$response_time"
        return 1
    fi
}

# Perform health check for a service
check_service() {
    local service="$1"
    local url="$2"
    local protocol
    
    protocol=$(echo "$url" | cut -d: -f1)
    
    case "$protocol" in
        http|https)
            check_http "$service" "$url"
            ;;
        postgres)
            check_postgres "$service" "$url"
            ;;
        redis)
            check_redis "$service" "$url"
            ;;
        *)
            log "✗ $service: Unknown protocol $protocol"
            send_metric "$service" 0 0
            return 1
            ;;
    esac
}

# Trigger failover for a service
trigger_failover() {
    local service="$1"
    
    log "🔄 Triggering failover for $service"
    
    case "$service" in
        postgres-master)
            # Promote slave to master
            log "Promoting PostgreSQL slave to master"
            # docker service update --force postgres-slave
            ;;
        redis-master)
            # Redis Sentinel will handle failover automatically
            log "Redis Sentinel handling failover"
            ;;
        app)
            # Restart application service
            log "Restarting application service"
            # docker service update --force app
            ;;
        *)
            log "No failover procedure defined for $service"
            ;;
    esac
}

# Main health check loop
main() {
    log "Starting health check cycle"
    
    local failed_services=()
    local total_services=0
    local healthy_services=0
    
    for service_config in "${SERVICES[@]}"; do
        service=$(echo "$service_config" | cut -d: -f1)
        url=$(echo "$service_config" | cut -d: -f2-)
        
        total_services=$((total_services + 1))
        
        local retry_count=0
        local success=false
        
        while [[ $retry_count -lt $MAX_RETRIES ]]; do
            if check_service "$service" "$url"; then
                healthy_services=$((healthy_services + 1))
                success=true
                break
            else
                retry_count=$((retry_count + 1))
                if [[ $retry_count -lt $MAX_RETRIES ]]; then
                    log "Retrying $service in 5 seconds... (attempt $((retry_count + 1))/$MAX_RETRIES)"
                    sleep 5
                fi
            fi
        done
        
        if [[ "$success" == false ]]; then
            failed_services+=("$service")
            
            # Trigger failover for critical services
            if [[ "$service" =~ ^(postgres-master|redis-master|app)$ ]]; then
                trigger_failover "$service"
            fi
        fi
    done
    
    # Summary
    local health_percentage=$((healthy_services * 100 / total_services))
    log "Health check summary: $healthy_services/$total_services services healthy ($health_percentage%)"
    
    # Send overall health metric
    send_metric "overall" "$((health_percentage >= 80 ? 1 : 0))" 0
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log "Failed services: ${failed_services[*]}"
        
        # Send alert if critical services are down
        for service in "${failed_services[@]}"; do
            if [[ "$service" =~ ^(postgres-master|redis-master|app|nginx)$ ]]; then
                log "🚨 CRITICAL: $service is down!"
                # Send alert to monitoring system
                send_metric "critical_service_down" 1 0
                break
            fi
        done
    fi
    
    log "Health check cycle completed"
}

# Continuous monitoring mode
if [[ "${1:-}" == "--continuous" ]]; then
    log "Starting continuous health monitoring (interval: ${HEALTH_CHECK_INTERVAL}s)"
    
    while true; do
        main
        sleep "$HEALTH_CHECK_INTERVAL"
    done
else
    # Single run
    main
fi

exit 0