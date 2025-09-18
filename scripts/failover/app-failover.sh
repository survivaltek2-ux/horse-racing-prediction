#!/bin/bash
set -e

# Application Failover Script
# This script handles application instance failover and load balancer updates

# Configuration
APP_INSTANCES=("hrp-app1" "hrp-app2" "hrp-app3" "hrp-app4" "hrp-app5" "hrp-app6")
HEALTH_CHECK_URL="/health"
HEALTH_CHECK_TIMEOUT=10
MIN_HEALTHY_INSTANCES=2
HAPROXY_STATS_URL="http://haproxy:8404/stats"
HAPROXY_ADMIN_URL="http://haproxy:8404/admin"

# Logging
LOG_FILE="/var/log/app-failover.log"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if an application instance is healthy
check_app_health() {
    local app_name="$1"
    local app_port="8000"
    
    # Get container IP
    local container_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$app_name" 2>/dev/null)
    
    if [ -z "$container_ip" ]; then
        log "Container $app_name not found or not running"
        return 1
    fi
    
    # Check health endpoint
    if curl -f -m "$HEALTH_CHECK_TIMEOUT" "http://$container_ip:$app_port$HEALTH_CHECK_URL" >/dev/null 2>&1; then
        return 0  # Healthy
    else
        return 1  # Unhealthy
    fi
}

# Get list of healthy instances
get_healthy_instances() {
    local healthy_instances=()
    
    for instance in "${APP_INSTANCES[@]}"; do
        if check_app_health "$instance"; then
            healthy_instances+=("$instance")
        fi
    done
    
    echo "${healthy_instances[@]}"
}

# Get list of unhealthy instances
get_unhealthy_instances() {
    local unhealthy_instances=()
    
    for instance in "${APP_INSTANCES[@]}"; do
        if ! check_app_health "$instance"; then
            unhealthy_instances+=("$instance")
        fi
    done
    
    echo "${unhealthy_instances[@]}"
}

# Restart an application instance
restart_app_instance() {
    local app_name="$1"
    
    log "Restarting application instance: $app_name"
    
    if docker restart "$app_name" >/dev/null 2>&1; then
        log "Successfully restarted $app_name"
        
        # Wait for instance to come back online
        local timeout=60
        local count=0
        
        while [ $count -lt $timeout ]; do
            if check_app_health "$app_name"; then
                log "$app_name is healthy after restart"
                return 0
            fi
            sleep 2
            count=$((count + 2))
        done
        
        log "WARNING: $app_name did not become healthy after restart"
        return 1
    else
        log "ERROR: Failed to restart $app_name"
        return 1
    fi
}

# Scale up application instances
scale_up_instances() {
    local target_count="$1"
    local current_count=$(docker ps --filter "name=hrp-app" --format "table {{.Names}}" | grep -c "hrp-app" || echo "0")
    
    if [ "$current_count" -ge "$target_count" ]; then
        log "Already have $current_count instances, no scale up needed"
        return 0
    fi
    
    local instances_to_add=$((target_count - current_count))
    log "Scaling up: adding $instances_to_add instances"
    
    for ((i=1; i<=instances_to_add; i++)); do
        local new_instance_name="hrp-app$((current_count + i))"
        
        log "Starting new instance: $new_instance_name"
        
        # Start new container (this would need to be adapted based on your Docker setup)
        if docker run -d --name "$new_instance_name" \
            --network hrp-network \
            -e POSTGRES_HOST=postgres-primary \
            -e REDIS_HOST=redis-primary \
            hrp-app:latest >/dev/null 2>&1; then
            
            log "Successfully started $new_instance_name"
            
            # Wait for instance to become healthy
            sleep 10
            if check_app_health "$new_instance_name"; then
                log "$new_instance_name is healthy"
            else
                log "WARNING: $new_instance_name is not healthy"
            fi
        else
            log "ERROR: Failed to start $new_instance_name"
        fi
    done
}

# Update HAProxy configuration
update_haproxy_config() {
    local action="$1"  # enable/disable
    local server_name="$2"
    
    log "Updating HAProxy: $action server $server_name"
    
    # This would typically involve HAProxy admin socket or API
    # For now, we'll log the action
    if [ "$action" = "disable" ]; then
        log "Would disable $server_name in HAProxy"
        # curl -X POST "$HAPROXY_ADMIN_URL?action=disable&server=$server_name" || true
    elif [ "$action" = "enable" ]; then
        log "Would enable $server_name in HAProxy"
        # curl -X POST "$HAPROXY_ADMIN_URL?action=enable&server=$server_name" || true
    fi
}

# Handle unhealthy instances
handle_unhealthy_instances() {
    local unhealthy_instances=($(get_unhealthy_instances))
    
    if [ ${#unhealthy_instances[@]} -eq 0 ]; then
        log "All application instances are healthy"
        return 0
    fi
    
    log "Found ${#unhealthy_instances[@]} unhealthy instances: ${unhealthy_instances[*]}"
    
    # Check if we have enough healthy instances
    local healthy_instances=($(get_healthy_instances))
    local healthy_count=${#healthy_instances[@]}
    
    if [ "$healthy_count" -lt "$MIN_HEALTHY_INSTANCES" ]; then
        log "CRITICAL: Only $healthy_count healthy instances (minimum: $MIN_HEALTHY_INSTANCES)"
        send_notification "CRITICAL: Application has only $healthy_count healthy instances!"
        
        # Try to restart unhealthy instances immediately
        for instance in "${unhealthy_instances[@]}"; do
            restart_app_instance "$instance" &
        done
        wait
        
        # Scale up if still not enough healthy instances
        scale_up_instances $((MIN_HEALTHY_INSTANCES + 1))
    else
        log "Have $healthy_count healthy instances, handling unhealthy ones"
        
        # Restart unhealthy instances one by one
        for instance in "${unhealthy_instances[@]}"; do
            # Disable in load balancer first
            update_haproxy_config "disable" "$instance"
            
            # Restart the instance
            restart_app_instance "$instance"
            
            # Re-enable in load balancer if healthy
            if check_app_health "$instance"; then
                update_haproxy_config "enable" "$instance"
            fi
        done
    fi
}

# Check overall application health
check_overall_health() {
    local healthy_instances=($(get_healthy_instances))
    local unhealthy_instances=($(get_unhealthy_instances))
    local healthy_count=${#healthy_instances[@]}
    local unhealthy_count=${#unhealthy_instances[@]}
    local total_count=$((healthy_count + unhealthy_count))
    
    log "Application health status:"
    log "  Total instances: $total_count"
    log "  Healthy instances: $healthy_count"
    log "  Unhealthy instances: $unhealthy_count"
    
    if [ "$healthy_count" -ge "$MIN_HEALTHY_INSTANCES" ]; then
        log "Application cluster is healthy"
        return 0
    else
        log "Application cluster is unhealthy"
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
        echo "$message" | mail -s "Application Failover Alert" "$ALERT_EMAIL" || true
    fi
}

# Main function
main() {
    local action="${1:-check}"
    
    case "$action" in
        "check")
            log "Checking application health..."
            check_overall_health
            ;;
        "failover")
            log "Handling application failover..."
            handle_unhealthy_instances
            ;;
        "scale-up")
            local target="${2:-6}"
            log "Scaling up to $target instances..."
            scale_up_instances "$target"
            ;;
        "restart")
            local instance="${2:-all}"
            if [ "$instance" = "all" ]; then
                log "Restarting all application instances..."
                for app in "${APP_INSTANCES[@]}"; do
                    restart_app_instance "$app" &
                done
                wait
            else
                log "Restarting instance: $instance"
                restart_app_instance "$instance"
            fi
            ;;
        *)
            echo "Usage: $0 {check|failover|scale-up [count]|restart [instance|all]}"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"