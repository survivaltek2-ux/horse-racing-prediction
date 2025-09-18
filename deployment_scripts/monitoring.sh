#!/bin/bash

# =============================================================================
# Monitoring and Health Check Script
# Horse Racing Prediction Application
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/monitoring"

# Create log directory
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default configuration
ENVIRONMENT="production"
ACTION="health-check"
ALERT_EMAIL=""
SLACK_WEBHOOK=""
CONTINUOUS_MODE=false
CHECK_INTERVAL=60

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
# Health Check Functions
# =============================================================================

check_application_health() {
    log "Checking application health..."
    
    local app_url="http://localhost:8000"
    local health_endpoint="$app_url/health"
    local status_code
    
    # Check if health endpoint exists and responds
    if status_code=$(curl -s -o /dev/null -w "%{http_code}" "$health_endpoint" 2>/dev/null); then
        if [[ "$status_code" == "200" ]]; then
            log "✓ Application health check: PASSED (HTTP $status_code)"
            return 0
        else
            error "✗ Application health check: FAILED (HTTP $status_code)"
            return 1
        fi
    else
        error "✗ Application health check: FAILED (Connection error)"
        return 1
    fi
}

check_database_health() {
    log "Checking database health..."
    
    if docker-compose exec -T postgres pg_isready -U "${POSTGRES_USER:-app_user}" &> /dev/null; then
        log "✓ Database health check: PASSED"
        
        # Check database connectivity
        if docker-compose exec -T postgres psql -U "${POSTGRES_USER:-app_user}" -d "${POSTGRES_DB:-horse_racing_db}" -c "SELECT 1;" &> /dev/null; then
            log "✓ Database connectivity: PASSED"
            return 0
        else
            error "✗ Database connectivity: FAILED"
            return 1
        fi
    else
        error "✗ Database health check: FAILED"
        return 1
    fi
}

check_redis_health() {
    log "Checking Redis health..."
    
    if docker-compose exec -T redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
        log "✓ Redis health check: PASSED"
        return 0
    else
        error "✗ Redis health check: FAILED"
        return 1
    fi
}

check_docker_services() {
    log "Checking Docker services..."
    
    local services=("app" "postgres" "redis" "nginx")
    local failed_services=()
    
    for service in "${services[@]}"; do
        if docker-compose ps "$service" 2>/dev/null | grep -q "Up"; then
            log "✓ Service $service: RUNNING"
        else
            error "✗ Service $service: NOT RUNNING"
            failed_services+=("$service")
        fi
    done
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log "✓ All Docker services: RUNNING"
        return 0
    else
        error "✗ Failed services: ${failed_services[*]}"
        return 1
    fi
}

check_disk_space() {
    log "Checking disk space..."
    
    local threshold=90
    local usage=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [[ $usage -lt $threshold ]]; then
        log "✓ Disk space: ${usage}% used (OK)"
        return 0
    else
        warn "⚠ Disk space: ${usage}% used (WARNING: Above ${threshold}%)"
        return 1
    fi
}

check_memory_usage() {
    log "Checking memory usage..."
    
    local threshold=90
    local usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    
    if [[ $usage -lt $threshold ]]; then
        log "✓ Memory usage: ${usage}% used (OK)"
        return 0
    else
        warn "⚠ Memory usage: ${usage}% used (WARNING: Above ${threshold}%)"
        return 1
    fi
}

check_cpu_usage() {
    log "Checking CPU usage..."
    
    local threshold=90
    local usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    
    if [[ $(echo "$usage < $threshold" | bc -l) -eq 1 ]]; then
        log "✓ CPU usage: ${usage}% used (OK)"
        return 0
    else
        warn "⚠ CPU usage: ${usage}% used (WARNING: Above ${threshold}%)"
        return 1
    fi
}

check_ssl_certificates() {
    log "Checking SSL certificates..."
    
    local cert_file="$PROJECT_ROOT/ssl/fullchain.pem"
    
    if [[ ! -f "$cert_file" ]]; then
        info "ℹ SSL certificate not found (SSL may not be configured)"
        return 0
    fi
    
    # Check certificate expiration
    local expiry_date=$(openssl x509 -enddate -noout -in "$cert_file" | cut -d= -f2)
    local expiry_epoch=$(date -d "$expiry_date" +%s)
    local current_epoch=$(date +%s)
    local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
    
    if [[ $days_until_expiry -gt 30 ]]; then
        log "✓ SSL certificate: Valid for $days_until_expiry more days"
        return 0
    elif [[ $days_until_expiry -gt 0 ]]; then
        warn "⚠ SSL certificate: Expires in $days_until_expiry days"
        return 1
    else
        error "✗ SSL certificate: EXPIRED"
        return 1
    fi
}

# =============================================================================
# Performance Monitoring
# =============================================================================

monitor_application_performance() {
    log "Monitoring application performance..."
    
    local app_url="http://localhost:8000"
    local response_time
    
    # Measure response time
    response_time=$(curl -o /dev/null -s -w "%{time_total}" "$app_url" 2>/dev/null || echo "0")
    
    if [[ $(echo "$response_time > 0" | bc -l) -eq 1 ]]; then
        if [[ $(echo "$response_time < 2.0" | bc -l) -eq 1 ]]; then
            log "✓ Application response time: ${response_time}s (Good)"
        elif [[ $(echo "$response_time < 5.0" | bc -l) -eq 1 ]]; then
            warn "⚠ Application response time: ${response_time}s (Slow)"
        else
            error "✗ Application response time: ${response_time}s (Very slow)"
        fi
    else
        error "✗ Application response time: Unable to measure"
    fi
}

monitor_database_performance() {
    log "Monitoring database performance..."
    
    # Check active connections
    local active_connections=$(docker-compose exec -T postgres psql -U "${POSTGRES_USER:-app_user}" -d "${POSTGRES_DB:-horse_racing_db}" -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | tr -d ' \n' || echo "0")
    
    log "Database active connections: $active_connections"
    
    # Check slow queries (if pg_stat_statements is enabled)
    local slow_queries=$(docker-compose exec -T postgres psql -U "${POSTGRES_USER:-app_user}" -d "${POSTGRES_DB:-horse_racing_db}" -t -c "SELECT count(*) FROM pg_stat_statements WHERE mean_time > 1000;" 2>/dev/null | tr -d ' \n' || echo "0")
    
    if [[ $slow_queries -gt 0 ]]; then
        warn "⚠ Found $slow_queries slow queries (>1s average)"
    else
        log "✓ No slow queries detected"
    fi
}

monitor_redis_performance() {
    log "Monitoring Redis performance..."
    
    # Get Redis info
    local redis_info=$(docker-compose exec -T redis redis-cli info stats 2>/dev/null || echo "")
    
    if [[ -n "$redis_info" ]]; then
        local total_commands=$(echo "$redis_info" | grep "total_commands_processed" | cut -d: -f2 | tr -d '\r')
        local keyspace_hits=$(echo "$redis_info" | grep "keyspace_hits" | cut -d: -f2 | tr -d '\r')
        local keyspace_misses=$(echo "$redis_info" | grep "keyspace_misses" | cut -d: -f2 | tr -d '\r')
        
        if [[ -n "$keyspace_hits" && -n "$keyspace_misses" && $((keyspace_hits + keyspace_misses)) -gt 0 ]]; then
            local hit_rate=$(( keyspace_hits * 100 / (keyspace_hits + keyspace_misses) ))
            log "Redis cache hit rate: ${hit_rate}%"
            
            if [[ $hit_rate -lt 80 ]]; then
                warn "⚠ Redis cache hit rate is low: ${hit_rate}%"
            fi
        fi
        
        log "Redis total commands processed: ${total_commands:-0}"
    else
        warn "⚠ Unable to get Redis performance metrics"
    fi
}

# =============================================================================
# Log Analysis
# =============================================================================

analyze_application_logs() {
    log "Analyzing application logs..."
    
    local log_file="$PROJECT_ROOT/logs/app.log"
    
    if [[ ! -f "$log_file" ]]; then
        info "Application log file not found: $log_file"
        return 0
    fi
    
    # Check for errors in the last hour
    local error_count=$(grep -c "ERROR" "$log_file" 2>/dev/null || echo "0")
    local warning_count=$(grep -c "WARNING" "$log_file" 2>/dev/null || echo "0")
    
    log "Recent log analysis:"
    log "  Errors: $error_count"
    log "  Warnings: $warning_count"
    
    if [[ $error_count -gt 10 ]]; then
        warn "⚠ High number of errors detected: $error_count"
    fi
    
    # Show recent errors
    if [[ $error_count -gt 0 ]]; then
        log "Recent errors:"
        tail -n 100 "$log_file" | grep "ERROR" | tail -n 5 || true
    fi
}

# =============================================================================
# Alerting Functions
# =============================================================================

send_email_alert() {
    local subject="$1"
    local message="$2"
    
    if [[ -z "$ALERT_EMAIL" ]]; then
        return 0
    fi
    
    log "Sending email alert to: $ALERT_EMAIL"
    
    # Use mail command if available
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "$subject" "$ALERT_EMAIL"
    elif command -v sendmail &> /dev/null; then
        {
            echo "To: $ALERT_EMAIL"
            echo "Subject: $subject"
            echo ""
            echo "$message"
        } | sendmail "$ALERT_EMAIL"
    else
        warn "No mail command available for sending alerts"
    fi
}

send_slack_alert() {
    local message="$1"
    
    if [[ -z "$SLACK_WEBHOOK" ]]; then
        return 0
    fi
    
    log "Sending Slack alert"
    
    local payload=$(cat << EOF
{
    "text": "🚨 Horse Racing Prediction Alert",
    "attachments": [
        {
            "color": "danger",
            "fields": [
                {
                    "title": "Environment",
                    "value": "$ENVIRONMENT",
                    "short": true
                },
                {
                    "title": "Timestamp",
                    "value": "$(date)",
                    "short": true
                },
                {
                    "title": "Message",
                    "value": "$message",
                    "short": false
                }
            ]
        }
    ]
}
EOF
)
    
    curl -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK" &> /dev/null || warn "Failed to send Slack alert"
}

# =============================================================================
# Comprehensive Health Check
# =============================================================================

run_comprehensive_health_check() {
    log "Running comprehensive health check..."
    
    local failed_checks=0
    local total_checks=0
    
    # Application checks
    ((total_checks++))
    check_application_health || ((failed_checks++))
    
    ((total_checks++))
    check_database_health || ((failed_checks++))
    
    ((total_checks++))
    check_redis_health || ((failed_checks++))
    
    ((total_checks++))
    check_docker_services || ((failed_checks++))
    
    # System checks
    ((total_checks++))
    check_disk_space || ((failed_checks++))
    
    ((total_checks++))
    check_memory_usage || ((failed_checks++))
    
    ((total_checks++))
    check_cpu_usage || ((failed_checks++))
    
    ((total_checks++))
    check_ssl_certificates || ((failed_checks++))
    
    # Performance monitoring
    monitor_application_performance
    monitor_database_performance
    monitor_redis_performance
    
    # Log analysis
    analyze_application_logs
    
    # Summary
    local success_rate=$(( (total_checks - failed_checks) * 100 / total_checks ))
    
    log "Health check summary:"
    log "  Total checks: $total_checks"
    log "  Passed: $((total_checks - failed_checks))"
    log "  Failed: $failed_checks"
    log "  Success rate: ${success_rate}%"
    
    if [[ $failed_checks -eq 0 ]]; then
        log "✓ All health checks passed!"
        return 0
    else
        error "✗ $failed_checks health checks failed"
        
        # Send alerts if configured
        local alert_message="Health check failed: $failed_checks out of $total_checks checks failed on $ENVIRONMENT environment"
        send_email_alert "Health Check Alert - $ENVIRONMENT" "$alert_message"
        send_slack_alert "$alert_message"
        
        return 1
    fi
}

# =============================================================================
# Continuous Monitoring
# =============================================================================

run_continuous_monitoring() {
    log "Starting continuous monitoring (interval: ${CHECK_INTERVAL}s)"
    log "Press Ctrl+C to stop"
    
    while true; do
        echo ""
        log "=== Health Check Cycle $(date) ==="
        
        run_comprehensive_health_check
        
        log "Next check in ${CHECK_INTERVAL} seconds..."
        sleep "$CHECK_INTERVAL"
    done
}

# =============================================================================
# Setup Monitoring Services
# =============================================================================

setup_prometheus_monitoring() {
    log "Setting up Prometheus monitoring..."
    
    # Create Prometheus configuration
    local prometheus_config="$PROJECT_ROOT/monitoring/prometheus.yml"
    mkdir -p "$(dirname "$prometheus_config")"
    
    cat > "$prometheus_config" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'horse-racing-app'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

    # Create alert rules
    local alert_rules="$PROJECT_ROOT/monitoring/alert_rules.yml"
    
    cat > "$alert_rules" << 'EOF'
groups:
  - name: horse_racing_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(flask_http_request_exceptions_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(flask_http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"
          description: "PostgreSQL database is not responding"

      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"
          description: "Redis cache is not responding"

      - alert: HighDiskUsage
        expr: (node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High disk usage"
          description: "Disk usage is {{ $value | humanizePercentage }}"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"
EOF

    log "Prometheus monitoring configuration created"
}

setup_grafana_dashboards() {
    log "Setting up Grafana dashboards..."
    
    local dashboard_dir="$PROJECT_ROOT/monitoring/grafana/dashboards"
    mkdir -p "$dashboard_dir"
    
    # Create main application dashboard
    cat > "$dashboard_dir/horse_racing_dashboard.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Horse Racing Prediction Dashboard",
    "tags": ["horse-racing"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(flask_http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(flask_http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(flask_http_request_exceptions_total[5m])",
            "legendFormat": "Errors/sec"
          }
        ]
      },
      {
        "id": 4,
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends",
            "legendFormat": "Active connections"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    log "Grafana dashboard configuration created"
}

# =============================================================================
# Usage and Help
# =============================================================================

show_usage() {
    cat << EOF
Monitoring and Health Check Script

Usage: $0 [environment] [action] [options]

Environments:
    development     Monitor development environment
    staging         Monitor staging environment
    production      Monitor production environment (default)

Actions:
    health-check    Run comprehensive health check (default)
    monitor         Start continuous monitoring
    setup-prometheus Setup Prometheus monitoring
    setup-grafana   Setup Grafana dashboards
    performance     Monitor performance metrics only
    logs            Analyze application logs

Options:
    --continuous            Run in continuous mode
    --interval SECONDS      Check interval for continuous mode (default: 60)
    --alert-email EMAIL     Email address for alerts
    --slack-webhook URL     Slack webhook URL for alerts
    --help                  Show this help message

Examples:
    $0 production health-check
    $0 staging monitor --continuous --interval 30
    $0 production health-check --alert-email admin@example.com
    $0 development setup-prometheus

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
            health-check|monitor|setup-prometheus|setup-grafana|performance|logs)
                ACTION="$1"
                shift
                ;;
            --continuous)
                CONTINUOUS_MODE=true
                shift
                ;;
            --interval)
                CHECK_INTERVAL="$2"
                shift 2
                ;;
            --alert-email)
                ALERT_EMAIL="$2"
                shift 2
                ;;
            --slack-webhook)
                SLACK_WEBHOOK="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use --help for usage information."
                ;;
        esac
    done
    
    log "Monitoring for environment: $ENVIRONMENT"
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
        health-check)
            if [[ "$CONTINUOUS_MODE" == "true" ]]; then
                run_continuous_monitoring
            else
                run_comprehensive_health_check
            fi
            ;;
        monitor)
            run_continuous_monitoring
            ;;
        setup-prometheus)
            setup_prometheus_monitoring
            ;;
        setup-grafana)
            setup_grafana_dashboards
            ;;
        performance)
            monitor_application_performance
            monitor_database_performance
            monitor_redis_performance
            ;;
        logs)
            analyze_application_logs
            ;;
        *)
            error "Unknown action: $ACTION"
            ;;
    esac
    
    log "Monitoring script completed!"
}

# Trap Ctrl+C for graceful shutdown
trap 'log "Monitoring stopped by user"; exit 0' INT

# Run main function with all arguments
main "$@"