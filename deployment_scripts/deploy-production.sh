#!/bin/bash
# Production Deployment Script for Horse Racing Prediction App

set -euo pipefail

# Configuration
STACK_NAME="horse-racing-prediction"
COMPOSE_FILE="docker-swarm-stack.yml"
BACKUP_ENABLED=true
HEALTH_CHECK_TIMEOUT=300
LOG_FILE="$(pwd)/logs/deployment/deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    log "${GREEN}✅ $1${NC}"
}

log_warning() {
    log "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    log "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running"
        exit 1
    fi
    
    # Check if Docker Swarm is initialized
    if ! docker node ls >/dev/null 2>&1; then
        log_error "Docker Swarm is not initialized"
        log_info "Run: docker swarm init"
        exit 1
    fi
    
    # Check if compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    # Check if required images are available
    local required_images=(
        "postgres:13"
        "redis:7-alpine"
        "nginx:alpine"
        "elasticsearch:7.17.0"
        "kibana:7.17.0"
        "logstash:7.17.0"
        "grafana/grafana:latest"
        "prom/prometheus:latest"
    )
    
    for image in "${required_images[@]}"; do
        if ! docker image inspect "$image" >/dev/null 2>&1; then
            log_warning "Image not found locally: $image"
            log_info "Pulling image: $image"
            docker pull "$image"
        fi
    done
    
    log_success "Prerequisites check completed"
}

# Create backup before deployment
create_backup() {
    if [[ "$BACKUP_ENABLED" == true ]]; then
        log_info "Creating backup before deployment..."
        
        # Create backup directory
        local backup_dir="/backups/pre-deployment/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        
        # Backup PostgreSQL if running
        if docker service ls --filter name="${STACK_NAME}_postgres-master" --format "{{.Name}}" | grep -q "${STACK_NAME}_postgres-master"; then
            log_info "Backing up PostgreSQL..."
            docker exec -t "$(docker ps -q --filter name="${STACK_NAME}_postgres-master")" \
                pg_dump -U postgres horse_racing_prediction > "$backup_dir/postgres_backup.sql" || true
        fi
        
        # Backup application data
        if [[ -d "/data/app" ]]; then
            log_info "Backing up application data..."
            tar -czf "$backup_dir/app_data.tar.gz" -C /data app/ || true
        fi
        
        log_success "Backup created: $backup_dir"
    fi
}

# Build application image
build_application() {
    log_info "Building application image..."
    
    # Build production image
    docker build -f Dockerfile.production -t "${STACK_NAME}-app:latest" .
    
    # Tag with timestamp for rollback capability
    local timestamp=$(date +%Y%m%d_%H%M%S)
    docker tag "${STACK_NAME}-app:latest" "${STACK_NAME}-app:$timestamp"
    
    log_success "Application image built: ${STACK_NAME}-app:latest"
}

# Deploy stack
deploy_stack() {
    log_info "Deploying Docker Swarm stack..."
    
    # Deploy the stack
    docker stack deploy -c "$COMPOSE_FILE" "$STACK_NAME"
    
    log_success "Stack deployment initiated"
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to become healthy..."
    
    local timeout=$HEALTH_CHECK_TIMEOUT
    local elapsed=0
    local check_interval=10
    
    while [[ $elapsed -lt $timeout ]]; do
        local unhealthy_services=()
        
        # Check each service
        while IFS= read -r service; do
            local replicas_info
            replicas_info=$(docker service ls --filter name="$service" --format "{{.Replicas}}")
            
            if [[ "$replicas_info" =~ ^([0-9]+)/([0-9]+) ]]; then
                local running="${BASH_REMATCH[1]}"
                local desired="${BASH_REMATCH[2]}"
                
                if [[ "$running" != "$desired" ]]; then
                    unhealthy_services+=("$service")
                fi
            fi
        done < <(docker service ls --filter name="${STACK_NAME}_" --format "{{.Name}}")
        
        if [[ ${#unhealthy_services[@]} -eq 0 ]]; then
            log_success "All services are healthy"
            return 0
        fi
        
        log_info "Waiting for services: ${unhealthy_services[*]} (${elapsed}s/${timeout}s)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done
    
    log_error "Timeout waiting for services to become healthy"
    return 1
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Wait a bit for services to fully start
    sleep 30
    
    # Run the health check script
    if [[ -f "scripts/health-checks/health-check.sh" ]]; then
        bash scripts/health-checks/health-check.sh
        local health_status=$?
        
        if [[ $health_status -eq 0 ]]; then
            log_success "Health checks passed"
            return 0
        else
            log_error "Health checks failed"
            return 1
        fi
    else
        log_warning "Health check script not found, skipping..."
        return 0
    fi
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    # Get previous version
    local previous_image
    previous_image=$(docker images "${STACK_NAME}-app" --format "{{.Tag}}" | grep -E '^[0-9]{8}_[0-9]{6}$' | sort -r | head -n 2 | tail -n 1)
    
    if [[ -n "$previous_image" ]]; then
        log_info "Rolling back to: ${STACK_NAME}-app:$previous_image"
        
        # Update service with previous image
        docker service update --image "${STACK_NAME}-app:$previous_image" "${STACK_NAME}_app"
        
        # Wait for rollback to complete
        sleep 60
        
        log_success "Rollback completed"
    else
        log_error "No previous version found for rollback"
    fi
}

# Cleanup old images
cleanup_old_images() {
    log_info "Cleaning up old images..."
    
    # Keep last 5 versions
    local old_images
    old_images=$(docker images "${STACK_NAME}-app" --format "{{.Tag}}" | grep -E '^[0-9]{8}_[0-9]{6}$' | sort -r | tail -n +6)
    
    for tag in $old_images; do
        log_info "Removing old image: ${STACK_NAME}-app:$tag"
        docker rmi "${STACK_NAME}-app:$tag" || true
    done
    
    # Remove dangling images
    docker image prune -f
    
    log_success "Image cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting production deployment of $STACK_NAME"
    
    # Check prerequisites
    check_prerequisites
    
    # Create backup
    create_backup
    
    # Build application
    build_application
    
    # Deploy stack
    deploy_stack
    
    # Wait for services
    if ! wait_for_services; then
        log_error "Services failed to become healthy, initiating rollback"
        rollback_deployment
        exit 1
    fi
    
    # Run health checks
    if ! run_health_checks; then
        log_error "Health checks failed, initiating rollback"
        rollback_deployment
        exit 1
    fi
    
    # Cleanup
    cleanup_old_images
    
    log_success "Production deployment completed successfully!"
    
    # Display service information
    log_info "Service status:"
    docker service ls --filter name="${STACK_NAME}_"
    
    log_info "Access URLs:"
    log_info "  Application: https://localhost"
    log_info "  Grafana: https://localhost:3000"
    log_info "  Kibana: https://localhost:5601"
}

# Handle script arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    rollback)
        rollback_deployment
        ;;
    health-check)
        run_health_checks
        ;;
    cleanup)
        cleanup_old_images
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health-check|cleanup}"
        exit 1
        ;;
esac

exit 0