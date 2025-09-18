#!/bin/bash

# =============================================================================
# Horse Racing Prediction - Automated Deployment Script
# =============================================================================
# This script provides full automation for deploying the Horse Racing 
# Prediction application to any server environment.
#
# Usage: ./auto-deploy.sh [environment] [action] [options]
# Example: ./auto-deploy.sh production deploy --ssl --monitoring
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/deployment"
CONFIG_DIR="$PROJECT_ROOT/config"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging setup
LOG_FILE="$LOG_DIR/auto-deploy-$(date +%Y%m%d-%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
ENVIRONMENT="production"
ACTION="deploy"
ENABLE_SSL=false
ENABLE_MONITORING=false
ENABLE_BACKUP=false
FORCE_REBUILD=false
SKIP_TESTS=false
DOMAIN=""
EMAIL=""
DB_BACKUP=true

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
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        error "Required command '$1' not found. Please install it first."
    fi
}

check_file() {
    if [[ ! -f "$1" ]]; then
        error "Required file '$1' not found."
    fi
}

check_directory() {
    if [[ ! -d "$1" ]]; then
        error "Required directory '$1' not found."
    fi
}

# =============================================================================
# System Requirements Check
# =============================================================================

check_system_requirements() {
    log "Checking system requirements..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "git" "curl" "openssl")
    for cmd in "${required_commands[@]}"; do
        check_command "$cmd"
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
    fi
    
    # Check available disk space (minimum 5GB)
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 5242880 ]]; then
        warn "Low disk space detected. Recommended minimum: 5GB"
    fi
    
    # Check memory (minimum 2GB)
    local available_memory=$(free -m | awk 'NR==2{print $7}')
    if [[ $available_memory -lt 2048 ]]; then
        warn "Low memory detected. Recommended minimum: 2GB"
    fi
    
    log "System requirements check completed."
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_environment() {
    log "Setting up environment for: $ENVIRONMENT"
    
    # Source environment configuration
    local env_file="$PROJECT_ROOT/.env.$ENVIRONMENT"
    if [[ -f "$env_file" ]]; then
        set -a
        source "$env_file"
        set +a
        log "Loaded environment configuration from $env_file"
    else
        warn "Environment file $env_file not found. Using defaults."
    fi
    
    # Create necessary directories
    local directories=(
        "$PROJECT_ROOT/data"
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/backups"
        "$PROJECT_ROOT/ssl"
        "$PROJECT_ROOT/monitoring"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log "Created directory: $dir"
    done
    
    # Set proper permissions
    chmod 755 "$PROJECT_ROOT/data"
    chmod 755 "$PROJECT_ROOT/logs"
    chmod 700 "$PROJECT_ROOT/ssl"
    
    log "Environment setup completed."
}

# =============================================================================
# Database Setup and Migration
# =============================================================================

setup_database() {
    log "Setting up database..."
    
    # Check if database service is running
    if docker-compose ps | grep -q "postgres.*Up"; then
        log "Database service is already running."
    else
        log "Starting database service..."
        docker-compose up -d postgres
        
        # Wait for database to be ready
        local max_attempts=30
        local attempt=1
        while [[ $attempt -le $max_attempts ]]; do
            if docker-compose exec -T postgres pg_isready -U "${POSTGRES_USER:-app_user}" &> /dev/null; then
                log "Database is ready."
                break
            fi
            info "Waiting for database to be ready... (attempt $attempt/$max_attempts)"
            sleep 2
            ((attempt++))
        done
        
        if [[ $attempt -gt $max_attempts ]]; then
            error "Database failed to start within expected time."
        fi
    fi
    
    # Run database migrations
    log "Running database migrations..."
    if [[ -f "$PROJECT_ROOT/migrate_enhanced_schema.py" ]]; then
        docker-compose exec -T app python migrate_enhanced_schema.py || warn "Migration script failed"
    fi
    
    # Initialize database if needed
    if [[ -f "$PROJECT_ROOT/init_database.py" ]]; then
        docker-compose exec -T app python init_database.py || warn "Database initialization failed"
    fi
    
    log "Database setup completed."
}

# =============================================================================
# SSL/TLS Certificate Setup
# =============================================================================

setup_ssl() {
    if [[ "$ENABLE_SSL" != "true" ]]; then
        return 0
    fi
    
    log "Setting up SSL/TLS certificates..."
    
    if [[ -z "$DOMAIN" ]]; then
        error "Domain name is required for SSL setup. Use --domain option."
    fi
    
    if [[ -z "$EMAIL" ]]; then
        error "Email address is required for SSL setup. Use --email option."
    fi
    
    # Check if certificates already exist
    local cert_dir="$PROJECT_ROOT/ssl"
    if [[ -f "$cert_dir/fullchain.pem" && -f "$cert_dir/privkey.pem" ]]; then
        log "SSL certificates already exist. Checking validity..."
        
        # Check certificate expiration
        local expiry_date=$(openssl x509 -enddate -noout -in "$cert_dir/fullchain.pem" | cut -d= -f2)
        local expiry_epoch=$(date -d "$expiry_date" +%s)
        local current_epoch=$(date +%s)
        local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
        
        if [[ $days_until_expiry -lt 30 ]]; then
            warn "SSL certificate expires in $days_until_expiry days. Renewing..."
            obtain_ssl_certificate
        else
            log "SSL certificate is valid for $days_until_expiry more days."
        fi
    else
        log "Obtaining new SSL certificate..."
        obtain_ssl_certificate
    fi
    
    log "SSL setup completed."
}

obtain_ssl_certificate() {
    # Use Let's Encrypt with certbot
    if ! command -v certbot &> /dev/null; then
        log "Installing certbot..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y certbot
        elif command -v yum &> /dev/null; then
            sudo yum install -y certbot
        else
            error "Cannot install certbot. Please install it manually."
        fi
    fi
    
    # Stop nginx temporarily for certificate generation
    docker-compose stop nginx || true
    
    # Obtain certificate
    sudo certbot certonly --standalone \
        --email "$EMAIL" \
        --agree-tos \
        --no-eff-email \
        -d "$DOMAIN" \
        --cert-path "$PROJECT_ROOT/ssl/fullchain.pem" \
        --key-path "$PROJECT_ROOT/ssl/privkey.pem"
    
    # Set proper permissions
    sudo chown -R $(whoami):$(whoami) "$PROJECT_ROOT/ssl"
    chmod 600 "$PROJECT_ROOT/ssl/privkey.pem"
    chmod 644 "$PROJECT_ROOT/ssl/fullchain.pem"
}

# =============================================================================
# Application Deployment
# =============================================================================

deploy_application() {
    log "Deploying application..."
    
    # Pull latest code if not in development
    if [[ "$ENVIRONMENT" != "development" ]]; then
        log "Pulling latest code from repository..."
        git pull origin main || warn "Failed to pull latest code"
    fi
    
    # Build or rebuild containers
    if [[ "$FORCE_REBUILD" == "true" ]]; then
        log "Force rebuilding containers..."
        docker-compose build --no-cache
    else
        log "Building containers..."
        docker-compose build
    fi
    
    # Run tests if not skipped
    if [[ "$SKIP_TESTS" != "true" ]]; then
        run_tests
    fi
    
    # Create backup before deployment
    if [[ "$DB_BACKUP" == "true" ]]; then
        create_backup
    fi
    
    # Deploy services
    log "Starting services..."
    docker-compose up -d
    
    # Wait for services to be healthy
    wait_for_services
    
    # Run post-deployment tasks
    post_deployment_tasks
    
    log "Application deployment completed."
}

run_tests() {
    log "Running tests..."
    
    # Build test environment
    docker-compose -f docker-compose.yml -f docker-compose.test.yml build test || {
        warn "Test environment not configured. Skipping tests."
        return 0
    }
    
    # Run tests
    docker-compose -f docker-compose.yml -f docker-compose.test.yml run --rm test || {
        error "Tests failed. Deployment aborted."
    }
    
    log "All tests passed."
}

wait_for_services() {
    log "Waiting for services to be healthy..."
    
    local services=("app" "postgres" "redis")
    local max_attempts=60
    
    for service in "${services[@]}"; do
        local attempt=1
        while [[ $attempt -le $max_attempts ]]; do
            if docker-compose ps "$service" | grep -q "Up (healthy)"; then
                log "Service $service is healthy."
                break
            elif docker-compose ps "$service" | grep -q "Up"; then
                info "Service $service is running, waiting for health check... (attempt $attempt/$max_attempts)"
            else
                warn "Service $service is not running."
            fi
            
            sleep 5
            ((attempt++))
        done
        
        if [[ $attempt -gt $max_attempts ]]; then
            error "Service $service failed to become healthy within expected time."
        fi
    done
    
    log "All services are healthy."
}

post_deployment_tasks() {
    log "Running post-deployment tasks..."
    
    # Update database schema if needed
    docker-compose exec -T app python -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('Database schema updated.')
" || warn "Failed to update database schema"
    
    # Clear caches
    docker-compose exec -T redis redis-cli FLUSHALL || warn "Failed to clear Redis cache"
    
    # Generate AI models if needed
    if [[ -f "$PROJECT_ROOT/train_models_direct.py" ]]; then
        log "Training AI models..."
        docker-compose exec -T app python train_models_direct.py || warn "AI model training failed"
    fi
    
    log "Post-deployment tasks completed."
}

# =============================================================================
# Monitoring Setup
# =============================================================================

setup_monitoring() {
    if [[ "$ENABLE_MONITORING" != "true" ]]; then
        return 0
    fi
    
    log "Setting up monitoring..."
    
    # Start monitoring services
    if [[ -f "$PROJECT_ROOT/docker-compose.monitoring.yml" ]]; then
        docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d prometheus grafana
        log "Monitoring services started."
    else
        warn "Monitoring configuration not found. Skipping monitoring setup."
    fi
    
    log "Monitoring setup completed."
}

# =============================================================================
# Backup Functions
# =============================================================================

create_backup() {
    if [[ "$ENABLE_BACKUP" != "true" && "$DB_BACKUP" != "true" ]]; then
        return 0
    fi
    
    log "Creating backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Database backup
    if docker-compose ps postgres | grep -q "Up"; then
        log "Creating database backup..."
        docker-compose exec -T postgres pg_dump -U "${POSTGRES_USER:-app_user}" "${POSTGRES_DB:-horse_racing_db}" > "$backup_dir/database.sql"
        log "Database backup created: $backup_dir/database.sql"
    fi
    
    # Application data backup
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        log "Creating application data backup..."
        tar -czf "$backup_dir/app_data.tar.gz" -C "$PROJECT_ROOT" data/
        log "Application data backup created: $backup_dir/app_data.tar.gz"
    fi
    
    # Configuration backup
    log "Creating configuration backup..."
    tar -czf "$backup_dir/config.tar.gz" -C "$PROJECT_ROOT" config/ .env* || true
    
    log "Backup completed: $backup_dir"
}

# =============================================================================
# Rollback Functions
# =============================================================================

rollback_deployment() {
    log "Rolling back deployment..."
    
    # Find latest backup
    local latest_backup=$(find "$PROJECT_ROOT/backups" -maxdepth 1 -type d -name "20*" | sort -r | head -n1)
    
    if [[ -z "$latest_backup" ]]; then
        error "No backup found for rollback."
    fi
    
    log "Rolling back to backup: $latest_backup"
    
    # Stop services
    docker-compose down
    
    # Restore database
    if [[ -f "$latest_backup/database.sql" ]]; then
        log "Restoring database..."
        docker-compose up -d postgres
        sleep 10
        docker-compose exec -T postgres psql -U "${POSTGRES_USER:-app_user}" -d "${POSTGRES_DB:-horse_racing_db}" < "$latest_backup/database.sql"
    fi
    
    # Restore application data
    if [[ -f "$latest_backup/app_data.tar.gz" ]]; then
        log "Restoring application data..."
        rm -rf "$PROJECT_ROOT/data"
        tar -xzf "$latest_backup/app_data.tar.gz" -C "$PROJECT_ROOT"
    fi
    
    # Restart services
    docker-compose up -d
    
    log "Rollback completed."
}

# =============================================================================
# Health Check Functions
# =============================================================================

health_check() {
    log "Performing health check..."
    
    local app_url="http://localhost:8000"
    if [[ "$ENABLE_SSL" == "true" && -n "$DOMAIN" ]]; then
        app_url="https://$DOMAIN"
    fi
    
    # Check application health
    local health_endpoint="$app_url/health"
    if curl -f -s "$health_endpoint" > /dev/null; then
        log "Application health check: PASSED"
    else
        error "Application health check: FAILED"
    fi
    
    # Check database connectivity
    if docker-compose exec -T postgres pg_isready -U "${POSTGRES_USER:-app_user}" &> /dev/null; then
        log "Database health check: PASSED"
    else
        error "Database health check: FAILED"
    fi
    
    # Check Redis connectivity
    if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
        log "Redis health check: PASSED"
    else
        error "Redis health check: FAILED"
    fi
    
    log "Health check completed successfully."
}

# =============================================================================
# Usage and Help
# =============================================================================

show_usage() {
    cat << EOF
Horse Racing Prediction - Automated Deployment Script

Usage: $0 [environment] [action] [options]

Environments:
    development     Deploy for development environment
    staging         Deploy for staging environment
    production      Deploy for production environment (default)

Actions:
    deploy          Full deployment (default)
    update          Update existing deployment
    rollback        Rollback to previous version
    backup          Create backup only
    health          Perform health check only
    stop            Stop all services
    start           Start all services
    restart         Restart all services

Options:
    --ssl                   Enable SSL/TLS with Let's Encrypt
    --domain DOMAIN         Domain name for SSL certificate
    --email EMAIL           Email for SSL certificate registration
    --monitoring            Enable monitoring (Prometheus/Grafana)
    --backup                Enable automatic backups
    --force-rebuild         Force rebuild of all containers
    --skip-tests            Skip running tests
    --no-db-backup          Skip database backup before deployment
    --help                  Show this help message

Examples:
    $0 production deploy --ssl --domain example.com --email admin@example.com
    $0 staging update --monitoring
    $0 development deploy --skip-tests
    $0 production rollback
    $0 production health

EOF
}

# =============================================================================
# Main Function
# =============================================================================

main() {
    log "Starting Horse Racing Prediction automated deployment..."
    log "Script: $0"
    log "Arguments: $*"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            development|staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            deploy|update|rollback|backup|health|stop|start|restart)
                ACTION="$1"
                shift
                ;;
            --ssl)
                ENABLE_SSL=true
                shift
                ;;
            --domain)
                DOMAIN="$2"
                shift 2
                ;;
            --email)
                EMAIL="$2"
                shift 2
                ;;
            --monitoring)
                ENABLE_MONITORING=true
                shift
                ;;
            --backup)
                ENABLE_BACKUP=true
                shift
                ;;
            --force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --no-db-backup)
                DB_BACKUP=false
                shift
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
    
    log "Configuration:"
    log "  Environment: $ENVIRONMENT"
    log "  Action: $ACTION"
    log "  SSL Enabled: $ENABLE_SSL"
    log "  Monitoring Enabled: $ENABLE_MONITORING"
    log "  Backup Enabled: $ENABLE_BACKUP"
    log "  Force Rebuild: $FORCE_REBUILD"
    log "  Skip Tests: $SKIP_TESTS"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Execute action
    case $ACTION in
        deploy)
            check_system_requirements
            setup_environment
            setup_ssl
            setup_database
            deploy_application
            setup_monitoring
            health_check
            ;;
        update)
            setup_environment
            deploy_application
            health_check
            ;;
        rollback)
            rollback_deployment
            health_check
            ;;
        backup)
            create_backup
            ;;
        health)
            health_check
            ;;
        stop)
            log "Stopping all services..."
            docker-compose down
            ;;
        start)
            log "Starting all services..."
            docker-compose up -d
            wait_for_services
            ;;
        restart)
            log "Restarting all services..."
            docker-compose restart
            wait_for_services
            ;;
        *)
            error "Unknown action: $ACTION"
            ;;
    esac
    
    log "Deployment script completed successfully!"
    log "Log file: $LOG_FILE"
}

# =============================================================================
# Script Entry Point
# =============================================================================

# Trap errors and cleanup
trap 'error "Script failed at line $LINENO"' ERR

# Run main function with all arguments
main "$@"