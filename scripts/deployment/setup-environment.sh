#!/bin/bash

# =============================================================================
# Environment Setup and Configuration Script
# Horse Racing Prediction Application
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ENVIRONMENT="production"
INTERACTIVE=true
VALIDATE_ONLY=false

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

prompt_user() {
    local prompt="$1"
    local default="$2"
    local response
    
    if [[ "$INTERACTIVE" == "false" ]]; then
        echo "$default"
        return
    fi
    
    read -p "$prompt [$default]: " response
    echo "${response:-$default}"
}

generate_random_password() {
    local length="${1:-32}"
    openssl rand -base64 "$length" | tr -d "=+/" | cut -c1-"$length"
}

generate_secret_key() {
    python3 -c "import secrets; print(secrets.token_urlsafe(64))"
}

# =============================================================================
# Environment Validation
# =============================================================================

validate_environment() {
    local env_file="$PROJECT_ROOT/.env.$ENVIRONMENT"
    
    log "Validating environment configuration for: $ENVIRONMENT"
    
    if [[ ! -f "$env_file" ]]; then
        warn "Environment file $env_file not found."
        return 1
    fi
    
    # Source the environment file
    set -a
    source "$env_file"
    set +a
    
    local validation_errors=0
    
    # Required variables
    local required_vars=(
        "SECRET_KEY"
        "DATABASE_URL"
        "POSTGRES_USER"
        "POSTGRES_PASSWORD"
        "POSTGRES_DB"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
            ((validation_errors++))
        fi
    done
    
    # Validate SECRET_KEY strength
    if [[ -n "${SECRET_KEY:-}" ]]; then
        if [[ ${#SECRET_KEY} -lt 32 ]]; then
            warn "SECRET_KEY should be at least 32 characters long"
            ((validation_errors++))
        fi
    fi
    
    # Validate database URL format
    if [[ -n "${DATABASE_URL:-}" ]]; then
        if [[ ! "$DATABASE_URL" =~ ^postgresql:// ]]; then
            warn "DATABASE_URL should start with postgresql://"
            ((validation_errors++))
        fi
    fi
    
    # Validate Redis configuration
    if [[ -n "${REDIS_URL:-}" ]]; then
        if [[ ! "$REDIS_URL" =~ ^redis:// ]]; then
            warn "REDIS_URL should start with redis://"
            ((validation_errors++))
        fi
    fi
    
    # Environment-specific validations
    case "$ENVIRONMENT" in
        production)
            validate_production_environment
            ;;
        staging)
            validate_staging_environment
            ;;
        development)
            validate_development_environment
            ;;
    esac
    
    if [[ $validation_errors -eq 0 ]]; then
        log "Environment validation passed."
        return 0
    else
        error "Environment validation failed with $validation_errors errors."
        return 1
    fi
}

validate_production_environment() {
    log "Validating production-specific configuration..."
    
    # Production should have DEBUG=False
    if [[ "${DEBUG:-}" == "True" || "${DEBUG:-}" == "true" ]]; then
        error "DEBUG should be False in production environment"
    fi
    
    # Production should have secure session settings
    if [[ "${SESSION_COOKIE_SECURE:-}" != "True" ]]; then
        warn "SESSION_COOKIE_SECURE should be True in production"
    fi
    
    if [[ "${SESSION_COOKIE_HTTPONLY:-}" != "True" ]]; then
        warn "SESSION_COOKIE_HTTPONLY should be True in production"
    fi
    
    # Check for SSL configuration
    if [[ -z "${SSL_CERT_PATH:-}" && -z "${DOMAIN:-}" ]]; then
        warn "SSL configuration not found. Consider enabling SSL for production."
    fi
}

validate_staging_environment() {
    log "Validating staging-specific configuration..."
    
    # Staging can have DEBUG=True but should warn
    if [[ "${DEBUG:-}" == "True" || "${DEBUG:-}" == "true" ]]; then
        info "DEBUG is enabled in staging environment"
    fi
}

validate_development_environment() {
    log "Validating development-specific configuration..."
    
    # Development warnings
    if [[ "${DEBUG:-}" != "True" && "${DEBUG:-}" != "true" ]]; then
        info "DEBUG is disabled in development environment"
    fi
}

# =============================================================================
# Environment Creation
# =============================================================================

create_environment_file() {
    local env_file="$PROJECT_ROOT/.env.$ENVIRONMENT"
    
    log "Creating environment file: $env_file"
    
    if [[ -f "$env_file" && "$INTERACTIVE" == "true" ]]; then
        local overwrite=$(prompt_user "Environment file already exists. Overwrite?" "n")
        if [[ "$overwrite" != "y" && "$overwrite" != "yes" ]]; then
            log "Keeping existing environment file."
            return 0
        fi
    fi
    
    # Gather configuration
    local app_name=$(prompt_user "Application name" "horse-racing-prediction")
    local app_version=$(prompt_user "Application version" "1.0.0")
    local secret_key=$(generate_secret_key)
    local db_password=$(generate_random_password 24)
    local redis_password=$(generate_random_password 24)
    
    # Database configuration
    local db_host=$(prompt_user "Database host" "postgres")
    local db_port=$(prompt_user "Database port" "5432")
    local db_name=$(prompt_user "Database name" "horse_racing_db")
    local db_user=$(prompt_user "Database user" "app_user")
    
    # Redis configuration
    local redis_host=$(prompt_user "Redis host" "redis")
    local redis_port=$(prompt_user "Redis port" "6379")
    
    # Environment-specific settings
    local debug_mode="False"
    local session_secure="True"
    local session_httponly="True"
    
    case "$ENVIRONMENT" in
        development)
            debug_mode="True"
            session_secure="False"
            ;;
        staging)
            debug_mode="False"
            session_secure="True"
            ;;
        production)
            debug_mode="False"
            session_secure="True"
            ;;
    esac
    
    # Create the environment file
    cat > "$env_file" << EOF
# =============================================================================
# $ENVIRONMENT Environment Configuration
# Horse Racing Prediction System
# Generated on: $(date)
# =============================================================================

# Application Configuration
FLASK_ENV=$ENVIRONMENT
SECRET_KEY=$secret_key
DEBUG=$debug_mode
APP_NAME=$app_name
APP_VERSION=$app_version
HOST=0.0.0.0
PORT=8000

# Database Configuration
DATABASE_URL=postgresql://$db_user:$db_password@$db_host:$db_port/$db_name
DATABASE_ECHO=False
POSTGRES_DB=$db_name
POSTGRES_USER=$db_user
POSTGRES_PASSWORD=$db_password
POSTGRES_HOST=$db_host
POSTGRES_PORT=$db_port

# Database Connection Pool Settings
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Redis Configuration
REDIS_URL=redis://:$redis_password@$redis_host:$redis_port/0
REDIS_HOST=$redis_host
REDIS_PORT=$redis_port
REDIS_PASSWORD=$redis_password
REDIS_DB=0

# Session Configuration
SESSION_COOKIE_SECURE=$session_secure
SESSION_COOKIE_HTTPONLY=$session_httponly
SESSION_COOKIE_SAMESITE=Lax
SESSION_PERMANENT=False
PERMANENT_SESSION_LIFETIME=3600

# Security Configuration
WTF_CSRF_ENABLED=True
WTF_CSRF_TIME_LIMIT=3600

# API Configuration
API_TIMEOUT=30
API_MAX_RETRIES=3
DEFAULT_API_PROVIDER=mock

# External API Keys (set these manually)
SAMPLE_API_KEY=
ODDS_API_API_KEY=
RAPID_API_API_KEY=
THERACINGAPI_USERNAME=
THERACINGAPI_PASSWORD=

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE_MAX_SIZE=10485760
LOG_FILE_BACKUP_COUNT=5

# File Upload Configuration
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=uploads
ALLOWED_EXTENSIONS=csv,json,txt

# Cache Configuration
CACHE_TYPE=redis
CACHE_DEFAULT_TIMEOUT=300
CACHE_KEY_PREFIX=hrp_

# Email Configuration (for notifications)
MAIL_SERVER=
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=
MAIL_PASSWORD=
MAIL_DEFAULT_SENDER=

# Monitoring Configuration
ENABLE_METRICS=True
METRICS_PORT=9090

# Backup Configuration
BACKUP_RETENTION_DAYS=30
BACKUP_SCHEDULE=0 2 * * *

# SSL Configuration (for production)
SSL_CERT_PATH=
SSL_KEY_PATH=
DOMAIN=

EOF

    # Set proper permissions
    chmod 600 "$env_file"
    
    log "Environment file created: $env_file"
    log "Please review and update the configuration as needed."
    
    # Show important notes
    info "Important notes:"
    info "1. Update external API keys in the environment file"
    info "2. Configure email settings for notifications"
    info "3. Set SSL certificate paths for production"
    info "4. Review and adjust database connection pool settings"
}

# =============================================================================
# Docker Environment Setup
# =============================================================================

setup_docker_environment() {
    log "Setting up Docker environment files..."
    
    local env_file="$PROJECT_ROOT/.env.$ENVIRONMENT"
    local docker_env_file="$PROJECT_ROOT/.env"
    
    if [[ ! -f "$env_file" ]]; then
        error "Environment file $env_file not found. Create it first."
    fi
    
    # Create Docker environment file
    log "Creating Docker environment file: $docker_env_file"
    cp "$env_file" "$docker_env_file"
    
    # Create docker-compose override for environment
    local override_file="$PROJECT_ROOT/docker-compose.$ENVIRONMENT.yml"
    
    case "$ENVIRONMENT" in
        production)
            create_production_override
            ;;
        staging)
            create_staging_override
            ;;
        development)
            create_development_override
            ;;
    esac
    
    log "Docker environment setup completed."
}

create_production_override() {
    local override_file="$PROJECT_ROOT/docker-compose.production.yml"
    
    cat > "$override_file" << 'EOF'
version: '3.8'

services:
  app:
    restart: always
    environment:
      - FLASK_ENV=production
      - DEBUG=False
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    restart: always
    environment:
      - POSTGRES_INITDB_ARGS=--auth-host=md5
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    command: >
      postgres
      -c max_connections=200
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
      -c maintenance_work_mem=64MB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=16MB
      -c default_statistics_target=100

  redis:
    restart: always
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.25'
        reservations:
          memory: 256M
          cpus: '0.1'
    command: >
      redis-server
      --maxmemory 256mb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
      --save 60 10000

  nginx:
    restart: always
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.25'
        reservations:
          memory: 128M
          cpus: '0.1'
EOF

    log "Created production Docker Compose override: $override_file"
}

create_staging_override() {
    local override_file="$PROJECT_ROOT/docker-compose.staging.yml"
    
    cat > "$override_file" << 'EOF'
version: '3.8'

services:
  app:
    restart: unless-stopped
    environment:
      - FLASK_ENV=staging
      - DEBUG=False
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

  postgres:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  redis:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.25'
EOF

    log "Created staging Docker Compose override: $override_file"
}

create_development_override() {
    local override_file="$PROJECT_ROOT/docker-compose.development.yml"
    
    cat > "$override_file" << 'EOF'
version: '3.8'

services:
  app:
    environment:
      - FLASK_ENV=development
      - DEBUG=True
    volumes:
      - .:/app
      - /app/__pycache__
    command: flask run --host=0.0.0.0 --port=8000 --debug

  postgres:
    ports:
      - "5432:5432"

  redis:
    ports:
      - "6379:6379"
EOF

    log "Created development Docker Compose override: $override_file"
}

# =============================================================================
# System Dependencies
# =============================================================================

install_system_dependencies() {
    log "Installing system dependencies..."
    
    # Detect OS
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        OS=$ID
    else
        error "Cannot detect operating system"
    fi
    
    case "$OS" in
        ubuntu|debian)
            install_debian_dependencies
            ;;
        centos|rhel|fedora)
            install_redhat_dependencies
            ;;
        *)
            warn "Unsupported operating system: $OS"
            warn "Please install Docker, Docker Compose, and other dependencies manually"
            ;;
    esac
}

install_debian_dependencies() {
    log "Installing dependencies for Debian/Ubuntu..."
    
    sudo apt-get update
    
    # Install required packages
    sudo apt-get install -y \
        curl \
        wget \
        git \
        openssl \
        ca-certificates \
        gnupg \
        lsb-release
    
    # Install Docker
    if ! command -v docker &> /dev/null; then
        log "Installing Docker..."
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io
        sudo usermod -aG docker $USER
    fi
    
    # Install Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
}

install_redhat_dependencies() {
    log "Installing dependencies for RedHat/CentOS/Fedora..."
    
    # Install required packages
    sudo yum install -y \
        curl \
        wget \
        git \
        openssl \
        ca-certificates
    
    # Install Docker
    if ! command -v docker &> /dev/null; then
        log "Installing Docker..."
        sudo yum install -y yum-utils
        sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
        sudo yum install -y docker-ce docker-ce-cli containerd.io
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker $USER
    fi
    
    # Install Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
}

# =============================================================================
# Usage and Help
# =============================================================================

show_usage() {
    cat << EOF
Environment Setup and Configuration Script

Usage: $0 [environment] [options]

Environments:
    development     Setup development environment
    staging         Setup staging environment
    production      Setup production environment (default)

Options:
    --create                Create new environment file
    --validate              Validate existing environment file
    --docker                Setup Docker environment files
    --install-deps          Install system dependencies
    --non-interactive       Run without user prompts
    --help                  Show this help message

Examples:
    $0 production --create
    $0 staging --validate
    $0 development --docker
    $0 production --install-deps --non-interactive

EOF
}

# =============================================================================
# Main Function
# =============================================================================

main() {
    local create_env=false
    local setup_docker=false
    local install_deps=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            development|staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            --create)
                create_env=true
                shift
                ;;
            --validate)
                VALIDATE_ONLY=true
                shift
                ;;
            --docker)
                setup_docker=true
                shift
                ;;
            --install-deps)
                install_deps=true
                shift
                ;;
            --non-interactive)
                INTERACTIVE=false
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
    
    log "Environment setup for: $ENVIRONMENT"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Execute requested actions
    if [[ "$install_deps" == "true" ]]; then
        install_system_dependencies
    fi
    
    if [[ "$create_env" == "true" ]]; then
        create_environment_file
    fi
    
    if [[ "$setup_docker" == "true" ]]; then
        setup_docker_environment
    fi
    
    if [[ "$VALIDATE_ONLY" == "true" ]]; then
        validate_environment
    fi
    
    # If no specific action requested, run validation
    if [[ "$create_env" == "false" && "$setup_docker" == "false" && "$install_deps" == "false" && "$VALIDATE_ONLY" == "false" ]]; then
        validate_environment
    fi
    
    log "Environment setup completed successfully!"
}

# Run main function with all arguments
main "$@"