#!/bin/bash

# DigitalOcean Automated Deployment Script
# Horse Racing Prediction Application
# 
# This script automates the complete deployment process on DigitalOcean:
# 1. Droplet provisioning with specified configurations
# 2. Security hardening (firewall, SSH, fail2ban)
# 3. Dependencies installation (Docker, monitoring tools)
# 4. Application deployment and configuration
# 5. Monitoring and logging setup
#
# Prerequisites:
# - DigitalOcean CLI (doctl) installed and authenticated
# - SSH key pair generated and added to DigitalOcean account
# - Environment variables configured (see .env.digitalocean)
#
# Usage: ./deploy-digitalocean.sh [environment] [action]
# Example: ./deploy-digitalocean.sh production deploy

set -euo pipefail

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Script metadata
SCRIPT_NAME="DigitalOcean Deployment Script"
SCRIPT_VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Environment configuration
ENVIRONMENT="${1:-production}"
ACTION="${2:-deploy}"

# Load environment-specific configuration
ENV_FILE="${PROJECT_ROOT}/.env.digitalocean"
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
fi

# DigitalOcean Configuration
DO_TOKEN="${DO_TOKEN:-}"
DO_REGION="${DO_REGION:-nyc1}"
DO_SIZE="${DO_SIZE:-s-2vcpu-4gb}"
DO_IMAGE="${DO_IMAGE:-ubuntu-22-04-x64}"
DO_SSH_KEY_NAME="${DO_SSH_KEY_NAME:-horse-racing-key}"
DO_DROPLET_NAME="${DO_DROPLET_NAME:-horse-racing-${ENVIRONMENT}}"
DO_DOMAIN="${DO_DOMAIN:-}"
DO_SUBDOMAIN="${DO_SUBDOMAIN:-app}"

# Application Configuration
APP_NAME="horse-racing-prediction"
APP_PORT="${APP_PORT:-8000}"
APP_USER="${APP_USER:-appuser}"
APP_DIR="/opt/${APP_NAME}"
BACKUP_DIR="/opt/backups"

# Database Configuration
DB_NAME="${DB_NAME:-horse_racing_db}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-$(openssl rand -base64 32)}"

# Redis Configuration
REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"

# SSL Configuration
SSL_EMAIL="${SSL_EMAIL:-admin@example.com}"
ENABLE_SSL="${ENABLE_SSL:-true}"

# Monitoring Configuration
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-$(openssl rand -base64 32)}"

# Logging
LOG_DIR="${PROJECT_ROOT}/logs/digitalocean"
LOG_FILE="${LOG_DIR}/deploy-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "${BLUE}ℹ️  $*${NC}"
}

log_success() {
    log "SUCCESS" "${GREEN}✅ $*${NC}"
}

log_warning() {
    log "WARNING" "${YELLOW}⚠️  $*${NC}"
}

log_error() {
    log "ERROR" "${RED}❌ $*${NC}"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        log "DEBUG" "${PURPLE}🔍 $*${NC}"
    fi
}

# Error handling
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Script failed at line $line_number with exit code $exit_code"
    log_error "Initiating cleanup and rollback procedures..."
    cleanup_on_error
    exit $exit_code
}

trap 'handle_error $LINENO' ERR

# Cleanup function
cleanup_on_error() {
    log_warning "Performing cleanup operations..."
    
    # Stop any running services
    if [[ -n "${DROPLET_IP:-}" ]]; then
        log_info "Attempting to clean up droplet resources..."
        ssh_execute "sudo systemctl stop docker || true"
        ssh_execute "sudo systemctl stop nginx || true"
    fi
    
    # Optionally destroy the droplet on critical failures
    if [[ "${CLEANUP_ON_ERROR:-false}" == "true" && -n "${DROPLET_ID:-}" ]]; then
        log_warning "Destroying droplet due to critical failure..."
        DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN" doctl compute droplet delete "$DROPLET_ID" --force || true
    fi
}

# Validation functions
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check if doctl is installed
    if ! command -v doctl &> /dev/null; then
        log_error "DigitalOcean CLI (doctl) is not installed"
        log_info "Install with: snap install doctl"
        exit 1
    fi
    
    # Check if doctl is authenticated with the token
    if ! DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN" doctl account get &> /dev/null; then
        log_error "DigitalOcean CLI authentication failed with provided token"
        log_info "Check your DO_TOKEN in .env.digitalocean file"
        exit 1
    fi
    
    # Check required environment variables
    if [[ -z "$DO_TOKEN" ]]; then
        log_error "DO_TOKEN environment variable is required"
        exit 1
    fi
    
    # Validate SSH key exists and get its ID
    log_info "Checking for SSH key: ${DO_SSH_KEY_NAME}"
    
    # Get the SSH key ID
    DO_SSH_KEY_ID=$(DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN" doctl compute ssh-key list --format ID,Name --no-header | grep "${DO_SSH_KEY_NAME}$" | awk '{print $1}')
    
    if [[ -z "$DO_SSH_KEY_ID" ]]; then
        log_error "SSH key '${DO_SSH_KEY_NAME}' not found in DigitalOcean account"
        log_info "Available SSH keys:"
        DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN" doctl compute ssh-key list --format Name --no-header
        log_info "Add your SSH key with: DIGITALOCEAN_ACCESS_TOKEN=\"$DO_TOKEN\" doctl compute ssh-key import ${DO_SSH_KEY_NAME} --public-key-file ~/.ssh/id_rsa.pub"
        exit 1
    fi
    
    log_info "Found SSH key '${DO_SSH_KEY_NAME}' with ID: ${DO_SSH_KEY_ID}"
    
    # Check if required files exist
    local required_files=(
        "${PROJECT_ROOT}/docker-compose.yml"
        "${PROJECT_ROOT}/Dockerfile.production"
        "${PROJECT_ROOT}/nginx.conf"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    log_success "Prerequisites validation completed"
}

# SSH execution helper
ssh_execute() {
    local command="$1"
    local user="${2:-root}"
    
    if [[ -z "${DROPLET_IP:-}" ]]; then
        log_error "Droplet IP not available for SSH execution"
        return 1
    fi
    
    log_debug "Executing SSH command: $command"
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=30 -o ServerAliveInterval=60 \
        "${user}@${DROPLET_IP}" "$command"
}

# File transfer helper
scp_transfer() {
    local source="$1"
    local destination="$2"
    local user="${3:-root}"
    
    if [[ -z "${DROPLET_IP:-}" ]]; then
        log_error "Droplet IP not available for file transfer"
        return 1
    fi
    
    log_debug "Transferring file: $source -> $destination"
    scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -r "$source" "${user}@${DROPLET_IP}:${destination}"
}

# Wait for service to be ready
wait_for_service() {
    local service_name="$1"
    local port="$2"
    local timeout="${3:-300}"
    local interval="${4:-10}"
    
    log_info "Waiting for $service_name to be ready on port $port..."
    
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if ssh_execute "curl -f http://localhost:$port/health 2>/dev/null" &> /dev/null; then
            log_success "$service_name is ready"
            return 0
        fi
        
        log_debug "Waiting for $service_name... (${elapsed}s/${timeout}s)"
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    
    log_error "$service_name failed to become ready within ${timeout}s"
    return 1
}

# =============================================================================
# DROPLET MANAGEMENT FUNCTIONS
# =============================================================================

create_droplet() {
    log_info "Creating DigitalOcean droplet: $DO_DROPLET_NAME"
    
    # Check if droplet already exists
    if DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN" doctl compute droplet list --format Name --no-header | grep -q "^${DO_DROPLET_NAME}$"; then
        log_info "Droplet '${DO_DROPLET_NAME}' already exists"
        DROPLET_ID=$(DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN" doctl compute droplet list --format ID,Name --no-header | grep "$DO_DROPLET_NAME" | awk '{print $1}')
        DROPLET_IP=$(DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN" doctl compute droplet get "$DROPLET_ID" --format PublicIPv4 --no-header)
        log_info "Using existing droplet: ID=$DROPLET_ID, IP=$DROPLET_IP"
        return 0
    fi
    
    # Create the droplet
    log_info "Creating new droplet with specifications:"
    log_info "  Region: $DO_REGION"
    log_info "  Size: $DO_SIZE"
    log_info "  Image: $DO_IMAGE"
    log_info "  SSH Key: $DO_SSH_KEY_NAME"
    
    DROPLET_ID=$(DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN" doctl compute droplet create "$DO_DROPLET_NAME" \
        --region "$DO_REGION" \
        --size "$DO_SIZE" \
        --image "$DO_IMAGE" \
        --ssh-keys "$DO_SSH_KEY_ID" \
        --enable-monitoring \
        --enable-ipv6 \
        --user-data-file "$(dirname "$0")/cloud-init.yml" \
        --tag-names "environment:${ENVIRONMENT},app:${APP_NAME}" \
        --format ID --no-header)
    
    if [[ -z "$DROPLET_ID" ]]; then
        log_error "Failed to create droplet"
        exit 1
    fi
    
    log_success "Droplet created with ID: $DROPLET_ID"
    
    # Wait for droplet to be active
    log_info "Waiting for droplet to become active..."
    local timeout=300
    local elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        local status=$(DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN" doctl compute droplet get "$DROPLET_ID" --format Status --no-header)
        if [[ "$status" == "active" ]]; then
            break
        fi
        
        log_debug "Droplet status: $status (${elapsed}s/${timeout}s)"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    
    if [[ $elapsed -ge $timeout ]]; then
        log_error "Droplet failed to become active within ${timeout}s"
        exit 1
    fi
    
    # Get droplet IP
    DROPLET_IP=$(DIGITALOCEAN_ACCESS_TOKEN="$DO_TOKEN" doctl compute droplet get "$DROPLET_ID" --format PublicIPv4 --no-header)
    log_success "Droplet is active with IP: $DROPLET_IP"
    
    # Wait for SSH to be available
    log_info "Waiting for SSH to become available..."
    timeout=600  # Increased to 10 minutes for Ubuntu 22.04 boot time
    elapsed=0
    retry_count=0
    
    while [[ $elapsed -lt $timeout ]]; do
        retry_count=$((retry_count + 1))
        
        if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
               -o ConnectTimeout=10 -o BatchMode=yes \
               "root@${DROPLET_IP}" "echo 'SSH Ready'" &> /dev/null; then
            log_success "SSH connection established after ${elapsed}s (${retry_count} attempts)"
            break
        fi
        
        # Log progress every 30 seconds
        if [[ $((elapsed % 30)) -eq 0 ]]; then
            log_info "Waiting for SSH... (${elapsed}s/${timeout}s, attempt ${retry_count})"
        fi
        
        sleep 15  # Check every 15 seconds instead of 10
        elapsed=$((elapsed + 15))
    done
    
    if [[ $elapsed -ge $timeout ]]; then
        log_error "SSH failed to become available within ${timeout}s after ${retry_count} attempts"
        log_info "You can try connecting manually: ssh root@${DROPLET_IP}"
        exit 1
    fi
    
    log_success "SSH is available"
}

# =============================================================================
# SECURITY CONFIGURATION FUNCTIONS
# =============================================================================

configure_security() {
    log_info "Configuring security measures..."
    
    # Update system packages
    log_info "Updating system packages..."
    ssh_execute "apt-get update && apt-get upgrade -y"
    
    # Configure firewall
    configure_firewall
    
    # Harden SSH configuration
    harden_ssh
    
    # Install and configure fail2ban
    if [[ "${ENABLE_FAIL2BAN:-true}" == "true" ]]; then
        install_fail2ban
    fi
    
    # Create application user
    create_app_user
    
    # Set up automatic security updates
    configure_auto_updates
    
    log_success "Security configuration completed"
}

configure_firewall() {
    log_info "Configuring UFW firewall..."
    
    # Install UFW if not present
    ssh_execute "apt-get install -y ufw"
    
    # Reset UFW to defaults
    ssh_execute "ufw --force reset"
    
    # Set default policies
    ssh_execute "ufw default deny incoming"
    ssh_execute "ufw default allow outgoing"
    
    # Allow SSH
    local ssh_port="${SSH_PORT:-22}"
    ssh_execute "ufw allow ${ssh_port}/tcp comment 'SSH'"
    
    # Allow HTTP and HTTPS
    ssh_execute "ufw allow 80/tcp comment 'HTTP'"
    ssh_execute "ufw allow 443/tcp comment 'HTTPS'"
    
    # Allow application port
    ssh_execute "ufw allow ${APP_PORT}/tcp comment 'Application'"
    
    # Allow monitoring ports (if monitoring is enabled)
    if [[ "${ENABLE_MONITORING:-true}" == "true" ]]; then
        ssh_execute "ufw allow 3000/tcp comment 'Grafana'"
        ssh_execute "ufw allow 9090/tcp comment 'Prometheus'"
    fi
    
    # Enable UFW
    ssh_execute "ufw --force enable"
    
    # Show status
    ssh_execute "ufw status verbose"
    
    log_success "Firewall configured successfully"
}

harden_ssh() {
    log_info "Hardening SSH configuration..."
    
    # Backup original SSH config
    ssh_execute "cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup"
    
    # Create hardened SSH configuration
    ssh_execute "cat > /etc/ssh/sshd_config << 'EOF'
# SSH Configuration - Security Hardened
Port ${SSH_PORT:-22}
Protocol 2

# Authentication
PermitRootLogin yes
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
PasswordAuthentication no
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM yes

# Security settings
X11Forwarding no
PrintMotd no
AcceptEnv LANG LC_*
Subsystem sftp /usr/lib/openssh/sftp-server

# Connection settings
ClientAliveInterval 300
ClientAliveCountMax 2
MaxAuthTries 3
MaxSessions 10
LoginGraceTime 60

# Logging
SyslogFacility AUTH
LogLevel INFO
EOF"
    
    # Validate SSH configuration
    ssh_execute "sshd -t"
    
    # Restart SSH service
    ssh_execute "systemctl restart sshd"
    
    log_success "SSH hardening completed"
}

install_fail2ban() {
    log_info "Installing and configuring fail2ban..."
    
    # Install fail2ban
    ssh_execute "apt-get install -y fail2ban"
    
    # Create custom jail configuration
    ssh_execute "cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
backend = systemd

[sshd]
enabled = true
port = ${SSH_PORT:-22}
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 3
EOF"
    
    # Start and enable fail2ban
    ssh_execute "systemctl enable fail2ban"
    ssh_execute "systemctl start fail2ban"
    
    # Check status
    ssh_execute "fail2ban-client status"
    
    log_success "Fail2ban configured successfully"
}

create_app_user() {
    log_info "Creating application user: $APP_USER"
    
    # Create user with home directory
    ssh_execute "useradd -m -s /bin/bash $APP_USER || true"
    
    # Add user to docker group (will be created later)
    ssh_execute "usermod -aG sudo $APP_USER || true"
    
    # Create application directory
    ssh_execute "mkdir -p $APP_DIR"
    ssh_execute "chown $APP_USER:$APP_USER $APP_DIR"
    
    # Create backup directory
    ssh_execute "mkdir -p $BACKUP_DIR"
    ssh_execute "chown $APP_USER:$APP_USER $BACKUP_DIR"
    
    log_success "Application user created successfully"
}

configure_auto_updates() {
    log_info "Configuring automatic security updates..."
    
    # Install unattended-upgrades
    ssh_execute "apt-get install -y unattended-upgrades"
    
    # Configure automatic updates
    ssh_execute "cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    \"\${distro_id}:\${distro_codename}-security\";
    \"\${distro_id}ESMApps:\${distro_codename}-apps-security\";
    \"\${distro_id}ESM:\${distro_codename}-infra-security\";
};

Unattended-Upgrade::AutoFixInterruptedDpkg \"true\";
Unattended-Upgrade::MinimalSteps \"true\";
Unattended-Upgrade::Remove-Unused-Dependencies \"true\";
Unattended-Upgrade::Automatic-Reboot \"false\";
EOF"
    
    # Enable automatic updates
    ssh_execute "cat > /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists \"1\";
APT::Periodic::Unattended-Upgrade \"1\";
EOF"
    
    log_success "Automatic security updates configured"
}

# =============================================================================
# DEPENDENCIES INSTALLATION FUNCTIONS
# =============================================================================

install_dependencies() {
    log_info "Installing required dependencies..."
    
    # Install basic packages
    install_basic_packages
    
    # Install Docker and Docker Compose
    install_docker
    
    # Install monitoring tools
    if [[ "${ENABLE_MONITORING:-true}" == "true" ]]; then
        install_monitoring_tools
    fi
    
    # Install SSL certificate tools
    if [[ "${ENABLE_SSL:-true}" == "true" ]]; then
        install_ssl_tools
    fi
    
    log_success "Dependencies installation completed"
}

install_basic_packages() {
    log_info "Installing basic system packages..."
    
    ssh_execute "apt-get update"
    ssh_execute "apt-get install -y \
        curl \
        wget \
        git \
        unzip \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        htop \
        vim \
        nano \
        jq \
        tree \
        rsync \
        cron \
        logrotate"
    
    log_success "Basic packages installed"
}

install_docker() {
    log_info "Installing Docker and Docker Compose..."
    
    # Add Docker's official GPG key
    ssh_execute "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg"
    
    # Add Docker repository
    ssh_execute "echo \"deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \$(lsb_release -cs) stable\" | tee /etc/apt/sources.list.d/docker.list > /dev/null"
    
    # Update package index
    ssh_execute "apt-get update"
    
    # Install Docker
    ssh_execute "apt-get install -y docker-ce docker-ce-cli containerd.io"
    
    # Install Docker Compose
    ssh_execute "curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose"
    ssh_execute "chmod +x /usr/local/bin/docker-compose"
    
    # Add app user to docker group
    ssh_execute "usermod -aG docker $APP_USER"
    
    # Start and enable Docker
    ssh_execute "systemctl start docker"
    ssh_execute "systemctl enable docker"
    
    # Verify installation
    ssh_execute "docker --version"
    ssh_execute "docker-compose --version"
    
    log_success "Docker and Docker Compose installed successfully"
}

install_monitoring_tools() {
    log_info "Installing monitoring tools..."
    
    # Install Node Exporter for Prometheus
    ssh_execute "wget https://github.com/prometheus/node_exporter/releases/latest/download/node_exporter-1.6.1.linux-amd64.tar.gz"
    ssh_execute "tar xvfz node_exporter-1.6.1.linux-amd64.tar.gz"
    ssh_execute "mv node_exporter-1.6.1.linux-amd64/node_exporter /usr/local/bin/"
    ssh_execute "rm -rf node_exporter-1.6.1.linux-amd64*"
    
    # Create node_exporter service
    ssh_execute "cat > /etc/systemd/system/node_exporter.service << 'EOF'
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=nobody
Group=nogroup
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
EOF"
    
    # Start and enable node_exporter
    ssh_execute "systemctl daemon-reload"
    ssh_execute "systemctl start node_exporter"
    ssh_execute "systemctl enable node_exporter"
    
    log_success "Monitoring tools installed"
}

install_ssl_tools() {
    log_info "Installing SSL certificate tools..."
    
    # Install Certbot for Let's Encrypt
    ssh_execute "apt-get install -y certbot python3-certbot-nginx"
    
    log_success "SSL tools installed"
}

# =============================================================================
# APPLICATION DEPLOYMENT FUNCTIONS
# =============================================================================

deploy_application() {
    log_info "Deploying application..."
    
    # Transfer application files
    transfer_application_files
    
    # Configure environment
    configure_application_environment
    
    # Build and start services
    build_and_start_services
    
    # Configure Nginx
    configure_nginx
    
    # Set up SSL certificates
    if [[ "${ENABLE_SSL:-true}" == "true" ]]; then
        setup_ssl_certificates
    fi
    
    # Configure backups
    if [[ "${ENABLE_BACKUPS:-true}" == "true" ]]; then
        configure_backups
    fi
    
    log_success "Application deployment completed"
}

transfer_application_files() {
    log_info "Transferring application files..."
    
    # Create temporary archive of application files
    local temp_archive="/tmp/app-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    log_info "Creating application archive..."
    tar -czf "$temp_archive" \
        --exclude='.git' \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.env*' \
        --exclude='logs' \
        --exclude='backups' \
        -C "$PROJECT_ROOT" .
    
    # Transfer archive to droplet
    log_info "Transferring archive to droplet..."
    scp_transfer "$temp_archive" "/tmp/app.tar.gz"
    
    # Extract archive on droplet
    ssh_execute "cd $APP_DIR && tar -xzf /tmp/app.tar.gz"
    ssh_execute "rm /tmp/app.tar.gz"
    ssh_execute "chown -R $APP_USER:$APP_USER $APP_DIR"
    
    # Clean up local archive
    rm -f "$temp_archive"
    
    log_success "Application files transferred successfully"
}

configure_application_environment() {
    log_info "Configuring application environment..."
    
    # Create production environment file
    ssh_execute "cat > $APP_DIR/.env.production << 'EOF'
# Production Environment Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=$(openssl rand -base64 32)

# Database Configuration
DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@postgres:5432/$DB_NAME

# Redis Configuration
REDIS_URL=redis://:$REDIS_PASSWORD@redis:6379/0

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=$APP_PORT

# Security Configuration
CSRF_SECRET_KEY=$(openssl rand -base64 32)
JWT_SECRET_KEY=$(openssl rand -base64 32)

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=9100
EOF"
    
    # Set proper permissions
    ssh_execute "chmod 600 $APP_DIR/.env.production"
    ssh_execute "chown $APP_USER:$APP_USER $APP_DIR/.env.production"
    
    log_success "Application environment configured"
}

build_and_start_services() {
    log_info "Building and starting services..."
    
    # Create Docker Compose override for production
    ssh_execute "cat > $APP_DIR/docker-compose.production.yml << 'EOF'
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.production
    environment:
      - FLASK_ENV=production
    env_file:
      - .env.production
    restart: unless-stopped
    depends_on:
      - postgres
      - redis
    networks:
      - app-network

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: $DB_NAME
      POSTGRES_USER: $DB_USER
      POSTGRES_PASSWORD: $DB_PASSWORD
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass $REDIS_PASSWORD
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - app-network

  nginx:
    image: nginx:alpine
    ports:
      - \"80:80\"
      - \"443:443\"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - app
    restart: unless-stopped
    networks:
      - app-network

volumes:
  postgres_data:
  redis_data:
  nginx_logs:

networks:
  app-network:
    driver: bridge
EOF"
    
    # Build and start services
    ssh_execute "cd $APP_DIR && docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d --build" "$APP_USER"
    
    # Wait for services to be ready
    wait_for_service "Application" "$APP_PORT"
    
    log_success "Services started successfully"
}

configure_nginx() {
    log_info "Configuring Nginx..."
    
    # Create Nginx configuration
    ssh_execute "cat > $APP_DIR/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    log_format main '\$remote_addr - \$remote_user [\$time_local] \"\$request\" '
                    '\$status \$body_bytes_sent \"\$http_referer\" '
                    '\"\$http_user_agent\" \"\$http_x_forwarded_for\"';
    
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=login:10m rate=1r/s;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection \"1; mode=block\";
    add_header Strict-Transport-Security \"max-age=31536000; includeSubDomains\" always;
    
    upstream app {
        server app:$APP_PORT;
    }
    
    server {
        listen 80;
        server_name $DO_DOMAIN www.$DO_DOMAIN;
        
        # Redirect HTTP to HTTPS
        return 301 https://\$server_name\$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name $DO_DOMAIN www.$DO_DOMAIN;
        
        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        
        # Application proxy
        location / {
            proxy_pass http://app;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
        
        # API rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://app;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        # Login rate limiting
        location /auth/login {
            limit_req zone=login burst=5 nodelay;
            proxy_pass http://app;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        # Health check
        location /health {
            proxy_pass http://app/health;
            access_log off;
        }
        
        # Static files
        location /static/ {
            alias /opt/horse-racing-prediction/static/;
            expires 1y;
            add_header Cache-Control \"public, immutable\";
        }
    }
}
EOF"
    
    log_success "Nginx configured successfully"
}

setup_ssl_certificates() {
    log_info "Setting up SSL certificates..."
    
    if [[ -z "$DO_DOMAIN" ]]; then
        log_warning "Domain not configured, skipping SSL setup"
        return 0
    fi
    
    # Create SSL directory
    ssh_execute "mkdir -p $APP_DIR/ssl"
    
    # Stop Nginx temporarily for certificate generation
    ssh_execute "cd $APP_DIR && docker-compose stop nginx" "$APP_USER"
    
    # Generate Let's Encrypt certificate
    ssh_execute "certbot certonly --standalone --non-interactive --agree-tos --email $SSL_EMAIL -d $DO_DOMAIN -d www.$DO_DOMAIN"
    
    # Copy certificates to application directory
    ssh_execute "cp /etc/letsencrypt/live/$DO_DOMAIN/fullchain.pem $APP_DIR/ssl/"
    ssh_execute "cp /etc/letsencrypt/live/$DO_DOMAIN/privkey.pem $APP_DIR/ssl/"
    ssh_execute "chown -R $APP_USER:$APP_USER $APP_DIR/ssl"
    
    # Set up certificate renewal
    ssh_execute "cat > /etc/cron.d/certbot-renewal << 'EOF'
0 12 * * * root certbot renew --quiet --post-hook \"cd $APP_DIR && docker-compose restart nginx\"
EOF"
    
    # Restart Nginx with SSL
    ssh_execute "cd $APP_DIR && docker-compose start nginx" "$APP_USER"
    
    log_success "SSL certificates configured successfully"
}

configure_backups() {
    log_info "Configuring backup system..."
    
    # Create backup script
    ssh_execute "cat > $BACKUP_DIR/backup.sh << 'EOF'
#!/bin/bash

BACKUP_DATE=\$(date +%Y%m%d_%H%M%S)
BACKUP_PATH=\"$BACKUP_DIR/\$BACKUP_DATE\"

# Create backup directory
mkdir -p \"\$BACKUP_PATH\"

# Backup database
cd $APP_DIR
docker-compose exec -T postgres pg_dump -U $DB_USER $DB_NAME > \"\$BACKUP_PATH/database.sql\"

# Backup application data
tar -czf \"\$BACKUP_PATH/app_data.tar.gz\" -C $APP_DIR data/

# Backup configuration
cp $APP_DIR/.env.production \"\$BACKUP_PATH/\"
cp $APP_DIR/docker-compose.production.yml \"\$BACKUP_PATH/\"

# Remove old backups (keep last 7 days)
find $BACKUP_DIR -type d -name \"20*\" -mtime +${BACKUP_RETENTION_DAYS:-7} -exec rm -rf {} +

echo \"Backup completed: \$BACKUP_PATH\"
EOF"
    
    # Make backup script executable
    ssh_execute "chmod +x $BACKUP_DIR/backup.sh"
    ssh_execute "chown $APP_USER:$APP_USER $BACKUP_DIR/backup.sh"
    
    # Set up daily backup cron job
    ssh_execute "cat > /etc/cron.d/app-backup << 'EOF'
0 2 * * * $APP_USER $BACKUP_DIR/backup.sh >> /var/log/backup.log 2>&1
EOF"
    
    log_success "Backup system configured successfully"
}

# =============================================================================
# MONITORING AND LOGGING FUNCTIONS
# =============================================================================

setup_monitoring() {
    log_info "Setting up monitoring and logging..."
    
    if [[ "${ENABLE_MONITORING:-true}" != "true" ]]; then
        log_info "Monitoring disabled, skipping setup"
        return 0
    fi
    
    # Deploy monitoring stack
    deploy_monitoring_stack
    
    # Configure log aggregation
    configure_log_aggregation
    
    # Set up health checks
    setup_health_checks
    
    # Configure alerting
    configure_alerting
    
    log_success "Monitoring and logging setup completed"
}

deploy_monitoring_stack() {
    log_info "Deploying monitoring stack..."
    
    # Create monitoring Docker Compose file
    ssh_execute "cat > $APP_DIR/docker-compose.monitoring.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - \"9090:9090\"
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    ports:
      - \"3000:3000\"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=$GRAFANA_PASSWORD
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
EOF"
    
    # Create Prometheus configuration
    ssh_execute "mkdir -p $APP_DIR/monitoring/prometheus"
    ssh_execute "cat > $APP_DIR/monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['$DROPLET_IP:9100']

  - job_name: 'application'
    static_configs:
      - targets: ['app:$APP_PORT']
    metrics_path: '/metrics'
EOF"
    
    # Start monitoring services
    ssh_execute "cd $APP_DIR && docker-compose -f docker-compose.monitoring.yml up -d" "$APP_USER"
    
    log_success "Monitoring stack deployed"
}

configure_log_aggregation() {
    log_info "Configuring log aggregation..."
    
    # Configure log rotation
    ssh_execute "cat > /etc/logrotate.d/app-logs << 'EOF'
$APP_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $APP_USER $APP_USER
    postrotate
        cd $APP_DIR && docker-compose restart app
    endscript
}
EOF"
    
    log_success "Log aggregation configured"
}

setup_health_checks() {
    log_info "Setting up health checks..."
    
    # Create health check script
    ssh_execute "cat > $APP_DIR/health-check.sh << 'EOF'
#!/bin/bash

# Check application health
if ! curl -f http://localhost:$APP_PORT/health >/dev/null 2>&1; then
    echo \"Application health check failed\"
    exit 1
fi

# Check database connectivity
if ! docker-compose exec -T postgres pg_isready -U $DB_USER >/dev/null 2>&1; then
    echo \"Database health check failed\"
    exit 1
fi

# Check Redis connectivity
if ! docker-compose exec -T redis redis-cli -a $REDIS_PASSWORD ping >/dev/null 2>&1; then
    echo \"Redis health check failed\"
    exit 1
fi

echo \"All health checks passed\"
EOF"
    
    ssh_execute "chmod +x $APP_DIR/health-check.sh"
    ssh_execute "chown $APP_USER:$APP_USER $APP_DIR/health-check.sh"
    
    # Set up health check cron job
    ssh_execute "cat > /etc/cron.d/health-check << 'EOF'
*/5 * * * * $APP_USER cd $APP_DIR && ./health-check.sh >> /var/log/health-check.log 2>&1
EOF"
    
    log_success "Health checks configured"
}

configure_alerting() {
    log_info "Configuring alerting..."
    
    # Create alerting script
    ssh_execute "cat > $APP_DIR/alert.sh << 'EOF'
#!/bin/bash

ALERT_TYPE=\"\$1\"
ALERT_MESSAGE=\"\$2\"

# Log alert
echo \"\$(date): [\$ALERT_TYPE] \$ALERT_MESSAGE\" >> /var/log/alerts.log

# Send notifications (if configured)
if [[ -n \"$SLACK_WEBHOOK_URL\" ]]; then
    curl -X POST -H 'Content-type: application/json' \
        --data \"{\\\"text\\\":\\\"[\$ALERT_TYPE] \$ALERT_MESSAGE\\\"}\" \
        \"$SLACK_WEBHOOK_URL\"
fi
EOF"
    
    ssh_execute "chmod +x $APP_DIR/alert.sh"
    ssh_execute "chown $APP_USER:$APP_USER $APP_DIR/alert.sh"
    
    log_success "Alerting configured"
}

# =============================================================================
# ERROR HANDLING AND ROLLBACK FUNCTIONS
# =============================================================================

cleanup_on_error() {
    local exit_code=$?
    log_error "Deployment failed with exit code: $exit_code"
    
    if [[ -n "${DROPLET_ID:-}" ]]; then
        log_info "Cleaning up resources..."
        
        # Ask user if they want to keep the droplet for debugging
        if [[ "${AUTO_CLEANUP:-false}" == "true" ]]; then
            destroy_droplet
        else
            log_warning "Droplet $DROPLET_ID left running for debugging"
            log_info "To destroy manually: DIGITALOCEAN_ACCESS_TOKEN=\"$DO_TOKEN\" doctl compute droplet delete $DROPLET_ID"
        fi
    fi
    
    exit $exit_code
}

rollback_deployment() {
    log_info "Rolling back deployment..."
    
    if [[ -n "$DROPLET_ID" ]]; then
        # Stop services
        ssh_execute "cd $APP_DIR && docker-compose down" "$APP_USER" || true
        
        # Remove application files
        ssh_execute "rm -rf $APP_DIR" || true
        
        # Optionally destroy droplet
        if [[ "${ROLLBACK_DESTROY_DROPLET:-false}" == "true" ]]; then
            destroy_droplet
        fi
    fi
    
    log_success "Rollback completed"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    log_info "Starting DigitalOcean deployment..."
    log_info "Timestamp: $(date)"
    log_info "Script version: $SCRIPT_VERSION"
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    # Validate prerequisites
    validate_prerequisites
    
    # Create or get droplet
    if [[ "${USE_EXISTING_DROPLET:-false}" == "true" ]]; then
        log_info "Using existing droplet: $EXISTING_DROPLET_ID"
        DROPLET_ID="$EXISTING_DROPLET_ID"
        get_droplet_info
    else
        create_droplet
        # SSH availability is already checked in create_droplet function
    fi
    
    # Configure security
    configure_security
    
    # Install dependencies
    install_dependencies
    
    # Deploy application
    deploy_application
    
    # Set up monitoring
    setup_monitoring
    
    # Final verification
    verify_deployment
    
    # Display deployment summary
    display_deployment_summary
    
    log_success "Deployment completed successfully!"
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check if services are running
    local services=("app" "postgres" "redis" "nginx")
    for service in "${services[@]}"; do
        if ! ssh_execute "cd $APP_DIR && docker-compose ps $service | grep -q 'Up'" "$APP_USER"; then
            log_error "Service $service is not running"
            return 1
        fi
    done
    
    # Check application health endpoint
    if ! ssh_execute "curl -f http://localhost:$APP_PORT/health >/dev/null 2>&1"; then
        log_error "Application health check failed"
        return 1
    fi
    
    # Check external access
    if [[ -n "$DO_DOMAIN" ]]; then
        if ! curl -f "https://$DO_DOMAIN/health" >/dev/null 2>&1; then
            log_warning "External health check failed - DNS may not be propagated yet"
        fi
    fi
    
    log_success "Deployment verification completed"
}

display_deployment_summary() {
    log_info "=== DEPLOYMENT SUMMARY ==="
    log_info "Droplet ID: $DROPLET_ID"
    log_info "Droplet IP: $DROPLET_IP"
    log_info "Application URL: http://$DROPLET_IP:$APP_PORT"
    
    if [[ -n "$DO_DOMAIN" ]]; then
        log_info "Domain URL: https://$DO_DOMAIN"
    fi
    
    log_info "Monitoring URLs:"
    log_info "  - Grafana: http://$DROPLET_IP:3000 (admin/$GRAFANA_PASSWORD)"
    log_info "  - Prometheus: http://$DROPLET_IP:9090"
    
    log_info "SSH Access: ssh $APP_USER@$DROPLET_IP"
    log_info "Application Directory: $APP_DIR"
    log_info "Backup Directory: $BACKUP_DIR"
    
    log_info "=== NEXT STEPS ==="
    log_info "1. Update DNS records to point $DO_DOMAIN to $DROPLET_IP"
    log_info "2. Monitor application logs: ssh $APP_USER@$DROPLET_IP 'cd $APP_DIR && docker-compose logs -f'"
    log_info "3. Set up regular backups and monitoring alerts"
    log_info "4. Review security settings and update as needed"
    
    # Save deployment info to file
    cat > "$PROJECT_ROOT/deployment-info.txt" << EOF
DigitalOcean Deployment Information
Generated: $(date)

Droplet Details:
- ID: $DROPLET_ID
- IP: $DROPLET_IP
- Region: $DO_REGION
- Size: $DO_SIZE

Access Information:
- SSH: ssh $APP_USER@$DROPLET_IP
- Application: http://$DROPLET_IP:$APP_PORT
- Domain: ${DO_DOMAIN:-"Not configured"}

Monitoring:
- Grafana: http://$DROPLET_IP:3000
- Prometheus: http://$DROPLET_IP:9090

Directories:
- Application: $APP_DIR
- Backups: $BACKUP_DIR
- Logs: $APP_DIR/logs

Configuration Files:
- Environment: $APP_DIR/.env.production
- Docker Compose: $APP_DIR/docker-compose.production.yml
- Nginx: $APP_DIR/nginx.conf
EOF
    
    log_info "Deployment information saved to: $PROJECT_ROOT/deployment-info.txt"
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    main "$@"
fi