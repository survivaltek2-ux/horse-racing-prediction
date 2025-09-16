#!/bin/bash

# Horse Racing Prediction System - Deployment Script
# This script installs all requirements and sets up the application for production deployment
# Compatible with Ubuntu/Debian and CentOS/RHEL systems

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="horse-racing-prediction"
APP_USER="hrp"
APP_DIR="/opt/$APP_NAME"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="hrp-app"
NGINX_AVAILABLE="/etc/nginx/sites-available"
NGINX_ENABLED="/etc/nginx/sites-enabled"
DOMAIN_NAME="${DOMAIN_NAME:-localhost}"
PORT="${PORT:-8000}"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons."
        error "Please run as a regular user with sudo privileges."
        exit 1
    fi
    
    # Check if user has sudo privileges
    if ! sudo -n true 2>/dev/null; then
        error "This script requires sudo privileges. Please ensure your user can run sudo commands."
        exit 1
    fi
}

# Detect OS
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$ID
        OS_VERSION=$VERSION_ID
    else
        error "Cannot detect operating system"
        exit 1
    fi
    
    log "Detected OS: $OS $OS_VERSION"
}

# Install system dependencies
install_system_dependencies() {
    log "Installing system dependencies..."
    
    case $OS in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y \
                python3 \
                python3-pip \
                python3-venv \
                python3-dev \
                build-essential \
                nginx \
                supervisor \
                git \
                curl \
                wget \
                unzip \
                sqlite3 \
                libsqlite3-dev \
                pkg-config \
                libssl-dev \
                libffi-dev \
                libjpeg-dev \
                libpng-dev \
                libfreetype6-dev \
                libblas-dev \
                liblapack-dev \
                gfortran \
                libatlas-base-dev \
                ufw
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                PKG_MANAGER="dnf"
            else
                PKG_MANAGER="yum"
            fi
            
            sudo $PKG_MANAGER update -y
            sudo $PKG_MANAGER install -y \
                python3 \
                python3-pip \
                python3-devel \
                gcc \
                gcc-c++ \
                make \
                nginx \
                supervisor \
                git \
                curl \
                wget \
                unzip \
                sqlite \
                sqlite-devel \
                openssl-devel \
                libffi-devel \
                libjpeg-turbo-devel \
                libpng-devel \
                freetype-devel \
                blas-devel \
                lapack-devel \
                atlas-devel \
                firewalld
            ;;
        *)
            error "Unsupported operating system: $OS"
            exit 1
            ;;
    esac
    
    log "System dependencies installed successfully"
}

# Create application user
create_app_user() {
    log "Creating application user: $APP_USER"
    
    if ! id "$APP_USER" &>/dev/null; then
        sudo useradd --system --shell /bin/bash --home-dir $APP_DIR --create-home $APP_USER
        log "User $APP_USER created"
    else
        warning "User $APP_USER already exists"
    fi
}

# Setup application directory
setup_app_directory() {
    log "Setting up application directory: $APP_DIR"
    
    # Create directory if it doesn't exist
    sudo mkdir -p $APP_DIR
    
    # Copy application files
    if [[ -f "app.py" ]]; then
        log "Copying application files..."
        sudo cp -r . $APP_DIR/
        sudo chown -R $APP_USER:$APP_USER $APP_DIR
        log "Application files copied to $APP_DIR"
    else
        error "app.py not found. Please run this script from the application directory."
        exit 1
    fi
    
    # Create necessary directories
    sudo -u $APP_USER mkdir -p $APP_DIR/{data,logs,backups,static-html}
    
    # Set proper permissions
    sudo chmod 755 $APP_DIR
    sudo chmod -R 755 $APP_DIR/static-html
}

# Setup Python virtual environment
setup_python_environment() {
    log "Setting up Python virtual environment..."
    
    # Create virtual environment
    sudo -u $APP_USER python3 -m venv $VENV_DIR
    
    # Upgrade pip
    sudo -u $APP_USER $VENV_DIR/bin/pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    if [[ -f "$APP_DIR/requirements.txt" ]]; then
        log "Installing Python dependencies from requirements.txt..."
        sudo -u $APP_USER $VENV_DIR/bin/pip install -r $APP_DIR/requirements.txt
    else
        error "requirements.txt not found in $APP_DIR"
        exit 1
    fi
    
    log "Python environment setup completed"
}

# Setup database
setup_database() {
    log "Setting up SQLite database..."
    
    # Create data directory
    sudo -u $APP_USER mkdir -p $APP_DIR/data
    
    # Initialize database
    cd $APP_DIR
    sudo -u $APP_USER $VENV_DIR/bin/python -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('Database tables created successfully')
"
    
    # Set database permissions
    sudo chmod 664 $APP_DIR/data/hrp_database.db 2>/dev/null || true
    
    log "Database setup completed"
}

# Create environment configuration
create_environment_config() {
    log "Creating environment configuration..."
    
    # Generate secret key
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    
    # Create .env file
    sudo -u $APP_USER tee $APP_DIR/.env > /dev/null <<EOF
# Production Environment Configuration
FLASK_ENV=production
SECRET_KEY=$SECRET_KEY
DEBUG=False

# API Configuration
API_TIMEOUT=30
API_MAX_RETRIES=3
API_RATE_LIMIT_DELAY=1.0
API_CACHE_DURATION=300
API_MAX_RACES=50
API_DEFAULT_DAYS=7

# Provider Selection
DEFAULT_API_PROVIDER=mock
FALLBACK_API_PROVIDER=sample

# Add your API keys here:
# SAMPLE_API_KEY=your_sample_api_key_here
# ODDS_API_API_KEY=your_odds_api_key_here
# RAPID_API_API_KEY=your_rapidapi_key_here
# THERACINGAPI_USERNAME=your_theracingapi_username_here
# THERACINGAPI_PASSWORD=your_theracingapi_password_here

# Database Configuration
DATABASE_URL=sqlite:///$APP_DIR/data/hrp_database.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=$APP_DIR/logs/app.log
EOF

    sudo chmod 600 $APP_DIR/.env
    log "Environment configuration created"
}

# Setup Gunicorn configuration
setup_gunicorn() {
    log "Setting up Gunicorn configuration..."
    
    sudo -u $APP_USER tee $APP_DIR/gunicorn.conf.py > /dev/null <<EOF
# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "127.0.0.1:$PORT"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "$APP_DIR/logs/gunicorn_access.log"
errorlog = "$APP_DIR/logs/gunicorn_error.log"
loglevel = "info"

# Process naming
proc_name = "$APP_NAME"

# Server mechanics
preload_app = True
daemon = False
pidfile = "$APP_DIR/gunicorn.pid"
user = "$APP_USER"
group = "$APP_USER"
tmp_upload_dir = None

# SSL (uncomment and configure if using HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
EOF

    log "Gunicorn configuration created"
}

# Setup Supervisor configuration
setup_supervisor() {
    log "Setting up Supervisor configuration..."
    
    sudo tee /etc/supervisor/conf.d/$SERVICE_NAME.conf > /dev/null <<EOF
[program:$SERVICE_NAME]
command=$VENV_DIR/bin/gunicorn --config $APP_DIR/gunicorn.conf.py app:app
directory=$APP_DIR
user=$APP_USER
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=$APP_DIR/logs/supervisor.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=5
environment=PATH="$VENV_DIR/bin"
EOF

    # Reload supervisor configuration
    sudo supervisorctl reread
    sudo supervisorctl update
    
    log "Supervisor configuration created"
}

# Setup Nginx configuration
setup_nginx() {
    log "Setting up Nginx configuration..."
    
    sudo tee $NGINX_AVAILABLE/$APP_NAME > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN_NAME;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss;
    
    # Static files
    location /static {
        alias $APP_DIR/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    location /static-html {
        alias $APP_DIR/static-html;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Main application
    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_redirect off;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
    }
    
    # Deny access to sensitive files
    location ~ /\. {
        deny all;
    }
    
    location ~ \.(env|conf|py)$ {
        deny all;
    }
}
EOF

    # Enable the site
    sudo ln -sf $NGINX_AVAILABLE/$APP_NAME $NGINX_ENABLED/
    
    # Remove default site if it exists
    sudo rm -f $NGINX_ENABLED/default
    
    # Test nginx configuration
    sudo nginx -t
    
    log "Nginx configuration created"
}

# Setup firewall
setup_firewall() {
    log "Setting up firewall..."
    
    case $OS in
        ubuntu|debian)
            # UFW configuration
            sudo ufw --force reset
            sudo ufw default deny incoming
            sudo ufw default allow outgoing
            sudo ufw allow ssh
            sudo ufw allow 'Nginx Full'
            sudo ufw --force enable
            ;;
        centos|rhel|fedora)
            # Firewalld configuration
            sudo systemctl enable firewalld
            sudo systemctl start firewalld
            sudo firewall-cmd --permanent --add-service=ssh
            sudo firewall-cmd --permanent --add-service=http
            sudo firewall-cmd --permanent --add-service=https
            sudo firewall-cmd --reload
            ;;
    esac
    
    log "Firewall configured"
}

# Create systemd service (alternative to supervisor)
create_systemd_service() {
    log "Creating systemd service..."
    
    sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Horse Racing Prediction Application
After=network.target

[Service]
Type=exec
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
Environment=PATH=$VENV_DIR/bin
ExecStart=$VENV_DIR/bin/gunicorn --config $APP_DIR/gunicorn.conf.py app:app
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME
    
    log "Systemd service created"
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    sudo tee /etc/logrotate.d/$APP_NAME > /dev/null <<EOF
$APP_DIR/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $APP_USER $APP_USER
    postrotate
        supervisorctl restart $SERVICE_NAME || systemctl restart $SERVICE_NAME
    endscript
}
EOF

    log "Log rotation configured"
}

# Create backup script
create_backup_script() {
    log "Creating backup script..."
    
    sudo -u $APP_USER tee $APP_DIR/backup.sh > /dev/null <<'EOF'
#!/bin/bash
# Backup script for Horse Racing Prediction System

BACKUP_DIR="/opt/horse-racing-prediction/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$DATE.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create backup
tar -czf $BACKUP_FILE \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='logs/*.log' \
    /opt/horse-racing-prediction/

# Keep only last 7 backups
find $BACKUP_DIR -name "backup_*.tar.gz" -type f -mtime +7 -delete

echo "Backup created: $BACKUP_FILE"
EOF

    sudo chmod +x $APP_DIR/backup.sh
    
    # Add to crontab for daily backups
    (sudo -u $APP_USER crontab -l 2>/dev/null; echo "0 2 * * * $APP_DIR/backup.sh") | sudo -u $APP_USER crontab -
    
    log "Backup script created and scheduled"
}

# Start services
start_services() {
    log "Starting services..."
    
    # Start and enable nginx
    sudo systemctl enable nginx
    sudo systemctl restart nginx
    
    # Choose between supervisor or systemd
    if command -v supervisorctl &> /dev/null; then
        sudo supervisorctl start $SERVICE_NAME
        log "Application started with Supervisor"
    else
        sudo systemctl start $SERVICE_NAME
        log "Application started with systemd"
    fi
    
    log "All services started successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check if application is responding
    sleep 5
    if curl -f http://localhost:$PORT/ > /dev/null 2>&1; then
        log "✓ Application is responding on port $PORT"
    else
        error "✗ Application is not responding on port $PORT"
        return 1
    fi
    
    # Check nginx
    if sudo systemctl is-active --quiet nginx; then
        log "✓ Nginx is running"
    else
        error "✗ Nginx is not running"
        return 1
    fi
    
    # Check application service
    if sudo supervisorctl status $SERVICE_NAME | grep -q RUNNING || sudo systemctl is-active --quiet $SERVICE_NAME; then
        log "✓ Application service is running"
    else
        error "✗ Application service is not running"
        return 1
    fi
    
    log "Deployment verification completed successfully"
}

# Print deployment summary
print_summary() {
    log "Deployment Summary"
    echo "===================="
    echo "Application: $APP_NAME"
    echo "User: $APP_USER"
    echo "Directory: $APP_DIR"
    echo "Port: $PORT"
    echo "Domain: $DOMAIN_NAME"
    echo "Database: SQLite ($APP_DIR/data/hrp_database.db)"
    echo ""
    echo "Services:"
    echo "- Nginx: http://$DOMAIN_NAME"
    echo "- Application: http://localhost:$PORT"
    echo ""
    echo "Management Commands:"
    echo "- Restart app: sudo supervisorctl restart $SERVICE_NAME"
    echo "- View logs: sudo tail -f $APP_DIR/logs/supervisor.log"
    echo "- Nginx logs: sudo tail -f /var/log/nginx/error.log"
    echo "- Backup: sudo -u $APP_USER $APP_DIR/backup.sh"
    echo ""
    echo "Configuration files:"
    echo "- Environment: $APP_DIR/.env"
    echo "- Gunicorn: $APP_DIR/gunicorn.conf.py"
    echo "- Nginx: $NGINX_AVAILABLE/$APP_NAME"
    echo "- Supervisor: /etc/supervisor/conf.d/$SERVICE_NAME.conf"
    echo ""
    warning "Don't forget to:"
    warning "1. Update API keys in $APP_DIR/.env"
    warning "2. Configure your domain name in DNS"
    warning "3. Set up SSL certificate for HTTPS"
    warning "4. Review and adjust firewall rules"
}

# Main deployment function
main() {
    log "Starting Horse Racing Prediction System deployment..."
    
    check_root
    detect_os
    install_system_dependencies
    create_app_user
    setup_app_directory
    setup_python_environment
    setup_database
    create_environment_config
    setup_gunicorn
    setup_supervisor
    setup_nginx
    setup_firewall
    create_systemd_service
    setup_log_rotation
    create_backup_script
    start_services
    
    if verify_deployment; then
        log "🎉 Deployment completed successfully!"
        print_summary
    else
        error "Deployment verification failed. Please check the logs."
        exit 1
    fi
}

# Run main function
main "$@"