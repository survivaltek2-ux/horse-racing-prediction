#!/bin/bash

# =============================================================================
# SSL/TLS Certificate Automation Script
# Horse Racing Prediction Application
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SSL_DIR="$PROJECT_ROOT/ssl"
LOG_DIR="$PROJECT_ROOT/logs/ssl"

# Create directories
mkdir -p "$SSL_DIR" "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default configuration
ENVIRONMENT="production"
ACTION="setup"
DOMAIN=""
EMAIL=""
STAGING=false
FORCE_RENEWAL=false
AUTO_RENEWAL=true

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
# Validation Functions
# =============================================================================

validate_domain() {
    local domain="$1"
    
    if [[ -z "$domain" ]]; then
        error "Domain is required"
        return 1
    fi
    
    # Basic domain validation
    if [[ ! "$domain" =~ ^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$ ]]; then
        error "Invalid domain format: $domain"
        return 1
    fi
    
    log "Domain validation passed: $domain"
    return 0
}

validate_email() {
    local email="$1"
    
    if [[ -z "$email" ]]; then
        error "Email is required for Let's Encrypt registration"
        return 1
    fi
    
    # Basic email validation
    if [[ ! "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        error "Invalid email format: $email"
        return 1
    fi
    
    log "Email validation passed: $email"
    return 0
}

check_dns_resolution() {
    local domain="$1"
    
    log "Checking DNS resolution for $domain..."
    
    if nslookup "$domain" &> /dev/null; then
        local ip=$(nslookup "$domain" | grep -A1 "Name:" | tail -1 | awk '{print $2}' || echo "")
        log "✓ DNS resolution successful: $domain -> $ip"
        return 0
    else
        error "✗ DNS resolution failed for $domain"
        return 1
    fi
}

check_domain_accessibility() {
    local domain="$1"
    
    log "Checking domain accessibility..."
    
    # Check if domain is accessible on port 80
    if curl -s --connect-timeout 10 "http://$domain" &> /dev/null; then
        log "✓ Domain is accessible on port 80"
        return 0
    else
        warn "⚠ Domain may not be accessible on port 80"
        return 1
    fi
}

# =============================================================================
# Certbot Installation and Setup
# =============================================================================

install_certbot() {
    log "Installing Certbot..."
    
    # Check if certbot is already installed
    if command -v certbot &> /dev/null; then
        log "Certbot is already installed"
        certbot --version
        return 0
    fi
    
    # Install certbot based on OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Debian
        if command -v apt-get &> /dev/null; then
            log "Installing Certbot on Ubuntu/Debian..."
            apt-get update
            apt-get install -y certbot python3-certbot-nginx
        # CentOS/RHEL
        elif command -v yum &> /dev/null; then
            log "Installing Certbot on CentOS/RHEL..."
            yum install -y epel-release
            yum install -y certbot python3-certbot-nginx
        # Amazon Linux
        elif command -v amazon-linux-extras &> /dev/null; then
            log "Installing Certbot on Amazon Linux..."
            amazon-linux-extras install -y epel
            yum install -y certbot python3-certbot-nginx
        else
            error "Unsupported Linux distribution"
            return 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            log "Installing Certbot on macOS..."
            brew install certbot
        else
            error "Homebrew is required to install Certbot on macOS"
            return 1
        fi
    else
        error "Unsupported operating system: $OSTYPE"
        return 1
    fi
    
    log "Certbot installation completed"
}

# =============================================================================
# Certificate Management
# =============================================================================

obtain_certificate() {
    local domain="$1"
    local email="$2"
    local staging_flag=""
    
    if [[ "$STAGING" == "true" ]]; then
        staging_flag="--staging"
        log "Using Let's Encrypt staging environment"
    fi
    
    log "Obtaining SSL certificate for $domain..."
    
    # Stop nginx temporarily for standalone mode
    if docker-compose ps nginx 2>/dev/null | grep -q "Up"; then
        log "Stopping nginx for certificate generation..."
        docker-compose stop nginx
    fi
    
    # Obtain certificate using standalone mode
    local certbot_cmd="certbot certonly \
        --standalone \
        --non-interactive \
        --agree-tos \
        --email $email \
        --domains $domain \
        $staging_flag \
        --cert-path $SSL_DIR \
        --key-path $SSL_DIR \
        --fullchain-path $SSL_DIR \
        --chain-path $SSL_DIR"
    
    if $certbot_cmd; then
        log "✓ Certificate obtained successfully"
        
        # Copy certificates to project SSL directory
        copy_certificates "$domain"
        
        # Start nginx again
        log "Starting nginx..."
        docker-compose start nginx
        
        return 0
    else
        error "✗ Failed to obtain certificate"
        
        # Start nginx again even if certificate failed
        docker-compose start nginx
        return 1
    fi
}

obtain_certificate_webroot() {
    local domain="$1"
    local email="$2"
    local webroot_path="$PROJECT_ROOT/public"
    local staging_flag=""
    
    if [[ "$STAGING" == "true" ]]; then
        staging_flag="--staging"
        log "Using Let's Encrypt staging environment"
    fi
    
    log "Obtaining SSL certificate for $domain using webroot method..."
    
    # Ensure webroot directory exists
    mkdir -p "$webroot_path"
    
    # Obtain certificate using webroot mode
    local certbot_cmd="certbot certonly \
        --webroot \
        --webroot-path $webroot_path \
        --non-interactive \
        --agree-tos \
        --email $email \
        --domains $domain \
        $staging_flag"
    
    if $certbot_cmd; then
        log "✓ Certificate obtained successfully"
        copy_certificates "$domain"
        return 0
    else
        error "✗ Failed to obtain certificate"
        return 1
    fi
}

copy_certificates() {
    local domain="$1"
    local cert_dir="/etc/letsencrypt/live/$domain"
    
    log "Copying certificates to project directory..."
    
    if [[ -d "$cert_dir" ]]; then
        cp "$cert_dir/fullchain.pem" "$SSL_DIR/"
        cp "$cert_dir/privkey.pem" "$SSL_DIR/"
        cp "$cert_dir/cert.pem" "$SSL_DIR/"
        cp "$cert_dir/chain.pem" "$SSL_DIR/"
        
        # Set appropriate permissions
        chmod 644 "$SSL_DIR/fullchain.pem" "$SSL_DIR/cert.pem" "$SSL_DIR/chain.pem"
        chmod 600 "$SSL_DIR/privkey.pem"
        
        log "✓ Certificates copied successfully"
    else
        error "Certificate directory not found: $cert_dir"
        return 1
    fi
}

renew_certificates() {
    log "Renewing SSL certificates..."
    
    local force_flag=""
    if [[ "$FORCE_RENEWAL" == "true" ]]; then
        force_flag="--force-renewal"
        log "Forcing certificate renewal"
    fi
    
    if certbot renew $force_flag --quiet; then
        log "✓ Certificate renewal completed"
        
        # Copy renewed certificates
        if [[ -n "$DOMAIN" ]]; then
            copy_certificates "$DOMAIN"
        fi
        
        # Reload nginx
        if docker-compose ps nginx 2>/dev/null | grep -q "Up"; then
            log "Reloading nginx configuration..."
            docker-compose exec nginx nginx -s reload
        fi
        
        return 0
    else
        error "✗ Certificate renewal failed"
        return 1
    fi
}

check_certificate_expiry() {
    local domain="$1"
    local cert_file="$SSL_DIR/fullchain.pem"
    
    if [[ ! -f "$cert_file" ]]; then
        warn "Certificate file not found: $cert_file"
        return 1
    fi
    
    log "Checking certificate expiry for $domain..."
    
    # Get certificate expiration date
    local expiry_date=$(openssl x509 -enddate -noout -in "$cert_file" | cut -d= -f2)
    local expiry_epoch=$(date -d "$expiry_date" +%s)
    local current_epoch=$(date +%s)
    local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
    
    log "Certificate expires in $days_until_expiry days ($expiry_date)"
    
    if [[ $days_until_expiry -gt 30 ]]; then
        log "✓ Certificate is valid for more than 30 days"
        return 0
    elif [[ $days_until_expiry -gt 0 ]]; then
        warn "⚠ Certificate expires in $days_until_expiry days - renewal recommended"
        return 1
    else
        error "✗ Certificate has expired!"
        return 1
    fi
}

# =============================================================================
# Nginx SSL Configuration
# =============================================================================

configure_nginx_ssl() {
    local domain="$1"
    
    log "Configuring Nginx for SSL..."
    
    # Create SSL configuration for nginx
    local nginx_ssl_conf="$PROJECT_ROOT/nginx/ssl.conf"
    mkdir -p "$(dirname "$nginx_ssl_conf")"
    
    cat > "$nginx_ssl_conf" << EOF
# SSL Configuration
ssl_certificate /etc/ssl/certs/fullchain.pem;
ssl_certificate_key /etc/ssl/private/privkey.pem;

# SSL Settings
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384:ECDHE-RSA-AES128-SHA:ECDHE-RSA-AES256-SHA:DHE-RSA-AES128-SHA256:DHE-RSA-AES256-SHA256:DHE-RSA-AES128-SHA:DHE-RSA-AES256-SHA:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA;
ssl_prefer_server_ciphers off;

# SSL Session
ssl_session_timeout 1d;
ssl_session_cache shared:SSL:50m;
ssl_stapling on;
ssl_stapling_verify on;

# Security Headers
add_header Strict-Transport-Security "max-age=63072000" always;
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "strict-origin-when-cross-origin";

# OCSP Stapling
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;
EOF

    # Update main nginx configuration
    local nginx_conf="$PROJECT_ROOT/nginx/nginx.conf"
    
    if [[ ! -f "$nginx_conf" ]]; then
        create_nginx_ssl_config "$domain"
    else
        log "Nginx configuration already exists. Please manually update it to include SSL settings."
        log "SSL configuration saved to: $nginx_ssl_conf"
    fi
    
    log "✓ Nginx SSL configuration completed"
}

create_nginx_ssl_config() {
    local domain="$1"
    local nginx_conf="$PROJECT_ROOT/nginx/nginx.conf"
    
    log "Creating Nginx SSL configuration..."
    
    cat > "$nginx_conf" << EOF
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    log_format main '\$remote_addr - \$remote_user [\$time_local] "\$request" '
                    '\$status \$body_bytes_sent "\$http_referer" '
                    '"\$http_user_agent" "\$http_x_forwarded_for"';
    
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
    
    # Upstream backend
    upstream app_backend {
        server app:8000;
        keepalive 32;
    }
    
    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name $domain;
        
        # Let's Encrypt challenge
        location /.well-known/acme-challenge/ {
            root /var/www/html;
        }
        
        # Redirect all other traffic to HTTPS
        location / {
            return 301 https://\$server_name\$request_uri;
        }
    }
    
    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name $domain;
        
        # Include SSL configuration
        include /etc/nginx/ssl.conf;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Referrer-Policy "strict-origin-when-cross-origin";
        
        # Static files
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
        
        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://app_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
        
        # Login endpoint with stricter rate limiting
        location /auth/login {
            limit_req zone=login burst=5 nodelay;
            proxy_pass http://app_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        # Main application
        location / {
            proxy_pass http://app_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        # Health check endpoint
        location /health {
            access_log off;
            proxy_pass http://app_backend;
            proxy_set_header Host \$host;
        }
    }
}
EOF

    log "✓ Nginx SSL configuration created"
}

# =============================================================================
# Auto-renewal Setup
# =============================================================================

setup_auto_renewal() {
    log "Setting up automatic certificate renewal..."
    
    # Create renewal script
    local renewal_script="$SCRIPT_DIR/ssl-renewal.sh"
    
    cat > "$renewal_script" << EOF
#!/bin/bash
# Automatic SSL certificate renewal script

cd "$PROJECT_ROOT"
"$SCRIPT_DIR/ssl-setup.sh" $ENVIRONMENT renew --domain "$DOMAIN" --email "$EMAIL"
EOF

    chmod +x "$renewal_script"
    
    # Add cron job for automatic renewal
    local cron_job="0 2 * * 0 $renewal_script >> $LOG_DIR/renewal.log 2>&1"
    
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "$renewal_script"; then
        log "Cron job for SSL renewal already exists"
    else
        log "Adding cron job for automatic SSL renewal..."
        (crontab -l 2>/dev/null; echo "$cron_job") | crontab -
        log "✓ Cron job added: Weekly renewal check on Sundays at 2 AM"
    fi
    
    log "✓ Auto-renewal setup completed"
}

# =============================================================================
# Self-signed Certificate (for development)
# =============================================================================

create_self_signed_certificate() {
    local domain="$1"
    
    log "Creating self-signed certificate for development..."
    
    # Generate private key
    openssl genrsa -out "$SSL_DIR/privkey.pem" 2048
    
    # Generate certificate signing request
    openssl req -new -key "$SSL_DIR/privkey.pem" -out "$SSL_DIR/cert.csr" -subj "/C=US/ST=State/L=City/O=Organization/CN=$domain"
    
    # Generate self-signed certificate
    openssl x509 -req -in "$SSL_DIR/cert.csr" -signkey "$SSL_DIR/privkey.pem" -out "$SSL_DIR/cert.pem" -days 365
    
    # Create fullchain (same as cert for self-signed)
    cp "$SSL_DIR/cert.pem" "$SSL_DIR/fullchain.pem"
    
    # Create empty chain file
    touch "$SSL_DIR/chain.pem"
    
    # Set permissions
    chmod 644 "$SSL_DIR/fullchain.pem" "$SSL_DIR/cert.pem" "$SSL_DIR/chain.pem"
    chmod 600 "$SSL_DIR/privkey.pem"
    
    # Clean up CSR
    rm "$SSL_DIR/cert.csr"
    
    log "✓ Self-signed certificate created for $domain"
    warn "⚠ This is a self-signed certificate - browsers will show security warnings"
}

# =============================================================================
# Certificate Information
# =============================================================================

show_certificate_info() {
    local cert_file="$SSL_DIR/fullchain.pem"
    
    if [[ ! -f "$cert_file" ]]; then
        error "Certificate file not found: $cert_file"
        return 1
    fi
    
    log "Certificate Information:"
    
    # Subject
    local subject=$(openssl x509 -subject -noout -in "$cert_file" | sed 's/subject=//')
    log "  Subject: $subject"
    
    # Issuer
    local issuer=$(openssl x509 -issuer -noout -in "$cert_file" | sed 's/issuer=//')
    log "  Issuer: $issuer"
    
    # Validity dates
    local not_before=$(openssl x509 -startdate -noout -in "$cert_file" | cut -d= -f2)
    local not_after=$(openssl x509 -enddate -noout -in "$cert_file" | cut -d= -f2)
    log "  Valid from: $not_before"
    log "  Valid until: $not_after"
    
    # Days until expiry
    local expiry_epoch=$(date -d "$not_after" +%s)
    local current_epoch=$(date +%s)
    local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
    log "  Days until expiry: $days_until_expiry"
    
    # Subject Alternative Names
    local san=$(openssl x509 -text -noout -in "$cert_file" | grep -A1 "Subject Alternative Name" | tail -1 | sed 's/^[[:space:]]*//' || echo "None")
    log "  Subject Alternative Names: $san"
    
    # Key size
    local key_size=$(openssl x509 -text -noout -in "$cert_file" | grep "Public-Key:" | sed 's/.*(\([0-9]*\) bit).*/\1/' || echo "Unknown")
    log "  Key size: ${key_size} bits"
}

# =============================================================================
# Usage and Help
# =============================================================================

show_usage() {
    cat << EOF
SSL/TLS Certificate Automation Script

Usage: $0 [environment] [action] [options]

Environments:
    development     Use self-signed certificates
    staging         Use Let's Encrypt staging environment
    production      Use Let's Encrypt production environment (default)

Actions:
    setup           Obtain and configure SSL certificates (default)
    renew           Renew existing certificates
    check           Check certificate status and expiry
    info            Show certificate information
    auto-renewal    Setup automatic renewal

Options:
    --domain DOMAIN         Domain name for the certificate (required)
    --email EMAIL           Email for Let's Encrypt registration (required for production)
    --staging               Use Let's Encrypt staging environment
    --force-renewal         Force certificate renewal
    --webroot               Use webroot method instead of standalone
    --no-auto-renewal       Skip automatic renewal setup
    --help                  Show this help message

Examples:
    $0 production setup --domain example.com --email admin@example.com
    $0 staging setup --domain staging.example.com --email admin@example.com
    $0 development setup --domain localhost
    $0 production renew --domain example.com --email admin@example.com
    $0 production check --domain example.com
    $0 production auto-renewal --domain example.com --email admin@example.com

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
            setup|renew|check|info|auto-renewal)
                ACTION="$1"
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
            --staging)
                STAGING=true
                shift
                ;;
            --force-renewal)
                FORCE_RENEWAL=true
                shift
                ;;
            --webroot)
                WEBROOT_MODE=true
                shift
                ;;
            --no-auto-renewal)
                AUTO_RENEWAL=false
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use --help for usage information."
                exit 1
                ;;
        esac
    done
    
    log "SSL setup for environment: $ENVIRONMENT"
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
        setup)
            if [[ "$ENVIRONMENT" == "development" ]]; then
                # Development environment - use self-signed certificates
                if [[ -z "$DOMAIN" ]]; then
                    DOMAIN="localhost"
                fi
                create_self_signed_certificate "$DOMAIN"
                configure_nginx_ssl "$DOMAIN"
            else
                # Production/Staging - use Let's Encrypt
                validate_domain "$DOMAIN" || exit 1
                validate_email "$EMAIL" || exit 1
                check_dns_resolution "$DOMAIN"
                check_domain_accessibility "$DOMAIN"
                
                install_certbot
                
                if [[ "${WEBROOT_MODE:-false}" == "true" ]]; then
                    obtain_certificate_webroot "$DOMAIN" "$EMAIL"
                else
                    obtain_certificate "$DOMAIN" "$EMAIL"
                fi
                
                configure_nginx_ssl "$DOMAIN"
                
                if [[ "$AUTO_RENEWAL" == "true" ]]; then
                    setup_auto_renewal
                fi
            fi
            ;;
        renew)
            validate_domain "$DOMAIN" || exit 1
            renew_certificates
            ;;
        check)
            if [[ -n "$DOMAIN" ]]; then
                check_certificate_expiry "$DOMAIN"
            else
                # Check all certificates
                for cert_dir in /etc/letsencrypt/live/*/; do
                    if [[ -d "$cert_dir" ]]; then
                        local domain=$(basename "$cert_dir")
                        check_certificate_expiry "$domain"
                    fi
                done
            fi
            ;;
        info)
            show_certificate_info
            ;;
        auto-renewal)
            validate_domain "$DOMAIN" || exit 1
            validate_email "$EMAIL" || exit 1
            setup_auto_renewal
            ;;
        *)
            error "Unknown action: $ACTION"
            exit 1
            ;;
    esac
    
    log "SSL setup completed!"
}

# Run main function with all arguments
main "$@"