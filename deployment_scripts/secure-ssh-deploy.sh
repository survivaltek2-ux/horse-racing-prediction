#!/bin/bash

# Secure SSH Deployment Script
# Horse Racing Prediction Application
# Version: 1.0.0
# Description: Securely transfers project files to remote droplet with integrity verification

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script configuration
SCRIPT_NAME="secure-ssh-deploy"
SCRIPT_VERSION="1.0.0"
LOG_FILE="/tmp/${SCRIPT_NAME}-$(date +%Y%m%d-%H%M%S).log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ✅ $message" | tee -a "$LOG_FILE"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} ⚠️  $message" | tee -a "$LOG_FILE"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ❌ $message" | tee -a "$LOG_FILE"
            ;;
    esac
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Error handler
error_exit() {
    log "ERROR" "$1"
    log "ERROR" "Deployment failed. Check log file: $LOG_FILE"
    exit 1
}

# Cleanup function
cleanup() {
    log "INFO" "Cleaning up temporary files..."
    rm -f /tmp/checksums_local.txt /tmp/checksums_remote.txt
    log "INFO" "Cleanup completed"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Configuration validation
validate_config() {
    log "INFO" "Validating configuration..."
    
    # Check required environment variables
    if [[ -z "${REMOTE_HOST:-}" ]]; then
        error_exit "REMOTE_HOST environment variable is required"
    fi
    
    if [[ -z "${REMOTE_USER:-}" ]]; then
        error_exit "REMOTE_USER environment variable is required"
    fi
    
    if [[ -z "${SSH_KEY_PATH:-}" ]]; then
        error_exit "SSH_KEY_PATH environment variable is required"
    fi
    
    # Validate SSH key exists and has correct permissions
    if [[ ! -f "$SSH_KEY_PATH" ]]; then
        error_exit "SSH key not found at: $SSH_KEY_PATH"
    fi
    
    # Check SSH key permissions (should be 600)
    local key_perms=$(stat -f "%A" "$SSH_KEY_PATH" 2>/dev/null || stat -c "%a" "$SSH_KEY_PATH" 2>/dev/null)
    if [[ "$key_perms" != "600" ]]; then
        log "WARNING" "SSH key permissions are $key_perms, should be 600. Fixing..."
        chmod 600 "$SSH_KEY_PATH"
        log "SUCCESS" "SSH key permissions corrected"
    fi
    
    log "SUCCESS" "Configuration validation completed"
}

# Test SSH connectivity
test_ssh_connection() {
    log "INFO" "Testing SSH connectivity to $REMOTE_USER@$REMOTE_HOST..."
    
    local ssh_opts=(
        -i "$SSH_KEY_PATH"
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
        -o ConnectTimeout=30
        -o ServerAliveInterval=60
        -o ServerAliveCountMax=3
        -o BatchMode=yes
    )
    
    if ssh "${ssh_opts[@]}" "$REMOTE_USER@$REMOTE_HOST" "echo 'SSH connection successful'" >/dev/null 2>&1; then
        log "SUCCESS" "SSH connection established successfully"
        return 0
    else
        error_exit "Failed to establish SSH connection to $REMOTE_USER@$REMOTE_HOST"
    fi
}

# Create remote directory structure
create_remote_directories() {
    log "INFO" "Creating remote directory structure..."
    
    local remote_app_dir="${REMOTE_APP_DIR:-/opt/horse-racing-app}"
    local ssh_opts=(
        -i "$SSH_KEY_PATH"
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
        -o ConnectTimeout=30
    )
    
    ssh "${ssh_opts[@]}" "$REMOTE_USER@$REMOTE_HOST" "
        sudo mkdir -p '$remote_app_dir'
        sudo chown $REMOTE_USER:$REMOTE_USER '$remote_app_dir'
        mkdir -p '$remote_app_dir'/{config,data,logs,backups}
    " || error_exit "Failed to create remote directories"
    
    log "SUCCESS" "Remote directory structure created"
}

# Calculate file checksums for integrity verification
calculate_checksums() {
    local source_dir="$1"
    local checksum_file="$2"
    
    log "INFO" "Calculating checksums for integrity verification..."
    
    # Find all files (including hidden) and calculate checksums
    find "$source_dir" -type f -not -path "*/.*" -not -path "*/__pycache__/*" -not -path "*/node_modules/*" \
        -exec sha256sum {} \; | sed "s|$source_dir/||g" | sort > "$checksum_file"
    
    local file_count=$(wc -l < "$checksum_file")
    log "INFO" "Calculated checksums for $file_count files"
}

# Transfer files with progress
transfer_files() {
    log "INFO" "Starting secure file transfer..."
    
    local source_dir="$(pwd)"
    local remote_app_dir="${REMOTE_APP_DIR:-/opt/horse-racing-app}"
    local ssh_opts=(
        -i "$SSH_KEY_PATH"
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
        -o ConnectTimeout=30
        -o ServerAliveInterval=60
    )
    
    # Create list of files to transfer
    local transfer_list="/tmp/transfer_list.txt"
    find . -type f \
        -not -path "./.git/*" \
        -not -path "./logs/*" \
        -not -path "./__pycache__/*" \
        -not -path "./node_modules/*" \
        -not -path "./venv/*" \
        -not -path "./.env.local" \
        > "$transfer_list"
    
    local total_files=$(wc -l < "$transfer_list")
    log "INFO" "Transferring $total_files files to remote server..."
    
    # Transfer files using rsync for better progress and resume capability
    rsync -avz --progress \
        --files-from="$transfer_list" \
        -e "ssh $(printf '%s ' "${ssh_opts[@]}")" \
        . "$REMOTE_USER@$REMOTE_HOST:$remote_app_dir/" \
        || error_exit "File transfer failed"
    
    log "SUCCESS" "File transfer completed successfully"
    rm -f "$transfer_list"
}

# Transfer deployment configuration files
transfer_deployment_configs() {
    log "INFO" "Transferring deployment configuration files..."
    
    local remote_app_dir="${REMOTE_APP_DIR:-/opt/horse-racing-app}"
    local ssh_opts=(
        -i "$SSH_KEY_PATH"
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
    )
    
    # List of critical deployment files
    local config_files=(
        "Dockerfile"
        "Dockerfile.production"
        "docker-compose.yml"
        "docker-compose.ha.yml"
        ".env.production"
        "requirements.txt"
        "nginx.conf"
        "digitalocean-app-spec.yml"
        "scripts/deployment/"
        "config/"
    )
    
    for config_file in "${config_files[@]}"; do
        if [[ -e "$config_file" ]]; then
            log "INFO" "Transferring $config_file..."
            
            if [[ -d "$config_file" ]]; then
                # Transfer directory
                rsync -avz --progress \
                    -e "ssh $(printf '%s ' "${ssh_opts[@]}")" \
                    "$config_file" "$REMOTE_USER@$REMOTE_HOST:$remote_app_dir/" \
                    || log "WARNING" "Failed to transfer directory: $config_file"
            else
                # Transfer file
                scp "${ssh_opts[@]}" "$config_file" \
                    "$REMOTE_USER@$REMOTE_HOST:$remote_app_dir/$config_file" \
                    || log "WARNING" "Failed to transfer file: $config_file"
            fi
        else
            log "WARNING" "Configuration file not found: $config_file"
        fi
    done
    
    log "SUCCESS" "Deployment configuration transfer completed"
}

# Verify file integrity
verify_integrity() {
    log "INFO" "Verifying file integrity..."
    
    local remote_app_dir="${REMOTE_APP_DIR:-/opt/horse-racing-app}"
    local ssh_opts=(
        -i "$SSH_KEY_PATH"
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
    )
    
    # Calculate local checksums
    calculate_checksums "." "/tmp/checksums_local.txt"
    
    # Calculate remote checksums
    ssh "${ssh_opts[@]}" "$REMOTE_USER@$REMOTE_HOST" "
        cd '$remote_app_dir'
        find . -type f -not -path '*/.*' -not -path '*/__pycache__/*' \
            -exec sha256sum {} \; | sed 's|./||g' | sort
    " > /tmp/checksums_remote.txt || error_exit "Failed to calculate remote checksums"
    
    # Compare checksums
    if diff /tmp/checksums_local.txt /tmp/checksums_remote.txt >/dev/null; then
        log "SUCCESS" "File integrity verification passed - all files transferred correctly"
    else
        log "ERROR" "File integrity verification failed - some files may be corrupted"
        log "INFO" "Checksum differences:"
        diff /tmp/checksums_local.txt /tmp/checksums_remote.txt || true
        error_exit "Integrity verification failed"
    fi
}

# Set remote permissions
set_remote_permissions() {
    log "INFO" "Setting appropriate permissions on remote files..."
    
    local remote_app_dir="${REMOTE_APP_DIR:-/opt/horse-racing-app}"
    local ssh_opts=(
        -i "$SSH_KEY_PATH"
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
    )
    
    ssh "${ssh_opts[@]}" "$REMOTE_USER@$REMOTE_HOST" "
        cd '$remote_app_dir'
        
        # Set directory permissions
        find . -type d -exec chmod 755 {} \;
        
        # Set file permissions
        find . -type f -exec chmod 644 {} \;
        
        # Set executable permissions for scripts
        find . -name '*.sh' -exec chmod 755 {} \;
        find . -name 'deploy*' -exec chmod 755 {} \;
        
        # Secure sensitive files
        if [[ -f '.env.production' ]]; then
            chmod 600 .env.production
        fi
        
        # Make sure deployment scripts are executable
        if [[ -d 'scripts/deployment' ]]; then
            chmod -R 755 scripts/deployment/
        fi
    " || error_exit "Failed to set remote permissions"
    
    log "SUCCESS" "Remote permissions set successfully"
}

# Main deployment function
main() {
    log "INFO" "🚀 Starting secure SSH deployment..."
    log "INFO" "Script: $SCRIPT_NAME v$SCRIPT_VERSION"
    log "INFO" "Timestamp: $(date)"
    log "INFO" "Log file: $LOG_FILE"
    
    # Validate configuration
    validate_config
    
    # Test SSH connection
    test_ssh_connection
    
    # Create remote directories
    create_remote_directories
    
    # Transfer project files
    transfer_files
    
    # Transfer deployment configurations
    transfer_deployment_configs
    
    # Set appropriate permissions
    set_remote_permissions
    
    # Verify file integrity
    verify_integrity
    
    log "SUCCESS" "🎉 Secure SSH deployment completed successfully!"
    log "INFO" "Remote application directory: ${REMOTE_APP_DIR:-/opt/horse-racing-app}"
    log "INFO" "Log file saved: $LOG_FILE"
    
    echo ""
    echo "Next steps:"
    echo "1. SSH into the remote server: ssh -i $SSH_KEY_PATH $REMOTE_USER@$REMOTE_HOST"
    echo "2. Navigate to the app directory: cd ${REMOTE_APP_DIR:-/opt/horse-racing-app}"
    echo "3. Run the deployment script: ./scripts/deployment/deploy-digitalocean.sh"
    echo ""
}

# Script usage information
usage() {
    cat << EOF
Usage: $0

Environment Variables Required:
  REMOTE_HOST     - Remote server hostname or IP address
  REMOTE_USER     - Remote server username
  SSH_KEY_PATH    - Path to SSH private key file
  REMOTE_APP_DIR  - Remote application directory (optional, default: /opt/horse-racing-app)

Example:
  export REMOTE_HOST="your-server.com"
  export REMOTE_USER="root"
  export SSH_KEY_PATH="~/.ssh/id_rsa"
  export REMOTE_APP_DIR="/opt/horse-racing-app"
  $0

Security Features:
  ✅ SSH key authentication (no passwords)
  ✅ Encrypted file transfer
  ✅ File integrity verification
  ✅ Secure permission setting
  ✅ Connection timeout handling
  ✅ Comprehensive error handling
  ✅ Progress feedback
  ✅ Detailed logging

EOF
}

# Check if help is requested
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

# Run main function
main "$@"