#!/bin/bash

# SSH Deployment Wrapper Script
# This script loads configuration and executes the secure SSH deployment

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Horse Racing Prediction - SSH Deployment${NC}"
echo "=================================================="

# Check if configuration file exists
CONFIG_FILE="$SCRIPT_DIR/ssh-deploy-config.local.env"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${YELLOW}⚠️  Configuration file not found: $CONFIG_FILE${NC}"
    echo ""
    echo "Please create the configuration file by copying the template:"
    echo "  cp $SCRIPT_DIR/ssh-deploy-config.env $CONFIG_FILE"
    echo ""
    echo "Then edit the configuration file with your server details:"
    echo "  nano $CONFIG_FILE"
    echo ""
    exit 1
fi

# Load configuration
echo "📋 Loading configuration from: $CONFIG_FILE"
source "$CONFIG_FILE"

# Validate required variables
required_vars=("REMOTE_HOST" "REMOTE_USER" "SSH_KEY_PATH")
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo -e "${RED}❌ Required variable $var is not set in configuration file${NC}"
        exit 1
    fi
done

# Expand tilde in SSH_KEY_PATH
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

# Display configuration summary
echo ""
echo "📊 Deployment Configuration:"
echo "  Remote Host: $REMOTE_HOST"
echo "  Remote User: $REMOTE_USER"
echo "  Remote Directory: ${REMOTE_APP_DIR:-/opt/horse-racing-app}"
echo "  SSH Key: $SSH_KEY_PATH"
echo ""

# Confirm deployment
read -p "🤔 Do you want to proceed with the deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled by user"
    exit 0
fi

# Change to project root
cd "$PROJECT_ROOT"

# Export configuration for the deployment script
export REMOTE_HOST
export REMOTE_USER
export SSH_KEY_PATH
export REMOTE_APP_DIR

# Execute the secure SSH deployment script
echo ""
echo "🔄 Starting secure SSH deployment..."
exec "$SCRIPT_DIR/secure-ssh-deploy.sh"