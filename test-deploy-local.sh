#!/bin/bash

# Horse Racing Prediction System - Local Test Deployment Script
# This script performs a test deployment using local Python environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

log "Starting Horse Racing Prediction System Local Test Deployment"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    error "Virtual environment not found. Please create one first with: python -m venv venv"
    exit 1
fi

# Activate virtual environment
log "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
log "Checking Python dependencies..."
if ! python -c "import flask" &> /dev/null; then
    warning "Flask not found. Installing requirements..."
    pip install -r requirements.txt
fi

# Create production environment file if it doesn't exist
if [ ! -f ".env.production" ]; then
    warning "Production environment file not found. Creating from template..."
    cp .env.example .env.production
    
    # Update some production settings
    sed -i '' 's/FLASK_ENV=development/FLASK_ENV=production/' .env.production
    sed -i '' 's/DEBUG=True/DEBUG=False/' .env.production
    
    warning "Please review and update .env.production with your production settings"
fi

# Create necessary directories
log "Creating necessary directories..."
mkdir -p data logs backups

# Set production environment variables
export FLASK_ENV=production
export DEBUG=False
export HOST=0.0.0.0
export PORT=8001  # Use different port to avoid conflicts

# Load production environment
if [ -f ".env.production" ]; then
    log "Loading production environment variables..."
    set -a  # automatically export all variables
    source .env.production
    set +a
fi

# Initialize database
log "Initializing database..."
python -c "
from app import app
from config.database_config import db
with app.app_context():
    db.create_all()
    print('Database initialized successfully')
"

# Test database connection
log "Testing database connection..."
python -c "
from config.database_config import check_database_health
if check_database_health():
    print('✓ Database connection successful')
else:
    print('✗ Database connection failed')
    exit(1)
"

log "Starting application in production mode with gunicorn..."
log "Application will be available at: http://localhost:8001/"
log "Press Ctrl+C to stop the server"

# Start the application with gunicorn
gunicorn --bind 0.0.0.0:8001 --workers 4 --timeout 120 --access-logfile - --error-logfile - app:app