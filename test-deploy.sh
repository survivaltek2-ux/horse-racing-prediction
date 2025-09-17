#!/bin/bash

# Horse Racing Prediction System - Test Deployment Script
# This script performs a test deployment using Docker

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

# Configuration
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.production"

log "Starting Horse Racing Prediction System Test Deployment"

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    error "Docker is not running. Please start Docker first."
    exit 1
fi

log "Docker is available and running"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    error "Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Use docker-compose or docker compose based on availability
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

log "Using $DOCKER_COMPOSE for deployment"

# Create environment file if it doesn't exist
if [ ! -f "$ENV_FILE" ]; then
    warning "Production environment file not found. Creating default..."
    cp .env.example "$ENV_FILE"
    warning "Please edit $ENV_FILE with your production settings"
fi

# Create necessary directories
log "Creating necessary directories..."
mkdir -p data logs backups ssl

# Stop any existing containers
log "Stopping existing containers..."
$DOCKER_COMPOSE -f $COMPOSE_FILE down --remove-orphans || true

# Build and start the application
log "Building and starting the application..."
$DOCKER_COMPOSE -f $COMPOSE_FILE build --no-cache
$DOCKER_COMPOSE -f $COMPOSE_FILE up -d

# Wait for services to be ready
log "Waiting for services to start..."
sleep 10

# Check if services are running
log "Checking service status..."
if $DOCKER_COMPOSE -f $COMPOSE_FILE ps | grep -q "Up"; then
    log "Services are running successfully!"
else
    error "Some services failed to start. Check logs:"
    $DOCKER_COMPOSE -f $COMPOSE_FILE logs
    exit 1
fi

# Test application health
log "Testing application health..."
sleep 5

if curl -f http://localhost:8000/ &> /dev/null; then
    log "✓ Application is responding on http://localhost:8000/"
else
    warning "Application health check failed. Checking logs..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE logs app
fi

# Show running containers
log "Running containers:"
$DOCKER_COMPOSE -f $COMPOSE_FILE ps

# Show logs
log "Recent application logs:"
$DOCKER_COMPOSE -f $COMPOSE_FILE logs --tail=20 app

log "Test deployment completed!"
log "Access the application at: http://localhost:8000/"
log "To stop the deployment: $DOCKER_COMPOSE -f $COMPOSE_FILE down"
log "To view logs: $DOCKER_COMPOSE -f $COMPOSE_FILE logs -f"