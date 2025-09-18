#!/bin/bash

# Deployment Validation Script (Non-Docker)
# Horse Racing Prediction System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Logging function
log() {
    echo -e "$1"
}

# Function to validate configuration files
validate_configs() {
    log "${BLUE}📋 Validating configuration files...${NC}"
    
    local configs=(
        ".env.production:Production environment variables"
        "docker-swarm-stack.yml:Docker Swarm stack configuration"
        "nginx/nginx-production.conf:Nginx production configuration"
        "ssl/cert.pem:SSL certificate"
        "ssl/key.pem:SSL private key"
        "ssl/dhparam.pem:SSL DH parameters"
    )
    
    local errors=0
    
    for config in "${configs[@]}"; do
        local file="${config%%:*}"
        local desc="${config##*:}"
        
        if [ -f "$PROJECT_ROOT/$file" ]; then
            log "${GREEN}✅ $desc: $file${NC}"
        else
            log "${RED}❌ Missing $desc: $file${NC}"
            ((errors++))
        fi
    done
    
    return $errors
}

# Function to validate environment variables
validate_env_vars() {
    log "${BLUE}🔧 Validating environment variables...${NC}"
    
    if [ ! -f "$PROJECT_ROOT/.env.production" ]; then
        log "${RED}❌ .env.production file not found${NC}"
        return 1
    fi
    
    source "$PROJECT_ROOT/.env.production"
    
    local required_vars=(
        "SECRET_KEY:Application secret key"
        "POSTGRES_PASSWORD:PostgreSQL password"
        "REDIS_PASSWORD:Redis password"
        "ENCRYPTION_KEY:Encryption key"
        "JWT_SECRET_KEY:JWT secret key"
        "GRAFANA_ADMIN_PASSWORD:Grafana admin password"
        "ELASTICSEARCH_PASSWORD:Elasticsearch password"
    )
    
    local errors=0
    
    for var_info in "${required_vars[@]}"; do
        local var_name="${var_info%%:*}"
        local var_desc="${var_info##*:}"
        local var_value="${!var_name:-}"
        
        if [ -n "$var_value" ] && [ "$var_value" != "your-"* ] && [ "$var_value" != "secure_"* ]; then
            log "${GREEN}✅ $var_desc ($var_name): Configured${NC}"
        else
            log "${RED}❌ $var_desc ($var_name): Not properly configured${NC}"
            ((errors++))
        fi
    done
    
    return $errors
}

# Function to validate application dependencies
validate_dependencies() {
    log "${BLUE}🐍 Validating Python dependencies...${NC}"
    
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        log "${GREEN}✅ Requirements file exists${NC}"
        
        # Check if virtual environment is active or Python packages are available
        if python3 -c "import flask, sqlalchemy, redis, prometheus_client" 2>/dev/null; then
            log "${GREEN}✅ Core Python dependencies available${NC}"
        else
            log "${YELLOW}⚠️  Some Python dependencies may not be installed${NC}"
            log "${BLUE}ℹ️  Run: pip install -r requirements.txt${NC}"
        fi
    else
        log "${RED}❌ Requirements file not found${NC}"
        return 1
    fi
}

# Function to validate SSL certificates
validate_ssl() {
    log "${BLUE}🔐 Validating SSL certificates...${NC}"
    
    local ssl_dir="$PROJECT_ROOT/ssl"
    local errors=0
    
    if [ -f "$ssl_dir/cert.pem" ]; then
        # Check certificate validity
        if openssl x509 -in "$ssl_dir/cert.pem" -text -noout >/dev/null 2>&1; then
            local expiry=$(openssl x509 -in "$ssl_dir/cert.pem" -enddate -noout | cut -d= -f2)
            log "${GREEN}✅ SSL certificate is valid (expires: $expiry)${NC}"
        else
            log "${RED}❌ SSL certificate is invalid${NC}"
            ((errors++))
        fi
    else
        log "${RED}❌ SSL certificate not found${NC}"
        ((errors++))
    fi
    
    if [ -f "$ssl_dir/key.pem" ]; then
        if openssl rsa -in "$ssl_dir/key.pem" -check -noout >/dev/null 2>&1; then
            log "${GREEN}✅ SSL private key is valid${NC}"
        else
            log "${RED}❌ SSL private key is invalid${NC}"
            ((errors++))
        fi
    else
        log "${RED}❌ SSL private key not found${NC}"
        ((errors++))
    fi
    
    return $errors
}

# Function to validate application code
validate_application() {
    log "${BLUE}🚀 Validating application code...${NC}"
    
    local errors=0
    
    # Check main application file
    if [ -f "$PROJECT_ROOT/app.py" ]; then
        if python3 -m py_compile "$PROJECT_ROOT/app.py" 2>/dev/null; then
            log "${GREEN}✅ Main application file compiles successfully${NC}"
        else
            log "${RED}❌ Main application file has syntax errors${NC}"
            ((errors++))
        fi
    else
        log "${RED}❌ Main application file (app.py) not found${NC}"
        ((errors++))
    fi
    
    # Check models
    if [ -d "$PROJECT_ROOT/models" ]; then
        log "${GREEN}✅ Models directory exists${NC}"
    else
        log "${RED}❌ Models directory not found${NC}"
        ((errors++))
    fi
    
    # Check templates
    if [ -d "$PROJECT_ROOT/templates" ]; then
        log "${GREEN}✅ Templates directory exists${NC}"
    else
        log "${RED}❌ Templates directory not found${NC}"
        ((errors++))
    fi
    
    # Check static files
    if [ -d "$PROJECT_ROOT/static" ]; then
        log "${GREEN}✅ Static files directory exists${NC}"
    else
        log "${RED}❌ Static files directory not found${NC}"
        ((errors++))
    fi
    
    return $errors
}

# Function to test application startup (dry run)
test_application_startup() {
    log "${BLUE}🧪 Testing application startup (dry run)...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables for testing
    export FLASK_ENV=production
    export SECRET_KEY="test-key-for-validation"
    export DATABASE_URL="sqlite:///test.db"
    
    # Try to import the application
    if python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from app import app
    print('✅ Application imports successfully')
except Exception as e:
    print(f'❌ Application import failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log "${GREEN}✅ Application startup test passed${NC}"
        return 0
    else
        log "${RED}❌ Application startup test failed${NC}"
        return 1
    fi
}

# Function to generate deployment summary
generate_summary() {
    log "${BLUE}📊 Deployment Validation Summary${NC}"
    log ""
    
    local total_errors=$1
    
    if [ $total_errors -eq 0 ]; then
        log "${GREEN}🎉 All validations passed! Deployment is ready.${NC}"
        log ""
        log "${BLUE}Next steps:${NC}"
        log "1. Ensure Docker is installed and running on target server"
        log "2. Copy all files to production server"
        log "3. Run: ./scripts/deployment/deploy-production.sh"
        log "4. Monitor logs and health checks"
        log ""
        log "${GREEN}Production deployment can proceed safely!${NC}"
    else
        log "${RED}❌ $total_errors validation(s) failed.${NC}"
        log ""
        log "${YELLOW}Please fix the issues above before deployment.${NC}"
        log ""
        log "${BLUE}Common fixes:${NC}"
        log "- Update .env.production with secure values"
        log "- Generate SSL certificates: ./scripts/deployment/setup-ssl.sh"
        log "- Install dependencies: pip install -r requirements.txt"
        log "- Check file permissions and paths"
    fi
}

# Main execution
main() {
    log "${GREEN}🔍 Starting Deployment Validation${NC}"
    log "Timestamp: $(date)"
    log "Project: Horse Racing Prediction System"
    log ""
    
    local total_errors=0
    
    # Run all validations
    validate_configs || ((total_errors += $?))
    echo ""
    
    validate_env_vars || ((total_errors += $?))
    echo ""
    
    validate_dependencies || ((total_errors += $?))
    echo ""
    
    validate_ssl || ((total_errors += $?))
    echo ""
    
    validate_application || ((total_errors += $?))
    echo ""
    
    test_application_startup || ((total_errors += $?))
    echo ""
    
    # Generate summary
    generate_summary $total_errors
    
    # Exit with appropriate code
    exit $total_errors
}

# Run main function
main "$@"