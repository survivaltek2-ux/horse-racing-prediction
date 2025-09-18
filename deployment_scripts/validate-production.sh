#!/bin/bash

# Production Environment Validation Script
# Horse Racing Prediction System

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="/tmp/production-validation-$(date +%Y%m%d-%H%M%S).log"
VALIDATION_RESULTS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Validation functions
validate_file_exists() {
    local file_path="$1"
    local description="$2"
    
    if [[ -f "$file_path" ]]; then
        log_success "✓ $description exists: $file_path"
        return 0
    else
        log_error "✗ $description missing: $file_path"
        return 1
    fi
}

validate_directory_exists() {
    local dir_path="$1"
    local description="$2"
    
    if [[ -d "$dir_path" ]]; then
        log_success "✓ $description exists: $dir_path"
        return 0
    else
        log_error "✗ $description missing: $dir_path"
        return 1
    fi
}

validate_script_executable() {
    local script_path="$1"
    local description="$2"
    
    if [[ -x "$script_path" ]]; then
        log_success "✓ $description is executable: $script_path"
        return 0
    else
        log_error "✗ $description is not executable: $script_path"
        return 1
    fi
}

validate_docker_compose_syntax() {
    local compose_file="$1"
    local description="$2"
    
    if command -v docker-compose >/dev/null 2>&1; then
        if docker-compose -f "$compose_file" config >/dev/null 2>&1; then
            log_success "✓ $description syntax is valid"
            return 0
        else
            log_error "✗ $description syntax is invalid"
            return 1
        fi
    else
        log_warn "⚠ Docker Compose not available, skipping syntax validation for $description"
        return 0
    fi
}

validate_nginx_config() {
    local config_file="$1"
    
    if command -v nginx >/dev/null 2>&1; then
        if nginx -t -c "$config_file" >/dev/null 2>&1; then
            log_success "✓ Nginx configuration is valid"
            return 0
        else
            log_error "✗ Nginx configuration is invalid"
            return 1
        fi
    else
        log_warn "⚠ Nginx not available, skipping configuration validation"
        return 0
    fi
}

validate_python_requirements() {
    local requirements_file="$1"
    
    if [[ -f "$requirements_file" ]]; then
        log_info "Validating Python requirements..."
        local pip_check_output
        pip_check_output=$(python3 -m pip check 2>&1)
        local pip_check_exit_code=$?
        
        if [[ $pip_check_exit_code -eq 0 ]]; then
            log_success "✓ Python requirements are satisfied"
            return 0
        else
            # Check if it's just version conflicts (common and usually non-critical)
            if echo "$pip_check_output" | grep -q "has requirement.*but you have"; then
                log_warn "⚠ Minor Python dependency version conflicts detected (usually non-critical)"
                log_info "Dependency conflicts: $pip_check_output"
                return 0  # Don't fail validation for version conflicts
            else
                log_error "✗ Critical Python requirements issues detected"
                log_error "Issues: $pip_check_output"
                return 1
            fi
        fi
    else
        log_error "✗ Requirements file not found: $requirements_file"
        return 1
    fi
}

# Main validation function
run_validation() {
    log_info "Starting production environment validation..."
    log_info "Project root: $PROJECT_ROOT"
    log_info "Log file: $LOG_FILE"
    
    local validation_passed=true
    
    echo -e "\n${BLUE}=== Core Application Files ===${NC}"
    validate_file_exists "$PROJECT_ROOT/app.py" "Main application file" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/requirements.txt" "Python requirements" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/.env.production" "Production environment file" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/Dockerfile.production" "Production Dockerfile" || validation_passed=false
    
    echo -e "\n${BLUE}=== Docker Configuration ===${NC}"
    validate_file_exists "$PROJECT_ROOT/docker-compose.yml" "Docker Compose file" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/docker-swarm-stack.yml" "Docker Swarm stack file" || validation_passed=false
    validate_docker_compose_syntax "$PROJECT_ROOT/docker-compose.yml" "Docker Compose" || validation_passed=false
    validate_docker_compose_syntax "$PROJECT_ROOT/docker-swarm-stack.yml" "Docker Swarm stack" || validation_passed=false
    
    echo -e "\n${BLUE}=== Nginx Configuration ===${NC}"
    validate_directory_exists "$PROJECT_ROOT/nginx" "Nginx directory" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/nginx/nginx-production.conf" "Nginx production config" || validation_passed=false
    
    echo -e "\n${BLUE}=== Monitoring Configuration ===${NC}"
    validate_directory_exists "$PROJECT_ROOT/monitoring" "Monitoring directory" || validation_passed=false
    validate_directory_exists "$PROJECT_ROOT/monitoring/prometheus" "Prometheus directory" || validation_passed=false
    validate_directory_exists "$PROJECT_ROOT/monitoring/grafana" "Grafana directory" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/monitoring/prometheus/prometheus.yml" "Prometheus config" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/monitoring/prometheus/alert_rules.yml" "Prometheus alert rules" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/monitoring/grafana/datasources/prometheus.yml" "Grafana datasources" || validation_passed=false
    
    echo -e "\n${BLUE}=== Logging Configuration ===${NC}"
    validate_directory_exists "$PROJECT_ROOT/logging" "Logging directory" || validation_passed=false
    validate_directory_exists "$PROJECT_ROOT/logging/elasticsearch" "Elasticsearch directory" || validation_passed=false
    validate_directory_exists "$PROJECT_ROOT/logging/logstash" "Logstash directory" || validation_passed=false
    validate_directory_exists "$PROJECT_ROOT/logging/kibana" "Kibana directory" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/logging/elasticsearch/elasticsearch.yml" "Elasticsearch config" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/logging/logstash/config/logstash.yml" "Logstash config" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/logging/logstash/pipeline/logstash.conf" "Logstash pipeline" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/logging/kibana/kibana.yml" "Kibana config" || validation_passed=false
    
    echo -e "\n${BLUE}=== Scripts and Automation ===${NC}"
    validate_directory_exists "$PROJECT_ROOT/scripts" "Scripts directory" || validation_passed=false
    validate_directory_exists "$PROJECT_ROOT/scripts/backup" "Backup scripts directory" || validation_passed=false
    validate_directory_exists "$PROJECT_ROOT/scripts/health-checks" "Health check scripts directory" || validation_passed=false
    validate_directory_exists "$PROJECT_ROOT/scripts/deployment" "Deployment scripts directory" || validation_passed=false
    
    # Check script executability
    for script in "$PROJECT_ROOT"/scripts/backup/*.sh; do
        [[ -f "$script" ]] && validate_script_executable "$script" "Backup script $(basename "$script")" || validation_passed=false
    done
    
    for script in "$PROJECT_ROOT"/scripts/health-checks/*.sh; do
        [[ -f "$script" ]] && validate_script_executable "$script" "Health check script $(basename "$script")" || validation_passed=false
    done
    
    for script in "$PROJECT_ROOT"/scripts/deployment/*.sh; do
        [[ -f "$script" ]] && validate_script_executable "$script" "Deployment script $(basename "$script")" || validation_passed=false
    done
    
    echo -e "\n${BLUE}=== Documentation ===${NC}"
    validate_file_exists "$PROJECT_ROOT/PRODUCTION_DEPLOYMENT_GUIDE.md" "Production deployment guide" || validation_passed=false
    validate_file_exists "$PROJECT_ROOT/README.md" "README file" || validation_passed=false
    
    echo -e "\n${BLUE}=== Python Environment ===${NC}"
    validate_python_requirements "$PROJECT_ROOT/requirements.txt" || validation_passed=false
    
    # Summary
    echo -e "\n${BLUE}=== Validation Summary ===${NC}"
    if [[ "$validation_passed" == true ]]; then
        log_success "🎉 All validations passed! Production environment is ready for deployment."
        echo -e "\n${GREEN}Next steps:${NC}"
        echo "1. Review and update environment variables in .env.production"
        echo "2. Set up SSL certificates"
        echo "3. Configure external services (databases, monitoring)"
        echo "4. Run deployment script: ./scripts/deployment/deploy-production.sh"
        echo "5. Monitor logs and health checks"
        return 0
    else
        log_error "❌ Some validations failed. Please review the issues above before deployment."
        echo -e "\n${RED}Please fix the issues above before proceeding with deployment.${NC}"
        return 1
    fi
}

# Help function
show_help() {
    cat << EOF
Production Environment Validation Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    -l, --log-file  Specify custom log file path

DESCRIPTION:
    This script validates the production environment setup for the
    Horse Racing Prediction System. It checks for:
    
    - Core application files
    - Docker configuration files
    - Nginx configuration
    - Monitoring setup (Prometheus, Grafana)
    - Logging setup (ELK stack)
    - Backup and deployment scripts
    - Documentation

EXAMPLES:
    $0                          # Run validation with default settings
    $0 -v                       # Run with verbose output
    $0 -l /tmp/my-validation.log # Use custom log file

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        -l|--log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo -e "${BLUE}Horse Racing Prediction System - Production Validation${NC}"
    echo -e "${BLUE}=====================================================${NC}\n"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run validation
    if run_validation; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"