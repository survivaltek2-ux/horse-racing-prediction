#!/bin/bash

# =============================================================================
# Deployment Scripts Test and Validation
# Horse Racing Prediction Application
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_LOG_DIR="$PROJECT_ROOT/logs/tests"

# Create test log directory
mkdir -p "$TEST_LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test configuration
TEST_ENVIRONMENT="development"
VERBOSE=false
QUICK_TEST=false
SKIP_DESTRUCTIVE=false

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

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

test_start() {
    local test_name="$1"
    ((TOTAL_TESTS++))
    log "🧪 Testing: $test_name"
}

test_pass() {
    local test_name="$1"
    ((PASSED_TESTS++))
    log "✅ PASS: $test_name"
}

test_fail() {
    local test_name="$1"
    local reason="${2:-Unknown error}"
    ((FAILED_TESTS++))
    error "❌ FAIL: $test_name - $reason"
}

test_skip() {
    local test_name="$1"
    local reason="${2:-Skipped}"
    ((SKIPPED_TESTS++))
    warn "⏭️  SKIP: $test_name - $reason"
}

# =============================================================================
# Script Validation Functions
# =============================================================================

test_script_exists() {
    local script_name="$1"
    local script_path="$SCRIPT_DIR/$script_name"
    
    test_start "Script exists: $script_name"
    
    if [[ -f "$script_path" ]]; then
        test_pass "Script exists: $script_name"
        return 0
    else
        test_fail "Script exists: $script_name" "File not found: $script_path"
        return 1
    fi
}

test_script_executable() {
    local script_name="$1"
    local script_path="$SCRIPT_DIR/$script_name"
    
    test_start "Script executable: $script_name"
    
    if [[ -x "$script_path" ]]; then
        test_pass "Script executable: $script_name"
        return 0
    else
        test_fail "Script executable: $script_name" "Script is not executable"
        return 1
    fi
}

test_script_syntax() {
    local script_name="$1"
    local script_path="$SCRIPT_DIR/$script_name"
    
    test_start "Script syntax: $script_name"
    
    if bash -n "$script_path" 2>/dev/null; then
        test_pass "Script syntax: $script_name"
        return 0
    else
        test_fail "Script syntax: $script_name" "Syntax error detected"
        return 1
    fi
}

test_script_help() {
    local script_name="$1"
    local script_path="$SCRIPT_DIR/$script_name"
    
    test_start "Script help: $script_name"
    
    if "$script_path" --help &>/dev/null; then
        test_pass "Script help: $script_name"
        return 0
    else
        test_fail "Script help: $script_name" "Help option not working"
        return 1
    fi
}

# =============================================================================
# Environment Setup Tests
# =============================================================================

test_environment_script() {
    local script_name="setup-environment.sh"
    
    test_start "Environment script validation"
    
    # Test script exists and is executable
    test_script_exists "$script_name" || return 1
    test_script_executable "$script_name" || return 1
    test_script_syntax "$script_name" || return 1
    test_script_help "$script_name" || return 1
    
    # Test environment validation function
    test_start "Environment validation function"
    if bash -c "source '$SCRIPT_DIR/$script_name'; validate_environment development" &>/dev/null; then
        test_pass "Environment validation function"
    else
        test_fail "Environment validation function" "Function not working"
        return 1
    fi
    
    test_pass "Environment script validation"
}

test_environment_file_creation() {
    test_start "Environment file creation"
    
    local test_env_file="$PROJECT_ROOT/.env.test"
    
    # Remove test file if it exists
    rm -f "$test_env_file"
    
    # Test environment file creation
    if "$SCRIPT_DIR/setup-environment.sh" test create-env &>/dev/null; then
        if [[ -f "$test_env_file" ]]; then
            test_pass "Environment file creation"
            rm -f "$test_env_file"
            return 0
        else
            test_fail "Environment file creation" "Environment file not created"
            return 1
        fi
    else
        test_fail "Environment file creation" "Script execution failed"
        return 1
    fi
}

# =============================================================================
# Database Setup Tests
# =============================================================================

test_database_script() {
    local script_name="setup-database.sh"
    
    test_start "Database script validation"
    
    # Test script exists and is executable
    test_script_exists "$script_name" || return 1
    test_script_executable "$script_name" || return 1
    test_script_syntax "$script_name" || return 1
    test_script_help "$script_name" || return 1
    
    test_pass "Database script validation"
}

test_database_connection() {
    if [[ "$SKIP_DESTRUCTIVE" == "true" ]]; then
        test_skip "Database connection test" "Destructive tests skipped"
        return 0
    fi
    
    test_start "Database connection test"
    
    # Start database container for testing
    if docker-compose up -d postgres &>/dev/null; then
        sleep 5  # Wait for database to start
        
        # Test database connection
        if docker-compose exec -T postgres pg_isready -U "${POSTGRES_USER:-app_user}" &>/dev/null; then
            test_pass "Database connection test"
            return 0
        else
            test_fail "Database connection test" "Cannot connect to database"
            return 1
        fi
    else
        test_fail "Database connection test" "Cannot start database container"
        return 1
    fi
}

# =============================================================================
# SSL Setup Tests
# =============================================================================

test_ssl_script() {
    local script_name="ssl-setup.sh"
    
    test_start "SSL script validation"
    
    # Test script exists and is executable
    test_script_exists "$script_name" || return 1
    test_script_executable "$script_name" || return 1
    test_script_syntax "$script_name" || return 1
    test_script_help "$script_name" || return 1
    
    test_pass "SSL script validation"
}

test_self_signed_certificate() {
    test_start "Self-signed certificate creation"
    
    local ssl_dir="$PROJECT_ROOT/ssl_test"
    mkdir -p "$ssl_dir"
    
    # Test self-signed certificate creation
    if SSL_DIR="$ssl_dir" "$SCRIPT_DIR/ssl-setup.sh" development setup --domain localhost &>/dev/null; then
        if [[ -f "$ssl_dir/fullchain.pem" && -f "$ssl_dir/privkey.pem" ]]; then
            test_pass "Self-signed certificate creation"
            rm -rf "$ssl_dir"
            return 0
        else
            test_fail "Self-signed certificate creation" "Certificate files not created"
            rm -rf "$ssl_dir"
            return 1
        fi
    else
        test_fail "Self-signed certificate creation" "Script execution failed"
        rm -rf "$ssl_dir"
        return 1
    fi
}

# =============================================================================
# Monitoring Tests
# =============================================================================

test_monitoring_script() {
    local script_name="monitoring.sh"
    
    test_start "Monitoring script validation"
    
    # Test script exists and is executable
    test_script_exists "$script_name" || return 1
    test_script_executable "$script_name" || return 1
    test_script_syntax "$script_name" || return 1
    test_script_help "$script_name" || return 1
    
    test_pass "Monitoring script validation"
}

test_health_check_functions() {
    test_start "Health check functions"
    
    # Test individual health check functions
    local functions_to_test=(
        "check_disk_space"
        "check_memory_usage"
        "check_cpu_usage"
    )
    
    for func in "${functions_to_test[@]}"; do
        if bash -c "source '$SCRIPT_DIR/monitoring.sh'; $func" &>/dev/null; then
            log "✓ Function $func works"
        else
            test_fail "Health check functions" "Function $func failed"
            return 1
        fi
    done
    
    test_pass "Health check functions"
}

# =============================================================================
# Backup and Rollback Tests
# =============================================================================

test_backup_script() {
    local script_name="backup-rollback.sh"
    
    test_start "Backup script validation"
    
    # Test script exists and is executable
    test_script_exists "$script_name" || return 1
    test_script_executable "$script_name" || return 1
    test_script_syntax "$script_name" || return 1
    test_script_help "$script_name" || return 1
    
    test_pass "Backup script validation"
}

test_backup_creation() {
    if [[ "$SKIP_DESTRUCTIVE" == "true" ]]; then
        test_skip "Backup creation test" "Destructive tests skipped"
        return 0
    fi
    
    test_start "Backup creation test"
    
    local test_backup_dir="$PROJECT_ROOT/backups_test"
    mkdir -p "$test_backup_dir"
    
    # Test application backup creation
    if BACKUP_DIR="$test_backup_dir" "$SCRIPT_DIR/backup-rollback.sh" development backup --type application --name test_backup &>/dev/null; then
        if [[ -d "$test_backup_dir/test_backup" ]]; then
            test_pass "Backup creation test"
            rm -rf "$test_backup_dir"
            return 0
        else
            test_fail "Backup creation test" "Backup directory not created"
            rm -rf "$test_backup_dir"
            return 1
        fi
    else
        test_fail "Backup creation test" "Script execution failed"
        rm -rf "$test_backup_dir"
        return 1
    fi
}

# =============================================================================
# Main Deployment Script Tests
# =============================================================================

test_main_deployment_script() {
    local script_name="auto-deploy.sh"
    
    test_start "Main deployment script validation"
    
    # Test script exists and is executable
    test_script_exists "$script_name" || return 1
    test_script_executable "$script_name" || return 1
    test_script_syntax "$script_name" || return 1
    test_script_help "$script_name" || return 1
    
    test_pass "Main deployment script validation"
}

test_deployment_dry_run() {
    test_start "Deployment dry run"
    
    # Test dry run functionality
    if "$SCRIPT_DIR/auto-deploy.sh" development deploy --dry-run &>/dev/null; then
        test_pass "Deployment dry run"
        return 0
    else
        test_fail "Deployment dry run" "Dry run failed"
        return 1
    fi
}

# =============================================================================
# Integration Tests
# =============================================================================

test_docker_compose_validation() {
    test_start "Docker Compose validation"
    
    cd "$PROJECT_ROOT"
    
    # Test docker-compose file syntax
    if docker-compose config &>/dev/null; then
        test_pass "Docker Compose validation"
        return 0
    else
        test_fail "Docker Compose validation" "Invalid docker-compose.yml"
        return 1
    fi
}

test_environment_variables() {
    test_start "Environment variables validation"
    
    local env_files=(
        ".env.development"
        ".env.production"
    )
    
    for env_file in "${env_files[@]}"; do
        if [[ -f "$PROJECT_ROOT/$env_file" ]]; then
            # Check for required variables
            local required_vars=(
                "FLASK_ENV"
                "POSTGRES_USER"
                "POSTGRES_PASSWORD"
                "POSTGRES_DB"
            )
            
            for var in "${required_vars[@]}"; do
                if grep -q "^$var=" "$PROJECT_ROOT/$env_file"; then
                    log "✓ Variable $var found in $env_file"
                else
                    test_fail "Environment variables validation" "Missing variable $var in $env_file"
                    return 1
                fi
            done
        else
            warn "Environment file not found: $env_file"
        fi
    done
    
    test_pass "Environment variables validation"
}

test_nginx_configuration() {
    test_start "Nginx configuration validation"
    
    local nginx_conf="$PROJECT_ROOT/nginx/nginx.conf"
    
    if [[ -f "$nginx_conf" ]]; then
        # Test nginx configuration syntax
        if docker run --rm -v "$nginx_conf:/etc/nginx/nginx.conf:ro" nginx:alpine nginx -t &>/dev/null; then
            test_pass "Nginx configuration validation"
            return 0
        else
            test_fail "Nginx configuration validation" "Invalid nginx configuration"
            return 1
        fi
    else
        test_skip "Nginx configuration validation" "Nginx config not found"
        return 0
    fi
}

# =============================================================================
# Performance Tests
# =============================================================================

test_script_performance() {
    if [[ "$QUICK_TEST" == "true" ]]; then
        test_skip "Script performance test" "Quick test mode"
        return 0
    fi
    
    test_start "Script performance test"
    
    local scripts_to_test=(
        "setup-environment.sh"
        "monitoring.sh"
        "backup-rollback.sh"
    )
    
    for script in "${scripts_to_test[@]}"; do
        local start_time=$(date +%s.%N)
        
        # Run script help command
        if "$SCRIPT_DIR/$script" --help &>/dev/null; then
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc -l)
            
            if (( $(echo "$duration < 5.0" | bc -l) )); then
                log "✓ Script $script performance: ${duration}s (Good)"
            else
                warn "⚠ Script $script performance: ${duration}s (Slow)"
            fi
        else
            test_fail "Script performance test" "Script $script failed to run"
            return 1
        fi
    done
    
    test_pass "Script performance test"
}

# =============================================================================
# Security Tests
# =============================================================================

test_script_security() {
    test_start "Script security validation"
    
    local scripts=(
        "auto-deploy.sh"
        "setup-environment.sh"
        "setup-database.sh"
        "ssl-setup.sh"
        "monitoring.sh"
        "backup-rollback.sh"
    )
    
    for script in "${scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        
        # Check for potential security issues
        if grep -q "eval\|exec\|system\|passthru" "$script_path"; then
            warn "⚠ Potential security issue in $script: dangerous functions found"
        fi
        
        # Check for hardcoded credentials
        if grep -qE "(password|secret|key)\s*=\s*['\"][^'\"]+['\"]" "$script_path"; then
            test_fail "Script security validation" "Hardcoded credentials found in $script"
            return 1
        fi
        
        # Check for proper variable quoting
        if grep -qE '\$[A-Za-z_][A-Za-z0-9_]*[^"]' "$script_path"; then
            warn "⚠ Unquoted variables found in $script"
        fi
    done
    
    test_pass "Script security validation"
}

# =============================================================================
# Cleanup Tests
# =============================================================================

cleanup_test_artifacts() {
    log "Cleaning up test artifacts..."
    
    # Remove test files and directories
    local cleanup_items=(
        "$PROJECT_ROOT/.env.test"
        "$PROJECT_ROOT/ssl_test"
        "$PROJECT_ROOT/backups_test"
        "$PROJECT_ROOT/logs/tests/*.tmp"
    )
    
    for item in "${cleanup_items[@]}"; do
        if [[ -e "$item" ]]; then
            rm -rf "$item"
            log "✓ Cleaned up: $item"
        fi
    done
}

# =============================================================================
# Test Execution Functions
# =============================================================================

run_basic_tests() {
    log "Running basic validation tests..."
    
    # Script validation tests
    test_environment_script
    test_database_script
    test_ssl_script
    test_monitoring_script
    test_backup_script
    test_main_deployment_script
    
    # Configuration tests
    test_docker_compose_validation
    test_environment_variables
    test_nginx_configuration
    
    # Security tests
    test_script_security
}

run_functional_tests() {
    log "Running functional tests..."
    
    # Environment tests
    test_environment_file_creation
    
    # SSL tests
    test_self_signed_certificate
    
    # Monitoring tests
    test_health_check_functions
    
    # Deployment tests
    test_deployment_dry_run
    
    # Database tests (if not skipping destructive tests)
    if [[ "$SKIP_DESTRUCTIVE" != "true" ]]; then
        test_database_connection
        test_backup_creation
    fi
}

run_performance_tests() {
    if [[ "$QUICK_TEST" != "true" ]]; then
        log "Running performance tests..."
        test_script_performance
    fi
}

# =============================================================================
# Test Report
# =============================================================================

generate_test_report() {
    local report_file="$TEST_LOG_DIR/test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
Deployment Scripts Test Report
==============================

Test Execution: $(date)
Environment: $TEST_ENVIRONMENT
Quick Test Mode: $QUICK_TEST
Skip Destructive: $SKIP_DESTRUCTIVE

Test Results:
-------------
Total Tests: $TOTAL_TESTS
Passed: $PASSED_TESTS
Failed: $FAILED_TESTS
Skipped: $SKIPPED_TESTS

Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%

EOF
    
    log "Test report generated: $report_file"
    
    # Display summary
    echo ""
    log "=== TEST SUMMARY ==="
    log "Total Tests: $TOTAL_TESTS"
    log "Passed: $PASSED_TESTS"
    log "Failed: $FAILED_TESTS"
    log "Skipped: $SKIPPED_TESTS"
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        log "🎉 All tests passed!"
        return 0
    else
        error "❌ $FAILED_TESTS tests failed"
        return 1
    fi
}

# =============================================================================
# Usage and Help
# =============================================================================

show_usage() {
    cat << EOF
Deployment Scripts Test and Validation

Usage: $0 [environment] [options]

Environments:
    development     Test in development environment (default)
    staging         Test in staging environment
    production      Test in production environment

Options:
    --quick                 Run quick tests only (skip performance tests)
    --skip-destructive      Skip tests that modify system state
    --verbose               Enable verbose output
    --help                  Show this help message

Test Categories:
    basic                   Script validation and syntax checks
    functional              Functional testing of script features
    performance             Performance and timing tests
    security                Security validation tests

Examples:
    $0 development
    $0 staging --quick
    $0 production --skip-destructive
    $0 development --verbose

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
                TEST_ENVIRONMENT="$1"
                shift
                ;;
            --quick)
                QUICK_TEST=true
                shift
                ;;
            --skip-destructive)
                SKIP_DESTRUCTIVE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
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
    
    log "Starting deployment scripts test suite"
    log "Environment: $TEST_ENVIRONMENT"
    log "Quick test mode: $QUICK_TEST"
    log "Skip destructive tests: $SKIP_DESTRUCTIVE"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Make all scripts executable
    chmod +x "$SCRIPT_DIR"/*.sh
    
    # Run test suites
    run_basic_tests
    run_functional_tests
    run_performance_tests
    
    # Cleanup
    cleanup_test_artifacts
    
    # Generate report
    generate_test_report
}

# Trap for cleanup on exit
trap cleanup_test_artifacts EXIT

# Run main function with all arguments
main "$@"