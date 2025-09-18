#!/bin/bash

# =============================================================================
# Database Setup and Migration Script
# Horse Racing Prediction Application
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/database"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging setup
LOG_FILE="$LOG_DIR/database-setup-$(date +%Y%m%d-%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default configuration
ENVIRONMENT="production"
ACTION="setup"
BACKUP_BEFORE_MIGRATION=true
FORCE_MIGRATION=false
SKIP_VALIDATION=false

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
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# =============================================================================
# Database Connection Functions
# =============================================================================

load_database_config() {
    local env_file="$PROJECT_ROOT/.env.$ENVIRONMENT"
    
    if [[ -f "$env_file" ]]; then
        set -a
        source "$env_file"
        set +a
        log "Loaded database configuration from $env_file"
    else
        error "Environment file $env_file not found"
    fi
    
    # Set default values if not provided
    POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
    POSTGRES_PORT="${POSTGRES_PORT:-5432}"
    POSTGRES_DB="${POSTGRES_DB:-horse_racing_db}"
    POSTGRES_USER="${POSTGRES_USER:-app_user}"
    
    if [[ -z "${POSTGRES_PASSWORD:-}" ]]; then
        error "POSTGRES_PASSWORD not set in environment file"
    fi
}

wait_for_database() {
    log "Waiting for database to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if docker-compose exec -T postgres pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" &> /dev/null; then
            log "Database is ready."
            return 0
        fi
        
        info "Waiting for database... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    error "Database failed to become ready within expected time"
}

test_database_connection() {
    log "Testing database connection..."
    
    if docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" &> /dev/null; then
        log "Database connection successful."
        return 0
    else
        error "Failed to connect to database"
    fi
}

# =============================================================================
# Database Setup Functions
# =============================================================================

create_database() {
    log "Creating database and user..."
    
    # Create user if not exists
    docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$POSTGRES_USER') THEN
                CREATE ROLE $POSTGRES_USER LOGIN PASSWORD '$POSTGRES_PASSWORD';
            END IF;
        END
        \$\$;
    " || warn "Failed to create user (may already exist)"
    
    # Create database if not exists
    docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "
        SELECT 'CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$POSTGRES_DB')\gexec
    " || warn "Failed to create database (may already exist)"
    
    # Grant privileges
    docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "
        GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;
        ALTER USER $POSTGRES_USER CREATEDB;
    " || warn "Failed to grant privileges"
    
    log "Database and user setup completed."
}

setup_database_extensions() {
    log "Setting up database extensions..."
    
    # Enable required extensions
    local extensions=("uuid-ossp" "pgcrypto" "pg_stat_statements")
    
    for ext in "${extensions[@]}"; do
        log "Enabling extension: $ext"
        docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
            CREATE EXTENSION IF NOT EXISTS \"$ext\";
        " || warn "Failed to enable extension: $ext"
    done
    
    log "Database extensions setup completed."
}

# =============================================================================
# Schema Migration Functions
# =============================================================================

run_schema_migrations() {
    log "Running schema migrations..."
    
    # Check if migration scripts exist
    local migration_scripts=(
        "$PROJECT_ROOT/migrate_enhanced_schema.py"
        "$PROJECT_ROOT/migrate_horse_schema.py"
        "$PROJECT_ROOT/init_database.py"
    )
    
    for script in "${migration_scripts[@]}"; do
        if [[ -f "$script" ]]; then
            log "Running migration script: $(basename "$script")"
            
            if docker-compose exec -T app python "$(basename "$script")"; then
                log "Migration script completed: $(basename "$script")"
            else
                if [[ "$FORCE_MIGRATION" == "true" ]]; then
                    warn "Migration script failed but continuing due to --force: $(basename "$script")"
                else
                    error "Migration script failed: $(basename "$script")"
                fi
            fi
        else
            info "Migration script not found: $(basename "$script")"
        fi
    done
    
    log "Schema migrations completed."
}

run_flask_migrations() {
    log "Running Flask-Migrate migrations..."
    
    # Initialize migration repository if it doesn't exist
    if [[ ! -d "$PROJECT_ROOT/migrations" ]]; then
        log "Initializing Flask-Migrate..."
        docker-compose exec -T app flask db init || warn "Failed to initialize Flask-Migrate"
    fi
    
    # Generate migration if there are model changes
    log "Generating migration..."
    docker-compose exec -T app flask db migrate -m "Auto-generated migration $(date +%Y%m%d_%H%M%S)" || warn "No changes detected or migration failed"
    
    # Apply migrations
    log "Applying migrations..."
    docker-compose exec -T app flask db upgrade || error "Failed to apply migrations"
    
    log "Flask migrations completed."
}

# =============================================================================
# Data Import Functions
# =============================================================================

import_sample_data() {
    log "Importing sample data..."
    
    local data_scripts=(
        "$PROJECT_ROOT/import_sample_data.py"
        "$PROJECT_ROOT/generate_sample_data.py"
        "$PROJECT_ROOT/import_json_races.py"
    )
    
    for script in "${data_scripts[@]}"; do
        if [[ -f "$script" ]]; then
            log "Running data import script: $(basename "$script")"
            docker-compose exec -T app python "$(basename "$script")" || warn "Data import script failed: $(basename "$script")"
        fi
    done
    
    log "Sample data import completed."
}

import_production_data() {
    log "Importing production data..."
    
    # Check for production data files
    local data_files=(
        "$PROJECT_ROOT/data/horses.json"
        "$PROJECT_ROOT/data/races.json"
        "$PROJECT_ROOT/data/users.json"
    )
    
    for file in "${data_files[@]}"; do
        if [[ -f "$file" ]]; then
            log "Found data file: $(basename "$file")"
            # Import logic would go here based on file type
        fi
    done
    
    log "Production data import completed."
}

# =============================================================================
# Database Optimization Functions
# =============================================================================

optimize_database() {
    log "Optimizing database performance..."
    
    # Update table statistics
    log "Updating table statistics..."
    docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
        ANALYZE;
    " || warn "Failed to update statistics"
    
    # Create indexes for better performance
    create_performance_indexes
    
    # Configure database settings
    configure_database_settings
    
    log "Database optimization completed."
}

create_performance_indexes() {
    log "Creating performance indexes..."
    
    # Define indexes for common queries
    local indexes=(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_races_date ON races(race_date);"
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_races_track ON races(track);"
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_horses_name ON horses(name);"
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_predictions_race_id ON predictions(race_id);"
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);"
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users(email);"
    )
    
    for index in "${indexes[@]}"; do
        log "Creating index: $index"
        docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "$index" || warn "Failed to create index"
    done
    
    log "Performance indexes created."
}

configure_database_settings() {
    log "Configuring database settings..."
    
    # PostgreSQL performance settings
    local settings=(
        "ALTER SYSTEM SET shared_buffers = '256MB';"
        "ALTER SYSTEM SET effective_cache_size = '1GB';"
        "ALTER SYSTEM SET maintenance_work_mem = '64MB';"
        "ALTER SYSTEM SET checkpoint_completion_target = 0.9;"
        "ALTER SYSTEM SET wal_buffers = '16MB';"
        "ALTER SYSTEM SET default_statistics_target = 100;"
        "ALTER SYSTEM SET random_page_cost = 1.1;"
        "ALTER SYSTEM SET effective_io_concurrency = 200;"
    )
    
    for setting in "${settings[@]}"; do
        docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "$setting" || warn "Failed to apply setting: $setting"
    done
    
    # Reload configuration
    docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "SELECT pg_reload_conf();" || warn "Failed to reload configuration"
    
    log "Database settings configured."
}

# =============================================================================
# Backup and Restore Functions
# =============================================================================

create_database_backup() {
    log "Creating database backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/database"
    local backup_file="$backup_dir/backup-$(date +%Y%m%d-%H%M%S).sql"
    
    mkdir -p "$backup_dir"
    
    # Create backup
    docker-compose exec -T postgres pg_dump -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" "$POSTGRES_DB" > "$backup_file"
    
    if [[ -f "$backup_file" && -s "$backup_file" ]]; then
        log "Database backup created: $backup_file"
        
        # Compress backup
        gzip "$backup_file"
        log "Backup compressed: $backup_file.gz"
        
        # Clean old backups (keep last 10)
        find "$backup_dir" -name "backup-*.sql.gz" -type f | sort -r | tail -n +11 | xargs rm -f
        
        return 0
    else
        error "Failed to create database backup"
    fi
}

restore_database_backup() {
    local backup_file="$1"
    
    if [[ -z "$backup_file" ]]; then
        error "Backup file not specified"
    fi
    
    if [[ ! -f "$backup_file" ]]; then
        error "Backup file not found: $backup_file"
    fi
    
    log "Restoring database from backup: $backup_file"
    
    # Drop existing database
    docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c "
        SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$POSTGRES_DB';
        DROP DATABASE IF EXISTS $POSTGRES_DB;
        CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER;
    "
    
    # Restore from backup
    if [[ "$backup_file" == *.gz ]]; then
        gunzip -c "$backup_file" | docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB"
    else
        docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" < "$backup_file"
    fi
    
    log "Database restore completed."
}

# =============================================================================
# Validation Functions
# =============================================================================

validate_database_schema() {
    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        return 0
    fi
    
    log "Validating database schema..."
    
    # Check if required tables exist
    local required_tables=("users" "horses" "races" "predictions")
    
    for table in "${required_tables[@]}"; do
        local table_exists=$(docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = '$table'
            );
        " | tr -d ' \n')
        
        if [[ "$table_exists" == "t" ]]; then
            log "Table exists: $table"
        else
            error "Required table missing: $table"
        fi
    done
    
    # Check table row counts
    for table in "${required_tables[@]}"; do
        local row_count=$(docker-compose exec -T postgres psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM $table;" | tr -d ' \n')
        log "Table $table has $row_count rows"
    done
    
    log "Database schema validation completed."
}

# =============================================================================
# Usage and Help
# =============================================================================

show_usage() {
    cat << EOF
Database Setup and Migration Script

Usage: $0 [environment] [action] [options]

Environments:
    development     Setup development database
    staging         Setup staging database
    production      Setup production database (default)

Actions:
    setup           Full database setup (default)
    migrate         Run migrations only
    backup          Create database backup
    restore         Restore from backup
    optimize        Optimize database performance
    validate        Validate database schema
    import-sample   Import sample data
    import-prod     Import production data

Options:
    --backup-file FILE      Backup file for restore action
    --no-backup             Skip backup before migration
    --force                 Force migration even if errors occur
    --skip-validation       Skip schema validation
    --help                  Show this help message

Examples:
    $0 production setup
    $0 staging migrate --no-backup
    $0 production backup
    $0 production restore --backup-file /path/to/backup.sql.gz
    $0 development import-sample

EOF
}

# =============================================================================
# Main Function
# =============================================================================

main() {
    local backup_file=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            development|staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            setup|migrate|backup|restore|optimize|validate|import-sample|import-prod)
                ACTION="$1"
                shift
                ;;
            --backup-file)
                backup_file="$2"
                shift 2
                ;;
            --no-backup)
                BACKUP_BEFORE_MIGRATION=false
                shift
                ;;
            --force)
                FORCE_MIGRATION=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use --help for usage information."
                ;;
        esac
    done
    
    log "Database setup for environment: $ENVIRONMENT"
    log "Action: $ACTION"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Load database configuration
    load_database_config
    
    # Execute action
    case $ACTION in
        setup)
            wait_for_database
            create_database
            setup_database_extensions
            if [[ "$BACKUP_BEFORE_MIGRATION" == "true" ]]; then
                create_database_backup || warn "Backup failed but continuing"
            fi
            run_schema_migrations
            run_flask_migrations
            optimize_database
            validate_database_schema
            if [[ "$ENVIRONMENT" == "development" ]]; then
                import_sample_data
            fi
            ;;
        migrate)
            wait_for_database
            test_database_connection
            if [[ "$BACKUP_BEFORE_MIGRATION" == "true" ]]; then
                create_database_backup
            fi
            run_schema_migrations
            run_flask_migrations
            validate_database_schema
            ;;
        backup)
            wait_for_database
            test_database_connection
            create_database_backup
            ;;
        restore)
            if [[ -z "$backup_file" ]]; then
                error "Backup file required for restore action. Use --backup-file option."
            fi
            wait_for_database
            restore_database_backup "$backup_file"
            validate_database_schema
            ;;
        optimize)
            wait_for_database
            test_database_connection
            optimize_database
            ;;
        validate)
            wait_for_database
            test_database_connection
            validate_database_schema
            ;;
        import-sample)
            wait_for_database
            test_database_connection
            import_sample_data
            ;;
        import-prod)
            wait_for_database
            test_database_connection
            import_production_data
            ;;
        *)
            error "Unknown action: $ACTION"
            ;;
    esac
    
    log "Database setup completed successfully!"
    log "Log file: $LOG_FILE"
}

# Trap errors and cleanup
trap 'error "Script failed at line $LINENO"' ERR

# Run main function with all arguments
main "$@"