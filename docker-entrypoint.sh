#!/bin/bash
set -e

# Wait for database to be ready
wait_for_db() {
    echo "Waiting for database to be ready..."
    while ! nc -z ${DB_HOST:-postgres-master} ${DB_PORT:-5432}; do
        echo "Database not ready, waiting..."
        sleep 2
    done
    echo "Database is ready!"
}

# Wait for Redis to be ready
wait_for_redis() {
    echo "Waiting for Redis to be ready..."
    while ! nc -z ${REDIS_HOST:-redis-master} ${REDIS_PORT:-6379}; do
        echo "Redis not ready, waiting..."
        sleep 2
    done
    echo "Redis is ready!"
}

# Initialize database if needed
init_db() {
    echo "Initializing database..."
    python3 -c "
from app import app, db
with app.app_context():
    try:
        db.create_all()
        print('Database tables created successfully')
    except Exception as e:
        print(f'Database initialization error: {e}')
"
}

# Run database migrations if needed
run_migrations() {
    echo "Running database migrations..."
    # Add migration commands here if using Flask-Migrate
    # flask db upgrade
}

# Set up logging directories
setup_logging() {
    echo "Setting up logging directories..."
    mkdir -p logs/{errors,audit,deployment,performance,security}
    chmod 755 logs logs/*
}

# Main execution
main() {
    echo "Starting Horse Racing Prediction Application..."
    
    # Setup logging
    setup_logging
    
    # Wait for dependencies
    if [ "${WAIT_FOR_DB:-true}" = "true" ]; then
        wait_for_db
    fi
    
    if [ "${WAIT_FOR_REDIS:-true}" = "true" ]; then
        wait_for_redis
    fi
    
    # Initialize database
    if [ "${INIT_DB:-true}" = "true" ]; then
        init_db
    fi
    
    # Run migrations
    if [ "${RUN_MIGRATIONS:-false}" = "true" ]; then
        run_migrations
    fi
    
    echo "Starting application with command: $@"
    exec "$@"
}

# Run main function
main "$@"