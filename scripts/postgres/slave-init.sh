#!/bin/bash
set -e

# PostgreSQL Slave Initialization Script
echo "Initializing PostgreSQL Slave for replication..."

# Wait for master to be ready
until pg_isready -h $POSTGRES_MASTER_SERVICE -p 5432 -U postgres; do
    echo "Waiting for PostgreSQL master to be ready..."
    sleep 2
done

# Stop PostgreSQL if running
pg_ctl stop -D /var/lib/postgresql/data -m fast || true

# Remove existing data directory
rm -rf /var/lib/postgresql/data/*

# Create base backup from master
PGPASSWORD=$POSTGRES_REPLICATION_PASSWORD pg_basebackup -h $POSTGRES_MASTER_SERVICE -D /var/lib/postgresql/data -U replicator -v -P -W

# Create standby.signal file
touch /var/lib/postgresql/data/standby.signal

# Configure PostgreSQL for standby
cat >> /var/lib/postgresql/data/postgresql.conf <<EOF

# Standby settings
hot_standby = on
primary_conninfo = 'host=$POSTGRES_MASTER_SERVICE port=5432 user=replicator password=$POSTGRES_REPLICATION_PASSWORD application_name=replica1'
primary_slot_name = 'replica_slot_1'
restore_command = 'cp /var/lib/postgresql/data/archive/%f %p'
recovery_target_timeline = 'latest'

# Performance settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_statement = 'all'
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on

# Connection settings
max_connections = 200
listen_addresses = '*'
EOF

# Set proper permissions
chown -R postgres:postgres /var/lib/postgresql/data
chmod 700 /var/lib/postgresql/data

echo "PostgreSQL Slave initialization completed."