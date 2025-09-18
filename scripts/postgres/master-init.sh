#!/bin/bash
set -e

# PostgreSQL Master Initialization Script
echo "Initializing PostgreSQL Master for replication..."

# Create replication user
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER $POSTGRES_REPLICATION_USER REPLICATION LOGIN CONNECTION LIMIT 3 ENCRYPTED PASSWORD '$POSTGRES_REPLICATION_PASSWORD';
    SELECT pg_create_physical_replication_slot('replica_slot_1');
    SELECT pg_create_physical_replication_slot('replica_slot_2');
EOSQL

# Configure PostgreSQL for replication
cat >> /var/lib/postgresql/data/postgresql.conf <<EOF

# Replication settings
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3
hot_standby = on
archive_mode = on
archive_command = 'test ! -f /var/lib/postgresql/data/archive/%f && cp %p /var/lib/postgresql/data/archive/%f'
synchronous_commit = on
synchronous_standby_names = 'replica1,replica2'

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
log_lock_waits = on

# Connection settings
max_connections = 200
listen_addresses = '*'
EOF

# Create archive directory
mkdir -p /var/lib/postgresql/data/archive
chown postgres:postgres /var/lib/postgresql/data/archive

echo "PostgreSQL Master initialization completed."