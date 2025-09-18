#!/bin/bash
set -e

# Wait for primary to be ready
until pg_isready -h postgres-primary -p 5432 -U postgres; do
  echo "Waiting for primary database to be ready..."
  sleep 2
done

# Check if this is the first run
if [ ! -s "$PGDATA/PG_VERSION" ]; then
    echo "Setting up streaming replication..."
    
    # Remove any existing data
    rm -rf $PGDATA/*
    
    # Create base backup from primary
    pg_basebackup -h postgres-primary -D $PGDATA -U replicator -v -P -W
    
    # Create recovery configuration
    cat > $PGDATA/postgresql.conf << EOF
# Replica configuration
hot_standby = on
max_standby_streaming_delay = 30s
wal_receiver_status_interval = 10s
hot_standby_feedback = on
EOF

    # Create standby signal file
    touch $PGDATA/standby.signal
    
    # Create primary connection info
    cat > $PGDATA/postgresql.auto.conf << EOF
primary_conninfo = 'host=postgres-primary port=5432 user=replicator password=repl_password application_name=replica'
primary_slot_name = 'replica_slot'
EOF

    echo "Replica setup completed"
fi

# Start PostgreSQL
exec postgres