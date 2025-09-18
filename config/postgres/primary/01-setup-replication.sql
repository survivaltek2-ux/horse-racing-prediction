-- Setup replication user and permissions
CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'repl_password';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE hrp_database TO replicator;
GRANT USAGE ON SCHEMA public TO replicator;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO replicator;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO replicator;

-- Create replication slot
SELECT pg_create_physical_replication_slot('replica_slot');

-- Configure pg_hba.conf for replication (this would typically be done via volume mount)
-- host replication replicator postgres-replica/32 md5