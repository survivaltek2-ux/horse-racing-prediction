-- PostgreSQL setup script for HRP migration
-- This script creates the required database and user for the migration

-- Create the user first
CREATE USER hrp_app WITH PASSWORD 'hrp_secure_password_2024';

-- Create the database
CREATE DATABASE hrp_database OWNER hrp_app;

-- Grant necessary privileges
GRANT ALL PRIVILEGES ON DATABASE hrp_database TO hrp_app;

-- Connect to the new database to set up additional permissions
\c hrp_database

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO hrp_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hrp_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hrp_app;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO hrp_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO hrp_app;

-- Display confirmation
SELECT 'PostgreSQL setup completed successfully!' as status;