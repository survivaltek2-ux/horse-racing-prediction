# PostgreSQL Migration Guide

## Overview

This guide provides comprehensive instructions for migrating the Horse Racing Prediction application from SQLite to PostgreSQL. The migration includes schema conversion, data transfer, validation, and performance optimization.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Migration Architecture](#migration-architecture)
3. [Pre-Migration Checklist](#pre-migration-checklist)
4. [Migration Process](#migration-process)
5. [Post-Migration Validation](#post-migration-validation)
6. [Rollback Procedures](#rollback-procedures)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)

## Prerequisites

### System Requirements

- **PostgreSQL**: Version 13 or higher
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2x current database size for migration workspace
- **Network**: Stable connection for replication setup

### Required Python Packages

```bash
pip install -r requirements.txt
```

Required packages:
- `psycopg2-binary>=2.9.0`
- `sqlalchemy>=1.4.0`
- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `matplotlib>=3.4.0`
- `psutil>=5.8.0`

### PostgreSQL Setup

1. **Install PostgreSQL**:
   ```bash
   # macOS
   brew install postgresql
   
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   
   # CentOS/RHEL
   sudo yum install postgresql-server postgresql-contrib
   ```

2. **Start PostgreSQL Service**:
   ```bash
   # macOS
   brew services start postgresql
   
   # Linux
   sudo systemctl start postgresql
   sudo systemctl enable postgresql
   ```

3. **Create Database and User**:
   ```sql
   -- Connect as postgres user
   sudo -u postgres psql
   
   -- Create database
   CREATE DATABASE hrp_database;
   
   -- Create user
   CREATE USER hrp_user WITH PASSWORD 'your_secure_password';
   
   -- Grant privileges
   GRANT ALL PRIVILEGES ON DATABASE hrp_database TO hrp_user;
   ALTER USER hrp_user CREATEDB;
   
   -- Exit
   \q
   ```

## Migration Architecture

### Components Overview

```
Migration System
├── Schema Conversion
│   ├── postgresql_schema.sql
│   └── schema_converter.py
├── Data Migration
│   ├── data_migrator.py
│   └── validation/data_validator.py
├── Backup & Recovery
│   ├── backup/backup_manager.py
│   └── rollback/rollback_manager.py
├── Performance Testing
│   ├── performance/performance_tester.py
│   └── performance/benchmark_suite.py
├── Redundancy
│   └── redundancy/replication_setup.py
└── Verification
    └── verification/health_checker.py
```

### Migration Flow

```mermaid
graph TD
    A[Pre-Migration Backup] --> B[Schema Analysis]
    B --> C[PostgreSQL Schema Creation]
    C --> D[Data Migration]
    D --> E[Data Validation]
    E --> F[Performance Testing]
    F --> G[Replication Setup]
    G --> H[Final Validation]
    H --> I[Production Cutover]
    I --> J[Post-Migration Monitoring]
```

## Pre-Migration Checklist

### 1. Environment Preparation

- [ ] PostgreSQL installed and configured
- [ ] Database user created with appropriate permissions
- [ ] Network connectivity verified
- [ ] Backup storage space available (3x current database size)
- [ ] Migration scripts downloaded and tested

### 2. Current Database Assessment

```bash
# Run database analysis
python scripts/migration/schema/schema_converter.py --analyze-only

# Check current database size
python -c "
import os
from config.database_config import DatabaseConfig
print(f'Current database size: {os.path.getsize(DatabaseConfig.DATABASE_PATH) / 1024 / 1024:.2f} MB')
"
```

### 3. Configuration Setup

1. **Environment Variables**:
   ```bash
   export POSTGRES_URL="postgresql://hrp_user:password@localhost:5432/hrp_database"
   export MIGRATION_WORKSPACE="/path/to/migration/workspace"
   export BACKUP_RETENTION_DAYS=30
   ```

2. **Configuration Files**:
   - Update `config/database_config.py` with PostgreSQL settings
   - Verify connection strings in migration scripts
   - Set appropriate logging levels

### 4. Dependencies Check

```bash
# Verify all dependencies
python scripts/migration/validation/dependency_checker.py

# Test database connections
python scripts/migration/validation/connection_tester.py
```

## Migration Process

### Phase 1: Backup and Preparation

1. **Create Full Backup**:
   ```bash
   python scripts/migration/backup/backup_manager.py --create-full-backup
   ```

2. **Analyze Current Schema**:
   ```bash
   python scripts/migration/schema/schema_converter.py --analyze
   ```

3. **Generate Migration Plan**:
   ```bash
   python scripts/migration/schema/schema_converter.py --generate-plan
   ```

### Phase 2: Schema Migration

1. **Create PostgreSQL Schema**:
   ```bash
   # Apply schema to PostgreSQL
   psql -h localhost -U hrp_user -d hrp_database -f scripts/migration/schema/postgresql_schema.sql
   ```

2. **Verify Schema Creation**:
   ```bash
   python scripts/migration/validation/schema_validator.py --verify-schema
   ```

### Phase 3: Data Migration

1. **Start Data Migration**:
   ```bash
   python scripts/migration/data/data_migrator.py --migrate-all
   ```

2. **Monitor Progress**:
   ```bash
   # In another terminal
   tail -f logs/data_migration.log
   ```

3. **Validate Data Integrity**:
   ```bash
   python scripts/migration/validation/data_validator.py --full-validation
   ```

### Phase 4: Performance Testing

1. **Run Performance Tests**:
   ```bash
   python scripts/migration/performance/performance_tester.py --test-type all
   ```

2. **Run Benchmark Suite**:
   ```bash
   python scripts/migration/performance/benchmark_suite.py --benchmark-type all
   ```

3. **Analyze Results**:
   ```bash
   # Review performance reports
   ls -la scripts/migration/performance/reports/
   ```

### Phase 5: Redundancy Setup

1. **Configure Replication**:
   ```bash
   python scripts/migration/redundancy/replication_setup.py --setup-primary
   ```

2. **Setup Standby Server** (if applicable):
   ```bash
   python scripts/migration/redundancy/replication_setup.py --setup-standby
   ```

### Phase 6: Final Validation

1. **Run Health Checks**:
   ```bash
   python scripts/migration/verification/health_checker.py --comprehensive
   ```

2. **Application Testing**:
   ```bash
   # Test application with PostgreSQL
   python app.py --database postgresql
   ```

## Post-Migration Validation

### Data Integrity Checks

1. **Record Count Validation**:
   ```sql
   -- Compare record counts
   SELECT 'races' as table_name, COUNT(*) as count FROM races
   UNION ALL
   SELECT 'horses', COUNT(*) FROM horses
   UNION ALL
   SELECT 'predictions', COUNT(*) FROM predictions
   UNION ALL
   SELECT 'users', COUNT(*) FROM users;
   ```

2. **Data Quality Checks**:
   ```bash
   python scripts/migration/validation/data_validator.py --quality-checks
   ```

3. **Referential Integrity**:
   ```bash
   python scripts/migration/validation/data_validator.py --referential-integrity
   ```

### Performance Validation

1. **Query Performance**:
   ```bash
   python scripts/migration/performance/performance_tester.py --test-type query
   ```

2. **Load Testing**:
   ```bash
   python scripts/migration/performance/performance_tester.py --test-type load
   ```

3. **Concurrent User Testing**:
   ```bash
   python scripts/migration/performance/benchmark_suite.py --benchmark-type concurrent
   ```

### Application Testing

1. **Functional Testing**:
   - Test all CRUD operations
   - Verify API endpoints
   - Check user authentication
   - Validate prediction algorithms

2. **Integration Testing**:
   - Test external API integrations
   - Verify data synchronization
   - Check scheduled tasks

## Rollback Procedures

### Automatic Rollback

If migration fails during execution:

```bash
python scripts/migration/rollback/rollback_manager.py --auto-rollback
```

### Manual Rollback

1. **Stop Application**:
   ```bash
   # Stop application services
   pkill -f "python app.py"
   ```

2. **Restore from Backup**:
   ```bash
   python scripts/migration/backup/backup_manager.py --restore-latest
   ```

3. **Verify Restoration**:
   ```bash
   python scripts/migration/validation/data_validator.py --verify-restoration
   ```

### Partial Rollback

For rolling back specific components:

```bash
# Rollback schema only
python scripts/migration/rollback/rollback_manager.py --rollback-schema

# Rollback data only
python scripts/migration/rollback/rollback_manager.py --rollback-data

# Rollback configuration
python scripts/migration/rollback/rollback_manager.py --rollback-config
```

## Performance Optimization

### Index Optimization

1. **Analyze Query Patterns**:
   ```sql
   -- Enable query logging
   ALTER SYSTEM SET log_statement = 'all';
   ALTER SYSTEM SET log_min_duration_statement = 100;
   SELECT pg_reload_conf();
   ```

2. **Create Additional Indexes**:
   ```sql
   -- Example indexes for common queries
   CREATE INDEX CONCURRENTLY idx_predictions_confidence ON predictions(confidence);
   CREATE INDEX CONCURRENTLY idx_races_date ON races(race_date);
   CREATE INDEX CONCURRENTLY idx_horses_name ON horses(horse_name);
   ```

3. **Monitor Index Usage**:
   ```sql
   SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
   FROM pg_stat_user_indexes
   ORDER BY idx_scan DESC;
   ```

### Connection Optimization

1. **Connection Pooling**:
   ```python
   # Update database_config.py
   SQLALCHEMY_ENGINE_OPTIONS = {
       'pool_size': 20,
       'max_overflow': 30,
       'pool_pre_ping': True,
       'pool_recycle': 3600
   }
   ```

2. **Connection Monitoring**:
   ```sql
   SELECT count(*) as active_connections 
   FROM pg_stat_activity 
   WHERE state = 'active';
   ```

### Query Optimization

1. **Analyze Slow Queries**:
   ```sql
   SELECT query, mean_time, calls, total_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

2. **Update Statistics**:
   ```sql
   ANALYZE;
   ```

## Troubleshooting

### Common Issues

#### 1. Connection Errors

**Problem**: Cannot connect to PostgreSQL
```
psycopg2.OperationalError: could not connect to server
```

**Solution**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check configuration
sudo -u postgres psql -c "SHOW config_file;"

# Verify pg_hba.conf settings
sudo nano /etc/postgresql/13/main/pg_hba.conf
```

#### 2. Permission Errors

**Problem**: Permission denied for database operations
```
psycopg2.errors.InsufficientPrivilege: permission denied
```

**Solution**:
```sql
-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE hrp_database TO hrp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hrp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hrp_user;
```

#### 3. Data Type Conversion Errors

**Problem**: Data type mismatch during migration
```
psycopg2.errors.InvalidTextRepresentation: invalid input syntax
```

**Solution**:
```bash
# Run data type analysis
python scripts/migration/validation/data_type_analyzer.py

# Apply data type conversions
python scripts/migration/data/data_converter.py --fix-types
```

#### 4. Performance Issues

**Problem**: Slow query performance after migration

**Solution**:
```sql
-- Update table statistics
ANALYZE;

-- Check for missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;

-- Create missing indexes
CREATE INDEX CONCURRENTLY idx_table_column ON table_name(column_name);
```

### Diagnostic Commands

1. **Database Status**:
   ```bash
   python scripts/migration/verification/health_checker.py --status
   ```

2. **Migration Progress**:
   ```bash
   python scripts/migration/data/data_migrator.py --status
   ```

3. **Performance Metrics**:
   ```bash
   python scripts/migration/performance/performance_tester.py --quick-test
   ```

4. **Log Analysis**:
   ```bash
   # Check migration logs
   tail -f logs/migration.log

   # Check PostgreSQL logs
   sudo tail -f /var/log/postgresql/postgresql-13-main.log
   ```

## Maintenance

### Daily Tasks

1. **Monitor Performance**:
   ```bash
   python scripts/migration/verification/health_checker.py --daily-check
   ```

2. **Check Replication Status**:
   ```sql
   SELECT client_addr, state, sync_state 
   FROM pg_stat_replication;
   ```

3. **Backup Verification**:
   ```bash
   python scripts/migration/backup/backup_manager.py --verify-backups
   ```

### Weekly Tasks

1. **Performance Analysis**:
   ```bash
   python scripts/migration/performance/performance_tester.py --weekly-report
   ```

2. **Index Maintenance**:
   ```sql
   REINDEX DATABASE hrp_database;
   ```

3. **Statistics Update**:
   ```sql
   ANALYZE;
   ```

### Monthly Tasks

1. **Full Backup**:
   ```bash
   python scripts/migration/backup/backup_manager.py --monthly-backup
   ```

2. **Performance Benchmark**:
   ```bash
   python scripts/migration/performance/benchmark_suite.py --monthly-benchmark
   ```

3. **Cleanup Old Logs**:
   ```bash
   find logs/ -name "*.log" -mtime +30 -delete
   ```

### Monitoring Queries

1. **Database Size**:
   ```sql
   SELECT pg_size_pretty(pg_database_size('hrp_database')) as database_size;
   ```

2. **Table Sizes**:
   ```sql
   SELECT schemaname, tablename, 
          pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
   FROM pg_tables 
   WHERE schemaname = 'public'
   ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
   ```

3. **Active Connections**:
   ```sql
   SELECT datname, usename, client_addr, state, query_start
   FROM pg_stat_activity
   WHERE state = 'active';
   ```

## Security Considerations

### Database Security

1. **User Permissions**:
   ```sql
   -- Create read-only user for reporting
   CREATE USER hrp_readonly WITH PASSWORD 'readonly_password';
   GRANT CONNECT ON DATABASE hrp_database TO hrp_readonly;
   GRANT USAGE ON SCHEMA public TO hrp_readonly;
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO hrp_readonly;
   ```

2. **SSL Configuration**:
   ```bash
   # Enable SSL in postgresql.conf
   ssl = on
   ssl_cert_file = 'server.crt'
   ssl_key_file = 'server.key'
   ```

3. **Authentication**:
   ```bash
   # Update pg_hba.conf for secure authentication
   hostssl all all 0.0.0.0/0 md5
   ```

### Application Security

1. **Connection String Security**:
   ```python
   # Use environment variables
   import os
   POSTGRES_URL = os.getenv('POSTGRES_URL')
   ```

2. **SQL Injection Prevention**:
   ```python
   # Always use parameterized queries
   cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
   ```

## Support and Resources

### Documentation

- [PostgreSQL Official Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Migration Best Practices](https://wiki.postgresql.org/wiki/Migration)

### Monitoring Tools

- **pgAdmin**: Web-based PostgreSQL administration
- **pg_stat_statements**: Query performance monitoring
- **pg_stat_activity**: Real-time activity monitoring

### Emergency Contacts

- **Database Administrator**: [Your DBA contact]
- **System Administrator**: [Your SysAdmin contact]
- **Development Team**: [Your Dev team contact]

---

**Note**: This guide should be customized based on your specific environment and requirements. Always test migration procedures in a development environment before applying to production.