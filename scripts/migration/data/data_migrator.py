#!/usr/bin/env python3
"""
PostgreSQL Data Migration Script
Migrates data from SQLite to PostgreSQL with comprehensive validation
"""

import os
import sys
import json
import logging
import hashlib
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.database_config import DatabaseConfig
from models.sqlalchemy_models import db, APICredentials, Race, Horse, Prediction, User

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MigrationStats:
    """Statistics for migration process"""
    table_name: str
    source_count: int = 0
    migrated_count: int = 0
    failed_count: int = 0
    validation_errors: List[str] = None
    start_time: datetime = None
    end_time: datetime = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.start_time is None:
            self.start_time = datetime.now()
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        if self.source_count == 0:
            return 100.0
        return (self.migrated_count / self.source_count) * 100

class PostgreSQLDataMigrator:
    """Handles data migration from SQLite to PostgreSQL"""
    
    def __init__(self, source_db_uri: str, target_db_uri: str, batch_size: int = 1000):
        self.source_db_uri = source_db_uri
        self.target_db_uri = target_db_uri
        self.batch_size = batch_size
        
        # Database connections
        self.source_engine = create_engine(source_db_uri)
        self.target_engine = create_engine(target_db_uri)
        
        # PostgreSQL connection for direct operations
        self.pg_conn = None
        self.pg_cursor = None
        
        # Migration statistics
        self.migration_stats: Dict[str, MigrationStats] = {}
        
        # Data validation rules
        self.validation_rules = {
            'users': {
                'required_fields': ['username', 'email'],
                'unique_fields': ['username', 'email'],
                'data_types': {
                    'id': int,
                    'username': str,
                    'email': str,
                    'is_active': bool
                }
            },
            'horses': {
                'required_fields': ['name'],
                'data_types': {
                    'id': int,
                    'name': str,
                    'age': int,
                    'sex': str,
                    'rating': (int, float),
                    'wins': int,
                    'places': int,
                    'runs': int
                },
                'constraints': {
                    'age': lambda x: 0 < x <= 30,
                    'wins': lambda x: x >= 0,
                    'places': lambda x: x >= 0,
                    'runs': lambda x: x >= 0,
                    'rating': lambda x: 0 <= x <= 200
                }
            },
            'races': {
                'required_fields': ['name', 'date', 'track'],
                'data_types': {
                    'id': int,
                    'name': str,
                    'date': (str, datetime),
                    'track': str,
                    'distance': (int, float),
                    'prize_money': (int, float)
                },
                'constraints': {
                    'distance': lambda x: x > 0,
                    'prize_money': lambda x: x >= 0
                }
            },
            'predictions': {
                'required_fields': ['race_id', 'horse_id'],
                'data_types': {
                    'id': int,
                    'race_id': int,
                    'horse_id': int,
                    'predicted_position': int,
                    'confidence': float
                },
                'constraints': {
                    'predicted_position': lambda x: x > 0,
                    'confidence': lambda x: 0.0 <= x <= 1.0
                }
            }
        }
    
    def connect_postgresql(self) -> bool:
        """Establish PostgreSQL connection"""
        try:
            # Parse connection string
            import urllib.parse as urlparse
            parsed = urlparse.urlparse(self.target_db_uri)
            
            self.pg_conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],  # Remove leading slash
                user=parsed.username,
                password=parsed.password
            )
            self.pg_cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            logger.info("PostgreSQL connection established")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def disconnect_postgresql(self):
        """Close PostgreSQL connection"""
        if self.pg_cursor:
            self.pg_cursor.close()
        if self.pg_conn:
            self.pg_conn.close()
        logger.info("PostgreSQL connection closed")
    
    def validate_record(self, table_name: str, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single record against defined rules"""
        errors = []
        
        if table_name not in self.validation_rules:
            return True, errors
        
        rules = self.validation_rules[table_name]
        
        # Check required fields
        for field in rules.get('required_fields', []):
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Check data types
        for field, expected_type in rules.get('data_types', {}).items():
            if field in record and record[field] is not None:
                if isinstance(expected_type, tuple):
                    if not isinstance(record[field], expected_type):
                        errors.append(f"Invalid type for {field}: expected {expected_type}, got {type(record[field])}")
                else:
                    if not isinstance(record[field], expected_type):
                        errors.append(f"Invalid type for {field}: expected {expected_type}, got {type(record[field])}")
        
        # Check constraints
        for field, constraint_func in rules.get('constraints', {}).items():
            if field in record and record[field] is not None:
                try:
                    if not constraint_func(record[field]):
                        errors.append(f"Constraint violation for {field}: {record[field]}")
                except Exception as e:
                    errors.append(f"Constraint check error for {field}: {e}")
        
        return len(errors) == 0, errors
    
    def calculate_record_checksum(self, record: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity verification"""
        # Sort keys for consistent hashing
        sorted_items = sorted(record.items())
        record_str = json.dumps(sorted_items, default=str, sort_keys=True)
        return hashlib.md5(record_str.encode()).hexdigest()
    
    def get_source_data(self, table_name: str) -> List[Dict[str, Any]]:
        """Retrieve data from source database"""
        logger.info(f"Retrieving data from source table: {table_name}")
        
        try:
            with self.source_engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {table_name}"))
                records = [dict(row._mapping) for row in result]
                logger.info(f"Retrieved {len(records)} records from {table_name}")
                return records
                
        except Exception as e:
            logger.error(f"Error retrieving data from {table_name}: {e}")
            return []
    
    def prepare_record_for_postgresql(self, table_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare record for PostgreSQL insertion"""
        prepared_record = record.copy()
        
        # Handle datetime fields
        datetime_fields = ['created_at', 'updated_at', 'date', 'timestamp']
        for field in datetime_fields:
            if field in prepared_record and prepared_record[field]:
                if isinstance(prepared_record[field], str):
                    try:
                        # Parse string datetime
                        prepared_record[field] = datetime.fromisoformat(
                            prepared_record[field].replace('Z', '+00:00')
                        )
                    except:
                        # If parsing fails, set to current time
                        prepared_record[field] = datetime.now(timezone.utc)
                elif not isinstance(prepared_record[field], datetime):
                    prepared_record[field] = datetime.now(timezone.utc)
        
        # Handle boolean fields
        boolean_fields = ['is_active', 'is_admin', 'is_verified']
        for field in boolean_fields:
            if field in prepared_record:
                if isinstance(prepared_record[field], str):
                    prepared_record[field] = prepared_record[field].lower() in ('true', '1', 'yes')
                elif prepared_record[field] is None:
                    prepared_record[field] = False
        
        # Handle numeric fields
        numeric_fields = ['age', 'wins', 'places', 'runs', 'rating', 'distance', 'prize_money']
        for field in numeric_fields:
            if field in prepared_record and prepared_record[field] is not None:
                try:
                    if field in ['distance', 'prize_money', 'rating']:
                        prepared_record[field] = float(prepared_record[field])
                    else:
                        prepared_record[field] = int(prepared_record[field])
                except (ValueError, TypeError):
                    prepared_record[field] = 0
        
        # Add checksum for integrity verification
        prepared_record['_migration_checksum'] = self.calculate_record_checksum(record)
        
        return prepared_record
    
    def insert_batch_to_postgresql(self, table_name: str, records: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """Insert batch of records to PostgreSQL"""
        if not records:
            return 0, []
        
        errors = []
        inserted_count = 0
        
        try:
            # Prepare column names and placeholders
            columns = list(records[0].keys())
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join(columns)
            
            # Prepare insert statement
            insert_sql = f"""
                INSERT INTO {table_name} ({column_names})
                VALUES ({placeholders})
                ON CONFLICT (id) DO UPDATE SET
                {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'id'])}
            """
            
            # Prepare data tuples
            data_tuples = []
            for record in records:
                data_tuple = tuple(record[col] for col in columns)
                data_tuples.append(data_tuple)
            
            # Execute batch insert
            execute_batch(self.pg_cursor, insert_sql, data_tuples, page_size=self.batch_size)
            self.pg_conn.commit()
            inserted_count = len(records)
            
            logger.info(f"Inserted {inserted_count} records into {table_name}")
            
        except Exception as e:
            self.pg_conn.rollback()
            error_msg = f"Batch insert failed for {table_name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            # Try individual inserts for error isolation
            inserted_count, individual_errors = self._insert_records_individually(table_name, records)
            errors.extend(individual_errors)
        
        return inserted_count, errors
    
    def _insert_records_individually(self, table_name: str, records: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """Insert records individually to isolate errors"""
        inserted_count = 0
        errors = []
        
        for i, record in enumerate(records):
            try:
                columns = list(record.keys())
                placeholders = ', '.join(['%s'] * len(columns))
                column_names = ', '.join(columns)
                
                insert_sql = f"""
                    INSERT INTO {table_name} ({column_names})
                    VALUES ({placeholders})
                    ON CONFLICT (id) DO UPDATE SET
                    {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'id'])}
                """
                
                data_tuple = tuple(record[col] for col in columns)
                self.pg_cursor.execute(insert_sql, data_tuple)
                self.pg_conn.commit()
                inserted_count += 1
                
            except Exception as e:
                self.pg_conn.rollback()
                error_msg = f"Failed to insert record {i} in {table_name}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
        
        return inserted_count, errors
    
    def migrate_table(self, table_name: str) -> MigrationStats:
        """Migrate a single table"""
        logger.info(f"Starting migration for table: {table_name}")
        
        stats = MigrationStats(table_name)
        self.migration_stats[table_name] = stats
        
        try:
            # Get source data
            source_records = self.get_source_data(table_name)
            stats.source_count = len(source_records)
            
            if stats.source_count == 0:
                logger.info(f"No data found in source table: {table_name}")
                stats.end_time = datetime.now()
                return stats
            
            # Process records in batches
            valid_records = []
            
            for record in source_records:
                # Validate record
                is_valid, validation_errors = self.validate_record(table_name, record)
                
                if is_valid:
                    # Prepare record for PostgreSQL
                    prepared_record = self.prepare_record_for_postgresql(table_name, record)
                    valid_records.append(prepared_record)
                else:
                    stats.failed_count += 1
                    stats.validation_errors.extend(validation_errors)
                    logger.warning(f"Validation failed for record in {table_name}: {validation_errors}")
                
                # Process batch when it reaches batch_size
                if len(valid_records) >= self.batch_size:
                    inserted_count, errors = self.insert_batch_to_postgresql(table_name, valid_records)
                    stats.migrated_count += inserted_count
                    stats.validation_errors.extend(errors)
                    valid_records = []
            
            # Process remaining records
            if valid_records:
                inserted_count, errors = self.insert_batch_to_postgresql(table_name, valid_records)
                stats.migrated_count += inserted_count
                stats.validation_errors.extend(errors)
            
            stats.end_time = datetime.now()
            
            logger.info(f"Migration completed for {table_name}: "
                       f"{stats.migrated_count}/{stats.source_count} records migrated "
                       f"({stats.success_rate:.1f}% success rate)")
            
        except Exception as e:
            stats.end_time = datetime.now()
            error_msg = f"Migration failed for {table_name}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            stats.validation_errors.append(error_msg)
        
        return stats
    
    def verify_migration(self, table_name: str) -> Dict[str, Any]:
        """Verify migration integrity for a table"""
        logger.info(f"Verifying migration for table: {table_name}")
        
        verification_result = {
            'table_name': table_name,
            'source_count': 0,
            'target_count': 0,
            'checksum_matches': 0,
            'checksum_mismatches': 0,
            'missing_records': [],
            'integrity_check': False
        }
        
        try:
            # Get source count
            with self.source_engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table_name}"))
                verification_result['source_count'] = result.scalar()
            
            # Get target count
            self.pg_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            verification_result['target_count'] = self.pg_cursor.fetchone()[0]
            
            # Verify checksums (if available)
            try:
                self.pg_cursor.execute(f"SELECT id, _migration_checksum FROM {table_name}")
                target_checksums = {row['id']: row['_migration_checksum'] for row in self.pg_cursor.fetchall()}
                
                with self.source_engine.connect() as conn:
                    result = conn.execute(text(f"SELECT * FROM {table_name}"))
                    for row in result:
                        record = dict(row._mapping)
                        record_id = record['id']
                        expected_checksum = self.calculate_record_checksum(record)
                        
                        if record_id in target_checksums:
                            if target_checksums[record_id] == expected_checksum:
                                verification_result['checksum_matches'] += 1
                            else:
                                verification_result['checksum_mismatches'] += 1
                        else:
                            verification_result['missing_records'].append(record_id)
            
            except Exception as e:
                logger.warning(f"Checksum verification failed for {table_name}: {e}")
            
            # Overall integrity check
            verification_result['integrity_check'] = (
                verification_result['source_count'] == verification_result['target_count'] and
                verification_result['checksum_mismatches'] == 0 and
                len(verification_result['missing_records']) == 0
            )
            
            logger.info(f"Verification completed for {table_name}: "
                       f"Integrity check {'PASSED' if verification_result['integrity_check'] else 'FAILED'}")
            
        except Exception as e:
            logger.error(f"Verification failed for {table_name}: {e}")
        
        return verification_result
    
    def generate_migration_report(self) -> Dict[str, Any]:
        """Generate comprehensive migration report"""
        total_source = sum(stats.source_count for stats in self.migration_stats.values())
        total_migrated = sum(stats.migrated_count for stats in self.migration_stats.values())
        total_failed = sum(stats.failed_count for stats in self.migration_stats.values())
        
        report = {
            'migration_timestamp': datetime.now().isoformat(),
            'source_database': self.source_db_uri,
            'target_database': self.target_db_uri,
            'batch_size': self.batch_size,
            'summary': {
                'total_tables': len(self.migration_stats),
                'total_source_records': total_source,
                'total_migrated_records': total_migrated,
                'total_failed_records': total_failed,
                'overall_success_rate': (total_migrated / total_source * 100) if total_source > 0 else 0
            },
            'table_details': {}
        }
        
        for table_name, stats in self.migration_stats.items():
            report['table_details'][table_name] = {
                'source_count': stats.source_count,
                'migrated_count': stats.migrated_count,
                'failed_count': stats.failed_count,
                'success_rate': stats.success_rate,
                'duration_seconds': stats.duration,
                'validation_errors': stats.validation_errors[:10],  # Limit to first 10 errors
                'total_validation_errors': len(stats.validation_errors)
            }
        
        return report
    
    def get_existing_source_tables(self) -> List[str]:
        """Get list of tables that exist in the source database"""
        try:
            with self.source_engine.connect() as conn:
                # For SQLite, get table names from sqlite_master
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
                existing_tables = [row[0] for row in result.fetchall()]
                logger.info(f"Found existing tables in source database: {existing_tables}")
                return existing_tables
        except Exception as e:
            logger.error(f"Failed to get source table list: {e}")
            return []

    def migrate_all_tables(self, tables: List[str] = None) -> bool:
        """Migrate all specified tables"""
        if tables is None:
            tables = ['users', 'horses', 'races', 'predictions', 'api_credentials']
        
        # Filter tables to only include those that exist in source database
        existing_tables = self.get_existing_source_tables()
        tables_to_migrate = [table for table in tables if table in existing_tables]
        
        if not tables_to_migrate:
            logger.warning("No tables found to migrate")
            return True
        
        logger.info(f"Starting migration for existing tables: {tables_to_migrate}")
        
        if not self.connect_postgresql():
            return False
        
        try:
            migration_order = ['users', 'horses', 'races', 'api_credentials', 'predictions']
            ordered_tables = [table for table in migration_order if table in tables_to_migrate]
            
            # Add any remaining tables
            for table in tables_to_migrate:
                if table not in ordered_tables:
                    ordered_tables.append(table)
            
            # Migrate tables in order
            for table_name in ordered_tables:
                self.migrate_table(table_name)
            
            # Generate and save report
            report = self.generate_migration_report()
            report_path = "scripts/migration/data/migration_report.json"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Migration completed. Report saved to: {report_path}")
            logger.info(f"Overall success rate: {report['summary']['overall_success_rate']:.1f}%")
            
            return report['summary']['overall_success_rate'] > 95.0
            
        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            return False
        
        finally:
            self.disconnect_postgresql()

def main():
    """Main migration process"""
    logger.info("Starting PostgreSQL data migration...")
    
    try:
        # Configuration
        source_db_uri = DatabaseConfig.SQLALCHEMY_DATABASE_URI
        target_db_uri = os.getenv('POSTGRES_URL', 'postgresql://hrp_user:password@localhost:5432/hrp_database')
        batch_size = int(os.getenv('MIGRATION_BATCH_SIZE', '1000'))
        
        # Initialize migrator
        migrator = PostgreSQLDataMigrator(source_db_uri, target_db_uri, batch_size)
        
        # Run migration
        success = migrator.migrate_all_tables()
        
        if success:
            logger.info("Data migration completed successfully!")
            return True
        else:
            logger.error("Data migration completed with errors!")
            return False
            
    except Exception as e:
        logger.error(f"Data migration failed: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)