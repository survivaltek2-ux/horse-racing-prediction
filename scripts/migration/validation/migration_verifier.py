#!/usr/bin/env python3
"""
PostgreSQL Migration Verification Script

This script provides comprehensive verification procedures to ensure
the migration from SQLite to PostgreSQL was successful and complete.

Features:
- Data completeness verification
- Schema structure validation
- Performance comparison
- Functional testing
- Data integrity checks
- Application compatibility verification
- Migration success certification

Author: Migration Team
Date: January 2025
Version: 1.0
"""

import os
import sys
import json
import time
import logging
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
import pandas as pd


@dataclass
class VerificationResult:
    """Verification result data structure"""
    test_name: str
    status: str  # 'PASS', 'FAIL', 'WARN'
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float
    expected: Any = None
    actual: Any = None


@dataclass
class MigrationSummary:
    """Migration verification summary"""
    overall_status: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    data_integrity_score: float
    performance_improvement: float
    migration_timestamp: datetime
    verification_timestamp: datetime


class MigrationVerifier:
    """Comprehensive migration verification system"""
    
    def __init__(self, config_file: str = None):
        """Initialize migration verifier"""
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.results: List[VerificationResult] = []
        self.start_time = datetime.now()
        
        # Database connections
        self.sqlite_engine = None
        self.postgres_engine = None
        
        self._setup_database_connections()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'databases': {
                'sqlite': {
                    'path': 'database.db'
                },
                'postgresql': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'hrp_database',
                    'username': 'hrp_app',
                    'password': os.getenv('POSTGRES_PASSWORD', 'password')
                }
            },
            'verification': {
                'data_completeness': True,
                'schema_structure': True,
                'data_integrity': True,
                'performance_comparison': True,
                'functional_testing': True,
                'application_compatibility': True
            },
            'thresholds': {
                'data_completeness_percent': 99.9,
                'performance_improvement_percent': 20,
                'max_acceptable_variance_percent': 0.1
            },
            'tables': ['races', 'horses', 'predictions', 'users', 'api_credentials'],
            'sample_size': 1000,
            'performance_queries': [
                "SELECT COUNT(*) FROM races",
                "SELECT COUNT(*) FROM horses",
                "SELECT COUNT(*) FROM predictions",
                """SELECT r.race_name, h.horse_name, p.confidence 
                   FROM races r 
                   JOIN predictions p ON r.id = p.race_id 
                   JOIN horses h ON p.horse_id = h.id 
                   LIMIT 100"""
            ]
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('migration_verifier')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = 'logs/verification'
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        log_file = os.path.join(log_dir, f'verification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_database_connections(self):
        """Setup database connections"""
        try:
            # SQLite connection
            sqlite_path = self.config['databases']['sqlite']['path']
            if os.path.exists(sqlite_path):
                self.sqlite_engine = create_engine(f'sqlite:///{sqlite_path}')
                self.logger.info(f"SQLite connection established: {sqlite_path}")
            else:
                self.logger.warning(f"SQLite database not found: {sqlite_path}")
            
            # PostgreSQL connection
            pg_config = self.config['databases']['postgresql']
            pg_url = f"postgresql://{pg_config['username']}:{pg_config['password']}@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
            self.postgres_engine = create_engine(pg_url, pool_pre_ping=True)
            self.logger.info("PostgreSQL connection established")
            
        except Exception as e:
            self.logger.error(f"Failed to setup database connections: {e}")
            raise
    
    def _record_result(self, test_name: str, status: str, message: str,
                      details: Dict[str, Any] = None, duration_ms: float = 0,
                      expected: Any = None, actual: Any = None):
        """Record a verification result"""
        result = VerificationResult(
            test_name=test_name,
            status=status,
            message=message,
            details=details or {},
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            expected=expected,
            actual=actual
        )
        self.results.append(result)
        
        # Log the result
        log_level = {
            'PASS': logging.INFO,
            'WARN': logging.WARNING,
            'FAIL': logging.ERROR
        }.get(status, logging.INFO)
        
        self.logger.log(log_level, f"{test_name}: {status} - {message}")
    
    def verify_data_completeness(self) -> bool:
        """Verify data completeness between SQLite and PostgreSQL"""
        if not self.sqlite_engine:
            self._record_result(
                'Data Completeness',
                'WARN',
                'SQLite database not available for comparison',
                {}
            )
            return True
        
        start_time = time.time()
        
        try:
            completeness_results = {}
            
            for table in self.config['tables']:
                # Get row counts from both databases
                with self.sqlite_engine.connect() as sqlite_conn:
                    sqlite_count = sqlite_conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
                
                with self.postgres_engine.connect() as pg_conn:
                    pg_count = pg_conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
                
                # Calculate completeness percentage
                if sqlite_count > 0:
                    completeness_percent = (pg_count / sqlite_count) * 100
                else:
                    completeness_percent = 100 if pg_count == 0 else 0
                
                completeness_results[table] = {
                    'sqlite_count': sqlite_count,
                    'postgresql_count': pg_count,
                    'completeness_percent': completeness_percent,
                    'missing_records': max(0, sqlite_count - pg_count)
                }
            
            # Calculate overall completeness
            total_sqlite = sum(r['sqlite_count'] for r in completeness_results.values())
            total_postgres = sum(r['postgresql_count'] for r in completeness_results.values())
            
            overall_completeness = (total_postgres / total_sqlite * 100) if total_sqlite > 0 else 100
            
            duration = (time.time() - start_time) * 1000
            threshold = self.config['thresholds']['data_completeness_percent']
            
            # Check for incomplete tables
            incomplete_tables = [
                table for table, result in completeness_results.items()
                if result['completeness_percent'] < threshold
            ]
            
            if incomplete_tables:
                self._record_result(
                    'Data Completeness',
                    'FAIL',
                    f'Incomplete data migration for tables: {", ".join(incomplete_tables)}',
                    {
                        'overall_completeness_percent': overall_completeness,
                        'table_results': completeness_results,
                        'incomplete_tables': incomplete_tables
                    },
                    duration,
                    threshold,
                    overall_completeness
                )
                return False
            
            if overall_completeness < threshold:
                self._record_result(
                    'Data Completeness',
                    'FAIL',
                    f'Overall data completeness ({overall_completeness:.2f}%) below threshold ({threshold}%)',
                    {
                        'overall_completeness_percent': overall_completeness,
                        'table_results': completeness_results
                    },
                    duration,
                    threshold,
                    overall_completeness
                )
                return False
            
            self._record_result(
                'Data Completeness',
                'PASS',
                f'Data migration complete ({overall_completeness:.2f}% completeness)',
                {
                    'overall_completeness_percent': overall_completeness,
                    'table_results': completeness_results,
                    'total_records_migrated': total_postgres
                },
                duration,
                threshold,
                overall_completeness
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Data Completeness',
                'FAIL',
                f'Data completeness verification failed: {str(e)}',
                {'error': str(e)},
                duration
            )
            return False
    
    def verify_schema_structure(self) -> bool:
        """Verify schema structure in PostgreSQL"""
        start_time = time.time()
        
        try:
            schema_info = {}
            
            with self.postgres_engine.connect() as conn:
                # Get table information
                tables_query = """
                    SELECT table_name, table_type
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """
                tables = conn.execute(text(tables_query)).fetchall()
                schema_info['tables'] = [{'name': row[0], 'type': row[1]} for row in tables]
                
                # Get column information
                columns_query = """
                    SELECT table_name, column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position
                """
                columns = conn.execute(text(columns_query)).fetchall()
                
                # Group columns by table
                table_columns = {}
                for row in columns:
                    table_name = row[0]
                    if table_name not in table_columns:
                        table_columns[table_name] = []
                    table_columns[table_name].append({
                        'name': row[1],
                        'type': row[2],
                        'nullable': row[3] == 'YES',
                        'default': row[4]
                    })
                
                schema_info['columns'] = table_columns
                
                # Get constraint information
                constraints_query = """
                    SELECT table_name, constraint_name, constraint_type
                    FROM information_schema.table_constraints
                    WHERE table_schema = 'public'
                    ORDER BY table_name, constraint_name
                """
                constraints = conn.execute(text(constraints_query)).fetchall()
                
                # Group constraints by table
                table_constraints = {}
                for row in constraints:
                    table_name = row[0]
                    if table_name not in table_constraints:
                        table_constraints[table_name] = []
                    table_constraints[table_name].append({
                        'name': row[1],
                        'type': row[2]
                    })
                
                schema_info['constraints'] = table_constraints
                
                # Get index information
                indexes_query = """
                    SELECT schemaname, tablename, indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    ORDER BY tablename, indexname
                """
                indexes = conn.execute(text(indexes_query)).fetchall()
                
                # Group indexes by table
                table_indexes = {}
                for row in indexes:
                    table_name = row[1]
                    if table_name not in table_indexes:
                        table_indexes[table_name] = []
                    table_indexes[table_name].append({
                        'name': row[2],
                        'definition': row[3]
                    })
                
                schema_info['indexes'] = table_indexes
            
            # Verify expected tables exist
            expected_tables = set(self.config['tables'])
            actual_tables = set(table['name'] for table in schema_info['tables'])
            missing_tables = expected_tables - actual_tables
            extra_tables = actual_tables - expected_tables
            
            # Verify primary keys exist
            tables_without_pk = []
            for table in expected_tables:
                if table in schema_info['constraints']:
                    pk_constraints = [c for c in schema_info['constraints'][table] if c['type'] == 'PRIMARY KEY']
                    if not pk_constraints:
                        tables_without_pk.append(table)
                else:
                    tables_without_pk.append(table)
            
            # Verify indexes exist
            tables_without_indexes = []
            for table in expected_tables:
                if table not in schema_info['indexes'] or len(schema_info['indexes'][table]) <= 1:  # Only primary key index
                    tables_without_indexes.append(table)
            
            duration = (time.time() - start_time) * 1000
            
            issues = []
            if missing_tables:
                issues.append(f"Missing tables: {', '.join(missing_tables)}")
            if tables_without_pk:
                issues.append(f"Tables without primary keys: {', '.join(tables_without_pk)}")
            
            warnings = []
            if extra_tables:
                warnings.append(f"Extra tables found: {', '.join(extra_tables)}")
            if tables_without_indexes:
                warnings.append(f"Tables with minimal indexing: {', '.join(tables_without_indexes)}")
            
            if issues:
                self._record_result(
                    'Schema Structure',
                    'FAIL',
                    f'Schema structure issues: {"; ".join(issues)}',
                    schema_info,
                    duration
                )
                return False
            
            if warnings:
                self._record_result(
                    'Schema Structure',
                    'WARN',
                    f'Schema structure warnings: {"; ".join(warnings)}',
                    schema_info,
                    duration
                )
                return False
            
            table_count = len(actual_tables)
            constraint_count = sum(len(constraints) for constraints in schema_info['constraints'].values())
            index_count = sum(len(indexes) for indexes in schema_info['indexes'].values())
            
            self._record_result(
                'Schema Structure',
                'PASS',
                f'Schema structure valid ({table_count} tables, {constraint_count} constraints, {index_count} indexes)',
                schema_info,
                duration
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Schema Structure',
                'FAIL',
                f'Schema structure verification failed: {str(e)}',
                {'error': str(e)},
                duration
            )
            return False
    
    def verify_data_integrity(self) -> bool:
        """Verify data integrity in PostgreSQL"""
        start_time = time.time()
        
        try:
            integrity_results = {}
            
            with self.postgres_engine.connect() as conn:
                # Check referential integrity
                foreign_key_violations = []
                
                # Check predictions -> races
                orphaned_predictions_race = conn.execute(text("""
                    SELECT COUNT(*) FROM predictions p
                    LEFT JOIN races r ON p.race_id = r.id
                    WHERE r.id IS NULL
                """)).fetchone()[0]
                
                if orphaned_predictions_race > 0:
                    foreign_key_violations.append(f"predictions->races: {orphaned_predictions_race} orphaned records")
                
                # Check predictions -> horses
                orphaned_predictions_horse = conn.execute(text("""
                    SELECT COUNT(*) FROM predictions p
                    LEFT JOIN horses h ON p.horse_id = h.id
                    WHERE h.id IS NULL
                """)).fetchone()[0]
                
                if orphaned_predictions_horse > 0:
                    foreign_key_violations.append(f"predictions->horses: {orphaned_predictions_horse} orphaned records")
                
                integrity_results['foreign_key_violations'] = foreign_key_violations
                
                # Check data constraints
                constraint_violations = []
                
                # Check confidence values
                invalid_confidence = conn.execute(text("""
                    SELECT COUNT(*) FROM predictions
                    WHERE confidence < 0 OR confidence > 1
                """)).fetchone()[0]
                
                if invalid_confidence > 0:
                    constraint_violations.append(f"invalid confidence values: {invalid_confidence} records")
                
                # Check horse ages
                invalid_ages = conn.execute(text("""
                    SELECT COUNT(*) FROM horses
                    WHERE age < 2 OR age > 20
                """)).fetchone()[0]
                
                if invalid_ages > 0:
                    constraint_violations.append(f"invalid horse ages: {invalid_ages} records")
                
                # Check for NULL values in required fields
                null_race_names = conn.execute(text("""
                    SELECT COUNT(*) FROM races WHERE race_name IS NULL
                """)).fetchone()[0]
                
                if null_race_names > 0:
                    constraint_violations.append(f"NULL race names: {null_race_names} records")
                
                null_horse_names = conn.execute(text("""
                    SELECT COUNT(*) FROM horses WHERE horse_name IS NULL
                """)).fetchone()[0]
                
                if null_horse_names > 0:
                    constraint_violations.append(f"NULL horse names: {null_horse_names} records")
                
                integrity_results['constraint_violations'] = constraint_violations
                
                # Check for duplicate records
                duplicate_checks = []
                
                # Check for duplicate horses
                duplicate_horses = conn.execute(text("""
                    SELECT horse_name, COUNT(*) as count
                    FROM horses
                    GROUP BY horse_name
                    HAVING COUNT(*) > 1
                """)).fetchall()
                
                if duplicate_horses:
                    duplicate_checks.append(f"duplicate horses: {len(duplicate_horses)} names")
                
                # Check for duplicate predictions
                duplicate_predictions = conn.execute(text("""
                    SELECT race_id, horse_id, algorithm_version, COUNT(*) as count
                    FROM predictions
                    GROUP BY race_id, horse_id, algorithm_version
                    HAVING COUNT(*) > 1
                """)).fetchall()
                
                if duplicate_predictions:
                    duplicate_checks.append(f"duplicate predictions: {len(duplicate_predictions)} combinations")
                
                integrity_results['duplicate_violations'] = duplicate_checks
            
            # Calculate integrity score
            total_violations = (
                len(integrity_results['foreign_key_violations']) +
                len(integrity_results['constraint_violations']) +
                len(integrity_results['duplicate_violations'])
            )
            
            integrity_score = max(0, 100 - (total_violations * 10))  # Deduct 10 points per violation type
            
            duration = (time.time() - start_time) * 1000
            
            if total_violations > 0:
                all_violations = (
                    integrity_results['foreign_key_violations'] +
                    integrity_results['constraint_violations'] +
                    integrity_results['duplicate_violations']
                )
                
                self._record_result(
                    'Data Integrity',
                    'FAIL',
                    f'Data integrity violations found: {"; ".join(all_violations)}',
                    {
                        **integrity_results,
                        'integrity_score': integrity_score,
                        'total_violations': total_violations
                    },
                    duration,
                    100,
                    integrity_score
                )
                return False
            
            self._record_result(
                'Data Integrity',
                'PASS',
                f'Data integrity verified (Score: {integrity_score}/100)',
                {
                    **integrity_results,
                    'integrity_score': integrity_score,
                    'total_violations': total_violations
                },
                duration,
                100,
                integrity_score
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Data Integrity',
                'FAIL',
                f'Data integrity verification failed: {str(e)}',
                {'error': str(e)},
                duration
            )
            return False
    
    def verify_performance_comparison(self) -> bool:
        """Compare performance between SQLite and PostgreSQL"""
        if not self.sqlite_engine:
            self._record_result(
                'Performance Comparison',
                'WARN',
                'SQLite database not available for performance comparison',
                {}
            )
            return True
        
        start_time = time.time()
        
        try:
            performance_results = {}
            queries = self.config['performance_queries']
            
            for i, query in enumerate(queries):
                query_name = f"query_{i+1}"
                
                # Test SQLite performance
                sqlite_times = []
                for _ in range(3):  # Run 3 times for average
                    query_start = time.time()
                    with self.sqlite_engine.connect() as conn:
                        result = conn.execute(text(query)).fetchall()
                    sqlite_times.append((time.time() - query_start) * 1000)
                
                sqlite_avg = sum(sqlite_times) / len(sqlite_times)
                
                # Test PostgreSQL performance
                postgres_times = []
                for _ in range(3):  # Run 3 times for average
                    query_start = time.time()
                    with self.postgres_engine.connect() as conn:
                        result = conn.execute(text(query)).fetchall()
                    postgres_times.append((time.time() - query_start) * 1000)
                
                postgres_avg = sum(postgres_times) / len(postgres_times)
                
                # Calculate improvement
                improvement_percent = ((sqlite_avg - postgres_avg) / sqlite_avg) * 100 if sqlite_avg > 0 else 0
                
                performance_results[query_name] = {
                    'query': query[:100] + '...' if len(query) > 100 else query,
                    'sqlite_avg_ms': sqlite_avg,
                    'postgresql_avg_ms': postgres_avg,
                    'improvement_percent': improvement_percent,
                    'sqlite_times': sqlite_times,
                    'postgresql_times': postgres_times
                }
            
            # Calculate overall performance improvement
            total_sqlite_time = sum(r['sqlite_avg_ms'] for r in performance_results.values())
            total_postgres_time = sum(r['postgresql_avg_ms'] for r in performance_results.values())
            
            overall_improvement = ((total_sqlite_time - total_postgres_time) / total_sqlite_time) * 100 if total_sqlite_time > 0 else 0
            
            duration = (time.time() - start_time) * 1000
            threshold = self.config['thresholds']['performance_improvement_percent']
            
            # Check for performance regressions
            regressions = [
                name for name, result in performance_results.items()
                if result['improvement_percent'] < 0
            ]
            
            if regressions:
                self._record_result(
                    'Performance Comparison',
                    'WARN',
                    f'Performance regressions detected in queries: {", ".join(regressions)}',
                    {
                        'overall_improvement_percent': overall_improvement,
                        'query_results': performance_results,
                        'regressions': regressions
                    },
                    duration,
                    threshold,
                    overall_improvement
                )
                return False
            
            if overall_improvement < threshold:
                self._record_result(
                    'Performance Comparison',
                    'WARN',
                    f'Overall performance improvement ({overall_improvement:.2f}%) below target ({threshold}%)',
                    {
                        'overall_improvement_percent': overall_improvement,
                        'query_results': performance_results
                    },
                    duration,
                    threshold,
                    overall_improvement
                )
                return False
            
            self._record_result(
                'Performance Comparison',
                'PASS',
                f'Performance improved by {overall_improvement:.2f}% (Target: {threshold}%)',
                {
                    'overall_improvement_percent': overall_improvement,
                    'query_results': performance_results,
                    'total_queries_tested': len(queries)
                },
                duration,
                threshold,
                overall_improvement
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Performance Comparison',
                'FAIL',
                f'Performance comparison failed: {str(e)}',
                {'error': str(e)},
                duration
            )
            return False
    
    def verify_functional_testing(self) -> bool:
        """Verify functional operations work correctly"""
        start_time = time.time()
        
        try:
            functional_results = {}
            
            with self.postgres_engine.connect() as conn:
                # Test basic CRUD operations
                test_table = 'races'  # Use races table for testing
                
                # Test INSERT
                insert_start = time.time()
                test_race_id = None
                try:
                    result = conn.execute(text(f"""
                        INSERT INTO {test_table} (race_name, race_date, track_name, race_type, distance, surface)
                        VALUES ('Test Race Verification', CURRENT_DATE, 'Test Track', 'flat', 1200.0, 'dirt')
                        RETURNING id
                    """))
                    test_race_id = result.fetchone()[0]
                    conn.commit()
                    insert_duration = (time.time() - insert_start) * 1000
                    functional_results['insert'] = {
                        'status': 'success',
                        'duration_ms': insert_duration,
                        'test_id': test_race_id
                    }
                except Exception as e:
                    functional_results['insert'] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                
                # Test SELECT
                if test_race_id:
                    select_start = time.time()
                    try:
                        result = conn.execute(text(f"""
                            SELECT race_name, race_date, track_name
                            FROM {test_table}
                            WHERE id = :race_id
                        """), {'race_id': test_race_id})
                        row = result.fetchone()
                        select_duration = (time.time() - select_start) * 1000
                        
                        if row and row[0] == 'Test Race Verification':
                            functional_results['select'] = {
                                'status': 'success',
                                'duration_ms': select_duration,
                                'data_verified': True
                            }
                        else:
                            functional_results['select'] = {
                                'status': 'failed',
                                'error': 'Data mismatch in SELECT'
                            }
                    except Exception as e:
                        functional_results['select'] = {
                            'status': 'failed',
                            'error': str(e)
                        }
                
                # Test UPDATE
                if test_race_id:
                    update_start = time.time()
                    try:
                        conn.execute(text(f"""
                            UPDATE {test_table}
                            SET race_name = 'Updated Test Race'
                            WHERE id = :race_id
                        """), {'race_id': test_race_id})
                        conn.commit()
                        update_duration = (time.time() - update_start) * 1000
                        functional_results['update'] = {
                            'status': 'success',
                            'duration_ms': update_duration
                        }
                    except Exception as e:
                        functional_results['update'] = {
                            'status': 'failed',
                            'error': str(e)
                        }
                
                # Test complex JOIN
                join_start = time.time()
                try:
                    result = conn.execute(text("""
                        SELECT COUNT(*)
                        FROM races r
                        LEFT JOIN predictions p ON r.id = p.race_id
                        LEFT JOIN horses h ON p.horse_id = h.id
                    """))
                    join_count = result.fetchone()[0]
                    join_duration = (time.time() - join_start) * 1000
                    functional_results['join'] = {
                        'status': 'success',
                        'duration_ms': join_duration,
                        'result_count': join_count
                    }
                except Exception as e:
                    functional_results['join'] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                
                # Test aggregate functions
                aggregate_start = time.time()
                try:
                    result = conn.execute(text("""
                        SELECT 
                            COUNT(*) as total_races,
                            AVG(distance) as avg_distance,
                            MIN(race_date) as earliest_race,
                            MAX(race_date) as latest_race
                        FROM races
                        WHERE distance IS NOT NULL
                    """))
                    aggregate_result = result.fetchone()
                    aggregate_duration = (time.time() - aggregate_start) * 1000
                    functional_results['aggregate'] = {
                        'status': 'success',
                        'duration_ms': aggregate_duration,
                        'total_races': aggregate_result[0],
                        'avg_distance': float(aggregate_result[1]) if aggregate_result[1] else None
                    }
                except Exception as e:
                    functional_results['aggregate'] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                
                # Clean up test data
                if test_race_id:
                    try:
                        conn.execute(text(f"DELETE FROM {test_table} WHERE id = :race_id"), {'race_id': test_race_id})
                        conn.commit()
                        functional_results['cleanup'] = {'status': 'success'}
                    except Exception as e:
                        functional_results['cleanup'] = {'status': 'failed', 'error': str(e)}
            
            # Evaluate functional test results
            failed_operations = [
                op for op, result in functional_results.items()
                if result.get('status') == 'failed'
            ]
            
            successful_operations = [
                op for op, result in functional_results.items()
                if result.get('status') == 'success'
            ]
            
            duration = (time.time() - start_time) * 1000
            
            if failed_operations:
                self._record_result(
                    'Functional Testing',
                    'FAIL',
                    f'Functional operations failed: {", ".join(failed_operations)}',
                    functional_results,
                    duration
                )
                return False
            
            self._record_result(
                'Functional Testing',
                'PASS',
                f'All functional operations successful ({len(successful_operations)} operations tested)',
                functional_results,
                duration
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Functional Testing',
                'FAIL',
                f'Functional testing failed: {str(e)}',
                {'error': str(e)},
                duration
            )
            return False
    
    def verify_data_consistency(self) -> bool:
        """Verify data consistency between SQLite and PostgreSQL"""
        if not self.sqlite_engine:
            self._record_result(
                'Data Consistency',
                'WARN',
                'SQLite database not available for consistency comparison',
                {}
            )
            return True
        
        start_time = time.time()
        
        try:
            consistency_results = {}
            sample_size = self.config['sample_size']
            
            for table in self.config['tables']:
                # Get sample data from both databases
                with self.sqlite_engine.connect() as sqlite_conn:
                    sqlite_sample = pd.read_sql(
                        f"SELECT * FROM {table} ORDER BY id LIMIT {sample_size}",
                        sqlite_conn
                    )
                
                with self.postgres_engine.connect() as pg_conn:
                    pg_sample = pd.read_sql(
                        f"SELECT * FROM {table} ORDER BY id LIMIT {sample_size}",
                        pg_conn
                    )
                
                # Compare data
                if len(sqlite_sample) != len(pg_sample):
                    consistency_results[table] = {
                        'status': 'mismatch',
                        'sqlite_rows': len(sqlite_sample),
                        'postgresql_rows': len(pg_sample),
                        'issue': 'Row count mismatch'
                    }
                    continue
                
                # Compare column values (excluding timestamp columns that might have slight differences)
                exclude_columns = ['created_at', 'updated_at']
                compare_columns = [col for col in sqlite_sample.columns if col not in exclude_columns]
                
                mismatches = 0
                for col in compare_columns:
                    if col in pg_sample.columns:
                        # Handle potential type differences
                        sqlite_values = sqlite_sample[col].astype(str).fillna('NULL')
                        pg_values = pg_sample[col].astype(str).fillna('NULL')
                        
                        if not sqlite_values.equals(pg_values):
                            mismatches += 1
                
                if mismatches > 0:
                    consistency_results[table] = {
                        'status': 'mismatch',
                        'mismatched_columns': mismatches,
                        'total_columns': len(compare_columns),
                        'sample_size': len(sqlite_sample)
                    }
                else:
                    consistency_results[table] = {
                        'status': 'consistent',
                        'sample_size': len(sqlite_sample),
                        'columns_compared': len(compare_columns)
                    }
            
            # Evaluate consistency
            inconsistent_tables = [
                table for table, result in consistency_results.items()
                if result['status'] == 'mismatch'
            ]
            
            consistent_tables = [
                table for table, result in consistency_results.items()
                if result['status'] == 'consistent'
            ]
            
            duration = (time.time() - start_time) * 1000
            
            if inconsistent_tables:
                self._record_result(
                    'Data Consistency',
                    'FAIL',
                    f'Data inconsistencies found in tables: {", ".join(inconsistent_tables)}',
                    consistency_results,
                    duration
                )
                return False
            
            self._record_result(
                'Data Consistency',
                'PASS',
                f'Data consistency verified for {len(consistent_tables)} tables',
                consistency_results,
                duration
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Data Consistency',
                'FAIL',
                f'Data consistency verification failed: {str(e)}',
                {'error': str(e)},
                duration
            )
            return False
    
    def run_all_verifications(self) -> MigrationSummary:
        """Run all verification tests"""
        self.logger.info("Starting comprehensive migration verification...")
        
        verification_functions = []
        
        if self.config['verification'].get('data_completeness', True):
            verification_functions.append(('Data Completeness', self.verify_data_completeness))
        
        if self.config['verification'].get('schema_structure', True):
            verification_functions.append(('Schema Structure', self.verify_schema_structure))
        
        if self.config['verification'].get('data_integrity', True):
            verification_functions.append(('Data Integrity', self.verify_data_integrity))
        
        if self.config['verification'].get('performance_comparison', True):
            verification_functions.append(('Performance Comparison', self.verify_performance_comparison))
        
        if self.config['verification'].get('functional_testing', True):
            verification_functions.append(('Functional Testing', self.verify_functional_testing))
        
        # Always run data consistency check
        verification_functions.append(('Data Consistency', self.verify_data_consistency))
        
        # Run verifications
        verification_results = {}
        for test_name, test_function in verification_functions:
            self.logger.info(f"Running {test_name} verification...")
            try:
                result = test_function()
                verification_results[test_name] = result
            except Exception as e:
                self.logger.error(f"Verification {test_name} failed with exception: {e}")
                verification_results[test_name] = False
        
        # Generate summary
        total_duration = (datetime.now() - self.start_time).total_seconds()
        summary = self._generate_summary(verification_results, total_duration)
        
        self.logger.info(f"Migration verification completed in {total_duration:.2f} seconds")
        self.logger.info(f"Overall status: {summary.overall_status}")
        
        return summary
    
    def _generate_summary(self, verification_results: Dict[str, bool], total_duration: float) -> MigrationSummary:
        """Generate migration verification summary"""
        # Count results by status
        status_counts = {'PASS': 0, 'WARN': 0, 'FAIL': 0}
        
        for result in self.results:
            status_counts[result.status] += 1
        
        # Determine overall status
        if status_counts['FAIL'] > 0:
            overall_status = 'FAIL'
        elif status_counts['WARN'] > 0:
            overall_status = 'WARN'
        else:
            overall_status = 'PASS'
        
        # Calculate data integrity score
        integrity_results = [r for r in self.results if r.test_name == 'Data Integrity']
        data_integrity_score = integrity_results[0].actual if integrity_results and integrity_results[0].actual else 0
        
        # Calculate performance improvement
        performance_results = [r for r in self.results if r.test_name == 'Performance Comparison']
        performance_improvement = performance_results[0].actual if performance_results and performance_results[0].actual else 0
        
        summary = MigrationSummary(
            overall_status=overall_status,
            total_tests=len(self.results),
            passed_tests=status_counts['PASS'],
            failed_tests=status_counts['FAIL'],
            warning_tests=status_counts['WARN'],
            data_integrity_score=data_integrity_score,
            performance_improvement=performance_improvement,
            migration_timestamp=datetime.now() - timedelta(hours=1),  # Estimate
            verification_timestamp=datetime.now()
        )
        
        return summary
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate detailed verification report"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"reports/migration_verification_report_{timestamp}.json"
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Run all verifications if not already done
        if not self.results:
            summary = self.run_all_verifications()
        else:
            summary = self._generate_summary({}, 0)
        
        # Prepare report data
        report_data = {
            'summary': asdict(summary),
            'detailed_results': [asdict(result) for result in self.results],
            'configuration': self.config,
            'verification_metadata': {
                'verifier_version': '1.0',
                'python_version': sys.version,
                'verification_duration_seconds': (datetime.now() - self.start_time).total_seconds()
            }
        }
        
        # Write report
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Migration verification report saved to: {output_file}")
        return output_file
    
    def generate_certificate(self, output_file: str = None) -> str:
        """Generate migration success certificate"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"reports/migration_certificate_{timestamp}.txt"
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Run verifications if needed
        if not self.results:
            summary = self.run_all_verifications()
        else:
            summary = self._generate_summary({}, 0)
        
        # Generate certificate content
        certificate_content = f"""
{'='*80}
                    POSTGRESQL MIGRATION CERTIFICATE
{'='*80}

This certificate confirms that the database migration from SQLite to PostgreSQL
has been successfully completed and verified.

MIGRATION DETAILS:
------------------
Migration Date: {summary.migration_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Verification Date: {summary.verification_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Overall Status: {summary.overall_status}

VERIFICATION RESULTS:
--------------------
Total Tests Conducted: {summary.total_tests}
Tests Passed: {summary.passed_tests}
Tests with Warnings: {summary.warning_tests}
Tests Failed: {summary.failed_tests}

Data Integrity Score: {summary.data_integrity_score}/100
Performance Improvement: {summary.performance_improvement:.2f}%

CERTIFICATION STATUS:
--------------------
"""
        
        if summary.overall_status == 'PASS':
            certificate_content += """
✓ MIGRATION CERTIFIED SUCCESSFUL

The PostgreSQL migration has been completed successfully with all verification
tests passing. The database is ready for production use.

Key Achievements:
- All data successfully migrated
- Schema structure properly implemented
- Data integrity maintained
- Performance improvements achieved
- Functional operations verified

"""
        elif summary.overall_status == 'WARN':
            certificate_content += """
⚠ MIGRATION COMPLETED WITH WARNINGS

The PostgreSQL migration has been completed but some warnings were identified.
Review the detailed verification report for specific issues that may need
attention.

"""
        else:
            certificate_content += """
✗ MIGRATION VERIFICATION FAILED

The PostgreSQL migration verification has identified critical issues that must
be resolved before the database can be considered ready for production use.
Please review the detailed verification report and address all failed tests.

"""
        
        certificate_content += f"""
VERIFICATION DETAILS:
--------------------
"""
        
        for result in self.results:
            status_symbol = {'PASS': '✓', 'WARN': '⚠', 'FAIL': '✗'}[result.status]
            certificate_content += f"{status_symbol} {result.test_name}: {result.message}\n"
        
        certificate_content += f"""

TECHNICAL SPECIFICATIONS:
------------------------
Source Database: SQLite
Target Database: PostgreSQL
Migration Tool Version: 1.0
Verification Tool Version: 1.0

AUTHORIZED BY:
-------------
Migration Team
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
"""
        
        # Write certificate
        with open(output_file, 'w') as f:
            f.write(certificate_content)
        
        self.logger.info(f"Migration certificate saved to: {output_file}")
        return output_file


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PostgreSQL Migration Verifier')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--output', '-o', help='Output report file path')
    parser.add_argument('--certificate', action='store_true', help='Generate migration certificate')
    parser.add_argument('--tests', nargs='+',
                       choices=['data_completeness', 'schema_structure', 'data_integrity',
                               'performance_comparison', 'functional_testing'],
                       help='Specific tests to run')
    
    args = parser.parse_args()
    
    try:
        # Initialize verifier
        verifier = MigrationVerifier(args.config)
        
        # Configure specific tests if requested
        if args.tests:
            for test in verifier.config['verification']:
                verifier.config['verification'][test] = test in args.tests
        
        # Run verification
        summary = verifier.run_all_verifications()
        
        # Generate report
        report_file = verifier.generate_report(args.output)
        
        # Generate certificate if requested
        if args.certificate:
            cert_file = verifier.generate_certificate()
            print(f"Migration certificate: {cert_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("MIGRATION VERIFICATION SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Status: {summary.overall_status}")
        print(f"Total Tests: {summary.total_tests}")
        print(f"Passed: {summary.passed_tests}")
        print(f"Warnings: {summary.warning_tests}")
        print(f"Failed: {summary.failed_tests}")
        print(f"Data Integrity Score: {summary.data_integrity_score}/100")
        print(f"Performance Improvement: {summary.performance_improvement:.2f}%")
        print(f"\nDetailed report: {report_file}")
        
        # Exit with appropriate code
        if summary.overall_status == 'FAIL':
            sys.exit(1)
        elif summary.overall_status == 'WARN':
            sys.exit(2)
        else:
            sys.exit(0)
    
    except Exception as e:
        print(f"Migration verification failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()