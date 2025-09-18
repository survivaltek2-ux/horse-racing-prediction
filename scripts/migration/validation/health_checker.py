#!/usr/bin/env python3
"""
PostgreSQL Migration Health Checker

This script provides comprehensive health checks and verification procedures
for the migrated PostgreSQL database system.

Features:
- Database connectivity and performance checks
- Data integrity verification
- Replication health monitoring
- Performance metrics validation
- Security configuration verification
- Backup system validation
- Application compatibility checks
- Real-time monitoring capabilities

Author: Migration Team
Date: January 2025
Version: 1.0
"""

import os
import sys
import time
import json
import logging
import psutil
import hashlib
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import psycopg2.extras
from psycopg2 import sql
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


@dataclass
class HealthCheckResult:
    """Health check result data structure"""
    check_name: str
    status: str  # 'PASS', 'WARN', 'FAIL'
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    database_connections: int
    query_performance: Dict[str, float]
    replication_lag: Optional[float]


class PostgreSQLHealthChecker:
    """Comprehensive PostgreSQL health checker"""
    
    def __init__(self, config_file: str = None):
        """Initialize health checker with configuration"""
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.results: List[HealthCheckResult] = []
        self.start_time = datetime.now()
        
        # Database connections
        self.primary_engine = None
        self.standby_engine = None
        self.primary_conn = None
        self.standby_conn = None
        
        self._setup_database_connections()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'database': {
                'primary': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'hrp_database',
                    'username': 'hrp_app',
                    'password': os.getenv('POSTGRES_PASSWORD', 'password')
                },
                'standby': {
                    'host': 'localhost',
                    'port': 5433,
                    'database': 'hrp_database',
                    'username': 'hrp_readonly',
                    'password': os.getenv('POSTGRES_READONLY_PASSWORD', 'password')
                }
            },
            'thresholds': {
                'response_time_ms': 1000,
                'cpu_usage_percent': 80,
                'memory_usage_percent': 85,
                'disk_usage_percent': 90,
                'replication_lag_seconds': 30,
                'connection_count_percent': 80
            },
            'checks': {
                'connectivity': True,
                'performance': True,
                'data_integrity': True,
                'replication': True,
                'security': True,
                'backup': True,
                'application': True,
                'monitoring': True
            },
            'application': {
                'url': 'http://localhost:5000',
                'health_endpoint': '/health',
                'api_endpoints': ['/api/races', '/api/horses', '/api/predictions']
            }
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
        logger = logging.getLogger('health_checker')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = 'logs/health_checks'
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        log_file = os.path.join(log_dir, f'health_check_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
            # Primary database connection
            primary_config = self.config['database']['primary']
            primary_url = f"postgresql://{primary_config['username']}:{primary_config['password']}@{primary_config['host']}:{primary_config['port']}/{primary_config['database']}"
            self.primary_engine = create_engine(primary_url, pool_pre_ping=True)
            
            # Standby database connection (if configured)
            if 'standby' in self.config['database']:
                standby_config = self.config['database']['standby']
                standby_url = f"postgresql://{standby_config['username']}:{standby_config['password']}@{standby_config['host']}:{standby_config['port']}/{standby_config['database']}"
                self.standby_engine = create_engine(standby_url, pool_pre_ping=True)
            
            self.logger.info("Database connections configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup database connections: {e}")
            raise
    
    def _record_result(self, check_name: str, status: str, message: str, 
                      details: Dict[str, Any] = None, severity: str = 'MEDIUM',
                      duration_ms: float = 0):
        """Record a health check result"""
        result = HealthCheckResult(
            check_name=check_name,
            status=status,
            message=message,
            details=details or {},
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            severity=severity
        )
        self.results.append(result)
        
        # Log the result
        log_level = {
            'PASS': logging.INFO,
            'WARN': logging.WARNING,
            'FAIL': logging.ERROR
        }.get(status, logging.INFO)
        
        self.logger.log(log_level, f"{check_name}: {status} - {message}")
    
    def check_database_connectivity(self) -> bool:
        """Check database connectivity and basic operations"""
        start_time = time.time()
        
        try:
            # Test primary database connection
            with self.primary_engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test")).fetchone()
                if result[0] != 1:
                    raise Exception("Basic query failed")
            
            primary_duration = (time.time() - start_time) * 1000
            
            # Test standby database connection (if configured)
            standby_duration = 0
            if self.standby_engine:
                standby_start = time.time()
                with self.standby_engine.connect() as conn:
                    result = conn.execute(text("SELECT 1 as test")).fetchone()
                    if result[0] != 1:
                        raise Exception("Standby basic query failed")
                standby_duration = (time.time() - standby_start) * 1000
            
            total_duration = (time.time() - start_time) * 1000
            
            details = {
                'primary_response_time_ms': primary_duration,
                'standby_response_time_ms': standby_duration if self.standby_engine else None,
                'total_response_time_ms': total_duration
            }
            
            # Check response time threshold
            threshold = self.config['thresholds']['response_time_ms']
            if primary_duration > threshold:
                self._record_result(
                    'Database Connectivity',
                    'WARN',
                    f'Primary database response time ({primary_duration:.2f}ms) exceeds threshold ({threshold}ms)',
                    details,
                    'MEDIUM',
                    total_duration
                )
                return False
            
            self._record_result(
                'Database Connectivity',
                'PASS',
                f'Database connectivity successful (Primary: {primary_duration:.2f}ms)',
                details,
                'LOW',
                total_duration
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Database Connectivity',
                'FAIL',
                f'Database connectivity failed: {str(e)}',
                {'error': str(e)},
                'CRITICAL',
                duration
            )
            return False
    
    def check_database_performance(self) -> bool:
        """Check database performance metrics"""
        start_time = time.time()
        
        try:
            performance_queries = {
                'simple_select': "SELECT COUNT(*) FROM races",
                'complex_join': """
                    SELECT r.race_name, h.horse_name, p.confidence 
                    FROM races r 
                    JOIN predictions p ON r.id = p.race_id 
                    JOIN horses h ON p.horse_id = h.id 
                    LIMIT 100
                """,
                'aggregate_query': """
                    SELECT track_name, COUNT(*) as race_count, AVG(distance) as avg_distance
                    FROM races 
                    GROUP BY track_name 
                    ORDER BY race_count DESC 
                    LIMIT 10
                """
            }
            
            performance_results = {}
            
            with self.primary_engine.connect() as conn:
                for query_name, query in performance_queries.items():
                    query_start = time.time()
                    try:
                        result = conn.execute(text(query)).fetchall()
                        query_duration = (time.time() - query_start) * 1000
                        performance_results[query_name] = {
                            'duration_ms': query_duration,
                            'rows_returned': len(result),
                            'status': 'success'
                        }
                    except Exception as e:
                        query_duration = (time.time() - query_start) * 1000
                        performance_results[query_name] = {
                            'duration_ms': query_duration,
                            'error': str(e),
                            'status': 'failed'
                        }
            
            # Check for slow queries
            slow_queries = []
            threshold = self.config['thresholds']['response_time_ms']
            
            for query_name, result in performance_results.items():
                if result['status'] == 'success' and result['duration_ms'] > threshold:
                    slow_queries.append(f"{query_name}: {result['duration_ms']:.2f}ms")
            
            total_duration = (time.time() - start_time) * 1000
            
            if slow_queries:
                self._record_result(
                    'Database Performance',
                    'WARN',
                    f'Slow queries detected: {", ".join(slow_queries)}',
                    performance_results,
                    'MEDIUM',
                    total_duration
                )
                return False
            
            failed_queries = [name for name, result in performance_results.items() 
                            if result['status'] == 'failed']
            
            if failed_queries:
                self._record_result(
                    'Database Performance',
                    'FAIL',
                    f'Failed queries: {", ".join(failed_queries)}',
                    performance_results,
                    'HIGH',
                    total_duration
                )
                return False
            
            avg_duration = statistics.mean([r['duration_ms'] for r in performance_results.values()])
            
            self._record_result(
                'Database Performance',
                'PASS',
                f'All performance queries successful (Avg: {avg_duration:.2f}ms)',
                performance_results,
                'LOW',
                total_duration
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Database Performance',
                'FAIL',
                f'Performance check failed: {str(e)}',
                {'error': str(e)},
                'HIGH',
                duration
            )
            return False
    
    def check_data_integrity(self) -> bool:
        """Check data integrity and consistency"""
        start_time = time.time()
        
        try:
            integrity_checks = []
            
            with self.primary_engine.connect() as conn:
                # Check for orphaned records
                orphan_checks = {
                    'orphaned_predictions_race': """
                        SELECT COUNT(*) FROM predictions p 
                        LEFT JOIN races r ON p.race_id = r.id 
                        WHERE r.id IS NULL
                    """,
                    'orphaned_predictions_horse': """
                        SELECT COUNT(*) FROM predictions p 
                        LEFT JOIN horses h ON p.horse_id = h.id 
                        WHERE h.id IS NULL
                    """
                }
                
                for check_name, query in orphan_checks.items():
                    result = conn.execute(text(query)).fetchone()
                    orphan_count = result[0]
                    integrity_checks.append({
                        'check': check_name,
                        'orphan_count': orphan_count,
                        'status': 'pass' if orphan_count == 0 else 'fail'
                    })
                
                # Check data consistency
                consistency_checks = {
                    'negative_confidence': """
                        SELECT COUNT(*) FROM predictions 
                        WHERE confidence < 0 OR confidence > 1
                    """,
                    'invalid_horse_age': """
                        SELECT COUNT(*) FROM horses 
                        WHERE age < 2 OR age > 20
                    """,
                    'future_race_dates': """
                        SELECT COUNT(*) FROM races 
                        WHERE race_date > CURRENT_DATE + INTERVAL '1 year'
                    """
                }
                
                for check_name, query in consistency_checks.items():
                    result = conn.execute(text(query)).fetchone()
                    invalid_count = result[0]
                    integrity_checks.append({
                        'check': check_name,
                        'invalid_count': invalid_count,
                        'status': 'pass' if invalid_count == 0 else 'fail'
                    })
                
                # Check referential integrity
                foreign_key_check = conn.execute(text("""
                    SELECT COUNT(*) FROM information_schema.table_constraints 
                    WHERE constraint_type = 'FOREIGN KEY'
                """)).fetchone()
                
                integrity_checks.append({
                    'check': 'foreign_key_constraints',
                    'constraint_count': foreign_key_check[0],
                    'status': 'pass' if foreign_key_check[0] > 0 else 'warn'
                })
            
            # Evaluate results
            failed_checks = [check for check in integrity_checks if check['status'] == 'fail']
            warning_checks = [check for check in integrity_checks if check['status'] == 'warn']
            
            total_duration = (time.time() - start_time) * 1000
            
            if failed_checks:
                self._record_result(
                    'Data Integrity',
                    'FAIL',
                    f'Data integrity violations found: {len(failed_checks)} failed checks',
                    {'checks': integrity_checks, 'failed_checks': failed_checks},
                    'HIGH',
                    total_duration
                )
                return False
            
            if warning_checks:
                self._record_result(
                    'Data Integrity',
                    'WARN',
                    f'Data integrity warnings: {len(warning_checks)} warning checks',
                    {'checks': integrity_checks, 'warning_checks': warning_checks},
                    'MEDIUM',
                    total_duration
                )
                return False
            
            self._record_result(
                'Data Integrity',
                'PASS',
                f'All data integrity checks passed ({len(integrity_checks)} checks)',
                {'checks': integrity_checks},
                'LOW',
                total_duration
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Data Integrity',
                'FAIL',
                f'Data integrity check failed: {str(e)}',
                {'error': str(e)},
                'HIGH',
                duration
            )
            return False
    
    def check_replication_health(self) -> bool:
        """Check replication status and lag"""
        if not self.standby_engine:
            self._record_result(
                'Replication Health',
                'WARN',
                'No standby server configured for replication check',
                {},
                'LOW',
                0
            )
            return True
        
        start_time = time.time()
        
        try:
            replication_info = {}
            
            # Check primary server replication status
            with self.primary_engine.connect() as conn:
                # Get replication slots
                slots_result = conn.execute(text("""
                    SELECT slot_name, active, restart_lsn, confirmed_flush_lsn
                    FROM pg_replication_slots
                """)).fetchall()
                
                replication_info['replication_slots'] = [
                    {
                        'slot_name': row[0],
                        'active': row[1],
                        'restart_lsn': str(row[2]),
                        'confirmed_flush_lsn': str(row[3])
                    }
                    for row in slots_result
                ]
                
                # Get WAL sender processes
                wal_senders = conn.execute(text("""
                    SELECT client_addr, state, sent_lsn, write_lsn, flush_lsn, replay_lsn,
                           write_lag, flush_lag, replay_lag
                    FROM pg_stat_replication
                """)).fetchall()
                
                replication_info['wal_senders'] = [
                    {
                        'client_addr': str(row[0]),
                        'state': row[1],
                        'sent_lsn': str(row[2]),
                        'write_lsn': str(row[3]),
                        'flush_lsn': str(row[4]),
                        'replay_lsn': str(row[5]),
                        'write_lag': str(row[6]) if row[6] else None,
                        'flush_lag': str(row[7]) if row[7] else None,
                        'replay_lag': str(row[8]) if row[8] else None
                    }
                    for row in wal_senders
                ]
            
            # Check standby server status
            with self.standby_engine.connect() as conn:
                # Check if in recovery mode
                recovery_status = conn.execute(text("SELECT pg_is_in_recovery()")).fetchone()
                replication_info['standby_in_recovery'] = recovery_status[0]
                
                # Get last received WAL
                if recovery_status[0]:
                    wal_status = conn.execute(text("""
                        SELECT pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn(),
                               pg_last_xact_replay_timestamp()
                    """)).fetchone()
                    
                    replication_info['standby_wal_status'] = {
                        'last_receive_lsn': str(wal_status[0]),
                        'last_replay_lsn': str(wal_status[1]),
                        'last_replay_timestamp': str(wal_status[2]) if wal_status[2] else None
                    }
            
            # Calculate replication lag
            replication_lag = None
            if replication_info.get('standby_wal_status', {}).get('last_replay_timestamp'):
                last_replay = datetime.fromisoformat(
                    replication_info['standby_wal_status']['last_replay_timestamp'].replace('Z', '+00:00')
                )
                replication_lag = (datetime.now().replace(tzinfo=last_replay.tzinfo) - last_replay).total_seconds()
                replication_info['replication_lag_seconds'] = replication_lag
            
            total_duration = (time.time() - start_time) * 1000
            
            # Evaluate replication health
            issues = []
            
            if not replication_info.get('standby_in_recovery'):
                issues.append("Standby server not in recovery mode")
            
            if not replication_info.get('wal_senders'):
                issues.append("No active WAL sender processes")
            
            if replication_lag and replication_lag > self.config['thresholds']['replication_lag_seconds']:
                issues.append(f"Replication lag ({replication_lag:.2f}s) exceeds threshold")
            
            if issues:
                self._record_result(
                    'Replication Health',
                    'FAIL',
                    f'Replication issues detected: {"; ".join(issues)}',
                    replication_info,
                    'HIGH',
                    total_duration
                )
                return False
            
            self._record_result(
                'Replication Health',
                'PASS',
                f'Replication healthy (Lag: {replication_lag:.2f}s)' if replication_lag else 'Replication healthy',
                replication_info,
                'LOW',
                total_duration
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Replication Health',
                'FAIL',
                f'Replication health check failed: {str(e)}',
                {'error': str(e)},
                'HIGH',
                duration
            )
            return False
    
    def check_system_resources(self) -> bool:
        """Check system resource utilization"""
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Get database-specific metrics
            with self.primary_engine.connect() as conn:
                # Connection count
                connection_result = conn.execute(text("""
                    SELECT count(*) as current_connections,
                           setting::int as max_connections
                    FROM pg_stat_activity, pg_settings 
                    WHERE pg_settings.name = 'max_connections'
                    GROUP BY setting
                """)).fetchone()
                
                current_connections = connection_result[0]
                max_connections = connection_result[1]
                connection_percent = (current_connections / max_connections) * 100
                
                # Database size
                db_size_result = conn.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as size,
                           pg_database_size(current_database()) as size_bytes
                """)).fetchone()
            
            metrics = {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'disk_usage_percent': disk.percent,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'database_connections': current_connections,
                'max_connections': max_connections,
                'connection_usage_percent': connection_percent,
                'database_size': db_size_result[0],
                'database_size_bytes': db_size_result[1]
            }
            
            # Check thresholds
            issues = []
            warnings = []
            
            thresholds = self.config['thresholds']
            
            if cpu_percent > thresholds['cpu_usage_percent']:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > thresholds['cpu_usage_percent'] * 0.8:
                warnings.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > thresholds['memory_usage_percent']:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            elif memory.percent > thresholds['memory_usage_percent'] * 0.8:
                warnings.append(f"Elevated memory usage: {memory.percent:.1f}%")
            
            if disk.percent > thresholds['disk_usage_percent']:
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            elif disk.percent > thresholds['disk_usage_percent'] * 0.8:
                warnings.append(f"Elevated disk usage: {disk.percent:.1f}%")
            
            if connection_percent > thresholds['connection_count_percent']:
                issues.append(f"High connection usage: {connection_percent:.1f}%")
            elif connection_percent > thresholds['connection_count_percent'] * 0.8:
                warnings.append(f"Elevated connection usage: {connection_percent:.1f}%")
            
            total_duration = (time.time() - start_time) * 1000
            
            if issues:
                self._record_result(
                    'System Resources',
                    'FAIL',
                    f'Resource issues detected: {"; ".join(issues)}',
                    metrics,
                    'HIGH',
                    total_duration
                )
                return False
            
            if warnings:
                self._record_result(
                    'System Resources',
                    'WARN',
                    f'Resource warnings: {"; ".join(warnings)}',
                    metrics,
                    'MEDIUM',
                    total_duration
                )
                return False
            
            self._record_result(
                'System Resources',
                'PASS',
                f'System resources healthy (CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, Disk: {disk.percent:.1f}%)',
                metrics,
                'LOW',
                total_duration
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'System Resources',
                'FAIL',
                f'System resource check failed: {str(e)}',
                {'error': str(e)},
                'HIGH',
                duration
            )
            return False
    
    def check_application_health(self) -> bool:
        """Check application health and API endpoints"""
        start_time = time.time()
        
        try:
            app_config = self.config.get('application', {})
            base_url = app_config.get('url', 'http://localhost:5000')
            
            endpoint_results = {}
            
            # Check main health endpoint
            health_endpoint = app_config.get('health_endpoint', '/health')
            try:
                response = requests.get(f"{base_url}{health_endpoint}", timeout=10)
                endpoint_results['health'] = {
                    'status_code': response.status_code,
                    'response_time_ms': response.elapsed.total_seconds() * 1000,
                    'content': response.text[:200] if response.text else None
                }
            except Exception as e:
                endpoint_results['health'] = {
                    'error': str(e),
                    'status': 'failed'
                }
            
            # Check API endpoints
            api_endpoints = app_config.get('api_endpoints', [])
            for endpoint in api_endpoints:
                try:
                    response = requests.get(f"{base_url}{endpoint}", timeout=10)
                    endpoint_results[endpoint] = {
                        'status_code': response.status_code,
                        'response_time_ms': response.elapsed.total_seconds() * 1000,
                        'content_length': len(response.content) if response.content else 0
                    }
                except Exception as e:
                    endpoint_results[endpoint] = {
                        'error': str(e),
                        'status': 'failed'
                    }
            
            # Evaluate results
            failed_endpoints = []
            slow_endpoints = []
            
            for endpoint, result in endpoint_results.items():
                if 'error' in result or result.get('status') == 'failed':
                    failed_endpoints.append(endpoint)
                elif result.get('status_code', 0) >= 400:
                    failed_endpoints.append(f"{endpoint} (HTTP {result['status_code']})")
                elif result.get('response_time_ms', 0) > self.config['thresholds']['response_time_ms']:
                    slow_endpoints.append(f"{endpoint} ({result['response_time_ms']:.0f}ms)")
            
            total_duration = (time.time() - start_time) * 1000
            
            if failed_endpoints:
                self._record_result(
                    'Application Health',
                    'FAIL',
                    f'Application endpoints failed: {", ".join(failed_endpoints)}',
                    endpoint_results,
                    'HIGH',
                    total_duration
                )
                return False
            
            if slow_endpoints:
                self._record_result(
                    'Application Health',
                    'WARN',
                    f'Slow application endpoints: {", ".join(slow_endpoints)}',
                    endpoint_results,
                    'MEDIUM',
                    total_duration
                )
                return False
            
            successful_endpoints = len([r for r in endpoint_results.values() 
                                      if r.get('status_code', 0) < 400 and 'error' not in r])
            
            self._record_result(
                'Application Health',
                'PASS',
                f'Application healthy ({successful_endpoints} endpoints checked)',
                endpoint_results,
                'LOW',
                total_duration
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Application Health',
                'FAIL',
                f'Application health check failed: {str(e)}',
                {'error': str(e)},
                'HIGH',
                duration
            )
            return False
    
    def check_backup_system(self) -> bool:
        """Check backup system health"""
        start_time = time.time()
        
        try:
            backup_info = {}
            
            # Check for recent backups
            backup_dirs = [
                'backups/postgresql',
                'backups/migration',
                '/var/backups/postgresql'
            ]
            
            recent_backups = []
            for backup_dir in backup_dirs:
                if os.path.exists(backup_dir):
                    for file in os.listdir(backup_dir):
                        file_path = os.path.join(backup_dir, file)
                        if os.path.isfile(file_path):
                            stat = os.stat(file_path)
                            recent_backups.append({
                                'file': file_path,
                                'size_bytes': stat.st_size,
                                'modified': datetime.fromtimestamp(stat.st_mtime),
                                'age_hours': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 3600
                            })
            
            # Sort by modification time
            recent_backups.sort(key=lambda x: x['modified'], reverse=True)
            backup_info['recent_backups'] = recent_backups[:10]  # Keep top 10
            
            # Check WAL archiving
            with self.primary_engine.connect() as conn:
                wal_info = conn.execute(text("""
                    SELECT name, setting FROM pg_settings 
                    WHERE name IN ('archive_mode', 'archive_command', 'wal_level')
                """)).fetchall()
                
                backup_info['wal_settings'] = {row[0]: row[1] for row in wal_info}
                
                # Check archive status
                archive_status = conn.execute(text("""
                    SELECT archived_count, failed_count, last_archived_wal, last_archived_time
                    FROM pg_stat_archiver
                """)).fetchone()
                
                if archive_status:
                    backup_info['archive_status'] = {
                        'archived_count': archive_status[0],
                        'failed_count': archive_status[1],
                        'last_archived_wal': archive_status[2],
                        'last_archived_time': str(archive_status[3]) if archive_status[3] else None
                    }
            
            # Evaluate backup health
            issues = []
            warnings = []
            
            # Check for recent backups (within 24 hours)
            if not recent_backups or recent_backups[0]['age_hours'] > 24:
                issues.append("No recent backups found (within 24 hours)")
            elif recent_backups[0]['age_hours'] > 12:
                warnings.append(f"Latest backup is {recent_backups[0]['age_hours']:.1f} hours old")
            
            # Check WAL archiving
            if backup_info.get('wal_settings', {}).get('archive_mode') != 'on':
                warnings.append("WAL archiving not enabled")
            
            if backup_info.get('archive_status', {}).get('failed_count', 0) > 0:
                warnings.append(f"WAL archive failures: {backup_info['archive_status']['failed_count']}")
            
            total_duration = (time.time() - start_time) * 1000
            
            if issues:
                self._record_result(
                    'Backup System',
                    'FAIL',
                    f'Backup issues detected: {"; ".join(issues)}',
                    backup_info,
                    'HIGH',
                    total_duration
                )
                return False
            
            if warnings:
                self._record_result(
                    'Backup System',
                    'WARN',
                    f'Backup warnings: {"; ".join(warnings)}',
                    backup_info,
                    'MEDIUM',
                    total_duration
                )
                return False
            
            backup_count = len(recent_backups)
            latest_age = recent_backups[0]['age_hours'] if recent_backups else 0
            
            self._record_result(
                'Backup System',
                'PASS',
                f'Backup system healthy ({backup_count} backups found, latest: {latest_age:.1f}h old)',
                backup_info,
                'LOW',
                total_duration
            )
            return True
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self._record_result(
                'Backup System',
                'FAIL',
                f'Backup system check failed: {str(e)}',
                {'error': str(e)},
                'HIGH',
                duration
            )
            return False
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        self.logger.info("Starting comprehensive health check...")
        
        check_functions = []
        
        if self.config['checks'].get('connectivity', True):
            check_functions.append(('connectivity', self.check_database_connectivity))
        
        if self.config['checks'].get('performance', True):
            check_functions.append(('performance', self.check_database_performance))
        
        if self.config['checks'].get('data_integrity', True):
            check_functions.append(('data_integrity', self.check_data_integrity))
        
        if self.config['checks'].get('replication', True):
            check_functions.append(('replication', self.check_replication_health))
        
        if self.config['checks'].get('application', True):
            check_functions.append(('application', self.check_application_health))
        
        if self.config['checks'].get('backup', True):
            check_functions.append(('backup', self.check_backup_system))
        
        # Add system resource check
        check_functions.append(('system_resources', self.check_system_resources))
        
        # Run checks
        check_results = {}
        for check_name, check_function in check_functions:
            self.logger.info(f"Running {check_name} check...")
            try:
                result = check_function()
                check_results[check_name] = result
            except Exception as e:
                self.logger.error(f"Check {check_name} failed with exception: {e}")
                check_results[check_name] = False
        
        # Generate summary
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        summary = self._generate_summary(check_results, total_duration)
        
        self.logger.info(f"Health check completed in {total_duration:.2f} seconds")
        self.logger.info(f"Overall status: {summary['overall_status']}")
        
        return summary
    
    def _generate_summary(self, check_results: Dict[str, bool], total_duration: float) -> Dict[str, Any]:
        """Generate health check summary"""
        # Count results by status
        status_counts = {'PASS': 0, 'WARN': 0, 'FAIL': 0}
        severity_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        
        for result in self.results:
            status_counts[result.status] += 1
            severity_counts[result.severity] += 1
        
        # Determine overall status
        if status_counts['FAIL'] > 0:
            overall_status = 'FAIL'
        elif status_counts['WARN'] > 0:
            overall_status = 'WARN'
        else:
            overall_status = 'PASS'
        
        # Calculate health score (0-100)
        total_checks = len(self.results)
        if total_checks > 0:
            health_score = (
                (status_counts['PASS'] * 100 + status_counts['WARN'] * 50) / total_checks
            )
        else:
            health_score = 0
        
        summary = {
            'overall_status': overall_status,
            'health_score': round(health_score, 1),
            'total_checks': total_checks,
            'status_counts': status_counts,
            'severity_counts': severity_counts,
            'check_results': check_results,
            'duration_seconds': total_duration,
            'timestamp': datetime.now().isoformat(),
            'detailed_results': [asdict(result) for result in self.results]
        }
        
        return summary
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate detailed health check report"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"reports/health_check_report_{timestamp}.json"
        
        # Ensure reports directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Run all checks if not already done
        if not self.results:
            summary = self.run_all_checks()
        else:
            summary = self._generate_summary({}, 0)
        
        # Write report
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Health check report saved to: {output_file}")
        return output_file
    
    def continuous_monitoring(self, interval_minutes: int = 5, duration_hours: int = 24):
        """Run continuous health monitoring"""
        self.logger.info(f"Starting continuous monitoring (interval: {interval_minutes}min, duration: {duration_hours}h)")
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            try:
                # Run basic checks
                self.check_database_connectivity()
                self.check_database_performance()
                self.check_system_resources()
                
                if self.standby_engine:
                    self.check_replication_health()
                
                # Sleep until next check
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                self.logger.info("Continuous monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PostgreSQL Migration Health Checker')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--output', '-o', help='Output report file path')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in minutes')
    parser.add_argument('--duration', type=int, default=24, help='Monitoring duration in hours')
    parser.add_argument('--checks', nargs='+', 
                       choices=['connectivity', 'performance', 'data_integrity', 'replication', 
                               'application', 'backup', 'system_resources'],
                       help='Specific checks to run')
    
    args = parser.parse_args()
    
    try:
        # Initialize health checker
        checker = PostgreSQLHealthChecker(args.config)
        
        # Configure specific checks if requested
        if args.checks:
            for check in checker.config['checks']:
                checker.config['checks'][check] = check in args.checks
        
        if args.continuous:
            # Run continuous monitoring
            checker.continuous_monitoring(args.interval, args.duration)
        else:
            # Run one-time health check
            summary = checker.run_all_checks()
            
            # Generate report
            report_file = checker.generate_report(args.output)
            
            # Print summary
            print(f"\n{'='*60}")
            print("POSTGRESQL HEALTH CHECK SUMMARY")
            print(f"{'='*60}")
            print(f"Overall Status: {summary['overall_status']}")
            print(f"Health Score: {summary['health_score']}/100")
            print(f"Total Checks: {summary['total_checks']}")
            print(f"Duration: {summary['duration_seconds']:.2f} seconds")
            print(f"\nStatus Breakdown:")
            for status, count in summary['status_counts'].items():
                print(f"  {status}: {count}")
            print(f"\nDetailed report saved to: {report_file}")
            
            # Exit with appropriate code
            if summary['overall_status'] == 'FAIL':
                sys.exit(1)
            elif summary['overall_status'] == 'WARN':
                sys.exit(2)
            else:
                sys.exit(0)
    
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()