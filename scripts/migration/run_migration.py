#!/usr/bin/env python3
"""
PostgreSQL Migration Orchestrator

This script orchestrates the complete migration process from SQLite to PostgreSQL,
coordinating all migration components and ensuring proper execution order.

Features:
- Complete migration workflow orchestration
- Pre-migration validation and preparation
- Schema creation and data migration
- Post-migration verification and validation
- Backup and rollback capabilities
- Comprehensive logging and reporting
- Error handling and recovery

Author: Migration Team
Date: January 2025
Version: 1.0
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Add migration modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'schema'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'validation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backup'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'rollback'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'performance'))

# Import migration modules
try:
    from schema_converter import PostgreSQLSchemaConverter
    from data_migrator import PostgreSQLDataMigrator
    from data_validator import DataValidator
    from health_checker import PostgreSQLHealthChecker
    from migration_verifier import MigrationVerifier
    from backup_manager import BackupManager
    from rollback_manager import RollbackManager, RollbackPhase
    from performance_tester import PerformanceTester
    from benchmark_suite import DatabaseBenchmark
except ImportError as e:
    print(f"Error importing migration modules: {e}")
    print("Please ensure all migration scripts are in the correct directories.")
    sys.exit(1)


@dataclass
class MigrationPhase:
    """Migration phase information"""
    name: str
    description: str
    required: bool = True
    completed: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0
    status: str = 'pending'  # pending, running, completed, failed, skipped
    error_message: Optional[str] = None
    artifacts: List[str] = None

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []


@dataclass
class MigrationConfig:
    """Migration configuration"""
    source_database: Dict[str, Any]
    target_database: Dict[str, Any]
    migration_options: Dict[str, Any]
    backup_options: Dict[str, Any]
    validation_options: Dict[str, Any]
    performance_options: Dict[str, Any]


class MigrationOrchestrator:
    """Main migration orchestrator"""
    
    def __init__(self, config_file: str = None, dry_run: bool = False):
        """Initialize migration orchestrator"""
        self.config = self._load_config(config_file)
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        self.start_time = datetime.now()
        
        # Migration phases
        self.phases = self._initialize_phases()
        
        # Migration components
        self.schema_converter = None
        self.data_migrator = None
        self.data_validator = None
        self.health_checker = None
        self.migration_verifier = None
        self.backup_manager = None
        self.rollback_manager = None
        self.performance_tester = None
        self.database_benchmark = None
        
        # Migration state
        self.migration_id = f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.migration_successful = False
        self.rollback_required = False
        
        self.logger.info(f"Migration orchestrator initialized (ID: {self.migration_id})")
        if self.dry_run:
            self.logger.info("DRY RUN MODE: No actual changes will be made")
    
    def _load_config(self, config_file: str) -> MigrationConfig:
        """Load migration configuration"""
        default_config = {
            'source_database': {
                'type': 'sqlite',
                'path': 'database.db'
            },
            'target_database': {
                'type': 'postgresql',
                'host': 'localhost',
                'port': 5432,
                'database': 'hrp_database',
                'username': 'hrp_app',
                'password': os.getenv('POSTGRES_PASSWORD', 'password')
            },
            'migration_options': {
                'batch_size': 1000,
                'parallel_workers': 4,
                'create_indexes': True,
                'create_constraints': True,
                'validate_data': True,
                'backup_before_migration': True,
                'run_performance_tests': True
            },
            'backup_options': {
                'create_source_backup': True,
                'create_target_backup': True,
                'backup_directory': 'backups',
                'retention_days': 30
            },
            'validation_options': {
                'validate_schema': True,
                'validate_data_integrity': True,
                'validate_data_completeness': True,
                'validate_performance': True,
                'sample_size': 1000
            },
            'performance_options': {
                'run_benchmarks': True,
                'performance_threshold_percent': 20,
                'load_test_duration_seconds': 300
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Deep merge configurations
                    for key, value in user_config.items():
                        if key in default_config and isinstance(default_config[key], dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
        
        return MigrationConfig(**default_config)
    
    def _construct_database_uri(self, db_config: Dict[str, Any]) -> str:
        """Construct database URI from configuration parameters"""
        if 'uri' in db_config:
            return db_config['uri']
        
        if db_config.get('type') == 'sqlite':
            return f"sqlite:///{db_config.get('path', 'database.db')}"
        elif db_config.get('type') == 'postgresql':
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 5432)
            database = db_config.get('database', 'hrp_production')
            username = db_config.get('username', 'hrp_app')
            password = db_config.get('password', '')
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        else:
            raise ValueError(f"Unsupported database type: {db_config.get('type')}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('migration_orchestrator')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = 'logs/migration'
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        log_file = os.path.join(log_dir, f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
    
    def _initialize_phases(self) -> List[MigrationPhase]:
        """Initialize migration phases"""
        phases = [
            MigrationPhase(
                name="pre_migration_validation",
                description="Validate source database and migration prerequisites",
                required=True
            ),
            MigrationPhase(
                name="backup_creation",
                description="Create backups of source and target databases",
                required=self.config.backup_options['create_source_backup']
            ),
            MigrationPhase(
                name="schema_conversion",
                description="Convert and create PostgreSQL schema",
                required=True
            ),
            MigrationPhase(
                name="data_migration",
                description="Migrate data from SQLite to PostgreSQL",
                required=True
            ),
            MigrationPhase(
                name="data_validation",
                description="Validate migrated data integrity and completeness",
                required=self.config.validation_options['validate_data_integrity']
            ),
            MigrationPhase(
                name="index_creation",
                description="Create indexes and constraints",
                required=self.config.migration_options['create_indexes']
            ),
            MigrationPhase(
                name="performance_testing",
                description="Run performance tests and benchmarks",
                required=self.config.performance_options['run_benchmarks']
            ),
            MigrationPhase(
                name="final_validation",
                description="Comprehensive migration verification",
                required=True
            ),
            MigrationPhase(
                name="health_check",
                description="System health and readiness verification",
                required=True
            ),
            MigrationPhase(
                name="migration_certification",
                description="Generate migration completion certificate",
                required=True
            )
        ]
        
        return phases
    
    def _initialize_components(self):
        """Initialize migration components"""
        try:
            # Construct database URIs
            source_uri = self._construct_database_uri(self.config.source_database)
            target_uri = self._construct_database_uri(self.config.target_database)
            
            # Schema converter
            self.schema_converter = PostgreSQLSchemaConverter(source_uri, target_uri)
            
            # Data migrator
            self.data_migrator = PostgreSQLDataMigrator(
                source_uri,
                target_uri,
                self.config.migration_options.get('batch_size', 1000)
            )
            
            # Data validator
            self.data_validator = DataValidator(source_uri, target_uri)
            
            # Health checker
            self.health_checker = PostgreSQLHealthChecker()
            
            # Migration verifier
            self.migration_verifier = MigrationVerifier()
            
            # Backup manager
            self.backup_manager = BackupManager(source_uri, target_uri)
            
            # Rollback manager
            self.rollback_manager = RollbackManager(source_uri, target_uri)
            
            # Performance tester
            self.performance_tester = PerformanceTester(source_uri, target_uri)
            
            # Database benchmark
            self.database_benchmark = DatabaseBenchmark(source_uri, target_uri)
            
            self.logger.info("Migration components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize migration components: {e}")
            raise
    
    def _start_phase(self, phase_name: str) -> MigrationPhase:
        """Start a migration phase"""
        phase = next((p for p in self.phases if p.name == phase_name), None)
        if not phase:
            raise ValueError(f"Unknown migration phase: {phase_name}")
        
        phase.status = 'running'
        phase.start_time = datetime.now()
        
        self.logger.info(f"Starting phase: {phase.description}")
        return phase
    
    def _complete_phase(self, phase_name: str, success: bool = True, 
                       error_message: str = None, artifacts: List[str] = None):
        """Complete a migration phase"""
        phase = next((p for p in self.phases if p.name == phase_name), None)
        if not phase:
            raise ValueError(f"Unknown migration phase: {phase_name}")
        
        phase.end_time = datetime.now()
        phase.duration_seconds = (phase.end_time - phase.start_time).total_seconds()
        phase.status = 'completed' if success else 'failed'
        phase.completed = success
        phase.error_message = error_message
        
        if artifacts:
            phase.artifacts.extend(artifacts)
        
        status_msg = "completed successfully" if success else f"failed: {error_message}"
        self.logger.info(f"Phase {phase.description} {status_msg} ({phase.duration_seconds:.2f}s)")
        
        return phase
    
    def _skip_phase(self, phase_name: str, reason: str = None):
        """Skip a migration phase"""
        phase = next((p for p in self.phases if p.name == phase_name), None)
        if not phase:
            raise ValueError(f"Unknown migration phase: {phase_name}")
        
        phase.status = 'skipped'
        phase.error_message = reason
        
        self.logger.info(f"Skipping phase: {phase.description} - {reason}")
        return phase
    
    def run_pre_migration_validation(self) -> bool:
        """Run pre-migration validation"""
        phase = self._start_phase('pre_migration_validation')
        
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Pre-migration validation")
                time.sleep(1)  # Simulate work
                self._complete_phase('pre_migration_validation', artifacts=['validation_report.json'])
                return True
            
            # Check source database
            source_path = self.config.source_database['path']
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source database not found: {source_path}")
            
            # Check PostgreSQL connection
            import psycopg2
            pg_config = self.config.target_database
            try:
                conn = psycopg2.connect(
                    host=pg_config['host'],
                    port=pg_config['port'],
                    database=pg_config['database'],
                    user=pg_config['username'],
                    password=pg_config['password']
                )
                conn.close()
            except psycopg2.Error as e:
                raise ConnectionError(f"Cannot connect to PostgreSQL: {e}")
            
            # Check disk space
            source_size = os.path.getsize(source_path)
            free_space = os.statvfs('.').f_frsize * os.statvfs('.').f_bavail
            
            if free_space < source_size * 3:  # Need 3x space for safety
                raise RuntimeError(f"Insufficient disk space. Need {source_size * 3} bytes, have {free_space}")
            
            # Validate migration scripts
            script_dir = os.path.dirname(__file__)
            required_scripts = [
                'schema/schema_converter.py',
                'data/data_migrator.py',
                'validation/data_validator.py'
            ]
            
            for script in required_scripts:
                script_path = os.path.join(script_dir, script)
                if not os.path.exists(script_path):
                    raise FileNotFoundError(f"Required migration script not found: {script_path}")
            
            self._complete_phase('pre_migration_validation', artifacts=['pre_migration_validation.log'])
            return True
            
        except Exception as e:
            self._complete_phase('pre_migration_validation', False, str(e))
            return False
    
    def run_backup_creation(self) -> bool:
        """Create database backups"""
        if not self.config.backup_options['create_source_backup']:
            self._skip_phase('backup_creation', 'Backup creation disabled in configuration')
            return True
        
        phase = self._start_phase('backup_creation')
        
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Backup creation")
                time.sleep(2)  # Simulate work
                self._complete_phase('backup_creation', artifacts=['backup_source.db', 'backup_target.sql'])
                return True
            
            # Initialize backup manager if not already done
            if not self.backup_manager:
                self.backup_manager = BackupManager()
            
            artifacts = []
            
            # Create source backup
            if self.config.backup_options['create_source_backup']:
                source_backup = self.backup_manager.create_sqlite_backup(
                    backup_type="full"
                )
                artifacts.append(source_backup)
                self.logger.info(f"Source backup created: {source_backup}")
            
            # Create target backup (if database exists)
            if self.config.backup_options['create_target_backup']:
                try:
                    target_backup = self.backup_manager.create_postgresql_backup(
                        backup_type="full"
                    )
                    artifacts.append(target_backup)
                    self.logger.info(f"Target backup created: {target_backup}")
                except Exception as e:
                    self.logger.warning(f"Could not create target backup: {e}")
            
            self._complete_phase('backup_creation', artifacts=artifacts)
            return True
            
        except Exception as e:
            self._complete_phase('backup_creation', False, str(e))
            return False
    
    def run_schema_conversion(self) -> bool:
        """Convert and create PostgreSQL schema"""
        phase = self._start_phase('schema_conversion')
        
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Schema conversion")
                time.sleep(3)  # Simulate work
                self._complete_phase('schema_conversion', artifacts=['postgresql_schema.sql'])
                return True
            
            # Initialize schema converter if not already done
            if not self.schema_converter:
                source_uri = self._construct_database_uri(self.config.source_database)
                target_uri = self._construct_database_uri(self.config.target_database)
                self.schema_converter = PostgreSQLSchemaConverter(source_uri, target_uri)
            
            # Convert schema
            schema_file = self.schema_converter.convert_schema(
                self.config.source_database['path'],
                self.config.target_database
            )
            
            self.logger.info(f"Schema conversion completed: {schema_file}")
            self._complete_phase('schema_conversion', artifacts=[schema_file])
            return True
            
        except Exception as e:
            self._complete_phase('schema_conversion', False, str(e))
            return False
    
    def run_data_migration(self) -> bool:
        """Migrate data from SQLite to PostgreSQL"""
        phase = self._start_phase('data_migration')
        
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Data migration")
                time.sleep(5)  # Simulate work
                self._complete_phase('data_migration', artifacts=['migration_report.json'])
                return True
            
            # Initialize data migrator if not already done
            if not self.data_migrator:
                source_uri = self._construct_database_uri(self.config.source_database)
                target_uri = self._construct_database_uri(self.config.target_database)
                self.data_migrator = PostgreSQLDataMigrator(
                    source_uri,
                    target_uri,
                    self.config.migration_options.get('batch_size', 1000)
                )
            
            # Run data migration
            migration_success = self.data_migrator.migrate_all_tables()
            
            self.logger.info(f"Data migration completed: {migration_success}")
            self._complete_phase('data_migration', artifacts=[f"Migration success: {migration_success}"])
            return True
            
        except Exception as e:
            self._complete_phase('data_migration', False, str(e))
            return False
    
    def run_data_validation(self) -> bool:
        """Validate migrated data"""
        if not self.config.validation_options['validate_data_integrity']:
            self._skip_phase('data_validation', 'Data validation disabled in configuration')
            return True
        
        phase = self._start_phase('data_validation')
        
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Data validation")
                time.sleep(3)  # Simulate work
                self._complete_phase('data_validation', artifacts=['validation_report.json'])
                return True
            
            # Initialize data validator if not already done
            if not self.data_validator:
                self.data_validator = DataValidator()
            
            # Run data validation
            validation_report = self.data_validator.validate_all_data(
                self.config.source_database['path'],
                self.config.target_database,
                sample_size=self.config.validation_options['sample_size']
            )
            
            self.logger.info(f"Data validation completed: {validation_report}")
            self._complete_phase('data_validation', artifacts=[validation_report])
            return True
            
        except Exception as e:
            self._complete_phase('data_validation', False, str(e))
            return False
    
    def run_index_creation(self) -> bool:
        """Create indexes and constraints"""
        if not self.config.migration_options['create_indexes']:
            self._skip_phase('index_creation', 'Index creation disabled in configuration')
            return True
        
        phase = self._start_phase('index_creation')
        
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Index creation")
                time.sleep(2)  # Simulate work
                self._complete_phase('index_creation', artifacts=['indexes_created.sql'])
                return True
            
            # Initialize schema converter if not already done
            if not self.schema_converter:
                source_uri = self._construct_database_uri(self.config.source_database)
                target_uri = self._construct_database_uri(self.config.target_database)
                self.schema_converter = PostgreSQLSchemaConverter(source_uri, target_uri)
            
            # Create indexes and constraints
            index_report = self.schema_converter.create_indexes_and_constraints(
                self.config.target_database
            )
            
            self.logger.info(f"Index creation completed: {index_report}")
            self._complete_phase('index_creation', artifacts=[index_report])
            return True
            
        except Exception as e:
            self._complete_phase('index_creation', False, str(e))
            return False
    
    def run_performance_testing(self) -> bool:
        """Run performance tests and benchmarks"""
        if not self.config.performance_options['run_benchmarks']:
            self._skip_phase('performance_testing', 'Performance testing disabled in configuration')
            return True
        
        phase = self._start_phase('performance_testing')
        
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Performance testing")
                time.sleep(4)  # Simulate work
                self._complete_phase('performance_testing', artifacts=['performance_report.json'])
                return True
            
            # Initialize performance components if not already done
            if not self.performance_tester:
                source_uri = self._construct_database_uri(self.config.source_database)
                target_uri = self._construct_database_uri(self.config.target_database)
                self.performance_tester = PerformanceTester(source_uri, target_uri)
            if not self.database_benchmark:
                source_uri = self._construct_database_uri(self.config.source_database)
                target_uri = self._construct_database_uri(self.config.target_database)
                self.database_benchmark = DatabaseBenchmark(source_uri, target_uri)
            
            artifacts = []
            
            # Run performance tests
            perf_report = self.performance_tester.run_comprehensive_tests(
                self.config.target_database
            )
            artifacts.append(perf_report)
            
            # Run benchmarks
            benchmark_report = self.database_benchmark.run_comprehensive_benchmarks()
            artifacts.append(benchmark_report)
            
            self.logger.info(f"Performance testing completed")
            self._complete_phase('performance_testing', artifacts=artifacts)
            return True
            
        except Exception as e:
            self._complete_phase('performance_testing', False, str(e))
            return False
    
    def run_final_validation(self) -> bool:
        """Run comprehensive migration verification"""
        phase = self._start_phase('final_validation')
        
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Final validation")
                time.sleep(3)  # Simulate work
                self._complete_phase('final_validation', artifacts=['final_validation_report.json'])
                return True
            
            # Initialize migration verifier if not already done
            if not self.migration_verifier:
                self.migration_verifier = MigrationVerifier()
            
            # Run comprehensive verification
            summary = self.migration_verifier.run_all_verifications()
            
            # Generate detailed report
            report_file = self.migration_verifier.generate_report()
            
            # Check if migration was successful
            success = summary.overall_status in ['PASS', 'WARN']
            
            if not success:
                self.rollback_required = True
            
            self.logger.info(f"Final validation completed: {summary.overall_status}")
            self._complete_phase('final_validation', success, 
                               None if success else f"Validation failed: {summary.overall_status}",
                               [report_file])
            return success
            
        except Exception as e:
            self._complete_phase('final_validation', False, str(e))
            self.rollback_required = True
            return False
    
    def run_health_check(self) -> bool:
        """Run system health and readiness verification"""
        phase = self._start_phase('health_check')
        
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Health check")
                time.sleep(2)  # Simulate work
                self._complete_phase('health_check', artifacts=['health_check_report.json'])
                return True
            
            # Initialize health checker if not already done
            if not self.health_checker:
                self.health_checker = PostgreSQLHealthChecker()
            
            # Run health checks
            health_report = self.health_checker.run_comprehensive_health_check(
                self.config.target_database
            )
            
            # Check if system is healthy
            success = health_report.get('overall_status') == 'healthy'
            
            self.logger.info(f"Health check completed: {health_report.get('overall_status')}")
            self._complete_phase('health_check', success,
                               None if success else "System health check failed",
                               [health_report.get('report_file')])
            return success
            
        except Exception as e:
            self._complete_phase('health_check', False, str(e))
            return False
    
    def run_migration_certification(self) -> bool:
        """Generate migration completion certificate"""
        phase = self._start_phase('migration_certification')
        
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Migration certification")
                time.sleep(1)  # Simulate work
                self._complete_phase('migration_certification', artifacts=['migration_certificate.txt'])
                return True
            
            # Initialize migration verifier if not already done
            if not self.migration_verifier:
                self.migration_verifier = MigrationVerifier()
            
            # Generate migration certificate
            cert_file = self.migration_verifier.generate_certificate()
            
            self.logger.info(f"Migration certificate generated: {cert_file}")
            self._complete_phase('migration_certification', artifacts=[cert_file])
            return True
            
        except Exception as e:
            self._complete_phase('migration_certification', False, str(e))
            return False
    
    def run_rollback(self) -> bool:
        """Run migration rollback"""
        self.logger.warning("Initiating migration rollback...")
        
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Migration rollback")
                return True
            
            # Initialize rollback manager if not already done
            if not self.rollback_manager:
                source_uri = self._construct_database_uri(self.config.source_database)
                target_uri = self._construct_database_uri(self.config.target_database)
                self.rollback_manager = RollbackManager(source_uri, target_uri)
            
            # Create rollback plan
            migration_id = f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            rollback_plan = self.rollback_manager.create_rollback_plan(
                migration_id,
                RollbackPhase.DATA_MIGRATION  # Roll back to before data migration
            )
            
            # Execute rollback
            rollback_success = self.rollback_manager.execute_rollback_plan(rollback_plan.plan_id)
            
            if rollback_success:
                self.logger.info("Migration rollback completed successfully")
            else:
                self.logger.error("Migration rollback failed")
            
            return rollback_success
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def run_migration(self) -> bool:
        """Run the complete migration process"""
        self.logger.info("Starting PostgreSQL migration process...")
        
        try:
            # Initialize components
            if not self.dry_run:
                self._initialize_components()
            
            # Define migration workflow
            migration_steps = [
                ('pre_migration_validation', self.run_pre_migration_validation),
                ('backup_creation', self.run_backup_creation),
                ('schema_conversion', self.run_schema_conversion),
                ('data_migration', self.run_data_migration),
                ('data_validation', self.run_data_validation),
                ('index_creation', self.run_index_creation),
                ('performance_testing', self.run_performance_testing),
                ('final_validation', self.run_final_validation),
                ('health_check', self.run_health_check),
                ('migration_certification', self.run_migration_certification)
            ]
            
            # Execute migration steps
            for step_name, step_function in migration_steps:
                phase = next((p for p in self.phases if p.name == step_name), None)
                
                # Skip non-required phases if they're not needed
                if phase and not phase.required:
                    if not self._should_run_phase(step_name):
                        self._skip_phase(step_name, 'Phase not required by configuration')
                        continue
                
                # Execute the step
                success = step_function()
                
                if not success:
                    self.logger.error(f"Migration step failed: {step_name}")
                    
                    # Check if rollback is required
                    if self.rollback_required or step_name in ['data_migration', 'final_validation']:
                        self.logger.warning("Critical failure detected, initiating rollback...")
                        rollback_success = self.run_rollback()
                        if not rollback_success:
                            self.logger.error("Rollback failed - manual intervention required")
                        return False
                    
                    # For non-critical failures, continue but mark migration as problematic
                    self.logger.warning(f"Non-critical failure in {step_name}, continuing migration...")
            
            # Check overall migration success
            critical_phases = ['schema_conversion', 'data_migration', 'final_validation']
            critical_failures = [
                p for p in self.phases 
                if p.name in critical_phases and not p.completed
            ]
            
            if critical_failures:
                self.logger.error(f"Critical phases failed: {[p.name for p in critical_failures]}")
                self.migration_successful = False
                return False
            
            # Migration completed successfully
            self.migration_successful = True
            total_duration = (datetime.now() - self.start_time).total_seconds()
            
            self.logger.info(f"Migration completed successfully in {total_duration:.2f} seconds")
            self._generate_migration_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed with exception: {e}")
            
            # Attempt rollback on critical failure
            try:
                self.run_rollback()
            except Exception as rollback_error:
                self.logger.error(f"Rollback also failed: {rollback_error}")
            
            return False
    
    def _should_run_phase(self, phase_name: str) -> bool:
        """Determine if a phase should be run based on configuration"""
        phase_config_map = {
            'backup_creation': self.config.backup_options['create_source_backup'],
            'data_validation': self.config.validation_options['validate_data_integrity'],
            'index_creation': self.config.migration_options['create_indexes'],
            'performance_testing': self.config.performance_options['run_benchmarks']
        }
        
        return phase_config_map.get(phase_name, True)
    
    def _generate_migration_summary(self):
        """Generate migration summary report"""
        summary = {
            'migration_id': self.migration_id,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'migration_successful': self.migration_successful,
            'dry_run': self.dry_run,
            'phases': [asdict(phase) for phase in self.phases],
            'configuration': asdict(self.config)
        }
        
        # Save summary to file
        summary_dir = 'reports/migration'
        os.makedirs(summary_dir, exist_ok=True)
        
        summary_file = os.path.join(summary_dir, f'migration_summary_{self.migration_id}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Migration summary saved to: {summary_file}")
        
        # Print summary to console
        print(f"\n{'='*80}")
        print("MIGRATION SUMMARY")
        print(f"{'='*80}")
        print(f"Migration ID: {self.migration_id}")
        print(f"Status: {'SUCCESS' if self.migration_successful else 'FAILED'}")
        print(f"Duration: {summary['total_duration_seconds']:.2f} seconds")
        print(f"Dry Run: {self.dry_run}")
        print(f"\nPhase Results:")
        
        for phase in self.phases:
            status_symbol = {
                'completed': '✓',
                'failed': '✗',
                'skipped': '-',
                'pending': '○'
            }.get(phase.status, '?')
            
            print(f"  {status_symbol} {phase.description} ({phase.status})")
            if phase.duration_seconds > 0:
                print(f"    Duration: {phase.duration_seconds:.2f}s")
            if phase.error_message:
                print(f"    Error: {phase.error_message}")
        
        print(f"\nDetailed report: {summary_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PostgreSQL Migration Orchestrator')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode (no actual changes)')
    parser.add_argument('--skip-backup', action='store_true', help='Skip backup creation')
    parser.add_argument('--skip-validation', action='store_true', help='Skip data validation')
    parser.add_argument('--skip-performance', action='store_true', help='Skip performance testing')
    parser.add_argument('--rollback-only', action='store_true', help='Only run rollback procedure')
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = MigrationOrchestrator(args.config, args.dry_run)
        
        # Apply command line overrides
        if args.skip_backup:
            orchestrator.config.backup_options['create_source_backup'] = False
        if args.skip_validation:
            orchestrator.config.validation_options['validate_data_integrity'] = False
        if args.skip_performance:
            orchestrator.config.performance_options['run_benchmarks'] = False
        
        # Run rollback only if requested
        if args.rollback_only:
            success = orchestrator.run_rollback()
            sys.exit(0 if success else 1)
        
        # Run migration
        success = orchestrator.run_migration()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nMigration interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Migration orchestrator failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()