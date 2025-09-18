#!/usr/bin/env python3
"""
Data Validation Script for PostgreSQL Migration
Comprehensive validation of migrated data integrity and quality
"""

import os
import sys
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.database_config import DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    table_name: str
    passed: bool
    expected_value: Any = None
    actual_value: Any = None
    error_message: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class TableValidationSummary:
    """Summary of validation results for a table"""
    table_name: str
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    critical_failures: int = 0
    warnings: int = 0
    validation_results: List[ValidationResult] = None
    
    def __post_init__(self):
        if self.validation_results is None:
            self.validation_results = []
    
    @property
    def success_rate(self) -> float:
        if self.total_checks == 0:
            return 100.0
        return (self.passed_checks / self.total_checks) * 100
    
    @property
    def is_valid(self) -> bool:
        return self.critical_failures == 0 and self.success_rate >= 95.0

class DataValidator:
    """Comprehensive data validation for PostgreSQL migration"""
    
    def __init__(self, source_db_uri: str, target_db_uri: str):
        self.source_db_uri = source_db_uri
        self.target_db_uri = target_db_uri
        
        # Database connections
        self.source_engine = create_engine(source_db_uri)
        self.target_engine = create_engine(target_db_uri)
        
        # PostgreSQL connection
        self.pg_conn = None
        self.pg_cursor = None
        
        # Validation results
        self.validation_summaries: Dict[str, TableValidationSummary] = {}
        
        # Critical validation rules
        self.critical_validations = {
            'record_count_match',
            'primary_key_integrity',
            'foreign_key_integrity',
            'required_fields_not_null'
        }
        
        # Data quality checks
        self.data_quality_checks = {
            'users': [
                ('email_format', self._validate_email_format),
                ('username_uniqueness', self._validate_username_uniqueness),
                ('active_status_consistency', self._validate_active_status)
            ],
            'horses': [
                ('age_range', self._validate_horse_age_range),
                ('performance_consistency', self._validate_horse_performance),
                ('rating_range', self._validate_horse_rating),
                ('name_not_empty', self._validate_horse_name)
            ],
            'races': [
                ('date_validity', self._validate_race_date),
                ('distance_positive', self._validate_race_distance),
                ('prize_money_non_negative', self._validate_prize_money),
                ('track_not_empty', self._validate_track_name)
            ],
            'predictions': [
                ('confidence_range', self._validate_confidence_range),
                ('position_positive', self._validate_predicted_position),
                ('race_horse_exists', self._validate_race_horse_references)
            ]
        }
    
    def connect_postgresql(self) -> bool:
        """Establish PostgreSQL connection"""
        try:
            import urllib.parse as urlparse
            parsed = urlparse.urlparse(self.target_db_uri)
            
            self.pg_conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password
            )
            self.pg_cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            logger.info("PostgreSQL connection established for validation")
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
    
    def validate_record_count(self, table_name: str) -> ValidationResult:
        """Validate that record counts match between source and target"""
        try:
            # Get source count
            with self.source_engine.connect() as conn:
                source_result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table_name}"))
                source_count = source_result.scalar()
            
            # Get target count
            self.pg_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            target_count = self.pg_cursor.fetchone()[0]
            
            passed = source_count == target_count
            
            return ValidationResult(
                check_name="record_count_match",
                table_name=table_name,
                passed=passed,
                expected_value=source_count,
                actual_value=target_count,
                error_message="" if passed else f"Record count mismatch: source={source_count}, target={target_count}",
                details={"source_count": source_count, "target_count": target_count}
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="record_count_match",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate record count: {e}"
            )
    
    def validate_primary_key_integrity(self, table_name: str) -> ValidationResult:
        """Validate primary key integrity"""
        try:
            # Check for duplicate primary keys
            self.pg_cursor.execute(f"""
                SELECT id, COUNT(*) as count 
                FROM {table_name} 
                GROUP BY id 
                HAVING COUNT(*) > 1
            """)
            duplicates = self.pg_cursor.fetchall()
            
            # Check for null primary keys
            self.pg_cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE id IS NULL")
            null_count = self.pg_cursor.fetchone()[0]
            
            passed = len(duplicates) == 0 and null_count == 0
            
            error_message = ""
            if len(duplicates) > 0:
                error_message += f"Found {len(duplicates)} duplicate primary keys. "
            if null_count > 0:
                error_message += f"Found {null_count} null primary keys."
            
            return ValidationResult(
                check_name="primary_key_integrity",
                table_name=table_name,
                passed=passed,
                error_message=error_message,
                details={"duplicate_count": len(duplicates), "null_count": null_count}
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="primary_key_integrity",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate primary key integrity: {e}"
            )
    
    def validate_foreign_key_integrity(self, table_name: str) -> ValidationResult:
        """Validate foreign key integrity"""
        try:
            foreign_key_checks = []
            
            if table_name == 'predictions':
                # Check race_id references
                self.pg_cursor.execute("""
                    SELECT COUNT(*) FROM predictions p 
                    LEFT JOIN races r ON p.race_id = r.id 
                    WHERE p.race_id IS NOT NULL AND r.id IS NULL
                """)
                orphaned_race_refs = self.pg_cursor.fetchone()[0]
                foreign_key_checks.append(("race_id", orphaned_race_refs))
                
                # Check horse_id references
                self.pg_cursor.execute("""
                    SELECT COUNT(*) FROM predictions p 
                    LEFT JOIN horses h ON p.horse_id = h.id 
                    WHERE p.horse_id IS NOT NULL AND h.id IS NULL
                """)
                orphaned_horse_refs = self.pg_cursor.fetchone()[0]
                foreign_key_checks.append(("horse_id", orphaned_horse_refs))
            
            total_orphaned = sum(count for _, count in foreign_key_checks)
            passed = total_orphaned == 0
            
            error_message = ""
            if not passed:
                error_details = [f"{fk}: {count} orphaned" for fk, count in foreign_key_checks if count > 0]
                error_message = f"Foreign key violations: {', '.join(error_details)}"
            
            return ValidationResult(
                check_name="foreign_key_integrity",
                table_name=table_name,
                passed=passed,
                error_message=error_message,
                details={"foreign_key_checks": dict(foreign_key_checks)}
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="foreign_key_integrity",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate foreign key integrity: {e}"
            )
    
    def validate_required_fields(self, table_name: str) -> ValidationResult:
        """Validate that required fields are not null"""
        try:
            required_fields = {
                'users': ['username', 'email'],
                'horses': ['name'],
                'races': ['name', 'date', 'track'],
                'predictions': ['race_id', 'horse_id']
            }
            
            if table_name not in required_fields:
                return ValidationResult(
                    check_name="required_fields_not_null",
                    table_name=table_name,
                    passed=True,
                    details={"message": "No required fields defined for this table"}
                )
            
            null_counts = {}
            for field in required_fields[table_name]:
                self.pg_cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {field} IS NULL")
                null_counts[field] = self.pg_cursor.fetchone()[0]
            
            total_nulls = sum(null_counts.values())
            passed = total_nulls == 0
            
            error_message = ""
            if not passed:
                null_fields = [f"{field}: {count}" for field, count in null_counts.items() if count > 0]
                error_message = f"Required fields with null values: {', '.join(null_fields)}"
            
            return ValidationResult(
                check_name="required_fields_not_null",
                table_name=table_name,
                passed=passed,
                error_message=error_message,
                details={"null_counts": null_counts}
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="required_fields_not_null",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate required fields: {e}"
            )
    
    def validate_data_checksums(self, table_name: str) -> ValidationResult:
        """Validate data integrity using checksums"""
        try:
            # Get records with checksums from target
            self.pg_cursor.execute(f"SELECT id, _migration_checksum FROM {table_name} WHERE _migration_checksum IS NOT NULL")
            target_checksums = {row['id']: row['_migration_checksum'] for row in self.pg_cursor.fetchall()}
            
            if not target_checksums:
                return ValidationResult(
                    check_name="data_checksum_validation",
                    table_name=table_name,
                    passed=True,
                    details={"message": "No checksums available for validation"}
                )
            
            # Calculate checksums from source data
            mismatched_checksums = 0
            with self.source_engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {table_name}"))
                for row in result:
                    record = dict(row._mapping)
                    record_id = record['id']
                    
                    if record_id in target_checksums:
                        # Calculate expected checksum
                        sorted_items = sorted(record.items())
                        record_str = json.dumps(sorted_items, default=str, sort_keys=True)
                        expected_checksum = hashlib.md5(record_str.encode()).hexdigest()
                        
                        if target_checksums[record_id] != expected_checksum:
                            mismatched_checksums += 1
            
            passed = mismatched_checksums == 0
            
            return ValidationResult(
                check_name="data_checksum_validation",
                table_name=table_name,
                passed=passed,
                expected_value=0,
                actual_value=mismatched_checksums,
                error_message="" if passed else f"Found {mismatched_checksums} records with checksum mismatches",
                details={"total_checked": len(target_checksums), "mismatched": mismatched_checksums}
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="data_checksum_validation",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate checksums: {e}"
            )
    
    # Data quality validation methods
    def _validate_email_format(self, table_name: str) -> ValidationResult:
        """Validate email format in users table"""
        try:
            self.pg_cursor.execute("""
                SELECT COUNT(*) FROM users 
                WHERE email IS NOT NULL 
                AND email !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
            """)
            invalid_emails = self.pg_cursor.fetchone()[0]
            
            passed = invalid_emails == 0
            
            return ValidationResult(
                check_name="email_format",
                table_name=table_name,
                passed=passed,
                actual_value=invalid_emails,
                error_message="" if passed else f"Found {invalid_emails} invalid email formats"
            )
        except Exception as e:
            return ValidationResult(
                check_name="email_format",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate email format: {e}"
            )
    
    def _validate_username_uniqueness(self, table_name: str) -> ValidationResult:
        """Validate username uniqueness"""
        try:
            self.pg_cursor.execute("""
                SELECT username, COUNT(*) as count 
                FROM users 
                WHERE username IS NOT NULL 
                GROUP BY username 
                HAVING COUNT(*) > 1
            """)
            duplicate_usernames = len(self.pg_cursor.fetchall())
            
            passed = duplicate_usernames == 0
            
            return ValidationResult(
                check_name="username_uniqueness",
                table_name=table_name,
                passed=passed,
                actual_value=duplicate_usernames,
                error_message="" if passed else f"Found {duplicate_usernames} duplicate usernames"
            )
        except Exception as e:
            return ValidationResult(
                check_name="username_uniqueness",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate username uniqueness: {e}"
            )
    
    def _validate_active_status(self, table_name: str) -> ValidationResult:
        """Validate active status consistency"""
        try:
            self.pg_cursor.execute("SELECT COUNT(*) FROM users WHERE is_active IS NULL")
            null_status_count = self.pg_cursor.fetchone()[0]
            
            passed = null_status_count == 0
            
            return ValidationResult(
                check_name="active_status_consistency",
                table_name=table_name,
                passed=passed,
                actual_value=null_status_count,
                error_message="" if passed else f"Found {null_status_count} users with null active status"
            )
        except Exception as e:
            return ValidationResult(
                check_name="active_status_consistency",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate active status: {e}"
            )
    
    def _validate_horse_age_range(self, table_name: str) -> ValidationResult:
        """Validate horse age is within reasonable range"""
        try:
            self.pg_cursor.execute("SELECT COUNT(*) FROM horses WHERE age < 1 OR age > 30")
            invalid_ages = self.pg_cursor.fetchone()[0]
            
            passed = invalid_ages == 0
            
            return ValidationResult(
                check_name="age_range",
                table_name=table_name,
                passed=passed,
                actual_value=invalid_ages,
                error_message="" if passed else f"Found {invalid_ages} horses with invalid ages"
            )
        except Exception as e:
            return ValidationResult(
                check_name="age_range",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate horse age range: {e}"
            )
    
    def _validate_horse_performance(self, table_name: str) -> ValidationResult:
        """Validate horse performance consistency (wins <= places <= runs)"""
        try:
            self.pg_cursor.execute("""
                SELECT COUNT(*) FROM horses 
                WHERE wins > places OR places > runs OR wins < 0 OR places < 0 OR runs < 0
            """)
            inconsistent_performance = self.pg_cursor.fetchone()[0]
            
            passed = inconsistent_performance == 0
            
            return ValidationResult(
                check_name="performance_consistency",
                table_name=table_name,
                passed=passed,
                actual_value=inconsistent_performance,
                error_message="" if passed else f"Found {inconsistent_performance} horses with inconsistent performance data"
            )
        except Exception as e:
            return ValidationResult(
                check_name="performance_consistency",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate horse performance: {e}"
            )
    
    def _validate_horse_rating(self, table_name: str) -> ValidationResult:
        """Validate horse rating is within reasonable range"""
        try:
            self.pg_cursor.execute("SELECT COUNT(*) FROM horses WHERE rating < 0 OR rating > 200")
            invalid_ratings = self.pg_cursor.fetchone()[0]
            
            passed = invalid_ratings == 0
            
            return ValidationResult(
                check_name="rating_range",
                table_name=table_name,
                passed=passed,
                actual_value=invalid_ratings,
                error_message="" if passed else f"Found {invalid_ratings} horses with invalid ratings"
            )
        except Exception as e:
            return ValidationResult(
                check_name="rating_range",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate horse rating: {e}"
            )
    
    def _validate_horse_name(self, table_name: str) -> ValidationResult:
        """Validate horse name is not empty"""
        try:
            self.pg_cursor.execute("SELECT COUNT(*) FROM horses WHERE name IS NULL OR TRIM(name) = ''")
            empty_names = self.pg_cursor.fetchone()[0]
            
            passed = empty_names == 0
            
            return ValidationResult(
                check_name="name_not_empty",
                table_name=table_name,
                passed=passed,
                actual_value=empty_names,
                error_message="" if passed else f"Found {empty_names} horses with empty names"
            )
        except Exception as e:
            return ValidationResult(
                check_name="name_not_empty",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate horse name: {e}"
            )
    
    def _validate_race_date(self, table_name: str) -> ValidationResult:
        """Validate race date is valid"""
        try:
            self.pg_cursor.execute("SELECT COUNT(*) FROM races WHERE date IS NULL")
            null_dates = self.pg_cursor.fetchone()[0]
            
            passed = null_dates == 0
            
            return ValidationResult(
                check_name="date_validity",
                table_name=table_name,
                passed=passed,
                actual_value=null_dates,
                error_message="" if passed else f"Found {null_dates} races with null dates"
            )
        except Exception as e:
            return ValidationResult(
                check_name="date_validity",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate race date: {e}"
            )
    
    def _validate_race_distance(self, table_name: str) -> ValidationResult:
        """Validate race distance is positive"""
        try:
            self.pg_cursor.execute("SELECT COUNT(*) FROM races WHERE distance <= 0")
            invalid_distances = self.pg_cursor.fetchone()[0]
            
            passed = invalid_distances == 0
            
            return ValidationResult(
                check_name="distance_positive",
                table_name=table_name,
                passed=passed,
                actual_value=invalid_distances,
                error_message="" if passed else f"Found {invalid_distances} races with invalid distances"
            )
        except Exception as e:
            return ValidationResult(
                check_name="distance_positive",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate race distance: {e}"
            )
    
    def _validate_prize_money(self, table_name: str) -> ValidationResult:
        """Validate prize money is non-negative"""
        try:
            self.pg_cursor.execute("SELECT COUNT(*) FROM races WHERE prize_money < 0")
            negative_prize_money = self.pg_cursor.fetchone()[0]
            
            passed = negative_prize_money == 0
            
            return ValidationResult(
                check_name="prize_money_non_negative",
                table_name=table_name,
                passed=passed,
                actual_value=negative_prize_money,
                error_message="" if passed else f"Found {negative_prize_money} races with negative prize money"
            )
        except Exception as e:
            return ValidationResult(
                check_name="prize_money_non_negative",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate prize money: {e}"
            )
    
    def _validate_track_name(self, table_name: str) -> ValidationResult:
        """Validate track name is not empty"""
        try:
            self.pg_cursor.execute("SELECT COUNT(*) FROM races WHERE track IS NULL OR TRIM(track) = ''")
            empty_tracks = self.pg_cursor.fetchone()[0]
            
            passed = empty_tracks == 0
            
            return ValidationResult(
                check_name="track_not_empty",
                table_name=table_name,
                passed=passed,
                actual_value=empty_tracks,
                error_message="" if passed else f"Found {empty_tracks} races with empty track names"
            )
        except Exception as e:
            return ValidationResult(
                check_name="track_not_empty",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate track name: {e}"
            )
    
    def _validate_confidence_range(self, table_name: str) -> ValidationResult:
        """Validate prediction confidence is between 0 and 1"""
        try:
            self.pg_cursor.execute("SELECT COUNT(*) FROM predictions WHERE confidence < 0 OR confidence > 1")
            invalid_confidence = self.pg_cursor.fetchone()[0]
            
            passed = invalid_confidence == 0
            
            return ValidationResult(
                check_name="confidence_range",
                table_name=table_name,
                passed=passed,
                actual_value=invalid_confidence,
                error_message="" if passed else f"Found {invalid_confidence} predictions with invalid confidence values"
            )
        except Exception as e:
            return ValidationResult(
                check_name="confidence_range",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate confidence range: {e}"
            )
    
    def _validate_predicted_position(self, table_name: str) -> ValidationResult:
        """Validate predicted position is positive"""
        try:
            self.pg_cursor.execute("SELECT COUNT(*) FROM predictions WHERE predicted_position <= 0")
            invalid_positions = self.pg_cursor.fetchone()[0]
            
            passed = invalid_positions == 0
            
            return ValidationResult(
                check_name="position_positive",
                table_name=table_name,
                passed=passed,
                actual_value=invalid_positions,
                error_message="" if passed else f"Found {invalid_positions} predictions with invalid positions"
            )
        except Exception as e:
            return ValidationResult(
                check_name="position_positive",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate predicted position: {e}"
            )
    
    def _validate_race_horse_references(self, table_name: str) -> ValidationResult:
        """Validate that race and horse references exist"""
        try:
            # This is covered by foreign key validation, but we can add specific checks
            self.pg_cursor.execute("""
                SELECT COUNT(*) FROM predictions p
                WHERE NOT EXISTS (SELECT 1 FROM races r WHERE r.id = p.race_id)
                   OR NOT EXISTS (SELECT 1 FROM horses h WHERE h.id = p.horse_id)
            """)
            invalid_references = self.pg_cursor.fetchone()[0]
            
            passed = invalid_references == 0
            
            return ValidationResult(
                check_name="race_horse_exists",
                table_name=table_name,
                passed=passed,
                actual_value=invalid_references,
                error_message="" if passed else f"Found {invalid_references} predictions with invalid race/horse references"
            )
        except Exception as e:
            return ValidationResult(
                check_name="race_horse_exists",
                table_name=table_name,
                passed=False,
                error_message=f"Failed to validate race/horse references: {e}"
            )
    
    def validate_table(self, table_name: str) -> TableValidationSummary:
        """Perform comprehensive validation for a table"""
        logger.info(f"Starting validation for table: {table_name}")
        
        summary = TableValidationSummary(table_name)
        
        # Core integrity checks
        core_checks = [
            self.validate_record_count,
            self.validate_primary_key_integrity,
            self.validate_foreign_key_integrity,
            self.validate_required_fields,
            self.validate_data_checksums
        ]
        
        for check_func in core_checks:
            try:
                result = check_func(table_name)
                summary.validation_results.append(result)
                summary.total_checks += 1
                
                if result.passed:
                    summary.passed_checks += 1
                else:
                    summary.failed_checks += 1
                    if result.check_name in self.critical_validations:
                        summary.critical_failures += 1
                    else:
                        summary.warnings += 1
                        
            except Exception as e:
                logger.error(f"Validation check failed for {table_name}: {e}")
                summary.failed_checks += 1
                summary.critical_failures += 1
        
        # Data quality checks
        if table_name in self.data_quality_checks:
            for check_name, check_func in self.data_quality_checks[table_name]:
                try:
                    result = check_func(table_name)
                    summary.validation_results.append(result)
                    summary.total_checks += 1
                    
                    if result.passed:
                        summary.passed_checks += 1
                    else:
                        summary.failed_checks += 1
                        summary.warnings += 1
                        
                except Exception as e:
                    logger.error(f"Data quality check {check_name} failed for {table_name}: {e}")
                    summary.failed_checks += 1
                    summary.warnings += 1
        
        logger.info(f"Validation completed for {table_name}: "
                   f"{summary.passed_checks}/{summary.total_checks} checks passed "
                   f"({summary.success_rate:.1f}% success rate)")
        
        return summary
    
    def validate_all_data(self, tables: List[str] = None) -> Dict[str, Any]:
        """Validate all data integrity and completeness - main method called by migration orchestrator"""
        return self.validate_all_tables(tables)
    
    def validate_all_tables(self, tables: List[str] = None) -> Dict[str, Any]:
        """Validate all specified tables"""
        if tables is None:
            tables = ['users', 'horses', 'races', 'predictions', 'api_credentials']
        
        logger.info(f"Starting validation for tables: {tables}")
        
        if not self.connect_postgresql():
            return {"error": "Failed to connect to PostgreSQL"}
        
        try:
            # Validate each table
            for table_name in tables:
                summary = self.validate_table(table_name)
                self.validation_summaries[table_name] = summary
            
            # Generate overall report
            report = self.generate_validation_report()
            
            # Save report
            report_path = "scripts/migration/validation/validation_report.json"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Validation completed. Report saved to: {report_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Validation process failed: {e}")
            return {"error": str(e)}
        
        finally:
            self.disconnect_postgresql()
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_checks = sum(summary.total_checks for summary in self.validation_summaries.values())
        total_passed = sum(summary.passed_checks for summary in self.validation_summaries.values())
        total_failed = sum(summary.failed_checks for summary in self.validation_summaries.values())
        total_critical = sum(summary.critical_failures for summary in self.validation_summaries.values())
        total_warnings = sum(summary.warnings for summary in self.validation_summaries.values())
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'source_database': self.source_db_uri,
            'target_database': self.target_db_uri,
            'summary': {
                'total_tables': len(self.validation_summaries),
                'total_checks': total_checks,
                'total_passed': total_passed,
                'total_failed': total_failed,
                'critical_failures': total_critical,
                'warnings': total_warnings,
                'overall_success_rate': (total_passed / total_checks * 100) if total_checks > 0 else 0,
                'migration_valid': total_critical == 0 and (total_passed / total_checks) >= 0.95 if total_checks > 0 else False
            },
            'table_summaries': {},
            'failed_checks': []
        }
        
        for table_name, summary in self.validation_summaries.items():
            report['table_summaries'][table_name] = {
                'total_checks': summary.total_checks,
                'passed_checks': summary.passed_checks,
                'failed_checks': summary.failed_checks,
                'critical_failures': summary.critical_failures,
                'warnings': summary.warnings,
                'success_rate': summary.success_rate,
                'is_valid': summary.is_valid
            }
            
            # Collect failed checks
            for result in summary.validation_results:
                if not result.passed:
                    report['failed_checks'].append({
                        'table': table_name,
                        'check': result.check_name,
                        'error': result.error_message,
                        'critical': result.check_name in self.critical_validations
                    })
        
        return report

def main():
    """Main validation process"""
    logger.info("Starting PostgreSQL data validation...")
    
    try:
        # Configuration
        source_db_uri = DatabaseConfig.SQLALCHEMY_DATABASE_URI
        target_db_uri = os.getenv('POSTGRES_URL', 'postgresql://hrp_user:password@localhost:5432/hrp_database')
        
        # Initialize validator
        validator = DataValidator(source_db_uri, target_db_uri)
        
        # Run validation
        report = validator.validate_all_tables()
        
        if 'error' in report:
            logger.error(f"Validation failed: {report['error']}")
            return False
        
        # Check results
        if report['summary']['migration_valid']:
            logger.info("Data validation completed successfully! Migration is valid.")
            return True
        else:
            logger.warning(f"Data validation completed with issues: "
                          f"{report['summary']['critical_failures']} critical failures, "
                          f"{report['summary']['warnings']} warnings")
            return False
            
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)