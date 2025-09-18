#!/usr/bin/env python3
"""
Migration Rollback Manager
Comprehensive rollback procedures for PostgreSQL migration failures
"""

import os
import sys
import json
import logging
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import psycopg2
from sqlalchemy import create_engine, text

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.database_config import DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rollback_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RollbackPhase(Enum):
    """Migration phases that can be rolled back"""
    SCHEMA_CREATION = "schema_creation"
    DATA_MIGRATION = "data_migration"
    VALIDATION = "validation"
    CUTOVER = "cutover"
    POST_MIGRATION = "post_migration"

class RollbackAction(Enum):
    """Types of rollback actions"""
    RESTORE_DATABASE = "restore_database"
    RESTORE_SCHEMA = "restore_schema"
    RESTORE_DATA = "restore_data"
    REVERT_CONFIG = "revert_config"
    CLEANUP_FILES = "cleanup_files"
    RESTART_SERVICES = "restart_services"

@dataclass
class RollbackStep:
    """Individual rollback step"""
    step_id: str
    phase: RollbackPhase
    action: RollbackAction
    description: str
    command: Optional[str] = None
    sql_script: Optional[str] = None
    backup_file: Optional[str] = None
    config_file: Optional[str] = None
    dependencies: List[str] = None
    timeout: int = 300
    critical: bool = True
    executed: bool = False
    success: bool = False
    error_message: Optional[str] = None
    execution_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class RollbackPlan:
    """Complete rollback plan"""
    plan_id: str
    migration_id: str
    created_at: datetime
    target_phase: RollbackPhase
    steps: List[RollbackStep]
    metadata: Dict[str, Any]
    executed: bool = False
    success: bool = False
    execution_start: Optional[datetime] = None
    execution_end: Optional[datetime] = None

class RollbackManager:
    """Manages migration rollback procedures"""
    
    def __init__(self, source_db_uri: str, target_db_uri: str, backup_dir: str = "backups"):
        self.source_db_uri = source_db_uri
        self.target_db_uri = target_db_uri
        self.backup_dir = Path(backup_dir)
        
        # Create rollback directory structure
        self.rollback_dir = Path("scripts/migration/rollback")
        self.rollback_dir.mkdir(exist_ok=True)
        
        self.plans_dir = self.rollback_dir / "plans"
        self.plans_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Database connections
        self.source_engine = create_engine(source_db_uri)
        self.target_engine = create_engine(target_db_uri)
        
        # Rollback registry
        self.registry_file = self.rollback_dir / "rollback_registry.json"
        self.registry = self._load_registry()
        
        # PostgreSQL connection details
        self.pg_conn_params = self._parse_postgres_uri(target_db_uri)
    
    def _parse_postgres_uri(self, uri: str) -> Dict[str, str]:
        """Parse PostgreSQL connection URI"""
        import urllib.parse as urlparse
        parsed = urlparse.urlparse(uri)
        
        return {
            'host': parsed.hostname or 'localhost',
            'port': str(parsed.port or 5432),
            'database': parsed.path[1:] if parsed.path else '',
            'user': parsed.username or '',
            'password': parsed.password or ''
        }
    
    def _load_registry(self) -> Dict[str, RollbackPlan]:
        """Load rollback registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    registry = {}
                    for plan_id, plan_data in data.items():
                        # Convert datetime strings back to datetime objects
                        plan_data['created_at'] = datetime.fromisoformat(plan_data['created_at'])
                        if plan_data.get('execution_start'):
                            plan_data['execution_start'] = datetime.fromisoformat(plan_data['execution_start'])
                        if plan_data.get('execution_end'):
                            plan_data['execution_end'] = datetime.fromisoformat(plan_data['execution_end'])
                        
                        # Convert steps
                        steps = []
                        for step_data in plan_data['steps']:
                            step_data['phase'] = RollbackPhase(step_data['phase'])
                            step_data['action'] = RollbackAction(step_data['action'])
                            if step_data.get('execution_time'):
                                step_data['execution_time'] = datetime.fromisoformat(step_data['execution_time'])
                            steps.append(RollbackStep(**step_data))
                        
                        plan_data['steps'] = steps
                        plan_data['target_phase'] = RollbackPhase(plan_data['target_phase'])
                        
                        registry[plan_id] = RollbackPlan(**plan_data)
                    
                    return registry
            except Exception as e:
                logger.warning(f"Failed to load rollback registry: {e}")
        
        return {}
    
    def _save_registry(self):
        """Save rollback registry to file"""
        try:
            data = {}
            for plan_id, plan in self.registry.items():
                plan_dict = asdict(plan)
                
                # Convert datetime objects to strings
                plan_dict['created_at'] = plan.created_at.isoformat()
                if plan.execution_start:
                    plan_dict['execution_start'] = plan.execution_start.isoformat()
                if plan.execution_end:
                    plan_dict['execution_end'] = plan.execution_end.isoformat()
                
                # Convert enum values to strings
                plan_dict['target_phase'] = plan.target_phase.value
                
                # Convert steps
                steps_data = []
                for step in plan.steps:
                    step_dict = asdict(step)
                    step_dict['phase'] = step.phase.value
                    step_dict['action'] = step.action.value
                    if step.execution_time:
                        step_dict['execution_time'] = step.execution_time.isoformat()
                    steps_data.append(step_dict)
                
                plan_dict['steps'] = steps_data
                data[plan_id] = plan_dict
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save rollback registry: {e}")
    
    def create_rollback_plan(self, migration_id: str, target_phase: RollbackPhase, 
                           backup_info: Dict[str, str] = None) -> RollbackPlan:
        """Create a comprehensive rollback plan"""
        logger.info(f"Creating rollback plan for migration {migration_id} to phase {target_phase.value}")
        
        plan_id = f"rollback_{migration_id}_{target_phase.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        steps = []
        
        # Define rollback steps based on target phase
        if target_phase in [RollbackPhase.POST_MIGRATION, RollbackPhase.CUTOVER]:
            steps.extend(self._create_cutover_rollback_steps(backup_info))
        
        if target_phase in [RollbackPhase.POST_MIGRATION, RollbackPhase.CUTOVER, RollbackPhase.VALIDATION]:
            steps.extend(self._create_validation_rollback_steps(backup_info))
        
        if target_phase in [RollbackPhase.POST_MIGRATION, RollbackPhase.CUTOVER, 
                           RollbackPhase.VALIDATION, RollbackPhase.DATA_MIGRATION]:
            steps.extend(self._create_data_migration_rollback_steps(backup_info))
        
        if target_phase in [RollbackPhase.POST_MIGRATION, RollbackPhase.CUTOVER, 
                           RollbackPhase.VALIDATION, RollbackPhase.DATA_MIGRATION, 
                           RollbackPhase.SCHEMA_CREATION]:
            steps.extend(self._create_schema_rollback_steps(backup_info))
        
        # Add cleanup steps
        steps.extend(self._create_cleanup_steps())
        
        # Create rollback plan
        plan = RollbackPlan(
            plan_id=plan_id,
            migration_id=migration_id,
            created_at=datetime.now(),
            target_phase=target_phase,
            steps=steps,
            metadata={
                'backup_info': backup_info or {},
                'source_db_uri': self.source_db_uri,
                'target_db_uri': self.target_db_uri
            }
        )
        
        # Save plan
        self.registry[plan_id] = plan
        self._save_registry()
        
        # Save detailed plan to file
        plan_file = self.plans_dir / f"{plan_id}.json"
        with open(plan_file, 'w') as f:
            json.dump(asdict(plan), f, indent=2, default=str)
        
        logger.info(f"Rollback plan created: {plan_id} with {len(steps)} steps")
        return plan
    
    def _create_cutover_rollback_steps(self, backup_info: Dict[str, str]) -> List[RollbackStep]:
        """Create rollback steps for cutover phase"""
        steps = []
        
        # Stop application services
        steps.append(RollbackStep(
            step_id="stop_application",
            phase=RollbackPhase.CUTOVER,
            action=RollbackAction.RESTART_SERVICES,
            description="Stop application services",
            command="pkill -f 'python.*app.py'",
            critical=False,
            timeout=30
        ))
        
        # Revert database connection configuration
        steps.append(RollbackStep(
            step_id="revert_db_config",
            phase=RollbackPhase.CUTOVER,
            action=RollbackAction.REVERT_CONFIG,
            description="Revert database configuration to SQLite",
            config_file="config/database_config.py",
            critical=True
        ))
        
        # Restart application with original configuration
        steps.append(RollbackStep(
            step_id="restart_application",
            phase=RollbackPhase.CUTOVER,
            action=RollbackAction.RESTART_SERVICES,
            description="Restart application with SQLite configuration",
            command="python3 app.py &",
            critical=True,
            dependencies=["revert_db_config"]
        ))
        
        return steps
    
    def _create_validation_rollback_steps(self, backup_info: Dict[str, str]) -> List[RollbackStep]:
        """Create rollback steps for validation phase"""
        steps = []
        
        # Clear validation results
        steps.append(RollbackStep(
            step_id="clear_validation_results",
            phase=RollbackPhase.VALIDATION,
            action=RollbackAction.CLEANUP_FILES,
            description="Clear validation results and temporary files",
            command="rm -rf logs/validation_* reports/validation_*",
            critical=False
        ))
        
        return steps
    
    def _create_data_migration_rollback_steps(self, backup_info: Dict[str, str]) -> List[RollbackStep]:
        """Create rollback steps for data migration phase"""
        steps = []
        
        # Restore PostgreSQL database from backup
        if backup_info and backup_info.get('postgresql_backup'):
            steps.append(RollbackStep(
                step_id="restore_postgresql_data",
                phase=RollbackPhase.DATA_MIGRATION,
                action=RollbackAction.RESTORE_DATA,
                description="Restore PostgreSQL database from pre-migration backup",
                backup_file=backup_info['postgresql_backup'],
                critical=True,
                timeout=1800  # 30 minutes
            ))
        
        # Clear migration logs
        steps.append(RollbackStep(
            step_id="clear_migration_logs",
            phase=RollbackPhase.DATA_MIGRATION,
            action=RollbackAction.CLEANUP_FILES,
            description="Clear data migration logs and temporary files",
            command="rm -rf logs/data_migration_* temp/migration_*",
            critical=False
        ))
        
        return steps
    
    def _create_schema_rollback_steps(self, backup_info: Dict[str, str]) -> List[RollbackStep]:
        """Create rollback steps for schema creation phase"""
        steps = []
        
        # Drop PostgreSQL schema
        steps.append(RollbackStep(
            step_id="drop_postgresql_schema",
            phase=RollbackPhase.SCHEMA_CREATION,
            action=RollbackAction.RESTORE_SCHEMA,
            description="Drop PostgreSQL schema and objects",
            sql_script="DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;",
            critical=True
        ))
        
        # Restore original PostgreSQL state
        if backup_info and backup_info.get('postgresql_schema_backup'):
            steps.append(RollbackStep(
                step_id="restore_postgresql_schema",
                phase=RollbackPhase.SCHEMA_CREATION,
                action=RollbackAction.RESTORE_SCHEMA,
                description="Restore original PostgreSQL schema",
                backup_file=backup_info['postgresql_schema_backup'],
                critical=True,
                dependencies=["drop_postgresql_schema"]
            ))
        
        return steps
    
    def _create_cleanup_steps(self) -> List[RollbackStep]:
        """Create general cleanup steps"""
        steps = []
        
        # Clean up migration artifacts
        steps.append(RollbackStep(
            step_id="cleanup_migration_artifacts",
            phase=RollbackPhase.SCHEMA_CREATION,
            action=RollbackAction.CLEANUP_FILES,
            description="Clean up migration artifacts and temporary files",
            command="rm -rf temp/migration_* logs/migration_temp_*",
            critical=False
        ))
        
        # Reset file permissions
        steps.append(RollbackStep(
            step_id="reset_permissions",
            phase=RollbackPhase.SCHEMA_CREATION,
            action=RollbackAction.CLEANUP_FILES,
            description="Reset file permissions",
            command="chmod -R 644 config/ && chmod +x scripts/migration/rollback/*.py",
            critical=False
        ))
        
        return steps
    
    def execute_rollback_plan(self, plan_id: str, dry_run: bool = False) -> bool:
        """Execute a rollback plan"""
        if plan_id not in self.registry:
            logger.error(f"Rollback plan not found: {plan_id}")
            return False
        
        plan = self.registry[plan_id]
        
        if dry_run:
            logger.info(f"DRY RUN: Executing rollback plan {plan_id}")
        else:
            logger.info(f"Executing rollback plan {plan_id}")
        
        plan.execution_start = datetime.now()
        plan.executed = True
        
        success = True
        executed_steps = []
        
        try:
            # Sort steps by dependencies
            sorted_steps = self._sort_steps_by_dependencies(plan.steps)
            
            for step in sorted_steps:
                logger.info(f"Executing step: {step.step_id} - {step.description}")
                
                if dry_run:
                    logger.info(f"DRY RUN: Would execute step {step.step_id}")
                    step.executed = True
                    step.success = True
                    step.execution_time = datetime.now()
                    continue
                
                step_success = self._execute_rollback_step(step)
                executed_steps.append(step)
                
                if not step_success and step.critical:
                    logger.error(f"Critical step failed: {step.step_id}")
                    success = False
                    break
                elif not step_success:
                    logger.warning(f"Non-critical step failed: {step.step_id}")
        
        except Exception as e:
            logger.error(f"Rollback execution failed: {e}")
            success = False
        
        finally:
            plan.execution_end = datetime.now()
            plan.success = success
            
            # Save updated plan
            self._save_registry()
            
            # Generate rollback report
            self._generate_rollback_report(plan, executed_steps)
        
        if success:
            logger.info(f"Rollback plan {plan_id} executed successfully")
        else:
            logger.error(f"Rollback plan {plan_id} failed")
        
        return success
    
    def _sort_steps_by_dependencies(self, steps: List[RollbackStep]) -> List[RollbackStep]:
        """Sort steps by their dependencies"""
        sorted_steps = []
        remaining_steps = steps.copy()
        step_ids = {step.step_id for step in steps}
        
        while remaining_steps:
            # Find steps with no unresolved dependencies
            ready_steps = []
            for step in remaining_steps:
                if all(dep in [s.step_id for s in sorted_steps] for dep in step.dependencies):
                    ready_steps.append(step)
            
            if not ready_steps:
                # If no steps are ready, there might be circular dependencies
                # Add remaining steps in order
                logger.warning("Possible circular dependencies detected, adding remaining steps in order")
                sorted_steps.extend(remaining_steps)
                break
            
            # Add ready steps to sorted list
            for step in ready_steps:
                sorted_steps.append(step)
                remaining_steps.remove(step)
        
        return sorted_steps
    
    def _execute_rollback_step(self, step: RollbackStep) -> bool:
        """Execute a single rollback step"""
        step.execution_time = datetime.now()
        step.executed = True
        
        try:
            if step.action == RollbackAction.RESTORE_DATABASE:
                success = self._restore_database(step)
            elif step.action == RollbackAction.RESTORE_SCHEMA:
                success = self._restore_schema(step)
            elif step.action == RollbackAction.RESTORE_DATA:
                success = self._restore_data(step)
            elif step.action == RollbackAction.REVERT_CONFIG:
                success = self._revert_config(step)
            elif step.action == RollbackAction.CLEANUP_FILES:
                success = self._cleanup_files(step)
            elif step.action == RollbackAction.RESTART_SERVICES:
                success = self._restart_services(step)
            else:
                logger.error(f"Unknown rollback action: {step.action}")
                success = False
            
            step.success = success
            
            if success:
                logger.info(f"Step completed successfully: {step.step_id}")
            else:
                logger.error(f"Step failed: {step.step_id}")
            
            return success
            
        except Exception as e:
            step.error_message = str(e)
            step.success = False
            logger.error(f"Step {step.step_id} failed with exception: {e}")
            return False
    
    def _restore_database(self, step: RollbackStep) -> bool:
        """Restore entire database from backup"""
        if not step.backup_file:
            logger.error("No backup file specified for database restore")
            return False
        
        try:
            # Use backup manager to restore
            from scripts.migration.backup.backup_manager import BackupManager
            
            backup_manager = BackupManager(self.source_db_uri, self.target_db_uri)
            
            # Extract backup ID from file path
            backup_id = Path(step.backup_file).stem
            
            # Restore based on database type
            if 'postgresql' in step.backup_file.lower():
                return backup_manager.restore_postgresql_backup(backup_id)
            elif 'sqlite' in step.backup_file.lower():
                return backup_manager.restore_sqlite_backup(backup_id)
            else:
                logger.error(f"Unknown database type in backup file: {step.backup_file}")
                return False
                
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    
    def _restore_schema(self, step: RollbackStep) -> bool:
        """Restore database schema"""
        try:
            if step.sql_script:
                # Execute SQL script directly
                with self.target_engine.connect() as conn:
                    conn.execute(text(step.sql_script))
                    conn.commit()
                return True
            
            elif step.backup_file:
                # Restore from backup file
                return self._restore_database(step)
            
            else:
                logger.error("No SQL script or backup file specified for schema restore")
                return False
                
        except Exception as e:
            logger.error(f"Schema restore failed: {e}")
            return False
    
    def _restore_data(self, step: RollbackStep) -> bool:
        """Restore database data"""
        return self._restore_database(step)
    
    def _revert_config(self, step: RollbackStep) -> bool:
        """Revert configuration files"""
        if not step.config_file:
            logger.error("No config file specified for revert")
            return False
        
        try:
            config_path = Path(step.config_file)
            backup_path = config_path.with_suffix(config_path.suffix + '.backup')
            
            if backup_path.exists():
                shutil.copy2(backup_path, config_path)
                logger.info(f"Reverted config file: {config_path}")
                return True
            else:
                logger.warning(f"No backup found for config file: {config_path}")
                return False
                
        except Exception as e:
            logger.error(f"Config revert failed: {e}")
            return False
    
    def _cleanup_files(self, step: RollbackStep) -> bool:
        """Clean up files and directories"""
        if not step.command:
            logger.error("No command specified for cleanup")
            return False
        
        try:
            result = subprocess.run(
                step.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=step.timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Cleanup completed: {step.command}")
                return True
            else:
                logger.warning(f"Cleanup command failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Cleanup command timed out: {step.command}")
            return False
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False
    
    def _restart_services(self, step: RollbackStep) -> bool:
        """Restart services"""
        if not step.command:
            logger.error("No command specified for service restart")
            return False
        
        try:
            result = subprocess.run(
                step.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=step.timeout
            )
            
            # For service commands, we might not get a return code of 0
            # Check if the command executed without throwing an exception
            logger.info(f"Service command executed: {step.command}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Service command timed out: {step.command}")
            return False
        except Exception as e:
            logger.error(f"Service restart failed: {e}")
            return False
    
    def _generate_rollback_report(self, plan: RollbackPlan, executed_steps: List[RollbackStep]):
        """Generate rollback execution report"""
        report = {
            'plan_id': plan.plan_id,
            'migration_id': plan.migration_id,
            'target_phase': plan.target_phase.value,
            'execution_start': plan.execution_start.isoformat() if plan.execution_start else None,
            'execution_end': plan.execution_end.isoformat() if plan.execution_end else None,
            'success': plan.success,
            'total_steps': len(plan.steps),
            'executed_steps': len(executed_steps),
            'successful_steps': len([s for s in executed_steps if s.success]),
            'failed_steps': len([s for s in executed_steps if not s.success]),
            'step_details': []
        }
        
        for step in executed_steps:
            step_detail = {
                'step_id': step.step_id,
                'description': step.description,
                'action': step.action.value,
                'phase': step.phase.value,
                'success': step.success,
                'execution_time': step.execution_time.isoformat() if step.execution_time else None,
                'error_message': step.error_message
            }
            report['step_details'].append(step_detail)
        
        # Save report
        report_file = self.rollback_dir / f"rollback_report_{plan.plan_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Rollback report saved: {report_file}")
    
    def list_rollback_plans(self) -> List[RollbackPlan]:
        """List all rollback plans"""
        return list(self.registry.values())
    
    def get_rollback_plan(self, plan_id: str) -> Optional[RollbackPlan]:
        """Get specific rollback plan"""
        return self.registry.get(plan_id)
    
    def delete_rollback_plan(self, plan_id: str) -> bool:
        """Delete a rollback plan"""
        if plan_id in self.registry:
            del self.registry[plan_id]
            self._save_registry()
            
            # Remove plan file
            plan_file = self.plans_dir / f"{plan_id}.json"
            if plan_file.exists():
                plan_file.unlink()
            
            logger.info(f"Rollback plan deleted: {plan_id}")
            return True
        
        return False

def main():
    """Main rollback management operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migration Rollback Manager")
    parser.add_argument("--create-plan", help="Create rollback plan for migration ID")
    parser.add_argument("--phase", choices=[p.value for p in RollbackPhase], 
                       help="Target rollback phase")
    parser.add_argument("--execute", help="Execute rollback plan by ID")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run")
    parser.add_argument("--list", action="store_true", help="List rollback plans")
    parser.add_argument("--backup-info", help="JSON file with backup information")
    
    args = parser.parse_args()
    
    # Configuration
    source_db_uri = DatabaseConfig.SQLALCHEMY_DATABASE_URI
    target_db_uri = os.getenv('POSTGRES_URL', 'postgresql://hrp_user:password@localhost:5432/hrp_database')
    
    # Initialize rollback manager
    rollback_manager = RollbackManager(source_db_uri, target_db_uri)
    
    if args.create_plan:
        if not args.phase:
            logger.error("Phase is required when creating rollback plan")
            return 1
        
        backup_info = {}
        if args.backup_info and Path(args.backup_info).exists():
            with open(args.backup_info, 'r') as f:
                backup_info = json.load(f)
        
        plan = rollback_manager.create_rollback_plan(
            args.create_plan,
            RollbackPhase(args.phase),
            backup_info
        )
        
        print(f"Rollback plan created: {plan.plan_id}")
        return 0
    
    elif args.execute:
        success = rollback_manager.execute_rollback_plan(args.execute, args.dry_run)
        return 0 if success else 1
    
    elif args.list:
        plans = rollback_manager.list_rollback_plans()
        for plan in plans:
            status = "Executed" if plan.executed else "Not executed"
            success = "Success" if plan.success else "Failed" if plan.executed else "N/A"
            print(f"{plan.plan_id}: {plan.target_phase.value} - {status} - {success}")
        return 0
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())