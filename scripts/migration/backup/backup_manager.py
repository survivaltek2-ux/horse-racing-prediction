#!/usr/bin/env python3
"""
Backup and Redundancy Manager for PostgreSQL Migration
Comprehensive backup, restore, and redundancy mechanisms
"""

import os
import sys
import json
import logging
import shutil
import subprocess
import gzip
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
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
        logging.FileHandler('logs/backup_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BackupInfo:
    """Information about a backup"""
    backup_id: str
    backup_type: str  # 'full', 'incremental', 'schema_only', 'data_only'
    database_type: str  # 'source', 'target'
    timestamp: datetime
    file_path: str
    file_size: int
    checksum: str
    compression: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'backup_id': self.backup_id,
            'backup_type': self.backup_type,
            'database_type': self.database_type,
            'timestamp': self.timestamp.isoformat(),
            'file_path': self.file_path,
            'file_size': self.file_size,
            'checksum': self.checksum,
            'compression': self.compression,
            'metadata': self.metadata
        }

class BackupManager:
    """Manages database backups and redundancy for migration"""
    
    def __init__(self, source_db_uri: str, target_db_uri: str, backup_dir: str = "backups"):
        self.source_db_uri = source_db_uri
        self.target_db_uri = target_db_uri
        self.backup_dir = Path(backup_dir)
        
        # Create backup directory structure
        self.backup_dir.mkdir(exist_ok=True)
        (self.backup_dir / "source").mkdir(exist_ok=True)
        (self.backup_dir / "target").mkdir(exist_ok=True)
        (self.backup_dir / "migration_snapshots").mkdir(exist_ok=True)
        
        # Database connections
        self.source_engine = create_engine(source_db_uri)
        self.target_engine = create_engine(target_db_uri)
        
        # Backup registry
        self.backup_registry_file = self.backup_dir / "backup_registry.json"
        self.backup_registry = self._load_backup_registry()
        
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
    
    def _load_backup_registry(self) -> Dict[str, BackupInfo]:
        """Load backup registry from file"""
        if self.backup_registry_file.exists():
            try:
                with open(self.backup_registry_file, 'r') as f:
                    data = json.load(f)
                    registry = {}
                    for backup_id, backup_data in data.items():
                        backup_data['timestamp'] = datetime.fromisoformat(backup_data['timestamp'])
                        registry[backup_id] = BackupInfo(**backup_data)
                    return registry
            except Exception as e:
                logger.warning(f"Failed to load backup registry: {e}")
        
        return {}
    
    def _save_backup_registry(self):
        """Save backup registry to file"""
        try:
            data = {backup_id: backup.to_dict() for backup_id, backup in self.backup_registry.items()}
            with open(self.backup_registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup registry: {e}")
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _compress_file(self, file_path: str) -> str:
        """Compress a file using gzip"""
        compressed_path = f"{file_path}.gz"
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original file
        os.remove(file_path)
        
        return compressed_path
    
    def _decompress_file(self, compressed_path: str) -> str:
        """Decompress a gzipped file"""
        if not compressed_path.endswith('.gz'):
            return compressed_path
        
        original_path = compressed_path[:-3]  # Remove .gz extension
        
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(original_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return original_path
    
    def create_sqlite_backup(self, backup_type: str = "full", compress: bool = True) -> Optional[BackupInfo]:
        """Create backup of SQLite source database"""
        logger.info(f"Creating SQLite backup (type: {backup_type})")
        
        try:
            timestamp = datetime.now()
            backup_id = f"sqlite_{backup_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Determine backup file path
            backup_filename = f"{backup_id}.db"
            backup_path = self.backup_dir / "source" / backup_filename
            
            # Create backup based on type
            if backup_type == "full":
                # Copy entire database file
                source_db_path = self.source_db_uri.replace('sqlite:///', '')
                if os.path.exists(source_db_path):
                    shutil.copy2(source_db_path, backup_path)
                else:
                    logger.error(f"Source database file not found: {source_db_path}")
                    return None
            
            elif backup_type == "schema_only":
                # Export schema only
                self._export_sqlite_schema(backup_path)
            
            elif backup_type == "data_only":
                # Export data only (without schema)
                self._export_sqlite_data(backup_path)
            
            else:
                logger.error(f"Unsupported backup type: {backup_type}")
                return None
            
            # Compress if requested
            final_path = str(backup_path)
            if compress:
                final_path = self._compress_file(str(backup_path))
            
            # Calculate checksum and file size
            file_size = os.path.getsize(final_path)
            checksum = self._calculate_file_checksum(final_path)
            
            # Create backup info
            backup_info = BackupInfo(
                backup_id=backup_id,
                backup_type=backup_type,
                database_type="source",
                timestamp=timestamp,
                file_path=final_path,
                file_size=file_size,
                checksum=checksum,
                compression=compress,
                metadata={
                    "source_uri": self.source_db_uri,
                    "backup_method": "file_copy" if backup_type == "full" else "sql_export"
                }
            )
            
            # Register backup
            self.backup_registry[backup_id] = backup_info
            self._save_backup_registry()
            
            logger.info(f"SQLite backup created: {backup_id} ({file_size} bytes)")
            return backup_info
            
        except Exception as e:
            logger.error(f"Failed to create SQLite backup: {e}")
            return None
    
    def _export_sqlite_schema(self, output_path: str):
        """Export SQLite schema to file"""
        with self.source_engine.connect() as conn:
            # Get schema
            result = conn.execute(text("SELECT sql FROM sqlite_master WHERE type='table'"))
            schema_statements = [row[0] for row in result if row[0]]
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write("-- SQLite Schema Export\n")
                f.write(f"-- Generated on: {datetime.now().isoformat()}\n\n")
                for statement in schema_statements:
                    f.write(f"{statement};\n\n")
    
    def _export_sqlite_data(self, output_path: str):
        """Export SQLite data to file"""
        with self.source_engine.connect() as conn:
            # Get table names
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            
            with open(output_path, 'w') as f:
                f.write("-- SQLite Data Export\n")
                f.write(f"-- Generated on: {datetime.now().isoformat()}\n\n")
                
                for table in tables:
                    f.write(f"-- Data for table: {table}\n")
                    
                    # Get column names
                    columns_result = conn.execute(text(f"PRAGMA table_info({table})"))
                    columns = [row[1] for row in columns_result]
                    
                    # Export data
                    data_result = conn.execute(text(f"SELECT * FROM {table}"))
                    for row in data_result:
                        values = []
                        for value in row:
                            if value is None:
                                values.append("NULL")
                            elif isinstance(value, str):
                                escaped_value = value.replace("'", "''")
                                values.append(f"'{escaped_value}'")
                            else:
                                values.append(str(value))
                        
                        columns_str = ', '.join(columns)
                        values_str = ', '.join(values)
                        f.write(f"INSERT INTO {table} ({columns_str}) VALUES ({values_str});\n")
                    
                    f.write("\n")
    
    def create_postgresql_backup(self, backup_type: str = "full", compress: bool = True) -> Optional[BackupInfo]:
        """Create backup of PostgreSQL target database"""
        logger.info(f"Creating PostgreSQL backup (type: {backup_type})")
        
        try:
            timestamp = datetime.now()
            backup_id = f"postgresql_{backup_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Determine backup file extension
            if backup_type == "schema_only":
                backup_filename = f"{backup_id}.schema.sql"
            elif backup_type == "data_only":
                backup_filename = f"{backup_id}.data.sql"
            else:
                backup_filename = f"{backup_id}.sql"
            
            backup_path = self.backup_dir / "target" / backup_filename
            
            # Build pg_dump command
            pg_dump_cmd = [
                "pg_dump",
                "-h", self.pg_conn_params['host'],
                "-p", self.pg_conn_params['port'],
                "-U", self.pg_conn_params['user'],
                "-d", self.pg_conn_params['database'],
                "-f", str(backup_path),
                "--verbose"
            ]
            
            # Add backup type specific options
            if backup_type == "schema_only":
                pg_dump_cmd.extend(["--schema-only", "--no-owner", "--no-privileges"])
            elif backup_type == "data_only":
                pg_dump_cmd.extend(["--data-only", "--no-owner"])
            else:  # full backup
                pg_dump_cmd.extend(["--no-owner", "--no-privileges"])
            
            # Set environment variable for password
            env = os.environ.copy()
            if self.pg_conn_params['password']:
                env['PGPASSWORD'] = self.pg_conn_params['password']
            
            # Execute pg_dump
            result = subprocess.run(
                pg_dump_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"pg_dump failed: {result.stderr}")
                return None
            
            # Compress if requested
            final_path = str(backup_path)
            if compress:
                final_path = self._compress_file(str(backup_path))
            
            # Calculate checksum and file size
            file_size = os.path.getsize(final_path)
            checksum = self._calculate_file_checksum(final_path)
            
            # Create backup info
            backup_info = BackupInfo(
                backup_id=backup_id,
                backup_type=backup_type,
                database_type="target",
                timestamp=timestamp,
                file_path=final_path,
                file_size=file_size,
                checksum=checksum,
                compression=compress,
                metadata={
                    "target_uri": self.target_db_uri,
                    "backup_method": "pg_dump",
                    "pg_dump_version": self._get_pg_dump_version()
                }
            )
            
            # Register backup
            self.backup_registry[backup_id] = backup_info
            self._save_backup_registry()
            
            logger.info(f"PostgreSQL backup created: {backup_id} ({file_size} bytes)")
            return backup_info
            
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL backup: {e}")
            return None
    
    def _get_pg_dump_version(self) -> str:
        """Get pg_dump version"""
        try:
            result = subprocess.run(["pg_dump", "--version"], capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def create_migration_snapshot(self, phase: str, metadata: Dict[str, Any] = None) -> Optional[BackupInfo]:
        """Create a snapshot at a specific migration phase"""
        logger.info(f"Creating migration snapshot for phase: {phase}")
        
        try:
            timestamp = datetime.now()
            backup_id = f"migration_snapshot_{phase}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Create snapshot directory
            snapshot_dir = self.backup_dir / "migration_snapshots" / backup_id
            snapshot_dir.mkdir(exist_ok=True)
            
            # Backup both databases
            source_backup = self.create_sqlite_backup("full", compress=True)
            target_backup = self.create_postgresql_backup("full", compress=True)
            
            if not source_backup or not target_backup:
                logger.error("Failed to create complete migration snapshot")
                return None
            
            # Create snapshot metadata
            snapshot_metadata = {
                "phase": phase,
                "timestamp": timestamp.isoformat(),
                "source_backup_id": source_backup.backup_id,
                "target_backup_id": target_backup.backup_id,
                "migration_metadata": metadata or {}
            }
            
            # Save metadata
            metadata_file = snapshot_dir / "snapshot_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(snapshot_metadata, f, indent=2)
            
            # Create snapshot info
            snapshot_info = BackupInfo(
                backup_id=backup_id,
                backup_type="migration_snapshot",
                database_type="both",
                timestamp=timestamp,
                file_path=str(snapshot_dir),
                file_size=source_backup.file_size + target_backup.file_size,
                checksum="",  # Not applicable for directories
                compression=True,
                metadata=snapshot_metadata
            )
            
            # Register snapshot
            self.backup_registry[backup_id] = snapshot_info
            self._save_backup_registry()
            
            logger.info(f"Migration snapshot created: {backup_id}")
            return snapshot_info
            
        except Exception as e:
            logger.error(f"Failed to create migration snapshot: {e}")
            return None
    
    def restore_postgresql_backup(self, backup_id: str, target_db: str = None) -> bool:
        """Restore PostgreSQL database from backup"""
        logger.info(f"Restoring PostgreSQL backup: {backup_id}")
        
        if backup_id not in self.backup_registry:
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        backup_info = self.backup_registry[backup_id]
        
        if backup_info.database_type != "target":
            logger.error(f"Backup is not a PostgreSQL backup: {backup_id}")
            return False
        
        try:
            # Decompress if necessary
            restore_file = backup_info.file_path
            if backup_info.compression:
                restore_file = self._decompress_file(backup_info.file_path)
            
            # Verify checksum
            current_checksum = self._calculate_file_checksum(restore_file)
            if backup_info.compression:
                # For compressed files, we need to check the compressed file checksum
                current_checksum = self._calculate_file_checksum(backup_info.file_path)
            
            if current_checksum != backup_info.checksum:
                logger.error(f"Backup file checksum mismatch for {backup_id}")
                return False
            
            # Use target database or default
            database = target_db or self.pg_conn_params['database']
            
            # Build psql command
            psql_cmd = [
                "psql",
                "-h", self.pg_conn_params['host'],
                "-p", self.pg_conn_params['port'],
                "-U", self.pg_conn_params['user'],
                "-d", database,
                "-f", restore_file,
                "--quiet"
            ]
            
            # Set environment variable for password
            env = os.environ.copy()
            if self.pg_conn_params['password']:
                env['PGPASSWORD'] = self.pg_conn_params['password']
            
            # Execute psql
            result = subprocess.run(
                psql_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"psql restore failed: {result.stderr}")
                return False
            
            # Clean up decompressed file if it was created
            if backup_info.compression and restore_file != backup_info.file_path:
                os.remove(restore_file)
            
            logger.info(f"PostgreSQL backup restored successfully: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore PostgreSQL backup: {e}")
            return False
    
    def restore_sqlite_backup(self, backup_id: str, target_path: str = None) -> bool:
        """Restore SQLite database from backup"""
        logger.info(f"Restoring SQLite backup: {backup_id}")
        
        if backup_id not in self.backup_registry:
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        backup_info = self.backup_registry[backup_id]
        
        if backup_info.database_type != "source":
            logger.error(f"Backup is not a SQLite backup: {backup_id}")
            return False
        
        try:
            # Decompress if necessary
            restore_file = backup_info.file_path
            if backup_info.compression:
                restore_file = self._decompress_file(backup_info.file_path)
            
            # Verify checksum
            current_checksum = self._calculate_file_checksum(restore_file)
            if backup_info.compression:
                current_checksum = self._calculate_file_checksum(backup_info.file_path)
            
            if current_checksum != backup_info.checksum:
                logger.error(f"Backup file checksum mismatch for {backup_id}")
                return False
            
            # Determine target path
            if target_path is None:
                target_path = self.source_db_uri.replace('sqlite:///', '')
            
            # Copy restored file to target location
            if backup_info.backup_type == "full":
                shutil.copy2(restore_file, target_path)
            else:
                # For schema/data only backups, we need to execute SQL
                with open(restore_file, 'r') as f:
                    sql_content = f.read()
                
                # Create new database and execute SQL
                target_engine = create_engine(f'sqlite:///{target_path}')
                with target_engine.connect() as conn:
                    conn.execute(text(sql_content))
                    conn.commit()
            
            # Clean up decompressed file if it was created
            if backup_info.compression and restore_file != backup_info.file_path:
                os.remove(restore_file)
            
            logger.info(f"SQLite backup restored successfully: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore SQLite backup: {e}")
            return False
    
    def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """Clean up backups older than retention period"""
        logger.info(f"Cleaning up backups older than {retention_days} days")
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        removed_count = 0
        
        backups_to_remove = []
        for backup_id, backup_info in self.backup_registry.items():
            if backup_info.timestamp < cutoff_date:
                backups_to_remove.append(backup_id)
        
        for backup_id in backups_to_remove:
            backup_info = self.backup_registry[backup_id]
            
            try:
                # Remove backup file/directory
                if os.path.isfile(backup_info.file_path):
                    os.remove(backup_info.file_path)
                elif os.path.isdir(backup_info.file_path):
                    shutil.rmtree(backup_info.file_path)
                
                # Remove from registry
                del self.backup_registry[backup_id]
                removed_count += 1
                
                logger.info(f"Removed old backup: {backup_id}")
                
            except Exception as e:
                logger.warning(f"Failed to remove backup {backup_id}: {e}")
        
        # Save updated registry
        self._save_backup_registry()
        
        logger.info(f"Cleanup completed: {removed_count} backups removed")
        return removed_count
    
    def list_backups(self, database_type: str = None, backup_type: str = None) -> List[BackupInfo]:
        """List available backups with optional filtering"""
        backups = list(self.backup_registry.values())
        
        if database_type:
            backups = [b for b in backups if b.database_type == database_type]
        
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        
        return backups
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """Get information about a specific backup"""
        return self.backup_registry.get(backup_id)
    
    def verify_backup_integrity(self, backup_id: str) -> bool:
        """Verify backup file integrity using checksum"""
        if backup_id not in self.backup_registry:
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        backup_info = self.backup_registry[backup_id]
        
        try:
            if not os.path.exists(backup_info.file_path):
                logger.error(f"Backup file not found: {backup_info.file_path}")
                return False
            
            current_checksum = self._calculate_file_checksum(backup_info.file_path)
            
            if current_checksum != backup_info.checksum:
                logger.error(f"Backup integrity check failed for {backup_id}: checksum mismatch")
                return False
            
            logger.info(f"Backup integrity verified: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify backup integrity: {e}")
            return False
    
    def generate_backup_report(self) -> Dict[str, Any]:
        """Generate comprehensive backup report"""
        backups = list(self.backup_registry.values())
        
        # Calculate statistics
        total_backups = len(backups)
        total_size = sum(b.file_size for b in backups)
        
        backup_types = {}
        database_types = {}
        
        for backup in backups:
            backup_types[backup.backup_type] = backup_types.get(backup.backup_type, 0) + 1
            database_types[backup.database_type] = database_types.get(backup.database_type, 0) + 1
        
        # Find oldest and newest backups
        if backups:
            oldest_backup = min(backups, key=lambda x: x.timestamp)
            newest_backup = max(backups, key=lambda x: x.timestamp)
        else:
            oldest_backup = newest_backup = None
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'backup_directory': str(self.backup_dir),
            'summary': {
                'total_backups': total_backups,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'backup_types': backup_types,
                'database_types': database_types
            },
            'oldest_backup': oldest_backup.to_dict() if oldest_backup else None,
            'newest_backup': newest_backup.to_dict() if newest_backup else None,
            'recent_backups': [b.to_dict() for b in sorted(backups, key=lambda x: x.timestamp, reverse=True)[:10]]
        }
        
        return report

def main():
    """Main backup management operations"""
    logger.info("Starting backup management operations...")
    
    try:
        # Configuration
        source_db_uri = DatabaseConfig.SQLALCHEMY_DATABASE_URI
        target_db_uri = os.getenv('POSTGRES_URL', 'postgresql://hrp_user:password@localhost:5432/hrp_database')
        
        # Initialize backup manager
        backup_manager = BackupManager(source_db_uri, target_db_uri)
        
        # Create pre-migration backups
        logger.info("Creating pre-migration backups...")
        
        source_backup = backup_manager.create_sqlite_backup("full", compress=True)
        if source_backup:
            logger.info(f"Source backup created: {source_backup.backup_id}")
        
        target_backup = backup_manager.create_postgresql_backup("full", compress=True)
        if target_backup:
            logger.info(f"Target backup created: {target_backup.backup_id}")
        
        # Create migration snapshot
        snapshot = backup_manager.create_migration_snapshot("pre_migration", {
            "migration_phase": "preparation",
            "source_backup": source_backup.backup_id if source_backup else None,
            "target_backup": target_backup.backup_id if target_backup else None
        })
        
        if snapshot:
            logger.info(f"Migration snapshot created: {snapshot.backup_id}")
        
        # Generate and save backup report
        report = backup_manager.generate_backup_report()
        report_path = backup_manager.backup_dir / "backup_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Backup operations completed. Report saved to: {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"Backup management failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)