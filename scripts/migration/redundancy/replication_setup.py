#!/usr/bin/env python3
"""
PostgreSQL Replication and Redundancy Setup
Configures streaming replication, failover, and high availability
"""

import os
import sys
import json
import logging
import subprocess
import time
import psycopg2
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/replication_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ReplicationNode:
    """Information about a replication node"""
    node_id: str
    role: str  # 'primary', 'standby', 'witness'
    host: str
    port: int
    database: str
    username: str
    password: str
    data_directory: str
    status: str = 'unknown'
    lag_bytes: int = 0
    lag_seconds: float = 0.0
    
    def get_connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class PostgreSQLReplicationManager:
    """Manages PostgreSQL streaming replication and high availability"""
    
    def __init__(self, config_file: str = "replication_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.nodes = self._initialize_nodes()
        
        # Create necessary directories
        self.scripts_dir = Path("scripts/replication")
        self.scripts_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Load replication configuration"""
        default_config = {
            "primary": {
                "host": "localhost",
                "port": 5432,
                "database": "hrp_database",
                "username": "hrp_user",
                "password": "password",
                "data_directory": "/usr/local/var/postgres"
            },
            "standby": {
                "host": "localhost",
                "port": 5433,
                "database": "hrp_database",
                "username": "hrp_user",
                "password": "password",
                "data_directory": "/usr/local/var/postgres_standby"
            },
            "replication": {
                "replication_user": "replicator",
                "replication_password": "repl_password",
                "wal_level": "replica",
                "max_wal_senders": 3,
                "wal_keep_segments": 64,
                "hot_standby": True,
                "archive_mode": True,
                "archive_command": "cp %p /usr/local/var/postgres/archive/%f"
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logger.warning(f"Failed to load config, using defaults: {e}")
        
        # Save default config
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _initialize_nodes(self) -> Dict[str, ReplicationNode]:
        """Initialize replication nodes from config"""
        nodes = {}
        
        # Primary node
        primary_config = self.config['primary']
        nodes['primary'] = ReplicationNode(
            node_id='primary',
            role='primary',
            host=primary_config['host'],
            port=primary_config['port'],
            database=primary_config['database'],
            username=primary_config['username'],
            password=primary_config['password'],
            data_directory=primary_config['data_directory']
        )
        
        # Standby node
        standby_config = self.config['standby']
        nodes['standby'] = ReplicationNode(
            node_id='standby',
            role='standby',
            host=standby_config['host'],
            port=standby_config['port'],
            database=standby_config['database'],
            username=standby_config['username'],
            password=standby_config['password'],
            data_directory=standby_config['data_directory']
        )
        
        return nodes
    
    def _execute_sql(self, node: ReplicationNode, sql: str, autocommit: bool = True) -> Optional[List]:
        """Execute SQL on a specific node"""
        try:
            conn = psycopg2.connect(
                host=node.host,
                port=node.port,
                database=node.database,
                user=node.username,
                password=node.password
            )
            
            if autocommit:
                conn.autocommit = True
            
            with conn.cursor() as cursor:
                cursor.execute(sql)
                
                if cursor.description:
                    return cursor.fetchall()
                
            conn.close()
            return []
            
        except Exception as e:
            logger.error(f"Failed to execute SQL on {node.node_id}: {e}")
            return None
    
    def _execute_system_command(self, command: List[str], cwd: str = None) -> Tuple[bool, str, str]:
        """Execute system command"""
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def setup_replication_user(self) -> bool:
        """Create replication user on primary"""
        logger.info("Setting up replication user...")
        
        primary = self.nodes['primary']
        repl_config = self.config['replication']
        
        # Create replication user
        create_user_sql = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '{repl_config['replication_user']}') THEN
                CREATE ROLE {repl_config['replication_user']} WITH REPLICATION LOGIN PASSWORD '{repl_config['replication_password']}';
            END IF;
        END
        $$;
        """
        
        result = self._execute_sql(primary, create_user_sql)
        if result is None:
            logger.error("Failed to create replication user")
            return False
        
        logger.info(f"Replication user '{repl_config['replication_user']}' created/verified")
        return True
    
    def configure_primary_server(self) -> bool:
        """Configure primary server for replication"""
        logger.info("Configuring primary server for replication...")
        
        primary = self.nodes['primary']
        repl_config = self.config['replication']
        
        # Generate postgresql.conf settings
        postgresql_conf_settings = f"""
# Replication settings
wal_level = {repl_config['wal_level']}
max_wal_senders = {repl_config['max_wal_senders']}
wal_keep_segments = {repl_config['wal_keep_segments']}
hot_standby = {'on' if repl_config['hot_standby'] else 'off'}
archive_mode = {'on' if repl_config['archive_mode'] else 'off'}
archive_command = '{repl_config['archive_command']}'

# Connection settings
listen_addresses = '*'
max_connections = 100

# Logging
log_replication_commands = on
log_min_messages = info
"""
        
        # Write postgresql.conf additions
        conf_file = self.scripts_dir / "postgresql_replication.conf"
        with open(conf_file, 'w') as f:
            f.write(postgresql_conf_settings)
        
        # Generate pg_hba.conf entries
        pg_hba_entries = f"""
# Replication connections
host replication {repl_config['replication_user']} {self.nodes['standby'].host}/32 md5
host replication {repl_config['replication_user']} 127.0.0.1/32 md5
host replication {repl_config['replication_user']} ::1/128 md5
"""
        
        # Write pg_hba.conf additions
        hba_file = self.scripts_dir / "pg_hba_replication.conf"
        with open(hba_file, 'w') as f:
            f.write(pg_hba_entries)
        
        logger.info("Primary server configuration files generated")
        logger.info(f"PostgreSQL config: {conf_file}")
        logger.info(f"pg_hba config: {hba_file}")
        logger.info("Please append these settings to your PostgreSQL configuration files and restart the server")
        
        return True
    
    def create_base_backup(self) -> bool:
        """Create base backup for standby server"""
        logger.info("Creating base backup for standby server...")
        
        primary = self.nodes['primary']
        standby = self.nodes['standby']
        repl_config = self.config['replication']
        
        # Ensure standby data directory exists and is empty
        standby_data_dir = Path(standby.data_directory)
        if standby_data_dir.exists():
            logger.warning(f"Standby data directory exists: {standby_data_dir}")
            logger.warning("Please ensure it's empty or remove it before proceeding")
        else:
            standby_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create pg_basebackup command
        pg_basebackup_cmd = [
            "pg_basebackup",
            "-h", primary.host,
            "-p", str(primary.port),
            "-U", repl_config['replication_user'],
            "-D", str(standby_data_dir),
            "-P",  # Show progress
            "-W",  # Force password prompt
            "-R",  # Write recovery.conf
            "-X", "stream"  # Stream WAL
        ]
        
        # Set environment for password
        env = os.environ.copy()
        env['PGPASSWORD'] = repl_config['replication_password']
        
        logger.info(f"Running pg_basebackup to {standby_data_dir}")
        
        try:
            result = subprocess.run(
                pg_basebackup_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"pg_basebackup failed: {result.stderr}")
                return False
            
            logger.info("Base backup completed successfully")
            logger.info(result.stdout)
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("pg_basebackup timed out")
            return False
        except Exception as e:
            logger.error(f"pg_basebackup failed: {e}")
            return False
    
    def configure_standby_server(self) -> bool:
        """Configure standby server"""
        logger.info("Configuring standby server...")
        
        standby = self.nodes['standby']
        primary = self.nodes['primary']
        repl_config = self.config['replication']
        
        # Generate recovery.conf (for PostgreSQL < 12) or postgresql.conf settings (for PostgreSQL >= 12)
        recovery_settings = f"""
# Standby server settings
standby_mode = 'on'
primary_conninfo = 'host={primary.host} port={primary.port} user={repl_config['replication_user']} password={repl_config['replication_password']}'
trigger_file = '{standby.data_directory}/trigger_failover'
restore_command = 'cp {primary.data_directory}/archive/%f %p'
archive_cleanup_command = 'pg_archivecleanup {primary.data_directory}/archive %r'

# Hot standby settings
hot_standby = on
max_standby_streaming_delay = 30s
max_standby_archive_delay = 30s
"""
        
        # Write recovery configuration
        recovery_file = self.scripts_dir / "recovery.conf"
        with open(recovery_file, 'w') as f:
            f.write(recovery_settings)
        
        # Generate standby-specific postgresql.conf settings
        standby_postgresql_conf = f"""
# Standby server specific settings
port = {standby.port}
hot_standby = on
max_standby_streaming_delay = 30s
max_standby_archive_delay = 30s

# Logging
log_min_messages = info
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
"""
        
        # Write standby postgresql.conf
        standby_conf_file = self.scripts_dir / "postgresql_standby.conf"
        with open(standby_conf_file, 'w') as f:
            f.write(standby_postgresql_conf)
        
        logger.info("Standby server configuration files generated")
        logger.info(f"Recovery config: {recovery_file}")
        logger.info(f"Standby PostgreSQL config: {standby_conf_file}")
        
        return True
    
    def start_standby_server(self) -> bool:
        """Start standby server"""
        logger.info("Starting standby server...")
        
        standby = self.nodes['standby']
        
        # Start PostgreSQL standby server
        pg_ctl_cmd = [
            "pg_ctl",
            "-D", standby.data_directory,
            "-l", f"{self.logs_dir}/postgresql_standby.log",
            "start"
        ]
        
        success, stdout, stderr = self._execute_system_command(pg_ctl_cmd)
        
        if not success:
            logger.error(f"Failed to start standby server: {stderr}")
            return False
        
        logger.info("Standby server started successfully")
        logger.info(stdout)
        
        # Wait for server to be ready
        time.sleep(5)
        
        return True
    
    def check_replication_status(self) -> Dict[str, any]:
        """Check replication status"""
        logger.info("Checking replication status...")
        
        primary = self.nodes['primary']
        standby = self.nodes['standby']
        
        status = {
            'primary': {'status': 'unknown', 'details': {}},
            'standby': {'status': 'unknown', 'details': {}},
            'replication': {'status': 'unknown', 'lag_bytes': 0, 'lag_seconds': 0}
        }
        
        # Check primary server
        try:
            # Check if primary is accepting connections
            primary_result = self._execute_sql(primary, "SELECT version();")
            if primary_result:
                status['primary']['status'] = 'running'
                status['primary']['details']['version'] = primary_result[0][0]
                
                # Get replication status from primary
                repl_status_sql = """
                SELECT client_addr, state, sent_lsn, write_lsn, flush_lsn, replay_lsn,
                       write_lag, flush_lag, replay_lag
                FROM pg_stat_replication;
                """
                
                repl_result = self._execute_sql(primary, repl_status_sql)
                if repl_result:
                    status['primary']['details']['replication_connections'] = len(repl_result)
                    if repl_result:
                        # Get details of first replication connection
                        conn_info = repl_result[0]
                        status['replication']['status'] = 'active'
                        status['replication']['client_addr'] = conn_info[0]
                        status['replication']['state'] = conn_info[1]
                        
                        # Calculate lag if available
                        if conn_info[7]:  # flush_lag
                            status['replication']['lag_seconds'] = float(conn_info[7].total_seconds())
                else:
                    status['primary']['details']['replication_connections'] = 0
                    status['replication']['status'] = 'no_connections'
            
        except Exception as e:
            status['primary']['status'] = 'error'
            status['primary']['details']['error'] = str(e)
        
        # Check standby server
        try:
            standby_result = self._execute_sql(standby, "SELECT version();")
            if standby_result:
                status['standby']['status'] = 'running'
                status['standby']['details']['version'] = standby_result[0][0]
                
                # Check if standby is in recovery mode
                recovery_sql = "SELECT pg_is_in_recovery();"
                recovery_result = self._execute_sql(standby, recovery_sql)
                if recovery_result and recovery_result[0][0]:
                    status['standby']['details']['in_recovery'] = True
                    
                    # Get last received LSN
                    lsn_sql = "SELECT pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn();"
                    lsn_result = self._execute_sql(standby, lsn_sql)
                    if lsn_result:
                        status['standby']['details']['last_receive_lsn'] = str(lsn_result[0][0])
                        status['standby']['details']['last_replay_lsn'] = str(lsn_result[0][1])
                else:
                    status['standby']['details']['in_recovery'] = False
            
        except Exception as e:
            status['standby']['status'] = 'error'
            status['standby']['details']['error'] = str(e)
        
        return status
    
    def perform_failover(self, promote_standby: bool = True) -> bool:
        """Perform failover from primary to standby"""
        logger.info("Performing failover...")
        
        standby = self.nodes['standby']
        
        if promote_standby:
            # Create trigger file to promote standby
            trigger_file = Path(standby.data_directory) / "trigger_failover"
            trigger_file.touch()
            
            logger.info(f"Trigger file created: {trigger_file}")
            
            # Wait for promotion
            time.sleep(10)
            
            # Check if standby is now primary
            recovery_sql = "SELECT pg_is_in_recovery();"
            result = self._execute_sql(standby, recovery_sql)
            
            if result and not result[0][0]:
                logger.info("Standby successfully promoted to primary")
                
                # Update node roles
                self.nodes['standby'].role = 'primary'
                self.nodes['primary'].role = 'failed'
                
                return True
            else:
                logger.error("Failover failed - standby still in recovery mode")
                return False
        
        return False
    
    def generate_replication_scripts(self):
        """Generate helper scripts for replication management"""
        logger.info("Generating replication management scripts...")
        
        # Start replication script
        start_script = f"""#!/bin/bash
# Start PostgreSQL Replication Setup

echo "Starting PostgreSQL Replication Setup..."

# Start primary server
echo "Starting primary server..."
pg_ctl -D {self.nodes['primary'].data_directory} -l logs/postgresql_primary.log start

# Wait for primary to be ready
sleep 5

# Start standby server
echo "Starting standby server..."
pg_ctl -D {self.nodes['standby'].data_directory} -l logs/postgresql_standby.log start

echo "Replication setup started. Check logs for details."
"""
        
        start_script_file = self.scripts_dir / "start_replication.sh"
        with open(start_script_file, 'w') as f:
            f.write(start_script)
        start_script_file.chmod(0o755)
        
        # Stop replication script
        stop_script = f"""#!/bin/bash
# Stop PostgreSQL Replication Setup

echo "Stopping PostgreSQL Replication Setup..."

# Stop standby server
echo "Stopping standby server..."
pg_ctl -D {self.nodes['standby'].data_directory} stop -m fast

# Stop primary server
echo "Stopping primary server..."
pg_ctl -D {self.nodes['primary'].data_directory} stop -m fast

echo "Replication setup stopped."
"""
        
        stop_script_file = self.scripts_dir / "stop_replication.sh"
        with open(stop_script_file, 'w') as f:
            f.write(stop_script)
        stop_script_file.chmod(0o755)
        
        # Status check script
        status_script = """#!/bin/bash
# Check PostgreSQL Replication Status

echo "Checking PostgreSQL Replication Status..."
python3 scripts/migration/redundancy/replication_setup.py --check-status
"""
        
        status_script_file = self.scripts_dir / "check_replication_status.sh"
        with open(status_script_file, 'w') as f:
            f.write(status_script)
        status_script_file.chmod(0o755)
        
        logger.info("Replication management scripts generated:")
        logger.info(f"  Start: {start_script_file}")
        logger.info(f"  Stop: {stop_script_file}")
        logger.info(f"  Status: {status_script_file}")
    
    def setup_complete_replication(self) -> bool:
        """Setup complete replication environment"""
        logger.info("Setting up complete PostgreSQL replication environment...")
        
        steps = [
            ("Setting up replication user", self.setup_replication_user),
            ("Configuring primary server", self.configure_primary_server),
            ("Creating base backup", self.create_base_backup),
            ("Configuring standby server", self.configure_standby_server),
            ("Generating management scripts", self.generate_replication_scripts)
        ]
        
        for step_name, step_function in steps:
            logger.info(f"Step: {step_name}")
            
            if not step_function():
                logger.error(f"Failed at step: {step_name}")
                return False
            
            logger.info(f"Completed: {step_name}")
        
        logger.info("PostgreSQL replication setup completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Apply the generated configuration files to your PostgreSQL installations")
        logger.info("2. Restart the primary PostgreSQL server")
        logger.info("3. Use the generated scripts to start replication")
        
        return True

def main():
    """Main replication setup operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PostgreSQL Replication Setup")
    parser.add_argument("--setup", action="store_true", help="Setup complete replication")
    parser.add_argument("--check-status", action="store_true", help="Check replication status")
    parser.add_argument("--failover", action="store_true", help="Perform failover")
    parser.add_argument("--config", default="replication_config.json", help="Configuration file")
    
    args = parser.parse_args()
    
    # Initialize replication manager
    replication_manager = PostgreSQLReplicationManager(args.config)
    
    if args.setup:
        success = replication_manager.setup_complete_replication()
        return 0 if success else 1
    
    elif args.check_status:
        status = replication_manager.check_replication_status()
        print(json.dumps(status, indent=2, default=str))
        return 0
    
    elif args.failover:
        success = replication_manager.perform_failover()
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())