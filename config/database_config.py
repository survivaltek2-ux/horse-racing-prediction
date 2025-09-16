"""
SQLite Database Configuration
Replaces Firebase configuration with SQLAlchemy and SQLite setup.
"""

import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Initialize SQLAlchemy
db = SQLAlchemy()
migrate = Migrate()

class DatabaseConfig:
    """Database configuration class."""
    
    # SQLite database file path
    DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'hrp_database.db')
    
    # SQLAlchemy configuration
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DATABASE_PATH}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False  # Set to True for SQL debugging
    
    # Database connection settings
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'connect_args': {
            'check_same_thread': False,  # Allow SQLite to be used across threads
            'timeout': 30
        }
    }

def init_database(app: Flask):
    """Initialize database with Flask app."""
    
    # Configure Flask app
    app.config['SQLALCHEMY_DATABASE_URI'] = DatabaseConfig.SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = DatabaseConfig.SQLALCHEMY_TRACK_MODIFICATIONS
    app.config['SQLALCHEMY_ECHO'] = DatabaseConfig.SQLALCHEMY_ECHO
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = DatabaseConfig.SQLALCHEMY_ENGINE_OPTIONS
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DatabaseConfig.DATABASE_PATH), exist_ok=True)
    
    return db

def create_tables():
    """Create all database tables."""
    try:
        db.create_all()
        print("✓ Database tables created successfully")
        return True
    except Exception as e:
        print(f"✗ Error creating database tables: {e}")
        return False

def get_db_session():
    """Get a database session for direct SQLAlchemy operations."""
    engine = create_engine(DatabaseConfig.SQLALCHEMY_DATABASE_URI)
    Session = sessionmaker(bind=engine)
    return Session()

def backup_database(backup_path=None):
    """Create a backup of the SQLite database."""
    if backup_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"data/backup_hrp_database_{timestamp}.db"
    
    try:
        import shutil
        shutil.copy2(DatabaseConfig.DATABASE_PATH, backup_path)
        print(f"✓ Database backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"✗ Error creating database backup: {e}")
        return None

def check_database_health():
    """Check database health and connectivity."""
    try:
        engine = create_engine(DatabaseConfig.SQLALCHEMY_DATABASE_URI)
        with engine.connect() as connection:
            result = connection.execute(db.text("SELECT 1"))
            result.fetchone()
        print("✓ Database connection healthy")
        return True
    except Exception as e:
        print(f"✗ Database health check failed: {e}")
        return False

def get_database_info():
    """Get database information and statistics."""
    try:
        engine = create_engine(DatabaseConfig.SQLALCHEMY_DATABASE_URI)
        with engine.connect() as connection:
            # Get table list
            tables_result = connection.execute(db.text(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ))
            tables = [row[0] for row in tables_result.fetchall()]
            
            # Get database size
            db_size = os.path.getsize(DatabaseConfig.DATABASE_PATH) if os.path.exists(DatabaseConfig.DATABASE_PATH) else 0
            
            info = {
                'database_path': DatabaseConfig.DATABASE_PATH,
                'database_size_mb': round(db_size / (1024 * 1024), 2),
                'tables': tables,
                'table_count': len(tables)
            }
            
            return info
    except Exception as e:
        print(f"✗ Error getting database info: {e}")
        return None

# Database utility functions
class DatabaseUtils:
    """Utility functions for database operations."""
    
    @staticmethod
    def execute_raw_sql(sql, params=None):
        """Execute raw SQL query."""
        try:
            engine = create_engine(DatabaseConfig.SQLALCHEMY_DATABASE_URI)
            with engine.connect() as connection:
                result = connection.execute(db.text(sql), params or {})
                return result.fetchall()
        except Exception as e:
            print(f"✗ Error executing SQL: {e}")
            return None
    
    @staticmethod
    def get_table_row_count(table_name):
        """Get row count for a specific table."""
        try:
            sql = f"SELECT COUNT(*) FROM {table_name}"
            result = DatabaseUtils.execute_raw_sql(sql)
            return result[0][0] if result else 0
        except Exception as e:
            print(f"✗ Error getting row count for {table_name}: {e}")
            return 0
    
    @staticmethod
    def vacuum_database():
        """Optimize database by running VACUUM."""
        try:
            engine = create_engine(DatabaseConfig.SQLALCHEMY_DATABASE_URI)
            with engine.connect() as connection:
                connection.execute(db.text("VACUUM"))
            print("✓ Database optimized successfully")
            return True
        except Exception as e:
            print(f"✗ Error optimizing database: {e}")
            return False

# Export main components
__all__ = [
    'db', 'migrate', 'DatabaseConfig', 'init_database', 'create_tables',
    'get_db_session', 'backup_database', 'check_database_health',
    'get_database_info', 'DatabaseUtils'
]