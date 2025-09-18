"""
Database configuration module for the Flask application.
Provides database initialization and SQLAlchemy setup.
"""

import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Initialize SQLAlchemy
db = SQLAlchemy()
migrate = Migrate()


class DatabaseConfig:
    """Database configuration class"""
    
    def __init__(self):
        self.database_url = self._get_database_url()
    
    def _get_database_url(self):
        """Get database URL from environment or use default SQLite"""
        database_url = os.getenv('DATABASE_URL')
        
        if database_url:
            # Handle PostgreSQL URL format
            if database_url.startswith('postgres://'):
                database_url = database_url.replace('postgres://', 'postgresql://', 1)
            return database_url
        
        # Default to SQLite
        return 'sqlite:///horse_racing.db'
    
    def get_config(self):
        """Get database configuration dictionary"""
        return {
            'SQLALCHEMY_DATABASE_URI': self.database_url,
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'SQLALCHEMY_ENGINE_OPTIONS': {
                'pool_pre_ping': True,
                'pool_recycle': 300,
            }
        }


def init_database(app):
    """Initialize database with Flask app"""
    config = DatabaseConfig()
    
    # Configure SQLAlchemy
    app.config.update(config.get_config())
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Create tables
    with app.app_context():
        try:
            db.create_all()
            app.logger.info("Database tables created successfully")
        except Exception as e:
            app.logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    return db


def create_tables():
    """Create all database tables"""
    try:
        db.create_all()
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating database tables: {str(e)}")
        raise


# Create global database config instance
database_config = DatabaseConfig()