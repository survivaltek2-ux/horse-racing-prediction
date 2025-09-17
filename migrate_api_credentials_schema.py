#!/usr/bin/env python3
"""
Database migration script to add username and password columns to api_credentials table
"""

import os
import sys
from sqlalchemy import text

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from config.database_config import db

def migrate_api_credentials_table():
    """Add username and password columns to api_credentials table if they don't exist"""
    
    with app.app_context():
        try:
            # Check if the table exists
            result = db.session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='api_credentials'"))
            table_exists = result.fetchone() is not None
            
            if not table_exists:
                print("api_credentials table doesn't exist. Creating all tables...")
                db.create_all()
                print("✓ All tables created successfully")
                return True
            
            # Check if username column exists
            result = db.session.execute(text("PRAGMA table_info(api_credentials)"))
            columns = [row[1] for row in result.fetchall()]
            
            has_username = 'username' in columns
            has_password = 'password' in columns
            
            print(f"Current columns in api_credentials: {columns}")
            print(f"Has username column: {has_username}")
            print(f"Has password column: {has_password}")
            
            # Add missing columns
            if not has_username:
                print("Adding username column...")
                db.session.execute(text("ALTER TABLE api_credentials ADD COLUMN username VARCHAR(200)"))
                print("✓ Username column added")
            
            if not has_password:
                print("Adding password column...")
                db.session.execute(text("ALTER TABLE api_credentials ADD COLUMN password VARCHAR(500)"))
                print("✓ Password column added")
            
            if not has_username or not has_password:
                db.session.commit()
                print("✓ Database schema migration completed successfully")
            else:
                print("✓ All required columns already exist")
            
            return True
            
        except Exception as e:
            print(f"✗ Error during migration: {e}")
            db.session.rollback()
            return False

if __name__ == "__main__":
    print("Starting API credentials table migration...")
    success = migrate_api_credentials_table()
    if success:
        print("Migration completed successfully!")
        sys.exit(0)
    else:
        print("Migration failed!")
        sys.exit(1)