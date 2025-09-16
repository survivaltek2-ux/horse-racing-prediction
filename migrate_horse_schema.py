#!/usr/bin/env python3
"""
Database Migration Script: Add Missing Horse Columns
This script adds the missing performance analytics and recent form/fitness fields to the horses table.
"""

import sqlite3
import os
from datetime import datetime

def get_database_path():
    """Get the path to the database file"""
    return os.path.join(os.path.dirname(__file__), 'data', 'hrp_database.db')

def backup_database():
    """Create a backup of the database before migration"""
    db_path = get_database_path()
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"✓ Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"✗ Failed to backup database: {e}")
        return None

def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in the table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns

def add_missing_columns():
    """Add missing columns to the horses table"""
    db_path = get_database_path()
    
    if not os.path.exists(db_path):
        print(f"✗ Database file not found: {db_path}")
        return False
    
    # Create backup first
    backup_path = backup_database()
    if not backup_path:
        print("✗ Migration aborted - could not create backup")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("Starting database migration...")
        
        # Define the missing columns with their SQL definitions
        missing_columns = [
            ('speed_rating', 'INTEGER'),
            ('class_rating', 'INTEGER'),
            ('distance_preference', 'VARCHAR(20)'),
            ('surface_preference', 'VARCHAR(20)'),
            ('track_bias_rating', 'INTEGER'),
            ('days_since_last_race', 'INTEGER'),
            ('fitness_level', 'INTEGER'),
            ('workout_times', 'TEXT'),
            ('injury_history', 'TEXT')
        ]
        
        # Check which columns are actually missing and add them
        columns_added = 0
        for column_name, column_type in missing_columns:
            if not check_column_exists(cursor, 'horses', column_name):
                try:
                    sql = f"ALTER TABLE horses ADD COLUMN {column_name} {column_type}"
                    cursor.execute(sql)
                    print(f"✓ Added column: {column_name} ({column_type})")
                    columns_added += 1
                except Exception as e:
                    print(f"✗ Failed to add column {column_name}: {e}")
                    conn.rollback()
                    conn.close()
                    return False
            else:
                print(f"- Column {column_name} already exists")
        
        # Commit the changes
        conn.commit()
        
        # Verify the schema
        print("\nVerifying updated schema...")
        cursor.execute("PRAGMA table_info(horses)")
        columns = cursor.fetchall()
        
        print(f"\nHorses table now has {len(columns)} columns:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        conn.close()
        
        print(f"\n✓ Migration completed successfully!")
        print(f"✓ Added {columns_added} new columns")
        print(f"✓ Backup saved at: {backup_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        return False

def main():
    """Main migration function"""
    print("Horse Racing Prediction - Database Migration")
    print("=" * 50)
    print("Adding missing performance analytics and fitness fields to horses table...")
    print()
    
    success = add_missing_columns()
    
    if success:
        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("The application should now work without schema errors.")
    else:
        print("\n" + "=" * 50)
        print("Migration failed!")
        print("Please check the error messages above and try again.")
    
    return success

if __name__ == "__main__":
    main()