#!/usr/bin/env python3
"""
Database migration script to add enhanced data columns to existing tables.
This script will safely add new columns without affecting existing data.
"""

import sqlite3
import os
from app import app

def get_db_path():
    """Get the database path from the app config"""
    with app.app_context():
        db_path = app.config.get('SQLALCHEMY_DATABASE_URI', '').replace('sqlite:///', '')
        if not db_path:
            db_path = 'hrp_database.db'
        return db_path

def add_horse_columns(cursor):
    """Add enhanced columns to the horses table"""
    horse_columns = [
        # Basic Information
        ('breed', 'VARCHAR(50)'),
        ('color', 'VARCHAR(30)'),
        ('sex', 'VARCHAR(20)'),
        ('height', 'VARCHAR(10)'),
        ('markings', 'TEXT'),
        
        # Pedigree
        ('sire', 'VARCHAR(100)'),
        ('dam', 'VARCHAR(100)'),
        ('sire_line', 'VARCHAR(100)'),
        ('dam_line', 'VARCHAR(100)'),
        ('breeding_value', 'INTEGER'),
        
        # Connections
        ('jockey', 'VARCHAR(100)'),
        ('trainer', 'VARCHAR(100)'),
        ('owner', 'VARCHAR(100)'),
        ('breeder', 'VARCHAR(100)'),
        ('stable', 'VARCHAR(100)'),
        
        # Physical Attributes
        ('weight', 'FLOAT'),
        ('body_condition', 'VARCHAR(20)'),
        ('conformation_score', 'INTEGER'),
        
        # Performance Analytics
        ('speed_rating', 'INTEGER'),
        ('class_rating', 'INTEGER'),
        ('distance_preference', 'VARCHAR(50)'),
        ('surface_preference', 'VARCHAR(20)'),
        ('track_bias_rating', 'INTEGER'),
        ('pace_style', 'VARCHAR(30)'),
        ('closing_kick', 'VARCHAR(20)'),
        
        # Training & Fitness
        ('days_since_last_race', 'INTEGER'),
        ('fitness_level', 'INTEGER'),
        ('training_intensity', 'VARCHAR(20)'),
        ('workout_times', 'TEXT'),
        ('injury_history', 'TEXT'),
        ('recovery_time', 'INTEGER'),
        
        # Behavioral & Racing Style
        ('temperament', 'VARCHAR(20)'),
        ('gate_behavior', 'VARCHAR(20)'),
        ('racing_tactics', 'VARCHAR(30)'),
        ('equipment_used', 'VARCHAR(100)'),
        ('medication_notes', 'VARCHAR(200)'),
        
        # Financial Information
        ('purchase_price', 'FLOAT'),
        ('current_value', 'FLOAT'),
        ('insurance_value', 'FLOAT'),
        ('stud_fee', 'FLOAT'),
        
        # Performance Statistics (some already exist)
        ('form', 'VARCHAR(20)'),
        ('rating', 'INTEGER'),
        ('wins', 'INTEGER'),
        ('places', 'INTEGER'),
        ('runs', 'INTEGER'),
        ('earnings', 'FLOAT')
    ]
    
    # Get existing columns
    cursor.execute("PRAGMA table_info(horses)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    print(f"Found {len(existing_columns)} existing horse columns")
    
    # Add new columns
    added_count = 0
    for column_name, column_type in horse_columns:
        if column_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE horses ADD COLUMN {column_name} {column_type}")
                print(f"Added horse column: {column_name}")
                added_count += 1
            except sqlite3.OperationalError as e:
                print(f"Warning: Could not add column {column_name}: {e}")
    
    print(f"Added {added_count} new horse columns")

def add_race_columns(cursor):
    """Add enhanced columns to the races table"""
    race_columns = [
        # Weather Conditions
        ('temperature', 'INTEGER'),
        ('humidity', 'INTEGER'),
        ('wind_speed', 'INTEGER'),
        ('wind_direction', 'VARCHAR(5)'),
        ('weather_description', 'VARCHAR(50)'),
        ('visibility', 'FLOAT'),
        
        # Track Conditions
        ('surface_type', 'VARCHAR(30)'),
        ('rail_position', 'VARCHAR(50)'),
        ('track_bias', 'VARCHAR(50)'),
        ('track_maintenance', 'VARCHAR(200)'),
        
        # Field Analysis
        ('field_size', 'INTEGER'),
        ('field_quality', 'VARCHAR(20)'),
        ('pace_scenario', 'VARCHAR(20)'),
        ('competitive_balance', 'VARCHAR(30)'),
        ('speed_figures_range', 'VARCHAR(20)'),
        
        # Betting Information
        ('total_pool', 'FLOAT'),
        ('win_pool', 'FLOAT'),
        ('exacta_pool', 'FLOAT'),
        ('trifecta_pool', 'FLOAT'),
        ('superfecta_pool', 'FLOAT'),
        ('morning_line_favorite', 'VARCHAR(10)'),
        
        # Race Conditions
        ('age_restrictions', 'VARCHAR(50)'),
        ('sex_restrictions', 'VARCHAR(50)'),
        ('weight_conditions', 'VARCHAR(100)'),
        ('claiming_price', 'FLOAT'),
        ('race_grade', 'VARCHAR(30)'),
        
        # Historical Data
        ('track_record', 'VARCHAR(20)'),
        ('average_winning_time', 'VARCHAR(20)'),
        ('course_record_holder', 'VARCHAR(100)'),
        ('similar_race_results', 'TEXT'),
        ('trainer_jockey_stats', 'TEXT'),
        
        # Media Coverage
        ('tv_coverage', 'VARCHAR(30)'),
        ('streaming_available', 'VARCHAR(10)'),
        ('featured_race', 'VARCHAR(10)')
    ]
    
    # Get existing columns
    cursor.execute("PRAGMA table_info(races)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    print(f"Found {len(existing_columns)} existing race columns")
    
    # Add new columns
    added_count = 0
    for column_name, column_type in race_columns:
        if column_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE races ADD COLUMN {column_name} {column_type}")
                print(f"Added race column: {column_name}")
                added_count += 1
            except sqlite3.OperationalError as e:
                print(f"Warning: Could not add column {column_name}: {e}")
    
    print(f"Added {added_count} new race columns")

def main():
    """Main migration function"""
    print("Starting database schema migration...")
    print("=" * 50)
    
    # Get database path
    db_path = get_db_path()
    print(f"Database path: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return
    
    # Create backup
    backup_path = f"{db_path}.backup_migration"
    import shutil
    shutil.copy2(db_path, backup_path)
    print(f"Created backup at: {backup_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Add horse columns
        print("\nMigrating horses table...")
        add_horse_columns(cursor)
        
        print("\nMigrating races table...")
        add_race_columns(cursor)
        
        # Commit changes
        conn.commit()
        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("Database schema has been updated with enhanced data columns.")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()