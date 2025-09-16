#!/usr/bin/env python3
"""
Migration script to move data from SQLite/JSON to Firebase
Run this script to migrate your existing data to Firebase Firestore
"""

import os
import sys
import json
from datetime import datetime
from config.firebase_config import firebase_config

def migrate_json_data():
    """Migrate existing JSON data to Firebase"""
    print("Starting migration of JSON data to Firebase...")
    
    # Check if Firebase is connected
    if not firebase_config.is_connected():
        print("Error: Firebase is not connected. Please check your configuration.")
        return False
    
    # Migrate races
    races_file = 'data/races.json'
    if os.path.exists(races_file):
        print(f"Migrating races from {races_file}...")
        try:
            with open(races_file, 'r') as f:
                races_data = json.load(f)
            
            for race_data in races_data:
                # Convert date string to datetime if needed
                if 'date' in race_data and isinstance(race_data['date'], str):
                    try:
                        race_data['date'] = datetime.fromisoformat(race_data['date'])
                    except:
                        pass
                
                race_id = firebase_config.save_race(race_data)
                print(f"Migrated race: {race_data.get('name', 'Unknown')} -> {race_id}")
            
            print(f"Successfully migrated {len(races_data)} races")
        except Exception as e:
            print(f"Error migrating races: {e}")
    
    # Migrate horses
    horses_file = 'data/horses.json'
    if os.path.exists(horses_file):
        print(f"Migrating horses from {horses_file}...")
        try:
            with open(horses_file, 'r') as f:
                horses_data = json.load(f)
            
            for horse_data in horses_data:
                # Convert date strings to datetime if needed
                if 'last_run' in horse_data and isinstance(horse_data['last_run'], str):
                    try:
                        horse_data['last_run'] = datetime.fromisoformat(horse_data['last_run'])
                    except:
                        pass
                
                horse_id = firebase_config.save_horse(horse_data)
                print(f"Migrated horse: {horse_data.get('name', 'Unknown')} -> {horse_id}")
            
            print(f"Successfully migrated {len(horses_data)} horses")
        except Exception as e:
            print(f"Error migrating horses: {e}")
    
    return True

def migrate_sqlite_data():
    """Migrate existing SQLite data to Firebase"""
    print("Starting migration of SQLite data to Firebase...")
    
    try:
        # Import SQLAlchemy models
        from models.api_credentials import APICredentials as SQLAPICredentials
        from models.race import Race as SQLRace
        from models.horse import Horse as SQLHorse
        from models.prediction import Prediction as SQLPrediction
        from models.user import User as SQLUser
        
        # Initialize database
        from app import db
        
        # Migrate API credentials
        print("Migrating API credentials...")
        sql_credentials = SQLAPICredentials.query.all()
        for cred in sql_credentials:
            cred_data = {
                'provider': cred.provider,
                'base_url': cred.base_url,
                'api_key': cred.api_key,
                'api_secret': cred.api_secret,
                'description': cred.description,
                'is_active': cred.is_active
            }
            cred_id = firebase_config.save_api_credential(cred_data)
            print(f"Migrated API credential: {cred.provider} -> {cred_id}")
        
        print(f"Successfully migrated {len(sql_credentials)} API credentials")
        
        # Migrate races
        print("Migrating races...")
        sql_races = SQLRace.query.all()
        for race in sql_races:
            race_data = {
                'name': race.name,
                'date': race.date,
                'time': race.time,
                'track': race.track,
                'distance': race.distance,
                'surface': race.surface,
                'race_class': race.race_class,
                'prize_money': race.prize_money,
                'weather': race.weather,
                'track_condition': race.track_condition
            }
            race_id = firebase_config.save_race(race_data)
            print(f"Migrated race: {race.name} -> {race_id}")
        
        print(f"Successfully migrated {len(sql_races)} races")
        
        # Migrate horses
        print("Migrating horses...")
        sql_horses = SQLHorse.query.all()
        for horse in sql_horses:
            horse_data = {
                'name': horse.name,
                'age': horse.age,
                'sex': horse.sex,
                'color': horse.color,
                'sire': horse.sire,
                'dam': horse.dam,
                'trainer': horse.trainer,
                'jockey': horse.jockey,
                'owner': horse.owner,
                'weight': horse.weight,
                'form': horse.form,
                'rating': horse.rating,
                'last_run': horse.last_run,
                'wins': horse.wins,
                'places': horse.places,
                'runs': horse.runs,
                'earnings': horse.earnings
            }
            horse_id = firebase_config.save_horse(horse_data)
            print(f"Migrated horse: {horse.name} -> {horse_id}")
        
        print(f"Successfully migrated {len(sql_horses)} horses")
        
        # Migrate predictions
        print("Migrating predictions...")
        sql_predictions = SQLPrediction.query.all()
        for pred in sql_predictions:
            pred_data = {
                'race_id': str(pred.race_id),
                'horse_id': str(pred.horse_id),
                'predicted_position': pred.predicted_position,
                'confidence': pred.confidence,
                'odds': pred.odds,
                'factors': pred.factors or {},
                'model_version': pred.model_version or 'v1.0'
            }
            pred_id = firebase_config.save_prediction(pred_data)
            print(f"Migrated prediction: Race {pred.race_id}, Horse {pred.horse_id} -> {pred_id}")
        
        print(f"Successfully migrated {len(sql_predictions)} predictions")
        
        return True
        
    except ImportError:
        print("SQLAlchemy models not found. Skipping SQLite migration.")
        return True
    except Exception as e:
        print(f"Error migrating SQLite data: {e}")
        return False

def create_sample_data():
    """Create sample data in Firebase for testing"""
    print("Creating sample data in Firebase...")
    
    # Sample API credential
    sample_cred = {
        'provider': 'sample_api',
        'base_url': 'https://api.example.com',
        'api_key': 'sample_key_123',
        'api_secret': 'sample_secret_456',
        'description': 'Sample API for testing',
        'is_active': True
    }
    cred_id = firebase_config.save_api_credential(sample_cred)
    print(f"Created sample API credential: {cred_id}")
    
    # Sample race
    sample_race = {
        'name': 'Sample Stakes',
        'date': datetime(2024, 12, 25, 14, 30),
        'time': '14:30',
        'track': 'Sample Downs',
        'distance': 1200.0,
        'surface': 'Turf',
        'race_class': 'Grade 1',
        'prize_money': 100000.0,
        'weather': 'Clear',
        'track_condition': 'Good'
    }
    race_id = firebase_config.save_race(sample_race)
    print(f"Created sample race: {race_id}")
    
    # Sample horse
    sample_horse = {
        'name': 'Thunder Bolt',
        'age': 4,
        'sex': 'Gelding',
        'color': 'Bay',
        'sire': 'Lightning Strike',
        'dam': 'Storm Cloud',
        'trainer': 'John Smith',
        'jockey': 'Mike Johnson',
        'owner': 'ABC Racing Stable',
        'weight': 126.0,
        'form': '1-2-1-3-1',
        'rating': 95,
        'wins': 8,
        'places': 12,
        'runs': 15,
        'earnings': 250000.0
    }
    horse_id = firebase_config.save_horse(sample_horse)
    print(f"Created sample horse: {horse_id}")
    
    # Sample prediction
    sample_prediction = {
        'race_id': race_id,
        'horse_id': horse_id,
        'predicted_position': 1,
        'confidence': 0.85,
        'odds': 3.5,
        'factors': {
            'form_score': 0.9,
            'track_condition_score': 0.8,
            'jockey_score': 0.85
        },
        'model_version': 'v2.0'
    }
    pred_id = firebase_config.save_prediction(sample_prediction)
    print(f"Created sample prediction: {pred_id}")
    
    print("Sample data created successfully!")

def main():
    """Main migration function"""
    print("Firebase Migration Tool")
    print("=" * 50)
    
    if not firebase_config.is_connected():
        print("Error: Firebase is not connected.")
        print("Please check your Firebase configuration:")
        print("1. Set FIREBASE_SERVICE_ACCOUNT_PATH or GOOGLE_APPLICATION_CREDENTIALS")
        print("2. Ensure your Firebase project is set up correctly")
        print("3. Check your internet connection")
        return
    
    print("Firebase connected successfully!")
    
    # Ask user what to migrate
    print("\nWhat would you like to do?")
    print("1. Migrate existing JSON data")
    print("2. Migrate existing SQLite data")
    print("3. Create sample data for testing")
    print("4. All of the above")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        migrate_json_data()
    elif choice == '2':
        migrate_sqlite_data()
    elif choice == '3':
        create_sample_data()
    elif choice == '4':
        migrate_json_data()
        migrate_sqlite_data()
        create_sample_data()
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\nMigration completed!")
    print("You can now update your application to use Firebase models.")

if __name__ == '__main__':
    main()