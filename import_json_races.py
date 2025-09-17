#!/usr/bin/env python3
"""
Simple JSON Race Data Import
Imports race data from JSON files directly into the database.
"""

import sys
import os
import json
from datetime import datetime

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def import_races_from_json():
    """Import races from JSON files"""
    print("📊 IMPORTING RACE DATA FROM JSON")
    print("-" * 40)
    
    try:
        from app import app, db
        from models.sqlalchemy_models import Race
        
        with app.app_context():
            # Clear existing races
            existing_count = Race.query.count()
            print(f"🗑️  Clearing {existing_count} existing races...")
            Race.query.delete()
            db.session.commit()
            
            # Import from sample_races.json
            json_file = 'data/sample_races.json'
            
            if not os.path.exists(json_file):
                print(f"❌ File not found: {json_file}")
                return False
            
            print(f"📁 Reading {json_file}...")
            
            with open(json_file, 'r') as f:
                races_data = json.load(f)
            
            print(f"📋 Found {len(races_data)} races to import")
            
            imported_count = 0
            
            for race_data in races_data:
                try:
                    # Parse date
                    if isinstance(race_data.get('date'), str):
                        try:
                            race_date = datetime.fromisoformat(race_data['date'].replace('Z', '+00:00'))
                        except:
                            race_date = datetime.now()
                    else:
                        race_date = datetime.now()
                    
                    # Create race with only the fields that exist in the model
                    race = Race(
                        name=race_data.get('name', f'Race {imported_count + 1}'),
                        date=race_date,
                        time=race_data.get('time', '12:00'),
                        track=race_data.get('track', 'Unknown Track'),
                        distance=float(race_data.get('distance', 1200)),
                        surface=race_data.get('surface', 'Dirt'),
                        race_class=race_data.get('race_class', 'Maiden'),
                        prize_money=float(race_data.get('prize_money', 10000)),
                        weather=race_data.get('weather', 'Clear'),
                        track_condition=race_data.get('track_condition', 'Good'),
                        status=race_data.get('status', 'upcoming')
                    )
                    
                    db.session.add(race)
                    imported_count += 1
                    
                    if imported_count % 10 == 0:
                        print(f"   ✅ Imported {imported_count} races...")
                        
                except Exception as e:
                    print(f"   ⚠️  Skipped race due to error: {e}")
                    continue
            
            # Commit all changes
            db.session.commit()
            
            print(f"\n🎉 Successfully imported {imported_count} races!")
            
            # Verify import
            total_races = Race.query.count()
            print(f"📊 Total races in database: {total_races}")
            
            # Show some sample races
            sample_races = Race.query.limit(3).all()
            print(f"\n📋 Sample imported races:")
            for race in sample_races:
                print(f"   • {race.name} at {race.track} ({race.date.strftime('%Y-%m-%d')})")
            
            return imported_count > 0
            
    except Exception as e:
        print(f"❌ Error importing races: {e}")
        import traceback
        traceback.print_exc()
        return False

def import_horses_from_json():
    """Import horses from JSON files if available"""
    print("\n🐎 IMPORTING HORSE DATA FROM JSON")
    print("-" * 40)
    
    try:
        from app import app, db
        from models.sqlalchemy_models import Horse
        
        with app.app_context():
            # Check for horse JSON files
            horse_files = ['data/sample_horses.json', 'data/horses.json']
            
            for json_file in horse_files:
                if os.path.exists(json_file):
                    print(f"📁 Reading {json_file}...")
                    
                    with open(json_file, 'r') as f:
                        horses_data = json.load(f)
                    
                    print(f"📋 Found {len(horses_data)} horses to import")
                    
                    imported_count = 0
                    
                    for horse_data in horses_data:
                        try:
                            # Create horse with basic fields
                            horse = Horse(
                                name=horse_data.get('name', f'Horse {imported_count + 1}'),
                                age=int(horse_data.get('age', 4)),
                                sex=horse_data.get('sex', 'M'),
                                color=horse_data.get('color', 'Bay'),
                                trainer=horse_data.get('trainer', 'Unknown Trainer'),
                                jockey=horse_data.get('jockey', 'Unknown Jockey'),
                                owner=horse_data.get('owner', 'Unknown Owner'),
                                weight=float(horse_data.get('weight', 120.0)),
                                form=horse_data.get('form', '1-1-1'),
                                rating=int(horse_data.get('rating', 80))
                            )
                            
                            db.session.add(horse)
                            imported_count += 1
                            
                        except Exception as e:
                            print(f"   ⚠️  Skipped horse due to error: {e}")
                            continue
                    
                    db.session.commit()
                    print(f"✅ Imported {imported_count} horses from {json_file}")
                    return True
            
            print("ℹ️  No horse JSON files found, skipping horse import")
            return True
            
    except Exception as e:
        print(f"❌ Error importing horses: {e}")
        return False

def main():
    """Main import function"""
    print("🚀 JSON DATA IMPORT")
    print("=" * 50)
    
    # Import races
    races_success = import_races_from_json()
    
    # Import horses if available
    horses_success = import_horses_from_json()
    
    # Summary
    print(f"\n{'='*50}")
    print("IMPORT SUMMARY")
    print(f"{'='*50}")
    
    if races_success:
        print("✅ Race data imported successfully")
    else:
        print("❌ Race data import failed")
    
    if horses_success:
        print("✅ Horse data processed")
    else:
        print("⚠️  Horse data import had issues")
    
    if races_success:
        print("\n🎉 DATABASE IS NOW POPULATED!")
        print("💡 Try refreshing the races page: http://localhost:5000/races")
    else:
        print("\n🚨 IMPORT FAILED")
        print("💡 Check the error messages above")
    
    return races_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)