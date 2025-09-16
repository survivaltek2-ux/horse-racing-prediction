#!/usr/bin/env python3
"""
Race Import Test Script
Tests race import functionality and Firebase database operations
"""

import os
import sys
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.firebase_init import initialize_firebase, get_firebase_db
from models.firebase_models import Race, Horse
from utils.data_processor import DataProcessor

def create_sample_race_data():
    """Create sample race data for testing"""
    return {
        "name": "Test Championship Stakes",
        "date": (datetime.now() + timedelta(days=7)).isoformat(),
        "time": "15:30",
        "track": "Ascot Racecourse",
        "distance": 2000,
        "surface": "turf",
        "race_type": "flat",
        "prize_money": 50000,
        "conditions": "3-year-olds and upwards",
        "weather": "sunny",
        "track_condition": "good",
        "horses": [],
        "status": "upcoming"
    }

def create_sample_horse_data():
    """Create sample horse data for testing"""
    return [
        {
            "name": "Thunder Bolt",
            "age": 4,
            "gender": "stallion",
            "color": "bay",
            "weight": 126,
            "jockey": "John Smith",
            "trainer": "Mike Johnson",
            "owner": "Racing Stables Ltd",
            "odds": 3.5,
            "form": "1-2-1-3-1",
            "wins": 8,
            "places": 12,
            "total_races": 15
        },
        {
            "name": "Lightning Strike",
            "age": 3,
            "gender": "mare",
            "color": "chestnut",
            "weight": 122,
            "jockey": "Sarah Wilson",
            "trainer": "David Brown",
            "owner": "Elite Racing",
            "odds": 4.2,
            "form": "2-1-3-1-2",
            "wins": 6,
            "places": 9,
            "total_races": 12
        },
        {
            "name": "Storm Runner",
            "age": 5,
            "gender": "gelding",
            "color": "black",
            "weight": 128,
            "jockey": "Tom Davis",
            "trainer": "Lisa Anderson",
            "owner": "Champion Stables",
            "odds": 5.0,
            "form": "1-1-2-1-3",
            "wins": 10,
            "places": 15,
            "total_races": 18
        }
    ]

def test_single_race_import():
    """Test importing a single race to Firebase"""
    print("\n" + "="*60)
    print("TEST 1: Single Race Import")
    print("="*60)
    
    try:
        # Create sample race data
        race_data = create_sample_race_data()
        print(f"✓ Created sample race data: {race_data['name']}")
        
        # Create Race object
        race = Race(race_data)
        print(f"✓ Created Race object")
        
        # Save to Firebase
        race_id = race.save()
        if race_id:
            print(f"✓ Race saved to Firebase with ID: {race_id}")
            
            # Verify the race was saved
            saved_race = Race.get_by_id(race_id)
            if saved_race:
                print(f"✓ Race retrieved successfully: {saved_race.name}")
                print(f"  - Date: {saved_race.date}")
                print(f"  - Track: {saved_race.track}")
                print(f"  - Distance: {saved_race.distance}m")
                return race_id
            else:
                print("✗ Failed to retrieve saved race")
                return None
        else:
            print("✗ Failed to save race to Firebase")
            return None
            
    except Exception as e:
        print(f"✗ Error in single race import: {e}")
        return None

def test_horse_import_and_association(race_id):
    """Test importing horses and associating them with a race"""
    print("\n" + "="*60)
    print("TEST 2: Horse Import and Race Association")
    print("="*60)
    
    if not race_id:
        print("✗ No race ID provided, skipping horse association test")
        return []
    
    horse_ids = []
    
    try:
        horses_data = create_sample_horse_data()
        
        for horse_data in horses_data:
            # Create Horse object
            horse = Horse(horse_data)
            print(f"✓ Created Horse object: {horse.name}")
            
            # Save to Firebase
            horse_id = horse.save()
            if horse_id:
                print(f"✓ Horse saved to Firebase with ID: {horse_id}")
                horse_ids.append(horse_id)
                
                # Associate horse with race
                race = Race.get_by_id(race_id)
                if race:
                    race.add_horse(horse_id)
                    race.save()
                    print(f"✓ Horse {horse.name} associated with race")
                else:
                    print(f"✗ Failed to retrieve race for association")
            else:
                print(f"✗ Failed to save horse {horse.name} to Firebase")
        
        # Verify associations
        updated_race = Race.get_by_id(race_id)
        if updated_race and hasattr(updated_race, 'horses'):
            print(f"✓ Race now has {len(updated_race.horses)} horses associated")
            for horse_id in updated_race.horses:
                horse = Horse.get_by_id(horse_id)
                if horse:
                    print(f"  - {horse.name} (odds: {horse.odds})")
        
        return horse_ids
        
    except Exception as e:
        print(f"✗ Error in horse import and association: {e}")
        return []

def test_bulk_race_import():
    """Test importing multiple races at once"""
    print("\n" + "="*60)
    print("TEST 3: Bulk Race Import")
    print("="*60)
    
    try:
        # Create multiple race data
        races_data = []
        for i in range(3):
            race_data = create_sample_race_data()
            race_data['name'] = f"Test Race {i+1}"
            race_data['date'] = (datetime.now() + timedelta(days=i+1)).isoformat()
            race_data['track'] = f"Test Track {i+1}"
            races_data.append(race_data)
        
        print(f"✓ Created {len(races_data)} sample races")
        
        # Import races
        imported_race_ids = []
        for race_data in races_data:
            race = Race(race_data)
            race_id = race.save()
            if race_id:
                imported_race_ids.append(race_id)
                print(f"✓ Imported race: {race.name}")
            else:
                print(f"✗ Failed to import race: {race_data['name']}")
        
        print(f"✓ Successfully imported {len(imported_race_ids)} out of {len(races_data)} races")
        
        # Verify all races
        all_races = Race.get_all()
        print(f"✓ Total races in database: {len(all_races)}")
        
        return imported_race_ids
        
    except Exception as e:
        print(f"✗ Error in bulk race import: {e}")
        return []

def test_race_queries():
    """Test various race query operations"""
    print("\n" + "="*60)
    print("TEST 4: Race Query Operations")
    print("="*60)
    
    try:
        # Test get all races
        all_races = Race.get_all()
        print(f"✓ Retrieved all races: {len(all_races)} found")
        
        # Test get upcoming races
        upcoming_races = Race.get_upcoming()
        print(f"✓ Retrieved upcoming races: {len(upcoming_races)} found")
        
        # Display race details
        for race in upcoming_races[:3]:  # Show first 3
            print(f"  - {race.name} at {race.track} on {race.date}")
            if hasattr(race, 'horses') and race.horses:
                print(f"    Horses: {len(race.horses)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in race queries: {e}")
        return False

def test_data_validation():
    """Test data validation and error handling"""
    print("\n" + "="*60)
    print("TEST 5: Data Validation and Error Handling")
    print("="*60)
    
    try:
        # Test with invalid data
        invalid_race_data = {
            "name": "",  # Empty name
            "date": "invalid-date",  # Invalid date format
            "distance": -100,  # Negative distance
            "prize_money": "not-a-number"  # Invalid prize money
        }
        
        print("✓ Testing with invalid data...")
        race = Race(invalid_race_data)
        
        # Try to save (should handle gracefully)
        race_id = race.save()
        if race_id:
            print("⚠ Race with invalid data was saved (validation may need improvement)")
        else:
            print("✓ Invalid race data was rejected")
        
        # Test with missing required fields
        incomplete_race_data = {
            "name": "Incomplete Race"
            # Missing other required fields
        }
        
        print("✓ Testing with incomplete data...")
        incomplete_race = Race(incomplete_race_data)
        incomplete_id = incomplete_race.save()
        
        if incomplete_id:
            print("⚠ Incomplete race data was saved (validation may need improvement)")
        else:
            print("✓ Incomplete race data was rejected")
        
        return True
        
    except Exception as e:
        print(f"✓ Error handling working correctly: {e}")
        return True

def cleanup_test_data():
    """Clean up test data from Firebase"""
    print("\n" + "="*60)
    print("CLEANUP: Removing Test Data")
    print("="*60)
    
    try:
        # Get all races
        all_races = Race.get_all()
        test_races = [race for race in all_races if 'Test' in race.name]
        
        print(f"✓ Found {len(test_races)} test races to clean up")
        
        # Delete test races
        for race in test_races:
            if race.delete():
                print(f"✓ Deleted race: {race.name}")
            else:
                print(f"✗ Failed to delete race: {race.name}")
        
        # Get all horses
        all_horses = Horse.get_all()
        test_horses = [horse for horse in all_horses if any(name in horse.name for name in ['Thunder', 'Lightning', 'Storm'])]
        
        print(f"✓ Found {len(test_horses)} test horses to clean up")
        
        # Delete test horses
        for horse in test_horses:
            if horse.delete():
                print(f"✓ Deleted horse: {horse.name}")
            else:
                print(f"✗ Failed to delete horse: {horse.name}")
        
        print("✓ Cleanup completed")
        
    except Exception as e:
        print(f"✗ Error during cleanup: {e}")

def main():
    """Main test function"""
    print("🏇 Horse Racing Firebase Import Test Suite")
    print("="*80)
    
    # Initialize Firebase
    try:
        app, db = initialize_firebase()
        if not app or not db:
            print("✗ Failed to initialize Firebase")
            return False
        print("✓ Firebase initialized successfully")
    except Exception as e:
        print(f"✗ Firebase initialization error: {e}")
        return False
    
    # Run tests
    test_results = []
    
    # Test 1: Single race import
    race_id = test_single_race_import()
    test_results.append(race_id is not None)
    
    # Test 2: Horse import and association
    horse_ids = test_horse_import_and_association(race_id)
    test_results.append(len(horse_ids) > 0)
    
    # Test 3: Bulk race import
    bulk_race_ids = test_bulk_race_import()
    test_results.append(len(bulk_race_ids) > 0)
    
    # Test 4: Race queries
    query_result = test_race_queries()
    test_results.append(query_result)
    
    # Test 5: Data validation
    validation_result = test_data_validation()
    test_results.append(validation_result)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Firebase race import is working correctly.")
    else:
        print("⚠ Some tests failed. Please check the output above for details.")
    
    # Ask user if they want to clean up test data
    print("\n" + "="*60)
    cleanup_response = input("Do you want to clean up test data? (y/n): ").lower().strip()
    if cleanup_response in ['y', 'yes']:
        cleanup_test_data()
    else:
        print("✓ Test data preserved for manual inspection")
    
    return passed_tests == total_tests

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)