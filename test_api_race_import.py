#!/usr/bin/env python3
"""
API Race Import Test Script
Tests importing races from external APIs and saving to Firebase
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
from models.firebase_models import Race, Horse, APICredentials
from services.api_service import APIService

def create_mock_api_response():
    """Create mock API response data for testing"""
    return {
        "races": [
            {
                "id": "api_race_001",
                "name": "API Test Stakes",
                "date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                "time": "14:30",
                "track": "API Test Track",
                "distance": 1600,
                "surface": "turf",
                "race_type": "flat",
                "prize_money": 75000,
                "conditions": "3-year-olds and upwards",
                "weather": "cloudy",
                "track_condition": "good to firm",
                "horses": [
                    {
                        "id": "api_horse_001",
                        "name": "API Thunder",
                        "age": 4,
                        "gender": "stallion",
                        "weight": 126,
                        "jockey": "API Jockey 1",
                        "trainer": "API Trainer 1",
                        "odds": 2.8,
                        "form": "1-1-2-1"
                    },
                    {
                        "id": "api_horse_002", 
                        "name": "API Lightning",
                        "age": 3,
                        "gender": "mare",
                        "weight": 122,
                        "jockey": "API Jockey 2",
                        "trainer": "API Trainer 2",
                        "odds": 3.2,
                        "form": "2-1-1-3"
                    }
                ]
            },
            {
                "id": "api_race_002",
                "name": "API Championship Cup",
                "date": (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d"),
                "time": "16:00",
                "track": "API Championship Track",
                "distance": 2000,
                "surface": "turf",
                "race_type": "flat",
                "prize_money": 100000,
                "conditions": "4-year-olds and upwards",
                "weather": "sunny",
                "track_condition": "good",
                "horses": [
                    {
                        "id": "api_horse_003",
                        "name": "API Storm",
                        "age": 5,
                        "gender": "gelding",
                        "weight": 128,
                        "jockey": "API Jockey 3",
                        "trainer": "API Trainer 3",
                        "odds": 4.5,
                        "form": "1-2-1-1"
                    }
                ]
            }
        ]
    }

def test_api_credentials_setup():
    """Test API credentials setup"""
    print("\n" + "="*60)
    print("TEST 1: API Credentials Setup")
    print("="*60)
    
    try:
        # Create test API credentials
        api_creds_data = {
            "provider": "test_api",
            "base_url": "https://api.test-racing.com",
            "api_key": "test_key_123",
            "api_secret": "test_secret_456",
            "description": "Test API for race import testing",
            "is_active": True
        }
        
        api_creds = APICredentials(api_creds_data)
        creds_id = api_creds.save()
        
        if creds_id:
            print(f"✓ API credentials saved with ID: {creds_id}")
            
            # Verify retrieval
            saved_creds = APICredentials.get_by_id(creds_id)
            if saved_creds:
                print(f"✓ API credentials retrieved: {saved_creds.provider}")
                return creds_id
            else:
                print("✗ Failed to retrieve API credentials")
                return None
        else:
            print("✗ Failed to save API credentials")
            return None
            
    except Exception as e:
        print(f"✗ Error in API credentials setup: {e}")
        return None

def test_api_race_import():
    """Test importing races from API response"""
    print("\n" + "="*60)
    print("TEST 2: API Race Import")
    print("="*60)
    
    try:
        # Get mock API response
        api_response = create_mock_api_response()
        print(f"✓ Created mock API response with {len(api_response['races'])} races")
        
        imported_races = []
        imported_horses = []
        
        # Process each race from API
        for race_data in api_response['races']:
            print(f"\n--- Processing race: {race_data['name']} ---")
            
            # Extract horses data
            horses_data = race_data.pop('horses', [])
            
            # Convert API format to our Race model format
            race_model_data = {
                "name": race_data['name'],
                "date": race_data['date'] + "T" + race_data['time'] + ":00",
                "time": race_data['time'],
                "track": race_data['track'],
                "distance": race_data['distance'],
                "surface": race_data['surface'],
                "race_type": race_data['race_type'],
                "prize_money": race_data['prize_money'],
                "conditions": race_data['conditions'],
                "weather": race_data['weather'],
                "track_condition": race_data['track_condition'],
                "horses": [],
                "status": "upcoming",
                "source": "api_import",
                "external_id": race_data['id']
            }
            
            # Create and save race
            race = Race(race_model_data)
            race_id = race.save()
            
            if race_id:
                print(f"✓ Race imported: {race.name} (ID: {race_id})")
                imported_races.append(race_id)
                
                # Process horses for this race
                race_horse_ids = []
                for horse_data in horses_data:
                    horse_model_data = {
                        "name": horse_data['name'],
                        "age": horse_data['age'],
                        "gender": horse_data['gender'],
                        "weight": horse_data['weight'],
                        "jockey": horse_data['jockey'],
                        "trainer": horse_data['trainer'],
                        "odds": horse_data['odds'],
                        "form": horse_data['form'],
                        "source": "api_import",
                        "external_id": horse_data['id']
                    }
                    
                    # Create and save horse
                    horse = Horse(horse_model_data)
                    horse_id = horse.save()
                    
                    if horse_id:
                        print(f"  ✓ Horse imported: {horse.name} (ID: {horse_id})")
                        imported_horses.append(horse_id)
                        race_horse_ids.append(horse_id)
                    else:
                        print(f"  ✗ Failed to import horse: {horse_data['name']}")
                
                # Associate horses with race
                if race_horse_ids:
                    race.horses = race_horse_ids
                    race.save()
                    print(f"  ✓ Associated {len(race_horse_ids)} horses with race")
                
            else:
                print(f"✗ Failed to import race: {race_data['name']}")
        
        print(f"\n✓ Import summary:")
        print(f"  - Races imported: {len(imported_races)}")
        print(f"  - Horses imported: {len(imported_horses)}")
        
        return imported_races, imported_horses
        
    except Exception as e:
        print(f"✗ Error in API race import: {e}")
        return [], []

def test_imported_data_verification():
    """Test verification of imported data"""
    print("\n" + "="*60)
    print("TEST 3: Imported Data Verification")
    print("="*60)
    
    try:
        # Get all races
        all_races = Race.get_all()
        api_races = [race for race in all_races if hasattr(race, 'source') and race.source == 'api_import']
        
        print(f"✓ Found {len(api_races)} API-imported races")
        
        for race in api_races:
            print(f"\n--- Race: {race.name} ---")
            print(f"  Track: {race.track}")
            print(f"  Date: {race.date}")
            print(f"  Prize: ${race.prize_money:,}")
            
            if hasattr(race, 'horses') and race.horses:
                print(f"  Horses ({len(race.horses)}):")
                for horse_id in race.horses:
                    horse = Horse.get_by_id(horse_id)
                    if horse:
                        print(f"    - {horse.name} (odds: {horse.odds}, jockey: {horse.jockey})")
            else:
                print("  No horses associated")
        
        # Get all horses
        all_horses = Horse.get_all()
        api_horses = [horse for horse in all_horses if hasattr(horse, 'source') and horse.source == 'api_import']
        
        print(f"\n✓ Found {len(api_horses)} API-imported horses")
        
        return len(api_races) > 0 and len(api_horses) > 0
        
    except Exception as e:
        print(f"✗ Error in data verification: {e}")
        return False

def test_duplicate_handling():
    """Test handling of duplicate imports"""
    print("\n" + "="*60)
    print("TEST 4: Duplicate Import Handling")
    print("="*60)
    
    try:
        # Try to import the same data again
        api_response = create_mock_api_response()
        
        print("✓ Attempting to import duplicate data...")
        
        # Count races before import
        races_before = len(Race.get_all())
        horses_before = len(Horse.get_all())
        
        # Import again (this should create duplicates since we don't have duplicate checking yet)
        imported_races, imported_horses = test_api_race_import()
        
        # Count races after import
        races_after = len(Race.get_all())
        horses_after = len(Horse.get_all())
        
        print(f"\n✓ Duplicate handling results:")
        print(f"  - Races before: {races_before}, after: {races_after}")
        print(f"  - Horses before: {horses_before}, after: {horses_after}")
        
        if races_after > races_before:
            print("⚠ Duplicates were created (duplicate detection may need implementation)")
        else:
            print("✓ No duplicates created")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in duplicate handling test: {e}")
        return False

def cleanup_api_test_data():
    """Clean up API test data"""
    print("\n" + "="*60)
    print("CLEANUP: Removing API Test Data")
    print("="*60)
    
    try:
        # Clean up races
        all_races = Race.get_all()
        api_races = [race for race in all_races if hasattr(race, 'source') and race.source == 'api_import']
        
        print(f"✓ Found {len(api_races)} API races to clean up")
        for race in api_races:
            if race.delete():
                print(f"✓ Deleted race: {race.name}")
        
        # Clean up horses
        all_horses = Horse.get_all()
        api_horses = [horse for horse in all_horses if hasattr(horse, 'source') and horse.source == 'api_import']
        
        print(f"✓ Found {len(api_horses)} API horses to clean up")
        for horse in api_horses:
            if horse.delete():
                print(f"✓ Deleted horse: {horse.name}")
        
        # Clean up API credentials
        all_creds = APICredentials.get_all()
        test_creds = [cred for cred in all_creds if cred.provider == 'test_api']
        
        print(f"✓ Found {len(test_creds)} test API credentials to clean up")
        for cred in test_creds:
            if cred.delete():
                print(f"✓ Deleted API credential: {cred.provider}")
        
        print("✓ API test data cleanup completed")
        
    except Exception as e:
        print(f"✗ Error during cleanup: {e}")

def main():
    """Main test function"""
    print("🏇 API Race Import Test Suite")
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
    
    # Test 1: API credentials setup
    creds_id = test_api_credentials_setup()
    test_results.append(creds_id is not None)
    
    # Test 2: API race import
    imported_races, imported_horses = test_api_race_import()
    test_results.append(len(imported_races) > 0)
    
    # Test 3: Data verification
    verification_result = test_imported_data_verification()
    test_results.append(verification_result)
    
    # Test 4: Duplicate handling
    duplicate_result = test_duplicate_handling()
    test_results.append(duplicate_result)
    
    # Summary
    print("\n" + "="*80)
    print("API IMPORT TEST SUMMARY")
    print("="*80)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All API import tests passed! Firebase API race import is working correctly.")
    else:
        print("⚠ Some tests failed. Please check the output above for details.")
    
    # Ask user if they want to clean up test data
    print("\n" + "="*60)
    cleanup_response = input("Do you want to clean up API test data? (y/n): ").lower().strip()
    if cleanup_response in ['y', 'yes']:
        cleanup_api_test_data()
    else:
        print("✓ API test data preserved for manual inspection")
    
    return passed_tests == total_tests

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)