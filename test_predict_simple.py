#!/usr/bin/env python3
"""
Simple Test Script for Horse Racing Prediction Function

This script provides a practical test of the predict functionality
using the actual application structure and data models.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.predictor import Predictor

def test_predictor_initialization():
    """Test that the Predictor class can be initialized"""
    print("🔧 Testing Predictor Initialization...")
    try:
        predictor = Predictor()
        print("✅ Predictor initialized successfully")
        
        # Check if models are set up
        if hasattr(predictor, 'models'):
            print(f"✅ Models available: {list(predictor.models.keys()) if predictor.models else 'None'}")
        
        # Check training status
        print(f"📊 Model trained: {getattr(predictor, 'is_trained', False)}")
        
        return predictor
    except Exception as e:
        print(f"❌ Error initializing Predictor: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_predictor_methods():
    """Test various Predictor methods"""
    print("\n🔍 Testing Predictor Methods...")
    
    predictor = Predictor()
    
    # Test get_performance_stats
    try:
        stats = predictor.get_performance_stats()
        print(f"✅ Performance stats: {type(stats)} with keys: {list(stats.keys()) if isinstance(stats, dict) else 'Not a dict'}")
    except Exception as e:
        print(f"❌ Error getting performance stats: {str(e)}")
    
    # Test get_model_info
    try:
        info = predictor.get_model_info()
        print(f"✅ Model info: {type(info)} with keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")
    except Exception as e:
        print(f"❌ Error getting model info: {str(e)}")

def test_with_mock_race():
    """Test prediction with a mock race object"""
    print("\n🏇 Testing with Mock Race...")
    
    predictor = Predictor()
    
    # Create a simple mock race object
    class MockRace:
        def __init__(self):
            self.id = 'test_race_001'
            self.name = 'Test Stakes'
            self.date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            self.location = 'Test Track'
            self.distance = 1.25
            self.track_condition = 'Fast'
            self.race_type = 'Stakes'
            self.purse = 100000
            self.horse_ids = ['horse_1', 'horse_2', 'horse_3']
            self.status = 'upcoming'
        
        def get_horses(self):
            """Return mock horses"""
            return [MockHorse(f'horse_{i}', f'Test Horse {i}') for i in range(1, 4)]
    
    class MockHorse:
        def __init__(self, horse_id, name):
            self.id = horse_id
            self.name = name
            self.age = 4
            self.breed = 'Thoroughbred'
            self.jockey = 'Test Jockey'
            self.trainer = 'Test Trainer'
            self.win_rate = 0.2
            self.place_rate = 0.4
            self.show_rate = 0.6
            self.earnings = 50000
            self.weight = 126
            self.odds = 5.0
            self.races_count = 10
            self.avg_position = 3.5
            self.position_std = 1.2
            self.class_level = 'Stakes'
        
        def get_form(self):
            """Return mock form data"""
            return [
                {'position': 1, 'time': 105.2, 'date': '2024-01-15'},
                {'position': 3, 'time': 107.1, 'date': '2024-01-01'},
                {'position': 2, 'time': 106.5, 'date': '2023-12-15'},
            ]
        
        def get_recent_performances(self, count=5):
            """Return mock recent performances"""
            return self.get_form()[:count]
    
    # Test prediction
    try:
        mock_race = MockRace()
        print(f"📋 Mock race created: {mock_race.name} with {len(mock_race.horse_ids)} horses")
        
        prediction = predictor.predict_race(mock_race)
        
        if prediction:
            print(f"✅ Prediction generated successfully!")
            print(f"   Race ID: {getattr(prediction, 'race_id', 'N/A')}")
            print(f"   Algorithm: {getattr(prediction, 'algorithm', 'N/A')}")
            
            if hasattr(prediction, 'predictions') and prediction.predictions:
                print(f"   Predictions count: {len(prediction.predictions)}")
                for i, pred in enumerate(prediction.predictions[:3], 1):  # Show first 3
                    if isinstance(pred, dict):
                        print(f"     {i}. Horse: {pred.get('horse_name', 'Unknown')}")
                        print(f"        Win prob: {pred.get('win_probability', 0):.3f}")
                        print(f"        Position: {pred.get('predicted_position', 'N/A')}")
            else:
                print("   No detailed predictions available")
        else:
            print("❌ Prediction returned None")
            
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

def test_data_processor():
    """Test the DataProcessor functionality"""
    print("\n📊 Testing DataProcessor...")
    
    try:
        from utils.data_processor import DataProcessor
        processor = DataProcessor()
        print("✅ DataProcessor initialized successfully")
        
        # Test feature importance
        try:
            features = processor.get_feature_importance()
            print(f"✅ Feature importance: {type(features)} with {len(features) if isinstance(features, (list, dict)) else 'unknown'} items")
        except Exception as e:
            print(f"❌ Error getting feature importance: {str(e)}")
            
    except Exception as e:
        print(f"❌ Error with DataProcessor: {str(e)}")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n⚠️  Testing Edge Cases...")
    
    predictor = Predictor()
    
    # Test with None
    try:
        result = predictor.predict_race(None)
        print(f"✅ None race handled: {result is None}")
    except Exception as e:
        print(f"❌ Error with None race: {str(e)}")
    
    # Test with empty object
    class EmptyRace:
        pass
    
    try:
        empty_race = EmptyRace()
        result = predictor.predict_race(empty_race)
        print(f"✅ Empty race handled: {result is None}")
    except Exception as e:
        print(f"❌ Error with empty race: {str(e)}")

def run_all_tests():
    """Run all tests and provide summary"""
    print("🏇 Horse Racing Prediction Function - Simple Test Suite")
    print("=" * 70)
    
    tests = [
        ("Predictor Initialization", test_predictor_initialization),
        ("Predictor Methods", test_predictor_methods),
        ("Mock Race Prediction", test_with_mock_race),
        ("DataProcessor", test_data_processor),
        ("Edge Cases", test_edge_cases)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            print("-" * 50)
            test_func()
            passed += 1
            print(f"✅ {test_name} completed")
        except Exception as e:
            print(f"❌ {test_name} failed: {str(e)}")
    
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"Tests completed: {passed}/{total}")
    print(f"Success rate: {(passed/total*100):.1f}%")
    
    if passed == total:
        print("🎉 All tests completed successfully!")
    else:
        print(f"⚠️  {total - passed} test(s) had issues")
    
    return passed == total

def interactive_test():
    """Run an interactive test session"""
    print("\n🎮 Interactive Test Mode")
    print("=" * 50)
    
    predictor = Predictor()
    
    while True:
        print("\nChoose an option:")
        print("1. Test basic prediction")
        print("2. Check model status")
        print("3. Test performance stats")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            test_with_mock_race()
        elif choice == "2":
            try:
                info = predictor.get_model_info()
                print(f"Model Info: {info}")
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "3":
            try:
                stats = predictor.get_performance_stats()
                print(f"Performance Stats: {stats}")
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "4":
            print("Goodbye! 👋")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("🏇 Horse Racing Prediction Test Script")
    print("Choose test mode:")
    print("1. Run all tests")
    print("2. Interactive mode")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        run_all_tests()
    elif choice == "2":
        interactive_test()
    else:
        print("Invalid choice. Running all tests by default.")
        run_all_tests()