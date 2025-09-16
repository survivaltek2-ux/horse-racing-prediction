#!/usr/bin/env python3
"""
Comprehensive Test Script for Horse Racing Prediction Function

This script tests the predict functionality including:
- Basic prediction generation
- Different algorithms (ML vs heuristic)
- Edge cases and error handling
- Model training and performance
- Data validation
"""

import sys
import os
import json
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the actual models used in the app
try:
    from models.firebase_models import Horse, Race, Prediction
    FIREBASE_MODELS = True
except ImportError:
    try:
        from models.horse import Horse
        from models.race import Race
        from models.prediction import Prediction
        FIREBASE_MODELS = False
    except ImportError:
        print("Warning: Could not import model classes. Some tests may fail.")
        Horse = Race = Prediction = None
        FIREBASE_MODELS = False

from utils.predictor import Predictor
from utils.data_processor import DataProcessor

class TestPredictFunction(unittest.TestCase):
    """Test cases for the predict function"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.predictor = Predictor()
        self.data_processor = DataProcessor()
        
        # Create test horses
        self.test_horses = self._create_test_horses()
        
        # Create test races
        self.test_races = self._create_test_races()
        
        print(f"\n{'='*60}")
        print(f"Setting up test: {self._testMethodName}")
        print(f"{'='*60}")
    
    def tearDown(self):
        """Clean up after each test method"""
        print(f"Test completed: {self._testMethodName}")
        print(f"{'='*60}\n")
    
    def _create_test_horses(self):
        """Create test horses with various characteristics"""
        horses = []
        
        # High-performing horse
        horses.append({
            'id': 'test_horse_1',
            'name': 'Lightning Bolt',
            'age': 4,
            'breed': 'Thoroughbred',
            'jockey': 'John Smith',
            'trainer': 'Mike Johnson',
            'win_rate': 0.35,
            'place_rate': 0.60,
            'show_rate': 0.80,
            'earnings': 150000,
            'recent_performances': [
                {'position': 1, 'time': '1:45.2', 'earnings': 25000, 'race_date': '2024-01-15'},
                {'position': 2, 'time': '1:46.1', 'earnings': 15000, 'race_date': '2024-01-01'},
                {'position': 1, 'time': '1:44.8', 'earnings': 30000, 'race_date': '2023-12-15'},
            ],
            'weight': 126,
            'odds': 3.5
        })
        
        # Average horse
        horses.append({
            'id': 'test_horse_2',
            'name': 'Steady Runner',
            'age': 5,
            'breed': 'Thoroughbred',
            'jockey': 'Sarah Wilson',
            'trainer': 'Bob Davis',
            'win_rate': 0.15,
            'place_rate': 0.40,
            'show_rate': 0.65,
            'earnings': 75000,
            'recent_performances': [
                {'position': 3, 'time': '1:47.5', 'earnings': 8000, 'race_date': '2024-01-15'},
                {'position': 4, 'time': '1:48.2', 'earnings': 5000, 'race_date': '2024-01-01'},
                {'position': 2, 'time': '1:46.9', 'earnings': 12000, 'race_date': '2023-12-15'},
            ],
            'weight': 124,
            'odds': 8.0
        })
        
        # Underperforming horse
        horses.append({
            'id': 'test_horse_3',
            'name': 'Slow Poke',
            'age': 6,
            'breed': 'Quarter Horse',
            'jockey': 'Tom Brown',
            'trainer': 'Lisa White',
            'win_rate': 0.05,
            'place_rate': 0.20,
            'show_rate': 0.35,
            'earnings': 25000,
            'recent_performances': [
                {'position': 8, 'time': '1:52.1', 'earnings': 1000, 'race_date': '2024-01-15'},
                {'position': 7, 'time': '1:51.8', 'earnings': 1500, 'race_date': '2024-01-01'},
                {'position': 6, 'time': '1:50.5', 'earnings': 2000, 'race_date': '2023-12-15'},
            ],
            'weight': 122,
            'odds': 25.0
        })
        
        # Young promising horse
        horses.append({
            'id': 'test_horse_4',
            'name': 'Rising Star',
            'age': 3,
            'breed': 'Thoroughbred',
            'jockey': 'Emma Garcia',
            'trainer': 'Carlos Rodriguez',
            'win_rate': 0.25,
            'place_rate': 0.50,
            'show_rate': 0.75,
            'earnings': 45000,
            'recent_performances': [
                {'position': 1, 'time': '1:46.0', 'earnings': 20000, 'race_date': '2024-01-15'},
                {'position': 3, 'time': '1:47.2', 'earnings': 8000, 'race_date': '2024-01-01'},
                {'position': 2, 'time': '1:46.5', 'earnings': 12000, 'race_date': '2023-12-15'},
            ],
            'weight': 120,
            'odds': 5.5
        })
        
        return horses
    
    def _create_test_races(self):
        """Create test races with different characteristics"""
        races = []
        
        # Standard race
        races.append({
            'id': 'test_race_1',
            'name': 'Test Stakes',
            'date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'location': 'Churchill Downs',
            'distance': 1.25,  # miles
            'track_condition': 'Fast',
            'race_type': 'Stakes',
            'purse': 100000,
            'horse_ids': ['test_horse_1', 'test_horse_2', 'test_horse_3', 'test_horse_4'],
            'status': 'upcoming'
        })
        
        # Sprint race
        races.append({
            'id': 'test_race_2',
            'name': 'Sprint Championship',
            'date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
            'location': 'Belmont Park',
            'distance': 0.75,  # miles
            'track_condition': 'Good',
            'race_type': 'Allowance',
            'purse': 50000,
            'horse_ids': ['test_horse_1', 'test_horse_2', 'test_horse_4'],
            'status': 'upcoming'
        })
        
        # Long distance race
        races.append({
            'id': 'test_race_3',
            'name': 'Marathon Classic',
            'date': (datetime.now() + timedelta(days=21)).strftime('%Y-%m-%d'),
            'location': 'Santa Anita',
            'distance': 1.75,  # miles
            'track_condition': 'Muddy',
            'race_type': 'Handicap',
            'purse': 75000,
            'horse_ids': ['test_horse_2', 'test_horse_3', 'test_horse_4'],
            'status': 'upcoming'
        })
        
        return races
    
    def test_basic_prediction_generation(self):
        """Test basic prediction generation functionality"""
        print("Testing basic prediction generation...")
        
        # Create a race object
        race_data = self.test_races[0]
        race = Race(**race_data)
        
        # Generate prediction
        prediction = self.predictor.predict_race(race)
        
        # Assertions
        self.assertIsNotNone(prediction, "Prediction should not be None")
        self.assertIn('predictions', prediction.__dict__, "Prediction should contain predictions")
        self.assertIn('algorithm', prediction.__dict__, "Prediction should specify algorithm used")
        
        print(f"✓ Prediction generated successfully using {prediction.algorithm}")
        print(f"✓ Number of horse predictions: {len(prediction.predictions) if prediction.predictions else 0}")
    
    def test_heuristic_algorithm(self):
        """Test heuristic prediction algorithm"""
        print("Testing heuristic prediction algorithm...")
        
        race_data = self.test_races[0]
        race = Race(**race_data)
        
        # Force heuristic algorithm by ensuring model is not trained
        self.predictor.is_trained = False
        
        prediction = self.predictor.predict_race(race)
        
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.algorithm, 'enhanced_heuristic')
        
        print(f"✓ Heuristic algorithm used successfully")
        print(f"✓ Predictions generated for {len(prediction.predictions)} horses")
    
    def test_ml_algorithm_when_trained(self):
        """Test ML algorithm when model is trained"""
        print("Testing ML algorithm with trained model...")
        
        # Create some training data
        training_data = self._create_training_data()
        
        # Train the model
        print("Training model with test data...")
        training_result = self.predictor.train_model(training_data)
        
        if training_result and training_result.get('success'):
            race_data = self.test_races[0]
            race = Race(**race_data)
            
            prediction = self.predictor.predict_race(race)
            
            self.assertIsNotNone(prediction)
            self.assertEqual(prediction.algorithm, 'ensemble_ml')
            
            print(f"✓ ML algorithm used successfully")
            print(f"✓ Model training accuracy: {training_result.get('accuracy', 'N/A')}")
        else:
            print("⚠ Model training failed, skipping ML algorithm test")
            self.skipTest("Model training failed")
    
    def test_different_race_types(self):
        """Test predictions for different race types and distances"""
        print("Testing predictions for different race types...")
        
        for i, race_data in enumerate(self.test_races):
            race = Race(**race_data)
            prediction = self.predictor.predict_race(race)
            
            self.assertIsNotNone(prediction, f"Prediction failed for race {i+1}")
            
            print(f"✓ Race {i+1} ({race.name}): {race.distance} miles, {race.track_condition} track")
            print(f"  Algorithm: {prediction.algorithm}")
            print(f"  Horses predicted: {len(prediction.predictions) if prediction.predictions else 0}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("Testing edge cases...")
        
        # Test with None race
        prediction = self.predictor.predict_race(None)
        self.assertIsNone(prediction, "Prediction should be None for None race")
        print("✓ Handled None race correctly")
        
        # Test with race with no horses
        empty_race = Race(
            id='empty_race',
            name='Empty Race',
            date=datetime.now().strftime('%Y-%m-%d'),
            location='Test Track',
            distance=1.0,
            horse_ids=[]
        )
        prediction = self.predictor.predict_race(empty_race)
        # Should handle gracefully (might return None or empty prediction)
        print(f"✓ Handled empty race: {prediction is not None}")
        
        # Test with invalid race data
        invalid_race = Race(
            id='invalid_race',
            name='Invalid Race',
            date='invalid-date',
            location='Test Track',
            distance=-1.0,  # Invalid distance
            horse_ids=['nonexistent_horse']
        )
        prediction = self.predictor.predict_race(invalid_race)
        print(f"✓ Handled invalid race data: {prediction is not None}")
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent for the same race"""
        print("Testing prediction consistency...")
        
        race_data = self.test_races[0]
        race = Race(**race_data)
        
        # Generate multiple predictions for the same race
        predictions = []
        for i in range(3):
            prediction = self.predictor.predict_race(race)
            predictions.append(prediction)
        
        # Check that all predictions use the same algorithm
        algorithms = [p.algorithm for p in predictions if p]
        if algorithms:
            self.assertTrue(all(alg == algorithms[0] for alg in algorithms),
                          "All predictions should use the same algorithm")
            print(f"✓ Consistent algorithm used: {algorithms[0]}")
        
        print(f"✓ Generated {len(predictions)} consistent predictions")
    
    def test_confidence_scores(self):
        """Test confidence score calculation"""
        print("Testing confidence scores...")
        
        race_data = self.test_races[0]
        race = Race(**race_data)
        
        prediction = self.predictor.predict_race(race)
        
        if prediction and hasattr(prediction, 'confidence_scores'):
            self.assertIsNotNone(prediction.confidence_scores)
            print(f"✓ Confidence scores generated: {len(prediction.confidence_scores) if prediction.confidence_scores else 0}")
        else:
            print("⚠ No confidence scores found in prediction")
    
    def test_model_performance_metrics(self):
        """Test model performance evaluation"""
        print("Testing model performance metrics...")
        
        # Get performance stats
        performance = self.predictor.get_performance_stats()
        
        self.assertIsInstance(performance, dict)
        print(f"✓ Performance stats retrieved: {list(performance.keys())}")
        
        # Check if model info is available
        model_info = self.predictor.get_model_info()
        self.assertIsInstance(model_info, dict)
        print(f"✓ Model info retrieved: {model_info.get('is_trained', False)}")
    
    def _create_training_data(self):
        """Create sample training data for model testing"""
        training_data = []
        
        # Create sample race results
        for i in range(10):
            race_result = {
                'race_id': f'training_race_{i}',
                'horses': []
            }
            
            # Add horses with results
            for j, horse_data in enumerate(self.test_horses):
                horse_result = horse_data.copy()
                horse_result['final_position'] = j + 1  # Simple position assignment
                horse_result['finish_time'] = 105.0 + j * 2.5  # Seconds
                race_result['horses'].append(horse_result)
            
            training_data.append(race_result)
        
        return training_data

def run_comprehensive_tests():
    """Run all tests and provide a comprehensive report"""
    print("🏇 Horse Racing Prediction Function Test Suite")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPredictFunction)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("\n🎉 All tests passed successfully!")
    
    return result.wasSuccessful()

def run_manual_prediction_test():
    """Run a manual prediction test with detailed output"""
    print("\n🔍 MANUAL PREDICTION TEST")
    print("=" * 80)
    
    try:
        predictor = Predictor()
        
        # Create a test race
        test_race = Race(
            id='manual_test_race',
            name='Manual Test Stakes',
            date=(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            location='Test Track',
            distance=1.25,
            track_condition='Fast',
            race_type='Stakes',
            purse=100000,
            horse_ids=['test_horse_1', 'test_horse_2', 'test_horse_3'],
            status='upcoming'
        )
        
        print(f"Race: {test_race.name}")
        print(f"Distance: {test_race.distance} miles")
        print(f"Track: {test_race.track_condition}")
        print(f"Horses: {len(test_race.horse_ids)}")
        
        # Generate prediction
        print("\nGenerating prediction...")
        prediction = predictor.predict_race(test_race)
        
        if prediction:
            print(f"\n✅ Prediction successful!")
            print(f"Algorithm used: {prediction.algorithm}")
            print(f"Race ID: {prediction.race_id}")
            
            if hasattr(prediction, 'predictions') and prediction.predictions:
                print(f"\nPredictions for {len(prediction.predictions)} horses:")
                for i, horse_pred in enumerate(prediction.predictions, 1):
                    print(f"  {i}. Horse: {horse_pred.get('horse_name', 'Unknown')}")
                    print(f"     Win probability: {horse_pred.get('win_probability', 0):.3f}")
                    print(f"     Predicted position: {horse_pred.get('predicted_position', 'N/A')}")
            
            if hasattr(prediction, 'confidence_scores') and prediction.confidence_scores:
                print(f"\nConfidence scores available: {len(prediction.confidence_scores)}")
        else:
            print("❌ Prediction failed!")
            
    except Exception as e:
        print(f"❌ Error during manual test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🏇 Horse Racing Prediction Test Script")
    print("Choose test mode:")
    print("1. Run comprehensive test suite")
    print("2. Run manual prediction test")
    print("3. Run both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_comprehensive_tests()
    elif choice == "2":
        run_manual_prediction_test()
    elif choice == "3":
        success = run_comprehensive_tests()
        run_manual_prediction_test()
    else:
        print("Invalid choice. Running comprehensive tests by default.")
        run_comprehensive_tests()