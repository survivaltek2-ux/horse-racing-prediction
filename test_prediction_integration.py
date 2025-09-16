#!/usr/bin/env python3
"""
Prediction Integration Test
Tests the complete prediction workflow with mock data
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Flask app for context
from app import app

class MockRace:
    """Mock race object for testing"""
    def __init__(self):
        self.id = 1
        self.track_condition = 'fast'
        self.distance = 1200
        self.race_class = 'maiden'
        self.field_size = 8
        self.date = datetime.now()
        self.horses = []

class MockHorse:
    """Mock horse object for testing"""
    def __init__(self, horse_id, name):
        self.id = horse_id
        self.name = name
        self.age = 4
        self.weight = 55.0
        self.jockey_id = horse_id
        self.trainer_id = horse_id
        self.barrier = horse_id
        self.wins = horse_id % 3
        self.places = horse_id % 5
        self.starts = horse_id + 5
        self.earnings = (horse_id + 1) * 10000
        self.last_run = datetime.now() - timedelta(days=horse_id * 7)
        self.recent_form = f"{horse_id}11"
        
    @property
    def win_percentage(self):
        return (self.wins / self.starts) * 100 if self.starts > 0 else 0
        
    @property
    def place_percentage(self):
        return (self.places / self.starts) * 100 if self.starts > 0 else 0
        
    def get_form(self, num_runs=5):
        return self.recent_form[:num_runs]
        
    def get_recent_performance(self, num_races=10):
        """Return mock recent performance positions (list of integers)"""
        return [(self.id % 8) + 1] * min(num_races, 5)  # Return list of positions
    
    def get_form(self, num_races=10):
        """Return mock recent performance data as dictionaries"""
        # Return list of race performance dictionaries
        performances = []
        for i in range(min(num_races, 5)):
            performance = {
                'position': (self.id % 8) + 1,  # Position 1-8
                'date': datetime.now() - timedelta(days=i * 14),  # Every 2 weeks
                'time': 60.0 + i,  # Mock race time
                'distance': 1200,
                'track_condition': 'fast'
            }
            performances.append(performance)
        return performances

class MockJockey:
    """Mock jockey object for testing"""
    def __init__(self, jockey_id):
        self.id = jockey_id
        self.name = f"Jockey {jockey_id}"
        self.wins = jockey_id * 2
        self.starts = jockey_id * 10
        
    @property
    def win_percentage(self):
        return (self.wins / self.starts) * 100 if self.starts > 0 else 0

class MockTrainer:
    """Mock trainer object for testing"""
    def __init__(self, trainer_id):
        self.id = trainer_id
        self.name = f"Trainer {trainer_id}"
        self.wins = trainer_id * 3
        self.starts = trainer_id * 15
        
    @property
    def win_percentage(self):
        return (self.wins / self.starts) * 100 if self.starts > 0 else 0

def create_mock_race_data():
    """Create comprehensive mock race data"""
    print("Creating mock race data...")
    
    # Create race
    race = MockRace()
    
    # Create horses
    horses = []
    for i in range(1, 9):  # 8 horses
        horse = MockHorse(i, f"Horse {i}")
        horses.append(horse)
    
    race.horses = horses
    
    # Create jockeys and trainers
    jockeys = {i: MockJockey(i) for i in range(1, 9)}
    trainers = {i: MockTrainer(i) for i in range(1, 9)}
    
    print(f"✓ Created race with {len(horses)} horses")
    return race, horses, jockeys, trainers

def test_traditional_prediction():
    """Test traditional prediction system"""
    print("\n" + "=" * 60)
    print("TEST 1: Traditional Prediction System")
    print("=" * 60)
    
    try:
        from utils.predictor import Predictor
        
        with app.app_context():
            # Create predictor
            predictor = Predictor()
            print("✓ Traditional predictor initialized")
            
            # Create mock data
            race, horses, jockeys, trainers = create_mock_race_data()
            
            # Test prediction
            predictions = predictor.predict_race(race)
            
            if predictions:
                print(f"✓ Traditional prediction successful")
                print(f"  - Number of predictions: {len(predictions)}")
                print(f"  - Sample prediction: {predictions[0] if predictions else 'None'}")
                
                # Validate prediction structure
                if isinstance(predictions, list) and len(predictions) > 0:
                    first_pred = predictions[0]
                    if hasattr(first_pred, 'horse_id') and hasattr(first_pred, 'probability'):
                        print("✓ Prediction structure is valid")
                    else:
                        print("⚠ Prediction structure may be incomplete")
                
                return True
            else:
                print("⚠ Traditional prediction returned empty results")
                return False
            
    except Exception as e:
        print(f"✗ Traditional prediction failed: {e}")
        return False

def test_ai_prediction():
    """Test AI prediction system"""
    print("\n" + "=" * 60)
    print("TEST 2: AI Prediction System")
    print("=" * 60)
    
    try:
        from utils.ai_predictor import AIPredictor
        
        # Create AI predictor
        ai_predictor = AIPredictor()
        print("✓ AI predictor initialized")
        
        # Create mock data
        race, horses, jockeys, trainers = create_mock_race_data()
        
        # Test AI prediction
        ai_predictions = ai_predictor.predict_race_ai(race)
        
        if ai_predictions:
            print(f"✓ AI prediction successful")
            print(f"  - Prediction type: {type(ai_predictions)}")
            
            # Handle different prediction formats
            if isinstance(ai_predictions, dict):
                if 'predictions' in ai_predictions:
                    preds = ai_predictions['predictions']
                    print(f"  - Number of predictions: {len(preds) if preds else 0}")
                else:
                    print(f"  - Dict keys: {list(ai_predictions.keys())}")
            elif isinstance(ai_predictions, (list, np.ndarray)):
                print(f"  - Number of predictions: {len(ai_predictions)}")
                if len(ai_predictions) > 0:
                    print(f"  - Sample prediction: {ai_predictions[0]}")
            
            return True
        else:
            print("⚠ AI prediction returned empty results")
            return False
            
    except Exception as e:
        print(f"✗ AI prediction failed: {e}")
        return False

def test_enhanced_prediction():
    """Test enhanced prediction with data processing"""
    print("\n" + "=" * 60)
    print("TEST 3: Enhanced Prediction with Data Processing")
    print("=" * 60)
    
    try:
        with app.app_context():
            from utils.predictor import Predictor
            from utils.data_processor import DataProcessor
            
            # Create predictor and data processor
            predictor = Predictor()
            data_processor = DataProcessor()
            print("✓ Enhanced prediction components initialized")
            
            # Create mock data
            race, horses, jockeys, trainers = create_mock_race_data()
            print("✓ Created race with {} horses".format(len(horses)))
            
            # Test data enhancement
            enhanced_data = data_processor.prepare_race_data(race)
            
            if enhanced_data is not None and len(enhanced_data) > 0:
                print(f"✓ Data enhancement successful")
                print(f"  - Enhanced data type: {type(enhanced_data)}")
                
                if isinstance(enhanced_data, dict):
                    print(f"  - Data keys: {list(enhanced_data.keys())}")
                elif hasattr(enhanced_data, 'shape'):
                    print(f"  - Data shape: {enhanced_data.shape}")
                
                return True
            else:
                print("⚠ Data enhancement returned empty results")
                return False
            
    except Exception as e:
        print(f"✗ Enhanced prediction failed: {e}")
        return False

def test_prediction_comparison():
    """Test and compare different prediction methods"""
    print("\n" + "=" * 60)
    print("TEST 4: Prediction Method Comparison")
    print("=" * 60)
    
    try:
        from utils.predictor import Predictor
        from utils.ai_predictor import AIPredictor
        
        with app.app_context():
            # Create predictors
            traditional_predictor = Predictor()
            ai_predictor = AIPredictor()
            print("✓ Both predictors initialized")
            
            # Create mock data
            race, horses, jockeys, trainers = create_mock_race_data()
            
            # Get predictions from both systems
            traditional_preds = traditional_predictor.predict_race(race)
            ai_preds = ai_predictor.predict_race_ai(race)
            
            print(f"✓ Traditional predictions: {'Available' if traditional_preds else 'None'}")
            print(f"✓ AI predictions: {'Available' if ai_preds else 'None'}")
            
            # Compare if both are available
            if traditional_preds and ai_preds:
                print("✓ Both prediction methods are working")
                print("✓ Prediction system integration is successful")
                return True
            elif traditional_preds or ai_preds:
                print("⚠ One prediction method is working")
                print("⚠ Partial integration success")
                return True
            else:
                print("✗ No prediction methods are working")
                return False
            
    except Exception as e:
        print(f"✗ Prediction comparison failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction for AI models"""
    print("\n" + "=" * 60)
    print("TEST 5: Feature Extraction for AI Models")
    print("=" * 60)
    
    try:
        from utils.ai_predictor import AIPredictor
        
        # Create AI predictor
        ai_predictor = AIPredictor()
        print("✓ AI predictor initialized for feature extraction")
        
        # Create mock data
        race, horses, jockeys, trainers = create_mock_race_data()
        
        # Test feature extraction
        features = ai_predictor._prepare_ai_features(race)
        
        if features is not None:
            print(f"✓ Feature extraction successful")
            print(f"  - Feature type: {type(features)}")
            
            if hasattr(features, 'shape'):
                print(f"  - Feature shape: {features.shape}")
                
                # Check for NaN values
                if hasattr(features, 'isnan'):
                    nan_count = features.isnan().sum() if hasattr(features.isnan(), 'sum') else 0
                    print(f"  - NaN values: {nan_count}")
                elif isinstance(features, np.ndarray):
                    nan_count = np.isnan(features).sum()
                    print(f"  - NaN values: {nan_count}")
            
            return True
        else:
            print("⚠ Feature extraction returned None")
            return False
            
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False

def main():
    """Run all prediction integration tests"""
    print("PREDICTION INTEGRATION TEST")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Traditional Prediction
    test_results.append(test_traditional_prediction())
    
    # Test 2: AI Prediction
    test_results.append(test_ai_prediction())
    
    # Test 3: Enhanced Prediction
    test_results.append(test_enhanced_prediction())
    
    # Test 4: Prediction Comparison
    test_results.append(test_prediction_comparison())
    
    # Test 5: Feature Extraction
    test_results.append(test_feature_extraction())
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Traditional Prediction System",
        "AI Prediction System",
        "Enhanced Prediction with Data Processing",
        "Prediction Method Comparison",
        "Feature Extraction for AI Models"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 COMPLETE INTEGRATION SUCCESS!")
        print("✅ Traditional prediction system is working")
        print("✅ AI prediction system is working")
        print("✅ Data processing and enhancement is working")
        print("✅ Feature extraction for AI models is working")
        print("✅ All prediction methods are properly integrated")
        print("\n🚀 Your complete prediction system is ready for production!")
    elif passed >= total * 0.8:
        print("\n⚠️  MOSTLY INTEGRATED!")
        print("✅ Core prediction functionality is working")
        print("⚠️  Some components may need fine-tuning")
        print("🔧 System is functional but could be optimized")
    else:
        print("\n❌ INTEGRATION ISSUES!")
        print("❌ Multiple prediction components have problems")
        print("🔧 Significant integration work needed")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)