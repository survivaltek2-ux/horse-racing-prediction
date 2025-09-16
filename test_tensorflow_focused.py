#!/usr/bin/env python3
"""
Focused TensorFlow and AI Integration Test
Tests core AI functionality with mock data, bypassing database issues
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tensorflow_installation():
    """Test TensorFlow installation and basic operations"""
    print("=" * 60)
    print("TEST 1: TensorFlow Installation and Basic Operations")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow imported successfully")
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Test basic tensor operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"✓ Basic tensor operations work: {c.numpy()}")
        
        # Test model creation
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        print(f"✓ Model creation and compilation successful")
        
        return True
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False

def test_pytorch_installation():
    """Test PyTorch installation and basic operations"""
    print("\n" + "=" * 60)
    print("TEST 2: PyTorch Installation and Basic Operations")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        print(f"✓ PyTorch imported successfully")
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Test basic tensor operations
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        z = x + y
        print(f"✓ Basic tensor operations work: {z}")
        
        # Test model creation
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        print(f"✓ Model creation successful")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

def create_mock_race_data():
    """Create mock race data for testing"""
    print("\n" + "=" * 60)
    print("TEST 3: Mock Data Creation")
    print("=" * 60)
    
    try:
        # Create mock race object
        class MockRace:
            def __init__(self):
                self.id = 1
                self.name = "Test Race"
                self.date = datetime.now()
                self.track = "Test Track"
                self.distance = 1200.0
                self.track_condition = "fast"
                self.surface = "dirt"
                self.weather = "clear"
                
        # Create mock horse objects
        class MockHorse:
            def __init__(self, horse_id, name):
                self.id = horse_id
                self.name = name
                self.age = np.random.randint(3, 8)
                self.weight = np.random.uniform(450, 550)
                self.wins = np.random.randint(0, 10)
                self.runs = np.random.randint(5, 30)
                self.places = np.random.randint(0, 15)
                self.rating = np.random.randint(60, 100)
                self.jockey = f"Jockey_{horse_id}"
                self.trainer = f"Trainer_{horse_id}"
                self.form = "1-2-3-1-2"  # Mock form string
                self.earnings = np.random.uniform(10000, 100000)
                self.last_run = datetime.now() - timedelta(days=np.random.randint(7, 60))
                
            @property
            def win_percentage(self):
                return self.wins / max(self.runs, 1) if self.runs > 0 else 0
                
            @property
            def place_percentage(self):
                return self.places / max(self.runs, 1) if self.runs > 0 else 0
                
            def get_recent_performance(self, limit=5):
                # Return mock recent positions
                return [np.random.randint(1, 8) for _ in range(min(limit, 5))]
                
            def get_form(self, num_races=3):
                # Return mock form data as list of dictionaries
                form_data = []
                for i in range(num_races):
                    form_data.append({
                        'position': np.random.randint(1, 8),
                        'date': datetime.now() - timedelta(days=i*14 + 7),
                        'track': f"Track_{i}",
                        'distance': np.random.uniform(1000, 2000)
                    })
                return form_data
        
        race = MockRace()
        horses = [MockHorse(i, f"Horse_{i}") for i in range(1, 9)]
        
        print(f"✓ Created mock race: {race.name}")
        print(f"✓ Created {len(horses)} mock horses")
        
        return race, horses
    except Exception as e:
        print(f"✗ Mock data creation failed: {e}")
        return None, None

def test_ai_predictor_initialization():
    """Test AI predictor initialization with mock data"""
    print("\n" + "=" * 60)
    print("TEST 4: AI Predictor Initialization")
    print("=" * 60)
    
    try:
        from utils.ai_predictor import AIPredictor
        
        # Initialize AI predictor
        ai_predictor = AIPredictor()
        print(f"✓ AI Predictor initialized successfully")
        
        # Check if TensorFlow models are available
        if hasattr(ai_predictor, 'tf_dnn_model'):
            print(f"✓ TensorFlow DNN model available")
        if hasattr(ai_predictor, 'tf_lstm_model'):
            print(f"✓ TensorFlow LSTM model available")
        if hasattr(ai_predictor, 'tf_cnn_model'):
            print(f"✓ TensorFlow CNN model available")
            
        # Check if PyTorch models are available
        if hasattr(ai_predictor, 'pytorch_dnn_model'):
            print(f"✓ PyTorch DNN model available")
        if hasattr(ai_predictor, 'pytorch_rnn_model'):
            print(f"✓ PyTorch RNN model available")
            
        return ai_predictor
    except Exception as e:
        print(f"✗ AI Predictor initialization failed: {e}")
        return None

def test_feature_extraction(ai_predictor, race, horses):
    """Test AI feature extraction with mock data"""
    print("\n" + "=" * 60)
    print("TEST 5: AI Feature Extraction")
    print("=" * 60)
    
    try:
        # Add horses to race for data processor
        race.horses = horses
        
        # Test feature preparation
        features = ai_predictor._prepare_ai_features(race)
        print(f"✓ Feature extraction successful")
        
        if features is not None:
            print(f"✓ Feature shape: {features.shape}")
            print(f"✓ Feature columns: {features.shape[1]}")
            
            # Check for NaN values
            nan_count = features.isnull().sum().sum()
            if nan_count == 0:
                print(f"✓ No NaN values in features")
            else:
                print(f"⚠ Warning: {nan_count} NaN values found in features")
        else:
            print(f"⚠ Warning: Feature extraction returned None (likely due to data processor issues)")
            
        return features
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return None

def test_model_predictions(ai_predictor, features):
    """Test individual model predictions"""
    print("\n" + "=" * 60)
    print("TEST 6: Individual Model Predictions")
    print("=" * 60)
    
    results = {}
    
    # Test TensorFlow models
    try:
        if hasattr(ai_predictor, 'tf_dnn_model') and ai_predictor.tf_dnn_model:
            pred = ai_predictor._predict_tensorflow_dnn(features)
            results['tf_dnn'] = pred
            print(f"✓ TensorFlow DNN prediction successful: shape {pred.shape}")
    except Exception as e:
        print(f"✗ TensorFlow DNN prediction failed: {e}")
    
    try:
        if hasattr(ai_predictor, 'tf_lstm_model') and ai_predictor.tf_lstm_model:
            pred = ai_predictor._predict_tensorflow_lstm(features)
            results['tf_lstm'] = pred
            print(f"✓ TensorFlow LSTM prediction successful: shape {pred.shape}")
    except Exception as e:
        print(f"✗ TensorFlow LSTM prediction failed: {e}")
    
    try:
        if hasattr(ai_predictor, 'tf_cnn_model') and ai_predictor.tf_cnn_model:
            pred = ai_predictor._predict_tensorflow_cnn(features)
            results['tf_cnn'] = pred
            print(f"✓ TensorFlow CNN prediction successful: shape {pred.shape}")
    except Exception as e:
        print(f"✗ TensorFlow CNN prediction failed: {e}")
    
    # Test PyTorch models
    try:
        if hasattr(ai_predictor, 'pytorch_dnn_model') and ai_predictor.pytorch_dnn_model:
            pred = ai_predictor._predict_pytorch_dnn(features)
            results['pytorch_dnn'] = pred
            print(f"✓ PyTorch DNN prediction successful: shape {pred.shape}")
    except Exception as e:
        print(f"✗ PyTorch DNN prediction failed: {e}")
    
    try:
        if hasattr(ai_predictor, 'pytorch_rnn_model') and ai_predictor.pytorch_rnn_model:
            pred = ai_predictor._predict_pytorch_rnn(features)
            results['pytorch_rnn'] = pred
            print(f"✓ PyTorch RNN prediction successful: shape {pred.shape}")
    except Exception as e:
        print(f"✗ PyTorch RNN prediction failed: {e}")
    
    return results

def test_ensemble_prediction(ai_predictor, race, horses):
    """Test ensemble prediction with mock data"""
    print("\n" + "=" * 60)
    print("TEST 7: Ensemble Prediction")
    print("=" * 60)
    
    try:
        # Add horses to race for data processor
        race.horses = horses
        
        # Test full prediction pipeline
        predictions = ai_predictor.predict_race_ai(race)
        print(f"✓ Ensemble prediction successful")
        
        if isinstance(predictions, dict):
            print(f"✓ Prediction structure: {list(predictions.keys())}")
            if 'predictions' in predictions:
                pred_data = predictions['predictions']
                if isinstance(pred_data, (list, dict)):
                    print(f"✓ Predictions available: {type(pred_data)}")
                else:
                    print(f"✓ Prediction value: {pred_data}")
        else:
            print(f"✓ Prediction type: {type(predictions)}")
        
        return predictions
    except Exception as e:
        print(f"✗ Ensemble prediction failed: {e}")
        return None

def main():
    """Run all focused TensorFlow and AI tests"""
    print("FOCUSED TENSORFLOW AND AI INTEGRATION TEST")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: TensorFlow Installation
    test_results.append(test_tensorflow_installation())
    
    # Test 2: PyTorch Installation
    test_results.append(test_pytorch_installation())
    
    # Test 3: Mock Data Creation
    race, horses = create_mock_race_data()
    test_results.append(race is not None and horses is not None)
    
    if race and horses:
        # Test 4: AI Predictor Initialization
        ai_predictor = test_ai_predictor_initialization()
        test_results.append(ai_predictor is not None)
        
        if ai_predictor:
            # Test 5: Feature Extraction
            features = test_feature_extraction(ai_predictor, race, horses)
            test_results.append(features is not None)
            
            if features is not None:
                # Test 6: Individual Model Predictions
                model_results = test_model_predictions(ai_predictor, features)
                test_results.append(len(model_results) > 0)
                
                # Test 7: Ensemble Prediction
                predictions = test_ensemble_prediction(ai_predictor, race, horses)
                test_results.append(predictions is not None)
            else:
                test_results.extend([False, False])
        else:
            test_results.extend([False, False, False])
    else:
        test_results.extend([False, False, False, False])
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "TensorFlow Installation",
        "PyTorch Installation", 
        "Mock Data Creation",
        "AI Predictor Initialization",
        "Feature Extraction",
        "Individual Model Predictions",
        "Ensemble Prediction"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TensorFlow and AI integration is working properly.")
    elif passed >= total * 0.7:
        print("⚠️  Most tests passed. Some minor issues may need attention.")
    else:
        print("❌ Multiple test failures. Significant issues need to be resolved.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)