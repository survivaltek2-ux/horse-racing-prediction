#!/usr/bin/env python3
"""
Simplified TensorFlow and AI Integration Test
This script tests TensorFlow functionality without relying on database queries
"""

import sys
import os
import numpy as np
import traceback
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tensorflow_basic():
    """Test basic TensorFlow functionality"""
    print("=" * 60)
    print("TESTING TENSORFLOW BASIC FUNCTIONALITY")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test model creation and training
        print("\n🧠 Creating and testing a simple neural network...")
        
        # Create sample data
        X = np.random.random((1000, 10))
        y = np.random.randint(0, 2, (1000, 1))
        
        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   Model created with {model.count_params():,} parameters")
        
        # Test training
        print("   Training model...")
        history = model.fit(X, y, epochs=5, batch_size=32, verbose=0, validation_split=0.2)
        
        # Test prediction
        test_input = np.random.random((5, 10))
        predictions = model.predict(test_input, verbose=0)
        
        print(f"   ✅ Training completed. Final accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"   ✅ Predictions generated: {predictions.flatten()}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_ai_predictor_models():
    """Test AI predictor model creation without database"""
    print("\n" + "=" * 60)
    print("TESTING AI PREDICTOR MODEL CREATION")
    print("=" * 60)
    
    try:
        from utils.ai_predictor import AIPredictor
        
        print("🚀 Creating AI Predictor...")
        predictor = AIPredictor()
        
        # Test model initialization
        print("🧠 Testing model architectures...")
        
        if hasattr(predictor, 'neural_models'):
            for model_name, model in predictor.neural_models.items():
                if model is not None:
                    print(f"   ✅ {model_name}: {model.count_params():,} parameters")
                    
                    # Test prediction with dummy data
                    if hasattr(model, 'input_shape') and model.input_shape:
                        input_shape = model.input_shape[1:]
                        dummy_input = np.random.random((1,) + input_shape)
                        prediction = model.predict(dummy_input, verbose=0)
                        print(f"      Prediction test: {prediction[0][0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Predictor test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_ai_feature_extraction():
    """Test AI feature extraction with mock data"""
    print("\n" + "=" * 60)
    print("TESTING AI FEATURE EXTRACTION")
    print("=" * 60)
    
    try:
        from utils.ai_predictor import AIPredictor
        import pandas as pd
        
        predictor = AIPredictor()
        
        # Create mock race data
        print("📊 Creating mock race data...")
        mock_race_data = pd.DataFrame({
            'horse_id': [1, 2, 3, 4, 5],
            'age': [4, 5, 3, 6, 4],
            'weight': [120, 118, 122, 115, 119],
            'win_rate': [0.25, 0.15, 0.35, 0.10, 0.20],
            'recent_speed_avg': [85, 82, 88, 80, 84],
            'best_speed': [92, 88, 95, 85, 90],
            'speed_consistency': [0.8, 0.7, 0.9, 0.6, 0.75],
            'recent_form_score': [7.5, 6.2, 8.1, 5.8, 7.0],
            'wins_last_5': [2, 1, 3, 0, 1],
            'places_last_5': [4, 3, 4, 2, 3],
            'days_since_last_run': [14, 21, 7, 35, 18]
        })
        
        print("🔧 Testing feature extraction methods...")
        
        # Test speed features
        speed_features = predictor._extract_speed_features(mock_race_data)
        print(f"   ✅ Speed features: shape {speed_features.shape}")
        
        # Test form features
        form_features = predictor._extract_form_features(mock_race_data)
        print(f"   ✅ Form features: shape {form_features.shape}")
        
        # Test sequence creation
        sequences = predictor._create_form_sequences(mock_race_data)
        print(f"   ✅ Form sequences: shape {sequences.shape}")
        
        # Test pattern matrices
        patterns = predictor._create_pattern_matrices(mock_race_data)
        print(f"   ✅ Pattern matrices: shape {patterns.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_ensemble_prediction():
    """Test ensemble prediction with mock data"""
    print("\n" + "=" * 60)
    print("TESTING ENSEMBLE PREDICTION")
    print("=" * 60)
    
    try:
        from utils.ai_predictor import AIPredictor
        
        predictor = AIPredictor()
        
        # Create mock predictions from different models
        mock_predictions = {
            'dnn_win': np.array([[0.8], [0.6], [0.9], [0.3], [0.7]]),
            'lstm_form': np.array([[0.75], [0.65], [0.85], [0.35], [0.72]]),
            'cnn_pattern': np.array([[0.82], [0.58], [0.88], [0.32], [0.68]])
        }
        
        # Create mock features
        mock_features = {
            'speed_features': np.random.random((5, 5)),
            'form_features': np.random.random((5, 6)),
            'form_sequences': np.random.random((5, 10, 6)),
            'pattern_matrices': np.random.random((5, 20, 20, 1))
        }
        
        print("🤖 Testing ensemble prediction...")
        ensemble_result = predictor._ensemble_ai_predictions(mock_predictions, mock_features)
        
        if ensemble_result:
            print(f"   ✅ Ensemble prediction completed")
            print(f"   Result type: {type(ensemble_result)}")
            if hasattr(ensemble_result, 'shape'):
                print(f"   Result shape: {ensemble_result.shape}")
        
        # Test confidence calculation
        print("📊 Testing confidence calculation...")
        confidence = predictor._calculate_ai_confidence(mock_predictions, mock_features)
        
        if confidence:
            print(f"   ✅ Confidence scores calculated: {type(confidence)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble prediction test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_pytorch_integration():
    """Test PyTorch integration"""
    print("\n" + "=" * 60)
    print("TESTING PYTORCH INTEGRATION")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Test model creation
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(10, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = TestModel()
        print(f"   ✅ PyTorch model created")
        
        # Test training
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Generate sample data
        X = torch.randn(1000, 10)
        y = torch.randint(0, 2, (1000, 1)).float()
        
        # Training loop
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        print(f"   ✅ Training completed. Final loss: {loss.item():.4f}")
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(5, 10)
            predictions = model(test_input)
            print(f"   ✅ Predictions: {predictions.flatten().tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run simplified tests"""
    print("🧪 SIMPLIFIED TENSORFLOW AND AI INTEGRATION TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("TensorFlow Basic Functionality", test_tensorflow_basic),
        ("AI Predictor Model Creation", test_ai_predictor_models),
        ("AI Feature Extraction", test_ai_feature_extraction),
        ("Ensemble Prediction", test_ensemble_prediction),
        ("PyTorch Integration", test_pytorch_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TensorFlow and AI core functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()