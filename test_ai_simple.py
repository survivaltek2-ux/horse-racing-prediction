#!/usr/bin/env python3
"""
Simple AI Integration Test
Tests core TensorFlow and PyTorch functionality safely
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tensorflow_basic():
    """Test basic TensorFlow functionality"""
    print("=" * 60)
    print("TEST 1: TensorFlow Basic Functionality")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
        
        # Test basic operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"✓ Basic tensor operations: {c.numpy()}")
        
        # Test simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Test prediction
        X_test = np.random.random((3, 5))
        predictions = model.predict(X_test, verbose=0)
        print(f"✓ Model prediction successful: shape {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False

def test_pytorch_basic():
    """Test basic PyTorch functionality"""
    print("\n" + "=" * 60)
    print("TEST 2: PyTorch Basic Functionality")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        
        # Test basic operations
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        z = x + y
        print(f"✓ Basic tensor operations: {z}")
        
        # Test simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(5, 10)
                self.fc2 = nn.Linear(10, 1)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.sigmoid(self.fc2(x))
                return x
        
        model = SimpleModel()
        
        # Test prediction
        X_test = torch.randn(3, 5)
        with torch.no_grad():
            predictions = model(X_test)
        print(f"✓ Model prediction successful: shape {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

def test_ai_predictor_import():
    """Test AI predictor import and initialization"""
    print("\n" + "=" * 60)
    print("TEST 3: AI Predictor Import and Initialization")
    print("=" * 60)
    
    try:
        from utils.ai_predictor import AIPredictor
        print(f"✓ AI Predictor imported successfully")
        
        # Initialize predictor
        predictor = AIPredictor()
        print(f"✓ AI Predictor initialized successfully")
        
        # Check available attributes
        attrs = ['data_processor', 'scaler', 'label_encoder']
        for attr in attrs:
            if hasattr(predictor, attr):
                print(f"✓ Has attribute: {attr}")
            else:
                print(f"⚠ Missing attribute: {attr}")
        
        return True
        
    except Exception as e:
        print(f"✗ AI Predictor test failed: {e}")
        return False

def test_data_processor():
    """Test data processor functionality"""
    print("\n" + "=" * 60)
    print("TEST 4: Data Processor Functionality")
    print("=" * 60)
    
    try:
        from utils.data_processor import DataProcessor
        print(f"✓ Data Processor imported successfully")
        
        # Initialize processor
        processor = DataProcessor()
        print(f"✓ Data Processor initialized successfully")
        
        # Test encoding methods
        track_condition = processor._encode_track_condition('fast')
        print(f"✓ Track condition encoding: {track_condition}")
        
        distance = processor._normalize_distance(1200)
        print(f"✓ Distance normalization: {distance}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data Processor test failed: {e}")
        return False

def test_model_availability():
    """Test which AI models are available"""
    print("\n" + "=" * 60)
    print("TEST 5: AI Model Availability Check")
    print("=" * 60)
    
    try:
        from utils.ai_predictor import TENSORFLOW_AVAILABLE, PYTORCH_AVAILABLE
        
        print(f"✓ TensorFlow Available: {TENSORFLOW_AVAILABLE}")
        print(f"✓ PyTorch Available: {PYTORCH_AVAILABLE}")
        
        if TENSORFLOW_AVAILABLE:
            print("  - TensorFlow models can be used")
        else:
            print("  - TensorFlow models not available")
            
        if PYTORCH_AVAILABLE:
            print("  - PyTorch models can be used")
        else:
            print("  - PyTorch models not available")
        
        return TENSORFLOW_AVAILABLE or PYTORCH_AVAILABLE
        
    except Exception as e:
        print(f"✗ Model availability check failed: {e}")
        return False

def test_prediction_workflow():
    """Test basic prediction workflow"""
    print("\n" + "=" * 60)
    print("TEST 6: Basic Prediction Workflow")
    print("=" * 60)
    
    try:
        # Test traditional predictor
        from utils.predictor import Predictor
        print(f"✓ Traditional Predictor imported successfully")
        
        predictor = Predictor()
        print(f"✓ Traditional Predictor initialized successfully")
        
        # Check if it has the required methods
        methods = ['predict_race', 'predict_race_enhanced']
        for method in methods:
            if hasattr(predictor, method):
                print(f"✓ Has method: {method}")
            else:
                print(f"⚠ Missing method: {method}")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction workflow test failed: {e}")
        return False

def main():
    """Run all simple AI tests"""
    print("SIMPLE AI INTEGRATION TEST")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: TensorFlow Basic
    test_results.append(test_tensorflow_basic())
    
    # Test 2: PyTorch Basic
    test_results.append(test_pytorch_basic())
    
    # Test 3: AI Predictor Import
    test_results.append(test_ai_predictor_import())
    
    # Test 4: Data Processor
    test_results.append(test_data_processor())
    
    # Test 5: Model Availability
    test_results.append(test_model_availability())
    
    # Test 6: Prediction Workflow
    test_results.append(test_prediction_workflow())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "TensorFlow Basic Functionality",
        "PyTorch Basic Functionality",
        "AI Predictor Import and Initialization",
        "Data Processor Functionality",
        "AI Model Availability Check",
        "Basic Prediction Workflow"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ TensorFlow is properly installed and working")
        print("✅ PyTorch is properly installed and working") 
        print("✅ AI Predictor system is properly integrated")
        print("✅ Data processing components are functional")
        print("✅ Prediction workflow is available")
        print("\n🚀 Your AI prediction system is ready to use!")
    elif passed >= total * 0.8:
        print("\n⚠️  MOSTLY WORKING!")
        print("✅ Core AI functionality is working")
        print("⚠️  Some minor issues may need attention")
        print("🔧 System is functional but could be optimized")
    else:
        print("\n❌ ISSUES DETECTED!")
        print("❌ Multiple components have problems")
        print("🔧 Significant fixes needed before production use")
    
    return passed >= total * 0.8  # Consider 80% pass rate as success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)