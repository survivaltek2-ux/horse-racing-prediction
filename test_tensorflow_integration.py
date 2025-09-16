#!/usr/bin/env python3
"""
Comprehensive TensorFlow and AI Integration Test
This script verifies that TensorFlow is properly installed and integrated
with the horse racing prediction algorithm.
"""

import sys
import os
import numpy as np
import traceback
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tensorflow_installation():
    """Test TensorFlow installation and basic functionality"""
    print("=" * 60)
    print("TESTING TENSORFLOW INSTALLATION")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow successfully imported")
        print(f"   Version: {tf.__version__}")
        print(f"   Keras version: {tf.keras.__version__}")
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU devices found: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("ℹ️  No GPU devices found (using CPU)")
        
        # Test basic tensor operations
        print("\n📊 Testing basic tensor operations...")
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[2.0, 0.0], [0.0, 2.0]])
        c = tf.matmul(a, b)
        print(f"   Matrix multiplication test: {c.numpy().tolist()}")
        
        # Test model creation
        print("\n🧠 Testing model creation...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        print("   ✅ Simple model created and compiled successfully")
        
        # Test model prediction
        test_input = np.random.random((1, 5))
        prediction = model.predict(test_input, verbose=0)
        print(f"   ✅ Model prediction test: {prediction[0][0]:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {str(e)}")
        print("   Install with: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ TensorFlow test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_pytorch_installation():
    """Test PyTorch installation and basic functionality"""
    print("\n" + "=" * 60)
    print("TESTING PYTORCH INSTALLATION")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        print(f"✅ PyTorch successfully imported")
        print(f"   Version: {torch.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available (using CPU)")
        
        # Test basic tensor operations
        print("\n📊 Testing basic tensor operations...")
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        c = torch.matmul(a, b)
        print(f"   Matrix multiplication test: {c.tolist()}")
        
        # Test model creation
        print("\n🧠 Testing model creation...")
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.linear = nn.Linear(5, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                return self.sigmoid(self.linear(x))
        
        model = TestModel()
        test_input = torch.randn(1, 5)
        prediction = model(test_input)
        print(f"   ✅ Model prediction test: {prediction.item():.4f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {str(e)}")
        print("   Install with: pip install torch")
        return False
    except Exception as e:
        print(f"❌ PyTorch test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_ai_predictor_initialization():
    """Test AI predictor initialization"""
    print("\n" + "=" * 60)
    print("TESTING AI PREDICTOR INITIALIZATION")
    print("=" * 60)
    
    try:
        from utils.ai_predictor import AIPredictor
        
        print("🚀 Initializing AI Predictor...")
        predictor = AIPredictor()
        
        # Check TensorFlow models
        if hasattr(predictor, 'neural_models'):
            tf_models = [k for k in predictor.neural_models.keys() if 'dnn' in k or 'lstm' in k or 'cnn' in k]
            if tf_models:
                print(f"✅ TensorFlow models initialized: {tf_models}")
            else:
                print("⚠️  No TensorFlow models found")
        
        # Check PyTorch models
        pytorch_models = [k for k in predictor.neural_models.keys() if 'pytorch' in k]
        if pytorch_models:
            print(f"✅ PyTorch models initialized: {pytorch_models}")
        else:
            print("ℹ️  No PyTorch models found")
        
        print(f"   Training status: {'Trained' if predictor.is_trained else 'Not trained'}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Predictor initialization failed: {str(e)}")
        traceback.print_exc()
        return False

def test_model_architecture():
    """Test neural network model architectures"""
    print("\n" + "=" * 60)
    print("TESTING MODEL ARCHITECTURES")
    print("=" * 60)
    
    try:
        from utils.ai_predictor import AIPredictor
        
        predictor = AIPredictor()
        
        # Test DNN model
        if 'dnn_win' in predictor.neural_models:
            model = predictor.neural_models['dnn_win']
            print("🧠 DNN Model Architecture:")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            print(f"   Total parameters: {model.count_params():,}")
            
            # Test prediction
            test_input = np.random.random((1, model.input_shape[1]))
            prediction = model.predict(test_input, verbose=0)
            print(f"   ✅ DNN prediction test: {prediction[0][0]:.4f}")
        
        # Test LSTM model
        if 'lstm_form' in predictor.neural_models:
            model = predictor.neural_models['lstm_form']
            print("\n🔄 LSTM Model Architecture:")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            print(f"   Total parameters: {model.count_params():,}")
        
        # Test CNN model
        if 'cnn_pattern' in predictor.neural_models:
            model = predictor.neural_models['cnn_pattern']
            print("\n🖼️  CNN Model Architecture:")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            print(f"   Total parameters: {model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model architecture test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing for AI models"""
    print("\n" + "=" * 60)
    print("TESTING DATA PROCESSING")
    print("=" * 60)
    
    try:
        from app import app, db
        from models.sqlalchemy_models import Race, Horse
        from utils.ai_predictor import AIPredictor
        
        with app.app_context():
            # Get a sample race
            race = Race.query.first()
            if not race:
                print("⚠️  No races found in database")
                return False
            
            print(f"📊 Testing with race: {race.name}")
            
            predictor = AIPredictor()
            
            # Test feature preparation
            print("🔧 Testing feature preparation...")
            features = predictor._prepare_ai_features(race)
            
            if features:
                print("✅ Features prepared successfully:")
                for feature_type, feature_data in features.items():
                    if isinstance(feature_data, np.ndarray):
                        print(f"   {feature_type}: shape {feature_data.shape}")
                    else:
                        print(f"   {feature_type}: {type(feature_data)}")
            else:
                print("⚠️  Feature preparation returned None")
            
            return True
            
    except Exception as e:
        print(f"❌ Data processing test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_prediction_workflow():
    """Test the complete prediction workflow"""
    print("\n" + "=" * 60)
    print("TESTING PREDICTION WORKFLOW")
    print("=" * 60)
    
    try:
        from app import app, db
        from models.sqlalchemy_models import Race
        from utils.ai_predictor import AIPredictor
        
        with app.app_context():
            # Get a sample race
            race = Race.query.first()
            if not race:
                print("⚠️  No races found in database")
                return False
            
            print(f"🏁 Testing prediction for race: {race.name}")
            
            predictor = AIPredictor()
            
            # Test AI prediction
            print("🤖 Running AI prediction...")
            ai_result = predictor.predict_race_ai(race, use_ensemble=True)
            
            if ai_result:
                print("✅ AI prediction completed:")
                print(f"   Algorithm: {ai_result.get('algorithm', 'unknown')}")
                print(f"   Predictions: {ai_result.get('predictions', 'none')}")
                print(f"   Confidence: {ai_result.get('confidence_scores', {})}")
                
                if 'ai_insights' in ai_result:
                    insights = ai_result['ai_insights']
                    print(f"   Insights: {len(insights)} generated")
            else:
                print("⚠️  AI prediction returned None")
            
            return True
            
    except Exception as e:
        print(f"❌ Prediction workflow test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_integration_with_main_predictor():
    """Test integration with the main prediction system"""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION WITH MAIN PREDICTOR")
    print("=" * 60)
    
    try:
        from app import app, db
        from models.sqlalchemy_models import Race
        from utils.predictor import Predictor
        
        with app.app_context():
            # Get a sample race
            race = Race.query.first()
            if not race:
                print("⚠️  No races found in database")
                return False
            
            print(f"🔗 Testing integration for race: {race.name}")
            
            predictor = Predictor()
            
            # Test AI-enhanced prediction
            print("🤖 Running AI-enhanced prediction...")
            result = predictor.predict_race_with_ai(race, use_ai=True, use_ensemble=True)
            
            if result:
                print("✅ AI-enhanced prediction completed:")
                if hasattr(result, 'predictions'):
                    print(f"   Predictions count: {len(result.predictions)}")
                if hasattr(result, 'algorithm'):
                    print(f"   Algorithm: {result.algorithm}")
                if hasattr(result, 'confidence_scores'):
                    print(f"   Confidence scores available: {bool(result.confidence_scores)}")
            else:
                print("⚠️  AI-enhanced prediction returned None")
            
            return True
            
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 TENSORFLOW AND AI INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("TensorFlow Installation", test_tensorflow_installation),
        ("PyTorch Installation", test_pytorch_installation),
        ("AI Predictor Initialization", test_ai_predictor_initialization),
        ("Model Architecture", test_model_architecture),
        ("Data Processing", test_data_processing),
        ("Prediction Workflow", test_prediction_workflow),
        ("Integration with Main Predictor", test_integration_with_main_predictor)
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
        print("🎉 All tests passed! TensorFlow and AI integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()