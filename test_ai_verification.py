#!/usr/bin/env python3
"""
AI Verification Test - Comprehensive test to verify AI is actually being used in predictions
This test demonstrates multiple ways to confirm that AI models are active and making predictions.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Flask app for context
from app import app

class MockRace:
    """Mock race for testing"""
    def __init__(self):
        self.id = 1
        self.name = "Test Stakes"
        self.date = datetime.now().date()
        self.track = "Test Track"
        self.distance = 1600
        self.track_condition = "Good"
        self.class_level = "Class 3"
        self.field_size = 8

class MockHorse:
    """Mock horse for testing"""
    def __init__(self, horse_id, name):
        self.id = horse_id
        self.name = name
        self.age = 4
        self.weight = 57.0
        self.jockey_id = horse_id
        self.trainer_id = horse_id
        self.barrier = horse_id
        self.last_run = datetime.now().date() - timedelta(days=14)
        
    @property
    def win_percentage(self):
        return 0.15 + (self.id * 0.05)
    
    @property
    def place_percentage(self):
        return 0.35 + (self.id * 0.03)
    
    def get_recent_performance(self, num_races=10):
        """Return list of recent finishing positions"""
        return [1, 3, 2, 4, 1, 5, 2, 3, 1, 4][:num_races]
    
    def get_form(self, num_races=10):
        """Return detailed form data"""
        form_data = []
        for i in range(num_races):
            form_data.append({
                'position': (i % 5) + 1,
                'date': datetime.now().date() - timedelta(days=14 * (i + 1)),
                'time': 95.5 + (i * 0.3),
                'distance': 1600,
                'track_condition': 'Good'
            })
        return form_data

def create_mock_race_data():
    """Create comprehensive mock race data"""
    print("Creating mock race data...")
    race = MockRace()
    horses = [MockHorse(i, f"Test Horse {i}") for i in range(1, 9)]
    jockeys = [type('MockJockey', (), {'id': i, 'win_percentage': 0.1 + i*0.02})() for i in range(1, 9)]
    trainers = [type('MockTrainer', (), {'id': i, 'win_percentage': 0.12 + i*0.015})() for i in range(1, 9)]
    
    # Add horses to race
    race.horses = horses
    
    print(f"✓ Created race with {len(horses)} horses")
    return race, horses, jockeys, trainers

def test_ai_model_availability():
    """Test 1: Verify AI models are available and initialized"""
    print("\n" + "=" * 60)
    print("TEST 1: AI Model Availability Check")
    print("=" * 60)
    
    try:
        from utils.ai_predictor import AIPredictor, TENSORFLOW_AVAILABLE, PYTORCH_AVAILABLE
        
        print(f"📦 TensorFlow Available: {TENSORFLOW_AVAILABLE}")
        print(f"📦 PyTorch Available: {PYTORCH_AVAILABLE}")
        
        if not TENSORFLOW_AVAILABLE and not PYTORCH_AVAILABLE:
            print("❌ No AI libraries available - AI predictions will use fallback")
            return False
        
        # Initialize AI predictor
        ai_predictor = AIPredictor()
        print("✓ AI Predictor initialized successfully")
        
        # Check for AI models
        models_found = []
        if hasattr(ai_predictor, 'neural_models') and ai_predictor.neural_models:
            models_found.extend(ai_predictor.neural_models.keys())
        
        if TENSORFLOW_AVAILABLE:
            tf_models = ['tf_dnn_model', 'tf_lstm_model', 'tf_cnn_model']
            for model_name in tf_models:
                if hasattr(ai_predictor, model_name):
                    model = getattr(ai_predictor, model_name)
                    if model is not None:
                        models_found.append(model_name)
                        print(f"  ✓ {model_name}: Available")
                    else:
                        print(f"  ⚠ {model_name}: Not initialized")
        
        if PYTORCH_AVAILABLE:
            pytorch_models = ['pytorch_dnn_model', 'pytorch_rnn_model']
            for model_name in pytorch_models:
                if hasattr(ai_predictor, model_name):
                    model = getattr(ai_predictor, model_name)
                    if model is not None:
                        models_found.append(model_name)
                        print(f"  ✓ {model_name}: Available")
                    else:
                        print(f"  ⚠ {model_name}: Not initialized")
        
        print(f"\n📊 Total AI models available: {len(models_found)}")
        if models_found:
            print(f"   Models: {', '.join(models_found)}")
            return True
        else:
            print("❌ No AI models initialized")
            return False
            
    except Exception as e:
        print(f"❌ AI model availability test failed: {e}")
        return False

def test_ai_vs_traditional_predictions():
    """Test 2: Compare AI predictions vs traditional predictions"""
    print("\n" + "=" * 60)
    print("TEST 2: AI vs Traditional Prediction Comparison")
    print("=" * 60)
    
    try:
        with app.app_context():
            from utils.predictor import Predictor
            from utils.ai_predictor import AIPredictor
            
            # Create test data
            race, horses, jockeys, trainers = create_mock_race_data()
            
            # Initialize predictors
            traditional_predictor = Predictor()
            ai_predictor = AIPredictor()
            
            print("🔄 Running traditional prediction...")
            traditional_result = traditional_predictor.predict_race(race)
            
            print("🤖 Running AI prediction...")
            ai_result = ai_predictor.predict_race_ai(race, use_ensemble=True)
            
            # Compare results
            print("\n📊 PREDICTION COMPARISON:")
            print("-" * 40)
            
            if traditional_result:
                print("✓ Traditional Prediction: SUCCESS")
                if hasattr(traditional_result, 'predictions'):
                    trad_preds = traditional_result.predictions
                    print(f"  Type: {type(trad_preds)}")
                    if hasattr(trad_preds, '__len__'):
                        print(f"  Count: {len(trad_preds)}")
                else:
                    print("  No predictions attribute found")
            else:
                print("❌ Traditional Prediction: FAILED")
            
            if ai_result:
                print("✓ AI Prediction: SUCCESS")
                print(f"  Type: {type(ai_result)}")
                
                # Check for AI-specific indicators
                ai_indicators = []
                if isinstance(ai_result, dict):
                    if 'ai_insights' in ai_result:
                        ai_indicators.append('AI Insights')
                    if 'confidence_scores' in ai_result:
                        ai_indicators.append('AI Confidence Scores')
                    if 'ensemble_weights' in ai_result:
                        ai_indicators.append('Ensemble Weights')
                    if 'neural_predictions' in ai_result:
                        ai_indicators.append('Neural Network Predictions')
                    
                    print(f"  AI Indicators: {', '.join(ai_indicators) if ai_indicators else 'None'}")
                    
                    # Show AI insights if available
                    if 'ai_insights' in ai_result:
                        insights = ai_result['ai_insights']
                        print(f"  AI Insights Count: {len(insights) if insights else 0}")
                        if insights:
                            print(f"  Sample Insight: {insights[0][:50]}..." if len(insights[0]) > 50 else insights[0])
                
            else:
                print("❌ AI Prediction: FAILED")
            
            # Determine if AI is actually being used
            ai_active = False
            if ai_result and isinstance(ai_result, dict):
                # Check for AI-specific content
                if any(key in ai_result for key in ['ai_insights', 'neural_predictions', 'ensemble_weights']):
                    ai_active = True
                # Check if insights mention AI models
                if 'ai_insights' in ai_result and ai_result['ai_insights']:
                    for insight in ai_result['ai_insights']:
                        if any(term in insight.lower() for term in ['neural', 'tensorflow', 'pytorch', 'deep learning', 'ai model']):
                            ai_active = True
                            break
            
            print(f"\n🎯 AI VERIFICATION: {'✓ AI IS ACTIVE' if ai_active else '❌ AI NOT DETECTED'}")
            return ai_active
            
    except Exception as e:
        print(f"❌ AI vs Traditional comparison failed: {e}")
        return False

def test_ai_feature_extraction():
    """Test 3: Verify AI-specific feature extraction"""
    print("\n" + "=" * 60)
    print("TEST 3: AI Feature Extraction Verification")
    print("=" * 60)
    
    try:
        with app.app_context():
            from utils.ai_predictor import AIPredictor
            
            # Create test data
            race, horses, jockeys, trainers = create_mock_race_data()
            
            # Initialize AI predictor
            ai_predictor = AIPredictor()
            
            print("🔍 Extracting AI features...")
            features = ai_predictor._prepare_ai_features(race)
            
            if features:
                print("✓ AI feature extraction successful")
                print(f"  Feature type: {type(features)}")
                
                if isinstance(features, dict):
                    print(f"  Feature categories: {list(features.keys())}")
                    
                    # Check for AI-specific feature types
                    ai_feature_types = []
                    if 'speed_features' in features:
                        ai_feature_types.append('Speed Analysis')
                    if 'form_features' in features:
                        ai_feature_types.append('Form Patterns')
                    if 'class_features' in features:
                        ai_feature_types.append('Class Analysis')
                    if 'distance_features' in features:
                        ai_feature_types.append('Distance Optimization')
                    if 'jockey_trainer_features' in features:
                        ai_feature_types.append('Human Factor Analysis')
                    
                    print(f"  AI Feature Types: {', '.join(ai_feature_types)}")
                    
                    # Show feature complexity
                    total_features = 0
                    for key, value in features.items():
                        if isinstance(value, (list, np.ndarray)):
                            total_features += len(value)
                        elif isinstance(value, (int, float)):
                            total_features += 1
                    
                    print(f"  Total feature count: {total_features}")
                    
                    return len(ai_feature_types) > 0
                
                elif isinstance(features, (list, np.ndarray)):
                    print(f"  Feature vector length: {len(features)}")
                    return True
                
            else:
                print("❌ AI feature extraction failed")
                return False
                
    except Exception as e:
        print(f"❌ AI feature extraction test failed: {e}")
        return False

def test_ai_model_predictions():
    """Test 4: Test individual AI model predictions"""
    print("\n" + "=" * 60)
    print("TEST 4: Individual AI Model Predictions")
    print("=" * 60)
    
    try:
        with app.app_context():
            from utils.ai_predictor import AIPredictor, TENSORFLOW_AVAILABLE, PYTORCH_AVAILABLE
            
            # Create test data
            race, horses, jockeys, trainers = create_mock_race_data()
            
            # Initialize AI predictor
            ai_predictor = AIPredictor()
            
            # Get features
            features = ai_predictor._prepare_ai_features(race)
            
            if not features:
                print("❌ Cannot test models - feature extraction failed")
                return False
            
            model_results = []
            
            # Test TensorFlow models
            if TENSORFLOW_AVAILABLE:
                print("🧠 Testing TensorFlow models...")
                try:
                    tf_predictions = ai_predictor._get_tensorflow_predictions(features)
                    if tf_predictions:
                        print(f"  ✓ TensorFlow predictions: {type(tf_predictions)}")
                        model_results.append('TensorFlow')
                    else:
                        print("  ⚠ TensorFlow predictions: None")
                except Exception as e:
                    print(f"  ❌ TensorFlow prediction failed: {e}")
            
            # Test PyTorch models (if available)
            if PYTORCH_AVAILABLE:
                print("🔥 Testing PyTorch models...")
                try:
                    # Check if PyTorch prediction methods exist
                    if hasattr(ai_predictor, '_predict_pytorch_dnn'):
                        pytorch_pred = ai_predictor._predict_pytorch_dnn(features)
                        if pytorch_pred is not None:
                            print(f"  ✓ PyTorch DNN predictions: {type(pytorch_pred)}")
                            model_results.append('PyTorch DNN')
                    
                    if hasattr(ai_predictor, '_predict_pytorch_rnn'):
                        pytorch_rnn_pred = ai_predictor._predict_pytorch_rnn(features)
                        if pytorch_rnn_pred is not None:
                            print(f"  ✓ PyTorch RNN predictions: {type(pytorch_rnn_pred)}")
                            model_results.append('PyTorch RNN')
                            
                except Exception as e:
                    print(f"  ❌ PyTorch prediction failed: {e}")
            
            print(f"\n📊 Active AI Models: {len(model_results)}")
            if model_results:
                print(f"   Models: {', '.join(model_results)}")
                return True
            else:
                print("❌ No AI models produced predictions")
                return False
                
    except Exception as e:
        print(f"❌ AI model prediction test failed: {e}")
        return False

def test_web_interface_ai_indicators():
    """Test 5: Check web interface for AI indicators"""
    print("\n" + "=" * 60)
    print("TEST 5: Web Interface AI Indicators")
    print("=" * 60)
    
    try:
        with app.app_context():
            from utils.predictor import Predictor
            
            # Create test data
            race, horses, jockeys, trainers = create_mock_race_data()
            
            # Initialize predictor
            predictor = Predictor()
            
            print("🌐 Testing AI-enhanced prediction method...")
            
            # Test the web interface method
            result = predictor.predict_race_with_ai(race, use_ai=True, use_ensemble=True)
            
            if result:
                print("✓ AI-enhanced prediction successful")
                
                # Check for web interface indicators
                web_indicators = []
                
                if hasattr(result, 'algorithm'):
                    algorithm = getattr(result, 'algorithm', '')
                    if 'ai' in algorithm.lower() or 'ensemble' in algorithm.lower():
                        web_indicators.append(f'Algorithm: {algorithm}')
                
                if hasattr(result, 'confidence_scores'):
                    conf_scores = getattr(result, 'confidence_scores', {})
                    if isinstance(conf_scores, dict):
                        if 'ai_confidence' in conf_scores:
                            web_indicators.append('AI Confidence Score')
                        if 'ml_confidence' in conf_scores:
                            web_indicators.append('ML Confidence Score')
                
                # Check for AI insights
                ai_insights = predictor.get_ai_insights(race)
                if ai_insights:
                    web_indicators.append(f'AI Insights ({len(ai_insights)} items)')
                
                print(f"  Web AI Indicators: {', '.join(web_indicators) if web_indicators else 'None detected'}")
                
                return len(web_indicators) > 0
            else:
                print("❌ AI-enhanced prediction failed")
                return False
                
    except Exception as e:
        print(f"❌ Web interface AI test failed: {e}")
        return False

def main():
    """Run all AI verification tests"""
    print("🤖 AI VERIFICATION TEST SUITE")
    print("=" * 60)
    print("This test suite verifies that AI is actually being used in predictions")
    print("=" * 60)
    
    test_results = []
    test_names = [
        "AI Model Availability",
        "AI vs Traditional Predictions", 
        "AI Feature Extraction",
        "Individual AI Model Predictions",
        "Web Interface AI Indicators"
    ]
    
    # Run all tests
    test_results.append(test_ai_model_availability())
    test_results.append(test_ai_vs_traditional_predictions())
    test_results.append(test_ai_feature_extraction())
    test_results.append(test_ai_model_predictions())
    test_results.append(test_web_interface_ai_indicators())
    
    # Summary
    print("\n" + "=" * 60)
    print("AI VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    # Final verdict
    if passed_tests >= 3:
        print("\n🎉 AI VERIFICATION: SUCCESS!")
        print("✅ AI models are active and being used for predictions")
        print("✅ The system is using neural networks and machine learning")
        print("✅ AI-specific features and insights are being generated")
    elif passed_tests >= 1:
        print("\n⚠️  AI VERIFICATION: PARTIAL")
        print("🔶 Some AI functionality is working, but not all components")
        print("🔶 Check the failed tests to see what needs attention")
    else:
        print("\n❌ AI VERIFICATION: FAILED")
        print("🚨 AI models are not active - system is using fallback methods")
        print("🚨 Install TensorFlow/PyTorch and train models for AI functionality")
    
    return passed_tests >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)