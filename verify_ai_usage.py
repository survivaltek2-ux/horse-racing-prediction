#!/usr/bin/env python3
"""
Simple AI Usage Verification
This script provides clear ways to verify that AI is actually being used in predictions.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_ai_libraries():
    """Check if AI libraries are installed and available"""
    print("🔍 CHECKING AI LIBRARIES")
    print("-" * 40)
    
    try:
        from utils.ai_predictor import TENSORFLOW_AVAILABLE, PYTORCH_AVAILABLE
        print(f"📦 TensorFlow: {'✅ Available' if TENSORFLOW_AVAILABLE else '❌ Not Available'}")
        print(f"📦 PyTorch: {'✅ Available' if PYTORCH_AVAILABLE else '❌ Not Available'}")
        
        if TENSORFLOW_AVAILABLE:
            import tensorflow as tf
            print(f"   TensorFlow version: {tf.__version__}")
        
        if PYTORCH_AVAILABLE:
            import torch
            print(f"   PyTorch version: {torch.__version__}")
        
        return TENSORFLOW_AVAILABLE or PYTORCH_AVAILABLE
    except Exception as e:
        print(f"❌ Error checking AI libraries: {e}")
        return False

def check_ai_models():
    """Check if AI models are initialized"""
    print("\n🧠 CHECKING AI MODELS")
    print("-" * 40)
    
    try:
        from utils.ai_predictor import AIPredictor
        
        predictor = AIPredictor()
        print("✅ AI Predictor initialized successfully")
        
        # Check for model attributes
        models_found = []
        
        # TensorFlow models
        tf_models = ['tf_dnn_model', 'tf_lstm_model', 'tf_cnn_model']
        for model_name in tf_models:
            if hasattr(predictor, model_name):
                model = getattr(predictor, model_name)
                if model is not None:
                    models_found.append(model_name)
                    print(f"   ✅ {model_name}: Initialized")
        
        # PyTorch models
        pytorch_models = ['pytorch_dnn_model', 'pytorch_rnn_model']
        for model_name in pytorch_models:
            if hasattr(predictor, model_name):
                model = getattr(predictor, model_name)
                if model is not None:
                    models_found.append(model_name)
                    print(f"   ✅ {model_name}: Initialized")
        
        print(f"\n📊 Total AI models available: {len(models_found)}")
        return len(models_found) > 0
        
    except Exception as e:
        print(f"❌ Error checking AI models: {e}")
        return False

def check_ai_prediction_methods():
    """Check if AI prediction methods exist"""
    print("\n🎯 CHECKING AI PREDICTION METHODS")
    print("-" * 40)
    
    try:
        from utils.ai_predictor import AIPredictor
        from utils.predictor import Predictor
        
        # Check AI predictor methods
        ai_predictor = AIPredictor()
        ai_methods = []
        
        if hasattr(ai_predictor, 'predict_race_ai'):
            ai_methods.append('predict_race_ai')
            print("   ✅ predict_race_ai: Available")
        
        if hasattr(ai_predictor, '_ensemble_ai_predictions'):
            ai_methods.append('_ensemble_ai_predictions')
            print("   ✅ _ensemble_ai_predictions: Available")
        
        if hasattr(ai_predictor, '_generate_ai_insights'):
            ai_methods.append('_generate_ai_insights')
            print("   ✅ _generate_ai_insights: Available")
        
        # Check main predictor AI integration
        main_predictor = Predictor()
        if hasattr(main_predictor, 'predict_race_with_ai'):
            ai_methods.append('predict_race_with_ai')
            print("   ✅ predict_race_with_ai: Available")
        
        if hasattr(main_predictor, 'get_ai_insights'):
            ai_methods.append('get_ai_insights')
            print("   ✅ get_ai_insights: Available")
        
        print(f"\n📊 AI prediction methods: {len(ai_methods)}")
        return len(ai_methods) > 0
        
    except Exception as e:
        print(f"❌ Error checking AI prediction methods: {e}")
        return False

def check_web_ai_routes():
    """Check if web application has AI routes"""
    print("\n🌐 CHECKING WEB AI ROUTES")
    print("-" * 40)
    
    try:
        from app import app
        
        ai_routes = []
        
        # Check for AI-specific routes
        for rule in app.url_map.iter_rules():
            if 'ai' in rule.rule.lower():
                ai_routes.append(rule.rule)
                print(f"   ✅ {rule.rule}: {', '.join(rule.methods)}")
        
        print(f"\n📊 AI web routes: {len(ai_routes)}")
        return len(ai_routes) > 0
        
    except Exception as e:
        print(f"❌ Error checking web AI routes: {e}")
        return False

def demonstrate_ai_vs_fallback():
    """Demonstrate the difference between AI and fallback predictions"""
    print("\n🔬 DEMONSTRATING AI vs FALLBACK")
    print("-" * 40)
    
    try:
        from utils.ai_predictor import AIPredictor
        
        # Create a simple mock race
        mock_race = type('MockRace', (), {
            'id': 1,
            'name': 'Test Race',
            'horses': []
        })()
        
        predictor = AIPredictor()
        
        # Try to get a prediction
        result = predictor.predict_race_ai(mock_race)
        
        if result:
            print("✅ AI prediction method executed")
            print(f"   Result type: {type(result)}")
            
            if isinstance(result, dict):
                print(f"   Keys: {list(result.keys())}")
                
                # Check for AI-specific indicators
                ai_indicators = []
                if 'ai_insights' in result:
                    ai_indicators.append('AI Insights')
                if 'confidence_scores' in result:
                    ai_indicators.append('Confidence Scores')
                if 'ensemble_weights' in result:
                    ai_indicators.append('Ensemble Weights')
                
                if ai_indicators:
                    print(f"   AI Indicators: {', '.join(ai_indicators)}")
                    
                    # Check insights for AI vs fallback
                    if 'ai_insights' in result and result['ai_insights']:
                        insights = result['ai_insights']
                        print(f"   Insights count: {len(insights)}")
                        
                        # Look for fallback indicators
                        fallback_terms = ['fallback', 'not trained', 'not available']
                        is_fallback = any(
                            any(term in insight.lower() for term in fallback_terms)
                            for insight in insights
                        )
                        
                        if is_fallback:
                            print("   ⚠️  Using FALLBACK prediction (AI models not trained)")
                        else:
                            print("   ✅ Using ACTUAL AI prediction")
                        
                        return not is_fallback
                
        return False
        
    except Exception as e:
        print(f"❌ Error demonstrating AI vs fallback: {e}")
        return False

def show_how_to_verify_ai():
    """Show users how to verify AI is being used"""
    print("\n📋 HOW TO VERIFY AI IS BEING USED")
    print("=" * 50)
    
    print("\n1. 🔍 CHECK THE WEB INTERFACE:")
    print("   • Look for 'AI Enabled' badge on prediction results")
    print("   • Check for 'AI Confidence' scores")
    print("   • Look for AI-specific insights in results")
    print("   • Use the '/predict_ai/<race_id>' route instead of '/predict/<race_id>'")
    
    print("\n2. 🧠 CHECK THE PREDICTION ALGORITHM:")
    print("   • AI predictions show algorithm as 'AI Ensemble' or similar")
    print("   • Traditional predictions show 'Random Forest' or 'Statistical'")
    print("   • AI results include neural network insights")
    
    print("\n3. 📊 CHECK THE CONFIDENCE SCORES:")
    print("   • AI predictions have separate 'ai_confidence' scores")
    print("   • Look for 'ml_confidence' vs 'ai_confidence' breakdown")
    print("   • Ensemble predictions combine multiple model confidences")
    
    print("\n4. 🔬 CHECK THE INSIGHTS:")
    print("   • AI insights mention 'neural networks', 'deep learning'")
    print("   • Fallback insights mention 'not trained' or 'fallback'")
    print("   • AI insights analyze speed patterns, form sequences")
    
    print("\n5. 🌐 CHECK THE URL:")
    print("   • AI predictions: /predict_ai/<race_id>")
    print("   • Traditional predictions: /predict/<race_id>")
    print("   • API endpoint: /api/predict_ai/<race_id>")

def main():
    """Run AI verification checks"""
    print("🤖 AI USAGE VERIFICATION")
    print("=" * 50)
    
    checks = [
        ("AI Libraries", check_ai_libraries),
        ("AI Models", check_ai_models),
        ("AI Prediction Methods", check_ai_prediction_methods),
        ("Web AI Routes", check_web_ai_routes),
        ("AI vs Fallback", demonstrate_ai_vs_fallback)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{'='*20} {name} {'='*20}")
        result = check_func()
        results.append(result)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*50}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {passed}/{total} checks")
    
    if passed >= 4:
        print("\n🎉 AI IS FULLY OPERATIONAL!")
        print("✅ Your system is using AI for predictions")
    elif passed >= 2:
        print("\n⚠️  AI IS PARTIALLY WORKING")
        print("🔶 Some AI components are active")
    else:
        print("\n❌ AI IS NOT ACTIVE")
        print("🚨 System is using fallback methods")
    
    # Show how to verify
    show_how_to_verify_ai()
    
    return passed >= 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)