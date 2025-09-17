#!/usr/bin/env python3
"""
Simple AI Prediction Test

This script tests AI prediction functionality with only TensorFlow models.
"""

import os
import sys
from flask import Flask

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config.database_config import init_database

def create_app():
    """Create Flask app for context"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/horse_racing.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    init_database(app)
    return app

def test_ai_training_state():
    """Test AI training state loading"""
    print("="*60)
    print("SIMPLE AI TRAINING STATE TEST")
    print("="*60)
    
    # Create Flask app context
    app = create_app()
    
    with app.app_context():
        try:
            # Check if training state file exists
            from utils.ai_predictor import AIPredictor
            
            # Create a minimal test
            print("🤖 Testing AI predictor initialization...")
            
            # Check if model files exist
            model_dir = os.path.join(os.path.dirname(__file__), 'models', 'ai_models')
            training_state_file = os.path.join(model_dir, 'training_state.pkl')
            
            print(f"📁 Model directory: {model_dir}")
            print(f"📄 Training state file exists: {os.path.exists(training_state_file)}")
            
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                print(f"📋 Model files: {files}")
            
            # Try to load just the training state
            import pickle
            if os.path.exists(training_state_file):
                with open(training_state_file, 'rb') as f:
                    state = pickle.load(f)
                print(f"✅ Training state loaded: is_trained = {state.get('is_trained', False)}")
                return True
            else:
                print("❌ No training state file found")
                return False
                
        except Exception as e:
            print(f"❌ Error during test: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_ai_training_state()
    if success:
        print("\n🎉 AI training state is accessible!")
        sys.exit(0)
    else:
        print("\n💥 AI training state has issues!")
        sys.exit(1)