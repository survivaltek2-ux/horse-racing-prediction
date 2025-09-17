#!/usr/bin/env python3
"""
Direct AI Model Training Script

This script trains the AI models directly within the Flask app context.
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

from utils.predictor import Predictor
from config.database_config import init_database

def create_app():
    """Create Flask app for context"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/horse_racing.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    init_database(app)
    return app

def train_ai_models():
    """Train AI models directly"""
    print("="*60)
    print("DIRECT AI MODEL TRAINING")
    print("="*60)
    
    # Create Flask app context
    app = create_app()
    
    with app.app_context():
        try:
            # Initialize predictor
            print("🤖 Initializing predictor...")
            predictor = Predictor()
            
            # Check current AI training status
            print(f"📊 Current AI training status: {predictor.ai_predictor.is_trained}")
            
            # Train AI models
            print("🚀 Training AI models...")
            success = predictor.train_ai_models()
            
            if success:
                print("✅ AI models trained successfully!")
                print(f"📊 New AI training status: {predictor.ai_predictor.is_trained}")
                return True
            else:
                print("❌ AI model training failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error during AI training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = train_ai_models()
    if success:
        print("\n🎉 AI models are now ready for predictions!")
        sys.exit(0)
    else:
        print("\n💥 AI model training failed!")
        sys.exit(1)