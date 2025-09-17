#!/usr/bin/env python3
"""
Direct Model Training Script

This script trains the AI models directly without requiring web authentication.
It uses the generated training data to train both the ML and AI models.
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils.predictor import Predictor
from utils.data_processor import DataProcessor
from config.database_config import init_database, db
from flask import Flask

def create_app():
    """Create Flask app for database context"""
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///horse_racing.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    init_database(app)
    return app

def load_training_data_to_db():
    """Load generated training data into the database"""
    try:
        from sqlalchemy import text
        
        # Load training races
        with open('data/training_races.json', 'r') as f:
            races_data = json.load(f)
        
        # Load training horses
        with open('data/training_horses.json', 'r') as f:
            horses_data = json.load(f)
        
        print(f"Loaded {len(races_data)} training races")
        print(f"Loaded {len(horses_data)} training horses")
        
        # Clear existing data to avoid duplicates
        db.session.execute(text("DELETE FROM race_horses"))
        db.session.execute(text("DELETE FROM races"))
        db.session.execute(text("DELETE FROM horses"))
        db.session.commit()
        
        # Add horses to database using direct SQL
        for horse_data in horses_data:
            db.session.execute(text("""
                INSERT INTO horses (id, name, age, sex, color, sire, dam, trainer, jockey, owner, weight, form, rating, last_run, wins, places, runs, earnings, created_at, updated_at)
                VALUES (:id, :name, :age, :sex, :color, :sire, :dam, :trainer, :jockey, :owner, :weight, :form, :rating, :last_run, :wins, :places, :runs, :earnings, datetime('now'), datetime('now'))
            """), {
                'id': horse_data['id'],
                'name': horse_data['name'],
                'age': horse_data['age'],
                'sex': horse_data.get('sex', 'Unknown'),
                'color': horse_data.get('color', 'Bay'),
                'sire': horse_data.get('sire', 'Unknown'),
                'dam': horse_data.get('dam', 'Unknown'),
                'trainer': horse_data.get('trainer', 'Unknown'),
                'jockey': horse_data.get('jockey', 'Unknown'),
                'owner': horse_data.get('owner', 'Unknown'),
                'weight': horse_data.get('weight', 500.0),
                'form': horse_data.get('form', ''),
                'rating': horse_data.get('rating', 0),
                'last_run': None,
                'wins': horse_data.get('wins', 0),
                'places': horse_data.get('places', 0),
                'runs': horse_data.get('runs', 0),
                'earnings': horse_data.get('earnings', 0.0)
            })
        
        # Add races to database using direct SQL
        for race_data in races_data:
            # Convert date string to datetime if needed
            race_date = race_data['date']
            if isinstance(race_date, str):
                from datetime import datetime
                race_date = datetime.strptime(race_date, '%Y-%m-%d').strftime('%Y-%m-%d %H:%M:%S')
            
            # Convert distance to float (handle strings like "1200m")
            distance = race_data['distance']
            if isinstance(distance, str):
                # Extract numeric part from strings like "1200m"
                import re
                distance_match = re.search(r'(\d+(?:\.\d+)?)', distance)
                distance = float(distance_match.group(1)) if distance_match else 1200.0
            
            db.session.execute(text("""
                INSERT INTO races (id, name, date, time, track, distance, surface, status, results, created_at, updated_at)
                VALUES (:id, :name, :date, :time, :track, :distance, :surface, :status, :results, datetime('now'), datetime('now'))
            """), {
                'id': race_data['id'],
                'name': race_data['name'],
                'date': race_date,
                'time': race_data.get('time', '12:00'),
                'track': race_data.get('location', race_data.get('track', 'Unknown Track')),
                'distance': distance,
                'surface': race_data.get('surface', 'Dirt'),
                'status': race_data.get('status', 'completed' if race_data.get('results') else 'upcoming'),
                'results': json.dumps(race_data.get('results', []))
            })
        
        db.session.commit()
        
        # Now add horse-race relationships
        print("Adding horse-race relationships...")
        for race_data in races_data:
            if 'horse_ids' in race_data:
                for horse_id in race_data['horse_ids']:
                    db.session.execute(text("""
                        INSERT INTO race_horses (race_id, horse_id)
                        VALUES (:race_id, :horse_id)
                    """), {
                        'race_id': race_data['id'],
                        'horse_id': horse_id
                    })
        
        db.session.commit()
        
        # Count completed races
        completed_races = [race for race in races_data if race.get('results')]
        print(f"Found {len(completed_races)} completed races with results")
        print("✅ Training data loaded into database successfully")
        
        return len(races_data), len(horses_data), len(completed_races)
        
    except Exception as e:
        print(f"Error loading training data to database: {e}")
        db.session.rollback()
        return None, None, None

def train_models():
    """Train the prediction models"""
    print("="*60)
    print("HORSE RACING AI MODEL TRAINING")
    print("="*60)
    
    # Create Flask app context
    app = create_app()
    
    with app.app_context():
        # Initialize components
        predictor = Predictor()
        data_processor = DataProcessor()
        
        # Load training data into database
        num_races, num_horses, num_completed = load_training_data_to_db()
        
        if not num_races:
            print("❌ No training data available")
            return False
        
        print(f"\n📊 Training Data Summary:")
        print(f"   - Total races: {num_races}")
        print(f"   - Races with results: {num_completed}")
        print(f"   - Total horses: {num_horses}")
        
        # Train the main ML model
        print(f"\n🤖 Training ML Models...")
        try:
            result = predictor.train_model()
            
            if result.get('success'):
                print(f"✅ ML Model training completed successfully")
                print(f"   Method: {result.get('method', 'ensemble')}")
                print(f"   Message: {result.get('message', 'Training completed')}")
                
                # Get model info
                model_info = predictor.get_model_info()
                print(f"   Model trained: {model_info.get('trained', False)}")
                print(f"   Model type: {model_info.get('model_type', 'Unknown')}")
                
            else:
                print(f"❌ ML Model training failed: {result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Error training ML models: {str(e)}")
            return False
        
        # Train AI models
        print(f"\n🧠 Training AI Models...")
        try:
            ai_result = predictor.train_ai_models()
            
            if ai_result:
                print(f"✅ AI Model training completed successfully")
            else:
                print(f"⚠️  AI Model training completed with fallback methods")
                
        except Exception as e:
            print(f"❌ Error training AI models: {str(e)}")
            print(f"   Continuing with ML models only...")
        
        # Get final model status
        print(f"\n📈 Final Model Status:")
        try:
            performance_stats = predictor.get_performance_stats()
            print(f"   Total predictions: {performance_stats.get('total_predictions', 0)}")
            print(f"   Completed predictions: {performance_stats.get('completed_predictions', 0)}")
            print(f"   Average confidence: {performance_stats.get('avg_confidence', 0):.2f}")
            
        except Exception as e:
            print(f"   Could not retrieve performance stats: {str(e)}")
        
        print(f"\n🎉 Model training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"✅ Your AI models are now ready for predictions!")
        
        return True

if __name__ == "__main__":
    success = train_models()
    if success:
        print(f"\n🚀 You can now make AI predictions in the web application!")
        sys.exit(0)
    else:
        print(f"\n❌ Training failed. Please check the errors above.")
        sys.exit(1)