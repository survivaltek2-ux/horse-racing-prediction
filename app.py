import os
import warnings
import logging

# Suppress TensorFlow CPU optimization warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs except errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Suppress OpenMP warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt

# Import comprehensive logging and error handling systems
from config.logging_config import debug_logger, performance_monitor, debug_route
from config.error_handler import error_handler, catch_exceptions, validate_form_data, log_database_operation

# Try to import data science libraries
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pandas not available: {e}")
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: numpy not available: {e}")
    NUMPY_AVAILABLE = False
    np = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None
    StandardScaler = None
from models.sqlalchemy_models import User, APICredentials, Horse as SQLHorse
from models.race import Race
from models.horse import Horse
from models.prediction import Prediction
from utils.data_processor import DataProcessor
from utils.predictor import Predictor
from forms import RaceForm, HorseForm, PredictionForm, RaceResultForm, AddHorseToRaceForm, LoginForm, RegisterForm, UserManagementForm, ChangePasswordForm, APICredentialsForm, APICredentialsTestForm
from services.api_service import api_service
from config.api_config import APIConfig
from config.database_config import init_database, db
import os
import uuid
import threading
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize comprehensive logging and error handling systems
debug_logger.init_app(app)
error_handler.init_app(app)

# Initialize SQLite database
init_database(app)

# Use environment variable for secret key, fallback to random for development
secret_key = os.getenv('SECRET_KEY')
if secret_key:
    app.secret_key = secret_key
else:
    # Generate a random key for development (not recommended for production)
    app.secret_key = os.urandom(24)
    print("⚠️  WARNING: Using random secret key. Set SECRET_KEY environment variable for production!")

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Initialize Bcrypt for password hashing
bcrypt = Bcrypt(app)

# Add hasattr to Jinja2 template globals
app.jinja_env.globals['hasattr'] = hasattr

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    try:
        return User.get_by_id(int(user_id))
    except (ValueError, TypeError):
        return None

# Initialize data processor and predictor
data_processor = DataProcessor()
predictor = Predictor()

# Persistent prediction status storage
import json
import fcntl

class PredictionStatusManager:
    def __init__(self, storage_file='prediction_status.json'):
        self.storage_file = storage_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure the storage file exists"""
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, 'w') as f:
                json.dump({}, f)
    
    def _read_status(self):
        """Read status from file with file locking"""
        try:
            with open(self.storage_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _write_status(self, data):
        """Write status to file with file locking"""
        with open(self.storage_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
            json.dump(data, f, indent=2, default=str)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Unlock
    
    def get(self, job_id):
        """Get job status"""
        data = self._read_status()
        return data.get(job_id)
    
    def set(self, job_id, status_data):
        """Set job status"""
        data = self._read_status()
        data[job_id] = status_data
        self._write_status(data)
    
    def delete(self, job_id):
        """Delete job status"""
        data = self._read_status()
        if job_id in data:
            del data[job_id]
            self._write_status(data)
    
    def __contains__(self, job_id):
        """Check if job exists"""
        data = self._read_status()
        return job_id in data
    
    def __getitem__(self, job_id):
        """Get job status (dict-like access)"""
        result = self.get(job_id)
        if result is None:
            raise KeyError(job_id)
        return result
    
    def __setitem__(self, job_id, status_data):
        """Set job status (dict-like access)"""
        self.set(job_id, status_data)

# Global prediction status manager
prediction_status = PredictionStatusManager()

# Simple thread test storage
thread_test_status = {}

# Initialize API service
api_service.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()
    print("Database tables created successfully")

@app.route('/')
def index():
    """Render the home page"""
    # Get some basic statistics for the dashboard
    total_races = len(Race.get_all_races())
    total_horses = len(Horse.get_all_horses())
    total_predictions = len(Prediction.get_all_predictions())
    upcoming_races = len(Race.get_upcoming_races())
    
    # Calculate additional stats for the dashboard
    predictions = Prediction.get_all_predictions()
    win_predictions = len(predictions) if predictions else 0  # Simplified calculation
    
    # Calculate accuracy (placeholder calculation - you may want to implement proper accuracy calculation)
    accuracy = 75 if total_predictions > 0 else 0  # Placeholder value
    avg_accuracy = 78 if total_predictions > 0 else 0  # Placeholder value
    
    stats = {
        'total_races': total_races,
        'total_horses': total_horses,
        'total_predictions': total_predictions,
        'upcoming_races': upcoming_races,
        'win_predictions': win_predictions,
        'accuracy': accuracy,
        'avg_accuracy': avg_accuracy
    }
    
    return render_template('index.html', stats=stats)

@app.route('/races')
def races():
    """Display list of races"""
    try:
        # First try to get upcoming races
        race_list = Race.get_upcoming_races()
        
        # If no upcoming races, get all races
        if not race_list:
            race_list = Race.get_all_races()
        
        return render_template('races.html', races=race_list)
    except Exception as e:
        print(f"Error getting races: {e}")
        flash('Error loading races. Please try again.', 'error')
        return render_template('races.html', races=[])

@app.route('/horses')
def horses():
    """Display list of all horses"""
    try:
        horse_list = Horse.get_all_horses()
        return render_template('horses.html', horses=horse_list)
    except Exception as e:
        print(f"Error getting horses: {e}")
        flash('Error loading horses. Please try again.', 'error')
        return render_template('horses.html', horses=[])

@app.route('/race/<int:race_id>')
def race_details(race_id):
    """Display details for a specific race"""
    race = Race.get_race_by_id(race_id)
    if not race:
        flash('Race not found', 'error')
        return redirect(url_for('races'))
    
    # Get horses for this race using the method
    horses = race.get_horses()
    return render_template('race_details.html', race=race, horses=horses)

@app.route('/predict/<int:race_id>', methods=['GET', 'POST'])
@login_required
def predict(race_id):
    """Generate and display predictions for a race"""
    race = Race.get_race_by_id(race_id)
    if not race:
        flash('Race not found.', 'error')
        return redirect(url_for('races'))
    
    form = PredictionForm()
    
    if form.validate_on_submit():
        # Generate prediction using selected algorithm
        algorithm = form.algorithm.data
        prediction = predictor.predict_race(race, {'algorithm': algorithm})
        
        if prediction is None:
            flash('Unable to generate prediction for this race. Please check that the race has horses and valid data.', 'error')
            return render_template('predict.html', race=race, form=form)
            
        return render_template('prediction_results.html', race=race, prediction=prediction, algorithm=algorithm)
    
    # GET request - show prediction form
    return render_template('predict.html', race=race, form=form)

@app.route('/predict_ai/<int:race_id>', methods=['GET', 'POST'])
@login_required
def predict_ai(race_id):
    """Generate AI-enhanced predictions for a race"""
    race = Race.get_race_by_id(race_id)
    if not race:
        flash('Race not found.', 'error')
        return redirect(url_for('races'))
    
    if request.method == 'POST':
        # Generate AI-enhanced prediction
        use_ai = request.form.get('use_ai', 'true').lower() == 'true'
        use_ensemble = request.form.get('use_ensemble', 'true').lower() == 'true'
        
        try:
            prediction = predictor.predict_race_with_ai(race, use_ai=use_ai, use_ensemble=use_ensemble)
            ai_insights = predictor.get_ai_insights(race)
            
            # Handle case where prediction is None
            if prediction is None:
                flash('Unable to generate prediction for this race. The race may not have enough horse data.', 'warning')
                return redirect(url_for('predict', race_id=race_id))
            
            return render_template('ai_prediction_results.html', 
                                 race=race, 
                                 prediction=prediction, 
                                 ai_insights=ai_insights,
                                 use_ai=use_ai,
                                 use_ensemble=use_ensemble,
                                 generated_at=datetime.now())
        except Exception as e:
            flash(f'Error generating AI prediction: {str(e)}', 'error')
            return redirect(url_for('predict', race_id=race_id))
    
    # GET request - show AI prediction form
    return render_template('predict_ai.html', race=race)

@app.route('/ai_prediction_results/<int:race_id>')
@login_required
def ai_prediction_results(race_id):
    """Display AI prediction results for a specific race"""
    app.logger.info(f"DEBUG ROUTE: ai_prediction_results called for race_id {race_id}")
    race = Race.get_race_by_id(race_id)
    if not race:
        flash('Race not found', 'error')
        return redirect(url_for('races'))
    
    try:
        # Generate AI-enhanced prediction with default settings
        use_ai = True
        use_ensemble = True
        
        app.logger.info(f"DEBUG ROUTE: About to call predict_race_with_ai")
        prediction = predictor.predict_race_with_ai(race, use_ai=use_ai, use_ensemble=use_ensemble)
        app.logger.info(f"DEBUG ROUTE: Got prediction: {type(prediction)}")
        if hasattr(prediction, 'predictions'):
            app.logger.info(f"DEBUG ROUTE: Prediction data: {prediction.predictions}")
        
        ai_insights = predictor.get_ai_insights(race)
        
        # Handle case where prediction is None
        if prediction is None:
            flash('Unable to generate prediction for this race. The race may not have enough horse data.', 'warning')
            return redirect(url_for('predict_ai', race_id=race_id))
        
        return render_template('ai_prediction_results.html', 
                             race=race, 
                             prediction=prediction, 
                             ai_insights=ai_insights,
                             use_ai=use_ai,
                             use_ensemble=use_ensemble,
                             generated_at=datetime.now())
    except Exception as e:
        app.logger.error(f"DEBUG ROUTE: Exception in ai_prediction_results: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error generating AI prediction: {str(e)}', 'error')
        return redirect(url_for('predict_ai', race_id=race_id))

@app.route('/api/predict_ai/<int:race_id>', methods=['POST'])
@login_required
def api_predict_ai(race_id):
    """API endpoint for AI-enhanced predictions"""
    race = Race.get_race_by_id(race_id)
    if not race:
        return jsonify({'error': 'Race not found'}), 404
    
    try:
        # Get parameters from request
        data = request.get_json() or {}
        use_ai = data.get('use_ai', True)
        use_ensemble = data.get('use_ensemble', True)
        
        # Generate AI prediction
        prediction = predictor.predict_race_with_ai(race, use_ai=use_ai, use_ensemble=use_ensemble)
        ai_insights = predictor.get_ai_insights(race)
        
        # Format response
        response = {
            'race_id': race_id,
            'race_name': race.name,
            'prediction': {
                'algorithm': getattr(prediction, 'algorithm', 'ai_ensemble'),
                'confidence_scores': getattr(prediction, 'confidence_scores', {}),
                'predictions': getattr(prediction, 'predictions', {}),
                'ai_insights': ai_insights
            },
            'ai_enabled': use_ai,
            'ensemble_enabled': use_ensemble,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/ai_insights/<int:race_id>', methods=['GET'])
@login_required
def api_ai_insights(race_id):
    """API endpoint to get AI insights for a race"""
    race = Race.get_race_by_id(race_id)
    if not race:
        return jsonify({'error': 'Race not found'}), 404
    
    try:
        insights = predictor.get_ai_insights(race)
        return jsonify({
            'race_id': race_id,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get insights: {str(e)}'}), 500

def run_ai_prediction_async(job_id, race_id, use_ai, use_ensemble):
    """Background function to run AI prediction"""
    print(f"🧵 Thread {threading.current_thread().name} starting for job: {job_id}")
    
    try:
        with app.app_context():
            print(f"✅ Flask app context established for job: {job_id}")
            print(f"🚀 Starting background prediction for job: {job_id}")
            print(f"🔍 Job parameters: race_id={race_id}, use_ai={use_ai}, use_ensemble={use_ensemble}")
            
            # Update status to processing - use proper status manager methods
            job_data = prediction_status.get(job_id) or {}
            job_data['status'] = 'processing'
            job_data['progress'] = 10
            prediction_status.set(job_id, job_data)
            print(f"📈 Job {job_id} progress: 10% (processing started)")
            
            # Get race data
            print(f"🏇 Fetching race data for race_id: {race_id}")
            race = Race.get_race_by_id(race_id)
            if not race:
                print(f"❌ Race {race_id} not found")
                job_data['status'] = 'error'
                job_data['error'] = 'Race not found'
                prediction_status.set(job_id, job_data)
                return
            
            print(f"✅ Race found: {race.name if hasattr(race, 'name') else 'Unknown'}")
            job_data['progress'] = 30
            prediction_status.set(job_id, job_data)
            print(f"📈 Job {job_id} progress: 30% (race data loaded)")
            
            # Run the prediction
            print(f"🤖 Starting prediction with AI={use_ai}, ensemble={use_ensemble}")
            if use_ai:
                job_data['progress'] = 50
                prediction_status.set(job_id, job_data)
                print(f"📈 Job {job_id} progress: 50% (running AI prediction)")
                predictions = predictor.predict_race_with_ai(race, use_ensemble=use_ensemble)
            else:
                job_data['progress'] = 50
                prediction_status.set(job_id, job_data)
                print(f"📈 Job {job_id} progress: 50% (running traditional prediction)")
                predictions = predictor.predict_race(race)
            
            print(f"✅ Prediction completed successfully")
            job_data['progress'] = 80
            prediction_status.set(job_id, job_data)
            print(f"📈 Job {job_id} progress: 80% (prediction complete, formatting results)")
            
            # Format the response - convert Prediction object to dict if needed
            if hasattr(predictions, 'to_dict'):
                # If it's a Prediction object, convert to dict
                predictions_dict = predictions.to_dict()
            elif hasattr(predictions, '__dict__'):
                # If it has attributes, convert to dict
                predictions_dict = {
                    'id': getattr(predictions, 'id', None),
                    'race_id': getattr(predictions, 'race_id', race_id),
                    'predictions': getattr(predictions, 'predictions', {}),
                    'algorithm': getattr(predictions, 'algorithm', 'ai_enhanced'),
                    'confidence_scores': getattr(predictions, 'confidence_scores', {}),
                    'ai_insights': getattr(predictions, 'ai_insights', []),
                    'created_at': getattr(predictions, 'created_at', datetime.now().isoformat())
                }
            else:
                # If it's already a dict or simple value
                predictions_dict = predictions if isinstance(predictions, dict) else {'predictions': predictions}
            
            response = {
                'race_id': race_id,
                'predictions': predictions_dict,
                'ai_enabled': use_ai,
                'ensemble_enabled': use_ensemble,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update status to completed
            print(f"🎉 Job {job_id} completed successfully")
            job_data['status'] = 'completed'
            job_data['progress'] = 100
            job_data['result'] = response
            job_data['completed_at'] = datetime.now().isoformat()
            prediction_status.set(job_id, job_data)
            print(f"📈 Job {job_id} progress: 100% (COMPLETED)")
            
    except Exception as e:
        print(f"❌ ERROR in background prediction job {job_id}: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        print(f"📋 Full traceback:\n{traceback.format_exc()}")
        job_data = prediction_status.get(job_id) or {}
        job_data['status'] = 'error'
        job_data['error'] = str(e)
        job_data['progress'] = 0
        prediction_status.set(job_id, job_data)
        print(f"💾 Error saved to job status")

@app.route('/api/predict_ai_async/<int:race_id>', methods=['POST'])
@login_required
def api_predict_ai_async(race_id):
    """Async API endpoint to generate AI-enhanced predictions"""
    try:
        data = request.get_json() or {}
        use_ai = data.get('use_ai', True)
        use_ensemble = data.get('use_ensemble', False)
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        job_data = {
            'status': 'started',
            'progress': 0,
            'race_id': race_id,
            'created_at': datetime.now().isoformat()
        }
        prediction_status[job_id] = job_data
        
        # Debug logging
        print(f"🔄 Created AI prediction job: {job_id} for race {race_id}")
        print(f"📊 Job data: {job_data}")
        
        # Start prediction in background thread
        print(f"🧵 Creating background thread for job: {job_id}")
        thread = threading.Thread(
            target=run_ai_prediction_async,
            args=(job_id, race_id, use_ai, use_ensemble),
            name=f"AI-Prediction-{job_id[:8]}"
        )
        # Remove daemon=True to prevent thread termination when request completes
        print(f"🚀 Starting background thread: {thread.name}")
        thread.start()
        print(f"✅ Background thread started successfully: {thread.name}")
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'AI prediction started'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start prediction: {str(e)}'}), 500

@app.route('/api/prediction_status/<job_id>', methods=['GET'])
@login_required
def api_prediction_status(job_id):
    """API endpoint to check prediction status"""
    # Debug logging
    print(f"🔍 Looking up job status for: {job_id}")
    
    if job_id not in prediction_status:
        print(f"❌ Job not found: {job_id}")
        # List all current jobs for debugging
        all_jobs = prediction_status._read_status()
        print(f"📋 Current jobs in storage: {list(all_jobs.keys())}")
        return jsonify({'error': 'Job not found'}), 404
    
    job_data = prediction_status[job_id].copy()
    print(f"✅ Found job: {job_id}, status: {job_data.get('status', 'unknown')}")
    
    # Clean up completed or errored jobs after 1 hour
    if job_data['status'] in ['completed', 'error']:
        created_at = datetime.fromisoformat(job_data['created_at'])
        if (datetime.now() - created_at).total_seconds() > 3600:  # 1 hour
            del prediction_status[job_id]
    
    return jsonify(job_data)

@app.route('/api/train_ai', methods=['POST'])
@login_required
def api_train_ai():
    """API endpoint to train AI models"""
    try:
        # Check if user has admin privileges (you might want to add this check)
        success = predictor.train_ai_models()
        
        return jsonify({
            'success': success,
            'message': 'AI models trained successfully' if success else 'AI training failed',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

def simple_thread_test(test_id):
    """Simple function to test if threads work under gunicorn"""
    print(f"🧵 Thread test {test_id} starting...")
    import time
    for i in range(5):
        time.sleep(1)
        thread_test_status[test_id] = {
            'status': 'running',
            'progress': (i + 1) * 20,
            'message': f'Step {i + 1}/5'
        }
        print(f"🔄 Thread test {test_id}: Step {i + 1}/5")
    
    thread_test_status[test_id] = {
        'status': 'completed',
        'progress': 100,
        'message': 'Test completed successfully'
    }
    print(f"✅ Thread test {test_id} completed!")

@app.route('/api/test_thread', methods=['POST'])
@login_required
def api_test_thread():
    """Test endpoint to check if threads work under gunicorn"""
    test_id = str(uuid.uuid4())[:8]
    
    # Initialize status
    thread_test_status[test_id] = {
        'status': 'started',
        'progress': 0,
        'message': 'Thread test starting...'
    }
    
    print(f"🧵 Creating thread test: {test_id}")
    
    # Create and start thread
    thread = threading.Thread(
        target=simple_thread_test,
        args=(test_id,),
        name=f"ThreadTest-{test_id}"
    )
    print(f"🚀 Starting thread test: {test_id}")
    thread.start()
    print(f"✅ Thread test started: {test_id}")
    
    return jsonify({
        'success': True,
        'test_id': test_id,
        'message': 'Thread test started'
    })

@app.route('/api/test_thread_status/<test_id>', methods=['GET'])
@login_required
def api_test_thread_status(test_id):
    """Get status of thread test"""
    if test_id in thread_test_status:
        return jsonify(thread_test_status[test_id])
    else:
        return jsonify({'error': 'Test not found'}), 404

@app.route('/add_race', methods=['GET', 'POST'])
@login_required
def add_race():
    """Add a new race"""
    form = RaceForm()
    
    # Populate horse choices
    horses = Horse.get_all_horses()
    horse_choices = [('', 'Select a horse (optional)')]
    for horse in horses:
        horse_choices.append((str(horse.id), f"{horse.name} - {horse.age}yo {horse.sex}"))
    form.assigned_horses.choices = horse_choices
    
    print(f"DEBUG: Request method: {request.method}")
    print(f"DEBUG: Form validation result: {form.validate_on_submit()}")
    print(f"DEBUG: Form data: {request.form}")
    if form.errors:
        print(f"DEBUG: Form errors: {form.errors}")
    
    if form.validate_on_submit():
        print("DEBUG: Form validation passed, creating race...")
        try:
            # Create new race from form data with all enhanced fields
            race = Race(
                # Basic Race Information
                name=form.name.data,
                date=form.date.data,
                track=form.location.data,
                distance=float(form.distance.data) if form.distance.data else 0.0,
                prize_money=float(form.purse.data) if form.purse.data else 0.0,
                description=form.description.data,
                
                # Weather Conditions
                temperature=form.temperature.data,
                humidity=form.humidity.data,
                wind_speed=form.wind_speed.data,
                wind_direction=form.wind_direction.data,
                weather_description=form.weather_description.data,
                visibility=form.visibility.data,
                
                # Track Conditions
                track_condition=form.track_condition.data,
                surface_type=form.surface_type.data,
                rail_position=form.rail_position.data,
                track_bias=form.track_bias.data,
                track_maintenance=form.track_maintenance.data,
                
                # Field Analysis
                field_size=form.field_size.data,
                field_quality=form.field_quality.data,
                pace_scenario=form.pace_scenario.data,
                competitive_balance=form.competitive_balance.data,
                speed_figures_range=form.speed_figures_range.data,
                
                # Betting Information
                total_pool=form.total_pool.data,
                win_pool=form.win_pool.data,
                exacta_pool=form.exacta_pool.data,
                trifecta_pool=form.trifecta_pool.data,
                superfecta_pool=form.superfecta_pool.data,
                morning_line_favorite=form.morning_line_favorite.data,
                
                # Race Conditions
                age_restrictions=form.age_restrictions.data,
                sex_restrictions=form.sex_restrictions.data,
                weight_conditions=form.weight_conditions.data,
                claiming_price=form.claiming_price.data,
                race_grade=form.race_grade.data,
                
                # Historical Data
                track_record=form.track_record.data,
                average_winning_time=form.average_winning_time.data,
                course_record_holder=form.course_record_holder.data,
                similar_race_results=form.similar_race_results.data,
                trainer_jockey_stats=form.trainer_jockey_stats.data,
                
                # Media Coverage
                tv_coverage=form.tv_coverage.data,
                streaming_available=form.streaming_available.data,
                featured_race=form.featured_race.data,
                
                # Legacy fields for compatibility
                surface=form.surface_type.data,  # Map to legacy field
                weather=form.weather_description.data  # Map to legacy field
            )
            
            # Save the race
            race.save()
            
            # Handle horse assignment if selected
            if form.assigned_horses.data:
                try:
                    horse = Horse.get_by_id(int(form.assigned_horses.data))
                    if horse:
                        race.add_horse(horse.id)
                        flash(f'Race created successfully and horse "{horse.name}" assigned!', 'success')
                    else:
                        flash('Race created successfully, but selected horse not found.', 'warning')
                except Exception as e:
                    flash(f'Race created successfully, but error assigning horse: {str(e)}', 'warning')
            else:
                flash('Race created successfully with enhanced data!', 'success')
            
            return redirect(url_for('races'))
        except Exception as e:
            flash(f'Error creating race: {str(e)}', 'error')
    
    return render_template('add_race.html', form=form)

@app.route('/add_horse', methods=['GET', 'POST'])
@login_required
def add_horse():
    """Add a new horse"""
    form = HorseForm()
    
    # Populate race choices
    races = Race.get_all_races()
    race_choices = [('', 'Select a race (optional)')]
    race_choices.extend([(str(race.id), f"{race.name} - {race.date if race.date else 'No date'}") for race in races])
    form.assigned_races.choices = race_choices
    
    if form.validate_on_submit():
        try:
            # Create new horse from form data using SQLAlchemy model
            # Only include fields that exist in the SQLAlchemy Horse model
            horse = SQLHorse(
                # Basic Information
                name=form.name.data,
                age=form.age.data,
                sex=form.sex.data,
                color=form.color.data,
                breed=form.breed.data,
                
                # Connections
                trainer=form.trainer.data,
                jockey=form.jockey.data,
                owner=form.owner.data,
                
                # Pedigree Information
                sire=form.sire.data,
                dam=form.dam.data,
                
                # Use the most recent speed rating for the speed_rating field
                speed_rating=form.speed_rating_race_1.data
            )
            
            # Save to database using SQLAlchemy
            db.session.add(horse)
            db.session.commit()
            
            # Handle race assignment if selected
            if form.assigned_races.data:
                try:
                    race = Race.get_race_by_id(int(form.assigned_races.data))
                    if race:
                        race.add_horse(horse.id)
                        flash(f'Horse added successfully and assigned to race "{race.name}"!', 'success')
                    else:
                        flash('Horse added successfully, but selected race not found.', 'warning')
                except Exception as e:
                    flash(f'Horse added successfully, but error assigning to race: {str(e)}', 'warning')
            else:
                flash('Horse added successfully with enhanced data!', 'success')
            
            return redirect(url_for('races'))
        except Exception as e:
            flash(f'Error adding horse: {str(e)}', 'error')
    
    return render_template('add_horse.html', form=form)

@app.route('/edit_horse/<int:horse_id>', methods=['GET', 'POST'])
@login_required
@debug_route
@performance_monitor
@catch_exceptions
def edit_horse(horse_id):
    """Edit an existing horse with enhanced data"""
    horse = Horse.get_by_id(horse_id)
    if not horse:
        flash('Horse not found.', 'error')
        return redirect(url_for('races'))
    
    form = HorseForm(obj=horse)
    
    # Populate race choices
    races = Race.get_all_races()
    race_choices = [('', 'Select a race (optional)')]
    for race in races:
        race_choices.append((str(race.id), f"{race.name} - {race.date}"))
    form.assigned_races.choices = race_choices
    
    if form.validate_on_submit():
        try:
            # Update horse with all enhanced fields
            horse.name = form.name.data
            horse.age = form.age.data
            horse.sex = form.sex.data
            horse.color = form.color.data
            horse.breed = form.breed.data
            horse.markings = form.markings.data
            
            # Pedigree Information
            horse.sire = form.sire.data
            horse.dam = form.dam.data
            horse.sire_line = form.sire_line.data
            horse.dam_line = form.dam_line.data
            horse.breeding_value = form.breeding_value.data
            
            # Connections
            horse.trainer = form.trainer.data
            horse.jockey = form.jockey.data
            horse.owner = form.owner.data
            horse.breeder = form.breeder.data
            horse.stable = form.stable.data
            
            # Physical Attributes
            horse.weight = form.weight.data
            horse.body_condition = form.body_condition.data
            horse.conformation_score = form.conformation_score.data
            horse.temperament = form.temperament.data
            
            # Performance Analytics
            horse.speed_rating = form.speed_rating.data
            horse.class_rating = form.class_rating.data
            horse.fitness_level = form.fitness_level.data
            horse.form_rating = form.form_rating.data
            horse.consistency_rating = form.consistency_rating.data
            
            # Training & Fitness
            horse.training_intensity = form.training_intensity.data
            horse.last_workout_date = form.last_workout_date.data
            horse.workout_quality = form.workout_quality.data
            horse.recovery_time = form.recovery_time.data
            horse.injury_history = form.injury_history.data
            
            # Behavioral & Racing Style
            horse.racing_style = form.racing_style.data
            horse.preferred_distance = form.preferred_distance.data
            horse.track_preference = form.track_preference.data
            horse.equipment_used = form.equipment_used.data
            horse.medication_notes = form.medication_notes.data
            
            # Financial Information
            horse.purchase_price = form.purchase_price.data
            horse.current_value = form.current_value.data
            horse.insurance_value = form.insurance_value.data
            horse.stud_fee = form.stud_fee.data
            horse.syndication_details = form.syndication_details.data
            
            # Additional Notes
            horse.additional_notes = form.additional_notes.data
            
            # Save the updated horse
            horse.save()
            
            # Handle race assignment if selected
            if form.assigned_races.data:
                try:
                    race = Race.get_race_by_id(int(form.assigned_races.data))
                    if race:
                        race.add_horse(horse.id)
                        flash(f'Horse updated successfully and assigned to race "{race.name}"!', 'success')
                    else:
                        flash('Horse updated successfully, but selected race not found.', 'warning')
                except Exception as e:
                    flash(f'Horse updated successfully, but error assigning to race: {str(e)}', 'warning')
            else:
                flash('Horse updated successfully with enhanced data!', 'success')
            
            return redirect(url_for('races'))
        except Exception as e:
            flash(f'Error updating horse: {str(e)}', 'error')
    
    return render_template('edit_horse.html', form=form, horse=horse)

@app.route('/edit_race/<int:race_id>', methods=['GET', 'POST'])
@login_required
@debug_route
@performance_monitor
@catch_exceptions
def edit_race(race_id):
    """Edit an existing race with enhanced data"""
    import logging
    logger = logging.getLogger('debug')
    logger.info(f"Starting edit_race for race_id: {race_id}")
    
    try:
        race = Race.get_race_by_id(race_id)
        logger.info(f"Retrieved race: {race.name if race else 'None'}")
        if not race:
            flash('Race not found.', 'error')
            return redirect(url_for('races'))
    except Exception as e:
        logger.error(f"Error retrieving race: {str(e)}")
        flash('Error retrieving race.', 'error')
        return redirect(url_for('races'))
    
    try:
        logger.info("Creating RaceForm instance with race data")
        form = RaceForm(obj=race)
        logger.info("RaceForm created and populated successfully")
    except Exception as e:
        logger.error(f"Error creating/populating RaceForm: {str(e)}")
        logger.error(f"Race data: {race.__dict__ if race else 'None'}")
        flash('Error creating form.', 'error')
        return redirect(url_for('races'))
    
    # Populate horse choices
    horses = Horse.get_all_horses()
    horse_choices = [('', 'Select a horse (optional)')]
    for horse in horses:
        horse_choices.append((str(horse.id), f"{horse.name} - {horse.age}yo {horse.sex}"))
    form.assigned_horses.choices = horse_choices
    
    if form.validate_on_submit():
        try:
            # Update race with all enhanced fields
            # Basic Race Information
            race.name = form.name.data
            race.date = form.date.data
            race.track = form.location.data
            race.distance = float(form.distance.data) if form.distance.data else 0.0
            race.prize_money = float(form.purse.data) if form.purse.data else 0.0
            race.description = form.description.data
            
            # Weather Conditions
            race.temperature = form.temperature.data
            race.humidity = form.humidity.data
            race.wind_speed = form.wind_speed.data
            race.wind_direction = form.wind_direction.data
            race.weather_description = form.weather_description.data
            race.visibility = form.visibility.data
            
            # Track Conditions
            race.track_condition = form.track_condition.data
            race.surface_type = form.surface_type.data
            race.rail_position = form.rail_position.data
            race.track_bias = form.track_bias.data
            race.track_maintenance = form.track_maintenance.data
            
            # Field Analysis
            race.field_size = form.field_size.data
            race.field_quality = form.field_quality.data
            race.pace_scenario = form.pace_scenario.data
            race.competitive_balance = form.competitive_balance.data
            race.speed_figures_range = form.speed_figures_range.data
            
            # Betting Information
            race.total_pool = form.total_pool.data
            race.win_pool = form.win_pool.data
            race.exacta_pool = form.exacta_pool.data
            race.trifecta_pool = form.trifecta_pool.data
            race.superfecta_pool = form.superfecta_pool.data
            race.morning_line_favorite = form.morning_line_favorite.data
            
            # Race Conditions
            race.age_restrictions = form.age_restrictions.data
            race.sex_restrictions = form.sex_restrictions.data
            race.weight_conditions = form.weight_conditions.data
            race.claiming_price = form.claiming_price.data
            race.race_grade = form.race_grade.data
            
            # Historical Data
            race.track_record = form.track_record.data
            race.average_winning_time = form.average_winning_time.data
            race.course_record_holder = form.course_record_holder.data
            race.similar_race_results = form.similar_race_results.data
            race.trainer_jockey_stats = form.trainer_jockey_stats.data
            
            # Media Coverage
            race.tv_coverage = form.tv_coverage.data
            race.streaming_available = form.streaming_available.data
            race.featured_race = form.featured_race.data
            
            # Legacy fields for compatibility
            race.surface = form.surface_type.data
            race.weather = form.weather_description.data
            
            # Save the updated race
            race.save()
            
            # Handle horse assignment if selected
            if form.assigned_horses.data:
                try:
                    horse = Horse.get_by_id(int(form.assigned_horses.data))
                    if horse:
                        race.add_horse(horse.id)
                        flash(f'Race updated successfully and horse "{horse.name}" assigned!', 'success')
                    else:
                        flash('Race updated successfully, but selected horse not found.', 'warning')
                except Exception as e:
                    flash(f'Race updated successfully, but error assigning horse: {str(e)}', 'warning')
            else:
                flash('Race updated successfully with enhanced data!', 'success')
            
            return redirect(url_for('races'))
        except Exception as e:
            flash(f'Error updating race: {str(e)}', 'error')
    
    # Convert string date to datetime object for template rendering
    if hasattr(race, 'date') and isinstance(race.date, str):
        try:
            from datetime import datetime
            race.date = datetime.strptime(race.date, '%Y-%m-%d')
        except (ValueError, TypeError):
            # If date parsing fails, use current date
            race.date = datetime.now()
    
    return render_template('edit_race.html', form=form, race=race)

@app.route('/history')
@login_required
def prediction_history():
    """Show prediction history"""
    predictions = Prediction.get_all_predictions()
    return render_template('history.html', predictions=predictions)

@app.route('/stats')
def statistics():
    """Display application statistics"""
    stats = predictor.get_performance_stats()
    all_races = Race.get_all_races()
    race_stats = {
        'total_races': len(all_races),
        'completed_races': len([r for r in all_races if hasattr(r, 'status') and r.status == 'completed']),
        'upcoming_races': len(Race.get_upcoming_races()),
        'total_horses': len(Horse.get_all_horses()),
        'total_predictions': len(Prediction.get_all_predictions())
    }
    return render_template('stats.html', stats=stats, race_stats=race_stats)

# API Integration Routes
@app.route('/api/import')
def api_import():
    """Display API import interface"""
    providers = api_service.get_available_providers()
    return render_template('api_import.html', providers=providers)

@app.route('/api/fetch_races', methods=['POST'])
def fetch_races():
    """Fetch races from API without importing"""
    provider = request.form.get('provider', 'mock')
    days_ahead = int(request.form.get('days_ahead', 7))
    
    try:
        races = api_service.fetch_upcoming_races(provider, days_ahead)
        races_data = []
        
        for race in races:
            races_data.append({
                'id': race.race_id,
                'name': race.name,
                'date': race.date.strftime('%Y-%m-%d %H:%M'),
                'location': race.location,
                'distance': race.distance,
                'track_condition': race.track_condition,
                'purse': race.purse,
                'horses_count': len(race.horses)
            })
        
        return jsonify({
            'success': True,
            'races': races_data,
            'count': len(races_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/import_races', methods=['POST'])
@login_required
def import_races():
    """Import races from API to database"""
    provider = request.form.get('provider', 'mock')
    days_ahead = int(request.form.get('days_ahead', 7))
    update_existing = request.form.get('update_existing') == 'on'
    
    try:
        stats = api_service.bulk_import_races(provider, days_ahead, update_existing)
        
        if stats['imported'] > 0 or stats['updated'] > 0:
            flash(f"Successfully imported {stats['imported']} races and updated {stats['updated']} races", 'success')
        
        if stats['errors'] > 0:
            flash(f"Encountered {stats['errors']} errors during import", 'warning')
        
        if stats['skipped'] > 0:
            flash(f"Skipped {stats['skipped']} existing races", 'info')
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        flash(f"Error importing races: {str(e)}", 'error')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/providers')
def api_providers():
    """Get available API providers"""
    providers = api_service.get_available_providers()
    return jsonify(providers)

@app.route('/api/test_connection/<provider>')
def test_api_connection(provider):
    """Test connection to API provider"""
    try:
        # Try to fetch a small number of races to test connection
        races = api_service.fetch_upcoming_races(provider, 1)
        
        return jsonify({
            'success': True,
            'provider': provider,
            'races_found': len(races),
            'message': f"Successfully connected to {provider}"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'provider': provider,
            'error': str(e)
        }), 500

@app.route('/api/train_model', methods=['POST'])
@login_required
def train_model():
    """Train the prediction model"""
    try:
        # Get training parameters from request
        data = request.get_json() or {}
        
        # Start training
        result = predictor.train_model()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Training failed due to an error'
        }), 500

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get information about the current model"""
    try:
        info = predictor.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting model info: {str(e)}'
        }), 500

@app.route('/api/model/backups', methods=['GET'])
def list_model_backups():
    """List available model backups"""
    try:
        backups = predictor.list_model_backups()
        return jsonify({
            'success': True,
            'backups': backups
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error listing backups: {str(e)}'
        }), 500

@app.route('/api/model/restore', methods=['POST'])
@login_required
def restore_model_backup():
    """Restore a model from backup"""
    try:
        data = request.get_json()
        if not data or 'timestamp' not in data:
            return jsonify({
                'success': False,
                'message': 'Timestamp is required'
            }), 400
        
        success = predictor.restore_backup(data['timestamp'])
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model restored successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to restore model'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error restoring backup: {str(e)}'
        }), 500

@app.route('/api/model/load', methods=['POST'])
@login_required
def load_model():
    """Load a previously trained model"""
    try:
        success = predictor.load_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model loaded successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No saved model found or failed to load'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error loading model: {str(e)}'
        }), 500

@app.route('/api/model_status')
def model_status():
    """Get current model training status and performance"""
    try:
        # Load model if not already loaded
        if not predictor.is_trained:
            predictor.load_model()
        
        # Get performance stats
        stats = predictor.get_performance_stats()
        
        # Check AI predictor training status
        ai_trained = False
        ai_model_type = "AI Models (TensorFlow + PyTorch)"
        try:
            ai_trained = predictor.ai_predictor.is_trained
        except:
            ai_trained = False
        
        return jsonify({
            'success': True,
            'is_trained': ai_trained,  # Use AI predictor status
            'traditional_ml_trained': predictor.is_trained,  # Keep traditional ML status
            'model_type': ai_model_type,
            'performance_stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/retrain_model', methods=['POST'])
@login_required
def retrain_model():
    """Retrain the model with fresh data"""
    try:
        # Reset the predictor
        predictor.is_trained = False
        predictor.model = RandomForestRegressor(n_estimators=100, random_state=42)
        predictor.scaler = StandardScaler()
        
        # Retrain
        result = predictor.train_model()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Retraining failed due to an error'
        }), 500

# AI Monitoring Routes
@app.route('/ai_monitor')
@login_required
def ai_monitor():
    """AI prediction monitoring dashboard"""
    return render_template('ai_monitor.html')

@app.route('/api/ai_monitor/jobs', methods=['GET'])
@login_required
def api_ai_monitor_jobs():
    """Get all AI prediction jobs and their status"""
    try:
        status_data = prediction_status._read_status()
        jobs = []
        
        for job_id, job_data in status_data.items():
            jobs.append({
                'job_id': job_id,
                'status': job_data.get('status', 'unknown'),
                'progress': job_data.get('progress', 0),
                'race_id': job_data.get('race_id'),
                'created_at': job_data.get('created_at'),
                'completed_at': job_data.get('completed_at'),
                'error': job_data.get('error'),
                'result': job_data.get('result')
            })
        
        # Sort by created_at descending (newest first)
        jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'jobs': jobs
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai_monitor/clear_completed', methods=['POST'])
@login_required
def api_ai_monitor_clear_completed():
    """Clear completed and failed jobs from the status tracker"""
    try:
        status_data = prediction_status._read_status()
        
        # Keep only jobs that are not completed or failed
        active_jobs = {
            job_id: job_data 
            for job_id, job_data in status_data.items() 
            if job_data.get('status') not in ['completed', 'failed', 'error']
        }
        
        prediction_status._write_status(active_jobs)
        
        return jsonify({
            'success': True,
            'message': 'Completed jobs cleared successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.get_by_username(form.username.data)
        if user and user.check_password(form.password.data):
            if user.is_active:
                login_user(user, remember=form.remember_me.data)
                next_page = request.args.get('next')
                flash(f'Welcome back, {user.username}!', 'success')
                return redirect(next_page) if next_page else redirect(url_for('index'))
            else:
                flash('Your account has been deactivated. Please contact an administrator.', 'error')
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('auth/login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = RegisterForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            role='user'
        )
        user.set_password(form.password.data)
        user.save()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('auth/register.html', form=form)

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    return render_template('auth/profile.html', user=current_user)

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change user password"""
    form = ChangePasswordForm()
    if form.validate_on_submit():
        if current_user.check_password(form.current_password.data):
            current_user.set_password(form.new_password.data)
            current_user.save()
            flash('Your password has been updated.', 'success')
            return redirect(url_for('profile'))
        else:
            flash('Current password is incorrect.', 'error')
    
    return render_template('auth/change_password.html', form=form)

# Admin Routes
@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard - requires admin role"""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.', 'error')
        return redirect(url_for('index'))
    
    users = User.get_all()
    stats = {
        'total_users': len(users),
        'active_users': len([u for u in users if u.is_active]),
        'admin_users': len([u for u in users if u.is_admin]),
        'total_races': len(Race.get_all_races()),
        'total_horses': len(Horse.get_all_horses()),
        'total_predictions': len(Prediction.get_all_predictions())
    }
    
    return render_template('admin/dashboard.html', stats=stats, users=users[:10])

@app.route('/admin/users')
@login_required
def admin_users():
    """User management page"""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.', 'error')
        return redirect(url_for('index'))
    
    users = User.get_all()
    return render_template('admin/users.html', users=users)

@app.route('/admin/user/new', methods=['GET', 'POST'])
@login_required
def admin_create_user():
    """Create new user"""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.', 'error')
        return redirect(url_for('index'))
    
    form = UserManagementForm()
    if form.validate_on_submit():
        # Check if username or email already exists
        if User.get_by_username(form.username.data):
            flash('Username already exists.', 'error')
            return render_template('admin/user_form.html', form=form, action='Create')
        
        if User.get_by_email(form.email.data):
            flash('Email already registered.', 'error')
            return render_template('admin/user_form.html', form=form, action='Create')
        
        user = User(
            username=form.username.data,
            email=form.email.data,
            is_admin=(form.role.data == 'admin'),
            is_active=form.is_active.data
        )
        if form.password.data:
            user.set_password(form.password.data)
        else:
            user.set_password('password123')  # Default password
        
        user.save()
        flash(f'User {user.username} created successfully.', 'success')
        return redirect(url_for('admin_users'))
    
    return render_template('admin/user_form.html', form=form, action='Create')

@app.route('/admin/user/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
def admin_edit_user(user_id):
    """Edit existing user"""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.', 'error')
        return redirect(url_for('index'))
    
    user = User.get_by_id(user_id)
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('admin_users'))
    
    form = UserManagementForm(obj=user)
    if form.validate_on_submit():
        # Check if username or email conflicts with other users
        existing_user = User.get_by_username(form.username.data)
        if existing_user and existing_user.id != user.id:
            flash('Username already exists.', 'error')
            return render_template('admin/user_form.html', form=form, action='Edit', user=user)
        
        existing_user = User.get_by_email(form.email.data)
        if existing_user and existing_user.id != user.id:
            flash('Email already registered.', 'error')
            return render_template('admin/user_form.html', form=form, action='Edit', user=user)
        
        user.username = form.username.data
        user.email = form.email.data
        user.is_admin = (form.role.data == 'admin')
        user.is_active = form.is_active.data
        
        if form.password.data:
            user.set_password(form.password.data)
        
        user.save()
        flash(f'User {user.username} updated successfully.', 'success')
        return redirect(url_for('admin_users'))
    
    return render_template('admin/user_form.html', form=form, action='Edit', user=user)

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    """Delete user"""
    if not current_user.is_admin:
        flash('Access denied. Administrator privileges required.', 'error')
        return redirect(url_for('index'))
    
    user = User.get_by_id(user_id)
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('admin_users'))
    
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'error')
        return redirect(url_for('admin_users'))
    
    username = user.username
    user.delete()
    flash(f'User {username} deleted successfully.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/api_credentials')
@login_required
def admin_api_credentials():
    """List all API credentials"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    credentials = APICredentials.get_all()
    return render_template('admin/api_credentials.html', credentials=credentials)


@app.route('/admin/api_credentials/new', methods=['GET', 'POST'])
@login_required
def admin_create_api_credentials():
    """Create new API credentials"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    form = APICredentialsForm()
    if form.validate_on_submit():
        try:
            # Determine the provider name
            provider_name = form.provider.data
            if provider_name == 'custom' and form.custom_provider_name.data:
                provider_name = form.custom_provider_name.data
            
            credentials = APICredentials(
                provider=provider_name,
                api_key=form.api_key.data if form.api_key.data else None,
                api_secret=form.api_secret.data if form.api_secret.data else None,
                username=form.username.data if form.username.data else None,
                password=form.password.data if form.password.data else None,
                base_url=form.base_url.data if form.base_url.data else None,
                description=form.description.data if form.description.data else None,
                is_active=form.is_active.data
            )
            credentials.save()
            # Refresh API manager with new credentials
            api_service.update_api_manager_with_db_credentials()
            flash(f'API credentials for {form.provider.data} created successfully.', 'success')
            return redirect(url_for('admin_api_credentials'))
        except Exception as e:
            flash(f'Error creating API credentials: {str(e)}', 'error')
    
    return render_template('admin/api_credentials_form.html', form=form, title='Add API Credentials')


@app.route('/admin/api_credentials/<int:credential_id>/edit', methods=['GET', 'POST'])
@login_required
def admin_edit_api_credentials(credential_id):
    """Edit existing API credentials"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    credentials = APICredentials.get_by_id(credential_id)
    if not credentials:
        flash('API credentials not found.', 'error')
        return redirect(url_for('admin_api_credentials'))
    
    form = APICredentialsForm(obj=credentials)
    
    # Set the provider field based on existing credentials
    if request.method == 'GET':
        # Check if the provider is one of the predefined ones
        predefined_providers = ['theracingapi', 'odds_api', 'rapid_api', 'sample', 'mock']
        if credentials.provider in predefined_providers:
            form.provider.data = credentials.provider
        else:
            form.provider.data = 'custom'
            form.custom_provider_name.data = credentials.provider
    
    if form.validate_on_submit():
        try:
            # Determine the provider name
            provider_name = form.provider.data
            if provider_name == 'custom' and form.custom_provider_name.data:
                provider_name = form.custom_provider_name.data
            
            credentials.provider = provider_name
            credentials.api_key = form.api_key.data if form.api_key.data else None
            credentials.api_secret = form.api_secret.data if form.api_secret.data else None
            credentials.username = form.username.data if form.username.data else None
            credentials.password = form.password.data if form.password.data else None
            credentials.base_url = form.base_url.data if form.base_url.data else None
            credentials.description = form.description.data if form.description.data else None
            credentials.is_active = form.is_active.data
            credentials.save()
            # Refresh API manager with updated credentials
            api_service.update_api_manager_with_db_credentials()
            flash(f'API credentials for {form.provider.data} updated successfully.', 'success')
            return redirect(url_for('admin_api_credentials'))
        except Exception as e:
            flash(f'Error updating API credentials: {str(e)}', 'error')
    
    return render_template('admin/api_credentials_form.html', form=form, credentials=credentials, title='Edit API Credentials')


@app.route('/admin/api_credentials/<int:credential_id>/delete', methods=['POST'])
@login_required
def admin_delete_api_credentials(credential_id):
    """Delete API credentials"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    credentials = APICredentials.get_by_id(credential_id)
    if not credentials:
        flash('API credentials not found.', 'error')
        return redirect(url_for('admin_api_credentials'))
    
    provider_name = credentials.provider
    credentials.delete()
    # Refresh API manager after deleting credentials
    api_service.update_api_manager_with_db_credentials()
    flash(f'API credentials for {provider_name} deleted successfully.', 'success')
    return redirect(url_for('admin_api_credentials'))


@app.route('/admin/api_credentials/<int:credential_id>/test', methods=['POST'])
@login_required
def admin_test_api_credentials(credential_id):
    """Test API credentials connection"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    credentials = APICredentials.get_by_id(credential_id)
    if not credentials:
        flash('API credentials not found.', 'error')
        return redirect(url_for('admin_api_credentials'))
    
    try:
        # Test the connection using the API service
        test_result = api_service.test_connection_with_credentials(credentials)
        if test_result.get('success'):
            flash(f'Connection to {credentials.provider} successful!', 'success')
        else:
            flash(f'Connection to {credentials.provider} failed: {test_result.get("error", "Unknown error")}', 'error')
    except Exception as e:
        flash(f'Error testing connection: {str(e)}', 'error')
    
    return redirect(url_for('admin_api_credentials'))

if __name__ == '__main__':
    with app.app_context():
        # Create admin user if it doesn't exist
        User.create_admin_user('admin', 'admin@example.com', 'admin123')
    
    # Production configuration
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 8000))
    
    app.run(debug=debug_mode, host=host, port=port)