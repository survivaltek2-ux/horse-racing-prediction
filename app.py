import os
import warnings

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
from models.sqlalchemy_models import User, APICredentials
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
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

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

@app.route('/add_race', methods=['GET', 'POST'])
@login_required
def add_race():
    """Add a new race"""
    form = RaceForm()
    
    if form.validate_on_submit():
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
    
    if form.validate_on_submit():
        try:
            # Create new horse from form data with all enhanced fields
            horse = Horse(
                # Basic Information
                name=form.name.data,
                age=form.age.data,
                sex=form.sex.data,
                color=form.color.data,
                breed=form.breed.data,
                height=form.height.data,
                markings=form.markings.data,
                
                # Pedigree Information
                sire=form.sire.data,
                dam=form.dam.data,
                sire_line=form.sire_line.data,
                dam_line=form.dam_line.data,
                breeding_value=form.breeding_value.data,
                
                # Connections
                trainer=form.trainer.data,
                jockey=form.jockey.data,
                owner=form.owner.data,
                breeder=form.breeder.data,
                stable=form.stable.data,
                
                # Physical Attributes
                weight=form.weight.data,
                body_condition=form.body_condition.data,
                conformation_score=form.conformation_score.data,
                temperament=form.temperament.data,
                
                # Performance Analytics
                speed_rating=form.speed_rating.data,
                class_rating=form.class_rating.data,
                distance_preference=form.distance_preference.data,
                surface_preference=form.surface_preference.data,
                track_bias_rating=form.track_bias_rating.data,
                pace_style=form.pace_style.data,
                closing_kick=form.closing_kick.data,
                
                # Training & Fitness
                days_since_last_race=form.days_since_last_race.data,
                fitness_level=form.fitness_level.data,
                training_intensity=form.training_intensity.data,
                workout_times=form.workout_times.data,
                injury_history=form.injury_history.data,
                recovery_time=form.recovery_time.data,
                
                # Behavioral & Racing Style
                gate_behavior=form.gate_behavior.data,
                racing_tactics=form.racing_tactics.data,
                equipment_used=form.equipment_used.data,
                medication_notes=form.medication_notes.data,
                
                # Financial Information
                purchase_price=form.purchase_price.data,
                current_value=form.current_value.data,
                insurance_value=form.insurance_value.data,
                stud_fee=form.stud_fee.data
            )
            horse.save()
            flash('Horse added successfully with enhanced data!', 'success')
            return redirect(url_for('races'))
        except Exception as e:
            flash(f'Error adding horse: {str(e)}', 'error')
    
    return render_template('add_horse.html', form=form)

@app.route('/edit_horse/<int:horse_id>', methods=['GET', 'POST'])
@login_required
def edit_horse(horse_id):
    """Edit an existing horse with enhanced data"""
    horse = Horse.get_by_id(horse_id)
    if not horse:
        flash('Horse not found.', 'error')
        return redirect(url_for('races'))
    
    form = HorseForm(obj=horse)
    
    if form.validate_on_submit():
        try:
            # Update horse with all enhanced fields
            horse.name = form.name.data
            horse.age = form.age.data
            horse.sex = form.sex.data
            horse.color = form.color.data
            horse.breed = form.breed.data
            horse.height = form.height.data
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
            flash('Horse updated successfully with enhanced data!', 'success')
            return redirect(url_for('races'))
        except Exception as e:
            flash(f'Error updating horse: {str(e)}', 'error')
    
    return render_template('edit_horse.html', form=form, horse=horse)

@app.route('/edit_race/<int:race_id>', methods=['GET', 'POST'])
@login_required
def edit_race(race_id):
    """Edit an existing race with enhanced data"""
    race = Race.get_race_by_id(race_id)
    if not race:
        flash('Race not found.', 'error')
        return redirect(url_for('races'))
    
    form = RaceForm(obj=race)
    
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
            flash('Race updated successfully with enhanced data!', 'success')
            return redirect(url_for('races'))
        except Exception as e:
            flash(f'Error updating race: {str(e)}', 'error')
    
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
        
        return jsonify({
            'success': True,
            'is_trained': predictor.is_trained,
            'model_type': 'RandomForestRegressor',
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
            role=form.role.data,
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
        user.role = form.role.data
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
    # Create admin user if it doesn't exist
    with app.app_context():
        User.create_admin_user('admin', 'admin@example.com', 'admin123')
    app.run(debug=True, port=8000)