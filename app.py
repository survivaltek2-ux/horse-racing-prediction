from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
from models.horse import Horse
from models.race import Race
from models.prediction import Prediction
from utils.data_processor import DataProcessor
from utils.predictor import Predictor
from forms import RaceForm, HorseForm, PredictionForm, RaceResultForm, AddHorseToRaceForm
from services.api_service import api_service
from config.api_config import APIConfig
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize data processor and predictor
data_processor = DataProcessor()
predictor = Predictor()

# Initialize API service
api_service.init_app(app)

@app.route('/')
def index():
    """Render the home page"""
    # Get some basic statistics for the dashboard
    total_races = len(Race.get_all_races())
    total_horses = len(Horse.get_all_horses())
    total_predictions = len(Prediction.get_all_predictions())
    upcoming_races = len(Race.get_upcoming_races())
    
    stats = {
        'total_races': total_races,
        'total_horses': total_horses,
        'total_predictions': total_predictions,
        'upcoming_races': upcoming_races
    }
    
    return render_template('index.html', stats=stats)

@app.route('/races')
def races():
    """Display list of upcoming races"""
    race_list = Race.get_upcoming_races()
    return render_template('races.html', races=race_list)

@app.route('/race/<int:race_id>')
def race_details(race_id):
    """Display details for a specific race"""
    race = Race.get_race_by_id(race_id)
    horses = race.get_horses()
    return render_template('race_details.html', race=race, horses=horses)

@app.route('/predict/<int:race_id>', methods=['GET', 'POST'])
def predict(race_id):
    """Generate and display predictions for a race"""
    race = Race.get_race_by_id(race_id)
    form = PredictionForm()
    
    if form.validate_on_submit():
        # Generate prediction using selected algorithm
        algorithm = form.algorithm.data
        prediction = predictor.predict_race(race, {'algorithm': algorithm})
        return render_template('prediction_results.html', race=race, prediction=prediction)
    
    # GET request - show prediction form
    return render_template('predict.html', race=race, form=form)

@app.route('/add_race', methods=['GET', 'POST'])
def add_race():
    """Add a new race"""
    form = RaceForm()
    
    if form.validate_on_submit():
        # Create new race from form data
        race_data = {
            'name': form.name.data,
            'date': form.date.data.isoformat(),
            'location': form.location.data,
            'distance': form.distance.data,
            'track_condition': form.track_condition.data,
            'purse': form.purse.data,
            'description': form.description.data
        }
        race = Race.create_race(race_data)
        flash('Race created successfully!', 'success')
        return redirect(url_for('races'))
    
    return render_template('add_race.html', form=form)

@app.route('/add_horse', methods=['GET', 'POST'])
def add_horse():
    """Add a new horse"""
    form = HorseForm()
    
    if form.validate_on_submit():
        # Create new horse from form data
        horse_data = {
            'name': form.name.data,
            'age': form.age.data,
            'breed': form.breed.data,
            'color': form.color.data,
            'jockey': form.jockey.data,
            'trainer': form.trainer.data,
            'owner': form.owner.data,
            'weight': form.weight.data
        }
        horse = Horse.create_horse(horse_data)
        flash('Horse added successfully!', 'success')
        return redirect(url_for('races'))
    
    return render_template('add_horse.html', form=form)

@app.route('/history')
def prediction_history():
    """Display prediction history"""
    predictions = Prediction.get_all_predictions()
    return render_template('history.html', predictions=predictions)

@app.route('/stats')
def statistics():
    """Display application statistics"""
    stats = predictor.get_performance_stats()
    race_stats = {
        'total_races': len(Race.get_all_races()),
        'completed_races': len([r for r in Race.get_all_races() if r.status == 'completed']),
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

if __name__ == '__main__':
    app.run(debug=True)