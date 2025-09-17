#!/usr/bin/env python3
"""
Check what races exist in the database
"""

from config.database_config import init_database, db
from models.race import Race
from flask import Flask

app = Flask(__name__)
init_database(app)

with app.app_context():
    try:
        races = Race.query.all()
        print(f"Found {len(races)} races in database:")
        for race in races[:10]:  # Show first 10 races
            print(f"  Race ID: {race.id}, Name: {race.name}, Date: {race.date}")
        
        if len(races) > 10:
            print(f"  ... and {len(races) - 10} more races")
            
    except Exception as e:
        print(f"Error querying races: {e}")
        print("Database might not be properly initialized")