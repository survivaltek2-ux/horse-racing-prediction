#!/usr/bin/env python3
"""
Create a simple test race for testing AI prediction workflow
"""

from config.database_config import init_database, db
from models.sqlalchemy_models import Race, Horse
from flask import Flask
from datetime import datetime, timedelta

app = Flask(__name__)
init_database(app)

with app.app_context():
    try:
        # Check if we already have races
        existing_races = db.session.query(Race).count()
        print(f"Existing races in database: {existing_races}")
        
        if existing_races == 0:
            # Create a simple test race
            test_race = Race(
                name="Test Race 1",
                date=datetime.now() + timedelta(days=1),
                track="Test Track",
                distance=1200,
                prize_money=10000,
                status="upcoming"
            )
            
            db.session.add(test_race)
            db.session.commit()
            
            print(f"✅ Created test race with ID: {test_race.id}")
            
            # Create some test horses for the race
            for i in range(3):
                horse = Horse(
                    name=f"Test Horse {i+1}",
                    age=4,
                    weight=55.0,
                    jockey=f"Test Jockey {i+1}",
                    trainer=f"Test Trainer {i+1}",
                    odds=2.5 + i,
                    race_id=test_race.id
                )
                db.session.add(horse)
            
            db.session.commit()
            print("✅ Created 3 test horses for the race")
        else:
            # Get the first race
            first_race = db.session.query(Race).first()
            print(f"✅ Using existing race with ID: {first_race.id}")
            
    except Exception as e:
        print(f"❌ Error creating test race: {e}")
        db.session.rollback()