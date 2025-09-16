#!/usr/bin/env python3
"""
Sample Data Import Script
Imports realistic horse racing data into the SQLite database.
"""

import sys
import os
import random
import json
from datetime import datetime, timedelta
from decimal import Decimal

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database_config import init_database, db
from models.sqlalchemy_models import Horse, Race, Prediction, User
from flask import Flask

# Sample data
HORSE_NAMES = [
    "Thunder Strike", "Lightning Bolt", "Midnight Express", "Golden Arrow", "Storm Chaser",
    "Fire Spirit", "Wind Walker", "Star Dancer", "Royal Thunder", "Silver Bullet",
    "Black Beauty", "Red Lightning", "Blue Moon", "White Storm", "Green Flash",
    "Crimson Tide", "Golden Eagle", "Silver Star", "Diamond Dust", "Ruby Runner",
    "Emerald Dream", "Sapphire Speed", "Amber Arrow", "Crystal Clear", "Platinum Power",
    "Iron Will", "Steel Storm", "Copper Coin", "Bronze Bullet", "Titanium Tiger",
    "Velvet Voice", "Silk Road", "Cotton Cloud", "Satin Smooth", "Lace Lightning",
    "Marble Magic", "Granite Grace", "Quartz Queen", "Jade Jumper", "Pearl Princess"
]

JOCKEY_NAMES = [
    "Mike Smith", "John Velazquez", "Javier Castellano", "Joel Rosario", "Irad Ortiz Jr.",
    "Jose Ortiz", "Flavien Prat", "Luis Saez", "Tyler Gaffalione", "Ricardo Santana Jr.",
    "Florent Geroux", "Manny Franco", "Dylan Davis", "Junior Alvarado", "Antonio Fresu",
    "Kendrick Carmouche", "Eric Cancel", "Jorge Vargas Jr.", "Reylu Gutierrez", "David Cohen"
]

TRAINER_NAMES = [
    "Bob Baffert", "Todd Pletcher", "Chad Brown", "Steve Asmussen", "Bill Mott",
    "Mark Casse", "Brad Cox", "John Sadler", "Richard Baltas", "Doug O'Neill",
    "Jerry Hollendorfer", "Tim Yakteen", "Phil D'Amato", "Leonard Powell", "Peter Miller",
    "Michael McCarthy", "Jeff Mullins", "Dan Blacker", "Craig Lewis", "Vladimir Cerin"
]

TRACK_NAMES = [
    "Churchill Downs", "Belmont Park", "Santa Anita", "Gulfstream Park", "Keeneland",
    "Saratoga", "Del Mar", "Oaklawn Park", "Fair Grounds", "Aqueduct",
    "Woodbine", "Arlington Park", "Pimlico", "Laurel Park", "Tampa Bay Downs"
]

RACE_TYPES = [
    "Maiden Special Weight", "Allowance", "Claiming", "Stakes", "Graded Stakes",
    "Handicap", "Starter Allowance", "Optional Claiming", "Maiden Claiming"
]

TRACK_CONDITIONS = ["Fast", "Good", "Muddy", "Sloppy", "Yielding", "Soft"]

def create_sample_horses(num_horses=40):
    """Create sample horses with realistic data."""
    horses = []
    
    for i in range(num_horses):
        horse = Horse(
            name=HORSE_NAMES[i % len(HORSE_NAMES)] + f" #{i+1}" if i >= len(HORSE_NAMES) else HORSE_NAMES[i],
            age=random.randint(2, 8),
            sex=random.choice(["M", "F", "G", "C"]),  # Male, Female, Gelding, Colt
            color=random.choice(["Bay", "Chestnut", "Black", "Gray", "Brown", "Roan"]),
            sire=f"Sire {random.randint(1, 100)}",
            dam=f"Dam {random.randint(1, 100)}",
            trainer=random.choice(TRAINER_NAMES),
            jockey=random.choice(JOCKEY_NAMES),
            owner=f"Owner {random.randint(1, 50)}",
            weight=round(random.uniform(115.0, 126.0), 1),
            form=f"{random.randint(1,3)}-{random.randint(1,5)}-{random.randint(1,8)}",
            rating=random.randint(75, 105),
            last_run=datetime.now() - timedelta(days=random.randint(7, 90)),
            wins=random.randint(0, 15),
            places=random.randint(0, 20),
            runs=random.randint(5, 50),
            earnings=round(random.uniform(10000.0, 500000.0), 2)
        )
        horses.append(horse)
    
    return horses

def create_sample_races(horses, num_races=15):
    """Create sample races with horses."""
    races = []
    
    for i in range(num_races):
        # Create race
        race_date = datetime.now() + timedelta(days=random.randint(-30, 30))
        race = Race(
            name=f"Race {i+1} - {random.choice(RACE_TYPES)}",
            date=race_date,
            time=f"{random.randint(10, 18):02d}:{random.randint(0, 59):02d}",
            track=random.choice(TRACK_NAMES),
            distance=float(random.choice([1000, 1200, 1400, 1600, 1800, 2000, 2400])),
            surface=random.choice(["Dirt", "Turf", "Synthetic"]),
            race_class=random.choice(["Class 1", "Class 2", "Class 3", "Class 4", "Stakes"]),
            prize_money=float(random.randint(25000, 500000)),
            weather=random.choice(["Clear", "Cloudy", "Light Rain", "Overcast"]),
            track_condition=random.choice(TRACK_CONDITIONS),
            status="upcoming" if race_date > datetime.now() else "completed"
        )
        
        # Add random horses to this race (4-12 horses per race)
        race_horses_list = random.sample(horses, random.randint(4, min(12, len(horses))))
        for horse in race_horses_list:
            race.horses.append(horse)
        
        # Set results if the race is in the past
        if race_date < datetime.now():
            race.status = "completed"
            # Create simple results JSON
            results = {
                "winner": random.choice(race_horses_list).name if race_horses_list else None,
                "winning_time": round(random.uniform(60.0, 150.0), 2),
                "positions": [horse.name for horse in random.sample(race_horses_list, len(race_horses_list))]
            }
            race.results = json.dumps(results)
        
        races.append(race)
    
    return races

def create_sample_predictions(races):
    """Create sample predictions for races."""
    predictions = []
    
    for race in races:
        if race.horses and random.choice([True, False]):  # 50% chance to have a prediction
            predicted_horse = random.choice(race.horses)
            
            prediction = Prediction(
                race_id=race.id,
                horse_id=predicted_horse.id,
                predicted_position=random.randint(1, 3),  # Predict to finish in top 3
                confidence=round(random.uniform(0.6, 0.95), 2),
                odds=round(random.uniform(2.0, 20.0), 1),
                model_version="sample_v1.0"
            )
            
            # Add prediction factors as JSON
            factors = {
                "recent_form": random.choice(['excellent', 'good', 'average']),
                "track_conditions": random.choice(['favorable', 'neutral', 'challenging']),
                "jockey_performance": random.choice(['strong', 'average', 'weak']),
                "trainer_stats": random.choice(['top_tier', 'mid_tier', 'developing']),
                "weight_factor": round(random.uniform(0.8, 1.2), 2),
                "distance_suitability": random.choice(['perfect', 'good', 'questionable'])
            }
            prediction.factors_dict = factors
            
            predictions.append(prediction)
    
    return predictions

def import_sample_data():
    """Main function to import all sample data."""
    
    # Create Flask app context
    app = Flask(__name__)
    init_database(app)
    
    with app.app_context():
        print("🚀 Starting sample data import...")
        
        # Check if data already exists
        existing_horses = Horse.query.count()
        existing_races = Race.query.count()
        
        if existing_horses > 0 or existing_races > 0:
            print(f"⚠️  Database already contains {existing_horses} horses and {existing_races} races.")
            response = input("Do you want to clear existing data and import fresh sample data? (y/N): ")
            if response.lower() != 'y':
                print("❌ Import cancelled.")
                return False
            
            # Clear existing data
            print("🗑️  Clearing existing data...")
            Prediction.query.delete()
            # Clear race-horse relationships
            db.session.execute(db.text("DELETE FROM race_horses"))
            Race.query.delete()
            Horse.query.delete()
            db.session.commit()
            print("✅ Existing data cleared.")
        
        try:
            # Get admin user for predictions
            admin_user = User.query.filter_by(username='admin').first()
            if not admin_user:
                print("❌ Admin user not found. Please create admin user first.")
                return False
            
            # Create and import horses
            print("🐎 Creating sample horses...")
            horses = create_sample_horses(40)
            for horse in horses:
                db.session.add(horse)
            db.session.commit()
            print(f"✅ Imported {len(horses)} horses")
            
            # Refresh horses to get IDs
            horses = Horse.query.all()
            
            # Create and import races
            print("🏁 Creating sample races...")
            races = create_sample_races(horses, 15)
            for race in races:
                db.session.add(race)
            db.session.commit()
            print(f"✅ Imported {len(races)} races")
            
            # Refresh races to get IDs
            races = Race.query.all()
            
            # Create and import predictions
            print("🔮 Creating sample predictions...")
            predictions = create_sample_predictions(races)
            for prediction in predictions:
                db.session.add(prediction)
            db.session.commit()
            print(f"✅ Imported {len(predictions)} predictions")
            
            # Print summary
            print("\n📊 Import Summary:")
            print(f"   🐎 Horses: {Horse.query.count()}")
            print(f"   🏁 Races: {Race.query.count()}")
            print(f"   🔮 Predictions: {Prediction.query.count()}")
            print(f"   👥 Users: {User.query.count()}")
            
            # Print some sample data
            print("\n🎯 Sample Data Preview:")
            sample_race = Race.query.first()
            if sample_race:
                print(f"   📅 Next Race: {sample_race.name} at {sample_race.track}")
                print(f"   🐎 Horses in race: {len(sample_race.horses)}")
            
            sample_horse = Horse.query.first()
            if sample_horse:
                print(f"   🏆 Sample Horse: {sample_horse.name} (Jockey: {sample_horse.jockey})")
            
            print("\n🎉 Sample data import completed successfully!")
            print("🌐 You can now view the data at: http://127.0.0.1:5000")
            
            return True
            
        except Exception as e:
            print(f"❌ Error importing sample data: {e}")
            db.session.rollback()
            return False

if __name__ == '__main__':
    success = import_sample_data()
    if not success:
        sys.exit(1)