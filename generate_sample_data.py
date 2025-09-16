#!/usr/bin/env python3
"""
Comprehensive Sample Data Generator for Horse Racing Prediction App
Generates realistic horses, races, and prediction data for testing and development.
"""

import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from models.sqlalchemy_models import db, Horse, Race, Prediction, User
from config.database_config import create_tables

# Sample data pools for realistic generation
HORSE_NAMES = [
    "Thunder Strike", "Lightning Bolt", "Midnight Express", "Golden Arrow", "Silver Bullet",
    "Fire Storm", "Wind Walker", "Star Dancer", "Royal Thunder", "Diamond Dust",
    "Blazing Glory", "Storm Chaser", "Mystic Moon", "Crimson Flash", "Azure Dream",
    "Phantom Rider", "Eclipse Runner", "Sunset Warrior", "Morning Glory", "Night Fury",
    "Steel Thunder", "Velvet Touch", "Iron Will", "Gentle Giant", "Swift Arrow",
    "Brave Heart", "Wild Spirit", "Noble Quest", "Majestic Pride", "Regal Force",
    "Cosmic Ray", "Solar Flare", "Lunar Eclipse", "Stellar Wind", "Galaxy Runner",
    "Ocean Wave", "Mountain Peak", "Desert Storm", "Forest Fire", "River Rush",
    "Golden Eagle", "Silver Hawk", "Bronze Tiger", "Platinum Wolf", "Crystal Fox",
    "Emerald Knight", "Ruby Queen", "Sapphire Prince", "Diamond King", "Pearl Maiden",
    "Thunder Cloud", "Lightning Strike", "Storm Front", "Wind Gust", "Rain Drop",
    "Snow Flake", "Ice Crystal", "Fire Spark", "Earth Quake", "Rock Slide",
    "Speed Demon", "Power House", "Grace Note", "Harmony Bell", "Rhythm King",
    "Melody Queen", "Symphony Star", "Jazz Master", "Blues Brother", "Rock Legend",
    "Pop Icon", "Classical Hero", "Opera Singer", "Ballet Dancer", "Tap Master",
    "Swing King", "Waltz Queen", "Tango Fire", "Salsa Heat", "Rumba Romance",
    "Cha Cha Charm", "Foxtrot Flash", "Quickstep Queen", "Slow Dance", "Fast Track",
    "Victory Lane", "Winner's Circle", "Champion's Choice", "Trophy Hunter", "Medal Winner",
    "Gold Rush", "Silver Lining", "Bronze Medal", "Platinum Prize", "Diamond Trophy",
    "Emerald Cup", "Ruby Ring", "Sapphire Star", "Crystal Crown", "Pearl Prize",
    "Thunder Bay", "Lightning Lake", "Storm Sea", "Wind River", "Rain Forest",
    "Snow Mountain", "Ice Valley", "Fire Hill", "Earth Field", "Rock Canyon",
    "Speed Limit", "Power Play", "Grace Period", "Harmony House", "Rhythm Section"
]

JOCKEY_NAMES = [
    "Mike Smith", "John Velazquez", "Javier Castellano", "Joel Rosario", "Irad Ortiz Jr.",
    "Jose Ortiz", "Flavien Prat", "Luis Saez", "Tyler Gaffalione", "Florent Geroux",
    "Ricardo Santana Jr.", "Manny Franco", "Dylan Davis", "Junior Alvarado", "Antonio Fresu",
    "Kendrick Carmouche", "Eric Cancel", "Trevor McCarthy", "Reylu Gutierrez", "David Cohen",
    "Alex Cintron", "Angel Cruz", "Carlos Hernandez", "Edwin Gonzalez", "Feargal Lynch",
    "Gabriel Saez", "Hector Diaz Jr.", "Ivan Carnero", "James Graham", "Kevin Krigger",
    "Luca Panici", "Manuel Aguilar", "Norberto Arroyo Jr.", "Orlando Bocachica", "Pablo Morales",
    "Quin Howey", "Rafael Bejarano", "Samy Camacho", "Tiago Pereira", "Umberto Rispoli",
    "Victor Carrasco", "William Antongeorgi", "Xavier Perez", "Yauheniya Tsimafeyeva", "Zoey Ziegler"
]

TRAINER_NAMES = [
    "Bob Baffert", "Chad Brown", "Todd Pletcher", "Bill Mott", "Steve Asmussen",
    "Mark Casse", "Brad Cox", "John Sadler", "Richard Baltas", "Doug O'Neill",
    "Jerry Hollendorfer", "Tim Yakteen", "Phil D'Amato", "Leonard Powell", "Michael McCarthy",
    "Peter Miller", "Richard Mandella", "Simon Callaghan", "Vladimir Cerin", "Wesley Ward",
    "Aidan Butler", "Brendan Walsh", "Carlos Martin", "Danny Gargan", "Eddie Kenneally",
    "Ferris Allen", "George Weaver", "Horacio DePaz", "Ian Wilkes", "Jorge Delgado",
    "Kelly Breen", "Larry Rivelli", "Mike Maker", "Norm Casse", "Orlando Noda",
    "Paulo Lobo", "Quincy Hamilton", "Robertino Diodoro", "Saffie Joseph Jr.", "Tom Amoss",
    "Uriah St. Lewis", "Victoria Oliver", "Wayne Catalano", "Xavier Aizpuru", "Yates Dooley"
]

TRACK_NAMES = [
    "Churchill Downs", "Belmont Park", "Saratoga", "Santa Anita", "Del Mar",
    "Keeneland", "Gulfstream Park", "Oaklawn Park", "Fair Grounds", "Aqueduct",
    "Woodbine", "Arlington Park", "Monmouth Park", "Pimlico", "Laurel Park",
    "Penn National", "Parx Racing", "Delaware Park", "Charles Town", "Mountaineer",
    "Turfway Park", "Tampa Bay Downs", "Remington Park", "Lone Star Park", "Sam Houston",
    "Golden Gate Fields", "Bay Meadows", "Hollywood Park", "Los Alamitos", "Pleasanton"
]

SURFACES = ["Dirt", "Turf", "Synthetic", "All Weather"]
TRACK_CONDITIONS = ["Fast", "Good", "Firm", "Yielding", "Soft", "Heavy", "Sloppy", "Muddy"]
WEATHER_CONDITIONS = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Overcast", "Sunny", "Windy"]
RACE_CLASSES = ["Maiden", "Allowance", "Claiming", "Stakes", "Graded Stakes", "Listed"]
HORSE_COLORS = ["Bay", "Chestnut", "Brown", "Black", "Gray", "Roan", "Palomino", "Pinto"]
HORSE_SEXES = ["Colt", "Filly", "Gelding", "Mare", "Stallion", "Ridgling"]

def generate_horses(count: int = 200) -> List[Dict[str, Any]]:
    """Generate realistic horse data"""
    horses = []
    used_names = set()
    
    for i in range(count):
        # Ensure unique names
        name = random.choice(HORSE_NAMES)
        counter = 1
        original_name = name
        while name in used_names:
            name = f"{original_name} {counter}"
            counter += 1
        used_names.add(name)
        
        age = random.randint(2, 8)
        runs = random.randint(0, 50)
        wins = random.randint(0, min(runs, 15))
        places = random.randint(wins, min(runs, wins + 10))
        
        # Generate realistic performance metrics
        speed_rating = random.randint(60, 120)
        class_rating = random.randint(1, 10)
        fitness_level = random.randint(1, 10)
        
        # Generate form string (recent race results)
        form_length = min(runs, 10)
        form_results = []
        for _ in range(form_length):
            result = random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                  weights=[15, 12, 10, 8, 6, 5, 4, 3, 2, 1])[0]
            form_results.append(str(result))
        form = "-".join(form_results) if form_results else ""
        
        # Generate workout times (JSON)
        workout_times = []
        for _ in range(random.randint(3, 8)):
            workout_times.append({
                "date": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
                "distance": random.choice(["3f", "4f", "5f", "6f"]),
                "time": f"{random.randint(35, 75)}.{random.randint(10, 99)}",
                "track_condition": random.choice(TRACK_CONDITIONS)
            })
        
        horse = {
            "name": name,
            "age": age,
            "sex": random.choice(HORSE_SEXES),
            "color": random.choice(HORSE_COLORS),
            "sire": f"Sire of {name}",
            "dam": f"Dam of {name}",
            "trainer": random.choice(TRAINER_NAMES),
            "jockey": random.choice(JOCKEY_NAMES),
            "owner": f"Owner of {name}",
            "weight": round(random.uniform(115, 130), 1),
            "form": form,
            "rating": random.randint(60, 120),
            "last_run": datetime.now() - timedelta(days=random.randint(7, 90)),
            "wins": wins,
            "places": places,
            "runs": runs,
            "earnings": round(random.uniform(5000, 500000), 2),
            "speed_rating": speed_rating,
            "class_rating": class_rating,
            "distance_preference": random.choice(["5f-7f", "6f-1m", "1m-1.25m", "1.25m-1.5m", "1.5m+"]),
            "surface_preference": random.choice(SURFACES),
            "track_bias_rating": random.randint(1, 10),
            "days_since_last_race": random.randint(7, 90),
            "fitness_level": fitness_level,
            "workout_times": json.dumps(workout_times),
            "injury_history": json.dumps([])  # Empty for now
        }
        horses.append(horse)
    
    return horses

def generate_races(count: int = 100) -> List[Dict[str, Any]]:
    """Generate realistic race data"""
    races = []
    
    for i in range(count):
        # Generate race date (mix of past, present, and future)
        days_offset = random.randint(-30, 60)
        race_date = datetime.now() + timedelta(days=days_offset)
        
        # Determine race status based on date
        if race_date < datetime.now() - timedelta(days=1):
            status = "completed"
        elif race_date < datetime.now() + timedelta(hours=2):
            status = "running"
        else:
            status = "upcoming"
        
        distance = random.choice([5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0, 10.0, 12.0])
        
        race = {
            "name": f"Race {i+1} - {random.choice(['Sprint', 'Mile', 'Route', 'Stakes', 'Handicap'])}",
            "date": race_date,
            "time": f"{random.randint(12, 18)}:{random.randint(0, 59):02d}",
            "track": random.choice(TRACK_NAMES),
            "distance": distance,
            "surface": random.choice(SURFACES),
            "race_class": random.choice(RACE_CLASSES),
            "prize_money": round(random.uniform(10000, 1000000), 2),
            "weather": random.choice(WEATHER_CONDITIONS),
            "track_condition": random.choice(TRACK_CONDITIONS),
            "status": status,
            "results": json.dumps({}) if status != "completed" else json.dumps({
                "winner": random.randint(1, 12),
                "finishing_order": list(range(1, random.randint(8, 13))),
                "times": [f"{random.randint(60, 150)}.{random.randint(10, 99)}" for _ in range(random.randint(8, 12))]
            })
        }
        races.append(race)
    
    return races

def generate_predictions(horses: List[Dict], races: List[Dict]) -> List[Dict[str, Any]]:
    """Generate realistic prediction data"""
    predictions = []
    
    for race_idx, race in enumerate(races):
        # Select random horses for this race (6-12 horses per race)
        num_horses = random.randint(6, 12)
        race_horses = random.sample(horses, min(num_horses, len(horses)))
        
        for position, horse in enumerate(race_horses, 1):
            # Generate realistic prediction metrics
            confidence = round(random.uniform(0.1, 0.95), 3)
            odds = round(random.uniform(1.5, 50.0), 2)
            
            # Generate prediction factors
            factors = {
                "form_score": round(random.uniform(0.1, 1.0), 3),
                "speed_rating": round(random.uniform(0.1, 1.0), 3),
                "class_rating": round(random.uniform(0.1, 1.0), 3),
                "jockey_rating": round(random.uniform(0.1, 1.0), 3),
                "trainer_rating": round(random.uniform(0.1, 1.0), 3),
                "track_suitability": round(random.uniform(0.1, 1.0), 3),
                "distance_suitability": round(random.uniform(0.1, 1.0), 3),
                "recent_performance": round(random.uniform(0.1, 1.0), 3)
            }
            
            prediction = {
                "race_id": race_idx + 1,  # Will be updated when races are saved
                "horse_id": horses.index(horse) + 1,  # Will be updated when horses are saved
                "predicted_position": position,
                "confidence": confidence,
                "odds": odds,
                "factors": json.dumps(factors),
                "model_version": random.choice(["v1.0", "v1.1", "v1.2", "heuristic"])
            }
            predictions.append(prediction)
    
    return predictions

def save_to_database(horses: List[Dict], races: List[Dict], predictions: List[Dict]):
    """Save all generated data to the database"""
    with app.app_context():
        try:
            # Clear existing data (optional - comment out if you want to keep existing data)
            print("Clearing existing data...")
            db.session.query(Prediction).delete()
            db.session.query(Horse).delete()
            db.session.query(Race).delete()
            db.session.commit()
            
            print(f"Saving {len(horses)} horses...")
            horse_objects = []
            for horse_data in horses:
                horse = Horse(**horse_data)
                db.session.add(horse)
                horse_objects.append(horse)
            db.session.commit()
            
            print(f"Saving {len(races)} races...")
            race_objects = []
            for race_data in races:
                race = Race(**race_data)
                db.session.add(race)
                race_objects.append(race)
            db.session.commit()
            
            # Assign horses to races randomly
            print("Assigning horses to races...")
            for race in race_objects:
                num_horses = random.randint(6, 12)
                race_horses = random.sample(horse_objects, min(num_horses, len(horse_objects)))
                race.horses = race_horses  # Use assignment instead of append to avoid duplicates
            db.session.commit()
            
            # Generate predictions based on actual race-horse assignments
            print("Generating predictions based on race assignments...")
            prediction_count = 0
            for race in race_objects:
                for position, horse in enumerate(race.horses, 1):
                    # Generate realistic prediction metrics
                    confidence = round(random.uniform(0.1, 0.95), 3)
                    odds = round(random.uniform(1.5, 50.0), 2)
                    
                    # Generate prediction factors
                    factors = {
                        "form_score": round(random.uniform(0.1, 1.0), 3),
                        "speed_rating": round(random.uniform(0.1, 1.0), 3),
                        "class_rating": round(random.uniform(0.1, 1.0), 3),
                        "jockey_rating": round(random.uniform(0.1, 1.0), 3),
                        "trainer_rating": round(random.uniform(0.1, 1.0), 3),
                        "track_suitability": round(random.uniform(0.1, 1.0), 3),
                        "distance_suitability": round(random.uniform(0.1, 1.0), 3),
                        "recent_performance": round(random.uniform(0.1, 1.0), 3)
                    }
                    
                    prediction = Prediction(
                        race_id=race.id,
                        horse_id=horse.id,
                        predicted_position=position,
                        confidence=confidence,
                        odds=odds,
                        factors=json.dumps(factors),
                        model_version=random.choice(["v1.0", "v1.1", "v1.2", "heuristic"])
                    )
                    db.session.add(prediction)
                    prediction_count += 1
            
            db.session.commit()
            print(f"Generated {prediction_count} predictions")
            
            print("Sample data generation completed successfully!")
            print(f"Generated: {len(horses)} horses, {len(races)} races, {len(predictions)} predictions")
            
        except Exception as e:
            print(f"Error saving to database: {e}")
            db.session.rollback()
            raise

def datetime_serializer(obj):
    """JSON serializer for datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def save_to_json_files(horses: List[Dict], races: List[Dict], predictions: List[Dict]):
    """Save generated data to JSON files for backup"""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    with open(f"{data_dir}/sample_horses.json", "w") as f:
        json.dump(horses, f, indent=2, default=datetime_serializer)
    
    with open(f"{data_dir}/sample_races.json", "w") as f:
        json.dump(races, f, indent=2, default=datetime_serializer)
    
    with open(f"{data_dir}/sample_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2, default=datetime_serializer)
    
    print(f"JSON files saved to {data_dir}/ directory")

def main():
    """Main function to generate and save sample data"""
    print("Starting sample data generation...")
    
    # Generate data
    print("Generating horses...")
    horses = generate_horses(200)
    
    print("Generating races...")
    races = generate_races(100)
    
    print("Generating predictions...")
    predictions = generate_predictions(horses, races)
    
    # Save to JSON files
    print("Saving to JSON files...")
    save_to_json_files(horses, races, predictions)
    
    # Save to database
    print("Saving to database...")
    save_to_database(horses, races, predictions)
    
    print("Sample data generation completed!")

if __name__ == "__main__":
    main()