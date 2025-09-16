#!/usr/bin/env python3
"""
Script to enhance existing horses and races with additional data fields.
This script will populate the new enhanced fields for all existing entries.
"""

import random
from datetime import datetime, timedelta
from app import app, db
from models.sqlalchemy_models import Horse, Race
from sqlalchemy import text

# Sample data for enhanced fields
BREEDS = ['Thoroughbred', 'Arabian', 'Quarter Horse', 'Standardbred', 'Paint Horse', 'Appaloosa']
COLORS = ['Bay', 'Chestnut', 'Black', 'Gray', 'Brown', 'Palomino', 'Pinto', 'Roan']
SIRE_LINES = ['Northern Dancer', 'Mr. Prospector', 'Storm Cat', 'A.P. Indy', 'Seattle Slew', 'Secretariat']
DAM_LINES = ['Somethingroyal', 'Courtly Dee', 'Almahmoud', 'La Troienne', 'Rough Shod II', 'Pocahontas']
TRAINERS = ['Bob Baffert', 'Todd Pletcher', 'Chad Brown', 'Steve Asmussen', 'Bill Mott', 'Mark Casse']
JOCKEYS = ['Irad Ortiz Jr.', 'Joel Rosario', 'John Velazquez', 'Javier Castellano', 'Mike Smith', 'Flavien Prat']
OWNERS = ['Godolphin', 'Juddmonte Farms', 'WinStar Farm', 'Coolmore', 'Claiborne Farm', 'Three Chimneys Farm']

TRACK_SURFACES = ['Dirt', 'Turf', 'Synthetic', 'All Weather']
WEATHER_CONDITIONS = ['Clear', 'Partly Cloudy', 'Overcast', 'Light Rain', 'Heavy Rain', 'Fog']
WIND_DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
FIELD_QUALITIES = ['Weak', 'Average', 'Strong', 'Elite']
PACE_SCENARIOS = ['Slow', 'Moderate', 'Fast', 'Contested']
RACE_GRADES = ['Grade 1', 'Grade 2', 'Grade 3', 'Listed', 'Allowance', 'Claiming', 'Maiden']

def generate_horse_enhanced_data():
    """Generate enhanced data for a horse"""
    return {
        # Basic Information (matching actual DB columns)
        'breed': random.choice(BREEDS),
        'color': random.choice(COLORS),
        'sex': random.choice(['Colt', 'Filly', 'Gelding', 'Mare', 'Stallion']),
        'height': f"{random.randint(15, 17)}.{random.randint(0, 3)}",
        'markings': random.choice(['Star', 'Blaze', 'White socks', 'No markings', 'Star and snip']),
        
        # Pedigree
        'sire': random.choice(SIRE_LINES),
        'dam': f"Dam of {random.choice(['Excellence', 'Victory', 'Champion', 'Glory', 'Pride'])}",
        'sire_line': random.choice(SIRE_LINES),
        'dam_line': random.choice(DAM_LINES),
        'breeding_value': random.randint(50000, 500000),
        
        # Connections
        'jockey': random.choice(JOCKEYS),
        'trainer': random.choice(TRAINERS),
        'owner': random.choice(OWNERS),
        'breeder': random.choice(OWNERS),
        'stable': random.choice(['Barn A', 'Barn B', 'Barn C', 'Main Stable']),
        
        # Physical Attributes
        'weight': random.randint(1000, 1200),
        'body_condition': random.choice(['fair', 'good', 'excellent']),
        'conformation_score': random.randint(7, 10),
        
        # Performance Analytics
        'speed_rating': random.randint(70, 110),
        'class_rating': random.randint(6, 10),
        'distance_preference': random.choice(['5f-7f', '6f-1m', '7f-1.25m', '1m-1.5m', '1m-1.75m', '1.5m+']),
        'surface_preference': random.choice(['dirt', 'turf', 'synthetic']),
        'track_bias_rating': random.randint(5, 10),
        'pace_style': random.choice(['front_runner', 'presser', 'stalker', 'closer']),
        'closing_kick': random.choice(['strong', 'average', 'weak']),
        
        # Training & Fitness
        'days_since_last_race': random.randint(14, 90),
        'fitness_level': random.randint(7, 10),
        'training_intensity': random.choice(['moderate', 'heavy', 'intense']),
        'workout_times': f'["{random.randint(58, 62)}.{random.randint(10, 99)}", "{random.randint(47, 51)}.{random.randint(10, 99)}"]',
        'injury_history': '[]',
        'recovery_time': random.randint(0, 30),
        
        # Behavioral & Racing Style
        'temperament': random.choice(['calm', 'eager', 'nervous']),
        'gate_behavior': random.choice(['good', 'excellent', 'average']),
        'racing_tactics': random.choice(['aggressive', 'patient', 'tactical']),
        'equipment_used': random.choice(['None', 'Blinkers', 'Tongue tie', 'Shadow roll']),
        'medication_notes': random.choice(['None', 'Lasix', 'Bute', 'Banamine']),
        
        # Financial Information
        'purchase_price': random.randint(50000, 1000000),
        'current_value': random.randint(75000, 1500000),
        'insurance_value': random.randint(100000, 2000000),
        'stud_fee': random.randint(5000, 100000),
        
        # Performance Statistics (using existing columns)
        'form': random.choice(['111', '211', '321', '123', '231']),
        'rating': random.randint(70, 110),
        'wins': random.randint(1, 8),
        'places': random.randint(2, 12),
        'runs': random.randint(5, 25),
        'earnings': random.randint(10000, 500000)
    }

def generate_race_enhanced_data():
    """Generate enhanced data for a race"""
    return {
        # Weather Conditions
        'temperature': random.randint(45, 85),
        'humidity': random.randint(40, 80),
        'wind_speed': random.randint(0, 15),
        'wind_direction': random.choice(WIND_DIRECTIONS),
        'weather_description': random.choice(WEATHER_CONDITIONS),
        'visibility': round(random.uniform(5.0, 10.0), 1),
        
        # Track Conditions
        'surface_type': random.choice(TRACK_SURFACES),
        'rail_position': f"{random.randint(0, 10)} feet out",
        'track_bias': random.choice(['None', 'Speed Favoring', 'Closer Favoring', 'Inside Bias']),
        'track_maintenance': random.choice(['Normal', 'Recently harrowed', 'Sealed', 'Watered']),
        
        # Field Analysis
        'field_size': random.randint(6, 14),
        'field_quality': random.choice(FIELD_QUALITIES),
        'pace_scenario': random.choice(PACE_SCENARIOS),
        'competitive_balance': random.choice(['Even', 'Top Heavy', 'Wide Open', 'Dominant Favorite']),
        'speed_figures_range': f"{random.randint(85, 95)}-{random.randint(105, 115)}",
        
        # Betting Information
        'total_pool': random.randint(500000, 2000000),
        'win_pool': random.randint(100000, 400000),
        'exacta_pool': random.randint(150000, 600000),
        'trifecta_pool': random.randint(200000, 800000),
        'superfecta_pool': random.randint(50000, 200000),
        'morning_line_favorite': f"{random.randint(2, 5)}-1",
        
        # Race Conditions
        'age_restrictions': random.choice(['3yo+', '4yo+', '3yo only', 'No restrictions']),
        'sex_restrictions': random.choice(['', 'Fillies and Mares', 'No restrictions']),
        'weight_conditions': f"Weight for age, {random.randint(118, 126)} lbs",
        'claiming_price': random.choice([None, 25000, 50000, 75000, 100000]),
        'race_grade': random.choice(RACE_GRADES),
        
        # Historical Data
        'track_record': f"1:{random.randint(32, 38)}.{random.randint(10, 99)}",
        'average_winning_time': f"1:{random.randint(34, 40)}.{random.randint(10, 99)}",
        'course_record_holder': random.choice(['Secretariat', 'Man o War', 'Citation', 'Seattle Slew']),
        'similar_race_results': f"Last year won by {random.choice(['favorite', 'longshot', 'second choice'])}",
        'trainer_jockey_stats': f"Leading trainer: {random.choice(TRAINERS)} (3 wins)",
        
        # Media Coverage
        'tv_coverage': random.choice(['Local', 'National', 'None']),
        'streaming_available': random.choice(['Yes', 'No']),
        'featured_race': random.choice(['Yes', 'No'])
    }

def enhance_existing_horses():
    """Enhance all existing horses with generated data"""
    # Use raw SQL to avoid SQLAlchemy column conflicts
    cursor = db.session.execute(text("SELECT id, name FROM horses"))
    horses = cursor.fetchall()
    print(f"Found {len(horses)} horses to enhance")
    
    for horse_id, horse_name in horses:
        enhanced_data = generate_horse_enhanced_data()
        
        # Build update query
        set_clauses = []
        params = {}
        for field, value in enhanced_data.items():
            set_clauses.append(f"{field} = :{field}")
            params[field] = value
        
        if set_clauses:
            params['horse_id'] = horse_id
            update_sql = f"UPDATE horses SET {', '.join(set_clauses)} WHERE id = :horse_id"
            db.session.execute(text(update_sql), params)
        
        print(f"Enhanced horse: {horse_name}")
    
    db.session.commit()
    print(f"Successfully enhanced {len(horses)} horses")

def enhance_existing_races():
    """Enhance all existing races with generated data"""
    # Use raw SQL to avoid SQLAlchemy column conflicts
    cursor = db.session.execute(text("SELECT id, name FROM races"))
    races = cursor.fetchall()
    print(f"Found {len(races)} races to enhance")
    
    for race_id, race_name in races:
        enhanced_data = generate_race_enhanced_data()
        
        # Build update query
        set_clauses = []
        params = {}
        for field, value in enhanced_data.items():
            set_clauses.append(f"{field} = :{field}")
            params[field] = value
        
        if set_clauses:
            params['race_id'] = race_id
            update_sql = f"UPDATE races SET {', '.join(set_clauses)} WHERE id = :race_id"
            db.session.execute(text(update_sql), params)
        
        print(f"Enhanced race: {race_name}")
    
    db.session.commit()
    print(f"Successfully enhanced {len(races)} races")

def main():
    """Main function to run the enhancement script"""
    with app.app_context():
        print("Starting data enhancement process...")
        print("=" * 50)
        
        try:
            # Enhance horses
            enhance_existing_horses()
            print()
            
            # Enhance races
            enhance_existing_races()
            print()
            
            print("=" * 50)
            print("Data enhancement completed successfully!")
            print("All existing horses and races now have enhanced data fields populated.")
            
        except Exception as e:
            print(f"Error during enhancement: {str(e)}")
            db.session.rollback()
            raise

if __name__ == "__main__":
    main()