#!/usr/bin/env python3
"""
Enhanced Recent Form Data Generator for Horse Racing Prediction System
Generates extensive form data for hundreds of horses with realistic racing histories
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid

# Expanded racing data for more realistic generation
TRACKS = [
    "Churchill Downs", "Belmont Park", "Santa Anita", "Gulfstream Park", "Keeneland",
    "Saratoga", "Del Mar", "Oaklawn Park", "Fair Grounds", "Tampa Bay Downs",
    "Aqueduct", "Woodbine", "Arlington Park", "Hawthorne", "Laurel Park",
    "Pimlico", "Monmouth Park", "Penn National", "Parx Racing", "Charles Town",
    "Mountaineer", "Turfway Park", "Golden Gate Fields", "Los Alamitos", "Emerald Downs",
    "Canterbury Park", "Prairie Meadows", "Remington Park", "Lone Star Park", "Sam Houston",
    "Turf Paradise", "Sunland Park", "Zia Park", "Ruidoso Downs", "Louisiana Downs"
]

SURFACES = ["Dirt", "Turf", "Synthetic", "All Weather"]
TRACK_CONDITIONS = ["Fast", "Good", "Firm", "Soft", "Yielding", "Heavy", "Sloppy", "Muddy"]
WEATHER_CONDITIONS = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Overcast", "Sunny", "Windy"]

DISTANCES = [
    ("5f", 5.0), ("5.5f", 5.5), ("6f", 6.0), ("6.5f", 6.5), ("7f", 7.0), ("7.5f", 7.5),
    ("1m", 8.0), ("1m 70y", 8.5), ("1m 1f", 9.0), ("1m 2f", 10.0), ("1m 3f", 11.0),
    ("1m 4f", 12.0), ("1m 5f", 13.0), ("1m 6f", 14.0), ("1m 7f", 15.0), ("2m", 16.0)
]

RACE_CLASSES = [
    "Maiden", "Claiming", "Allowance", "Stakes", "Graded Stakes", "Listed",
    "Handicap", "Conditions", "Novice", "Nursery", "Selling", "Optional Claiming"
]

JOCKEYS = [
    "J. Rosario", "I. Ortiz Jr.", "F. Geroux", "J. Castellano", "L. Saez", "M. Franco",
    "T. Gaffalione", "R. Santana Jr.", "J. Alvarado", "K. Carmouche", "D. Davis",
    "L. Reyes", "A. Beschizza", "E. Cancel", "H. Diaz Jr.", "J. Lezcano", "M. Mena",
    "R. Maragh", "J. Bravo", "C. Lopez", "L. Davis", "M. Smith", "J. Velazquez",
    "R. Bejarano", "D. Van Dyke", "A. Gryder", "E. Prado", "C. Nakatani", "G. Stevens"
]

TRAINERS = [
    "B. Baffert", "T. Pletcher", "C. Brown", "B. Cox", "S. Asmussen", "M. Casse",
    "J. Toscano", "R. Atras", "G. Motion", "K. McPeek", "D. Romans", "I. Wilkes",
    "M. Maker", "W. Mott", "J. Shirreffs", "R. Mandella", "P. Miller", "J. Sadler",
    "G. Lewis", "M. Stidham", "L. Rice", "D. Gargan", "R. Rodriguez", "C. Clement",
    "H. Bond", "T. Amoss", "A. Delacour", "M. Hennig", "J. Sharp", "K. Dooley"
]

# Horse name components for generating realistic names
HORSE_NAME_PREFIXES = [
    "Thunder", "Lightning", "Storm", "Fire", "Wind", "Star", "Golden", "Silver", "Royal",
    "Mighty", "Swift", "Bold", "Brave", "Noble", "Wild", "Free", "Dancing", "Flying",
    "Blazing", "Shining", "Crimson", "Midnight", "Dawn", "Sunset", "Ocean", "Mountain",
    "Desert", "Forest", "River", "Valley", "Diamond", "Ruby", "Emerald", "Sapphire",
    "Crystal", "Shadow", "Spirit", "Dream", "Magic", "Wonder", "Victory", "Champion",
    "Legend", "Hero", "Warrior", "Knight", "Prince", "King", "Queen", "Angel"
]

HORSE_NAME_SUFFIXES = [
    "Runner", "Dancer", "Flyer", "Bolt", "Strike", "Blaze", "Storm", "Wind", "Fire",
    "Star", "Moon", "Sun", "Light", "Shadow", "Spirit", "Dream", "Hope", "Glory",
    "Pride", "Joy", "Grace", "Beauty", "Power", "Force", "Speed", "Flash", "Dash",
    "Rush", "Charge", "Quest", "Journey", "Adventure", "Destiny", "Fortune", "Treasure",
    "Jewel", "Crown", "Throne", "Empire", "Kingdom", "Legend", "Myth", "Tale", "Song",
    "Melody", "Harmony", "Rhythm", "Beat", "Pulse", "Heart", "Soul", "Mind", "Will"
]

def generate_horse_name() -> str:
    """Generate a realistic horse name"""
    if random.random() < 0.7:  # 70% chance of prefix + suffix
        prefix = random.choice(HORSE_NAME_PREFIXES)
        suffix = random.choice(HORSE_NAME_SUFFIXES)
        return f"{prefix} {suffix}"
    else:  # 30% chance of single word with modifier
        base = random.choice(HORSE_NAME_PREFIXES + HORSE_NAME_SUFFIXES)
        modifiers = ["'s", " Boy", " Girl", " King", " Queen", " Star", " Dream"]
        if random.random() < 0.5:
            return base + random.choice(modifiers)
        return base

def generate_race_time(distance_furlongs: float, surface: str, track_condition: str) -> str:
    """Generate realistic race times based on distance and conditions"""
    # Base times per furlong (in seconds)
    base_time_per_furlong = {
        "Dirt": 12.0,
        "Turf": 12.5,
        "Synthetic": 12.2,
        "All Weather": 12.3
    }
    
    # Condition adjustments
    condition_adjustments = {
        "Fast": 0.0, "Good": 0.2, "Firm": 0.0, "Soft": 0.5,
        "Yielding": 1.0, "Heavy": 1.5, "Sloppy": 0.8, "Muddy": 1.2
    }
    
    base_seconds = distance_furlongs * base_time_per_furlong.get(surface, 12.0)
    condition_adj = condition_adjustments.get(track_condition, 0.0)
    
    # Add some randomness for individual horse performance
    random_factor = random.uniform(-2.0, 3.0)
    
    total_seconds = base_seconds + condition_adj + random_factor
    
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    if minutes > 0:
        return f"{minutes}:{seconds:05.2f}"
    else:
        return f"{seconds:.2f}"

def generate_sectional_times(distance_furlongs: float, final_time: str) -> Dict[str, str]:
    """Generate realistic sectional times"""
    try:
        # Parse final time
        if ":" in final_time:
            minutes, seconds = final_time.split(":")
            total_seconds = int(minutes) * 60 + float(seconds)
        else:
            total_seconds = float(final_time)
    except:
        total_seconds = distance_furlongs * 12.0  # Fallback
    
    sectionals = {}
    
    if distance_furlongs >= 4:
        quarter_time = total_seconds * 0.25 + random.uniform(-1, 1)
        sectionals["quarter_mile"] = f"{quarter_time:.2f}"
    
    if distance_furlongs >= 6:
        half_time = total_seconds * 0.5 + random.uniform(-1.5, 1.5)
        sectionals["half_mile"] = f"{half_time:.2f}"
    
    if distance_furlongs >= 8:
        three_quarter_time = total_seconds * 0.75 + random.uniform(-1, 1)
        sectionals["three_quarter_mile"] = f"{three_quarter_time:.2f}"
    
    return sectionals

def generate_horse_quality() -> str:
    """Generate horse quality level"""
    qualities = ["Elite", "High Class", "Good", "Average", "Moderate", "Poor"]
    weights = [0.05, 0.15, 0.25, 0.35, 0.15, 0.05]
    return random.choices(qualities, weights=weights)[0]

def generate_recent_performances(horse_name: str, num_performances: int = None) -> List[Dict[str, Any]]:
    """Generate realistic recent racing performances for a horse"""
    if num_performances is None:
        # Vary the number of performances more realistically
        num_performances = random.choices(
            [5, 8, 10, 12, 15, 18, 20, 25, 30],
            weights=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02]
        )[0]
    
    performances = []
    current_date = datetime.now()
    horse_quality = generate_horse_quality()
    
    # Quality affects performance consistency
    quality_modifiers = {
        "Elite": {"win_chance": 0.35, "place_chance": 0.65, "position_variance": 2},
        "High Class": {"win_chance": 0.25, "place_chance": 0.55, "position_variance": 3},
        "Good": {"win_chance": 0.18, "place_chance": 0.45, "position_variance": 4},
        "Average": {"win_chance": 0.12, "place_chance": 0.35, "position_variance": 5},
        "Moderate": {"win_chance": 0.08, "place_chance": 0.25, "position_variance": 6},
        "Poor": {"win_chance": 0.04, "place_chance": 0.15, "position_variance": 7}
    }
    
    quality_mod = quality_modifiers[horse_quality]
    
    for i in range(num_performances):
        # Generate race date (going backwards in time)
        days_ago = random.randint(7 + i * 10, 30 + i * 15)
        race_date = current_date - timedelta(days=days_ago)
        
        # Select race details
        track = random.choice(TRACKS)
        distance_info = random.choice(DISTANCES)
        distance, distance_furlongs = distance_info
        surface = random.choice(SURFACES)
        track_condition = random.choice(TRACK_CONDITIONS)
        weather = random.choice(WEATHER_CONDITIONS)
        race_class = random.choice(RACE_CLASSES)
        
        # Generate field size
        total_runners = random.randint(6, 16)
        
        # Generate position based on horse quality
        if random.random() < quality_mod["win_chance"]:
            position = 1
        elif random.random() < quality_mod["place_chance"]:
            position = random.randint(2, 3)
        else:
            base_position = random.randint(4, total_runners)
            variance = quality_mod["position_variance"]
            position = max(1, min(total_runners, base_position + random.randint(-variance, variance)))
        
        # Generate race time and sectionals
        race_time = generate_race_time(distance_furlongs, surface, track_condition)
        sectionals = generate_sectional_times(distance_furlongs, race_time)
        
        # Generate margin (distance behind winner)
        if position == 1:
            margin = 0
        else:
            # Margins increase with position, but with some randomness
            base_margin = (position - 1) * random.uniform(0.5, 2.5)
            margin = round(base_margin + random.uniform(-0.5, 1.0), 1)
            margin = max(0.1, margin)
        
        # Generate prize money based on race class and position
        class_multipliers = {
            "Graded Stakes": 200000, "Stakes": 100000, "Listed": 75000,
            "Allowance": 50000, "Handicap": 60000, "Conditions": 40000,
            "Optional Claiming": 35000, "Claiming": 25000, "Maiden": 30000,
            "Novice": 25000, "Nursery": 20000, "Selling": 15000
        }
        
        base_prize = class_multipliers.get(race_class, 30000)
        total_prize = base_prize * random.uniform(0.8, 1.5)
        
        # Prize distribution by position
        position_percentages = {
            1: 0.60, 2: 0.20, 3: 0.12, 4: 0.06, 5: 0.02
        }
        
        if position <= 5:
            earnings = total_prize * position_percentages.get(position, 0.01)
        else:
            earnings = total_prize * 0.005  # Small amount for other positions
        
        # Generate other race details
        jockey = random.choice(JOCKEYS)
        trainer = random.choice(TRAINERS)
        weight = round(random.uniform(115, 126), 1)
        odds = generate_realistic_odds(position, total_runners)
        
        performance = {
            "race_id": f"race_{race_date.strftime('%Y%m%d')}_{random.randint(1000, 9999)}",
            "date": race_date.strftime('%Y-%m-%d'),
            "track": track,
            "race_name": f"{race_class} - {distance}",
            "distance": distance,
            "distance_furlongs": distance_furlongs,
            "surface": surface,
            "track_condition": track_condition,
            "weather": weather,
            "position": position,
            "total_runners": total_runners,
            "time": race_time,
            "margin": margin,
            "earnings": round(earnings, 2),
            "jockey": jockey,
            "trainer": trainer,
            "weight_carried": weight,
            "odds": odds,
            "race_class": race_class,
            "prize_money": round(total_prize),
            "going": track_condition,
            "race_time": race_time,
            "sectional_times": sectionals
        }
        
        performances.append(performance)
    
    # Sort by date (most recent first)
    performances.sort(key=lambda x: x['date'], reverse=True)
    return performances

def generate_realistic_odds(position: int, total_runners: int) -> float:
    """Generate realistic betting odds based on finishing position"""
    if position == 1:
        # Winners typically had reasonable odds
        return round(random.uniform(1.5, 15.0), 1)
    elif position <= 3:
        # Place getters had moderate odds
        return round(random.uniform(3.0, 25.0), 1)
    else:
        # Others had longer odds
        return round(random.uniform(8.0, 99.0), 1)

def create_new_horse(horse_id: int) -> Dict[str, Any]:
    """Create a new horse with comprehensive data"""
    name = generate_horse_name()
    
    # Generate basic horse attributes
    age = random.randint(2, 8)
    breeds = ["Thoroughbred", "Arabian", "Quarter Horse", "Standardbred", "Paint"]
    breed = random.choice(breeds)
    
    jockey = random.choice(JOCKEYS)
    trainer = random.choice(TRAINERS)
    
    # Generate recent performances
    recent_performances = generate_recent_performances(name)
    
    # Calculate statistics from performances
    wins = sum(1 for p in recent_performances if p['position'] == 1)
    places = sum(1 for p in recent_performances if p['position'] <= 3)
    total_races = len(recent_performances)
    
    win_rate = wins / total_races if total_races > 0 else 0
    place_rate = places / total_races if total_races > 0 else 0
    show_rate = sum(1 for p in recent_performances if p['position'] <= 3) / total_races if total_races > 0 else 0
    
    total_earnings = sum(p['earnings'] for p in recent_performances)
    
    # Generate additional attributes
    speed_rating = random.randint(60, 120)
    class_rating = random.randint(50, 100)
    
    distance_preferences = ["Sprint", "Mile", "Route", "Marathon"]
    distance_preference = random.choice(distance_preferences)
    
    surface_preference = random.choice(SURFACES)
    
    track_bias_rating = random.randint(70, 95)
    fitness_level = random.choice(["Excellent", "Good", "Fair", "Poor"])
    
    # Generate workout times
    workout_times = []
    for _ in range(random.randint(3, 8)):
        workout_date = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
        workout_distance = random.choice(["3f", "4f", "5f", "6f"])
        workout_time = f"{random.randint(35, 75)}.{random.randint(10, 99)}"
        workout_times.append(f"{workout_date}: {workout_distance} in {workout_time}")
    
    # Generate injury history
    injury_history = []
    if random.random() < 0.3:  # 30% chance of having injury history
        for _ in range(random.randint(1, 3)):
            injury_date = (datetime.now() - timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d')
            injuries = ["Tendon strain", "Ankle sprain", "Muscle pull", "Bruised sole", "Minor cut"]
            injury = random.choice(injuries)
            injury_history.append(f"{injury_date}: {injury}")
    
    horse = {
        "id": horse_id,
        "name": name,
        "age": age,
        "breed": breed,
        "jockey": jockey,
        "trainer": trainer,
        "win_rate": win_rate,
        "place_rate": place_rate,
        "show_rate": show_rate,
        "earnings": total_earnings,
        "recent_performances": recent_performances,
        "speed_rating": speed_rating,
        "class_rating": class_rating,
        "distance_preference": distance_preference,
        "surface_preference": surface_preference,
        "track_bias_rating": track_bias_rating,
        "fitness_level": fitness_level,
        "workout_times": json.dumps(workout_times),
        "injury_history": json.dumps(injury_history)
    }
    
    return horse

def main():
    """Main function to generate extensive form data"""
    print("🐎 Enhanced Form Data Generator")
    print("=" * 50)
    
    # Load existing horses or create new list
    try:
        with open('data/horses.json', 'r') as f:
            horses = json.load(f)
        print(f"📂 Loaded {len(horses)} existing horses")
    except FileNotFoundError:
        horses = []
        print("📂 Creating new horses database")
    
    # Determine how many horses to generate
    target_horses = 200  # Generate 200 horses total
    current_count = len(horses)
    
    if current_count >= target_horses:
        print(f"✅ Already have {current_count} horses. Enhancing existing data...")
        # Enhance existing horses with more performances
        for i, horse in enumerate(horses):
            print(f"🔄 Enhancing {horse['name']} ({i+1}/{len(horses)})")
            additional_performances = generate_recent_performances(
                horse['name'], 
                random.randint(10, 25)  # Add 10-25 more performances
            )
            
            # Merge with existing performances
            existing_performances = horse.get('recent_performances', [])
            all_performances = existing_performances + additional_performances
            
            # Remove duplicates and sort by date
            seen_races = set()
            unique_performances = []
            for perf in all_performances:
                race_key = (perf['date'], perf['track'], perf['race_name'])
                if race_key not in seen_races:
                    seen_races.add(race_key)
                    unique_performances.append(perf)
            
            unique_performances.sort(key=lambda x: x['date'], reverse=True)
            horse['recent_performances'] = unique_performances
            
            # Recalculate statistics
            wins = sum(1 for p in unique_performances if p['position'] == 1)
            places = sum(1 for p in unique_performances if p['position'] <= 3)
            total_races = len(unique_performances)
            
            horse['win_rate'] = wins / total_races if total_races > 0 else 0
            horse['place_rate'] = places / total_races if total_races > 0 else 0
            horse['show_rate'] = places / total_races if total_races > 0 else 0
            horse['earnings'] = sum(p['earnings'] for p in unique_performances)
    
    else:
        horses_to_generate = target_horses - current_count
        print(f"🏗️  Generating {horses_to_generate} new horses...")
        
        for i in range(horses_to_generate):
            horse_id = current_count + i + 1
            new_horse = create_new_horse(horse_id)
            horses.append(new_horse)
            
            if (i + 1) % 20 == 0:
                print(f"✅ Generated {i + 1}/{horses_to_generate} horses")
    
    # Save updated data
    print("💾 Saving enhanced horse data...")
    with open('data/horses.json', 'w') as f:
        json.dump(horses, f, indent=2)
    
    # Generate summary statistics
    total_horses = len(horses)
    total_performances = sum(len(horse.get('recent_performances', [])) for horse in horses)
    avg_performances = total_performances / total_horses if total_horses > 0 else 0
    
    print("\n📊 Enhanced Data Summary:")
    print(f"🐎 Total horses: {total_horses}")
    print(f"🏁 Total performances: {total_performances}")
    print(f"📈 Average performances per horse: {avg_performances:.1f}")
    
    # Show sample data
    if horses:
        sample_horse = horses[0]
        print(f"\n🎯 Sample horse: {sample_horse['name']}")
        print(f"   Performances: {len(sample_horse.get('recent_performances', []))}")
        print(f"   Win rate: {sample_horse.get('win_rate', 0):.1%}")
        print(f"   Earnings: ${sample_horse.get('earnings', 0):,.2f}")
    
    print("\n🎉 Enhanced form data generation complete!")

if __name__ == "__main__":
    main()