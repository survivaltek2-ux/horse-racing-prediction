#!/usr/bin/env python3
"""
Comprehensive Training Data Generator for Horse Racing AI
Generates large amounts of diverse, realistic training data
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sqlite3
import os

class ComprehensiveDataGenerator:
    def __init__(self):
        self.horse_names = [
            "Thunder Strike", "Lightning Bolt", "Storm Chaser", "Wind Runner", "Fire Spirit",
            "Golden Arrow", "Silver Bullet", "Midnight Express", "Dawn Breaker", "Star Gazer",
            "Royal Thunder", "Crimson Flash", "Blue Lightning", "Green Arrow", "Black Beauty",
            "White Lightning", "Red Storm", "Purple Rain", "Orange Sunset", "Yellow Thunder",
            "Diamond Dust", "Ruby Fire", "Emerald Wind", "Sapphire Storm", "Pearl Lightning",
            "Blazing Comet", "Racing Fury", "Speed Demon", "Victory Dance", "Champion's Pride",
            "Galloping Ghost", "Flying Eagle", "Roaring Lion", "Charging Bull", "Soaring Falcon",
            "Mighty Stallion", "Noble Steed", "Brave Heart", "Wild Spirit", "Free Runner",
            "Power Surge", "Energy Blast", "Force Field", "Turbo Charge", "Rocket Fuel",
            "Laser Beam", "Sonic Boom", "Thunder Clap", "Lightning Strike", "Storm Front",
            "Hurricane Force", "Tornado Alley", "Cyclone Spin", "Typhoon Wind", "Blizzard Rush",
            "Avalanche Run", "Earthquake Shake", "Volcano Burst", "Tsunami Wave", "Meteor Shower",
            "Galaxy Quest", "Cosmic Ray", "Stellar Wind", "Nebula Drift", "Pulsar Beat",
            "Quantum Leap", "Atomic Flash", "Nuclear Fusion", "Solar Flare", "Lunar Eclipse",
            "Mars Rover", "Jupiter Storm", "Saturn Ring", "Neptune Wave", "Pluto Ice",
            "Alpha Centauri", "Beta Star", "Gamma Ray", "Delta Force", "Epsilon Wave",
            "Zeta Prime", "Eta Storm", "Theta Flash", "Iota Wind", "Kappa Strike",
            "Lambda Light", "Mu Force", "Nu Energy", "Xi Power", "Omicron Blast",
            "Pi Circle", "Rho Flow", "Sigma Sum", "Tau Time", "Upsilon Up",
            "Phi Golden", "Chi Life", "Psi Mind", "Omega End", "Alpha Begin",
            "Ace of Spades", "King of Hearts", "Queen of Diamonds", "Jack of Clubs", "Joker Wild",
            "Royal Flush", "Straight Shot", "Full House", "Three Kind", "Two Pair",
            "High Card", "Lucky Seven", "Magic Eight", "Perfect Ten", "Eleven Eleven",
            "Dozen Roses", "Baker's Dozen", "Sweet Sixteen", "Lucky Seventeen", "Golden Eighteen",
            "Prime Nineteen", "Score Twenty", "Blackjack", "Deuce Wild", "Trey High",
            "Four Leaf", "High Five", "Six Shooter", "Lucky Seven", "Crazy Eight", "Cloud Nine"
        ]
        
        self.track_names = [
            "Churchill Downs", "Belmont Park", "Saratoga", "Santa Anita", "Del Mar",
            "Keeneland", "Gulfstream Park", "Oaklawn Park", "Fair Grounds", "Aqueduct",
            "Pimlico", "Woodbine", "Arlington Park", "Hawthorne", "Ellis Park",
            "Turfway Park", "Tampa Bay Downs", "Laurel Park", "Penn National", "Charles Town",
            "Mountaineer", "Presque Isle", "Finger Lakes", "Monmouth Park", "Meadowlands",
            "Freehold", "Harrington", "Dover Downs", "Delaware Park", "Colonial Downs",
            "Shenandoah Downs", "Rosecroft", "Ocean Downs", "Timonium", "Great Barrington",
            "Suffolk Downs", "Rockingham Park", "Seabrook", "The Downs", "Fonner Park",
            "Lincoln", "Prairie Meadows", "Canterbury Park", "Running Aces", "Emerald Downs",
            "Grants Pass", "Portland Meadows", "Turf Paradise", "Rillito", "Sonoita",
            "Zia Park", "Sunray Park", "Ruidoso Downs", "Albuquerque", "SunRay Park",
            "Lone Star Park", "Sam Houston", "Retama Park", "Manor Downs", "Gillespie County",
            "La Bahia Downs", "Delta Downs", "Evangeline Downs", "Louisiana Downs", "Fair Grounds",
            "Oaklawn Park", "Southland", "Will Rogers Downs", "Remington Park", "Blue Ribbon Downs",
            "Anthony Downs", "Sunland Park", "Ruidoso Downs", "Albuquerque", "Zia Park"
        ]
        
        self.jockey_names = [
            "Mike Smith", "John Velazquez", "Javier Castellano", "Joel Rosario", "Irad Ortiz Jr.",
            "Jose Ortiz", "Flavien Prat", "Luis Saez", "Tyler Gaffalione", "Florent Geroux",
            "Ricardo Santana Jr.", "Manny Franco", "Dylan Davis", "Junior Alvarado", "Antonio Fresu",
            "Kendrick Carmouche", "Eric Cancel", "David Cohen", "Trevor McCarthy", "Nik Juarez",
            "Alex Cintron", "Angel Cruz", "Emisael Jaramillo", "Reylu Gutierrez", "Carlos Hernandez",
            "Edgard Zayas", "Leonel Reyes", "Samy Camacho", "Edwin Gonzalez", "Marcos Meneses",
            "Victor Carrasco", "Jorge Vargas Jr.", "Jaime Rodriguez", "Jesus Castanon", "Miguel Vasquez",
            "Ruben Fuentes", "Francisco Arrieta", "Ry Eikleberry", "Colby Hernandez", "Alex Birzer",
            "Stewart Elliott", "Channing Hill", "Declan Carroll", "James Graham", "Julien Leparoux",
            "Brian Hernandez Jr.", "Corey Lanerie", "Gabriel Saez", "Adam Beschizza", "Martin Garcia",
            "Drayden Van Dyke", "Juan Hernandez", "Umberto Rispoli", "Abel Cedillo", "Hector Berrios",
            "Edwin Maldonado", "Kazushi Kimura", "Tiago Pereira", "Kyle Frey", "Geovanni Franco",
            "Ramon Vazquez", "Edgar Morales", "Alexis Centeno", "Orlando Bocachica", "Cristian Torres",
            "Rene Diaz", "Wilmer Garcia", "Angel Suarez", "Yomar Ortiz", "Jermaine Bridgmohan",
            "Chris Landeros", "Robby Albarado", "Mitchell Murrill", "Diego Saenz", "Jareth Loveberry",
            "Colton Murray", "Jake Olesiak", "Dane Nelson", "Alex Anaya", "Isaias Enriquez",
            "Francisco Calderon", "Agustin Bracho", "Cesar Morales", "Juan Carlos Diaz", "Rodolfo Guerra"
        ]
        
        self.trainer_names = [
            "Bob Baffert", "Chad Brown", "Todd Pletcher", "Bill Mott", "Steve Asmussen",
            "Mark Casse", "Brad Cox", "John Sadler", "Richard Baltas", "Doug O'Neill",
            "Jerry Hollendorfer", "Peter Miller", "Phil D'Amato", "Leonard Powell", "Tim Yakteen",
            "Michael McCarthy", "Jeff Mullins", "Dan Blacker", "Craig Lewis", "Vladimir Cerin",
            "Robertino Diodoro", "Ron Moquett", "Karl Broberg", "Joe Sharp", "Tom Amoss",
            "Al Stall Jr.", "Wayne Catalano", "Bret Calhoun", "Steve Margolis", "Michael Maker",
            "Ian Wilkes", "Kenny McPeek", "Dale Romans", "Brendan Walsh", "Graham Motion",
            "Kiaran McLaughlin", "Christophe Clement", "Linda Rice", "Rudy Rodriguez", "Jorge Abreu",
            "Gary Contessa", "David Jacobson", "Michelle Nevin", "Jeremiah Englehart", "John Kimmel",
            "Randi Persaud", "Jason Servis", "Jorge Navarro", "Marcus Vitali", "Claudio Gonzalez",
            "Carlos Martin", "Saffie Joseph Jr.", "Antonio Sano", "Gustavo Delgado", "Victor Barboza",
            "Ralph Nicks", "Kathleen O'Connell", "David Fawkes", "Michael Yates", "Jane Cibelli",
            "Norm Casse", "Josie Carroll", "Kevin Attard", "Catherine Day Phillips", "Gail Cox",
            "Roger Attfield", "Mark Frostad", "Stuart Simon", "Mike Mattine", "Robert Tiller",
            "Danny Gargan", "Amira Chichakly", "George Weaver", "Rob Atras", "Horacio DePaz",
            "Juan Alvarado", "Guadalupe Preciado", "Genaro Vallejo", "Eoin Harty", "Philip Gleaves"
        ]
        
        self.race_types = [
            "Maiden", "Allowance", "Claiming", "Stakes", "Handicap", "Graded Stakes",
            "Listed Stakes", "Conditions", "Starter Allowance", "Optional Claiming"
        ]
        
        self.surfaces = ["Dirt", "Turf", "Synthetic", "All Weather"]
        self.track_conditions = ["Fast", "Good", "Firm", "Yielding", "Soft", "Heavy", "Sloppy", "Muddy"]
        self.weather_conditions = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Windy", "Hot", "Cold"]
        
    def generate_horse_data(self, num_horses: int = 500) -> List[Dict[str, Any]]:
        """Generate diverse horse data with realistic performance profiles"""
        horses = []
        
        for i in range(num_horses):
            # Create performance profile based on horse quality tier
            quality_tier = random.choices(
                ["elite", "good", "average", "poor"],
                weights=[0.05, 0.15, 0.60, 0.20]
            )[0]
            
            if quality_tier == "elite":
                base_win_rate = random.uniform(0.25, 0.45)
                base_place_rate = random.uniform(0.45, 0.70)
                base_show_rate = random.uniform(0.65, 0.85)
                base_earnings = random.uniform(500000, 2000000)
            elif quality_tier == "good":
                base_win_rate = random.uniform(0.15, 0.30)
                base_place_rate = random.uniform(0.30, 0.50)
                base_show_rate = random.uniform(0.45, 0.70)
                base_earnings = random.uniform(100000, 600000)
            elif quality_tier == "average":
                base_win_rate = random.uniform(0.08, 0.20)
                base_place_rate = random.uniform(0.20, 0.40)
                base_show_rate = random.uniform(0.35, 0.60)
                base_earnings = random.uniform(25000, 150000)
            else:  # poor
                base_win_rate = random.uniform(0.02, 0.12)
                base_place_rate = random.uniform(0.10, 0.25)
                base_show_rate = random.uniform(0.20, 0.45)
                base_earnings = random.uniform(5000, 50000)
            
            # Generate recent performance data
            recent_performances = []
            for j in range(random.randint(3, 10)):
                performance_date = datetime.now() - timedelta(days=random.randint(7, 365))
                finish_position = self._generate_finish_position(base_win_rate, base_place_rate, base_show_rate)
                
                recent_performances.append({
                    "date": performance_date.strftime("%Y-%m-%d"),
                    "track": random.choice(self.track_names),
                    "distance": random.choice([1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]),
                    "surface": random.choice(self.surfaces),
                    "finish_position": finish_position,
                    "field_size": random.randint(6, 16),
                    "time": f"{random.randint(60, 180)}.{random.randint(10, 99)}",
                    "odds": round(random.uniform(1.5, 50.0), 1)
                })
            
            horse = {
                "id": str(uuid.uuid4()),
                "name": random.choice(self.horse_names) + f" #{i+1}",
                "age": random.randint(2, 8),
                "breed": random.choice(["Thoroughbred", "Arabian", "Quarter Horse", "Standardbred"]),
                "jockey": random.choice(self.jockey_names),
                "trainer": random.choice(self.trainer_names),
                "win_rate": round(base_win_rate, 3),
                "place_rate": round(base_place_rate, 3),
                "show_rate": round(base_show_rate, 3),
                "earnings": int(base_earnings),
                "recent_performances": recent_performances,
                "quality_tier": quality_tier,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            horses.append(horse)
        
        return horses
    
    def _generate_finish_position(self, win_rate: float, place_rate: float, show_rate: float) -> int:
        """Generate realistic finish position based on horse's ability"""
        rand = random.random()
        if rand < win_rate:
            return 1
        elif rand < place_rate:
            return random.choice([2, 3])
        elif rand < show_rate:
            return random.choice([2, 3, 4])
        else:
            return random.randint(4, 16)
    
    def generate_race_data(self, horses: List[Dict], num_races: int = 200) -> List[Dict[str, Any]]:
        """Generate diverse race data with realistic scenarios"""
        races = []
        
        for i in range(num_races):
            race_date = datetime.now() - timedelta(days=random.randint(1, 730))
            
            # Select horses for this race (6-16 horses)
            field_size = random.randint(6, 16)
            race_horses = random.sample(horses, field_size)
            
            # Generate race conditions
            distance = random.choice([1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400])
            surface = random.choice(self.surfaces)
            track_condition = random.choice(self.track_conditions)
            weather = random.choice(self.weather_conditions)
            
            # Generate race results based on horse quality and conditions
            results = self._generate_race_results(race_horses, surface, distance, track_condition)
            
            race = {
                "id": str(uuid.uuid4()),
                "name": f"Race {i+1} - {random.choice(self.race_types)}",
                "date": race_date.strftime("%Y-%m-%d"),
                "time": f"{random.randint(12, 18)}:{random.randint(0, 59):02d}",
                "track": random.choice(self.track_names),
                "distance": distance,
                "surface": surface,
                "track_condition": track_condition,
                "weather": weather,
                "race_type": random.choice(self.race_types),
                "purse": random.randint(10000, 500000),
                "field_size": field_size,
                "status": "completed",
                "results": results,
                "horse_ids": [horse["id"] for horse in race_horses],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            races.append(race)
        
        return races
    
    def _generate_race_results(self, horses: List[Dict], surface: str, distance: int, track_condition: str) -> List[Dict]:
        """Generate realistic race results based on horse abilities and conditions"""
        results = []
        
        # Calculate performance scores for each horse
        horse_scores = []
        for horse in horses:
            base_score = horse["win_rate"] * 100
            
            # Adjust for surface preference
            if surface == "Turf":
                base_score *= random.uniform(0.8, 1.2)
            elif surface == "Synthetic":
                base_score *= random.uniform(0.9, 1.1)
            
            # Adjust for distance preference
            if distance <= 1400:  # Sprint
                if horse["age"] <= 4:
                    base_score *= random.uniform(1.0, 1.1)
            else:  # Route
                if horse["age"] >= 4:
                    base_score *= random.uniform(1.0, 1.1)
            
            # Adjust for track condition
            if track_condition in ["Sloppy", "Muddy", "Heavy"]:
                base_score *= random.uniform(0.8, 1.2)
            
            # Add random variance
            final_score = base_score * random.uniform(0.7, 1.3)
            horse_scores.append((horse, final_score))
        
        # Sort by score (highest first) and assign positions
        horse_scores.sort(key=lambda x: x[1], reverse=True)
        
        for position, (horse, score) in enumerate(horse_scores, 1):
            # Generate realistic race time
            base_time = 60 + (distance / 20)  # Rough time calculation
            time_variance = random.uniform(-5, 10)
            race_time = base_time + time_variance + (position * 0.5)
            
            # Generate odds based on horse quality and position
            if position <= 3:
                odds = random.uniform(1.5, 8.0)
            elif position <= 6:
                odds = random.uniform(5.0, 15.0)
            else:
                odds = random.uniform(10.0, 50.0)
            
            result = {
                "horse_id": horse["id"],
                "horse_name": horse["name"],
                "jockey": horse["jockey"],
                "trainer": horse["trainer"],
                "position": position,
                "time": f"{int(race_time // 60)}:{int(race_time % 60):02d}.{random.randint(10, 99)}",
                "odds": round(odds, 1),
                "margin": round(random.uniform(0.1, 5.0), 1) if position > 1 else 0.0
            }
            results.append(result)
        
        return results
    
    def save_to_database(self, horses: List[Dict], races: List[Dict], db_path: str = "instance/horse_racing.db"):
        """Save generated data to SQLite database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Insert horses - matching actual schema
            for horse in horses:
                # Calculate derived stats
                wins = int(horse["win_rate"] * 20)  # Assume 20 runs on average
                places = int(horse["place_rate"] * 20)
                runs = random.randint(15, 30)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO horses 
                    (name, age, sex, color, sire, dam, trainer, jockey, owner, weight, form, rating, last_run, wins, places, runs, earnings, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    horse["name"], horse["age"], 
                    random.choice(["Colt", "Filly", "Gelding", "Mare", "Stallion"]),
                    random.choice(["Bay", "Chestnut", "Black", "Gray", "Brown", "Roan"]),
                    f"Sire of {horse['name']}", f"Dam of {horse['name']}",
                    horse["trainer"], horse["jockey"], 
                    f"Owner of {horse['name']}", round(random.uniform(110, 130), 1),
                    "".join([str(random.randint(1, 9)) for _ in range(5)]),  # Form string
                    int(horse["win_rate"] * 100 + random.randint(50, 100)),  # Rating
                    (datetime.now() - timedelta(days=random.randint(7, 60))).strftime("%Y-%m-%d"),
                    wins, places, runs, horse["earnings"],
                    horse["created_at"], horse["updated_at"]
                ))
            
            # Insert races - matching actual schema
            for race in races:
                cursor.execute("""
                    INSERT OR REPLACE INTO races 
                    (name, date, time, track, distance, surface, race_class, prize_money, weather, track_condition, status, results, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    race["name"], race["date"], race["time"], race["track"],
                    race["distance"], race["surface"], race.get("race_type", "Maiden"),
                    race.get("purse", 50000), race.get("weather", "Clear"),
                    race.get("track_condition", "Fast"), race["status"],
                    json.dumps(race["results"]), race["created_at"], race["updated_at"]
                ))
                
                # Get the race ID from the database
                race_db_id = cursor.lastrowid
                
                # Insert race-horse relationships using database IDs
                for horse in horses:
                    if horse["id"] in race["horse_ids"]:
                        # Find horse database ID
                        cursor.execute("SELECT id FROM horses WHERE name = ?", (horse["name"],))
                        horse_result = cursor.fetchone()
                        if horse_result:
                            horse_db_id = horse_result[0]
                            cursor.execute("""
                                INSERT OR REPLACE INTO race_horses (race_id, horse_id)
                                VALUES (?, ?)
                            """, (race_db_id, horse_db_id))
            
            conn.commit()
            print(f"✅ Successfully saved {len(horses)} horses and {len(races)} races to database")
            
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def generate_and_save_data(self, num_horses: int = 500, num_races: int = 200):
        """Generate and save comprehensive training data"""
        print(f"🐎 Generating {num_horses} horses...")
        horses = self.generate_horse_data(num_horses)
        
        print(f"🏁 Generating {num_races} races...")
        races = self.generate_race_data(horses, num_races)
        
        print("💾 Saving to database...")
        self.save_to_database(horses, races)
        
        # Also save to JSON files for backup
        with open("data/generated_horses.json", "w") as f:
            json.dump(horses, f, indent=2)
        
        with open("data/generated_races.json", "w") as f:
            json.dump(races, f, indent=2)
        
        print("✅ Data generation completed successfully!")
        print(f"   - {len(horses)} horses generated")
        print(f"   - {len(races)} races generated")
        print(f"   - Total race entries: {sum(len(race['horse_ids']) for race in races)}")

if __name__ == "__main__":
    generator = ComprehensiveDataGenerator()
    
    # Generate large dataset for training
    generator.generate_and_save_data(num_horses=1000, num_races=500)