#!/usr/bin/env python3
"""
Training Data Generator for Horse Racing Prediction System

This script generates realistic race results and historical data for training ML models.
It uses the existing enhanced race and horse data to create believable outcomes
based on statistical models and racing factors.
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import copy

class TrainingDataGenerator:
    """Generates realistic training data for horse racing predictions"""
    
    def __init__(self):
        self.horses_data = []
        self.races_data = []
        self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing enhanced data"""
        try:
            with open('data/horses_enhanced.json', 'r') as f:
                self.horses_data = json.load(f)
            print(f"Loaded {len(self.horses_data)} horses")
            
            with open('data/races_enhanced.json', 'r') as f:
                self.races_data = json.load(f)
            print(f"Loaded {len(self.races_data)} races")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            
    def calculate_horse_probability(self, horse_id: int, race_data: Dict) -> float:
        """Calculate win probability for a horse based on various factors"""
        horse = self.get_horse_by_id(horse_id)
        if not horse:
            return 0.1
            
        # Base probability from horse quality
        base_prob = 0.1
        
        # Factor in actual win rate from horse data
        win_rate = horse.get('win_rate', 0.1)
        base_prob += win_rate * 0.4  # Win rate is a strong predictor
        
        # Factor in place rate (consistency)
        place_rate = horse.get('place_rate', 0.2)
        base_prob += place_rate * 0.2
        
        # Factor in earnings (quality indicator)
        earnings = horse.get('earnings', 0)
        if earnings > 100000:
            base_prob += 0.15  # High earner bonus
        elif earnings > 50000:
            base_prob += 0.1   # Medium earner bonus
        elif earnings > 10000:
            base_prob += 0.05  # Low earner bonus
            
        # Factor in recent performances
        if 'recent_performances' in horse and horse['recent_performances']:
            recent_perfs = horse['recent_performances'][:5]  # Last 5 races
            
            # Calculate recent form
            recent_wins = sum(1 for perf in recent_perfs if perf.get('position', 99) == 1)
            recent_places = sum(1 for perf in recent_perfs if perf.get('position', 99) <= 3)
            
            if len(recent_perfs) > 0:
                recent_win_rate = recent_wins / len(recent_perfs)
                recent_place_rate = recent_places / len(recent_perfs)
                base_prob += recent_win_rate * 0.15
                base_prob += recent_place_rate * 0.1
                
            # Factor in track surface preference (from recent performances)
            race_surface = race_data.get('track_data', {}).get('surface', 'Dirt')
            surface_performances = [p for p in recent_perfs if p.get('surface', '').lower() == race_surface.lower()]
            if surface_performances:
                surface_wins = sum(1 for p in surface_performances if p.get('position', 99) == 1)
                if len(surface_performances) > 0:
                    surface_win_rate = surface_wins / len(surface_performances)
                    base_prob += surface_win_rate * 0.1
                    
        # Factor in age (peak performance typically 3-6 years)
        age = horse.get('age', 4)
        if 3 <= age <= 6:
            base_prob += 0.05
        elif age > 8:
            base_prob -= 0.05
                
        # Factor in jockey skill (simplified since jockey is a string)
        if 'jockey' in horse and horse['jockey']:
            # Use a simple hash-based rating for consistency
            jockey_hash = hash(horse['jockey']) % 100
            jockey_rating = (jockey_hash + 50) / 150.0  # Range 0.33-1.0
            base_prob += jockey_rating * 0.1
            
        # Factor in trainer success (simplified since trainer is a string)
        if 'trainer' in horse and horse['trainer']:
            # Use a simple hash-based rating for consistency
            trainer_hash = hash(horse['trainer']) % 100
            trainer_rating = (trainer_hash + 30) / 130.0  # Range 0.23-1.0
            base_prob += trainer_rating * 0.1
            
        # Weather factors
        weather = race_data.get('weather_data', {})
        if weather.get('condition') == 'Heavy Rain' and horse.get('track_preferences', {}).get('wet_track', False):
            base_prob += 0.05
        elif weather.get('condition') == 'Clear' and not horse.get('track_preferences', {}).get('wet_track', True):
            base_prob += 0.05
            
        # Normalize probability
        return max(0.01, min(0.95, base_prob))
        
    def get_horse_by_id(self, horse_id: int) -> Dict:
        """Get horse data by ID"""
        for horse in self.horses_data:
            if horse.get('id') == horse_id:
                return horse
        return None
        
    def generate_race_results(self, race_data: Dict) -> Dict:
        """Generate realistic race results for a race"""
        horse_ids = race_data.get('horse_ids', [])
        if not horse_ids:
            return {}
            
        # Calculate probabilities for each horse
        probabilities = {}
        for horse_id in horse_ids:
            prob = self.calculate_horse_probability(horse_id, race_data)
            probabilities[horse_id] = prob
            
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        else:
            # Equal probabilities if calculation fails
            prob_each = 1.0 / len(horse_ids)
            probabilities = {k: prob_each for k in horse_ids}
            
        # Generate finishing order using weighted random selection
        remaining_horses = list(horse_ids)
        finishing_order = []
        
        for position in range(1, len(horse_ids) + 1):
            if len(remaining_horses) == 1:
                winner = remaining_horses[0]
            else:
                # Weight selection by probability (higher prob = more likely to finish well)
                weights = [probabilities[h] * (len(remaining_horses) - position + 1) for h in remaining_horses]
                
                # Handle edge cases where weights might be zero or invalid
                weights = np.array(weights)
                weights = np.nan_to_num(weights, nan=0.1, posinf=1.0, neginf=0.1)
                
                # Ensure all weights are positive
                weights = np.maximum(weights, 0.01)
                
                # Normalize weights
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    # Equal weights if all are zero
                    weights = np.ones(len(remaining_horses)) / len(remaining_horses)
                
                winner = np.random.choice(remaining_horses, p=weights)
                
            finishing_order.append(winner)
            remaining_horses.remove(winner)
            
        # Generate race times and earnings
        results = {}
        base_time = self.calculate_base_time(race_data)
        total_purse = race_data.get('purse', 50000)
        
        # Prize distribution (typical horse racing)
        prize_percentages = [0.6, 0.2, 0.1, 0.05, 0.03, 0.02] + [0.0] * max(0, len(finishing_order) - 6)
        
        for i, horse_id in enumerate(finishing_order):
            position = i + 1
            
            # Calculate finish time (winner gets base time, others get progressively slower)
            time_penalty = i * random.uniform(0.1, 0.5)  # 0.1-0.5 seconds per position
            finish_time = base_time + time_penalty
            
            # Calculate earnings
            if i < len(prize_percentages):
                earnings = total_purse * prize_percentages[i]
            else:
                earnings = 0
                
            results[str(horse_id)] = {
                'position': position,
                'time': round(finish_time, 2),
                'earnings': round(earnings, 2),
                'margin': round(time_penalty, 2) if i > 0 else 0.0
            }
            
        return results
        
    def calculate_base_time(self, race_data: Dict) -> float:
        """Calculate realistic base time for race distance"""
        distance_str = race_data.get('distance', '1200m')
        distance = int(distance_str.replace('m', ''))
        
        # Base times (in seconds) for different distances
        # These are realistic times for average horses
        base_times = {
            1000: 60.0,
            1200: 72.0,
            1400: 84.0,
            1600: 96.0,
            1800: 108.0,
            2000: 120.0,
            2400: 144.0
        }
        
        # Find closest distance
        closest_distance = min(base_times.keys(), key=lambda x: abs(x - distance))
        base_time = base_times[closest_distance]
        
        # Adjust for track conditions
        track_condition = race_data.get('track_data', {}).get('condition', 'Good')
        if track_condition == 'Heavy':
            base_time *= 1.05
        elif track_condition == 'Soft':
            base_time *= 1.02
        elif track_condition == 'Fast':
            base_time *= 0.98
            
        # Add some randomness
        base_time += random.uniform(-2.0, 2.0)
        
        return max(base_time, 30.0)  # Minimum 30 seconds
        
    def generate_historical_races(self, num_historical_races: int = 50) -> List[Dict]:
        """Generate historical races with results for training"""
        historical_races = []
        
        # Create historical races based on existing race templates
        for i in range(num_historical_races):
            # Pick a random existing race as template
            template_race = random.choice(self.races_data)
            
            # Create historical version
            historical_race = copy.deepcopy(template_race)
            
            # Set historical date (1-365 days ago)
            days_ago = random.randint(1, 365)
            historical_date = datetime.now() - timedelta(days=days_ago)
            historical_race['date'] = historical_date.strftime('%Y-%m-%d')
            historical_race['id'] = 1000 + i  # Unique ID for historical races
            historical_race['name'] = f"Historical {template_race['name']} #{i+1}"
            historical_race['status'] = 'completed'
            
            # Generate results
            results = self.generate_race_results(historical_race)
            historical_race['results'] = results
            
            # Update horse recent form based on results
            self.update_horse_form(results, historical_date)
            
            historical_races.append(historical_race)
            
        return historical_races
        
    def update_horse_form(self, results: Dict, race_date: datetime):
        """Update horse form data based on race results"""
        for horse_id_str, result in results.items():
            horse_id = int(horse_id_str)
            horse = self.get_horse_by_id(horse_id)
            
            if horse:
                # Update earnings
                horse['earnings'] = horse.get('earnings', 0) + result['earnings']
                
                # Recalculate win/place rates based on recent performances
                if 'recent_performances' in horse:
                    perfs = horse['recent_performances']
                    
                    # Count wins and places
                    wins = sum(1 for p in perfs if p.get('position', 99) == 1)
                    places = sum(1 for p in perfs if p.get('position', 99) <= 3)
                    total_races = len(perfs)
                    
                    if total_races > 0:
                        horse['win_rate'] = wins / total_races
                        horse['place_rate'] = places / total_races
                        horse['show_rate'] = sum(1 for p in perfs if p.get('position', 99) <= 3) / total_races
                
    def save_training_data(self, historical_races: List[Dict]):
        """Save generated training data"""
        # Combine existing races with historical races
        all_races = self.races_data + historical_races
        
        # Save to training data file
        with open('data/training_races.json', 'w') as f:
            json.dump(all_races, f, indent=2)
            
        # Save updated horses data
        with open('data/training_horses.json', 'w') as f:
            json.dump(self.horses_data, f, indent=2)
            
        print(f"Generated {len(historical_races)} historical races")
        print(f"Total training races: {len(all_races)}")
        print(f"Races with results: {len([r for r in all_races if r.get('results')])}")
        
    def generate_training_dataset(self, num_historical_races: int = 100):
        """Generate complete training dataset"""
        print("Generating training data...")
        print(f"Target: {num_historical_races} historical races with results")
        
        # Generate historical races
        historical_races = self.generate_historical_races(num_historical_races)
        
        # Save training data
        self.save_training_data(historical_races)
        
        # Generate summary statistics
        self.print_training_summary(historical_races)
        
    def print_training_summary(self, historical_races: List[Dict]):
        """Print summary of generated training data"""
        print("\n" + "="*50)
        print("TRAINING DATA SUMMARY")
        print("="*50)
        
        total_races = len(self.races_data) + len(historical_races)
        completed_races = len(historical_races)
        
        print(f"Total Races: {total_races}")
        print(f"Completed Races with Results: {completed_races}")
        print(f"Upcoming Races: {len(self.races_data)}")
        print(f"Total Horses: {len(self.horses_data)}")
        
        # Analyze results distribution
        all_results = []
        for race in historical_races:
            if race.get('results'):
                all_results.extend(race['results'].values())
                
        if all_results:
            avg_time = np.mean([r['time'] for r in all_results])
            avg_earnings = np.mean([r['earnings'] for r in all_results])
            
            print(f"\nRace Statistics:")
            print(f"Average Finish Time: {avg_time:.2f} seconds")
            print(f"Average Earnings per Horse: ${avg_earnings:.2f}")
            
        print(f"\nFiles Created:")
        print(f"- data/training_races.json ({total_races} races)")
        print(f"- data/training_horses.json ({len(self.horses_data)} horses)")
        print("\nReady for ML training! 🏇")

def main():
    """Main function to generate training data"""
    generator = TrainingDataGenerator()
    
    # Generate 100 historical races with results
    generator.generate_training_dataset(num_historical_races=100)
    
    print("\n✅ Training data generation complete!")
    print("You can now train your ML models with realistic race results.")

if __name__ == "__main__":
    main()