#!/usr/bin/env python3
"""
Enhanced Horse Racing Prediction System
Utilizes comprehensive horse and race data for advanced predictions.
"""

import json
import random
import math
from datetime import datetime
from typing import Dict, List, Tuple

class EnhancedPredictionSystem:
    def __init__(self):
        self.horses_data = None
        self.races_data = None
        self.prediction_weights = {
            # Traditional factors
            'recent_form': 0.15,
            'speed_figures': 0.20,
            'class_rating': 0.12,
            
            # Enhanced factors
            'pedigree_rating': 0.08,
            'physical_fitness': 0.10,
            'training_form': 0.08,
            'track_suitability': 0.12,
            'pace_scenario': 0.10,
            'betting_confidence': 0.05
        }

    def load_enhanced_data(self):
        """Load enhanced horse and race data"""
        print("Loading enhanced data...")
        
        with open('data/horses_enhanced.json', 'r') as f:
            self.horses_data = {horse['id']: horse for horse in json.load(f)}
        
        with open('data/races_enhanced.json', 'r') as f:
            self.races_data = {race['id']: race for race in json.load(f)}
        
        print(f"Loaded {len(self.horses_data)} enhanced horses and {len(self.races_data)} enhanced races")

    def calculate_pedigree_score(self, horse):
        """Calculate pedigree-based performance score"""
        pedigree = horse.get('pedigree', {})
        
        # Base score from bloodline rating
        base_score = pedigree.get('bloodline_rating', 75) / 100
        
        # Adjust for sire/dam earnings
        sire_earnings = pedigree.get('sire_earnings', 0)
        dam_earnings = pedigree.get('dam_earnings', 0)
        
        earnings_bonus = min((sire_earnings + dam_earnings) / 5000000, 0.2)
        
        # Dosage index consideration (optimal around 2.0 for versatility)
        dosage = pedigree.get('dosage_index', 2.0)
        dosage_factor = 1.0 - abs(dosage - 2.0) * 0.1
        
        return (base_score + earnings_bonus) * dosage_factor

    def calculate_physical_fitness_score(self, horse):
        """Calculate physical condition and fitness score"""
        physical = horse.get('physical_stats', {})
        health = horse.get('health_data', {})
        
        # Conformation score
        conformation = physical.get('conformation_score', 80) / 100
        
        # Health status
        health_status = health.get('overall_health', 'Good')
        health_multiplier = {
            'Excellent': 1.0,
            'Good': 0.9,
            'Minor issues': 0.8,
            'Recovering': 0.7,
            'Under monitoring': 0.6
        }.get(health_status, 0.8)
        
        # Fitness level
        fitness = health.get('fitness_level', 80) / 100
        
        # Recovery rate
        recovery = health.get('recovery_rate', 85) / 100
        
        return (conformation * 0.3 + fitness * 0.4 + recovery * 0.3) * health_multiplier

    def calculate_training_form_score(self, horse):
        """Calculate training and workout form score"""
        training = horse.get('training_data', {})
        
        # Fitness level
        fitness = training.get('fitness_level', 80) / 100
        
        # Workout rating
        gallop_rating = training.get('gallop_rating', 80) / 100
        
        # Gate training (important for race start)
        gate_score = training.get('gate_training_score', 80) / 100
        
        # Recent workout (fresher is better, but not too fresh)
        last_workout = training.get('last_workout_date', '2025-01-01')
        try:
            days_since = (datetime.now() - datetime.strptime(last_workout, '%Y-%m-%d')).days
            workout_freshness = max(0.5, 1.0 - (days_since - 7) * 0.05) if days_since <= 14 else 0.3
        except:
            workout_freshness = 0.7
        
        return (fitness * 0.4 + gallop_rating * 0.3 + gate_score * 0.2 + workout_freshness * 0.1)

    def calculate_track_suitability_score(self, horse, race):
        """Calculate how well horse suits the race conditions"""
        preferences = horse.get('racing_preferences', {})
        track_data = race.get('track_data', {})
        weather_data = race.get('weather_data', {})
        
        # Surface preference
        surface = track_data.get('surface', 'Dirt').lower()
        surface_prefs = preferences.get('surface_preference', {})
        surface_score = surface_prefs.get(surface, 0.8)
        
        # Track condition preference
        condition = track_data.get('condition', 'Good').lower()
        condition_prefs = preferences.get('track_condition_preference', {})
        condition_score = condition_prefs.get(condition, 0.8)
        
        # Distance preference
        distance_pref = preferences.get('preferred_distance', 'Mile')
        race_distance = race.get('distance', '1200m')
        
        # Simple distance matching
        distance_score = 0.9  # Default good match
        if 'sprint' in distance_pref.lower() and ('1000' in race_distance or '1200' in race_distance):
            distance_score = 1.0
        elif 'mile' in distance_pref.lower() and ('1400' in race_distance or '1600' in race_distance):
            distance_score = 1.0
        elif 'route' in distance_pref.lower() and ('1800' in race_distance or '2000' in race_distance):
            distance_score = 1.0
        
        # Weather tolerance
        temp = weather_data.get('temperature_f', 70)
        health_data = horse.get('health_data', {})
        heat_tolerance = health_data.get('heat_tolerance', 80) / 100
        
        temp_score = 1.0
        if temp > 85:
            temp_score = heat_tolerance
        elif temp < 50:
            temp_score = min(1.0, heat_tolerance + 0.2)
        
        return (surface_score * 0.3 + condition_score * 0.3 + distance_score * 0.3 + temp_score * 0.1)

    def calculate_pace_scenario_score(self, horse, race):
        """Calculate how horse fits the projected pace scenario"""
        preferences = horse.get('racing_preferences', {})
        pace_analysis = race.get('pace_analysis', {})
        behavioral = horse.get('behavioral_data', {})
        
        running_style = preferences.get('running_style', 'Stalker')
        pace_scenario = pace_analysis.get('projected_early_pace', 'Moderate')
        pace_advantage = pace_analysis.get('pace_advantage', 'Balanced')
        
        # Style vs pace scenario matching
        style_scores = {
            'Front runner': {
                'Slow': 1.0, 'Moderate': 0.9, 'Fast': 0.6, 'Very Fast': 0.4
            },
            'Stalker': {
                'Slow': 0.8, 'Moderate': 1.0, 'Fast': 0.9, 'Very Fast': 0.7
            },
            'Closer': {
                'Slow': 0.6, 'Moderate': 0.8, 'Fast': 1.0, 'Very Fast': 1.0
            },
            'Deep closer': {
                'Slow': 0.4, 'Moderate': 0.6, 'Fast': 0.9, 'Very Fast': 1.0
            }
        }
        
        pace_score = style_scores.get(running_style, {}).get(pace_scenario, 0.8)
        
        # Pace advantage bonus
        advantage_bonus = 1.0
        if pace_advantage == 'Speed' and 'Front' in running_style:
            advantage_bonus = 1.2
        elif pace_advantage == 'Stalkers' and running_style == 'Stalker':
            advantage_bonus = 1.2
        elif pace_advantage == 'Closers' and 'Closer' in running_style:
            advantage_bonus = 1.2
        
        # Temperament factor
        temperament = behavioral.get('temperament', 'Balanced')
        temp_factor = {
            'Calm': 1.0, 'Balanced': 1.0, 'Energetic': 0.95,
            'Nervous': 0.85, 'Aggressive': 0.9
        }.get(temperament, 0.9)
        
        return pace_score * advantage_bonus * temp_factor

    def calculate_betting_confidence_score(self, horse_id, race):
        """Calculate confidence based on betting patterns"""
        betting_data = race.get('betting_data', {})
        morning_line_odds = betting_data.get('morning_line_odds', {})
        
        horse_odds = morning_line_odds.get(str(horse_id), 10.0)
        
        # Convert odds to implied probability
        implied_prob = 1 / (horse_odds + 1)
        
        # Public confidence factor
        public_confidence = betting_data.get('public_confidence', 0.7)
        
        # Betting interest
        betting_interest = betting_data.get('betting_interest', 'Moderate')
        interest_multiplier = {
            'Low': 0.9, 'Moderate': 1.0, 'High': 1.1, 'Very High': 1.2
        }.get(betting_interest, 1.0)
        
        return implied_prob * public_confidence * interest_multiplier

    def calculate_speed_figure_score(self, horse):
        """Enhanced speed figure calculation"""
        speed_figs = horse.get('speed_figures', {})
        
        # Multiple speed figure sources
        beyer = speed_figs.get('beyer_speed_figure', 80)
        timeform = speed_figs.get('timeform_rating', 90)
        speed_index = speed_figs.get('speed_index', 90)
        
        # Normalize and combine
        beyer_norm = min(beyer / 120, 1.0)
        timeform_norm = min(timeform / 140, 1.0)
        speed_norm = min(speed_index / 120, 1.0)
        
        # Weighted average
        combined_speed = (beyer_norm * 0.4 + timeform_norm * 0.4 + speed_norm * 0.2)
        
        # Consistency bonus
        consistency = speed_figs.get('consistency_index', 0.7)
        
        return combined_speed * (0.8 + consistency * 0.2)

    def calculate_class_rating_score(self, horse, race):
        """Calculate class rating based on race level and horse experience"""
        speed_figs = horse.get('speed_figures', {})
        class_rating = speed_figs.get('class_rating', 80) / 105  # Normalize
        
        # Race class level
        race_type = race.get('race_type', 'Allowance')
        race_purse = race.get('purse', 50000)
        
        # Class level hierarchy
        class_hierarchy = {
            'Maiden': 1, 'Claiming': 2, 'Allowance': 3, 'Stakes': 4,
            'Listed': 5, 'Group 3': 6, 'Group 2': 7, 'Group 1': 8
        }
        
        race_class_level = class_hierarchy.get(race_type, 3)
        
        # Horse's preferred class level
        preferences = horse.get('racing_preferences', {})
        horse_class = preferences.get('class_level', 'Allowance')
        horse_class_level = class_hierarchy.get(horse_class, 3)
        
        # Class match factor
        class_diff = abs(race_class_level - horse_class_level)
        class_match = max(0.6, 1.0 - class_diff * 0.1)
        
        return class_rating * class_match

    def calculate_recent_form_score(self, horse):
        """Enhanced recent form calculation"""
        performances = horse.get('recent_performances', [])
        
        if not performances:
            return 0.5
        
        # Weight recent performances more heavily
        weights = [0.4, 0.3, 0.2, 0.1]  # Last 4 races
        form_score = 0
        total_weight = 0
        
        for i, perf in enumerate(performances[:4]):
            if i >= len(weights):
                break
                
            position = perf.get('position', 10)
            total_runners = perf.get('total_runners', 10)
            
            # Position score (1st = 1.0, last = 0.0)
            pos_score = max(0, (total_runners - position + 1) / total_runners)
            
            # Margin consideration
            margin = perf.get('margin', 5)
            if isinstance(margin, str):
                try:
                    margin = float(margin)
                except:
                    margin = 5
            
            # Close finishes get bonus
            margin_factor = 1.0
            if position <= 3 and margin <= 2:
                margin_factor = 1.1
            elif margin > 10:
                margin_factor = 0.9
            
            form_score += pos_score * margin_factor * weights[i]
            total_weight += weights[i]
        
        return form_score / total_weight if total_weight > 0 else 0.5

    def predict_race(self, race_id):
        """Generate comprehensive race prediction"""
        race = self.races_data.get(race_id)
        if not race:
            return None
        
        predictions = []
        
        for horse_id in race['horse_ids']:
            horse = self.horses_data.get(horse_id)
            if not horse:
                continue
            
            # Calculate all scoring components
            scores = {
                'recent_form': self.calculate_recent_form_score(horse),
                'speed_figures': self.calculate_speed_figure_score(horse),
                'class_rating': self.calculate_class_rating_score(horse, race),
                'pedigree_rating': self.calculate_pedigree_score(horse),
                'physical_fitness': self.calculate_physical_fitness_score(horse),
                'training_form': self.calculate_training_form_score(horse),
                'track_suitability': self.calculate_track_suitability_score(horse, race),
                'pace_scenario': self.calculate_pace_scenario_score(horse, race),
                'betting_confidence': self.calculate_betting_confidence_score(horse_id, race)
            }
            
            # Calculate weighted total score
            total_score = sum(scores[factor] * weight for factor, weight in self.prediction_weights.items())
            
            # Convert to win probability
            win_probability = min(0.95, max(0.01, total_score))
            
            predictions.append({
                'horse_id': horse_id,
                'horse_name': horse['name'],
                'total_score': round(total_score, 3),
                'win_probability': round(win_probability, 3),
                'detailed_scores': {k: round(v, 3) for k, v in scores.items()},
                'confidence': 'High' if win_probability > 0.7 else 'Medium' if win_probability > 0.4 else 'Low'
            })
        
        # Sort by win probability
        predictions.sort(key=lambda x: x['win_probability'], reverse=True)
        
        # Normalize probabilities to sum to 1
        total_prob = sum(p['win_probability'] for p in predictions)
        if total_prob > 0:
            for pred in predictions:
                pred['normalized_probability'] = round(pred['win_probability'] / total_prob, 3)
        
        return {
            'race_id': race_id,
            'race_name': race['name'],
            'race_date': race['date'],
            'predictions': predictions,
            'race_analysis': {
                'field_size': len(race['horse_ids']),
                'field_quality': race.get('field_analysis', {}).get('field_quality', 'Unknown'),
                'pace_scenario': race.get('pace_analysis', {}).get('projected_early_pace', 'Unknown'),
                'weather': race.get('weather_data', {}).get('condition', 'Unknown'),
                'track_condition': race.get('track_data', {}).get('condition', 'Unknown')
            }
        }

    def analyze_all_races(self):
        """Analyze and predict all races"""
        print("Generating enhanced predictions for all races...")
        
        all_predictions = []
        
        for race_id in self.races_data.keys():
            prediction = self.predict_race(race_id)
            if prediction:
                all_predictions.append(prediction)
        
        return all_predictions

def main():
    predictor = EnhancedPredictionSystem()
    predictor.load_enhanced_data()
    
    # Generate predictions for all races
    all_predictions = predictor.analyze_all_races()
    
    # Save results
    with open('enhanced_predictions_results.json', 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    # Display summary
    print(f"\n{'='*80}")
    print("ENHANCED PREDICTION SYSTEM RESULTS")
    print(f"{'='*80}")
    print(f"Total races analyzed: {len(all_predictions)}")
    
    # Show top 3 races with highest confidence predictions
    confident_races = sorted(all_predictions, 
                           key=lambda x: x['predictions'][0]['win_probability'] if x['predictions'] else 0, 
                           reverse=True)[:3]
    
    for i, race in enumerate(confident_races, 1):
        print(f"\n{i}. {race['race_name']} ({race['race_date']})")
        print(f"   Field: {race['race_analysis']['field_size']} horses, {race['race_analysis']['field_quality']} quality")
        print(f"   Conditions: {race['race_analysis']['track_condition']} track, {race['race_analysis']['weather']} weather")
        
        top_3 = race['predictions'][:3]
        for j, pred in enumerate(top_3, 1):
            print(f"   {j}. {pred['horse_name']} - {pred['win_probability']:.1%} ({pred['confidence']} confidence)")
    
    print(f"\nDetailed results saved to: enhanced_predictions_results.json")

if __name__ == "__main__":
    main()