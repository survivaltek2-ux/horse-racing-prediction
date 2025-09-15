import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from models.horse import Horse
from models.race import Race
from models.prediction import Prediction
from utils.data_processor import DataProcessor

class Predictor:
    """Main prediction engine for horse racing"""
    
    def __init__(self):
        """Initialize the predictor"""
        self.data_processor = DataProcessor()
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def predict_race(self, race, params=None):
        """Generate predictions for a race"""
        if not race:
            return None
            
        # Prepare the race data
        race_data = self.data_processor.prepare_race_data(race)
        if race_data is None or race_data.empty:
            return None
        
        # Use simple heuristic-based prediction if no trained model
        predictions = self._heuristic_prediction(race_data, race)
        
        # Create and save the prediction
        prediction = Prediction.create_prediction({
            'race_id': race.id,
            'predictions': predictions,
            'algorithm': 'heuristic'
        })
        
        return prediction
    
    def _heuristic_prediction(self, race_data, race):
        """Generate predictions using heuristic rules"""
        predictions = {}
        
        # Calculate scores for each horse
        horse_scores = []
        
        for _, horse_row in race_data.iterrows():
            score = 0
            
            # Win rate contribution (40% weight)
            score += horse_row['win_rate'] * 0.4
            
            # Recent form contribution (30% weight)
            if horse_row['avg_position'] > 0:
                # Lower average position is better (1st place = 1, 2nd = 2, etc.)
                position_score = max(0, (10 - horse_row['avg_position']) / 10)
                score += position_score * 0.3
            
            # Consistency contribution (15% weight)
            if horse_row['position_std'] > 0:
                # Lower standard deviation is better (more consistent)
                consistency_score = max(0, (5 - horse_row['position_std']) / 5)
                score += consistency_score * 0.15
            
            # Age factor (10% weight)
            # Horses aged 3-6 are typically in their prime
            age = horse_row['age']
            if 3 <= age <= 6:
                age_score = 1.0
            elif age < 3:
                age_score = 0.7  # Young horses
            elif age <= 8:
                age_score = 0.8  # Older but experienced
            else:
                age_score = 0.5  # Very old horses
            
            score += age_score * 0.1
            
            # Track condition adjustment (5% weight)
            score += horse_row['track_condition'] * 0.05
            
            horse_scores.append({
                'horse_id': int(horse_row['horse_id']),
                'score': max(0, min(1, score))  # Normalize between 0 and 1
            })
        
        # Sort horses by score
        horse_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Convert scores to probabilities
        total_score = sum(horse['score'] for horse in horse_scores)
        if total_score == 0:
            # If all scores are 0, assign equal probabilities
            equal_prob = 1.0 / len(horse_scores)
            for horse in horse_scores:
                horse['win_prob'] = equal_prob
        else:
            # Normalize scores to probabilities
            for horse in horse_scores:
                horse['win_prob'] = horse['score'] / total_score
        
        # Calculate place and show probabilities
        for i, horse in enumerate(horse_scores):
            # Place probability (1st or 2nd)
            if i == 0:  # Favorite
                horse['place_prob'] = min(0.9, horse['win_prob'] + 0.3)
            elif i == 1:  # Second favorite
                horse['place_prob'] = min(0.8, horse['win_prob'] + 0.25)
            else:
                horse['place_prob'] = min(0.7, horse['win_prob'] + 0.2)
            
            # Show probability (1st, 2nd, or 3rd)
            if i <= 2:  # Top 3 favorites
                horse['show_prob'] = min(0.95, horse['place_prob'] + 0.2)
            else:
                horse['show_prob'] = min(0.8, horse['place_prob'] + 0.15)
        
        # Build predictions dictionary
        for horse in horse_scores:
            predictions[str(horse['horse_id'])] = {
                'win_prob': round(horse['win_prob'], 3),
                'place_prob': round(horse['place_prob'], 3),
                'show_prob': round(horse['show_prob'], 3)
            }
        
        return predictions
    
    def get_performance_stats(self):
        """Get performance statistics for the predictor"""
        predictions = Prediction.get_all_predictions()
        
        if not predictions:
            return {
                'total_predictions': 0,
                'accuracy': 0,
                'avg_accuracy': 0,
                'win_predictions': 0,
                'place_predictions': 0,
                'show_predictions': 0
            }
        
        # Calculate statistics
        total_predictions = len(predictions)
        accurate_predictions = sum(1 for pred in predictions if pred.accuracy and pred.accuracy >= 1.0)
        
        # Calculate average accuracy for completed predictions
        completed_predictions = [pred for pred in predictions if pred.accuracy is not None]
        avg_accuracy = sum(pred.accuracy for pred in completed_predictions) / len(completed_predictions) if completed_predictions else 0
        
        # Count different types of successful predictions
        win_predictions = sum(1 for pred in predictions if pred.accuracy and pred.accuracy >= 1.0)
        place_predictions = sum(1 for pred in predictions if pred.accuracy and pred.accuracy >= 0.5)
        show_predictions = sum(1 for pred in predictions if pred.accuracy and pred.accuracy >= 0.3)
        
        return {
            'total_predictions': total_predictions,
            'accuracy': round(accurate_predictions / total_predictions * 100, 1) if total_predictions > 0 else 0,
            'avg_accuracy': round(avg_accuracy * 100, 1),
            'win_predictions': win_predictions,
            'place_predictions': place_predictions,
            'show_predictions': show_predictions,
            'completed_predictions': len(completed_predictions)
        }
    
    def train_model(self, training_data=None):
        """Train the prediction model with historical data"""
        # This would be implemented with actual historical data
        # For now, we'll use the heuristic approach
        self.is_trained = True
        return True