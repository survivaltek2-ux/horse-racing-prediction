import json
import os
from datetime import datetime
from models.sqlalchemy_models import Race, Horse

class Prediction:
    """Class representing a race prediction with probabilities for each horse"""
    
    # Path to store prediction data
    DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'predictions.json')
    
    def __init__(self, id=None, race_id=None, date=None, predictions=None, 
                 algorithm=None, accuracy=None, actual_results=None, **kwargs):
        """Initialize a prediction with its attributes"""
        self.id = id
        self.race_id = race_id
        self.date = date or datetime.now().strftime('%Y-%m-%d')
        self.predictions = predictions or {}  # {horse_id: {'win_prob': 0.xx, 'place_prob': 0.xx, 'show_prob': 0.xx}}
        self.algorithm = algorithm or 'default'
        self.accuracy = accuracy  # Set after race is completed
        self.actual_results = actual_results  # Actual race results
        
        # Additional attributes can be passed as kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def get_all_predictions(cls):
        """Retrieve all predictions from the data store"""
        try:
            if not os.path.exists(cls.DATA_FILE):
                # Create data directory if it doesn't exist
                os.makedirs(os.path.dirname(cls.DATA_FILE), exist_ok=True)
                return []
                
            with open(cls.DATA_FILE, 'r') as f:
                predictions_data = json.load(f)
                
            return [cls(**prediction_data) for prediction_data in predictions_data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    @classmethod
    def get_prediction_by_id(cls, prediction_id):
        """Get a specific prediction by ID"""
        predictions = cls.get_all_predictions()
        for prediction in predictions:
            if prediction.id == prediction_id:
                return prediction
        return None
    
    @classmethod
    def get_predictions_for_race(cls, race_id):
        """Get all predictions for a specific race"""
        predictions = cls.get_all_predictions()
        return [pred for pred in predictions if pred.race_id == race_id]
    
    @classmethod
    def create_prediction(cls, prediction_data):
        """Create a new prediction and save to data store"""
        predictions = cls.get_all_predictions()
        
        # Generate a new ID if not provided
        if 'id' not in prediction_data or not prediction_data['id']:
            prediction_data['id'] = len(predictions) + 1
            
        # Create the new prediction instance
        new_prediction = cls(**prediction_data)
        
        # Add to the list and save
        predictions.append(new_prediction)
        cls._save_predictions(predictions)
        
        return new_prediction
    
    def update(self, **kwargs):
        """Update prediction attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Save the updated data
        predictions = self.get_all_predictions()
        for i, prediction in enumerate(predictions):
            if prediction.id == self.id:
                predictions[i] = self
                break
        
        self._save_predictions(predictions)
        return self
    
    def get_race(self):
        """Get the race this prediction is for"""
        return Race.get_by_id(self.race_id)
    
    def get_top_picks(self, limit=3):
        """Get the top picks (horses with highest win probability)"""
        if not self.predictions:
            return []
            
        # Sort horses by win probability
        sorted_horses = sorted(
            self.predictions.items(),
            key=lambda x: x[1]['win_prob'],
            reverse=True
        )
        
        # Get the top N horses
        top_horses = []
        for horse_id, probs in sorted_horses[:limit]:
            horse = Horse.get_by_id(int(horse_id))
            if horse:
                top_horses.append({
                    'horse': horse,
                    'win_probability': probs['win_prob'],
                    'place_probability': probs.get('place_prob', 0),
                    'show_probability': probs.get('show_prob', 0)
                })
                
        return top_horses
    
    def evaluate_accuracy(self):
        """Evaluate the accuracy of this prediction after the race is completed"""
        race = self.get_race()
        if not race or race.status != "completed":
            return None
            
        # Get the actual results
        self.actual_results = race.results
        
        # Calculate accuracy based on whether the top pick won
        top_picks = self.get_top_picks(1)
        if not top_picks:
            self.accuracy = 0
            return self.accuracy
            
        top_horse = top_picks[0]['horse']
        
        # Check if the top pick won
        winner_id = None
        for horse_id, result in race.results.items():
            if result['position'] == 1:
                winner_id = int(horse_id)
                break
                
        if winner_id == top_horse.id:
            self.accuracy = 1.0
        else:
            # Calculate partial accuracy based on position
            for horse_id, result in race.results.items():
                if int(horse_id) == top_horse.id:
                    position = result['position']
                    if position == 2:
                        self.accuracy = 0.5  # 50% accuracy for second place
                    elif position == 3:
                        self.accuracy = 0.3  # 30% accuracy for third place
                    else:
                        self.accuracy = 0
                    break
            else:
                self.accuracy = 0
                
        # Update the prediction
        self.update()
        
        return self.accuracy
    
    @classmethod
    def get_prediction_history(cls):
        """Get history of predictions with their accuracy"""
        predictions = cls.get_all_predictions()
        
        # Sort by date, most recent first
        return sorted(
            predictions,
            key=lambda x: datetime.strptime(x.date, '%Y-%m-%d'),
            reverse=True
        )
    
    def to_dict(self):
        """Convert prediction object to dictionary for serialization"""
        return {
            'id': self.id,
            'race_id': self.race_id,
            'date': self.date,
            'predictions': self.predictions,
            'algorithm': self.algorithm,
            'accuracy': self.accuracy,
            'actual_results': self.actual_results
        }
    
    @classmethod
    def _save_predictions(cls, predictions):
        """Save all predictions to the data store"""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.DATA_FILE), exist_ok=True)
        
        # Convert prediction objects to dictionaries
        predictions_data = [prediction.to_dict() for prediction in predictions]
        
        # Save to file
        with open(cls.DATA_FILE, 'w') as f:
            json.dump(predictions_data, f, indent=2)