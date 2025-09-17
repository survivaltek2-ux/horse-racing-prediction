import pandas as pd
import os
import json
from datetime import datetime

class Horse:
    """Class representing a horse with its racing attributes and history"""
    
    # Path to store horse data
    DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'horses.json')
    
    def __init__(self, id=None, name=None, age=None, sex=None, color=None, breed=None, jockey=None, trainer=None, 
                 win_rate=None, place_rate=None, show_rate=None, earnings=None, 
                 recent_performances=None, 
                 # Speed Rating Fields
                 speed_rating_race_1=None, speed_rating_race_2=None, speed_rating_race_3=None,
                 highest_speed_rating_distance=None,
                 # Beaten Lengths Fields
                 beaten_lengths_race_1=None, beaten_lengths_race_2=None, beaten_lengths_race_3=None,
                 # Pace Analysis Fields
                 running_style=None, pace_versatility=None, early_pace_ability=None, late_pace_ability=None,
                 # Pedigree Fields
                 sire=None, dam=None, pedigree_speed_rating=None, pedigree_stamina_rating=None,
                 distance_pedigree_suitability=None,
                 # Connections Fields
                 owner=None, trainer_win_percentage=None, trainer_win_percentage_distance=None,
                 trainer_win_percentage_surface=None, jockey_win_percentage=None, trainer_jockey_combo_wins=None,
                 # Race Assignment Fields
                 assigned_races=None, post_position=None, morning_line_odds=None,
                 **kwargs):
        """Initialize a horse with its racing performance attributes"""
        # Basic Information
        self.id = id
        self.name = name
        self.age = age
        self.sex = sex
        self.color = color
        self.breed = breed
        self.jockey = jockey
        self.trainer = trainer
        self.win_rate = win_rate or 0.0
        self.place_rate = place_rate or 0.0
        self.show_rate = show_rate or 0.0
        self.earnings = earnings or 0.0
        self.recent_performances = recent_performances or []
        
        # Speed Rating Fields
        self.speed_rating_race_1 = speed_rating_race_1
        self.speed_rating_race_2 = speed_rating_race_2
        self.speed_rating_race_3 = speed_rating_race_3
        self.highest_speed_rating_distance = highest_speed_rating_distance
        
        # Beaten Lengths Fields
        self.beaten_lengths_race_1 = beaten_lengths_race_1
        self.beaten_lengths_race_2 = beaten_lengths_race_2
        self.beaten_lengths_race_3 = beaten_lengths_race_3
        
        # Pace Analysis Fields
        self.running_style = running_style
        self.pace_versatility = pace_versatility
        self.early_pace_ability = early_pace_ability
        self.late_pace_ability = late_pace_ability
        
        # Pedigree Fields
        self.sire = sire
        self.dam = dam
        self.pedigree_speed_rating = pedigree_speed_rating
        self.pedigree_stamina_rating = pedigree_stamina_rating
        self.distance_pedigree_suitability = distance_pedigree_suitability
        
        # Connections Fields
        self.owner = owner
        self.trainer_win_percentage = trainer_win_percentage
        self.trainer_win_percentage_distance = trainer_win_percentage_distance
        self.trainer_win_percentage_surface = trainer_win_percentage_surface
        self.jockey_win_percentage = jockey_win_percentage
        self.trainer_jockey_combo_wins = trainer_jockey_combo_wins
        
        # Race Assignment Fields
        self.assigned_races = assigned_races
        self.post_position = post_position
        self.morning_line_odds = morning_line_odds
        
        # Additional attributes can be passed as kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def get_all_horses(cls):
        """Retrieve all horses from the data store"""
        try:
            if not os.path.exists(cls.DATA_FILE):
                # Create data directory if it doesn't exist
                os.makedirs(os.path.dirname(cls.DATA_FILE), exist_ok=True)
                return []
                
            with open(cls.DATA_FILE, 'r') as f:
                horses_data = json.load(f)
                
            return [cls(**horse_data) for horse_data in horses_data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    @classmethod
    def get_horse_by_id(cls, horse_id):
        """Get a specific horse by ID"""
        horses = cls.get_all_horses()
        for horse in horses:
            if horse.id == horse_id:
                return horse
        return None
    
    @classmethod
    def create_horse(cls, horse_data):
        """Create a new horse and save to data store"""
        horses = cls.get_all_horses()
        
        # Generate a new ID if not provided
        if 'id' not in horse_data or not horse_data['id']:
            horse_data['id'] = len(horses) + 1
            
        # Create the new horse instance
        new_horse = cls(**horse_data)
        
        # Add to the list and save
        horses.append(new_horse)
        cls._save_horses(horses)
        
        return new_horse
    
    def update(self, **kwargs):
        """Update horse attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Save the updated data
        horses = self.get_all_horses()
        for i, horse in enumerate(horses):
            if horse.id == self.id:
                horses[i] = self
                break
        
        self._save_horses(horses)
        return self
    
    def add_race_result(self, race_id, position, time, earnings, race_date=None):
        """Add a new race result to the horse's performance history"""
        if race_date is None:
            race_date = datetime.now().strftime('%Y-%m-%d')
            
        result = {
            'race_id': race_id,
            'position': position,
            'time': time,
            'earnings': earnings,
            'date': race_date
        }
        
        if not hasattr(self, 'recent_performances') or self.recent_performances is None:
            self.recent_performances = []
            
        self.recent_performances.append(result)
        
        # Update statistics
        self._update_statistics()
        
        # Save changes
        self.update()
        
        return result
    
    def _update_statistics(self):
        """Update horse statistics based on race results"""
        if not self.recent_performances:
            return
            
        total_races = len(self.recent_performances)
        wins = sum(1 for result in self.recent_performances if result['position'] == 1)
        places = sum(1 for result in self.recent_performances if result['position'] == 2)
        shows = sum(1 for result in self.recent_performances if result['position'] == 3)
        
        self.win_rate = wins / total_races if total_races > 0 else 0
        self.place_rate = places / total_races if total_races > 0 else 0
        self.show_rate = shows / total_races if total_races > 0 else 0
        
        # Update total earnings
        self.earnings = sum(result['earnings'] for result in self.recent_performances)
    
    def get_form(self, num_races=5):
        """Get the horse's recent form (last N races)"""
        if not self.recent_performances:
            return []
            
        # Sort by date, most recent first
        sorted_performances = sorted(
            self.recent_performances, 
            key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'),
            reverse=True
        )
        
        return sorted_performances[:num_races]
    
    def to_dict(self):
        """Convert horse object to dictionary for serialization"""
        return {
            # Basic Information
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'sex': self.sex,
            'color': self.color,
            'breed': self.breed,
            'jockey': self.jockey,
            'trainer': self.trainer,
            'win_rate': self.win_rate,
            'place_rate': self.place_rate,
            'show_rate': self.show_rate,
            'earnings': self.earnings,
            'recent_performances': self.recent_performances,
            
            # Speed Rating Fields
            'speed_rating_race_1': getattr(self, 'speed_rating_race_1', None),
            'speed_rating_race_2': getattr(self, 'speed_rating_race_2', None),
            'speed_rating_race_3': getattr(self, 'speed_rating_race_3', None),
            'highest_speed_rating_distance': getattr(self, 'highest_speed_rating_distance', None),
            
            # Beaten Lengths Fields
            'beaten_lengths_race_1': getattr(self, 'beaten_lengths_race_1', None),
            'beaten_lengths_race_2': getattr(self, 'beaten_lengths_race_2', None),
            'beaten_lengths_race_3': getattr(self, 'beaten_lengths_race_3', None),
            
            # Pace Analysis Fields
            'running_style': getattr(self, 'running_style', None),
            'pace_versatility': getattr(self, 'pace_versatility', None),
            'early_pace_ability': getattr(self, 'early_pace_ability', None),
            'late_pace_ability': getattr(self, 'late_pace_ability', None),
            
            # Pedigree Fields
            'sire': getattr(self, 'sire', None),
            'dam': getattr(self, 'dam', None),
            'pedigree_speed_rating': getattr(self, 'pedigree_speed_rating', None),
            'pedigree_stamina_rating': getattr(self, 'pedigree_stamina_rating', None),
            'distance_pedigree_suitability': getattr(self, 'distance_pedigree_suitability', None),
            
            # Connections Fields
            'owner': getattr(self, 'owner', None),
            'trainer_win_percentage': getattr(self, 'trainer_win_percentage', None),
            'trainer_win_percentage_distance': getattr(self, 'trainer_win_percentage_distance', None),
            'trainer_win_percentage_surface': getattr(self, 'trainer_win_percentage_surface', None),
            'jockey_win_percentage': getattr(self, 'jockey_win_percentage', None),
            'trainer_jockey_combo_wins': getattr(self, 'trainer_jockey_combo_wins', None),
            
            # Race Assignment Fields
            'assigned_races': getattr(self, 'assigned_races', None),
            'post_position': getattr(self, 'post_position', None),
            'morning_line_odds': getattr(self, 'morning_line_odds', None)
        }
    
    @classmethod
    def _save_horses(cls, horses):
        """Save all horses to the data store"""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.DATA_FILE), exist_ok=True)
        
        # Convert horse objects to dictionaries
        horses_data = [horse.to_dict() for horse in horses]
        
        # Save to file
        with open(cls.DATA_FILE, 'w') as f:
            json.dump(horses_data, f, indent=2)