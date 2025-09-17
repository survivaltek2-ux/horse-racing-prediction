import json
import os
from datetime import datetime
from models.horse import Horse

class Race:
    """Class representing a horse race with its details and participants"""
    
    # Path to store race data
    DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'races.json')
    
    def __init__(self, id=None, name=None, date=None, location=None, distance=None, 
                 track_condition=None, race_type=None, purse=None, horse_ids=None, 
                 results=None, status="upcoming", **kwargs):
        """Initialize a race with its attributes"""
        self.id = id
        self.name = name
        self.date = date
        self.location = location
        self.distance = distance
        self.track_condition = track_condition
        self.race_type = race_type
        self.purse = purse or 0.0
        self.horse_ids = horse_ids or []
        self.results = results or {}
        self.status = status  # "upcoming", "in_progress", "completed"
        
        # Additional attributes can be passed as kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def get_all_races(cls):
        """Retrieve all races from the data store"""
        try:
            if not os.path.exists(cls.DATA_FILE):
                # Create data directory if it doesn't exist
                os.makedirs(os.path.dirname(cls.DATA_FILE), exist_ok=True)
                return []
                
            with open(cls.DATA_FILE, 'r') as f:
                races_data = json.load(f)
                
            return [cls(**race_data) for race_data in races_data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    @classmethod
    def get_race_by_id(cls, race_id):
        """Get a specific race by ID"""
        races = cls.get_all_races()
        for race in races:
            if race.id == race_id:
                return race
        return None
    
    @classmethod
    def get_upcoming_races(cls):
        """Get all upcoming races"""
        races = cls.get_all_races()
        return [race for race in races if race.status == "upcoming"]
    
    @classmethod
    def create_race(cls, race_data):
        """Create a new race and save to data store"""
        races = cls.get_all_races()
        
        # Generate a new ID if not provided
        if 'id' not in race_data or not race_data['id']:
            race_data['id'] = len(races) + 1
            
        # Create the new race instance
        new_race = cls(**race_data)
        
        # Add to the list and save
        races.append(new_race)
        cls._save_races(races)
        
        return new_race
    
    def update(self, **kwargs):
        """Update race attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Save the updated data
        races = self.get_all_races()
        for i, race in enumerate(races):
            if race.id == self.id:
                races[i] = self
                break
        
        self._save_races(races)
        return self
    
    def add_horse(self, horse_id):
        """Add a horse to the race"""
        if horse_id not in self.horse_ids:
            self.horse_ids.append(horse_id)
            self.update()
        return self
    
    def remove_horse(self, horse_id):
        """Remove a horse from the race"""
        if horse_id in self.horse_ids:
            self.horse_ids.remove(horse_id)
            self.update()
        return self
    
    def get_horses(self):
        """Get all horses participating in this race"""
        return [Horse.get_horse_by_id(horse_id) for horse_id in self.horse_ids]
    
    @property
    def horses(self):
        """Property to get all horses participating in this race"""
        return self.get_horses()
    
    def record_result(self, horse_id, position, time, earnings=None):
        """Record the result for a horse in this race"""
        if not self.results:
            self.results = {}
            
        if earnings is None:
            # Calculate earnings based on position and purse
            if position == 1:
                earnings = self.purse * 0.6  # Winner gets 60%
            elif position == 2:
                earnings = self.purse * 0.2  # Second gets 20%
            elif position == 3:
                earnings = self.purse * 0.1  # Third gets 10%
            else:
                earnings = 0
        
        # Record the result
        self.results[str(horse_id)] = {
            'position': position,
            'time': time,
            'earnings': earnings
        }
        
        # Update the race status if all horses have results
        if len(self.results) == len(self.horse_ids):
            self.status = "completed"
            
        # Update the race
        self.update()
        
        # Update the horse's performance record
        horse = Horse.get_horse_by_id(horse_id)
        if horse:
            horse.add_race_result(self.id, position, time, earnings, self.date)
        
        return self
    
    def get_winner(self):
        """Get the winning horse of this race"""
        if not self.results or self.status != "completed":
            return None
            
        # Find the horse with position 1
        for horse_id, result in self.results.items():
            if result['position'] == 1:
                return Horse.get_horse_by_id(int(horse_id))
        
        return None
    
    def save(self):
        """Save this race to the data store"""
        print(f"DEBUG: Starting save for race: {self.name}")
        try:
            races = self.get_all_races()
            print(f"DEBUG: Retrieved {len(races)} existing races")
            
            # If this race doesn't have an ID, generate one
            if not self.id:
                self.id = max([race.id for race in races], default=0) + 1
                print(f"DEBUG: Generated new race ID: {self.id}")
            
            # Check if this race already exists (update) or is new (create)
            race_exists = False
            for i, race in enumerate(races):
                if race.id == self.id:
                    races[i] = self
                    race_exists = True
                    break
            
            # If race doesn't exist, add it to the list
            if not race_exists:
                races.append(self)
                print(f"DEBUG: Added new race to list. Total races: {len(races)}")
            else:
                print(f"DEBUG: Updated existing race at index {i}")
            
            # Save all races
            print(f"DEBUG: Calling _save_races with {len(races)} races")
            self._save_races(races)
            print(f"DEBUG: Successfully saved race: {self.name}")
            return self
        except Exception as e:
            print(f"DEBUG: Error in save method: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def to_dict(self):
        """Convert race object to dictionary for serialization"""
        # Convert datetime to string if it exists
        date_str = self.date.isoformat() if isinstance(self.date, datetime) else self.date
        
        race_dict = {
            'id': self.id,
            'name': self.name,
            'date': date_str,
            'location': self.location,
            'distance': self.distance,
            'track_condition': self.track_condition,
            'race_type': self.race_type,
            'purse': self.purse,
            'horse_ids': self.horse_ids,
            'results': self.results,
            'status': self.status
        }
        
        # Add any additional attributes that were set via kwargs
        # Exclude properties that contain non-serializable objects
        excluded_attrs = {'horses'}  # horses property contains Horse objects
        
        for attr_name in dir(self):
            if (not attr_name.startswith('_') and 
                attr_name not in race_dict and 
                attr_name not in excluded_attrs and
                not callable(getattr(self, attr_name))):
                
                try:
                    attr_value = getattr(self, attr_name)
                    # Only include serializable values
                    if isinstance(attr_value, (str, int, float, bool, list, dict, type(None))):
                        # For lists, check if all items are serializable
                        if isinstance(attr_value, list):
                            if all(isinstance(item, (str, int, float, bool, dict, type(None))) for item in attr_value):
                                race_dict[attr_name] = attr_value
                        else:
                            race_dict[attr_name] = attr_value
                except Exception:
                    # Skip attributes that cause errors when accessed
                    continue
        
        return race_dict
    
    @classmethod
    def _save_races(cls, races):
        """Save all races to the data store"""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.DATA_FILE), exist_ok=True)
        
        # Convert race objects to dictionaries
        races_data = [race.to_dict() for race in races]
        
        # Save to file
        with open(cls.DATA_FILE, 'w') as f:
            json.dump(races_data, f, indent=2)