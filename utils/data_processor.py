import pandas as pd
import numpy as np
from models.horse import Horse
from models.race import Race

class DataProcessor:
    """Utility class for processing horse racing data for prediction"""
    
    def __init__(self):
        """Initialize the data processor"""
        pass
    
    def prepare_race_data(self, race):
        """Prepare data for a specific race for prediction"""
        if not race:
            return None
            
        # Get all horses in the race
        horses = race.get_horses()
        if not horses:
            return None
            
        # Create a dataframe with horse features
        data = []
        for horse in horses:
            # Extract basic horse features
            horse_data = {
                'horse_id': horse.id,
                'age': horse.age,
                'win_rate': horse.win_rate,
                'place_rate': horse.place_rate,
                'show_rate': horse.show_rate,
                'earnings': horse.earnings,
            }
            
            # Add recent performance metrics
            recent_form = horse.get_form(5)  # Get last 5 races
            
            # Calculate average finish position
            if recent_form:
                avg_position = sum(result['position'] for result in recent_form) / len(recent_form)
                horse_data['avg_position'] = avg_position
                
                # Calculate consistency (standard deviation of positions)
                if len(recent_form) > 1:
                    positions = [result['position'] for result in recent_form]
                    horse_data['position_std'] = np.std(positions)
                else:
                    horse_data['position_std'] = 0
                    
                # Calculate average time (if available and comparable)
                times = [float(result['time']) for result in recent_form if 'time' in result and result['time']]
                if times:
                    horse_data['avg_time'] = sum(times) / len(times)
                else:
                    horse_data['avg_time'] = 0
                    
                # Calculate recent form trend (improvement or decline)
                if len(recent_form) >= 3:
                    # More weight to recent races
                    weighted_positions = [
                        recent_form[0]['position'] * 0.5,  # Most recent race
                        recent_form[1]['position'] * 0.3,  # Second most recent
                        recent_form[2]['position'] * 0.2   # Third most recent
                    ]
                    horse_data['recent_trend'] = sum(weighted_positions)
                else:
                    horse_data['recent_trend'] = avg_position
            else:
                # No recent form data
                horse_data['avg_position'] = 0
                horse_data['position_std'] = 0
                horse_data['avg_time'] = 0
                horse_data['recent_trend'] = 0
            
            # Add race-specific features
            horse_data['track_condition'] = self._encode_track_condition(race.track_condition)
            horse_data['distance'] = self._normalize_distance(race.distance)
            
            # Add the horse data to our dataset
            data.append(horse_data)
        
        # Convert to dataframe
        return pd.DataFrame(data)
    
    def _encode_track_condition(self, condition):
        """Encode track condition as a numerical value"""
        condition_map = {
            'fast': 1.0,
            'good': 0.9,
            'yielding': 0.8,
            'soft': 0.7,
            'heavy': 0.6,
            'slow': 0.5,
            'sloppy': 0.4,
            'muddy': 0.3
        }
        
        # Default to 'fast' if condition is unknown
        return condition_map.get(condition.lower() if condition else '', 1.0)
    
    def _normalize_distance(self, distance):
        """Normalize race distance to a standard format (in meters)"""
        if not distance:
            return 0
            
        # Handle different distance formats
        try:
            # If it's already a number, assume it's in meters
            return float(distance)
        except ValueError:
            # Try to parse common formats
            distance = distance.lower()
            
            # Handle furlongs (1 furlong = 201.168 meters)
            if 'furlong' in distance or 'f' in distance:
                furlongs = float(distance.replace('furlongs', '').replace('furlong', '').replace('f', '').strip())
                return furlongs * 201.168
                
            # Handle miles (1 mile = 1609.34 meters)
            if 'mile' in distance or 'm' in distance:
                miles = float(distance.replace('miles', '').replace('mile', '').replace('m', '').strip())
                return miles * 1609.34
                
            # Default case
            return 0