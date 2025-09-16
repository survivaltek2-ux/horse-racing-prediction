import pandas as pd
import numpy as np
from models.firebase_models import Horse, Race

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
                'age': self._validate_numeric(horse.age, 2, 15),  # Validate age range
                'weight': self._validate_numeric(getattr(horse, 'weight', 120), 100, 140),  # Racing weight
                'win_rate': self._validate_rate(horse.win_rate),
                'place_rate': self._validate_rate(horse.place_rate),
                'show_rate': self._validate_rate(horse.show_rate),
                'earnings': self._normalize_earnings(horse.earnings),
            }
            
            # Add recent performance metrics
            recent_form = horse.get_form(10)  # Get last 10 races for better analysis
            
            # Calculate performance metrics
            if recent_form:
                positions = [result['position'] for result in recent_form if result.get('position')]
                
                if positions:
                    horse_data['avg_position'] = sum(positions) / len(positions)
                    horse_data['position_std'] = np.std(positions) if len(positions) > 1 else 0
                    horse_data['best_position'] = min(positions)
                    horse_data['worst_position'] = max(positions)
                    
                    # Calculate win percentage in recent form
                    wins = sum(1 for pos in positions if pos == 1)
                    horse_data['recent_win_rate'] = wins / len(positions)
                    
                    # Calculate place percentage (top 3) in recent form
                    places = sum(1 for pos in positions if pos <= 3)
                    horse_data['recent_place_rate'] = places / len(positions)
                    
                    # Calculate form trend (last 5 vs previous 5)
                    if len(positions) >= 10:
                        recent_5 = positions[:5]
                        previous_5 = positions[5:10]
                        horse_data['form_trend'] = (sum(previous_5) / 5) - (sum(recent_5) / 5)  # Positive = improving
                    else:
                        horse_data['form_trend'] = 0
                        
                    # Calculate consistency score (lower std = more consistent)
                    horse_data['consistency_score'] = 1 / (1 + horse_data['position_std'])
                else:
                    self._set_default_performance_metrics(horse_data)
                    
                # Calculate average time and speed metrics
                times = [float(result['time']) for result in recent_form if 'time' in result and result['time']]
                if times:
                    horse_data['avg_time'] = sum(times) / len(times)
                    horse_data['best_time'] = min(times)
                    horse_data['time_consistency'] = np.std(times)
                else:
                    horse_data['avg_time'] = 0
                    horse_data['best_time'] = 0
                    horse_data['time_consistency'] = 0
                    
                # Days since last race
                if recent_form:
                    last_race_date = recent_form[0].get('date')
                    if last_race_date:
                        from datetime import datetime
                        try:
                            if isinstance(last_race_date, str):
                                last_date = datetime.strptime(last_race_date, '%Y-%m-%d')
                            else:
                                last_date = last_race_date
                            days_since = (datetime.now() - last_date).days
                            horse_data['days_since_last_race'] = min(days_since, 365)  # Cap at 1 year
                        except:
                            horse_data['days_since_last_race'] = 30  # Default
                    else:
                        horse_data['days_since_last_race'] = 30
                else:
                    horse_data['days_since_last_race'] = 365
            else:
                # No recent form data
                self._set_default_performance_metrics(horse_data)
                horse_data['days_since_last_race'] = 365
            
            # Add race-specific features
            horse_data['track_condition'] = self._encode_track_condition(race.track_condition)
            horse_data['distance'] = self._normalize_distance(race.distance)
            horse_data['distance_preference'] = self._calculate_distance_preference(horse, race.distance)
            
            # Add jockey and trainer ratings (if available)
            horse_data['jockey_rating'] = self._get_jockey_rating(getattr(horse, 'jockey', None))
            horse_data['trainer_rating'] = self._get_trainer_rating(getattr(horse, 'trainer', None))
            
            # Add class level and competition strength
            horse_data['class_level'] = self._encode_class_level(getattr(race, 'class_level', 'maiden'))
            horse_data['field_size'] = len(horses)
            
            # Add the horse data to our dataset
            data.append(horse_data)
        
        # Convert to dataframe and validate
        df = pd.DataFrame(data)
        return self._validate_and_clean_data(df)
    
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
        """Normalize distance to a standard scale"""
        try:
            # Convert distance to meters if needed
            if isinstance(distance, str):
                # Handle different distance formats
                if 'f' in distance.lower():  # furlongs
                    distance_num = float(distance.lower().replace('f', ''))
                    distance_meters = distance_num * 201.168  # 1 furlong = 201.168 meters
                elif 'm' in distance.lower():  # meters
                    distance_meters = float(distance.lower().replace('m', ''))
                else:
                    distance_meters = float(distance)
            else:
                distance_meters = float(distance)
            
            # Normalize to 0-1 scale (assuming races are between 1000m and 4000m)
            normalized = (distance_meters - 1000) / (4000 - 1000)
            return max(0, min(1, normalized))  # Clamp between 0 and 1
            
        except (ValueError, TypeError):
            # Default to middle distance if parsing fails
            return 0.5
    
    def _validate_numeric(self, value, min_val, max_val):
        """Validate and clamp numeric values to reasonable ranges"""
        try:
            num_val = float(value) if value is not None else min_val
            return max(min_val, min(max_val, num_val))
        except (ValueError, TypeError):
            return min_val
    
    def _validate_rate(self, rate):
        """Validate rate values (should be between 0 and 1)"""
        try:
            rate_val = float(rate) if rate is not None else 0.0
            return max(0.0, min(1.0, rate_val))
        except (ValueError, TypeError):
            return 0.0
    
    def _normalize_earnings(self, earnings):
        """Normalize earnings to a reasonable scale"""
        try:
            earnings_val = float(earnings) if earnings is not None else 0
            # Log scale for earnings (to handle wide range)
            if earnings_val > 0:
                return np.log10(earnings_val + 1) / 10  # Normalize log scale
            return 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _set_default_performance_metrics(self, horse_data):
        """Set default performance metrics when no form data is available"""
        horse_data['avg_position'] = 8.0  # Assume middle of field
        horse_data['position_std'] = 3.0
        horse_data['best_position'] = 8
        horse_data['worst_position'] = 12
        horse_data['recent_win_rate'] = 0.0
        horse_data['recent_place_rate'] = 0.0
        horse_data['form_trend'] = 0.0
        horse_data['consistency_score'] = 0.25
        horse_data['avg_time'] = 0
        horse_data['best_time'] = 0
        horse_data['time_consistency'] = 0
    
    def _calculate_distance_preference(self, horse, race_distance):
        """Calculate how well suited the horse is to this distance"""
        try:
            # Get horse's form and analyze distance performance
            recent_form = horse.get_form(20)  # Look at more races for distance analysis
            if not recent_form:
                return 0.5  # Neutral preference
            
            # Find races at similar distances
            race_distance_meters = self._distance_to_meters(race_distance)
            similar_distance_results = []
            
            for result in recent_form:
                if 'distance' in result:
                    result_distance = self._distance_to_meters(result['distance'])
                    # Consider distances within 20% as similar
                    if abs(result_distance - race_distance_meters) / race_distance_meters <= 0.2:
                        similar_distance_results.append(result['position'])
            
            if similar_distance_results:
                # Better performance at similar distances = higher preference
                avg_position = sum(similar_distance_results) / len(similar_distance_results)
                # Convert to preference score (lower position = higher preference)
                preference = max(0, 1 - (avg_position - 1) / 15)  # Assuming max 16 horses
                return min(1, preference)
            
            return 0.5  # Neutral if no similar distance data
        except:
            return 0.5
    
    def _distance_to_meters(self, distance):
        """Convert distance to meters"""
        try:
            if isinstance(distance, str):
                if 'f' in distance.lower():
                    return float(distance.lower().replace('f', '')) * 201.168
                elif 'm' in distance.lower():
                    return float(distance.lower().replace('m', ''))
                else:
                    return float(distance)
            return float(distance)
        except:
            return 1600  # Default distance
    
    def _get_jockey_rating(self, jockey):
        """Get jockey rating (placeholder - would be based on jockey stats)"""
        if not jockey:
            return 0.5
        # This would be calculated from jockey's win rate, experience, etc.
        # For now, return a default rating
        return 0.6
    
    def _get_trainer_rating(self, trainer):
        """Get trainer rating (placeholder - would be based on trainer stats)"""
        if not trainer:
            return 0.5
        # This would be calculated from trainer's success rate, stable form, etc.
        # For now, return a default rating
        return 0.6
    
    def _encode_class_level(self, class_level):
        """Encode race class level as numeric value"""
        class_map = {
            'group 1': 1.0,
            'group 2': 0.9,
            'group 3': 0.8,
            'listed': 0.7,
            'handicap': 0.6,
            'allowance': 0.5,
            'claiming': 0.4,
            'maiden': 0.3
        }
        return class_map.get(class_level.lower() if class_level else '', 0.5)
    
    def _validate_and_clean_data(self, df):
        """Validate and clean the prepared data"""
        if df is None or df.empty:
            return df
        
        # Fill any remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Remove any duplicate horses
        df = df.drop_duplicates(subset=['horse_id'], keep='first')
        
        # Ensure all required columns exist
        required_columns = [
            'age', 'weight', 'win_rate', 'avg_position', 'position_std',
            'track_condition', 'distance_preference', 'jockey_rating'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.5  # Default value
        
        return df
    
    def get_feature_importance(self):
        """Get the importance of different features for prediction"""
        return {
            'recent_form': 0.25,
            'win_rate': 0.20,
            'track_condition': 0.15,
            'distance_preference': 0.15,
            'age': 0.10,
            'earnings': 0.10,
            'consistency': 0.05
        }