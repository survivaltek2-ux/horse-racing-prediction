#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Horse Racing ML Models

This script processes the generated training data and creates feature matrices
suitable for machine learning models.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

class HorseRacingDataProcessor:
    """Processes horse racing data for ML training"""
    
    def __init__(self):
        self.horses_data = []
        self.races_data = []
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_training_data(self):
        """Load the generated training data"""
        try:
            with open('data/training_horses.json', 'r') as f:
                self.horses_data = json.load(f)
            print(f"Loaded {len(self.horses_data)} horses")
            
            with open('data/training_races.json', 'r') as f:
                self.races_data = json.load(f)
            print(f"Loaded {len(self.races_data)} races")
            
            # Filter to only completed races with results
            self.completed_races = [r for r in self.races_data if r.get('results')]
            print(f"Found {len(self.completed_races)} completed races with results")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return False
        return True
        
    def get_horse_by_id(self, horse_id: int) -> Dict:
        """Get horse data by ID"""
        for horse in self.horses_data:
            if horse.get('id') == horse_id:
                return horse
        return None
        
    def extract_horse_features(self, horse_id: int, race_date: str) -> Dict:
        """Extract features for a horse at a specific race date"""
        horse = self.get_horse_by_id(horse_id)
        if not horse:
            return {}
            
        features = {}
        
        # Basic horse features
        features['horse_age'] = horse.get('age', 4)
        features['horse_earnings'] = horse.get('earnings', 0)
        features['horse_win_rate'] = horse.get('win_rate', 0.1)
        features['horse_place_rate'] = horse.get('place_rate', 0.2)
        features['horse_show_rate'] = horse.get('show_rate', 0.3)
        
        # Jockey and trainer features (encoded)
        features['jockey'] = horse.get('jockey', 'Unknown')
        features['trainer'] = horse.get('trainer', 'Unknown')
        
        # Recent performance features
        if 'recent_performances' in horse and horse['recent_performances']:
            recent_perfs = horse['recent_performances'][:5]  # Last 5 races
            
            # Recent form metrics
            features['recent_races_count'] = len(recent_perfs)
            features['recent_wins'] = sum(1 for p in recent_perfs if p.get('position', 99) == 1)
            features['recent_places'] = sum(1 for p in recent_perfs if p.get('position', 99) <= 3)
            features['recent_avg_position'] = np.mean([p.get('position', 10) for p in recent_perfs])
            features['recent_avg_earnings'] = np.mean([p.get('earnings', 0) for p in recent_perfs])
            
            # Days since last race
            if recent_perfs:
                last_race_date = datetime.strptime(recent_perfs[0].get('date', race_date), '%Y-%m-%d')
                current_race_date = datetime.strptime(race_date, '%Y-%m-%d')
                features['days_since_last_race'] = (current_race_date - last_race_date).days
            else:
                features['days_since_last_race'] = 365  # Default if no recent races
                
            # Surface performance
            dirt_perfs = [p for p in recent_perfs if p.get('surface', '').lower() == 'dirt']
            turf_perfs = [p for p in recent_perfs if p.get('surface', '').lower() == 'turf']
            
            features['dirt_win_rate'] = (sum(1 for p in dirt_perfs if p.get('position', 99) == 1) / 
                                       max(len(dirt_perfs), 1))
            features['turf_win_rate'] = (sum(1 for p in turf_perfs if p.get('position', 99) == 1) / 
                                       max(len(turf_perfs), 1))
        else:
            # Default values for horses with no recent performances
            features['recent_races_count'] = 0
            features['recent_wins'] = 0
            features['recent_places'] = 0
            features['recent_avg_position'] = 8
            features['recent_avg_earnings'] = 0
            features['days_since_last_race'] = 365
            features['dirt_win_rate'] = 0.1
            features['turf_win_rate'] = 0.1
            
        return features
        
    def extract_race_features(self, race: Dict) -> Dict:
        """Extract features from race data"""
        features = {}
        
        # Basic race features
        features['race_distance'] = self.parse_distance(race.get('distance', '1200m'))
        features['race_purse'] = race.get('purse', 50000)
        features['field_size'] = len(race.get('horse_ids', []))
        
        # Track features
        track_data = race.get('track_data', {})
        features['track_surface'] = track_data.get('surface', 'Dirt')
        features['track_condition'] = track_data.get('condition', 'Good')
        
        # Weather features
        weather_data = race.get('weather_data', {})
        features['weather_condition'] = weather_data.get('condition', 'Clear')
        features['temperature'] = weather_data.get('temperature', 20)
        features['humidity'] = weather_data.get('humidity', 50)
        features['wind_speed'] = weather_data.get('wind_speed', 5)
        
        # Betting features (if available)
        betting_data = race.get('betting_data', {})
        features['total_pool'] = betting_data.get('total_pool', features['race_purse'])
        
        return features
        
    def parse_distance(self, distance_str: str) -> int:
        """Parse distance string to meters"""
        if 'm' in distance_str:
            return int(distance_str.replace('m', ''))
        elif 'f' in distance_str:
            # Convert furlongs to meters (1 furlong = 201.168 meters)
            furlongs = float(distance_str.replace('f', ''))
            return int(furlongs * 201.168)
        else:
            return 1200  # Default
            
    def create_training_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Create training dataset with features and targets"""
        training_data = []
        
        for race in self.completed_races:
            race_features = self.extract_race_features(race)
            race_date = race.get('date', '2024-01-01')
            results = race.get('results', {})
            
            for horse_id in race.get('horse_ids', []):
                # Extract horse features
                horse_features = self.extract_horse_features(horse_id, race_date)
                
                # Combine features
                combined_features = {**race_features, **horse_features}
                combined_features['horse_id'] = horse_id
                combined_features['race_id'] = race.get('id', 0)
                
                # Target variables
                if str(horse_id) in results:
                    result = results[str(horse_id)]
                    combined_features['position'] = result['position']
                    combined_features['finish_time'] = result['time']
                    combined_features['earnings'] = result['earnings']
                    combined_features['won'] = 1 if result['position'] == 1 else 0
                    combined_features['placed'] = 1 if result['position'] <= 3 else 0
                else:
                    # Skip horses without results
                    continue
                    
                training_data.append(combined_features)
                
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        print(f"Created dataset with {len(df)} horse-race combinations")
        
        return df
        
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        categorical_columns = ['jockey', 'trainer', 'track_surface', 'track_condition', 'weather_condition']
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col + '_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col + '_encoded'] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                    
        return df_encoded
        
    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare feature matrix and target vectors"""
        # Define feature columns (excluding targets and IDs)
        feature_columns = [
            'horse_age', 'horse_earnings', 'horse_win_rate', 'horse_place_rate', 'horse_show_rate',
            'recent_races_count', 'recent_wins', 'recent_places', 'recent_avg_position', 
            'recent_avg_earnings', 'days_since_last_race', 'dirt_win_rate', 'turf_win_rate',
            'race_distance', 'race_purse', 'field_size', 'temperature', 'humidity', 'wind_speed',
            'total_pool', 'jockey_encoded', 'trainer_encoded', 'track_surface_encoded',
            'track_condition_encoded', 'weather_condition_encoded'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Feature matrix
        X = df[available_features].values
        
        # Target vectors
        targets = {
            'win': df['won'].values,
            'place': df['placed'].values,
            'position': df['position'].values,
            'finish_time': df['finish_time'].values
        }
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Available features: {available_features}")
        
        return X, targets, available_features
        
    def save_processed_data(self, X: np.ndarray, targets: Dict, feature_names: List[str]):
        """Save processed data and preprocessing objects"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_win_train, y_win_test = train_test_split(
            X_scaled, targets['win'], test_size=0.2, random_state=42, stratify=targets['win']
        )
        
        # Save training and test sets
        np.save('data/X_train.npy', X_train)
        np.save('data/X_test.npy', X_test)
        np.save('data/y_win_train.npy', y_win_train)
        np.save('data/y_win_test.npy', y_win_test)
        
        # Save other targets
        for target_name, target_values in targets.items():
            if target_name != 'win':
                _, _, y_train, y_test = train_test_split(
                    X_scaled, target_values, test_size=0.2, random_state=42
                )
                np.save(f'data/y_{target_name}_train.npy', y_train)
                np.save(f'data/y_{target_name}_test.npy', y_test)
        
        # Save preprocessing objects
        with open('data/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open('data/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
            
        with open('data/feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
            
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Win rate in training set: {np.mean(y_win_train):.3f}")
        print(f"Win rate in test set: {np.mean(y_win_test):.3f}")
        
    def process_data(self):
        """Main processing pipeline"""
        print("Starting data preprocessing...")
        
        # Load data
        if not self.load_training_data():
            return False
            
        # Create dataset
        df = self.create_training_dataset()
        if df.empty:
            print("No training data available!")
            return False
            
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df)
        
        # Prepare features and targets
        X, targets, feature_names = self.prepare_features_and_targets(df_encoded)
        
        # Save processed data
        self.save_processed_data(X, targets, feature_names)
        
        # Save the full processed dataset
        df_encoded.to_csv('data/processed_dataset.csv', index=False)
        
        print("\n" + "="*50)
        print("DATA PREPROCESSING COMPLETE")
        print("="*50)
        print(f"Total samples: {len(df_encoded)}")
        print(f"Features: {len(feature_names)}")
        print(f"Win rate: {np.mean(targets['win']):.3f}")
        print(f"Average field size: {df_encoded['field_size'].mean():.1f}")
        print(f"Average race distance: {df_encoded['race_distance'].mean():.0f}m")
        print("\nFiles created:")
        print("- data/X_train.npy, data/X_test.npy (feature matrices)")
        print("- data/y_*_train.npy, data/y_*_test.npy (target vectors)")
        print("- data/scaler.pkl (feature scaler)")
        print("- data/label_encoders.pkl (categorical encoders)")
        print("- data/processed_dataset.csv (full dataset)")
        print("\n✅ Ready for ML model training!")
        
        return True

def main():
    """Main function"""
    processor = HorseRacingDataProcessor()
    success = processor.process_data()
    
    if success:
        print("\n🎯 Data preprocessing completed successfully!")
        print("You can now train ML models using the processed data.")
    else:
        print("\n❌ Data preprocessing failed!")

if __name__ == "__main__":
    main()