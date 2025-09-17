import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fall back to RandomForest if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Using RandomForest as primary model.")

from models.sqlalchemy_models import Horse, Race
from models.sqlalchemy_models import Prediction as SQLPrediction
from models.prediction import Prediction
from config.database_config import db
from utils.data_processor import DataProcessor
from utils.ai_predictor import AIPredictor

class Predictor:
    """Enhanced prediction engine for horse racing with advanced ML algorithms"""
    
    def __init__(self):
        """Initialize the predictor with ensemble models and AI capabilities"""
        self.data_processor = DataProcessor()
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
        # Initialize AI predictor
        self.ai_predictor = AIPredictor()
        
        # Initialize multiple models for ensemble
        self.models = {}
        self._initialize_models()
        
        # Main ensemble model
        self.ensemble_model = None
        self.is_trained = False
        self.feature_importance = {}
        self.model_performance = {}
    
    def _initialize_models(self):
        """Initialize different ML models for ensemble"""
        # Random Forest with optimized parameters
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        # Create ensemble model
        estimators = [(name, model) for name, model in self.models.items()]
        self.ensemble_model = VotingRegressor(estimators=estimators)
    
    def predict_race(self, race, params=None):
        """Generate enhanced predictions for a race using ensemble models"""
        if not race:
            return None
            
        # Prepare the race data with enhanced features
        race_data = self._prepare_enhanced_race_data(race)
        if race_data is None or len(race_data) == 0:
            return None
        
        # Use ML model if trained, otherwise fall back to enhanced heuristics
        if self.is_trained and self.ensemble_model is not None:
            print(f"DEBUG: Using ML prediction for race {race.id}", flush=True)
            predictions = self._ml_prediction(race_data, race)
            algorithm = 'ensemble_ml'
            print(f"DEBUG: ML predictions type: {type(predictions)}", flush=True)
            if isinstance(predictions, dict) and predictions:
                first_key = list(predictions.keys())[0]
                first_pred = predictions[first_key]
                print(f"DEBUG: First ML prediction: {first_pred}", flush=True)
        else:
            print(f"DEBUG: Using heuristic prediction for race {race.id}", flush=True)
            predictions = self._enhanced_heuristic_prediction(race_data, race)
            algorithm = 'enhanced_heuristic'
            print(f"DEBUG: Heuristic predictions type: {type(predictions)}", flush=True)
            if isinstance(predictions, dict) and predictions:
                first_key = list(predictions.keys())[0]
                first_pred = predictions[first_key]
                print(f"DEBUG: First heuristic prediction: {first_pred}", flush=True)
        
        # Create and save individual prediction records for each horse
        prediction_records = []
        try:
            # Sort horses by win probability to assign predicted positions
            sorted_horses = sorted(predictions.items(), key=lambda x: x[1]['win_probability'], reverse=True)
            
            for position, (horse_id, pred_data) in enumerate(sorted_horses, 1):
                # Create individual prediction record
                prediction_record = SQLPrediction(
                    race_id=race.id,
                    horse_id=horse_id,
                    predicted_position=position,
                    confidence=pred_data.get('confidence', 0.5),
                    odds=0.0,  # Will be updated when odds are available
                    factors=json.dumps(pred_data),  # Store prediction data as JSON string
                    model_version=algorithm
                )
                
                db.session.add(prediction_record)
                prediction_records.append(prediction_record)
            
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            print(f"Error saving predictions: {e}")
        
        # Create a Prediction object for compatibility with AI system
        prediction_obj = Prediction.create_prediction({
            'race_id': race.id,
            'predictions': predictions,
            'algorithm': algorithm,
            'confidence_scores': self._calculate_confidence_scores(predictions)
        })
        
        return prediction_obj
    
    def _prepare_enhanced_race_data(self, race):
        """Prepare race data with enhanced feature engineering"""
        try:
            # Get base race data
            race_data = self.data_processor.prepare_race_data(race)
            if race_data is None or len(race_data) == 0:
                return None
            
            # Add enhanced features
            race_data = self._add_enhanced_features(race_data, race)
            
            return race_data
        except Exception as e:
            print(f"Error preparing enhanced race data: {str(e)}")
            return None
    
    def _add_enhanced_features(self, race_data, race):
        """Add advanced features for better prediction accuracy"""
        try:
            # Speed ratings based on recent performances
            race_data['speed_rating'] = race_data.apply(self._calculate_speed_rating, axis=1)
            
            # Class rating (quality of competition)
            race_data['class_rating'] = race_data.apply(self._calculate_class_rating, axis=1)
            
            # Trainer/Jockey combination effectiveness
            race_data['trainer_jockey_combo'] = race_data.apply(self._calculate_combo_rating, axis=1)
            
            # Distance suitability score
            race_data['distance_suitability'] = race_data.apply(
                lambda x: self._calculate_distance_suitability(x, race), axis=1
            )
            
            # Recent form trend (improving/declining)
            race_data['form_trend'] = race_data.apply(self._calculate_form_trend, axis=1)
            
            # Weight-adjusted performance
            race_data['weight_adjusted_rating'] = race_data.apply(self._calculate_weight_adjustment, axis=1)
            
            return race_data
        except Exception as e:
            print(f"Error adding enhanced features: {str(e)}")
            return race_data
    
    def _ml_prediction(self, race_data, race):
        """Generate predictions using trained ML ensemble model"""
        try:
            # Prepare features for prediction
            features = self._extract_prediction_features(race_data)
            if len(features) == 0:
                return self._enhanced_heuristic_prediction(race_data, race)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get ensemble predictions
            ml_scores = self.ensemble_model.predict(features_scaled)
            
            # Get individual model predictions for confidence calculation
            individual_predictions = {}
            for name, model in self.models.items():
                try:
                    individual_predictions[name] = model.predict(features_scaled)
                except:
                    continue
            
            # Convert ML scores to horse predictions
            predictions = self._convert_ml_scores_to_predictions(race_data, ml_scores, individual_predictions)
            
            return predictions
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            # Fallback to enhanced heuristic prediction
            return self._enhanced_heuristic_prediction(race_data, race)
    
    def _extract_prediction_features(self, race_data):
        """Extract features from race data for ML prediction"""
        try:
            # Define feature columns that match training
            feature_columns = [
                'win_rate', 'avg_position', 'position_std', 'age', 'races_count',
                'speed_rating', 'class_rating', 'distance_suitability', 'form_trend',
                'weight_adjusted_rating', 'track_condition', 'trainer_jockey_combo',
                'recent_performance', 'distance_category', 'age_category'
            ]
            
            # Ensure all features exist with default values
            for col in feature_columns:
                if col not in race_data.columns:
                    race_data[col] = 0.5
            
            # Return features as numpy array
            return race_data[feature_columns].fillna(0.5).values
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.array([])
    
    def _convert_ml_scores_to_predictions(self, race_data, ml_scores, individual_predictions):
        """Convert ML model scores to race predictions"""
        predictions = {}
        
        try:
            # Normalize ML scores to probabilities
            exp_scores = np.exp(ml_scores - np.max(ml_scores))
            probabilities = exp_scores / np.sum(exp_scores)
            
            # Calculate confidence based on model agreement
            confidence_scores = []
            if individual_predictions:
                # Calculate variance across individual models
                all_preds = np.array(list(individual_predictions.values()))
                pred_variance = np.var(all_preds, axis=0)
                confidence_scores = 1 - (pred_variance / np.max(pred_variance))
            else:
                confidence_scores = [0.7] * len(probabilities)  # Default confidence
            
            # Create predictions for each horse
            for i, (_, horse_row) in enumerate(race_data.iterrows()):
                horse_id = int(horse_row['horse_id'])
                predictions[horse_id] = {
                    'win_probability': float(probabilities[i]),
                    'confidence': float(confidence_scores[i]),
                    'method': 'ensemble_ml',
                    'individual_model_scores': {
                        name: float(scores[i]) for name, scores in individual_predictions.items()
                    } if individual_predictions else {}
                }
            
            return predictions
            
        except Exception as e:
            print(f"Error converting ML scores: {e}")
            # Fallback to equal probabilities
            equal_prob = 1.0 / len(race_data)
            for _, horse_row in race_data.iterrows():
                horse_id = int(horse_row['horse_id'])
                predictions[horse_id] = {
                    'win_probability': equal_prob,
                    'confidence': 0.5,
                    'method': 'fallback'
                }
            return predictions
    
    def _calculate_individual_confidence(self, horse_row):
        """Calculate confidence score for individual horse prediction"""
        confidence_factors = []
        
        # Data completeness factor
        required_fields = ['win_rate', 'avg_position', 'age', 'races_count']
        completeness = sum(1 for field in required_fields if horse_row.get(field) is not None) / len(required_fields)
        confidence_factors.append(completeness)
        
        # Sample size factor (more races = higher confidence)
        races_count = horse_row.get('races_count', 0)
        sample_factor = min(1.0, races_count / 10)  # Normalize to 10 races
        confidence_factors.append(sample_factor)
        
        # Consistency factor (lower std = higher confidence)
        position_std = horse_row.get('position_std', 5)
        consistency_factor = max(0, (5 - position_std) / 5)
        confidence_factors.append(consistency_factor)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_confidence_scores(self, predictions):
        """Calculate overall confidence scores for the predictions"""
        if not predictions:
            return {'overall': 0.0}
        
        print(f"DEBUG: _calculate_confidence_scores called with predictions type: {type(predictions)}", flush=True)
        print(f"DEBUG: predictions keys: {list(predictions.keys()) if isinstance(predictions, dict) else 'Not a dict'}", flush=True)
        if isinstance(predictions, dict) and predictions:
            first_key = list(predictions.keys())[0]
            first_pred = predictions[first_key]
            print(f"DEBUG: First prediction structure: {first_pred}", flush=True)
            print(f"DEBUG: First prediction keys: {list(first_pred.keys()) if isinstance(first_pred, dict) else 'Not a dict'}", flush=True)
        
        # Calculate confidence based on prediction spread and quality
        try:
            win_probs = [pred['win_probability'] for pred in predictions.values()]
        except KeyError as e:
            print(f"DEBUG: KeyError in _calculate_confidence_scores: {e}", flush=True)
            print(f"DEBUG: Available keys in first prediction: {list(list(predictions.values())[0].keys()) if predictions else 'No predictions'}", flush=True)
            # Try old format as fallback
            try:
                win_probs = [pred['win_prob'] for pred in predictions.values()]
                print("DEBUG: Successfully used old 'win_prob' format", flush=True)
            except KeyError:
                print("DEBUG: Neither 'win_probability' nor 'win_prob' found, using default", flush=True)
                win_probs = [0.5] * len(predictions)
        
        # Higher spread = lower confidence (more uncertainty)
        prob_std = np.std(win_probs) if len(win_probs) > 1 else 0
        spread_confidence = max(0, 1 - (prob_std * 2))  # Normalize std to confidence
        
        # Higher max probability = higher confidence in top pick
        max_prob = max(win_probs) if win_probs else 0
        top_pick_confidence = min(1.0, max_prob * 1.5)  # Scale up confidence
        
        # Overall confidence is average of factors
        overall_confidence = (spread_confidence + top_pick_confidence) / 2
        
        return {
            'overall': round(overall_confidence, 3),
            'spread_confidence': round(spread_confidence, 3),
            'top_pick_confidence': round(top_pick_confidence, 3)
        }
    
    def _prepare_ml_race_data(self, race_data, race):
        """Prepare race data with enhanced features for ML prediction"""
        enhanced_data = race_data.copy()
        
        # Add enhanced features
        enhanced_data = self._add_enhanced_features(enhanced_data, race)
        
        # Select features for ML model
        feature_columns = [
            'win_rate', 'avg_position', 'position_std', 'age', 'races_count',
            'speed_rating', 'class_rating', 'distance_suitability', 'form_trend',
            'weight_adjusted_rating', 'track_condition', 'trainer_jockey_combo',
            'recent_performance', 'distance_category', 'age_category'
        ]
        
        # Ensure all features exist with default values
        for col in feature_columns:
            if col not in enhanced_data.columns:
                enhanced_data[col] = 0.5
        
        return enhanced_data[feature_columns].fillna(0.5)
    
    def _add_enhanced_features(self, race_data, race):
        """Add enhanced features to race data"""
        try:
            enhanced_data = race_data.copy()
            
            # Speed rating (normalized performance metric)
            enhanced_data['speed_rating'] = self._calculate_speed_rating(enhanced_data)
            
            # Class rating (based on race quality)
            enhanced_data['class_rating'] = self._calculate_class_rating(enhanced_data, race)
            
            # Distance suitability
            enhanced_data['distance_suitability'] = self._calculate_distance_suitability(enhanced_data, race)
            
            # Form trend (recent performance trend)
            enhanced_data['form_trend'] = self._calculate_form_trend(enhanced_data)
            
            # Weight-adjusted rating
            enhanced_data['weight_adjusted_rating'] = self._calculate_weight_adjustment(enhanced_data)
            
            # Trainer-jockey combination rating
            enhanced_data['trainer_jockey_combo'] = self._calculate_trainer_jockey_rating(enhanced_data)
            
            # Recent performance score
            enhanced_data['recent_performance'] = self._calculate_recent_performance(enhanced_data)
            
            # Categorical features
            enhanced_data['distance_category'] = self._categorize_distance(race)
            enhanced_data['age_category'] = self._categorize_age(enhanced_data['age'])
            
            return enhanced_data
        except Exception as e:
            print(f"Error in _add_enhanced_features: {e}")
            return race_data
    
    def _calculate_speed_rating(self, data):
        """Calculate speed rating based on performance metrics"""
        speed_rating = []
        for _, row in data.iterrows():
            # Base speed from win rate and average position
            base_speed = row.get('win_rate', 0.5) * 0.6
            if row.get('avg_position', 0) > 0:
                position_speed = max(0, (10 - row['avg_position']) / 10) * 0.4
            else:
                position_speed = 0.3
            
            rating = min(1.0, base_speed + position_speed)
            speed_rating.append(rating)
        
        return speed_rating
    
    def _calculate_class_rating(self, data, race):
        """Calculate class rating based on race quality"""
        # Simple implementation - can be enhanced with actual race class data
        base_rating = 0.5
        if hasattr(race, 'race_class'):
            class_mapping = {'Grade 1': 1.0, 'Grade 2': 0.8, 'Grade 3': 0.6, 'Listed': 0.4}
            base_rating = class_mapping.get(race.race_class, 0.5)
        
        return [base_rating] * len(data)
    
    def _calculate_distance_suitability(self, data, race):
        """Calculate how suitable the race distance is for each horse"""
        # Simple implementation based on age and experience
        suitability = []
        race_distance_str = getattr(race, 'distance', '1200m')  # Default distance
        
        # Parse distance string to extract numeric value
        try:
            if isinstance(race_distance_str, str):
                # Extract numeric part from strings like "1200m", "1.5km", etc.
                import re
                distance_match = re.search(r'(\d+(?:\.\d+)?)', race_distance_str)
                if distance_match:
                    race_distance = float(distance_match.group(1))
                    # Convert km to meters if needed
                    if 'km' in race_distance_str.lower():
                        race_distance *= 1000
                else:
                    race_distance = 1200  # Default fallback
            else:
                race_distance = float(race_distance_str)
        except (ValueError, AttributeError):
            race_distance = 1200  # Default fallback
        
        for _, row in data.iterrows():
            age = row.get('age', 5)
            races_count = row.get('races_count', 0)
            
            # Younger horses better at shorter distances, experienced horses at longer
            if race_distance < 1200:
                suit = 0.8 if age < 5 else 0.6
            elif race_distance < 1600:
                suit = 0.9  # Sweet spot for most horses
            else:
                suit = 0.7 if races_count > 10 else 0.5
            
            suitability.append(suit)
        
        return suitability
    
    def _calculate_form_trend(self, data):
        """Calculate recent form trend"""
        # Simple implementation - can be enhanced with actual recent race data
        form_trend = []
        for _, row in data.iterrows():
            win_rate = row.get('win_rate', 0.5)
            consistency = 1 - (row.get('position_std', 2.5) / 5)
            trend = (win_rate + consistency) / 2
            form_trend.append(max(0, min(1, trend)))
        
        return form_trend
    
    def _calculate_weight_adjustment(self, data):
        """Calculate weight-adjusted performance rating"""
        # Simple implementation - assumes weight data would be available
        return [0.5] * len(data)  # Placeholder
    
    def _calculate_trainer_jockey_rating(self, data):
        """Calculate trainer-jockey combination effectiveness"""
        # Simple implementation - would use historical trainer/jockey data
        return [0.5] * len(data)  # Placeholder
    
    def _calculate_recent_performance(self, data):
        """Calculate recent performance score"""
        recent_perf = []
        for _, row in data.iterrows():
            win_rate = row.get('win_rate', 0.5)
            avg_pos = row.get('avg_position', 5)
            
            # Weight recent performance more heavily
            if avg_pos > 0:
                pos_score = max(0, (6 - avg_pos) / 5)
                perf = (win_rate * 0.6) + (pos_score * 0.4)
            else:
                perf = win_rate
            
            recent_perf.append(max(0, min(1, perf)))
        
        return recent_perf
    
    def _categorize_distance(self, race):
        """Categorize race distance"""
        distance_str = getattr(race, 'distance', '1200m')
        
        # Parse distance string to extract numeric value
        try:
            if isinstance(distance_str, str):
                # Extract numeric part from strings like "1200m", "1.5km", etc.
                import re
                distance_match = re.search(r'(\d+(?:\.\d+)?)', distance_str)
                if distance_match:
                    distance = float(distance_match.group(1))
                    # Convert km to meters if needed
                    if 'km' in distance_str.lower():
                        distance *= 1000
                else:
                    distance = 1200  # Default fallback
            else:
                distance = float(distance_str)
        except (ValueError, AttributeError):
            distance = 1200  # Default fallback
        
        if distance < 1200:
            return 0  # Sprint
        elif distance < 1600:
            return 1  # Mile
        else:
            return 2  # Distance
    
    def _categorize_age(self, ages):
        """Categorize horse ages"""
        categories = []
        for age in ages:
            if age < 3:
                categories.append(0)  # Young
            elif age <= 6:
                categories.append(1)  # Prime
            else:
                categories.append(2)  # Veteran
        return categories
    
    def _ml_prediction(self, features):
        """Generate ML-based predictions using ensemble model"""
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get ensemble prediction
            predictions = self.ensemble_model.predict(features_scaled)
            
            # Convert to probabilities (softmax-like normalization)
            exp_preds = np.exp(predictions - np.max(predictions))
            probabilities = exp_preds / np.sum(exp_preds)
            
            return probabilities
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            # Fallback to equal probabilities
            return np.ones(len(features)) / len(features)
            
        except Exception as e:
            print(f"ML prediction failed: {str(e)}")
            return self._enhanced_heuristic_prediction(race_data, race)
    
    def _enhanced_heuristic_prediction(self, race_data, race):
        """Generate predictions using enhanced heuristic rules with advanced features"""
        predictions = {}
        
        # Calculate enhanced scores for each horse
        horse_scores = []
        
        for _, horse_row in race_data.iterrows():
            score = 0
            
            # Core performance metrics (50% total weight)
            score += horse_row.get('win_rate', 0) * 0.25
            
            if horse_row.get('avg_position', 0) > 0:
                position_score = max(0, (10 - horse_row['avg_position']) / 10)
                score += position_score * 0.15
            
            if horse_row.get('position_std', 0) > 0:
                consistency_score = max(0, (5 - horse_row['position_std']) / 5)
                score += consistency_score * 0.1
            
            # Enhanced features (35% total weight)
            score += horse_row.get('speed_rating', 0.5) * 0.15
            score += horse_row.get('class_rating', 0.5) * 0.1
            score += horse_row.get('distance_suitability', 0.5) * 0.05
            score += horse_row.get('form_trend', 0.5) * 0.05
            
            # Age and physical factors (10% weight)
            age = horse_row.get('age', 5)
            if 3 <= age <= 6:
                age_score = 1.0
            elif age < 3:
                age_score = 0.7
            elif age <= 8:
                age_score = 0.8
            else:
                age_score = 0.5
            score += age_score * 0.05
            
            # Weight adjustment
            score += horse_row.get('weight_adjusted_rating', 0.5) * 0.05
            
            # Environmental factors (5% weight)
            score += horse_row.get('track_condition', 0.5) * 0.03
            score += horse_row.get('trainer_jockey_combo', 0.5) * 0.02
            
            horse_scores.append({
                'horse_id': int(horse_row['horse_id']),
                'score': max(0, min(1, score)),
                'confidence': self._calculate_individual_confidence(horse_row)
            })
        
        # Sort horses by score
        horse_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Convert scores to probabilities with confidence weighting
        total_score = sum(horse['score'] for horse in horse_scores)
        if total_score == 0:
            equal_prob = 1.0 / len(horse_scores)
            for horse in horse_scores:
                horse['win_probability'] = equal_prob
        else:
            for horse in horse_scores:
                horse['win_probability'] = horse['score'] / total_score
        
        # Calculate place and show probabilities
        for i, horse in enumerate(horse_scores):
            # Place probability (1st or 2nd)
            if i == 0:  # Favorite
                horse['place_probability'] = min(0.9, horse['win_probability'] + 0.3)
            elif i == 1:  # Second favorite
                horse['place_probability'] = min(0.8, horse['win_probability'] + 0.25)
            else:
                horse['place_probability'] = min(0.7, horse['win_probability'] + 0.2)
            
            # Show probability (1st, 2nd, or 3rd)
            if i <= 2:  # Top 3 favorites
                horse['show_probability'] = min(0.95, horse['place_probability'] + 0.2)
            else:
                horse['show_probability'] = min(0.8, horse['place_probability'] + 0.15)
        
        # Build predictions dictionary
        for horse in horse_scores:
            predictions[horse['horse_id']] = {
                'win_probability': round(horse['win_probability'], 3),
                'place_probability': round(horse['place_probability'], 3),
                'show_probability': round(horse['show_probability'], 3),
                'confidence': horse['confidence'],
                'method': 'enhanced_heuristic'
            }
        
        return predictions
    
    def get_performance_stats(self):
        """Get performance statistics for the predictor"""
        predictions = Prediction.get_all_predictions()
        
        if not predictions:
            return {
                'total_predictions': 0,
                'accuracy': 0,
                'avg_confidence': 0,
                'win_predictions': 0,
                'place_predictions': 0,
                'show_predictions': 0
            }
        
        # Calculate statistics based on available attributes
        total_predictions = len(predictions)
        
        # Use confidence as a proxy for accuracy (high confidence predictions)
        high_confidence_predictions = sum(1 for pred in predictions if hasattr(pred, 'confidence') and pred.confidence and pred.confidence >= 0.8)
        
        # Calculate average confidence for all predictions
        predictions_with_confidence = [pred for pred in predictions if hasattr(pred, 'confidence') and pred.confidence is not None]
        avg_confidence = sum(pred.confidence for pred in predictions_with_confidence) / len(predictions_with_confidence) if predictions_with_confidence else 0
        
        # Count predictions by confidence levels (using confidence as proxy for different bet types)
        win_predictions = sum(1 for pred in predictions if hasattr(pred, 'confidence') and pred.confidence and pred.confidence >= 0.8)
        place_predictions = sum(1 for pred in predictions if hasattr(pred, 'confidence') and pred.confidence and pred.confidence >= 0.6)
        show_predictions = sum(1 for pred in predictions if hasattr(pred, 'confidence') and pred.confidence and pred.confidence >= 0.4)
        
        return {
            'total_predictions': total_predictions,
            'accuracy': round(high_confidence_predictions / total_predictions * 100, 1) if total_predictions > 0 else 0,
            'avg_confidence': round(avg_confidence * 100, 1),
            'win_predictions': win_predictions,
            'place_predictions': place_predictions,
            'show_predictions': show_predictions,
            'completed_predictions': len(predictions_with_confidence)
        }
    
    def train_model(self, training_data=None):
        """Train the enhanced ensemble model with historical data"""
        try:
            # Get historical race data if not provided
            if training_data is None:
                training_data = self._prepare_training_data()
            
            if training_data is None or len(training_data) < 10:
                print("Insufficient training data. Using heuristic approach.")
                self.is_trained = True
                return {'success': True, 'method': 'heuristic', 'message': 'Using heuristic predictions due to insufficient data'}
            
            # Prepare enhanced features
            enhanced_training_data = self._prepare_enhanced_training_data(training_data)
            
            # Extract features and targets
            feature_columns = [
                'win_rate', 'avg_position', 'position_std', 'age', 'races_count',
                'speed_rating', 'class_rating', 'distance_suitability', 'form_trend',
                'weight_adjusted_rating', 'track_condition', 'trainer_jockey_combo',
                'recent_performance', 'distance_category', 'age_category'
            ]
            
            X = enhanced_training_data[feature_columns].fillna(0.5)
            y = enhanced_training_data['actual_position'].fillna(5)
            
            # Convert positions to win probabilities for better training
            y_prob = self._convert_positions_to_probabilities(y)
            
            if len(X) == 0:
                print("No valid features extracted. Using heuristic approach.")
                self.is_trained = True
                return {'success': True, 'method': 'heuristic', 'message': 'Using heuristic predictions due to invalid features'}
            
            # Scale features
            features_scaled = self.scaler.fit_transform(X)
            
            # Train ensemble model
            self.ensemble_model.fit(features_scaled, y_prob)
            
            # Train individual models for comparison
            for name, model in self.models.items():
                try:
                    model.fit(features_scaled, y_prob)
                    print(f"Trained {name} model successfully")
                except Exception as e:
                    print(f"Error training {name}: {e}")
            
            # Calculate and store feature importance
            self._calculate_feature_importance(X.columns)
            
            # Evaluate model performance
            self._evaluate_model_performance(features_scaled, y_prob)
            
            self.is_trained = True
            
            # Calculate training accuracy
            predictions = self.ensemble_model.predict(features_scaled)
            accuracy = self._calculate_accuracy(predictions, y_prob)
            
            # Save the trained model
            self._save_model()
            
            return {
                'success': True, 
                'method': 'ensemble_machine_learning',
                'accuracy': round(accuracy * 100, 2),
                'training_samples': len(X),
                'message': f'Enhanced ensemble model trained successfully with {len(X)} samples. Accuracy: {round(accuracy * 100, 2)}%'
            }
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            self.is_trained = True  # Fall back to heuristic
            return {'success': False, 'error': str(e), 'message': 'Training failed, using heuristic approach'}
    
    def _prepare_enhanced_training_data(self, training_data):
        """Prepare training data with enhanced features"""
        enhanced_data = training_data.copy()
        
        # Add enhanced features (simplified for training)
        enhanced_data['speed_rating'] = self._calculate_speed_rating(enhanced_data)
        enhanced_data['class_rating'] = [0.5] * len(enhanced_data)  # Placeholder
        enhanced_data['distance_suitability'] = [0.7] * len(enhanced_data)  # Placeholder
        enhanced_data['form_trend'] = self._calculate_form_trend(enhanced_data)
        enhanced_data['weight_adjusted_rating'] = [0.5] * len(enhanced_data)  # Placeholder
        enhanced_data['trainer_jockey_combo'] = [0.5] * len(enhanced_data)  # Placeholder
        enhanced_data['recent_performance'] = self._calculate_recent_performance(enhanced_data)
        enhanced_data['distance_category'] = [1] * len(enhanced_data)  # Default to mile
        enhanced_data['age_category'] = self._categorize_age(enhanced_data['age'])
        
        return enhanced_data
    
    def _convert_positions_to_probabilities(self, positions):
        """Convert race positions to win probabilities for training"""
        # Convert positions to probabilities (1st place = high prob, last place = low prob)
        max_pos = positions.max()
        probabilities = (max_pos - positions + 1) / max_pos
        return probabilities.clip(0.01, 0.99)  # Avoid extreme values
    
    def _calculate_feature_importance(self, feature_names):
        """Calculate and store feature importance from ensemble model"""
        try:
            # Get feature importance from Random Forest (first estimator)
            if hasattr(self.ensemble_model.estimators_[0], 'feature_importances_'):
                importance = self.ensemble_model.estimators_[0].feature_importances_
                self.feature_importance = dict(zip(feature_names, importance))
                
                # Print top features
                sorted_features = sorted(self.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                print("Top 5 most important features:")
                for feature, importance in sorted_features[:5]:
                    print(f"  {feature}: {importance:.3f}")
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
    
    def _evaluate_model_performance(self, X_scaled, y_true):
        """Evaluate model performance and store metrics"""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Cross-validation scores
            cv_scores = cross_val_score(self.ensemble_model, X_scaled, y_true, cv=5, scoring='r2')
            
            # Predictions for evaluation
            y_pred = self.ensemble_model.predict(X_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            self.model_performance = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'mse': mse,
                'r2': r2,
                'training_samples': len(X_scaled)
            }
            
            print(f"Model Performance:")
            print(f"  Cross-validation R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"  Training R²: {r2:.3f}")
            print(f"  MSE: {mse:.3f}")
            
        except Exception as e:
            print(f"Error evaluating model performance: {e}")
    
    def _prepare_training_data(self):
        """Prepare training data from historical races"""
        try:
            # Get all completed races with results
            races = Race.get_all()
            training_data = []
            
            for race in races:
                # Only use races with results for training
                if hasattr(race, 'results') and race.results:
                    race_data = self.data_processor.prepare_race_data(race)
                    if race_data is not None and len(race_data) > 0:
                        # Add race results to the data
                        for _, horse_row in race_data.iterrows():
                            horse_id = horse_row['horse_id']
                            # Find the result for this horse
                            horse_result = next((r for r in race.results if r.get('horse_id') == horse_id), None)
                            if horse_result:
                                horse_row['actual_position'] = horse_result.get('position', 999)
                                horse_row['won'] = 1 if horse_result.get('position') == 1 else 0
                                training_data.append(horse_row.to_dict())
            
            return pd.DataFrame(training_data) if training_data else None
            
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            return None
    
    def _extract_features_and_targets(self, training_data):
        """Extract features and target variables from training data"""
        try:
            # Feature columns
            feature_columns = [
                'age', 'weight', 'win_rate', 'avg_position', 'position_std',
                'track_condition', 'distance_preference', 'jockey_rating'
            ]
            
            # Filter available columns
            available_features = [col for col in feature_columns if col in training_data.columns]
            
            if not available_features:
                return [], []
            
            # Extract features
            features = training_data[available_features].fillna(0).values
            
            # Extract targets (win probability)
            targets = training_data['won'].values
            
            return features, targets
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return [], []
    
    def _calculate_accuracy(self, predictions, targets):
        """Calculate prediction accuracy"""
        try:
            # Convert probabilities to binary predictions
            binary_predictions = (predictions > 0.5).astype(int)
            accuracy = np.mean(binary_predictions == targets)
            return accuracy
        except:
            return 0.0
    
    def _save_model(self):
        """Save the trained ensemble model to disk with metadata and versioning"""
        try:
            import joblib
            import os
            import json
            from datetime import datetime
            
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained')
            os.makedirs(models_dir, exist_ok=True)
            
            # Create timestamp for versioning
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save current model as backup if it exists
            current_model_path = os.path.join(models_dir, 'predictor_model.pkl')
            if os.path.exists(current_model_path):
                backup_path = os.path.join(models_dir, f'predictor_model_backup_{timestamp}.pkl')
                os.rename(current_model_path, backup_path)
                print(f"Previous model backed up to {backup_path}")
            
            # Save ensemble model, individual models, and scaler
            model_path = os.path.join(models_dir, 'predictor_model.pkl')
            ensemble_path = os.path.join(models_dir, 'ensemble_model.pkl')
            scaler_path = os.path.join(models_dir, 'predictor_scaler.pkl')
            metadata_path = os.path.join(models_dir, 'model_metadata.json')
            
            joblib.dump(self.ensemble_model, model_path)
            joblib.dump(self.ensemble_model, ensemble_path)
            joblib.dump(self.models, os.path.join(models_dir, 'individual_models.pkl'))
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'model_type': 'VotingRegressor_Ensemble',
                'training_date': datetime.now().isoformat(),
                'version': timestamp,
                'sklearn_version': self._get_sklearn_version(),
                'feature_columns': [
                    'win_rate', 'avg_position', 'position_std', 'age', 'races_count',
                    'speed_rating', 'class_rating', 'distance_suitability', 'form_trend',
                    'weight_adjusted_rating', 'track_condition', 'trainer_jockey_combo',
                    'recent_performance', 'distance_category', 'age_category'
                ],
                'ensemble_models': list(self.models.keys()),
                'xgboost_available': XGBOOST_AVAILABLE,
                'feature_importance': getattr(self, 'feature_importance', {}),
                'model_performance': getattr(self, 'model_performance', {}),
                'performance_stats': self.get_performance_stats(),
                'is_trained': self.is_trained
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Ensemble model saved to {model_path}")
            print(f"Metadata saved to {metadata_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self):
        """Load a previously trained ensemble model from disk with metadata validation"""
        try:
            import joblib
            import os
            import json
            
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained')
            model_path = os.path.join(models_dir, 'predictor_model.pkl')
            ensemble_path = os.path.join(models_dir, 'ensemble_model.pkl')
            individual_models_path = os.path.join(models_dir, 'individual_models.pkl')
            scaler_path = os.path.join(models_dir, 'predictor_scaler.pkl')
            metadata_path = os.path.join(models_dir, 'model_metadata.json')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                # Load ensemble model and scaler
                self.ensemble_model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load individual models if available
                if os.path.exists(individual_models_path):
                    self.models = joblib.load(individual_models_path)
                
                # Load metadata if available
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    print(f"Ensemble model loaded successfully")
                    print(f"Model type: {metadata.get('model_type', 'Unknown')}")
                    print(f"Training date: {metadata.get('training_date', 'Unknown')}")
                    print(f"Version: {metadata.get('version', 'Unknown')}")
                    print(f"Ensemble models: {metadata.get('ensemble_models', [])}")
                    
                    # Load feature importance and performance metrics
                    self.feature_importance = metadata.get('feature_importance', {})
                    self.model_performance = metadata.get('model_performance', {})
                    
                    # Validate sklearn version compatibility
                    saved_version = metadata.get('sklearn_version', '')
                    current_version = self._get_sklearn_version()
                    if saved_version and saved_version != current_version:
                        print(f"Warning: Model trained with sklearn {saved_version}, current version is {current_version}")
                    
                    # Store metadata for later use
                    self.model_metadata = metadata
                else:
                    print("Ensemble model loaded successfully (no metadata found)")
                    self.model_metadata = None
                
                self.is_trained = True
                return True
            else:
                print("No saved ensemble model found")
                return False
                
        except Exception as e:
            print(f"Error loading ensemble model: {str(e)}")
            return False
    
    def _get_sklearn_version(self):
        """Get the current sklearn version"""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return "unknown"
    
    def get_model_info(self):
        """Get information about the current ensemble model"""
        if not self.is_trained:
            return {'trained': False, 'message': 'No model trained'}
        
        info = {
            'trained': True,
            'model_type': 'VotingRegressor_Ensemble',
            'is_trained': self.is_trained,
            'ensemble_models': list(self.models.keys()),
            'xgboost_available': XGBOOST_AVAILABLE
        }
        
        # Add feature importance if available
        if hasattr(self, 'feature_importance') and self.feature_importance:
            info['feature_importance'] = self.feature_importance
        
        # Add performance metrics if available
        if hasattr(self, 'model_performance') and self.model_performance:
            info['model_performance'] = self.model_performance
        
        # Add metadata if available
        if hasattr(self, 'model_metadata') and self.model_metadata:
            info.update({
                'training_date': self.model_metadata.get('training_date'),
                'version': self.model_metadata.get('version'),
                'sklearn_version': self.model_metadata.get('sklearn_version'),
                'feature_count': len(self.model_metadata.get('feature_columns', [])),
                'ensemble_models_metadata': self.model_metadata.get('ensemble_models', [])
            })
        
        return info
    
    def list_model_backups(self):
        """List available model backups"""
        try:
            import os
            import glob
            
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained')
            backup_pattern = os.path.join(models_dir, 'predictor_model_backup_*.pkl')
            backups = glob.glob(backup_pattern)
            
            backup_info = []
            for backup in backups:
                filename = os.path.basename(backup)
                # Extract timestamp from filename
                timestamp = filename.replace('predictor_model_backup_', '').replace('.pkl', '')
                backup_info.append({
                    'filename': filename,
                    'timestamp': timestamp,
                    'path': backup
                })
            
            return sorted(backup_info, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            print(f"Error listing backups: {str(e)}")
            return []
    
    def restore_backup(self, backup_timestamp):
        """Restore a model from backup"""
        try:
            import os
            import shutil
            
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained')
            backup_path = os.path.join(models_dir, f'predictor_model_backup_{backup_timestamp}.pkl')
            current_path = os.path.join(models_dir, 'predictor_model.pkl')
            
            if os.path.exists(backup_path):
                # Backup current model first
                if os.path.exists(current_path):
                    temp_backup = os.path.join(models_dir, 'predictor_model_temp_backup.pkl')
                    shutil.copy2(current_path, temp_backup)
                
                # Restore from backup
                shutil.copy2(backup_path, current_path)
                print(f"Model restored from backup: {backup_timestamp}")
                
                # Try to load the restored model
                if self.load_model():
                    return True
                else:
                    # If loading fails, restore the temp backup
                    if os.path.exists(temp_backup):
                        shutil.copy2(temp_backup, current_path)
                        os.remove(temp_backup)
                    return False
            else:
                print(f"Backup not found: {backup_timestamp}")
                return False
                
        except Exception as e:
            print(f"Error restoring backup: {str(e)}")
            return False
    
    def predict_race_with_ai(self, race, use_ai=True, use_ensemble=True):
        """Enhanced prediction using both traditional ML and AI models"""
        try:
            import logging
            import sys
            
            # Force output to stdout immediately
            print(f"DEBUG: Starting predict_race_with_ai for race {race.id}", flush=True)
            sys.stdout.flush()
            
            # Get traditional ML prediction
            ml_prediction = self.predict_race(race)
            print(f"DEBUG: ML prediction type: {type(ml_prediction)}", flush=True)
            print(f"DEBUG: ML prediction has predictions attr: {hasattr(ml_prediction, 'predictions')}", flush=True)
            if hasattr(ml_prediction, 'predictions'):
                print(f"DEBUG: ML predictions: {ml_prediction.predictions}", flush=True)
            
            if not use_ai:
                return ml_prediction
            
            # Get AI prediction
            ai_prediction = self.ai_predictor.predict_race_ai(race, use_ensemble)
            print(f"DEBUG: AI prediction type: {type(ai_prediction)}", flush=True)
            print(f"DEBUG: AI prediction: {ai_prediction}", flush=True)
            
            # Combine predictions intelligently
            combined_prediction = self._combine_ml_ai_predictions(ml_prediction, ai_prediction)
            print(f"DEBUG: Combined prediction type: {type(combined_prediction)}", flush=True)
            if hasattr(combined_prediction, 'predictions'):
                print(f"DEBUG: Combined predictions: {combined_prediction.predictions}", flush=True)
            
            return combined_prediction
            
        except Exception as e:
            print(f"Error in AI-enhanced prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return self.predict_race(race)  # Fallback to traditional ML
    
    def _combine_ml_ai_predictions(self, ml_pred, ai_pred):
        """Intelligently combine ML and AI predictions"""
        try:
            # Extract prediction values from ML prediction
            ml_predictions = ml_pred.predictions if hasattr(ml_pred, 'predictions') else {}
            ml_confidence = getattr(ml_pred, 'confidence_scores', {}).get('overall', 0.7)
            
            # Extract AI prediction values
            ai_predictions = ai_pred.get('predictions', {}) if isinstance(ai_pred, dict) else {}
            ai_confidence = ai_pred.get('confidence_scores', {}).get('overall', 0.6) if isinstance(ai_pred, dict) else 0.6
            
            # Enhance ML predictions with AI insights
            enhanced_predictions = {}
            for horse_id, ml_data in ml_predictions.items():
                # Get ML win probability
                ml_win_prob = ml_data.get('win_probability', 0.5) if isinstance(ml_data, dict) else 0.5
                
                # Get AI prediction for this horse (if available)
                ai_win_prob = 0.5  # Default
                if str(horse_id) in ai_predictions:
                    ai_data = ai_predictions[str(horse_id)]
                    ai_win_prob = ai_data.get('win_probability', 0.5) if isinstance(ai_data, dict) else ai_data
                
                # Combine ML and AI scores
                total_confidence = ml_confidence + ai_confidence
                if total_confidence > 0:
                    combined_win_prob = (ml_win_prob * ml_confidence + ai_win_prob * ai_confidence) / total_confidence
                else:
                    combined_win_prob = (ml_win_prob + ai_win_prob) / 2
                
                enhanced_predictions[horse_id] = {
                    'win_probability': combined_win_prob,
                    'place_probability': ml_data.get('place_probability', combined_win_prob + 0.2),
                    'show_probability': ml_data.get('show_probability', combined_win_prob + 0.3),
                    'ml_score': ml_win_prob,
                    'ai_score': ai_win_prob,
                    'confidence': (ml_confidence + ai_confidence) / 2,
                    'method': 'ml_ai_ensemble'
                }
            
            # Create enhanced prediction object
            enhanced_pred = Prediction.create_prediction({
                'race_id': ml_pred.race_id if hasattr(ml_pred, 'race_id') else None,
                'predictions': enhanced_predictions,
                'algorithm': 'ml_ai_ensemble',
                'confidence_scores': {
                    'overall': (ml_confidence + ai_confidence) / 2,
                    'ml_confidence': ml_confidence,
                    'ai_confidence': ai_confidence
                },
                'ai_insights': ai_pred.get('ai_insights', []) if isinstance(ai_pred, dict) else [],
                'ml_details': ml_predictions,
                'ai_details': ai_pred if isinstance(ai_pred, dict) else {}
            })
            
            return enhanced_pred
            
        except Exception as e:
            print(f"Error combining predictions: {str(e)}")
            return ml_pred
    
    def train_ai_models(self, training_data=None):
        """Train the AI models with historical data"""
        try:
            if training_data is None:
                # Generate training data from existing races
                training_data = self._prepare_ai_training_data()
            
            success = self.ai_predictor.train_ai_models(training_data)
            if success:
                print("AI models trained successfully")
            else:
                print("AI model training failed")
            
            return success
            
        except Exception as e:
            print(f"Error training AI models: {str(e)}")
            return False
    
    def _prepare_ai_training_data(self):
        """Prepare training data for AI models from historical races"""
        try:
            # This would fetch historical race data
            # For now, return empty to use the fallback in ai_predictor
            return None
            
        except Exception as e:
            print(f"Error preparing AI training data: {str(e)}")
            return None
    
    def get_ai_insights(self, race):
        """Get AI-powered insights for a race"""
        try:
            ai_prediction = self.ai_predictor.predict_race_ai(race, use_ensemble=True)
            return ai_prediction.get('ai_insights', [])
        except Exception as e:
            print(f"Error getting AI insights: {str(e)}")
            return ["AI insights temporarily unavailable"]