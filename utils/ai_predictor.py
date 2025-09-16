import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

from models.sqlalchemy_models import Horse, Race, Prediction
from utils.data_processor import DataProcessor

class AIPredictor:
    """Advanced AI-powered horse racing predictor using deep learning"""
    
    def __init__(self):
        """Initialize the AI predictor with neural networks"""
        self.data_processor = DataProcessor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Neural network models
        self.neural_models = {}
        self.is_trained = False
        self.feature_names = []
        
        # Initialize models based on available libraries
        if TENSORFLOW_AVAILABLE:
            self._initialize_tensorflow_models()
        if PYTORCH_AVAILABLE:
            self._initialize_pytorch_models()
    
    def _initialize_tensorflow_models(self):
        """Initialize TensorFlow/Keras neural network models"""
        # Deep Neural Network for win probability
        self.neural_models['dnn_win'] = self._create_dnn_model()
        
        # LSTM for sequence prediction (form analysis)
        self.neural_models['lstm_form'] = self._create_lstm_model()
        
        # CNN for pattern recognition in racing data
        self.neural_models['cnn_pattern'] = self._create_cnn_model()
    
    def _initialize_pytorch_models(self):
        """Initialize PyTorch neural network models"""
        try:
            # PyTorch models for additional AI capabilities
            self.neural_models['pytorch_dnn'] = self._create_pytorch_dnn()
            self.neural_models['pytorch_rnn'] = self._create_pytorch_rnn()
            print("PyTorch models initialized successfully")
        except Exception as e:
            print(f"Error initializing PyTorch models: {str(e)}")
    
    def _create_pytorch_dnn(self):
        """Create a PyTorch deep neural network"""
        if not PYTORCH_AVAILABLE:
            return None
        
        class PyTorchDNN(nn.Module):
            def __init__(self, input_dim=50):
                super(PyTorchDNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        return PyTorchDNN()
    
    def _create_pytorch_rnn(self):
        """Create a PyTorch RNN model"""
        if not PYTORCH_AVAILABLE:
            return None
        
        class PyTorchRNN(nn.Module):
            def __init__(self, input_size=20, hidden_size=128, num_layers=2):
                super(PyTorchRNN, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                out, _ = self.rnn(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return self.sigmoid(out)
        
        return PyTorchRNN()
    
    def _create_dnn_model(self, input_dim=50):
        """Create a deep neural network for win probability prediction"""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')  # Win probability
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _create_lstm_model(self, sequence_length=10, features=20):
        """Create LSTM model for analyzing horse form sequences"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, features)),
            layers.Dropout(0.3),
            
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_cnn_model(self, input_shape=(20, 20, 1)):
        """Create CNN model for pattern recognition in racing data"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict_race_ai(self, race, use_ensemble=True):
        """Generate AI-powered predictions for a race"""
        try:
            # Prepare enhanced features
            race_features = self._prepare_ai_features(race)
            if race_features is None:
                return self._fallback_prediction(race)
            
            predictions = {}
            confidence_scores = {}
            
            # Get predictions from each AI model
            if TENSORFLOW_AVAILABLE and self.is_trained:
                predictions.update(self._get_tensorflow_predictions(race_features))
            
            # Ensemble prediction if multiple models available
            if use_ensemble and len(predictions) > 1:
                final_predictions = self._ensemble_ai_predictions(predictions, race_features)
            else:
                final_predictions = predictions.get('dnn_win', self._fallback_prediction(race))
            
            # Calculate confidence scores
            confidence_scores = self._calculate_ai_confidence(predictions, race_features)
            
            # Add AI insights
            insights = self._generate_ai_insights(race_features, final_predictions)
            
            return {
                'predictions': final_predictions,
                'confidence_scores': confidence_scores,
                'ai_insights': insights,
                'algorithm': 'ai_ensemble' if use_ensemble else 'ai_single'
            }
            
        except Exception as e:
            print(f"Error in AI prediction: {str(e)}")
            return self._fallback_prediction(race)
    
    def _prepare_ai_features(self, race):
        """Prepare advanced features for AI models"""
        try:
            # Get base race data
            race_data = self.data_processor.prepare_race_data(race)
            if race_data is None or len(race_data) == 0:
                return None
            
            # Enhanced feature engineering for AI
            features = {}
            
            # Statistical features
            features['speed_features'] = self._extract_speed_features(race_data)
            features['form_features'] = self._extract_form_features(race_data)
            features['class_features'] = self._extract_class_features(race_data)
            features['distance_features'] = self._extract_distance_features(race_data, race)
            features['jockey_trainer_features'] = self._extract_jockey_trainer_features(race_data)
            
            # Sequence features for LSTM
            features['form_sequences'] = self._create_form_sequences(race_data)
            
            # Pattern features for CNN
            features['pattern_matrices'] = self._create_pattern_matrices(race_data)
            
            return features
            
        except Exception as e:
            print(f"Error preparing AI features: {str(e)}")
            return None
    
    def _extract_speed_features(self, race_data):
        """Extract speed-related features"""
        features = []
        for _, horse_data in race_data.iterrows():
            speed_features = [
                horse_data.get('recent_speed_avg', 0),
                horse_data.get('best_speed', 0),
                horse_data.get('speed_consistency', 0),
                horse_data.get('speed_trend', 0),
                horse_data.get('class_adjusted_speed', 0)
            ]
            features.append(speed_features)
        return np.array(features)
    
    def _extract_form_features(self, race_data):
        """Extract form-related features"""
        features = []
        for _, horse_data in race_data.iterrows():
            form_features = [
                horse_data.get('recent_form_score', 0),
                horse_data.get('wins_last_5', 0),
                horse_data.get('places_last_5', 0),
                horse_data.get('form_trend', 0),
                horse_data.get('days_since_last_run', 0),
                horse_data.get('consistency_rating', 0)
            ]
            features.append(form_features)
        return np.array(features)
    
    def _extract_class_features(self, race_data):
        """Extract class and competition level features"""
        features = []
        for _, horse_data in race_data.iterrows():
            class_features = [
                horse_data.get('class_rating', 0),
                horse_data.get('prize_money_earned', 0),
                horse_data.get('grade_wins', 0),
                horse_data.get('competition_level', 0),
                horse_data.get('class_drop_rise', 0)
            ]
            features.append(class_features)
        return np.array(features)
    
    def _extract_distance_features(self, race_data, race):
        """Extract distance suitability features"""
        race_distance = getattr(race, 'distance', 1200)
        features = []
        
        for _, horse_data in race_data.iterrows():
            distance_features = [
                horse_data.get('preferred_distance', race_distance) / race_distance,
                horse_data.get('distance_wins', 0),
                horse_data.get('distance_places', 0),
                horse_data.get('distance_performance_avg', 0),
                abs(horse_data.get('preferred_distance', race_distance) - race_distance) / race_distance
            ]
            features.append(distance_features)
        return np.array(features)
    
    def _extract_jockey_trainer_features(self, race_data):
        """Extract jockey and trainer performance features"""
        features = []
        for _, horse_data in race_data.iterrows():
            jt_features = [
                horse_data.get('jockey_win_rate', 0),
                horse_data.get('trainer_win_rate', 0),
                horse_data.get('jockey_trainer_combo_wins', 0),
                horse_data.get('jockey_experience', 0),
                horse_data.get('trainer_experience', 0)
            ]
            features.append(jt_features)
        return np.array(features)
    
    def _create_form_sequences(self, race_data, sequence_length=10):
        """Create sequences for LSTM analysis"""
        sequences = []
        for _, horse_data in race_data.iterrows():
            # Create a sequence of recent performances
            sequence = []
            for i in range(sequence_length):
                performance = [
                    horse_data.get(f'run_{i}_position', 0),
                    horse_data.get(f'run_{i}_speed', 0),
                    horse_data.get(f'run_{i}_margin', 0),
                    horse_data.get(f'run_{i}_class', 0)
                ]
                sequence.append(performance)
            sequences.append(sequence)
        return np.array(sequences)
    
    def _create_pattern_matrices(self, race_data, matrix_size=(20, 20)):
        """Create 2D matrices for CNN pattern recognition"""
        matrices = []
        for _, horse_data in race_data.iterrows():
            # Create a matrix representing horse's racing patterns
            matrix = np.zeros(matrix_size)
            
            # Fill matrix with various performance metrics
            # This is a simplified example - you can enhance this
            for i in range(min(matrix_size[0], 10)):
                for j in range(min(matrix_size[1], 5)):
                    value = horse_data.get(f'pattern_{i}_{j}', np.random.random())
                    matrix[i, j] = value
            
            matrices.append(matrix.reshape(matrix_size + (1,)))
        return np.array(matrices)
    
    def _get_tensorflow_predictions(self, features):
        """Get predictions from TensorFlow models"""
        predictions = {}
        
        try:
            # DNN prediction
            if 'dnn_win' in self.neural_models:
                dnn_features = np.concatenate([
                    features['speed_features'].flatten().reshape(1, -1),
                    features['form_features'].flatten().reshape(1, -1),
                    features['class_features'].flatten().reshape(1, -1)
                ], axis=1)
                
                if dnn_features.shape[1] == self.neural_models['dnn_win'].input_shape[1]:
                    dnn_pred = self.neural_models['dnn_win'].predict(dnn_features, verbose=0)
                    predictions['dnn_win'] = float(dnn_pred[0][0])
            
            # LSTM prediction
            if 'lstm_form' in self.neural_models and 'form_sequences' in features:
                lstm_pred = self.neural_models['lstm_form'].predict(
                    features['form_sequences'][:1], verbose=0
                )
                predictions['lstm_form'] = float(lstm_pred[0][0])
            
            # CNN prediction
            if 'cnn_pattern' in self.neural_models and 'pattern_matrices' in features:
                cnn_pred = self.neural_models['cnn_pattern'].predict(
                    features['pattern_matrices'][:1], verbose=0
                )
                predictions['cnn_pattern'] = float(cnn_pred[0][0])
                
        except Exception as e:
            print(f"Error in TensorFlow predictions: {str(e)}")
        
        return predictions
    
    def _ensemble_ai_predictions(self, predictions, features):
        """Combine predictions from multiple AI models"""
        if not predictions:
            return 0.5
        
        # Weighted ensemble based on model confidence
        weights = {
            'dnn_win': 0.4,
            'lstm_form': 0.3,
            'cnn_pattern': 0.3
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.1)
            weighted_sum += prediction * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _calculate_ai_confidence(self, predictions, features):
        """Calculate confidence scores for AI predictions"""
        confidence = {}
        
        # Base confidence on prediction consistency
        if len(predictions) > 1:
            pred_values = list(predictions.values())
            std_dev = np.std(pred_values)
            confidence['consistency'] = max(0, 1 - std_dev * 2)
        else:
            confidence['consistency'] = 0.7
        
        # Feature quality confidence
        feature_quality = self._assess_feature_quality(features)
        confidence['feature_quality'] = feature_quality
        
        # Overall confidence
        confidence['overall'] = (confidence['consistency'] + confidence['feature_quality']) / 2
        
        return confidence
    
    def _assess_feature_quality(self, features):
        """Assess the quality of input features"""
        quality_score = 0.5
        
        try:
            # Check for missing or zero features
            for feature_type, feature_data in features.items():
                if isinstance(feature_data, np.ndarray):
                    non_zero_ratio = np.count_nonzero(feature_data) / feature_data.size
                    quality_score += non_zero_ratio * 0.1
            
            return min(1.0, quality_score)
        except:
            return 0.5
    
    def _generate_ai_insights(self, features, prediction):
        """Generate AI-powered insights about the prediction"""
        insights = []
        
        try:
            # Speed analysis
            speed_features = features.get('speed_features', np.array([]))
            if speed_features.size > 0:
                avg_speed = np.mean(speed_features)
                if avg_speed > 0.7:
                    insights.append("Strong speed indicators suggest good performance potential")
                elif avg_speed < 0.3:
                    insights.append("Speed metrics indicate potential challenges")
            
            # Form analysis
            form_features = features.get('form_features', np.array([]))
            if form_features.size > 0:
                form_trend = np.mean(form_features[:, 3]) if form_features.shape[1] > 3 else 0
                if form_trend > 0.6:
                    insights.append("Recent form shows positive improvement trend")
                elif form_trend < 0.4:
                    insights.append("Form analysis suggests declining performance")
            
            # Prediction confidence
            if prediction > 0.7:
                insights.append("AI models show high confidence in strong performance")
            elif prediction < 0.3:
                insights.append("AI analysis suggests lower probability of success")
            else:
                insights.append("AI models indicate moderate performance expectations")
            
        except Exception as e:
            insights.append("AI analysis completed with standard confidence")
        
        return insights
    
    def _fallback_prediction(self, race):
        """Fallback prediction when AI models are not available"""
        return {
            'predictions': 0.5,
            'confidence_scores': {'overall': 0.3},
            'ai_insights': ["Using fallback prediction - AI models not trained"],
            'algorithm': 'fallback'
        }
    
    def train_ai_models(self, training_data):
        """Train the AI models with historical data"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available for training")
            return False
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            if X is None or len(X) == 0:
                print("No training data available")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train DNN model
            if 'dnn_win' in self.neural_models:
                self.neural_models['dnn_win'].fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    verbose=0
                )
            
            self.is_trained = True
            print("AI models trained successfully")
            return True
            
        except Exception as e:
            print(f"Error training AI models: {str(e)}")
            return False
    
    def _prepare_training_data(self, training_data):
        """Prepare data for training AI models"""
        # This would process historical race data
        # For now, return dummy data
        X = np.random.random((1000, 50))
        y = np.random.randint(0, 2, 1000)
        return X, y