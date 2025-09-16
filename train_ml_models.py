#!/usr/bin/env python3
"""
ML Model Training for Horse Racing Predictions

This script demonstrates how to train machine learning models using the processed data.
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib

class HorseRacingMLTrainer:
    """Trains ML models for horse racing predictions"""
    
    def __init__(self):
        self.models = {}
        self.feature_names = []
        
    def load_processed_data(self):
        """Load the preprocessed training data"""
        try:
            # Load training and test sets
            self.X_train = np.load('data/X_train.npy')
            self.X_test = np.load('data/X_test.npy')
            self.y_win_train = np.load('data/y_win_train.npy')
            self.y_win_test = np.load('data/y_win_test.npy')
            
            # Load feature names
            with open('data/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
                
            print(f"Loaded training data: {self.X_train.shape}")
            print(f"Loaded test data: {self.X_test.shape}")
            print(f"Features: {len(self.feature_names)}")
            print(f"Win rate in training: {np.mean(self.y_win_train):.3f}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please run data_preprocessing.py first!")
            return False
            
    def train_models(self):
        """Train multiple ML models"""
        print("\nTraining ML models...")
        
        # Define models to train
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        # Train each model
        for model_name, model in models_to_train.items():
            print(f"\nTraining {model_name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_win_train)
            
            # Make predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            y_pred_proba_test = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            train_accuracy = accuracy_score(self.y_win_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_win_test, y_pred_test)
            test_auc = roc_auc_score(self.y_win_test, y_pred_proba_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_win_train, cv=5, scoring='roc_auc')
            
            print(f"Training Accuracy: {train_accuracy:.3f}")
            print(f"Test Accuracy: {test_accuracy:.3f}")
            print(f"Test AUC: {test_auc:.3f}")
            print(f"CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Store the model
            self.models[model_name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'test_auc': test_auc,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std()
            }
            
        # Find best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['test_auc'])
        print(f"\nBest model: {best_model_name} (AUC: {self.models[best_model_name]['test_auc']:.3f})")
        
        return best_model_name
        
    def analyze_feature_importance(self, model_name: str):
        """Analyze feature importance for tree-based models"""
        if model_name not in self.models:
            return
            
        model = self.models[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create feature importance pairs
            feature_importance = list(zip(self.feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 10 Feature Importances ({model_name}):")
            print("-" * 50)
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print(f"{i+1:2d}. {feature:25s} {importance:.4f}")
                
    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")
        
        for model_name, model_data in self.models.items():
            filename = f'models/ml_{model_name}.joblib'
            joblib.dump(model_data['model'], filename)
            print(f"Saved {model_name} to {filename}")
            
        # Save model performance summary
        performance_summary = {}
        for model_name, model_data in self.models.items():
            performance_summary[model_name] = {
                'test_accuracy': model_data['test_accuracy'],
                'test_auc': model_data['test_auc'],
                'cv_auc_mean': model_data['cv_auc_mean'],
                'cv_auc_std': model_data['cv_auc_std']
            }
            
        with open('models/model_performance.pkl', 'wb') as f:
            pickle.dump(performance_summary, f)
            
    def generate_sample_predictions(self, best_model_name: str):
        """Generate sample predictions to demonstrate model usage"""
        if best_model_name not in self.models:
            return
            
        model = self.models[best_model_name]['model']
        
        # Get a few test samples
        sample_indices = np.random.choice(len(self.X_test), size=5, replace=False)
        
        print(f"\nSample Predictions ({best_model_name}):")
        print("-" * 60)
        print("Sample | Actual | Predicted | Win Probability")
        print("-" * 60)
        
        for i, idx in enumerate(sample_indices):
            actual = self.y_win_test[idx]
            predicted = model.predict([self.X_test[idx]])[0]
            probability = model.predict_proba([self.X_test[idx]])[0][1]
            
            print(f"{i+1:6d} | {actual:6d} | {predicted:9d} | {probability:13.3f}")
            
    def train_and_evaluate(self):
        """Main training pipeline"""
        print("Starting ML model training...")
        
        # Load data
        if not self.load_processed_data():
            return False
            
        # Train models
        best_model_name = self.train_models()
        
        # Analyze feature importance
        self.analyze_feature_importance(best_model_name)
        
        # Save models
        self.save_models()
        
        # Generate sample predictions
        self.generate_sample_predictions(best_model_name)
        
        print("\n" + "="*60)
        print("ML MODEL TRAINING COMPLETE")
        print("="*60)
        print(f"Best performing model: {best_model_name}")
        print(f"Test AUC: {self.models[best_model_name]['test_auc']:.3f}")
        print(f"Models saved to: models/ directory")
        print("\n✅ ML models ready for horse racing predictions!")
        
        return True

def main():
    """Main function"""
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    trainer = HorseRacingMLTrainer()
    success = trainer.train_and_evaluate()
    
    if success:
        print("\n🏆 ML model training completed successfully!")
        print("You can now use these models to predict horse racing outcomes.")
    else:
        print("\n❌ ML model training failed!")

if __name__ == "__main__":
    main()