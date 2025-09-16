#!/usr/bin/env python3
"""
Enhanced ML Training Script for Horse Racing Predictions

This script provides comprehensive model training with hyperparameter tuning,
detailed evaluation, and model comparison for horse racing predictions.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, accuracy_score, roc_auc_score, 
                           confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class EnhancedHorseRacingTrainer:
    """Enhanced ML trainer with comprehensive evaluation and tuning"""
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.feature_names = []
        self.results_summary = {}
        
    def load_data(self):
        """Load preprocessed training data"""
        try:
            self.X_train = np.load('data/X_train.npy')
            self.X_test = np.load('data/X_test.npy')
            self.y_win_train = np.load('data/y_win_train.npy')
            self.y_win_test = np.load('data/y_win_test.npy')
            
            # Load additional targets
            self.y_place_train = np.load('data/y_place_train.npy')
            self.y_place_test = np.load('data/y_place_test.npy')
            self.y_position_train = np.load('data/y_position_train.npy')
            self.y_position_test = np.load('data/y_position_test.npy')
            
            with open('data/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
                
            print(f"✅ Data loaded successfully")
            print(f"Training samples: {self.X_train.shape[0]}")
            print(f"Test samples: {self.X_test.shape[0]}")
            print(f"Features: {len(self.feature_names)}")
            print(f"Win rate: {np.mean(self.y_win_train):.3f}")
            print(f"Place rate: {np.mean(self.y_place_train):.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
            
    def define_models(self):
        """Define models with hyperparameter grids for tuning"""
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'svm': {
                'model': SVC(random_state=42, class_weight='balanced', probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
    def train_with_hyperparameter_tuning(self, target='win'):
        """Train models with hyperparameter tuning"""
        print(f"\n🔧 Training models for {target} prediction with hyperparameter tuning...")
        
        # Select target
        if target == 'win':
            y_train, y_test = self.y_win_train, self.y_win_test
        elif target == 'place':
            y_train, y_test = self.y_place_train, self.y_place_test
        else:
            print(f"❌ Unknown target: {target}")
            return
            
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, config in self.model_configs.items():
            print(f"\n🏃 Training {model_name}...")
            
            try:
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'],
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(self.X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Calibrate probabilities
                calibrated_model = CalibratedClassifierCV(best_model, cv=3)
                calibrated_model.fit(self.X_train, y_train)
                
                # Predictions
                y_pred_train = calibrated_model.predict(self.X_train)
                y_pred_test = calibrated_model.predict(self.X_test)
                y_pred_proba_test = calibrated_model.predict_proba(self.X_test)[:, 1]
                
                # Metrics
                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc = accuracy_score(y_test, y_pred_test)
                test_auc = roc_auc_score(y_test, y_pred_proba_test)
                
                # Cross-validation scores
                cv_scores = cross_val_score(calibrated_model, self.X_train, y_train, 
                                          cv=cv, scoring='roc_auc')
                
                # Store results
                self.best_models[f"{model_name}_{target}"] = {
                    'model': calibrated_model,
                    'best_params': grid_search.best_params_,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'test_auc': test_auc,
                    'cv_auc_mean': cv_scores.mean(),
                    'cv_auc_std': cv_scores.std(),
                    'predictions': y_pred_proba_test,
                    'target': target
                }
                
                print(f"✅ {model_name}: AUC={test_auc:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue
                
    def evaluate_models(self, target='win'):
        """Comprehensive model evaluation"""
        print(f"\n📊 Evaluating {target} prediction models...")
        
        target_models = {k: v for k, v in self.best_models.items() if v['target'] == target}
        
        if not target_models:
            print(f"❌ No models found for target: {target}")
            return
            
        # Find best model
        best_model_name = max(target_models.keys(), key=lambda x: target_models[x]['test_auc'])
        best_model_data = target_models[best_model_name]
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"AUC: {best_model_data['test_auc']:.3f}")
        print(f"Accuracy: {best_model_data['test_accuracy']:.3f}")
        print(f"Best params: {best_model_data['best_params']}")
        
        # Detailed evaluation for best model
        y_test = self.y_win_test if target == 'win' else self.y_place_test
        y_pred_proba = best_model_data['predictions']
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print(f"\n📈 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance (if available)
        model = best_model_data['model']
        if hasattr(model.base_estimator, 'feature_importances_'):
            self.analyze_feature_importance(model.base_estimator, target)
        elif hasattr(model, 'feature_importances_'):
            self.analyze_feature_importance(model, target)
            
        return best_model_name, best_model_data
        
    def analyze_feature_importance(self, model, target):
        """Analyze and display feature importance"""
        if not hasattr(model, 'feature_importances_'):
            return
            
        importances = model.feature_importances_
        feature_importance = list(zip(self.feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔍 Top 10 Feature Importances ({target}):")
        print("-" * 60)
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"{i+1:2d}. {feature:30s} {importance:.4f}")
            
    def generate_predictions_analysis(self):
        """Generate detailed predictions analysis"""
        print(f"\n🎯 Generating predictions analysis...")
        
        # Load processed dataset for context
        try:
            df = pd.read_csv('data/processed_dataset.csv')
            
            # Get best win model
            win_models = {k: v for k, v in self.best_models.items() if v['target'] == 'win'}
            if not win_models:
                print("❌ No win models available")
                return
                
            best_win_model_name = max(win_models.keys(), key=lambda x: win_models[x]['test_auc'])
            best_model = win_models[best_win_model_name]['model']
            
            # Sample predictions
            sample_indices = np.random.choice(len(self.X_test), size=10, replace=False)
            
            print(f"\n🔮 Sample Predictions ({best_win_model_name}):")
            print("-" * 80)
            print("Sample | Actual | Predicted | Win Prob | Horse Age | Win Rate | Earnings")
            print("-" * 80)
            
            for i, idx in enumerate(sample_indices):
                actual = self.y_win_test[idx]
                predicted = best_model.predict([self.X_test[idx]])[0]
                probability = best_model.predict_proba([self.X_test[idx]])[0][1]
                
                # Get additional context (assuming feature order)
                age = self.X_test[idx][0] if len(self.X_test[idx]) > 0 else 0
                win_rate = self.X_test[idx][2] if len(self.X_test[idx]) > 2 else 0
                earnings = self.X_test[idx][1] if len(self.X_test[idx]) > 1 else 0
                
                print(f"{i+1:6d} | {actual:6d} | {predicted:9d} | {probability:8.3f} | "
                      f"{age:8.1f} | {win_rate:8.3f} | {earnings:8.0f}")
                      
        except Exception as e:
            print(f"❌ Error in predictions analysis: {e}")
            
    def save_models_and_results(self):
        """Save trained models and results"""
        print(f"\n💾 Saving models and results...")
        
        # Save individual models
        for model_name, model_data in self.best_models.items():
            filename = f'models/enhanced_{model_name}.joblib'
            joblib.dump(model_data['model'], filename)
            
        # Save comprehensive results
        results_summary = {}
        for model_name, model_data in self.best_models.items():
            results_summary[model_name] = {
                'test_auc': model_data['test_auc'],
                'test_accuracy': model_data['test_accuracy'],
                'cv_auc_mean': model_data['cv_auc_mean'],
                'cv_auc_std': model_data['cv_auc_std'],
                'best_params': model_data['best_params'],
                'target': model_data['target']
            }
            
        with open('models/enhanced_model_results.pkl', 'wb') as f:
            pickle.dump(results_summary, f)
            
        # Save feature importance if available
        feature_importances = {}
        for model_name, model_data in self.best_models.items():
            model = model_data['model']
            if hasattr(model.base_estimator, 'feature_importances_'):
                feature_importances[model_name] = dict(zip(
                    self.feature_names, 
                    model.base_estimator.feature_importances_
                ))
            elif hasattr(model, 'feature_importances_'):
                feature_importances[model_name] = dict(zip(
                    self.feature_names, 
                    model.feature_importances_
                ))
                
        if feature_importances:
            with open('models/feature_importances.pkl', 'wb') as f:
                pickle.dump(feature_importances, f)
                
        print(f"✅ Saved {len(self.best_models)} models")
        
    def print_final_summary(self):
        """Print comprehensive training summary"""
        print("\n" + "="*80)
        print("🏁 ENHANCED TRAINING COMPLETE")
        print("="*80)
        
        # Win prediction models
        win_models = {k: v for k, v in self.best_models.items() if v['target'] == 'win'}
        if win_models:
            best_win = max(win_models.keys(), key=lambda x: win_models[x]['test_auc'])
            print(f"🏆 Best Win Predictor: {best_win}")
            print(f"   AUC: {win_models[best_win]['test_auc']:.3f}")
            print(f"   Accuracy: {win_models[best_win]['test_accuracy']:.3f}")
            
        # Place prediction models
        place_models = {k: v for k, v in self.best_models.items() if v['target'] == 'place'}
        if place_models:
            best_place = max(place_models.keys(), key=lambda x: place_models[x]['test_auc'])
            print(f"🥉 Best Place Predictor: {best_place}")
            print(f"   AUC: {place_models[best_place]['test_auc']:.3f}")
            print(f"   Accuracy: {place_models[best_place]['test_accuracy']:.3f}")
            
        print(f"\n📁 Models saved to: models/ directory")
        print(f"📊 Results saved to: models/enhanced_model_results.pkl")
        print(f"🔍 Feature importance: models/feature_importances.pkl")
        print("\n✅ Ready for production use!")
        
    def run_complete_training(self):
        """Run the complete enhanced training pipeline"""
        print("🚀 Starting Enhanced ML Training Pipeline...")
        
        # Load data
        if not self.load_data():
            return False
            
        # Define models
        self.define_models()
        
        # Train for win prediction
        self.train_with_hyperparameter_tuning('win')
        
        # Train for place prediction  
        self.train_with_hyperparameter_tuning('place')
        
        # Evaluate models
        self.evaluate_models('win')
        self.evaluate_models('place')
        
        # Generate predictions analysis
        self.generate_predictions_analysis()
        
        # Save everything
        self.save_models_and_results()
        
        # Final summary
        self.print_final_summary()
        
        return True

def main():
    """Main execution function"""
    import os
    os.makedirs('models', exist_ok=True)
    
    trainer = EnhancedHorseRacingTrainer()
    success = trainer.run_complete_training()
    
    if success:
        print("\n🎉 Enhanced training completed successfully!")
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()