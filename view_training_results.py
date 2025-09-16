#!/usr/bin/env python3
"""
View Enhanced Training Results

This script loads and displays the comprehensive training results
from the enhanced training pipeline.
"""

import pickle
import pandas as pd
import numpy as np

def load_and_display_results():
    """Load and display comprehensive training results"""
    
    try:
        # Load results
        with open('models/enhanced_model_results.pkl', 'rb') as f:
            results = pickle.load(f)
            
        print("🏆 ENHANCED TRAINING RESULTS SUMMARY")
        print("=" * 80)
        
        # Separate win and place models
        win_models = {k: v for k, v in results.items() if v['target'] == 'win'}
        place_models = {k: v for k, v in results.items() if v['target'] == 'place'}
        
        # Win prediction results
        if win_models:
            print("\n🎯 WIN PREDICTION MODELS")
            print("-" * 60)
            win_df = pd.DataFrame(win_models).T
            win_df = win_df.sort_values('test_auc', ascending=False)
            
            print(f"{'Model':<25} {'Test AUC':<10} {'Test Acc':<10} {'CV AUC':<15}")
            print("-" * 60)
            for idx, row in win_df.iterrows():
                model_name = idx.replace('_win', '')
                cv_score = f"{row['cv_auc_mean']:.3f}±{row['cv_auc_std']:.3f}"
                print(f"{model_name:<25} {row['test_auc']:<10.3f} {row['test_accuracy']:<10.3f} {cv_score:<15}")
                
        # Place prediction results
        if place_models:
            print("\n🥉 PLACE PREDICTION MODELS")
            print("-" * 60)
            place_df = pd.DataFrame(place_models).T
            place_df = place_df.sort_values('test_auc', ascending=False)
            
            print(f"{'Model':<25} {'Test AUC':<10} {'Test Acc':<10} {'CV AUC':<15}")
            print("-" * 60)
            for idx, row in place_df.iterrows():
                model_name = idx.replace('_place', '')
                cv_score = f"{row['cv_auc_mean']:.3f}±{row['cv_auc_std']:.3f}"
                print(f"{model_name:<25} {row['test_auc']:<10.3f} {row['test_accuracy']:<10.3f} {cv_score:<15}")
        
        # Best models summary
        print("\n🏆 BEST MODELS")
        print("-" * 40)
        if win_models:
            best_win = max(win_models.keys(), key=lambda x: win_models[x]['test_auc'])
            print(f"Best Win Predictor: {best_win}")
            print(f"  AUC: {win_models[best_win]['test_auc']:.3f}")
            print(f"  Accuracy: {win_models[best_win]['test_accuracy']:.3f}")
            print(f"  Best Params: {win_models[best_win]['best_params']}")
            
        if place_models:
            best_place = max(place_models.keys(), key=lambda x: place_models[x]['test_auc'])
            print(f"\nBest Place Predictor: {best_place}")
            print(f"  AUC: {place_models[best_place]['test_auc']:.3f}")
            print(f"  Accuracy: {place_models[best_place]['test_accuracy']:.3f}")
            print(f"  Best Params: {place_models[best_place]['best_params']}")
            
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
        
    # Load feature importance if available
    try:
        with open('models/feature_importances.pkl', 'rb') as f:
            feature_importance = pickle.load(f)
            
        print("\n🔍 FEATURE IMPORTANCE (Top Features)")
        print("-" * 60)
        
        for model_name, features in feature_importance.items():
            if 'win' in model_name:  # Show win model feature importance
                print(f"\n{model_name}:")
                sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(sorted_features[:10]):
                    print(f"  {i+1:2d}. {feature:<30} {importance:.4f}")
                break  # Just show one model to avoid clutter
                
    except Exception as e:
        print(f"⚠️  Feature importance not available: {e}")
        
    print("\n✅ Results loaded successfully!")
    print(f"📁 Models available in: models/ directory")
    print(f"📊 Use joblib.load() to load specific models for predictions")

if __name__ == "__main__":
    load_and_display_results()