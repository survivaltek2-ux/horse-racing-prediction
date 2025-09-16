#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.race import Race
from utils.predictor import Predictor

def test_prediction():
    print("=== Testing Prediction Debug ===")
    
    # Get race 1 which has horses
    race = Race.get_race_by_id(1)
    print(f"Race found: {race}")
    print(f"Race name: {race.name if race else 'None'}")
    print(f"Race horse_ids: {race.horse_ids if race else 'None'}")
    
    if race:
        horses = race.get_horses()
        print(f"Horses retrieved: {len(horses) if horses else 0}")
        for i, horse in enumerate(horses[:3]):  # Show first 3 horses
            print(f"  Horse {i+1}: {horse.name if horse else 'None'} (ID: {horse.id if horse else 'None'})")
    
    # Test prediction
    predictor = Predictor()
    print("\n=== Calling predict_race ===")
    prediction = predictor.predict_race(race, {'algorithm': 'enhanced_heuristic'})
    print(f"Prediction result: {prediction}")
    print(f"Prediction type: {type(prediction)}")
    
    if prediction:
        print("Prediction successful!")
        try:
            top_picks = prediction.get_top_picks(3)
            print(f"Top picks: {top_picks}")
        except Exception as e:
            print(f"Error getting top picks: {e}")
    else:
        print("Prediction failed - returned None")

if __name__ == "__main__":
    test_prediction()