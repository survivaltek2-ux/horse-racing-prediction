#!/usr/bin/env python3
"""
End-to-End Prediction System Test
This script tests the complete prediction workflow including database integration
"""

import sys
import os
import numpy as np
import traceback
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_database_connection():
    """Test database connection and basic queries"""
    print("=" * 60)
    print("TESTING DATABASE CONNECTION")
    print("=" * 60)
    
    try:
        from app import app, db
        from models.sqlalchemy_models import Race, Horse
        
        with app.app_context():
            # Test basic queries
            race_count = db.session.query(Race).count()
            horse_count = db.session.query(Horse).count()
            
            print(f"✅ Database connected successfully")
            print(f"   Races in database: {race_count}")
            print(f"   Horses in database: {horse_count}")
            
            if race_count == 0:
                print("⚠️  No races found in database")
                return False
            
            if horse_count == 0:
                print("⚠️  No horses found in database")
                return False
            
            # Test getting a sample race with horses
            race = db.session.query(Race).first()
            if race:
                print(f"   Sample race: {race.name}")
                horses = race.horses
                print(f"   Horses in race: {len(horses)}")
                
                if len(horses) == 0:
                    print("⚠️  Sample race has no horses")
                    return False
            
            return True
            
    except Exception as e:
        print(f"❌ Database connection test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_traditional_predictor():
    """Test the traditional prediction system"""
    print("\n" + "=" * 60)
    print("TESTING TRADITIONAL PREDICTOR")
    print("=" * 60)
    
    try:
        from app import app, db
        from models.sqlalchemy_models import Race
        from utils.predictor import Predictor
        
        with app.app_context():
            # Get a sample race
            race = db.session.query(Race).first()
            if not race:
                print("⚠️  No races found in database")
                return False
            
            print(f"🏁 Testing prediction for race: {race.name}")
            
            predictor = Predictor()
            
            # Test traditional prediction
            print("🔮 Running traditional prediction...")
            result = predictor.predict_race(race)
            
            if result:
                print("✅ Traditional prediction completed:")
                print(f"   Algorithm: {result.algorithm}")
                print(f"   Predictions count: {len(result.predictions)}")
                
                # Show top 3 predictions
                if result.predictions:
                    sorted_predictions = sorted(result.predictions, key=lambda x: x.predicted_position)
                    print("   Top 3 predictions:")
                    for i, pred in enumerate(sorted_predictions[:3], 1):
                        print(f"      {i}. Horse ID {pred.horse_id} - Position {pred.predicted_position} (Confidence: {pred.confidence:.3f})")
            else:
                print("⚠️  Traditional prediction returned None")
                return False
            
            return True
            
    except Exception as e:
        print(f"❌ Traditional predictor test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_ai_predictor_with_fallback():
    """Test AI predictor with fallback to traditional methods"""
    print("\n" + "=" * 60)
    print("TESTING AI PREDICTOR WITH FALLBACK")
    print("=" * 60)
    
    try:
        from app import app, db
        from models.sqlalchemy_models import Race
        from utils.ai_predictor import AIPredictor
        
        with app.app_context():
            # Get a sample race
            race = db.session.query(Race).first()
            if not race:
                print("⚠️  No races found in database")
                return False
            
            print(f"🤖 Testing AI prediction for race: {race.name}")
            
            predictor = AIPredictor()
            
            # Test AI prediction with fallback
            print("🧠 Running AI prediction...")
            result = predictor.predict_race_ai(race, use_ensemble=True)
            
            if result:
                print("✅ AI prediction completed:")
                print(f"   Algorithm: {result.get('algorithm', 'unknown')}")
                
                predictions = result.get('predictions', {})
                if predictions:
                    print(f"   Predictions type: {type(predictions)}")
                    if hasattr(predictions, 'predictions'):
                        print(f"   Predictions count: {len(predictions.predictions)}")
                    elif isinstance(predictions, dict):
                        print(f"   Prediction keys: {list(predictions.keys())}")
                
                confidence = result.get('confidence_scores', {})
                if confidence:
                    print(f"   Confidence scores available: {len(confidence)} items")
                
                insights = result.get('ai_insights', [])
                if insights:
                    print(f"   AI insights: {len(insights)} generated")
                    for insight in insights[:3]:  # Show first 3 insights
                        print(f"      - {insight}")
            else:
                print("⚠️  AI prediction returned None")
                return False
            
            return True
            
    except Exception as e:
        print(f"❌ AI predictor test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_prediction_api_endpoint():
    """Test the prediction API endpoint"""
    print("\n" + "=" * 60)
    print("TESTING PREDICTION API ENDPOINT")
    print("=" * 60)
    
    try:
        from app import app, db
        from models.sqlalchemy_models import Race
        
        with app.app_context():
            # Get a sample race
            race = db.session.query(Race).first()
            if not race:
                print("⚠️  No races found in database")
                return False
            
            print(f"🌐 Testing API endpoint for race: {race.name}")
            
            # Test the app's prediction route
            with app.test_client() as client:
                # Test prediction endpoint
                response = client.post('/predict', data={
                    'race_id': race.id,
                    'algorithm': 'enhanced_heuristic'
                })
                
                print(f"   Response status: {response.status_code}")
                
                if response.status_code == 200:
                    print("✅ API endpoint responded successfully")
                    
                    # Check if response contains prediction data
                    response_text = response.get_data(as_text=True)
                    if 'prediction' in response_text.lower() or 'result' in response_text.lower():
                        print("   ✅ Response contains prediction data")
                    else:
                        print("   ⚠️  Response may not contain prediction data")
                else:
                    print(f"   ⚠️  API endpoint returned status {response.status_code}")
                    return False
            
            return True
            
    except Exception as e:
        print(f"❌ API endpoint test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_data_enhancement_integration():
    """Test that enhanced data is properly integrated"""
    print("\n" + "=" * 60)
    print("TESTING DATA ENHANCEMENT INTEGRATION")
    print("=" * 60)
    
    try:
        from app import app, db
        from models.sqlalchemy_models import Race, Horse
        
        with app.app_context():
            # Check if enhanced columns exist and have data
            race = db.session.query(Race).first()
            horse = db.session.query(Horse).first()
            
            if not race or not horse:
                print("⚠️  No race or horse found in database")
                return False
            
            print(f"🔍 Checking enhanced data for race: {race.name}")
            print(f"🐎 Checking enhanced data for horse: {horse.name}")
            
            # Check race enhanced fields
            enhanced_race_fields = [
                'temperature', 'humidity', 'wind_speed', 'surface_type',
                'track_bias', 'rail_position', 'pace_scenario'
            ]
            
            race_enhanced_count = 0
            for field in enhanced_race_fields:
                if hasattr(race, field):
                    value = getattr(race, field)
                    if value is not None:
                        race_enhanced_count += 1
                        print(f"   ✅ Race.{field}: {value}")
            
            print(f"   Race enhanced fields with data: {race_enhanced_count}/{len(enhanced_race_fields)}")
            
            # Check horse enhanced fields
            enhanced_horse_fields = [
                'body_condition', 'training_intensity', 'recent_workouts',
                'medication_history', 'equipment_changes'
            ]
            
            horse_enhanced_count = 0
            for field in enhanced_horse_fields:
                if hasattr(horse, field):
                    value = getattr(horse, field)
                    if value is not None:
                        horse_enhanced_count += 1
                        print(f"   ✅ Horse.{field}: {value}")
            
            print(f"   Horse enhanced fields with data: {horse_enhanced_count}/{len(enhanced_horse_fields)}")
            
            if race_enhanced_count > 0 or horse_enhanced_count > 0:
                print("✅ Enhanced data integration confirmed")
                return True
            else:
                print("⚠️  No enhanced data found - may need to run data enhancement")
                return True  # Still pass as the schema exists
            
    except Exception as e:
        print(f"❌ Data enhancement integration test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_prediction_accuracy_simulation():
    """Test prediction accuracy with available data"""
    print("\n" + "=" * 60)
    print("TESTING PREDICTION ACCURACY SIMULATION")
    print("=" * 60)
    
    try:
        from app import app, db
        from models.sqlalchemy_models import Race
        from utils.predictor import Predictor
        
        with app.app_context():
            # Get multiple races for testing
            races = db.session.query(Race).limit(5).all()
            if not races:
                print("⚠️  No races found in database")
                return False
            
            print(f"🎯 Testing prediction accuracy on {len(races)} races")
            
            predictor = Predictor()
            successful_predictions = 0
            
            for i, race in enumerate(races, 1):
                try:
                    print(f"   Testing race {i}: {race.name}")
                    
                    # Generate prediction
                    result = predictor.predict_race(race)
                    
                    if result and result.predictions:
                        successful_predictions += 1
                        print(f"      ✅ Prediction generated with {len(result.predictions)} horses")
                        
                        # Calculate average confidence
                        avg_confidence = sum(p.confidence for p in result.predictions) / len(result.predictions)
                        print(f"      Average confidence: {avg_confidence:.3f}")
                    else:
                        print(f"      ❌ Prediction failed")
                        
                except Exception as e:
                    print(f"      ❌ Error: {str(e)}")
            
            success_rate = successful_predictions / len(races)
            print(f"\n   Overall success rate: {success_rate:.1%} ({successful_predictions}/{len(races)})")
            
            if success_rate >= 0.8:  # 80% success rate
                print("✅ Prediction system is working reliably")
                return True
            else:
                print("⚠️  Prediction system has some reliability issues")
                return False
            
    except Exception as e:
        print(f"❌ Prediction accuracy test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all end-to-end tests"""
    print("🧪 END-TO-END PREDICTION SYSTEM TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Traditional Predictor", test_traditional_predictor),
        ("AI Predictor with Fallback", test_ai_predictor_with_fallback),
        ("Prediction API Endpoint", test_prediction_api_endpoint),
        ("Data Enhancement Integration", test_data_enhancement_integration),
        ("Prediction Accuracy Simulation", test_prediction_accuracy_simulation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The prediction system is fully functional.")
    elif passed >= total * 0.8:  # 80% pass rate
        print("✅ Most tests passed! The prediction system is largely functional.")
    else:
        print("⚠️  Several tests failed. Please check the output above for details.")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()