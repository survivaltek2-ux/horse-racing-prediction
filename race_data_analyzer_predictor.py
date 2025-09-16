#!/usr/bin/env python3
"""
Race Data Analyzer & Predictor
Finds races with the most comprehensive data and generates AI predictions
"""

import json
import sys
import os
from datetime import datetime
from collections import defaultdict
import statistics

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.ai_predictor import AIPredictor
    from utils.predictor import predict_race_outcome
except ImportError as e:
    print(f"Warning: Could not import prediction modules: {e}")
    AIPredictor = None
    predict_race_outcome = None

class RaceDataAnalyzer:
    def __init__(self):
        self.horses = []
        self.races = []
        self.load_data()
    
    def load_data(self):
        """Load horse and race data from JSON files"""
        try:
            with open('data/horses.json', 'r') as f:
                self.horses = json.load(f)
            
            with open('data/races.json', 'r') as f:
                self.races = json.load(f)
                
            print(f"✅ Loaded {len(self.horses)} horses and {len(self.races)} races")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    def analyze_race_data_richness(self):
        """Analyze each race for data richness and competitive strength"""
        race_analysis = []
        
        for race in self.races:
            horse_ids = race.get('horse_ids', [])
            horse_count = len(horse_ids)
            
            if horse_count == 0:
                continue
            
            # Calculate comprehensive data metrics
            total_performances = 0
            total_earnings = 0
            ratings = []
            win_rates = []
            recent_form_scores = []
            
            race_horses = []
            
            for horse_id in horse_ids:
                horse = next((h for h in self.horses if h['id'] == horse_id), None)
                if horse:
                    race_horses.append(horse)
                    performances = len(horse.get('recent_performances', []))
                    total_performances += performances
                    total_earnings += horse.get('total_earnings', 0)
                    
                    # Collect competitive metrics
                    if 'rating' in horse:
                        ratings.append(horse['rating'])
                    if 'win_rate' in horse:
                        win_rates.append(horse['win_rate'])
                    
                    # Calculate recent form score (last 5 races)
                    recent_perfs = horse.get('recent_performances', [])[:5]
                    if recent_perfs:
                        form_score = sum(1 for p in recent_perfs if p.get('position', 99) <= 3) / len(recent_perfs)
                        recent_form_scores.append(form_score)
            
            # Calculate competitive metrics
            avg_rating = statistics.mean(ratings) if ratings else 0
            rating_std = statistics.stdev(ratings) if len(ratings) > 1 else 0
            avg_win_rate = statistics.mean(win_rates) if win_rates else 0
            avg_form_score = statistics.mean(recent_form_scores) if recent_form_scores else 0
            
            # Calculate data richness score
            data_score = (
                horse_count * 10 +  # Field size weight
                total_performances * 0.5 +  # Performance history weight
                (total_earnings / 10000) +  # Earnings weight (scaled)
                avg_rating * 2 +  # Quality weight
                rating_std * 5  # Competitive balance weight
            )
            
            race_analysis.append({
                'race': race,
                'horse_count': horse_count,
                'total_performances': total_performances,
                'total_earnings': total_earnings,
                'avg_rating': avg_rating,
                'rating_std': rating_std,
                'avg_win_rate': avg_win_rate,
                'avg_form_score': avg_form_score,
                'data_score': data_score,
                'horses': race_horses,
                'competitiveness': rating_std * avg_rating,  # High ratings with good spread
                'purse_per_horse': race['purse'] / horse_count if horse_count > 0 else 0
            })
        
        # Sort by data richness score
        race_analysis.sort(key=lambda x: x['data_score'], reverse=True)
        return race_analysis
    
    def predict_race(self, race_data):
        """Generate predictions for a race using available prediction methods"""
        race = race_data['race']
        horses = race_data['horses']
        
        predictions = {
            'race_id': race['id'],
            'race_name': race['name'],
            'predictions': [],
            'analysis': {},
            'confidence': 0.0
        }
        
        try:
            # Try AI Predictor first
            if AIPredictor:
                ai_predictor = AIPredictor()
                ai_result = ai_predictor.predict_race(race['id'])
                if ai_result and 'predictions' in ai_result:
                    predictions['predictions'] = ai_result['predictions']
                    predictions['confidence'] = ai_result.get('confidence', 0.7)
                    predictions['method'] = 'AI Neural Network'
                    return predictions
        except Exception as e:
            print(f"⚠️  AI prediction failed for race {race['id']}: {e}")
        
        try:
            # Fallback to traditional predictor
            if predict_race_outcome:
                traditional_result = predict_race_outcome(race['id'])
                if traditional_result:
                    predictions['predictions'] = traditional_result.get('predictions', [])
                    predictions['confidence'] = 0.6
                    predictions['method'] = 'Traditional Algorithm'
                    return predictions
        except Exception as e:
            print(f"⚠️  Traditional prediction failed for race {race['id']}: {e}")
        
        # Manual prediction based on form analysis
        horse_scores = []
        for horse in horses:
            score = self.calculate_horse_score(horse, race)
            horse_scores.append({
                'horse_id': horse['id'],
                'horse_name': horse['name'],
                'score': score,
                'win_probability': 0.0,
                'place_probability': 0.0,
                'show_probability': 0.0
            })
        
        # Sort by score and assign probabilities
        horse_scores.sort(key=lambda x: x['score'], reverse=True)
        total_score = sum(h['score'] for h in horse_scores)
        
        if total_score > 0:
            for i, horse_pred in enumerate(horse_scores):
                base_prob = horse_pred['score'] / total_score
                # Adjust probabilities based on position
                position_factor = max(0.1, 1.0 - (i * 0.1))
                horse_pred['win_probability'] = base_prob * position_factor
                horse_pred['place_probability'] = min(0.8, base_prob * 1.5)
                horse_pred['show_probability'] = min(0.9, base_prob * 2.0)
        
        predictions['predictions'] = horse_scores
        predictions['confidence'] = 0.5
        predictions['method'] = 'Form Analysis'
        
        return predictions
    
    def calculate_horse_score(self, horse, race):
        """Calculate a comprehensive score for a horse in a specific race"""
        score = 0.0
        
        # Base rating
        score += horse.get('rating', 50)
        
        # Win rate factor
        score += horse.get('win_rate', 0) * 20
        
        # Recent form (last 5 races)
        recent_perfs = horse.get('recent_performances', [])[:5]
        if recent_perfs:
            recent_positions = [p.get('position', 99) for p in recent_perfs]
            avg_position = sum(recent_positions) / len(recent_positions)
            score += max(0, (10 - avg_position) * 2)  # Better recent form = higher score
            
            # Consistency bonus
            if all(pos <= 5 for pos in recent_positions):
                score += 10
        
        # Earnings factor
        total_earnings = horse.get('total_earnings', 0)
        if total_earnings > 0:
            score += min(20, total_earnings / 10000)  # Cap earnings bonus
        
        # Distance suitability (simplified)
        race_distance = race.get('distance', '1200m')
        if 'recent_performances' in horse and horse['recent_performances']:
            # Check if horse has run similar distances
            similar_distances = [p for p in horse['recent_performances'] 
                               if p.get('distance') == race_distance]
            if similar_distances:
                score += 5  # Distance experience bonus
        
        # Track condition factor
        track_condition = race.get('track_condition', 'Good')
        if 'recent_performances' in horse and horse['recent_performances']:
            good_conditions = [p for p in horse['recent_performances'] 
                             if p.get('track_condition') == track_condition and p.get('position', 99) <= 3]
            if good_conditions:
                score += len(good_conditions) * 2
        
        return max(0, score)
    
    def generate_report(self, top_races, predictions):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*80)
        print("🏆 RACE DATA ANALYSIS & PREDICTION REPORT")
        print("="*80)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Analyzed: {len(self.races)} races, {len(self.horses)} horses")
        
        print(f"\n🎯 TOP {len(top_races)} RACES BY DATA RICHNESS:")
        print("-" * 80)
        
        for i, race_data in enumerate(top_races, 1):
            race = race_data['race']
            print(f"\n#{i} 🏁 {race['name']}")
            print(f"    📍 {race['location']} | 📅 {race['date']} | 💰 ${race['purse']:,.0f}")
            print(f"    🐎 {race_data['horse_count']} horses | 📈 {race_data['total_performances']} total performances")
            print(f"    ⭐ Avg Rating: {race_data['avg_rating']:.1f} | 🎲 Competitiveness: {race_data['competitiveness']:.1f}")
            print(f"    💵 Total Earnings: ${race_data['total_earnings']:,.0f} | 📊 Data Score: {race_data['data_score']:.1f}")
            
            # Show predictions if available
            race_predictions = next((p for p in predictions if p['race_id'] == race['id']), None)
            if race_predictions and race_predictions['predictions']:
                print(f"    🔮 Prediction Method: {race_predictions['method']} (Confidence: {race_predictions['confidence']:.1%})")
                print(f"    🥇 Top 3 Predictions:")
                
                for j, pred in enumerate(race_predictions['predictions'][:3], 1):
                    horse_name = pred.get('horse_name', f"Horse {pred.get('horse_id', 'Unknown')}")
                    win_prob = pred.get('win_probability', 0)
                    print(f"       {j}. {horse_name} - Win: {win_prob:.1%}")
        
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"    • Average horses per top race: {sum(r['horse_count'] for r in top_races) / len(top_races):.1f}")
        print(f"    • Total performances in top races: {sum(r['total_performances'] for r in top_races):,}")
        print(f"    • Average purse value: ${sum(r['race']['purse'] for r in top_races) / len(top_races):,.0f}")
        print(f"    • Successful predictions: {len([p for p in predictions if p['predictions']])}/{len(predictions)}")

def main():
    """Main execution function"""
    print("🚀 Starting Race Data Analysis & Prediction System")
    print("="*60)
    
    # Initialize analyzer
    analyzer = RaceDataAnalyzer()
    
    # Analyze race data richness
    print("\n🔍 Analyzing race data richness...")
    race_analysis = analyzer.analyze_race_data_richness()
    
    # Get top 5 races with most data
    top_races = race_analysis[:5]
    
    print(f"\n🎯 Found {len(top_races)} top races for prediction")
    
    # Generate predictions for top races
    print("\n🔮 Generating predictions...")
    predictions = []
    
    for race_data in top_races:
        print(f"   Predicting: {race_data['race']['name']}...")
        try:
            prediction = analyzer.predict_race(race_data)
            predictions.append(prediction)
            print(f"   ✅ Success - {prediction.get('method', 'Unknown')} method")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            predictions.append({
                'race_id': race_data['race']['id'],
                'race_name': race_data['race']['name'],
                'predictions': [],
                'error': str(e)
            })
    
    # Generate comprehensive report
    analyzer.generate_report(top_races, predictions)
    
    # Save results to file
    results = {
        'timestamp': datetime.now().isoformat(),
        'top_races': [r['race'] for r in top_races],
        'race_analysis': [{
            'race_id': r['race']['id'],
            'race_name': r['race']['name'],
            'data_score': r['data_score'],
            'horse_count': r['horse_count'],
            'total_performances': r['total_performances'],
            'competitiveness': r['competitiveness']
        } for r in top_races],
        'predictions': predictions
    }
    
    try:
        with open('race_analysis_predictions.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: race_analysis_predictions.json")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")
    
    print("\n🏁 Analysis complete!")

if __name__ == "__main__":
    main()