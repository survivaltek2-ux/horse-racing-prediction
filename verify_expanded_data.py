#!/usr/bin/env python3
"""
Comprehensive verification script for the expanded form data
"""

import json
from datetime import datetime, timedelta
import statistics

def main():
    print('🐎 COMPREHENSIVE FORM DATA ANALYSIS')
    print('=' * 50)
    
    # Load the data
    with open('data/horses.json', 'r') as f:
        horses = json.load(f)
    
    # Basic statistics
    total_horses = len(horses)
    total_performances = sum(len(horse.get('recent_performances', [])) for horse in horses)
    avg_performances = total_performances / total_horses if total_horses > 0 else 0
    
    print(f'📊 Dataset Overview:')
    print(f'   Total horses: {total_horses:,}')
    print(f'   Total performances: {total_performances:,}')
    print(f'   Average performances per horse: {avg_performances:.1f}')
    
    # Performance distribution
    performance_counts = [len(horse.get('recent_performances', [])) for horse in horses]
    print(f'   Min performances per horse: {min(performance_counts)}')
    print(f'   Max performances per horse: {max(performance_counts)}')
    print(f'   Median performances per horse: {statistics.median(performance_counts):.1f}')
    
    # Track and surface variety
    tracks = set()
    surfaces = set()
    distances = set()
    jockeys = set()
    trainers = set()
    
    for horse in horses:
        for perf in horse.get('recent_performances', []):
            tracks.add(perf.get('track', ''))
            surfaces.add(perf.get('surface', ''))
            distances.add(perf.get('distance', ''))
            jockeys.add(perf.get('jockey', ''))
            trainers.add(perf.get('trainer', ''))
    
    print(f'\n🏁 Racing Variety:')
    print(f'   Unique tracks: {len(tracks)}')
    print(f'   Unique surfaces: {len(surfaces)}')
    print(f'   Unique distances: {len(distances)}')
    print(f'   Unique jockeys: {len(jockeys)}')
    print(f'   Unique trainers: {len(trainers)}')
    
    # Sample some tracks and surfaces
    print(f'   Sample tracks: {list(tracks)[:10]}')
    print(f'   Sample surfaces: {list(surfaces)}')
    
    # Earnings analysis
    total_earnings = []
    for horse in horses:
        horse_earnings = sum(float(perf.get('earnings', 0)) for perf in horse.get('recent_performances', []))
        total_earnings.append(horse_earnings)
    
    print(f'\n💰 Earnings Analysis:')
    print(f'   Total prize money: ${sum(total_earnings):,.2f}')
    print(f'   Average earnings per horse: ${statistics.mean(total_earnings):,.2f}')
    print(f'   Highest earning horse: ${max(total_earnings):,.2f}')
    
    # Sample horse details
    sample_horse = horses[0]
    print(f'\n🎯 Sample Horse: {sample_horse.get("name", "Unknown")}')
    print(f'   Total performances: {len(sample_horse.get("recent_performances", []))}')
    if sample_horse.get('recent_performances'):
        recent = sample_horse['recent_performances'][0]
        print(f'   Most recent race: {recent.get("date", "")} at {recent.get("track", "")}')
        print(f'   Position: {recent.get("position", "")} / Distance: {recent.get("distance", "")}')
        print(f'   Time: {recent.get("time", "")} / Margin: {recent.get("margin", "")}')
    
    # Test integration with Horse model
    print(f'\n🔧 Integration Test:')
    try:
        from models.horse import Horse
        
        # Test a few horses
        test_horses = horses[:3]
        for horse_data in test_horses:
            horse = Horse(horse_data)
            form = horse.get_form()
            print(f'   {horse.name}: Form method returns {len(form)} entries')
            
            if form:
                latest = form[0]
                print(f'     Latest: {latest.get("date", "")} - Position {latest.get("position", "")}')
        
        print(f'   ✅ Integration test passed!')
        
    except Exception as e:
        print(f'   ❌ Integration test failed: {e}')
    
    print(f'\n✅ Data quality verification complete!')
    print(f'🚀 Ready for prediction algorithms with {total_performances:,} race performances!')

if __name__ == "__main__":
    main()