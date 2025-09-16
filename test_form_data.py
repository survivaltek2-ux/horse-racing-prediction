#!/usr/bin/env python3
"""
Test Form Data Integration
==========================

This script tests that the generated form data integrates correctly with the Horse model
and can be accessed through the existing methods.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.horse import Horse

def test_form_data_integration():
    """Test that form data is properly integrated"""
    
    print("Testing Form Data Integration")
    print("=" * 40)
    
    # Load all horses
    horses = Horse.get_all_horses()
    
    if not horses:
        print("❌ No horses found!")
        return False
    
    print(f"✅ Found {len(horses)} horses")
    
    # Test each horse's form data
    horses_with_form = 0
    total_performances = 0
    
    for horse in horses:
        print(f"\n🐎 Testing {horse.name}:")
        
        # Check if horse has recent performances
        if hasattr(horse, 'recent_performances') and horse.recent_performances:
            horses_with_form += 1
            num_performances = len(horse.recent_performances)
            total_performances += num_performances
            
            print(f"  ✅ Has {num_performances} recent performances")
            
            # Test get_form method
            form = horse.get_form(5)
            print(f"  ✅ get_form(5) returns {len(form)} entries")
            
            # Show sample performance data
            if form:
                latest = form[0]
                print(f"  📊 Latest race: {latest['date']} at {latest['track']}")
                print(f"      Position: {latest['position']}/{latest['total_runners']}")
                print(f"      Distance: {latest['distance']} on {latest['surface']}")
                print(f"      Earnings: ${latest['earnings']:,.2f}")
            
            # Test statistics
            print(f"  📈 Win Rate: {horse.win_rate:.1%}")
            print(f"  📈 Place Rate: {horse.place_rate:.1%}")
            print(f"  💰 Total Earnings: ${horse.earnings:,.2f}")
            
            # Calculate days since last race from form data
            if form:
                from datetime import datetime
                last_race_date = datetime.strptime(form[0]['date'], '%Y-%m-%d')
                days_since = (datetime.now() - last_race_date).days
                print(f"  📅 Days since last race: {days_since}")
            else:
                print(f"  📅 Days since last race: Unknown")
            
        else:
            print(f"  ❌ No recent performances found")
    
    print(f"\n📊 Summary:")
    print(f"  - Horses with form data: {horses_with_form}/{len(horses)}")
    print(f"  - Total performances: {total_performances}")
    print(f"  - Average performances per horse: {total_performances/len(horses):.1f}")
    
    # Test specific scenarios
    print(f"\n🧪 Testing specific scenarios:")
    
    # Test horse with best win rate
    best_horse = max(horses, key=lambda h: h.win_rate)
    print(f"  🏆 Best win rate: {best_horse.name} ({best_horse.win_rate:.1%})")
    
    # Test horse with most earnings
    richest_horse = max(horses, key=lambda h: h.earnings)
    print(f"  💰 Highest earnings: {richest_horse.name} (${richest_horse.earnings:,.2f})")
    
    # Test horse with most recent race
    def get_days_since_last_race(horse):
        if hasattr(horse, 'recent_performances') and horse.recent_performances:
            from datetime import datetime
            last_race_date = datetime.strptime(horse.recent_performances[0]['date'], '%Y-%m-%d')
            return (datetime.now() - last_race_date).days
        return 999  # Large number for horses with no recent races
    
    most_recent = min(horses, key=get_days_since_last_race)
    days_ago = get_days_since_last_race(most_recent)
    print(f"  🏃 Most recent race: {most_recent.name} ({days_ago} days ago)")
    
    success = horses_with_form == len(horses) and total_performances > 0
    
    if success:
        print(f"\n✅ All tests passed! Form data integration is working correctly.")
    else:
        print(f"\n❌ Some tests failed. Check the form data generation.")
    
    return success

def test_form_data_quality():
    """Test the quality and realism of generated form data"""
    
    print(f"\n🔍 Testing Form Data Quality")
    print("=" * 40)
    
    horses = Horse.get_all_horses()
    
    if not horses:
        return False
    
    # Sample a horse for detailed testing
    test_horse = horses[0]
    print(f"Testing detailed data for: {test_horse.name}")
    
    if not test_horse.recent_performances:
        print("❌ No performances to test")
        return False
    
    performances = test_horse.recent_performances
    
    # Test data completeness
    required_fields = [
        'race_id', 'date', 'track', 'race_name', 'distance', 'surface',
        'track_condition', 'position', 'total_runners', 'time', 'earnings'
    ]
    
    print(f"\n📋 Testing data completeness:")
    for field in required_fields:
        if all(field in perf for perf in performances):
            print(f"  ✅ {field}: Present in all performances")
        else:
            print(f"  ❌ {field}: Missing in some performances")
    
    # Test data realism
    print(f"\n🎯 Testing data realism:")
    
    # Check date ordering (should be chronological, newest first)
    dates = [perf['date'] for perf in performances]
    if dates == sorted(dates, reverse=True):
        print(f"  ✅ Dates are properly ordered (newest first)")
    else:
        print(f"  ❌ Dates are not properly ordered")
    
    # Check position ranges
    positions = [perf['position'] for perf in performances]
    if all(1 <= pos <= perf['total_runners'] for pos, perf in zip(positions, performances)):
        print(f"  ✅ All positions are within valid range")
    else:
        print(f"  ❌ Some positions are outside valid range")
    
    # Check earnings are non-negative
    earnings = [perf['earnings'] for perf in performances]
    if all(e >= 0 for e in earnings):
        print(f"  ✅ All earnings are non-negative")
    else:
        print(f"  ❌ Some earnings are negative")
    
    # Show sample performance
    print(f"\n📄 Sample performance:")
    sample = performances[0]
    for key, value in sample.items():
        if key != 'sectional_times':  # Skip complex nested data
            print(f"  {key}: {value}")
    
    return True

def main():
    """Main test function"""
    
    try:
        integration_success = test_form_data_integration()
        quality_success = test_form_data_quality()
        
        if integration_success and quality_success:
            print(f"\n🎉 All tests completed successfully!")
            print(f"The recent form data has been generated and integrated correctly.")
        else:
            print(f"\n⚠️  Some tests failed. Please check the output above.")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()