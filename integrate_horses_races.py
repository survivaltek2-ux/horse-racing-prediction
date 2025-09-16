#!/usr/bin/env python3
"""
Script to integrate the expanded horse data with existing races
"""

import json
import random
from datetime import datetime, timedelta

def main():
    print('🔗 INTEGRATING HORSES WITH RACES')
    print('=' * 50)
    
    # Load data
    with open('data/horses.json', 'r') as f:
        horses = json.load(f)
    
    with open('data/races.json', 'r') as f:
        races = json.load(f)
    
    print(f'📊 Loaded {len(horses)} horses and {len(races)} races')
    
    # Get all horse IDs
    horse_ids = [horse['id'] for horse in horses]
    
    # Assign horses to races
    for race in races:
        race_id = race.get('id', 0)
        
        if race_id == 1:
            # Keep the existing assignment for race 1
            print(f'🏁 Race {race_id}: Keeping existing {len(race.get("horse_ids", []))} horses')
            continue
        
        # For other races, assign random horses
        num_horses = random.randint(8, 16)  # Realistic field size
        selected_horses = random.sample(horse_ids, num_horses)
        race['horse_ids'] = selected_horses
        
        print(f'🏁 Race {race_id}: Assigned {len(selected_horses)} horses')
        print(f'   Horses: {selected_horses[:5]}{"..." if len(selected_horses) > 5 else ""}')
    
    # Create additional races to showcase more horses
    additional_races = []
    next_race_id = max(race['id'] for race in races) + 1
    
    # Create 10 more races with different characteristics
    race_templates = [
        {"name": "Sprint Championship", "distance": "1000m", "purse": 80000.0},
        {"name": "Mile Classic", "distance": "1600m", "purse": 100000.0},
        {"name": "Derby Trial", "distance": "2000m", "purse": 120000.0},
        {"name": "Fillies Stakes", "distance": "1400m", "purse": 75000.0},
        {"name": "Handicap Race", "distance": "1800m", "purse": 90000.0},
        {"name": "Maiden Special", "distance": "1200m", "purse": 40000.0},
        {"name": "Turf Championship", "distance": "2200m", "purse": 150000.0},
        {"name": "Juvenile Stakes", "distance": "1000m", "purse": 60000.0},
        {"name": "Oaks Trial", "distance": "1600m", "purse": 85000.0},
        {"name": "Cup Final", "distance": "2400m", "purse": 200000.0}
    ]
    
    base_date = datetime.now() + timedelta(days=1)
    
    for i, template in enumerate(race_templates):
        race_date = base_date + timedelta(days=i)
        
        # Select horses for this race
        num_horses = random.randint(10, 18)
        selected_horses = random.sample(horse_ids, num_horses)
        
        new_race = {
            "id": next_race_id + i,
            "name": template["name"],
            "date": race_date.strftime("%Y-%m-%d"),
            "location": random.choice([
                "Churchill Downs", "Belmont Park", "Santa Anita", 
                "Keeneland", "Saratoga", "Del Mar", "Gulfstream Park"
            ]),
            "distance": template["distance"],
            "track_condition": random.choice(["Good", "Firm", "Soft", "Heavy"]),
            "race_type": random.choice(["Stakes", "Allowance", "Maiden", "Claiming"]),
            "purse": template["purse"],
            "horse_ids": selected_horses,
            "results": {},
            "status": "upcoming"
        }
        
        additional_races.append(new_race)
        print(f'🆕 Created race {new_race["id"]}: {new_race["name"]} ({len(selected_horses)} horses)')
    
    # Combine all races
    all_races = races + additional_races
    
    # Save updated races
    with open('data/races.json', 'w') as f:
        json.dump(all_races, f, indent=2)
    
    # Generate integration statistics
    total_assignments = 0
    assigned_horses = set()
    
    for race in all_races:
        race_horses = race.get('horse_ids', [])
        total_assignments += len(race_horses)
        assigned_horses.update(race_horses)
    
    print(f'\n📈 Integration Complete!')
    print(f'   Total races: {len(all_races)}')
    print(f'   Total horse assignments: {total_assignments}')
    print(f'   Unique horses assigned: {len(assigned_horses)}')
    print(f'   Coverage: {len(assigned_horses)}/{len(horses)} horses ({len(assigned_horses)/len(horses)*100:.1f}%)')
    
    # Show sample assignments
    print(f'\n🎯 Sample Race Assignments:')
    for race in all_races[:3]:
        horses_in_race = race.get('horse_ids', [])
        print(f'   {race["name"]}: {len(horses_in_race)} horses')

if __name__ == "__main__":
    main()