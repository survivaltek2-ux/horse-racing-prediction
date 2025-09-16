#!/usr/bin/env python3
"""
Verify the horse-race integration
"""

import json

def main():
    with open('data/races.json', 'r') as f:
        races = json.load(f)

    print('✅ INTEGRATION VERIFICATION')
    print('=' * 40)

    total_horses_assigned = 0
    for race in races:
        horse_count = len(race.get('horse_ids', []))
        total_horses_assigned += horse_count
        race_name = race['name'][:25]
        purse = race['purse']
        print(f'🏁 {race_name:<25} | {horse_count:2d} horses | {race["date"]} | ${purse:,.0f}')

    print(f'\n📊 Summary:')
    print(f'   Total races: {len(races)}')
    print(f'   Total assignments: {total_horses_assigned}')
    print(f'   Average horses per race: {total_horses_assigned/len(races):.1f}')

if __name__ == "__main__":
    main()