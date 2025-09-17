#!/usr/bin/env python3
"""
Migration script to update prediction data format from old keys to new keys.
Changes:
- win_prob -> win_probability
- place_prob -> place_probability  
- show_prob -> show_probability
"""

import json
import os
from datetime import datetime

def migrate_predictions_file(file_path):
    """Migrate predictions file from old format to new format"""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return False
    
    # Create backup
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Creating backup: {backup_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Create backup
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Track changes
        total_predictions = len(data)
        updated_predictions = 0
        
        # Migrate each prediction
        for prediction in data:
            if 'predictions' in prediction and prediction['predictions']:
                prediction_updated = False
                
                for horse_id, horse_data in prediction['predictions'].items():
                    # Update key names if they exist
                    if 'win_prob' in horse_data:
                        horse_data['win_probability'] = horse_data.pop('win_prob')
                        prediction_updated = True
                    
                    if 'place_prob' in horse_data:
                        horse_data['place_probability'] = horse_data.pop('place_prob')
                        prediction_updated = True
                    
                    if 'show_prob' in horse_data:
                        horse_data['show_probability'] = horse_data.pop('show_prob')
                        prediction_updated = True
                
                if prediction_updated:
                    updated_predictions += 1
        
        # Write updated data
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Migration completed:")
        print(f"  Total predictions: {total_predictions}")
        print(f"  Updated predictions: {updated_predictions}")
        print(f"  Backup created: {backup_path}")
        
        return True
        
    except Exception as e:
        print(f"Error migrating {file_path}: {e}")
        return False

def main():
    """Main migration function"""
    print("Starting prediction format migration...")
    
    # Files to migrate
    files_to_migrate = [
        'data/predictions.json',
        'static-html/data/predictions.json'
    ]
    
    success_count = 0
    
    for file_path in files_to_migrate:
        print(f"\nMigrating {file_path}...")
        if migrate_predictions_file(file_path):
            success_count += 1
        else:
            print(f"Failed to migrate {file_path}")
    
    print(f"\nMigration summary: {success_count}/{len(files_to_migrate)} files migrated successfully")

if __name__ == "__main__":
    main()