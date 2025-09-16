#!/usr/bin/env python3
"""
Database Race Verification Script
This script checks what race data exists in the database to troubleshoot the races page.
"""

import sys
import os
from datetime import datetime

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_sqlite_database():
    """Check SQLite database for race data"""
    print("🗄️  CHECKING SQLITE DATABASE")
    print("-" * 40)
    
    try:
        from app import app, db
        from models.race import Race
        from models.horse import Horse
        
        with app.app_context():
            # Check if tables exist
            try:
                race_count = Race.query.count()
                print(f"📊 Total races in database: {race_count}")
                
                if race_count > 0:
                    # Get some sample races
                    races = Race.query.limit(5).all()
                    print(f"\n📋 Sample races:")
                    for race in races:
                        print(f"   • ID: {race.id}, Name: {race.name}, Date: {race.date}")
                        print(f"     Track: {race.track}, Distance: {race.distance}m")
                        
                        # Check horses in this race
                        if hasattr(race, 'horses') and race.horses:
                            print(f"     Horses: {len(race.horses)} entries")
                        else:
                            print(f"     Horses: No entries")
                        print()
                else:
                    print("❌ No races found in database")
                
                # Check horses too
                horse_count = Horse.query.count()
                print(f"🐎 Total horses in database: {horse_count}")
                
                return race_count > 0
                
            except Exception as e:
                print(f"❌ Error querying database: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return False

def check_json_data_files():
    """Check JSON data files for race data"""
    print("\n📁 CHECKING JSON DATA FILES")
    print("-" * 40)
    
    import json
    
    data_files = [
        'data/races.json',
        'data/races_enhanced.json',
        'data/sample_races.json'
    ]
    
    total_races = 0
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    count = len(data)
                    total_races += count
                    print(f"✅ {file_path}: {count} races")
                    
                    if count > 0 and len(data) <= 3:
                        # Show sample data for small files
                        for i, race in enumerate(data[:2]):
                            print(f"   Sample {i+1}: {race.get('name', 'Unknown')} - {race.get('track', 'Unknown track')}")
                            
                elif isinstance(data, dict) and 'races' in data:
                    count = len(data['races'])
                    total_races += count
                    print(f"✅ {file_path}: {count} races (nested)")
                else:
                    print(f"⚠️  {file_path}: Unknown format")
                    
            except Exception as e:
                print(f"❌ {file_path}: Error reading - {e}")
        else:
            print(f"❌ {file_path}: File not found")
    
    print(f"\n📊 Total races in JSON files: {total_races}")
    return total_races > 0

def check_races_route():
    """Check if the races route is working"""
    print("\n🌐 CHECKING RACES ROUTE")
    print("-" * 40)
    
    try:
        from app import app
        
        with app.test_client() as client:
            # Test the races page
            response = client.get('/races')
            print(f"📄 /races route status: {response.status_code}")
            
            if response.status_code == 200:
                content = response.get_data(as_text=True)
                
                # Check for common indicators
                if 'No races found' in content or 'no races' in content.lower():
                    print("⚠️  Page shows 'No races found' message")
                elif '<table' in content or 'race-card' in content:
                    print("✅ Page contains race display elements")
                else:
                    print("❓ Page loaded but content unclear")
                
                # Check for errors
                if 'error' in content.lower() or 'exception' in content.lower():
                    print("❌ Page contains error messages")
                
                return response.status_code == 200
            else:
                print(f"❌ Route failed with status {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing races route: {e}")
        return False

def check_template_file():
    """Check if the races template exists"""
    print("\n📄 CHECKING RACES TEMPLATE")
    print("-" * 40)
    
    template_path = 'templates/races.html'
    
    if os.path.exists(template_path):
        print(f"✅ Template found: {template_path}")
        
        try:
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Check for key elements
            checks = [
                ('{% for race in races %}', 'Race loop'),
                ('{{ race.name }}', 'Race name display'),
                ('{{ race.track }}', 'Race track display'),
                ('No races found', 'Empty state message')
            ]
            
            for pattern, description in checks:
                if pattern in content:
                    print(f"   ✅ {description}: Found")
                else:
                    print(f"   ❌ {description}: Missing")
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading template: {e}")
            return False
    else:
        print(f"❌ Template not found: {template_path}")
        return False

def suggest_fixes():
    """Suggest potential fixes based on findings"""
    print("\n🔧 SUGGESTED FIXES")
    print("=" * 50)
    
    print("\n1. 📊 IF NO DATA IN DATABASE:")
    print("   • Run: python3 import_sample_data.py")
    print("   • Or: python3 generate_sample_data.py")
    print("   • Check data/ folder for JSON files to import")
    
    print("\n2. 🌐 IF ROUTE ISSUES:")
    print("   • Check app.py for /races route definition")
    print("   • Verify template path in route handler")
    print("   • Check for any route errors in terminal")
    
    print("\n3. 📄 IF TEMPLATE ISSUES:")
    print("   • Verify templates/races.html exists")
    print("   • Check template syntax for race display")
    print("   • Ensure proper variable names (races vs race_list)")
    
    print("\n4. 🔄 IF DATA EXISTS BUT NOT SHOWING:")
    print("   • Check database connection in route")
    print("   • Verify query logic in races route")
    print("   • Check for JavaScript errors in browser console")

def main():
    """Run comprehensive database and races page check"""
    print("🔍 RACE DATABASE VERIFICATION")
    print("=" * 50)
    
    # Run all checks
    db_has_races = check_sqlite_database()
    json_has_races = check_json_data_files()
    route_works = check_races_route()
    template_exists = check_template_file()
    
    # Summary
    print(f"\n{'='*50}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*50}")
    
    print(f"Database has races: {'✅' if db_has_races else '❌'}")
    print(f"JSON files have races: {'✅' if json_has_races else '❌'}")
    print(f"Races route works: {'✅' if route_works else '❌'}")
    print(f"Template exists: {'✅' if template_exists else '❌'}")
    
    # Diagnosis
    if not db_has_races and not json_has_races:
        print("\n🚨 DIAGNOSIS: NO RACE DATA FOUND")
        print("The database and JSON files are empty. You need to import or generate race data.")
    elif not db_has_races and json_has_races:
        print("\n🚨 DIAGNOSIS: DATA NOT IMPORTED")
        print("JSON files contain race data but it hasn't been imported to the database.")
    elif db_has_races and not route_works:
        print("\n🚨 DIAGNOSIS: ROUTE OR TEMPLATE ISSUE")
        print("Database has races but the web page isn't displaying them correctly.")
    elif db_has_races and route_works:
        print("\n✅ DIAGNOSIS: SYSTEM SHOULD BE WORKING")
        print("Database has races and route works. Check browser for display issues.")
    
    # Show suggestions
    suggest_fixes()
    
    return db_has_races and route_works

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)