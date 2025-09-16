#!/usr/bin/env python3
"""
Script to fix the admin user password hash compatibility issue
"""

import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.firebase_init import initialize_firebase, get_firebase_db
from models.firebase_models import User

def fix_admin_password():
    """Fix the admin user password hash"""
    
    print("Fixing admin user password hash...")
    print("=" * 50)
    
    # Initialize Firebase
    try:
        initialize_firebase()
        print("✓ Firebase initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Firebase: {e}")
        return False
    
    # Admin user credentials
    admin_username = os.getenv('ADMIN_USERNAME', 'admin')
    admin_email = os.getenv('ADMIN_EMAIL', 'admin@hrp.com')
    admin_password = os.getenv('ADMIN_PASSWORD')
    
    if not admin_password:
        print("❌ Error: ADMIN_PASSWORD environment variable is required!")
        print("Please set: ADMIN_PASSWORD=your_secure_password")
        return False
    
    try:
        # Get Firebase database
        db = get_firebase_db()
        if not db:
            print("✗ Failed to get Firebase database")
            return False
        
        # Delete existing admin user if it exists
        print("Checking for existing admin user...")
        existing_user = User.get_by_username(admin_username)
        if existing_user:
            print(f"✓ Found existing admin user: {existing_user.id}")
            
            # Delete the existing user document
            try:
                db.collection("users").document(existing_user.id).delete()
                print("✓ Deleted existing admin user")
            except Exception as e:
                print(f"✗ Failed to delete existing user: {e}")
                return False
        else:
            print("No existing admin user found")
        
        # Create new admin user with correct password hash
        print("Creating new admin user with correct password hash...")
        from werkzeug.security import generate_password_hash
        
        user_data = {
            'username': admin_username,
            'email': admin_email,
            'password_hash': generate_password_hash(admin_password),
            'is_admin': True,
            'is_active': True,
            'created_at': datetime.now().isoformat()
        }
        
        # Create and save the user
        admin_user = User(user_data)
        user_id = admin_user.save()
        
        if user_id:
            print("✓ Admin user created successfully!")
            print(f"  Username: {admin_username}")
            print(f"  Email: {admin_email}")
            print(f"  User ID: {user_id}")
            print(f"  Is Admin: True")
            print(f"  Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("  Password: [HIDDEN FOR SECURITY]")
            return True
        else:
            print("✗ Failed to save admin user")
            return False
            
    except Exception as e:
        print(f"✗ Error fixing admin user: {e}")
        return False

def main():
    """Main function"""
    print("Admin Password Hash Fix Tool")
    print("=" * 50)
    
    success = fix_admin_password()
    
    if success:
        print("\n🎉 Admin user password hash fixed successfully!")
        print("\nYou can now log in with:")
        print("  Username: admin")
        print("  Password: HRP_Admin_2024!")
    else:
        print("\n❌ Failed to fix admin user password hash")
        sys.exit(1)

if __name__ == "__main__":
    main()