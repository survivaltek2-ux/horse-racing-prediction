#!/usr/bin/env python3
"""
Script to create an admin user in Firebase
"""

import sys
import os
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.firebase_init import initialize_firebase
from models.firebase_models import User

def create_admin_user():
    """Create an admin user in Firebase"""
    
    print("Creating admin user in Firebase...")
    print("=" * 50)
    
    # Initialize Firebase
    try:
        initialize_firebase()
        print("✓ Firebase initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Firebase: {e}")
        return False
    
    # Admin user credentials
    admin_username = "admin"
    admin_email = "admin@hrp.com"
    admin_password = "HRP_Admin_2024!"
    
    try:
        # Use the create_admin_user class method
        admin_user = User.create_admin_user(admin_username, admin_email, admin_password)
        
        if admin_user:
            print("✓ Admin user created successfully!")
            print(f"  Username: {admin_username}")
            print(f"  Email: {admin_email}")
            print(f"  Password: {admin_password}")
            print(f"  Is Admin: {admin_user.is_admin}")
            print(f"  Created: {admin_user.created_at}")
            return True
        else:
            print("✗ Failed to create admin user")
            return False
            
    except Exception as e:
        print(f"✗ Error creating admin user: {e}")
        return False

def main():
    """Main function"""
    print("Firebase Admin User Creator")
    print("=" * 50)
    
    success = create_admin_user()
    
    if success:
        print("\n🎉 Admin user setup completed successfully!")
        print("\nYou can now log in with:")
        print("  Username: admin")
        print("  Password: HRP_Admin_2024!")
    else:
        print("\n❌ Failed to create admin user")
        sys.exit(1)

if __name__ == "__main__":
    main()