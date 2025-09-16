#!/usr/bin/env python3
"""
Fix Password Hashes Script
Updates existing scrypt-based password hashes to pbkdf2:sha256 in the SQLite database.
"""

import sys
import os
from werkzeug.security import generate_password_hash

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database_config import init_database, db
from models.sqlalchemy_models import User
from flask import Flask

def fix_password_hashes():
    """Fix all scrypt-based password hashes in the database."""
    
    # Create Flask app context
    app = Flask(__name__)
    init_database(app)
    
    with app.app_context():
        print("🔍 Checking for users with scrypt-based password hashes...")
        
        # Find all users with scrypt-based password hashes
        users_with_scrypt = User.query.filter(User.password_hash.like('scrypt:%')).all()
        
        if not users_with_scrypt:
            print("✅ No scrypt-based password hashes found!")
            return
        
        print(f"📝 Found {len(users_with_scrypt)} users with scrypt-based password hashes:")
        
        for user in users_with_scrypt:
            print(f"   - {user.username} (ID: {user.id})")
        
        # Reset admin password specifically
        admin_user = User.query.filter_by(username='admin').first()
        if admin_user and admin_user.password_hash.startswith('scrypt:'):
            print("\n🔧 Updating admin user password hash...")
            
            # Generate new password hash with pbkdf2:sha256
            new_password_hash = generate_password_hash('admin123', method='pbkdf2:sha256')
            admin_user.password_hash = new_password_hash
            
            print(f"   Old hash: {admin_user.password_hash[:50]}...")
            print(f"   New hash: {new_password_hash[:50]}...")
        
        # Update any other users (if they exist)
        for user in users_with_scrypt:
            if user.username != 'admin':
                print(f"\n🔧 Updating {user.username} password hash...")
                # For other users, we'll need to reset to a default password
                # In production, you'd want to force them to reset their password
                default_password = f"{user.username}123"  # Simple default
                new_password_hash = generate_password_hash(default_password, method='pbkdf2:sha256')
                user.password_hash = new_password_hash
                print(f"   ⚠️  Password reset to: {default_password}")
        
        # Commit changes
        try:
            db.session.commit()
            print("\n✅ Successfully updated all password hashes!")
            print("🔐 All users now use pbkdf2:sha256 password hashing.")
            
            # Verify the changes
            print("\n🔍 Verifying changes...")
            remaining_scrypt = User.query.filter(User.password_hash.like('scrypt:%')).count()
            if remaining_scrypt == 0:
                print("✅ Verification successful: No scrypt hashes remaining!")
            else:
                print(f"❌ Warning: {remaining_scrypt} scrypt hashes still found!")
                
        except Exception as e:
            print(f"❌ Error updating password hashes: {e}")
            db.session.rollback()
            return False
    
    return True

if __name__ == '__main__':
    print("🚀 Starting password hash migration...")
    success = fix_password_hashes()
    
    if success:
        print("\n🎉 Password hash migration completed successfully!")
        print("📋 Summary:")
        print("   - Admin password: admin123 (pbkdf2:sha256)")
        print("   - All password hashes now use secure pbkdf2:sha256 method")
        print("   - No more scrypt compatibility issues!")
    else:
        print("\n❌ Password hash migration failed!")
        sys.exit(1)