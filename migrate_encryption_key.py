#!/usr/bin/env python3
"""
Migration script to move encryption key from file to environment variable.
This script helps secure the encryption key by moving it from a plain text file
to an environment variable.
"""

import os
import sys
from pathlib import Path

def migrate_encryption_key():
    """Migrate encryption key from file to environment variable"""
    
    # Path to the encryption key file
    key_file = Path(__file__).parent / 'data' / '.encryption_key'
    
    print("🔐 Encryption Key Migration Tool")
    print("=" * 40)
    
    # Check if key file exists
    if not key_file.exists():
        print("❌ No encryption key file found at:", key_file)
        print("✅ This is good - it means you're already using environment variables!")
        return
    
    # Read the existing key
    try:
        with open(key_file, 'rb') as f:
            encryption_key = f.read().decode().strip()
        
        print(f"✅ Found encryption key in file: {key_file}")
        print(f"🔑 Your encryption key is: {encryption_key}")
        print()
        print("📋 To complete the migration:")
        print("1. Set the ENCRYPTION_KEY environment variable:")
        print(f"   Windows (PowerShell): $env:ENCRYPTION_KEY='{encryption_key}'")
        print(f"   Windows (CMD): set ENCRYPTION_KEY={encryption_key}")
        print(f"   Linux/Mac: export ENCRYPTION_KEY='{encryption_key}'")
        print()
        print("2. Add it to your .env file:")
        print(f"   ENCRYPTION_KEY={encryption_key}")
        print()
        print("3. After setting the environment variable, you can safely delete the key file:")
        print(f"   rm '{key_file}' (Linux/Mac)")
        print(f"   del '{key_file}' (Windows)")
        print()
        print("⚠️  IMPORTANT: Make sure to set the environment variable before deleting the file!")
        print("⚠️  Keep this key secure - losing it will make encrypted data unrecoverable!")
        
        # Ask if user wants to delete the file
        response = input("\n🗑️  Delete the key file now? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            # Check if environment variable is set
            if os.getenv('ENCRYPTION_KEY') == encryption_key:
                os.remove(key_file)
                print("✅ Key file deleted successfully!")
                print("✅ Migration complete!")
            else:
                print("❌ Environment variable not set correctly. Please set it first.")
                print(f"   Current value: {os.getenv('ENCRYPTION_KEY', 'Not set')}")
                print(f"   Expected value: {encryption_key}")
        else:
            print("✅ Key file preserved. Remember to delete it after setting the environment variable.")
            
    except Exception as e:
        print(f"❌ Error reading key file: {e}")
        return

if __name__ == "__main__":
    migrate_encryption_key()