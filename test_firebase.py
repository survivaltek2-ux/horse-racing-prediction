#!/usr/bin/env python3
"""
Firebase Connection Test Script
Tests Firebase connectivity and basic operations
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.firebase_init import initialize_firebase, get_firebase_db
from config.firebase_config import FirebaseConfig

def test_firebase_connection():
    """Test Firebase connection and basic operations"""
    print("Firebase Connection Test")
    print("=" * 50)
    
    # Test 1: Check environment variables
    print("\n1. Checking environment variables...")
    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
    if service_account_path:
        print(f"✓ FIREBASE_SERVICE_ACCOUNT_PATH: {service_account_path}")
        if os.path.exists(service_account_path):
            print(f"✓ Service account file exists")
        else:
            print(f"✗ Service account file not found")
            return False
    else:
        print("✗ FIREBASE_SERVICE_ACCOUNT_PATH not set")
        return False
    
    # Test 2: Initialize Firebase
    print("\n2. Initializing Firebase...")
    try:
        app, db = initialize_firebase()
        if app and db:
            print("✓ Firebase initialized successfully")
        else:
            print("✗ Firebase initialization failed")
            return False
    except Exception as e:
        print(f"✗ Firebase initialization error: {e}")
        return False
    
    # Test 3: Test Firestore connection
    print("\n3. Testing Firestore connection...")
    try:
        # Try to access a collection (this will fail if API is disabled)
        collections = db.collections()
        collection_list = list(collections)
        print(f"✓ Firestore connection successful")
        print(f"✓ Found {len(collection_list)} collections")
        for col in collection_list:
            print(f"  - {col.id}")
    except Exception as e:
        print(f"✗ Firestore connection failed: {e}")
        if "SERVICE_DISABLED" in str(e):
            print("  → Cloud Firestore API is disabled")
            print("  → Enable it at: https://console.developers.google.com/apis/api/firestore.googleapis.com/overview")
        return False
    
    # Test 4: Test basic operations
    print("\n4. Testing basic operations...")
    try:
        # Try to create a test document
        test_ref = db.collection('test').document('connection_test')
        test_ref.set({
            'test': True,
            'timestamp': 'test_connection',
            'message': 'Firebase connection test successful'
        })
        print("✓ Write operation successful")
        
        # Try to read the document
        doc = test_ref.get()
        if doc.exists:
            print("✓ Read operation successful")
            print(f"  Data: {doc.to_dict()}")
        else:
            print("✗ Read operation failed - document not found")
        
        # Clean up test document
        test_ref.delete()
        print("✓ Delete operation successful")
        
    except Exception as e:
        print(f"✗ Basic operations failed: {e}")
        return False
    
    # Test 5: Test Firebase Config class
    print("\n5. Testing Firebase Config class...")
    try:
        firebase_config = FirebaseConfig()
        if firebase_config.is_connected():
            print("✓ Firebase Config class connected successfully")
        else:
            print("✗ Firebase Config class connection failed")
            return False
    except Exception as e:
        print(f"✗ Firebase Config class error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All Firebase tests passed successfully!")
    print("Firebase is properly configured and working.")
    return True

if __name__ == '__main__':
    success = test_firebase_connection()
    sys.exit(0 if success else 1)