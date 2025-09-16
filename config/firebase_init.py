"""
Firebase Initialization Module
Sets up Firebase connection and configuration for the application
"""

import os
import firebase_admin
from firebase_admin import credentials, firestore

# Global Firebase app instance
firebase_app = None
db = None

def initialize_firebase():
    """Initialize Firebase with service account credentials"""
    global firebase_app, db
    
    if firebase_app is not None:
        return firebase_app, db
    
    try:
        # Get service account path from environment
        service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
        
        if not service_account_path:
            print("Warning: FIREBASE_SERVICE_ACCOUNT_PATH not set. Firebase features will be disabled.")
            print("To enable Firebase, set FIREBASE_SERVICE_ACCOUNT_PATH in your .env file")
            return None, None
        
        if not os.path.exists(service_account_path):
            print(f"Warning: Firebase service account file not found: {service_account_path}")
            print("Firebase features will be disabled.")
            return None, None
        
        # Initialize Firebase Admin SDK
        cred = credentials.Certificate(service_account_path)
        firebase_app = firebase_admin.initialize_app(cred)
        
        # Initialize Firestore
        db = firestore.client()
        
        print("Firebase initialized successfully")
        return firebase_app, db
        
    except Exception as e:
        print(f"Warning: Firebase initialization failed: {e}")
        print("Firebase features will be disabled. The app will run in local mode.")
        return None, None

def get_firebase_db():
    """Get Firestore database instance"""
    global db
    if db is None:
        initialize_firebase()
    return db

# Initialize Firebase when module is imported
try:
    initialize_firebase()
except Exception as e:
    print(f"Warning: Firebase initialization failed during import: {e}")
    print("Firebase features will be disabled.")