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
        # Priority 1: Try environment variables for service account (most secure)
        firebase_project_id = os.getenv('FIREBASE_PROJECT_ID')
        firebase_private_key = os.getenv('FIREBASE_PRIVATE_KEY')
        firebase_client_email = os.getenv('FIREBASE_CLIENT_EMAIL')
        
        if firebase_project_id and firebase_private_key and firebase_client_email:
            # Use environment variables (recommended for production)
            service_account_info = {
                "type": "service_account",
                "project_id": firebase_project_id,
                "private_key": firebase_private_key.replace('\\n', '\n'),
                "client_email": firebase_client_email,
                "token_uri": "https://oauth2.googleapis.com/token",
            }
            cred = credentials.Certificate(service_account_info)
            firebase_app = firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("Firebase initialized with environment variables (secure mode)")
            return firebase_app, db
        
        # Priority 2: Try service account file (for local development only)
        service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
        if service_account_path and os.path.exists(service_account_path):
            print("WARNING: Using service account file. This should only be used for local development!")
            print("For production, use environment variables: FIREBASE_PROJECT_ID, FIREBASE_PRIVATE_KEY, FIREBASE_CLIENT_EMAIL")
            cred = credentials.Certificate(service_account_path)
            firebase_app = firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("Firebase initialized successfully")
            return firebase_app, db
        
        # Priority 3: Try default credentials (Google Cloud environment)
        try:
            cred = credentials.ApplicationDefault()
            firebase_app = firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("Firebase initialized with default credentials (Google Cloud environment)")
            return firebase_app, db
        except Exception as default_cred_error:
            print(f"Warning: Firebase initialization failed: {default_cred_error}")
            print("Firebase features will be disabled. Please set up Firebase credentials:")
            print("1. For production: Set FIREBASE_PROJECT_ID, FIREBASE_PRIVATE_KEY, FIREBASE_CLIENT_EMAIL")
            print("2. For local development: Set FIREBASE_SERVICE_ACCOUNT_PATH")
            print("3. For Google Cloud: Use default credentials")
            return None, None
        
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