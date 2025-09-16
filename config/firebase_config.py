import os
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
from cryptography.fernet import Fernet

class FirebaseConfig:
    """Firebase configuration and database service"""
    
    def __init__(self):
        self.db = None
        self.encryption_key = None
        self._initialize_firebase()
        self._initialize_encryption()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
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
                    firebase_admin.initialize_app(cred)
                    print("Firebase initialized with environment variables (secure mode)")
                
                # Priority 2: Try service account file (for local development only)
                elif os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH'):
                    service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
                    if os.path.exists(service_account_path):
                        print("WARNING: Using service account file. This should only be used for local development!")
                        print("For production, use environment variables: FIREBASE_PROJECT_ID, FIREBASE_PRIVATE_KEY, FIREBASE_CLIENT_EMAIL")
                        cred = credentials.Certificate(service_account_path)
                        firebase_admin.initialize_app(cred)
                    else:
                        print(f"Warning: Firebase service account file not found: {service_account_path}")
                        return
                
                # Priority 3: Try default credentials (Google Cloud environment)
                else:
                    try:
                        cred = credentials.ApplicationDefault()
                        firebase_admin.initialize_app(cred)
                        print("Firebase initialized with default credentials (Google Cloud environment)")
                    except Exception as e:
                        print(f"Warning: Firebase not initialized - {e}")
                        print("Please set up Firebase credentials:")
                        print("1. For production: Set FIREBASE_PROJECT_ID, FIREBASE_PRIVATE_KEY, FIREBASE_CLIENT_EMAIL")
                        print("2. For local development: Set FIREBASE_SERVICE_ACCOUNT_PATH")
                        print("3. For Google Cloud: Use default credentials")
                        return
            
            # Get Firestore client
            self.db = firestore.client()
            print("Firebase initialized successfully")
            
        except Exception as e:
            print(f"Error initializing Firebase: {e}")
            self.db = None
    
    def _initialize_encryption(self):
        """Initialize encryption for sensitive data"""
        encryption_key = os.getenv('ENCRYPTION_KEY')
        if not encryption_key:
            # Generate a new key for development (in production, use a secure key management system)
            encryption_key = Fernet.generate_key().decode()
            print(f"Generated new encryption key: {encryption_key}")
            print("Please set ENCRYPTION_KEY environment variable with this key")
        
        self.encryption_key = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not data:
            return data
        return self.encryption_key.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not encrypted_data:
            return encrypted_data
        try:
            return self.encryption_key.decrypt(encrypted_data.encode()).decode()
        except Exception:
            return encrypted_data  # Return as-is if decryption fails
    
    def is_connected(self) -> bool:
        """Check if Firebase is properly connected"""
        return self.db is not None
    
    # API Credentials methods
    def get_api_credentials(self) -> List[Dict[str, Any]]:
        """Get all API credentials"""
        if not self.db:
            return []
        
        try:
            docs = self.db.collection('api_credentials').stream()
            credentials = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                # Decrypt sensitive fields
                if 'api_key' in data and data['api_key']:
                    data['api_key'] = self.decrypt_data(data['api_key'])
                if 'api_secret' in data and data['api_secret']:
                    data['api_secret'] = self.decrypt_data(data['api_secret'])
                credentials.append(data)
            return credentials
        except Exception as e:
            print(f"Error getting API credentials: {e}")
            return []
    
    def get_api_credential(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific API credential"""
        if not self.db:
            return None
        
        try:
            doc = self.db.collection('api_credentials').document(credential_id).get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                # Decrypt sensitive fields
                if 'api_key' in data and data['api_key']:
                    data['api_key'] = self.decrypt_data(data['api_key'])
                if 'api_secret' in data and data['api_secret']:
                    data['api_secret'] = self.decrypt_data(data['api_secret'])
                return data
            return None
        except Exception as e:
            print(f"Error getting API credential: {e}")
            return None
    
    def save_api_credential(self, data: Dict[str, Any]) -> str:
        """Save API credential to Firebase"""
        if not self.db:
            raise Exception("Firebase not connected")
        
        try:
            # Encrypt sensitive fields
            if 'api_key' in data and data['api_key']:
                data['api_key'] = self.encrypt_data(data['api_key'])
            if 'api_secret' in data and data['api_secret']:
                data['api_secret'] = self.encrypt_data(data['api_secret'])
            
            # Add timestamp
            data['created_at'] = datetime.utcnow()
            data['updated_at'] = datetime.utcnow()
            
            # Save to Firestore
            doc_ref = self.db.collection('api_credentials').add(data)
            return doc_ref[1].id
        except Exception as e:
            print(f"Error saving API credential: {e}")
            raise
    
    def update_api_credential(self, credential_id: str, data: Dict[str, Any]) -> bool:
        """Update API credential in Firebase"""
        if not self.db:
            return False
        
        try:
            # Encrypt sensitive fields
            if 'api_key' in data and data['api_key']:
                data['api_key'] = self.encrypt_data(data['api_key'])
            if 'api_secret' in data and data['api_secret']:
                data['api_secret'] = self.encrypt_data(data['api_secret'])
            
            # Add timestamp
            data['updated_at'] = datetime.utcnow()
            
            # Update in Firestore
            self.db.collection('api_credentials').document(credential_id).update(data)
            return True
        except Exception as e:
            print(f"Error updating API credential: {e}")
            return False
    
    def delete_api_credential(self, credential_id: str) -> bool:
        """Delete API credential from Firebase"""
        if not self.db:
            return False
        
        try:
            self.db.collection('api_credentials').document(credential_id).delete()
            return True
        except Exception as e:
            print(f"Error deleting API credential: {e}")
            return False
    
    # Race data methods
    def get_races(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get races from Firebase"""
        if not self.db:
            return []
        
        try:
            docs = self.db.collection('races').limit(limit).stream()
            races = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                races.append(data)
            return races
        except Exception as e:
            print(f"Error getting races: {e}")
            return []
    
    def get_race(self, race_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific race"""
        if not self.db:
            return None
        
        try:
            doc = self.db.collection('races').document(race_id).get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            return None
        except Exception as e:
            print(f"Error getting race: {e}")
            return None
    
    def save_race(self, data: Dict[str, Any]) -> str:
        """Save race to Firebase"""
        if not self.db:
            raise Exception("Firebase not connected")
        
        try:
            data['created_at'] = datetime.utcnow()
            data['updated_at'] = datetime.utcnow()
            doc_ref = self.db.collection('races').add(data)
            return doc_ref[1].id
        except Exception as e:
            print(f"Error saving race: {e}")
            raise
    
    def update_race(self, race_id: str, data: Dict[str, Any]) -> bool:
        """Update race in Firebase"""
        if not self.db:
            return False
        
        try:
            data['updated_at'] = datetime.utcnow()
            self.db.collection('races').document(race_id).update(data)
            return True
        except Exception as e:
            print(f"Error updating race: {e}")
            return False
    
    def delete_race(self, race_id: str) -> bool:
        """Delete race from Firebase"""
        if not self.db:
            return False
        
        try:
            self.db.collection('races').document(race_id).delete()
            return True
        except Exception as e:
            print(f"Error deleting race: {e}")
            return False
    
    # Horse data methods
    def get_horses(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get horses from Firebase"""
        if not self.db:
            return []
        
        try:
            docs = self.db.collection('horses').limit(limit).stream()
            horses = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                horses.append(data)
            return horses
        except Exception as e:
            print(f"Error getting horses: {e}")
            return []
    
    def save_horse(self, data: Dict[str, Any]) -> str:
        """Save horse to Firebase"""
        if not self.db:
            raise Exception("Firebase not connected")
        
        try:
            data['created_at'] = datetime.utcnow()
            data['updated_at'] = datetime.utcnow()
            doc_ref = self.db.collection('horses').add(data)
            return doc_ref[1].id
        except Exception as e:
            print(f"Error saving horse: {e}")
            raise
    
    # Prediction data methods
    def save_prediction(self, data: Dict[str, Any]) -> str:
        """Save prediction to Firebase"""
        if not self.db:
            raise Exception("Firebase not connected")
        
        try:
            data['created_at'] = datetime.utcnow()
            doc_ref = self.db.collection('predictions').add(data)
            return doc_ref[1].id
        except Exception as e:
            print(f"Error saving prediction: {e}")
            raise
    
    def get_predictions(self, race_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get predictions from Firebase"""
        if not self.db:
            return []
        
        try:
            query = self.db.collection('predictions')
            if race_id:
                query = query.where('race_id', '==', race_id)
            
            docs = query.limit(limit).stream()
            predictions = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                predictions.append(data)
            return predictions
        except Exception as e:
            print(f"Error getting predictions: {e}")
            return []

# Global Firebase instance
firebase_config = FirebaseConfig()