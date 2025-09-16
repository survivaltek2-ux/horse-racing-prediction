from typing import Dict, Any, List, Optional
from datetime import datetime
from flask_login import UserMixin
from config.firebase_init import get_firebase_db

class FirebaseModel:
    """Base class for Firebase models"""
    
    def __init__(self, data: dict = None):
        self.db = get_firebase_db()
        self.collection_name = None
        
        # Handle case where Firebase is not initialized
        if self.db is None:
            print("Warning: Firebase not initialized. Some features may not work.")
        if data:
            for key, value in data.items():
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        data = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_') and k not in ['db', 'collection_name']:
                data[k] = v
        return data
    
    def save(self) -> str:
        """Save model to Firestore"""
        if not self.db:
            print("Warning: Firebase not initialized. Cannot save data.")
            return None
            
        if not self.collection_name:
            raise ValueError("collection_name must be set in the model")
        
        data = self.to_dict()
        
        # Remove id from data if it exists
        if 'id' in data:
            doc_id = data.pop('id')
        else:
            doc_id = None
        
        # Add timestamps
        if not hasattr(self, 'created_at') or not self.created_at:
            data['created_at'] = datetime.now().isoformat()
        data['updated_at'] = datetime.now().isoformat()
        
        try:
            if doc_id and hasattr(self, 'id') and self.id:
                # Update existing document
                doc_ref = self.db.collection(self.collection_name).document(doc_id)
                doc_ref.update(data)
                return doc_id
            else:
                # Create new document
                doc_ref = self.db.collection(self.collection_name).add(data)[1]
                self.id = doc_ref.id
                return doc_ref.id
        except Exception as e:
            print(f"Error saving to Firestore: {e}")
            return None
    
    def delete(self) -> bool:
        """Delete model from Firestore"""
        if not self.db:
            print("Warning: Firebase not initialized. Cannot delete data.")
            return False
            
        if not self.collection_name:
            raise ValueError("collection_name must be set in the model")
        
        if not hasattr(self, 'id') or not self.id:
            return False
        
        try:
            self.db.collection(self.collection_name).document(self.id).delete()
            return True
        except Exception as e:
            print(f"Error deleting from Firestore: {e}")
            return False

class APICredentials(FirebaseModel):
    """Firebase model for API credentials"""
    
    def __init__(self, data: Dict[str, Any] = None):
        self.id = None
        self.provider = ""
        self.base_url = ""
        self.api_key = ""
        self.api_secret = ""
        self.description = ""
        self.is_active = True
        self.created_at = None
        self.updated_at = None
        
        super().__init__(data)
        self.collection_name = "races"
        self.collection_name = "api_credentials"
    
    @classmethod
    def get_all(cls) -> List['APICredentials']:
        """Get all API credentials"""
        try:
            db = get_firebase_db()
            if not db:
                return []
            
            docs = db.collection("api_credentials").stream()
            credentials = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                credentials.append(cls(data))
            return credentials
        except Exception as e:
            print(f"Error getting API credentials: {e}")
            return []
    
    @classmethod
    def get_by_id(cls, credential_id: str) -> Optional['APICredentials']:
        """Get API credential by ID"""
        try:
            db = get_firebase_db()
            if not db:
                return None
            
            doc = db.collection("api_credentials").document(credential_id).get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return cls(data)
            return None
        except Exception as e:
            print(f"Error getting API credential by ID: {e}")
            return None
    
    @classmethod
    def get_by_provider(cls, provider: str) -> Optional['APICredentials']:
        """Get active API credential by provider"""
        try:
            db = get_firebase_db()
            if not db:
                return None
            
            docs = db.collection("api_credentials").where("provider", "==", provider).where("is_active", "==", True).limit(1).stream()
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                return cls(data)
            return None
        except Exception as e:
            print(f"Error getting API credential by provider: {e}")
            return None
    

    
    def test_connection(self) -> Dict[str, Any]:
        """Test API connection"""
        # This would integrate with your existing API testing logic
        # For now, return a basic response
        return {
            'success': True,
            'message': f'Connection test for {self.provider}',
            'provider': self.provider,
            'base_url': self.base_url
        }

class Race(FirebaseModel):
    """Firebase model for races"""
    
    def __init__(self, data: Dict[str, Any] = None):
        self.id = None
        self.name = ""
        self.date = None
        self.time = ""
        self.track = ""
        self.distance = 0.0
        self.surface = ""
        self.race_class = ""
        self.prize_money = 0.0
        self.weather = ""
        self.track_condition = ""
        self.horses = []  # List of horse IDs
        self.results = {}  # Race results
        self.created_at = None
        self.updated_at = None
        
        super().__init__(data)
        self.collection_name = "races"
    
    @classmethod
    def get_all(cls, limit: int = 100) -> List['Race']:
        """Get all races"""
        try:
            db = get_firebase_db()
            if not db:
                return []
            
            docs = db.collection("races").limit(limit).stream()
            races = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                races.append(cls(data))
            return races
        except Exception as e:
            print(f"Error getting races: {e}")
            return []
    
    @classmethod
    def get_by_id(cls, race_id: str) -> Optional['Race']:
        """Get race by ID"""
        try:
            db = get_firebase_db()
            if not db:
                return None
            
            doc = db.collection("races").document(race_id).get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return cls(data)
            return None
        except Exception as e:
            print(f"Error getting race by ID: {e}")
            return None
    
    @classmethod
    def get_upcoming(cls, limit: int = 50) -> List['Race']:
        """Get upcoming races"""
        try:
            db = get_firebase_db()
            if not db:
                return []
            
            from datetime import datetime
            current_time = datetime.now().isoformat()
            
            docs = db.collection("races").where("date", ">", current_time).order_by("date").limit(limit).stream()
            races = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                races.append(cls(data))
            return races
        except Exception as e:
            print(f"Error getting upcoming races: {e}")
            return []
    

    
    def add_horse(self, horse_id: str):
        """Add horse to race"""
        if horse_id not in self.horses:
            self.horses.append(horse_id)
    
    def remove_horse(self, horse_id: str):
        """Remove horse from race"""
        if horse_id in self.horses:
            self.horses.remove(horse_id)

class Horse(FirebaseModel):
    """Firebase model for horses"""
    
    def __init__(self, data: Dict[str, Any] = None):
        self.id = None
        self.name = ""
        self.age = 0
        self.sex = ""
        self.color = ""
        self.sire = ""
        self.dam = ""
        self.trainer = ""
        self.jockey = ""
        self.owner = ""
        self.weight = 0.0
        self.form = ""
        self.rating = 0
        self.last_run = None
        self.wins = 0
        self.places = 0
        self.runs = 0
        self.earnings = 0.0
        self.created_at = None
        self.updated_at = None
        
        super().__init__(data)
        self.collection_name = "horses"
    
    @classmethod
    def get_all(cls, limit: int = 100) -> List['Horse']:
        """Get all horses"""
        try:
            db = get_firebase_db()
            if not db:
                return []
            
            docs = db.collection("horses").limit(limit).stream()
            horses = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                horses.append(cls(data))
            return horses
        except Exception as e:
            print(f"Error getting horses: {e}")
            return []
    
    @classmethod
    def get_by_id(cls, horse_id: str) -> Optional['Horse']:
        """Get horse by ID"""
        try:
            db = get_firebase_db()
            if not db:
                return None
            
            doc = db.collection("horses").document(horse_id).get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return cls(data)
            return None
        except Exception as e:
            print(f"Error getting horse by ID: {e}")
            return None
    
    @classmethod
    def search_by_name(cls, name: str) -> List['Horse']:
        """Search horses by name"""
        try:
            db = get_firebase_db()
            if not db:
                return []
            
            # Firestore doesn't support case-insensitive search, so we'll get all and filter
            docs = db.collection("horses").stream()
            horses = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                horse = cls(data)
                if name.lower() in horse.name.lower():
                    horses.append(horse)
            return horses
        except Exception as e:
            print(f"Error searching horses by name: {e}")
            return []
    

    
    @property
    def win_percentage(self) -> float:
        """Calculate win percentage"""
        if self.runs == 0:
            return 0.0
        return (self.wins / self.runs) * 100
    
    @property
    def place_percentage(self) -> float:
        """Calculate place percentage"""
        if self.runs == 0:
            return 0.0
        return (self.places / self.runs) * 100

class Prediction(FirebaseModel):
    """Firebase model for predictions"""
    
    def __init__(self, data: Dict[str, Any] = None):
        self.id = None
        self.race_id = ""
        self.horse_id = ""
        self.predicted_position = 0
        self.confidence = 0.0
        self.odds = 0.0
        self.factors = {}  # Dictionary of prediction factors
        self.model_version = ""
        self.created_at = None
        
        super().__init__(data)
        self.collection_name = "predictions"
    
    @classmethod
    def get_by_race(cls, race_id: str) -> List['Prediction']:
        """Get predictions for a race"""
        try:
            db = get_firebase_db()
            if not db:
                return []
            
            docs = db.collection("predictions").where("race_id", "==", race_id).stream()
            predictions = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                predictions.append(cls(data))
            return predictions
        except Exception as e:
            print(f"Error getting predictions by race: {e}")
            return []
    
    @classmethod
    def get_all(cls, limit: int = 100) -> List['Prediction']:
        """Get all predictions"""
        try:
            db = get_firebase_db()
            if not db:
                return []
            
            docs = db.collection("predictions").limit(limit).stream()
            predictions = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                predictions.append(cls(data))
            return predictions
        except Exception as e:
            print(f"Error getting predictions: {e}")
            return []
    


class User(UserMixin, FirebaseModel):
    """Firebase model for users"""
    
    def __init__(self, data: Dict[str, Any] = None):
        self.id = None
        self.username = ""
        self.email = ""
        self.password_hash = ""
        self.is_admin = False
        self._is_active = True  # Use private attribute to avoid conflict with UserMixin
        self.created_at = None
        self.last_login = None
        
        super().__init__(data)
        self.collection_name = "users"
    
    def check_password(self, password: str) -> bool:
        """Check if password is correct"""
        from werkzeug.security import check_password_hash
        return check_password_hash(self.password_hash, password)
    
    def set_password(self, password: str):
        """Set password hash"""
        from werkzeug.security import generate_password_hash
        self.password_hash = generate_password_hash(password)
    
    def get_id(self):
        """Flask-Login required method - return user ID as string"""
        return str(self.id)
    
    @property
    def is_active(self):
        """Override UserMixin is_active property"""
        return self._is_active
    
    @is_active.setter
    def is_active(self, value):
        """Set active status"""
        self._is_active = value
    
    def is_admin_user(self) -> bool:
        """Check if user has admin privileges"""
        return self.is_admin
    
    @classmethod
    def get_by_id(cls, user_id: str) -> Optional['User']:
        """Get user by ID - required for Flask-Login"""
        try:
            db = get_firebase_db()
            if not db:
                return None
            
            doc = db.collection("users").document(user_id).get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return cls(data)
            return None
        except Exception as e:
            print(f"Error getting user by ID: {e}")
            return None
    
    @classmethod
    def get_by_username(cls, username: str) -> Optional['User']:
        """Get user by username"""
        try:
            db = get_firebase_db()
            if not db:
                return None
            
            docs = db.collection("users").where("username", "==", username).limit(1).stream()
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                return cls(data)
            return None
        except Exception as e:
            print(f"Error getting user by username: {e}")
            return None
    
    @classmethod
    def get_by_email(cls, email: str) -> Optional['User']:
        """Get user by email"""
        try:
            db = get_firebase_db()
            if not db:
                return None
            
            docs = db.collection("users").where("email", "==", email).limit(1).stream()
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                return cls(data)
            return None
        except Exception as e:
            print(f"Error getting user by email: {e}")
            return None
    
    @classmethod
    def create_admin_user(cls, username: str, email: str, password: str) -> 'User':
        """Create admin user if it doesn't exist"""
        try:
            # Check if admin user already exists
            existing_user = cls.get_by_username(username)
            if existing_user:
                print(f"Admin user '{username}' already exists")
                return existing_user
            
            # Create new admin user
            from werkzeug.security import generate_password_hash
            
            user_data = {
                'username': username,
                'email': email,
                'password_hash': generate_password_hash(password),
                'is_admin': True,
                'created_at': datetime.now().isoformat()
            }
            
            user = cls(user_data)
            user_id = user.save()
            user.user_id = user_id
            
            print(f"Admin user '{username}' created successfully")
            return user
            
        except Exception as e:
            print(f"Error creating admin user: {e}")
            return None