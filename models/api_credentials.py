"""
API Credentials model for secure storage of API keys and credentials
"""
import json
import os
from datetime import datetime
from cryptography.fernet import Fernet
import base64
import uuid

class APICredentials:
    """Model for storing API credentials securely"""
    
    def __init__(self, credential_id=None, provider=None, api_key=None, api_secret=None, 
                 base_url=None, is_active=True, created_by=None, created_at=None, 
                 updated_at=None, description=None):
        self.id = credential_id or str(uuid.uuid4())
        self.provider = provider  # e.g., 'racing_api', 'odds_api', etc.
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.is_active = is_active
        self.created_by = created_by  # User ID who created this credential
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()
        self.description = description
        self._encryption_key = None
    
    @staticmethod
    def get_encryption_key():
        """Get encryption key for credentials from environment variable"""
        # First try to get from environment variable
        encryption_key = os.getenv('ENCRYPTION_KEY')
        
        if encryption_key:
            # If it's a string, encode it to bytes
            if isinstance(encryption_key, str):
                return encryption_key.encode()
            return encryption_key
        
        # Fallback: check for legacy key file (for backward compatibility)
        key_file = os.path.join(os.path.dirname(__file__), '..', 'data', '.encryption_key')
        if os.path.exists(key_file):
            print("WARNING: Using legacy encryption key file. Please migrate to ENCRYPTION_KEY environment variable.")
            with open(key_file, 'rb') as f:
                return f.read()
        
        # If no key found, raise an error with instructions
        raise ValueError(
            "No encryption key found. Please set the ENCRYPTION_KEY environment variable.\n"
            "Generate a new key with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    
    def encrypt_sensitive_data(self):
        """Encrypt sensitive credential data"""
        if not self._encryption_key:
            self._encryption_key = self.get_encryption_key()
        
        fernet = Fernet(self._encryption_key)
        
        if self.api_key:
            self.api_key = base64.b64encode(fernet.encrypt(self.api_key.encode())).decode()
        if self.api_secret:
            self.api_secret = base64.b64encode(fernet.encrypt(self.api_secret.encode())).decode()
    
    def decrypt_sensitive_data(self):
        """Decrypt sensitive credential data"""
        if not self._encryption_key:
            self._encryption_key = self.get_encryption_key()
        
        fernet = Fernet(self._encryption_key)
        
        try:
            if self.api_key:
                self.api_key = fernet.decrypt(base64.b64decode(self.api_key.encode())).decode()
            if self.api_secret:
                self.api_secret = fernet.decrypt(base64.b64decode(self.api_secret.encode())).decode()
        except Exception:
            # Data might not be encrypted yet
            pass
    
    def to_dict(self, include_sensitive=False):
        """Convert credentials to dictionary"""
        data = {
            'id': self.id,
            'provider': self.provider,
            'base_url': self.base_url,
            'is_active': self.is_active,
            'created_by': self.created_by,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'description': self.description
        }
        
        if include_sensitive:
            data.update({
                'api_key': self.api_key,
                'api_secret': self.api_secret
            })
        else:
            # Mask sensitive data for display
            data.update({
                'api_key': '***' + (self.api_key[-4:] if self.api_key and len(self.api_key) > 4 else '***'),
                'api_secret': '***' + (self.api_secret[-4:] if self.api_secret and len(self.api_secret) > 4 else '***') if self.api_secret else None
            })
        
        return data
    
    @classmethod
    def from_dict(cls, data):
        """Create credentials from dictionary"""
        return cls(
            credential_id=data.get('id'),
            provider=data.get('provider'),
            api_key=data.get('api_key'),
            api_secret=data.get('api_secret'),
            base_url=data.get('base_url'),
            is_active=data.get('is_active', True),
            created_by=data.get('created_by'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            description=data.get('description')
        )
    
    @staticmethod
    def get_data_file():
        """Get the path to the credentials data file"""
        return os.path.join(os.path.dirname(__file__), '..', 'data', 'api_credentials.json')
    
    @classmethod
    def load_all_credentials(cls):
        """Load all credentials from JSON file"""
        data_file = cls.get_data_file()
        try:
            with open(data_file, 'r') as f:
                credentials_data = json.load(f)
                credentials = []
                for cred_data in credentials_data:
                    cred = cls.from_dict(cred_data)
                    cred.decrypt_sensitive_data()
                    credentials.append(cred)
                return credentials
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    @classmethod
    def save_all_credentials(cls, credentials):
        """Save all credentials to JSON file"""
        data_file = cls.get_data_file()
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        
        credentials_data = []
        for cred in credentials:
            # Create a copy to encrypt
            cred_copy = cls.from_dict(cred.to_dict(include_sensitive=True))
            cred_copy.encrypt_sensitive_data()
            credentials_data.append(cred_copy.to_dict(include_sensitive=True))
        
        with open(data_file, 'w') as f:
            json.dump(credentials_data, f, indent=2)
    
    @classmethod
    def get_by_id(cls, credential_id):
        """Get credentials by ID"""
        credentials = cls.load_all_credentials()
        for cred in credentials:
            if cred.id == credential_id:
                return cred
        return None
    
    @classmethod
    def get_by_provider(cls, provider):
        """Get active credentials by provider"""
        credentials = cls.load_all_credentials()
        for cred in credentials:
            if cred.provider == provider and cred.is_active:
                return cred
        return None
    
    @classmethod
    def get_all_providers(cls):
        """Get list of all configured providers"""
        credentials = cls.load_all_credentials()
        providers = set()
        for cred in credentials:
            if cred.is_active:
                providers.add(cred.provider)
        return list(providers)
    
    @classmethod
    def get_all(cls):
        """Get all credentials"""
        return cls.load_all_credentials()
    
    def get_decrypted_api_key(self):
        """Get decrypted API key"""
        if not self.api_key:
            return None
        
        if not self._encryption_key:
            self._encryption_key = self.get_encryption_key()
        
        fernet = Fernet(self._encryption_key)
        
        try:
            return fernet.decrypt(base64.b64decode(self.api_key.encode())).decode()
        except Exception:
            # Data might not be encrypted yet
            return self.api_key
    
    def get_decrypted_api_secret(self):
        """Get decrypted API secret"""
        if not self.api_secret:
            return None
        
        if not self._encryption_key:
            self._encryption_key = self.get_encryption_key()
        
        fernet = Fernet(self._encryption_key)
        
        try:
            return fernet.decrypt(base64.b64decode(self.api_secret.encode())).decode()
        except Exception:
            # Data might not be encrypted yet
            return self.api_secret
    
    def save(self):
        """Save this credential to the database"""
        credentials = self.load_all_credentials()
        
        # Update timestamp
        self.updated_at = datetime.now().isoformat()
        
        # Check if credential exists
        for i, cred in enumerate(credentials):
            if cred.id == self.id:
                credentials[i] = self
                break
        else:
            credentials.append(self)
        
        self.save_all_credentials(credentials)
    
    def delete(self):
        """Delete this credential from the database"""
        credentials = self.load_all_credentials()
        credentials = [cred for cred in credentials if cred.id != self.id]
        self.save_all_credentials(credentials)
    
    @classmethod
    def get_credentials_by_user(cls, user_id):
        """Get all credentials created by a specific user"""
        credentials = cls.load_all_credentials()
        return [cred for cred in credentials if cred.created_by == user_id]
    
    def __repr__(self):
        return f'<APICredentials {self.provider}>'