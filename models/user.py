"""
User model for authentication and user management
"""
import json
import os
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

class User(UserMixin):
    """User model with authentication capabilities"""
    
    def __init__(self, user_id=None, username=None, email=None, password_hash=None, 
                 role='user', is_active=True, created_at=None, last_login=None):
        self.id = user_id or str(uuid.uuid4())
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role  # 'admin' or 'user'
        self._is_active = is_active
        self.created_at = created_at or datetime.now().isoformat()
        self.last_login = last_login
    
    @property
    def is_active(self):
        """Flask-Login required property"""
        return self._is_active
    
    @is_active.setter
    def is_active(self, value):
        """Set active status"""
        self._is_active = value
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        """Check if user has admin role"""
        return self.role == 'admin'
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.now().isoformat()
    
    def to_dict(self):
        """Convert user to dictionary for JSON storage"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'password_hash': self.password_hash,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at,
            'last_login': self.last_login
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create user from dictionary"""
        return cls(
            user_id=data.get('id'),
            username=data.get('username'),
            email=data.get('email'),
            password_hash=data.get('password_hash'),
            role=data.get('role', 'user'),
            is_active=data.get('is_active', True),
            created_at=data.get('created_at'),
            last_login=data.get('last_login')
        )
    
    @staticmethod
    def get_data_file():
        """Get path to users data file"""
        return os.path.join(os.path.dirname(__file__), '..', 'data', 'users.json')
    
    @classmethod
    def load_all_users(cls):
        """Load all users from JSON file"""
        data_file = cls.get_data_file()
        if not os.path.exists(data_file):
            return []
        
        try:
            with open(data_file, 'r') as f:
                users_data = json.load(f)
                return [cls.from_dict(user_data) for user_data in users_data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    @classmethod
    def save_all_users(cls, users):
        """Save all users to JSON file"""
        data_file = cls.get_data_file()
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        
        users_data = [user.to_dict() for user in users]
        with open(data_file, 'w') as f:
            json.dump(users_data, f, indent=2)
    
    @classmethod
    def get_by_id(cls, user_id):
        """Get user by ID"""
        users = cls.load_all_users()
        for user in users:
            if user.id == user_id:
                return user
        return None
    
    @classmethod
    def get_by_username(cls, username):
        """Get user by username"""
        users = cls.load_all_users()
        for user in users:
            if user.username == username:
                return user
        return None
    
    @classmethod
    def get_by_email(cls, email):
        """Get user by email"""
        users = cls.load_all_users()
        for user in users:
            if user.email == email:
                return user
        return None
    
    def save(self):
        """Save user to database"""
        users = self.load_all_users()
        
        # Update existing user or add new one
        for i, user in enumerate(users):
            if user.id == self.id:
                users[i] = self
                break
        else:
            users.append(self)
        
        self.save_all_users(users)
        return self
    
    def delete(self):
        """Delete user from database"""
        users = self.load_all_users()
        users = [user for user in users if user.id != self.id]
        self.save_all_users(users)
    
    @classmethod
    def create_admin_user(cls, username, email, password):
        """Create an admin user"""
        admin = cls(username=username, email=email, role='admin')
        admin.set_password(password)
        return admin.save()
    
    @classmethod
    def get_all_users(cls):
        """Get all users"""
        return cls.load_all_users()
    
    @classmethod
    def get_users_by_role(cls, role):
        """Get users by role"""
        users = cls.load_all_users()
        return [user for user in users if user.role == role]
    
    def __repr__(self):
        return f'<User {self.username}>'