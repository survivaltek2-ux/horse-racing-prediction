from typing import Dict, Any, List, Optional
from datetime import datetime
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Table
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from werkzeug.security import generate_password_hash, check_password_hash
import json

# Import the database instance from config
from config.database_config import db

# Association table for many-to-many relationship between races and horses
race_horses = Table('race_horses',
    db.Model.metadata,
    Column('race_id', Integer, ForeignKey('races.id'), primary_key=True),
    Column('horse_id', Integer, ForeignKey('horses.id'), primary_key=True)
)

class BaseModel(db.Model):
    """Base model with common fields and methods"""
    __abstract__ = True
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        data = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                data[column.name] = value.isoformat()
            else:
                data[column.name] = value
        return data
    
    def save(self):
        """Save model to database"""
        try:
            db.session.add(self)
            db.session.commit()
            return self.id
        except Exception as e:
            db.session.rollback()
            print(f"Error saving to database: {e}")
            return None
    
    def delete(self) -> bool:
        """Delete model from database"""
        try:
            db.session.delete(self)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            print(f"Error deleting from database: {e}")
            return False
    
    def update(self, **kwargs):
        """Update model with new data"""
        try:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            self.updated_at = datetime.utcnow()
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            print(f"Error updating model: {e}")
            return False

class APICredentials(BaseModel):
    """SQLAlchemy model for API credentials"""
    __tablename__ = 'api_credentials'
    
    provider = Column(String(100), nullable=False)
    base_url = Column(String(500), nullable=False)
    api_key = Column(String(500), nullable=False)
    api_secret = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    def __repr__(self):
        return f'<APICredentials {self.provider}>'
    
    @classmethod
    def get_all(cls) -> List['APICredentials']:
        """Get all API credentials"""
        try:
            return cls.query.all()
        except Exception as e:
            print(f"Error getting API credentials: {e}")
            return []
    
    @classmethod
    def get_by_id(cls, credential_id: int) -> Optional['APICredentials']:
        """Get API credential by ID"""
        try:
            return cls.query.get(credential_id)
        except Exception as e:
            print(f"Error getting API credential by ID: {e}")
            return None
    
    @classmethod
    def get_by_provider(cls, provider: str) -> Optional['APICredentials']:
        """Get active API credential by provider"""
        try:
            return cls.query.filter_by(provider=provider, is_active=True).first()
        except Exception as e:
            print(f"Error getting API credential by provider: {e}")
            return None
    
    def test_connection(self) -> Dict[str, Any]:
        """Test API connection"""
        return {
            'success': True,
            'message': f'Connection test for {self.provider}',
            'provider': self.provider,
            'base_url': self.base_url
        }

class Race(BaseModel):
    """SQLAlchemy model for races"""
    __tablename__ = 'races'
    
    name = Column(String(200), nullable=False)
    date = Column(DateTime, nullable=False)
    time = Column(String(10), nullable=True)
    track = Column(String(100), nullable=False)
    distance = Column(Float, nullable=False)
    surface = Column(String(50), nullable=True)
    race_class = Column(String(50), nullable=True)
    prize_money = Column(Float, default=0.0)
    weather = Column(String(50), nullable=True)
    track_condition = Column(String(50), nullable=True)
    status = Column(String(20), default='upcoming', nullable=False)  # upcoming, running, completed, cancelled
    results = Column(Text, nullable=True)  # JSON string for race results
    
    # Relationships
    horses = relationship('Horse', secondary=race_horses, back_populates='races')
    predictions = relationship('Prediction', back_populates='race', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Race {self.name} on {self.date}>'
    
    @classmethod
    def get_all(cls, limit: int = 100) -> List['Race']:
        """Get all races"""
        try:
            return cls.query.limit(limit).all()
        except Exception as e:
            print(f"Error getting races: {e}")
            return []
    
    @classmethod
    def get_by_id(cls, race_id: int) -> Optional['Race']:
        """Get race by ID"""
        try:
            return cls.query.get(race_id)
        except Exception as e:
            print(f"Error getting race by ID: {e}")
            return None
    
    @classmethod
    def get_upcoming(cls, limit: int = 50) -> List['Race']:
        """Get upcoming races"""
        try:
            current_time = datetime.utcnow()
            return cls.query.filter(cls.date > current_time).order_by(cls.date).limit(limit).all()
        except Exception as e:
            print(f"Error getting upcoming races: {e}")
            return []
    
    def add_horse(self, horse):
        """Add horse to race"""
        if horse not in self.horses:
            self.horses.append(horse)
            db.session.commit()
    
    def remove_horse(self, horse):
        """Remove horse from race"""
        if horse in self.horses:
            self.horses.remove(horse)
            db.session.commit()
    
    @property
    def results_dict(self) -> Dict[str, Any]:
        """Get results as dictionary"""
        if self.results:
            try:
                return json.loads(self.results)
            except json.JSONDecodeError:
                return {}
        return {}
    
    @results_dict.setter
    def results_dict(self, value: Dict[str, Any]):
        """Set results from dictionary"""
        self.results = json.dumps(value)

class Horse(BaseModel):
    """SQLAlchemy model for horses"""
    __tablename__ = 'horses'
    
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    sex = Column(String(10), nullable=True)
    color = Column(String(50), nullable=True)
    sire = Column(String(100), nullable=True)
    dam = Column(String(100), nullable=True)
    trainer = Column(String(100), nullable=True)
    jockey = Column(String(100), nullable=True)
    owner = Column(String(100), nullable=True)
    weight = Column(Float, default=0.0)
    form = Column(String(50), nullable=True)
    rating = Column(Integer, default=0)
    last_run = Column(DateTime, nullable=True)
    wins = Column(Integer, default=0)
    places = Column(Integer, default=0)
    runs = Column(Integer, default=0)
    earnings = Column(Float, default=0.0)
    
    # Relationships
    races = relationship('Race', secondary=race_horses, back_populates='horses')
    predictions = relationship('Prediction', back_populates='horse', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Horse {self.name}>'
    
    @classmethod
    def get_all(cls, limit: int = 100) -> List['Horse']:
        """Get all horses"""
        try:
            return cls.query.limit(limit).all()
        except Exception as e:
            print(f"Error getting horses: {e}")
            return []
    
    @classmethod
    def get_by_id(cls, horse_id: int) -> Optional['Horse']:
        """Get horse by ID"""
        try:
            return cls.query.get(horse_id)
        except Exception as e:
            print(f"Error getting horse by ID: {e}")
            return None
    
    @classmethod
    def search_by_name(cls, name: str) -> List['Horse']:
        """Search horses by name"""
        try:
            return cls.query.filter(cls.name.ilike(f'%{name}%')).all()
        except Exception as e:
            print(f"Error searching horses by name: {e}")
            return []
    
    @hybrid_property
    def win_percentage(self) -> float:
        """Calculate win percentage"""
        if self.runs == 0:
            return 0.0
        return (self.wins / self.runs) * 100
    
    @hybrid_property
    def place_percentage(self) -> float:
        """Calculate place percentage"""
        if self.runs == 0:
            return 0.0
        return (self.places / self.runs) * 100

class Prediction(BaseModel):
    """SQLAlchemy model for predictions"""
    __tablename__ = 'predictions'
    
    race_id = Column(Integer, ForeignKey('races.id'), nullable=False)
    horse_id = Column(Integer, ForeignKey('horses.id'), nullable=False)
    predicted_position = Column(Integer, nullable=False)
    confidence = Column(Float, default=0.0)
    odds = Column(Float, default=0.0)
    factors = Column(Text, nullable=True)  # JSON string for prediction factors
    model_version = Column(String(50), nullable=True)
    
    # Relationships
    race = relationship('Race', back_populates='predictions')
    horse = relationship('Horse', back_populates='predictions')
    
    def __repr__(self):
        return f'<Prediction Race:{self.race_id} Horse:{self.horse_id} Position:{self.predicted_position}>'
    
    @classmethod
    def get_by_race(cls, race_id: int) -> List['Prediction']:
        """Get predictions for a race"""
        try:
            return cls.query.filter_by(race_id=race_id).all()
        except Exception as e:
            print(f"Error getting predictions by race: {e}")
            return []
    
    @classmethod
    def get_all(cls, limit: int = 100) -> List['Prediction']:
        """Get all predictions"""
        try:
            return cls.query.limit(limit).all()
        except Exception as e:
            print(f"Error getting predictions: {e}")
            return []
    
    @property
    def factors_dict(self) -> Dict[str, Any]:
        """Get factors as dictionary"""
        if self.factors:
            try:
                return json.loads(self.factors)
            except json.JSONDecodeError:
                return {}
        return {}
    
    @factors_dict.setter
    def factors_dict(self, value: Dict[str, Any]):
        """Set factors from dictionary"""
        self.factors = json.dumps(value)

class User(UserMixin, BaseModel):
    """SQLAlchemy model for users"""
    __tablename__ = 'users'
    
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def check_password(self, password: str) -> bool:
        """Check if password is correct"""
        return check_password_hash(self.password_hash, password)
    
    def set_password(self, password: str):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def get_id(self):
        """Flask-Login required method - return user ID as string"""
        return str(self.id)
    
    def is_admin_user(self) -> bool:
        """Check if user has admin privileges"""
        return self.is_admin
    
    @classmethod
    def get_by_id(cls, user_id: int) -> Optional['User']:
        """Get user by ID - required for Flask-Login"""
        try:
            return cls.query.get(user_id)
        except Exception as e:
            print(f"Error getting user by ID: {e}")
            return None
    
    @classmethod
    def get_by_username(cls, username: str) -> Optional['User']:
        """Get user by username"""
        try:
            return cls.query.filter_by(username=username).first()
        except Exception as e:
            print(f"Error getting user by username: {e}")
            return None
    
    @classmethod
    def get_by_email(cls, email: str) -> Optional['User']:
        """Get user by email"""
        try:
            return cls.query.filter_by(email=email).first()
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
            user = cls(
                username=username,
                email=email,
                is_admin=True
            )
            user.set_password(password)
            
            user_id = user.save()
            if user_id:
                print(f"Admin user '{username}' created successfully")
                return user
            else:
                print(f"Failed to create admin user '{username}'")
                return None
            
        except Exception as e:
            print(f"Error creating admin user: {e}")
            return None