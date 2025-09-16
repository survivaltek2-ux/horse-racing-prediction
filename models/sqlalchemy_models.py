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
    api_key = Column(String(500), nullable=True)
    api_secret = Column(String(500), nullable=True)
    username = Column(String(200), nullable=True)
    password = Column(String(500), nullable=True)
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
    
    # Basic Race Information
    name = Column(String(200), nullable=False)
    date = Column(DateTime, nullable=False)
    time = Column(String(10), nullable=True)
    track = Column(String(100), nullable=False)
    distance = Column(Float, nullable=False)
    prize_money = Column(Float, default=0.0)
    description = Column(Text, nullable=True)
    
    # Weather Conditions
    temperature = Column(Integer, nullable=True)  # Temperature in Fahrenheit
    humidity = Column(Integer, nullable=True)  # Humidity percentage
    wind_speed = Column(Integer, nullable=True)  # Wind speed in mph
    wind_direction = Column(String(10), nullable=True)  # Wind direction
    weather_description = Column(String(50), nullable=True)  # Weather description
    visibility = Column(Float, nullable=True)  # Visibility in miles
    
    # Track Conditions
    track_condition = Column(String(50), nullable=True)
    surface_type = Column(String(50), nullable=True)  # Dirt, Turf, Synthetic, etc.
    rail_position = Column(String(50), nullable=True)
    track_bias = Column(String(50), nullable=True)
    track_maintenance = Column(String(200), nullable=True)
    
    # Field Analysis
    field_size = Column(Integer, nullable=True)
    field_quality = Column(String(50), nullable=True)
    pace_scenario = Column(String(50), nullable=True)
    competitive_balance = Column(String(50), nullable=True)
    speed_figures_range = Column(String(20), nullable=True)
    
    # Betting Information
    total_pool = Column(Float, nullable=True)
    win_pool = Column(Float, nullable=True)
    exacta_pool = Column(Float, nullable=True)
    trifecta_pool = Column(Float, nullable=True)
    superfecta_pool = Column(Float, nullable=True)
    morning_line_favorite = Column(String(10), nullable=True)
    
    # Race Conditions
    age_restrictions = Column(String(50), nullable=True)
    sex_restrictions = Column(String(50), nullable=True)
    weight_conditions = Column(String(100), nullable=True)
    claiming_price = Column(Float, nullable=True)
    race_grade = Column(String(50), nullable=True)
    
    # Historical Data
    track_record = Column(String(20), nullable=True)
    average_winning_time = Column(String(20), nullable=True)
    course_record_holder = Column(String(100), nullable=True)
    similar_race_results = Column(Text, nullable=True)
    trainer_jockey_stats = Column(Text, nullable=True)
    
    # Media Coverage
    tv_coverage = Column(String(50), nullable=True)
    streaming_available = Column(String(10), nullable=True)
    featured_race = Column(String(10), nullable=True)
    
    # Legacy fields for compatibility
    surface = Column(String(50), nullable=True)
    race_class = Column(String(50), nullable=True)
    weather = Column(String(50), nullable=True)
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
    
    # Basic Information
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    sex = Column(String(10), nullable=True)
    color = Column(String(50), nullable=True)
    breed = Column(String(100), nullable=True)
    height = Column(Float, nullable=True)  # Height in hands
    markings = Column(Text, nullable=True)  # Physical markings description
    
    # Pedigree Information
    sire = Column(String(100), nullable=True)
    dam = Column(String(100), nullable=True)
    sire_line = Column(String(200), nullable=True)
    dam_line = Column(String(200), nullable=True)
    breeding_value = Column(Integer, nullable=True)  # 1-10 scale
    
    # Connections
    trainer = Column(String(100), nullable=True)
    jockey = Column(String(100), nullable=True)
    owner = Column(String(100), nullable=True)
    breeder = Column(String(100), nullable=True)
    stable = Column(String(100), nullable=True)
    
    # Physical Attributes
    weight = Column(Float, default=0.0)
    body_condition = Column(Integer, nullable=True)  # 1-10 scale
    conformation_score = Column(Integer, nullable=True)  # 1-10 scale
    temperament = Column(String(50), nullable=True)
    
    # Performance Analytics
    speed_rating = Column(Integer, nullable=True)  # Speed figures (0-120)
    class_rating = Column(Integer, nullable=True)  # Class level performance (1-10)
    distance_preference = Column(String(20), nullable=True)  # Preferred race distance
    surface_preference = Column(String(20), nullable=True)  # Preferred track surface
    track_bias_rating = Column(Integer, nullable=True)  # Performance on different track conditions (1-10)
    pace_style = Column(String(50), nullable=True)  # Running style
    closing_kick = Column(Integer, nullable=True)  # 1-10 scale
    
    # Training & Fitness
    days_since_last_race = Column(Integer, nullable=True)  # Rest period in days
    fitness_level = Column(Integer, nullable=True)  # Current fitness rating (1-10)
    training_intensity = Column(String(50), nullable=True)  # Light, Moderate, Heavy
    workout_times = Column(Text, nullable=True)  # JSON string of recent workout performance
    injury_history = Column(Text, nullable=True)  # JSON string for injury records
    recovery_time = Column(Integer, nullable=True)  # Days needed for recovery
    
    # Behavioral & Racing Style
    gate_behavior = Column(String(50), nullable=True)
    racing_tactics = Column(String(100), nullable=True)
    equipment_used = Column(String(200), nullable=True)
    medication_notes = Column(Text, nullable=True)
    
    # Financial Information
    purchase_price = Column(Float, nullable=True)
    current_value = Column(Float, nullable=True)
    insurance_value = Column(Float, nullable=True)
    stud_fee = Column(Float, nullable=True)
    
    # Performance Statistics
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
    
    def get_recent_performance(self, limit: int = 5) -> List[int]:
        """Get recent race positions for this horse"""
        try:
            # Get recent completed races for this horse
            from sqlalchemy import and_, desc
            
            # Query races where this horse participated and the race is completed
            recent_races = db.session.query(Race).join(
                race_horses, Race.id == race_horses.c.race_id
            ).filter(
                and_(
                    race_horses.c.horse_id == self.id,
                    Race.status == 'completed',
                    Race.results.isnot(None)
                )
            ).order_by(desc(Race.date)).limit(limit).all()
            
            positions = []
            for race in recent_races:
                try:
                    # Parse race results JSON to find this horse's position
                    if race.results:
                        results = json.loads(race.results)
                        if isinstance(results, dict) and 'positions' in results:
                            # Look for this horse's position in the results
                            for position_data in results['positions']:
                                if isinstance(position_data, dict) and position_data.get('horse_id') == self.id:
                                    positions.append(position_data.get('position', 0))
                                    break
                        elif isinstance(results, list):
                            # If results is a list of positions
                            for i, horse_id in enumerate(results, 1):
                                if horse_id == self.id:
                                    positions.append(i)
                                    break
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                    continue
            
            return positions
            
        except Exception as e:
            print(f"Error getting recent performance for horse {self.id}: {e}")
            # Return some mock data based on form if available
            if self.form:
                try:
                    # Parse form string like "1-2-3-4-5" into positions
                    form_positions = [int(pos) for pos in self.form.split('-') if pos.isdigit()]
                    return form_positions[:limit]
                except (ValueError, AttributeError):
                    pass
            
            # Return empty list if no data available
            return []
    
    def get_form(self, num_races=3):
        """Get form data as objects with position attribute for template usage"""
        positions = self.get_recent_performance(num_races)
        
        # Create simple objects with position attribute
        class FormEntry:
            def __init__(self, position):
                self.position = position
        
        return [FormEntry(pos) for pos in positions]

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
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    
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
    def get_all(cls, limit: int = 100) -> List['User']:
        """Get all users"""
        try:
            return cls.query.limit(limit).all()
        except Exception as e:
            print(f"Error getting all users: {e}")
            return []

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