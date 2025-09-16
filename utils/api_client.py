"""
Horse Racing API Client
Handles integration with external horse racing data APIs
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RaceData:
    """Data structure for race information from API"""
    race_id: str
    name: str
    date: datetime
    location: str
    distance: str
    track_condition: str
    purse: float
    horses: List[Dict[str, Any]]
    status: str = 'upcoming'

@dataclass
class HorseData:
    """Data structure for horse information from API"""
    horse_id: str
    name: str
    age: int
    breed: str
    color: str
    jockey: str
    trainer: str
    owner: str
    weight: float
    odds: Optional[float] = None

class HorseRacingAPIClient:
    """
    Generic API client for horse racing data
    Supports multiple data sources and providers
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HorseRacingPrediction/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

class TheRacingAPI(HorseRacingAPIClient):
    """
    Implementation for TheRacingAPI service
    Provides real horse racing data from TheRacingAPI.com
    """
    
    def __init__(self, username: str = None, password: str = None):
        super().__init__()
        self.username = username or os.getenv('THERACINGAPI_USERNAME', '')
        self.password = password or os.getenv('THERACINGAPI_PASSWORD', '')
        self.base_url = 'https://api.theracingapi.com/v1'
        
        # Set up HTTP Basic Authentication
        if self.username and self.password:
            from requests.auth import HTTPBasicAuth
            self.session.auth = HTTPBasicAuth(self.username, self.password)
    
    def get_upcoming_races(self, days_ahead: int = 7) -> List[RaceData]:
        """Fetch upcoming races from TheRacingAPI"""
        # Note: Current subscription only has access to /courses endpoint
        # Race data endpoints (/meetings, /racecards) require a higher subscription plan
        logger.info("TheRacingAPI: Race data endpoints not available with current subscription")
        logger.info("TheRacingAPI: Only /courses endpoint is accessible")
        
        # Return empty list quickly instead of trying non-existent endpoints
        return []
    
    def get_race_details(self, race_id: str) -> Optional[RaceData]:
        """Fetch detailed information for a specific race from TheRacingAPI"""
        url = f"{self.base_url}/races/{race_id}"
        data = self._make_request(url)
        
        if not data:
            return None
        
        try:
            return self._parse_theracing_race_data(data.get('race', {}), data.get('meeting', {}))
        except Exception as e:
            logger.error(f"Error parsing TheRacingAPI race details: {e}")
            return None
    
    def get_race_horses(self, race_id: str) -> List[HorseData]:
        """Fetch horses participating in a specific race from TheRacingAPI"""
        url = f"{self.base_url}/races/{race_id}/runners"
        data = self._make_request(url)
        
        if not data:
            return []
        
        horses = []
        for runner_info in data.get('runners', []):
            try:
                horse = self._parse_theracing_horse_data(runner_info)
                horses.append(horse)
            except Exception as e:
                logger.error(f"Error parsing TheRacingAPI horse data: {e}")
                continue
        
        return horses
    
    def _parse_theracing_race_data(self, race_info: Dict, meeting_info: Dict) -> RaceData:
        """Parse race data from TheRacingAPI response"""
        race_time = race_info.get('start_time', '')
        meeting_date = meeting_info.get('date', '')
        
        # Combine date and time
        if meeting_date and race_time:
            datetime_str = f"{meeting_date} {race_time}"
            race_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        else:
            race_datetime = datetime.now()
        
        return RaceData(
            race_id=str(race_info.get('id', '')),
            name=race_info.get('name', 'Unknown Race'),
            date=race_datetime,
            location=meeting_info.get('name', 'Unknown Track'),
            distance=f"{race_info.get('distance', 0)}m",
            track_condition=race_info.get('going', 'Good'),
            purse=float(race_info.get('prize', 0)),
            horses=[],
            status=race_info.get('status', 'upcoming')
        )
    
    def _parse_theracing_horse_data(self, runner_info: Dict) -> HorseData:
        """Parse horse data from TheRacingAPI response"""
        horse_info = runner_info.get('horse', {})
        jockey_info = runner_info.get('jockey', {})
        trainer_info = runner_info.get('trainer', {})
        
        return HorseData(
            horse_id=str(horse_info.get('id', '')),
            name=horse_info.get('name', 'Unknown Horse'),
            age=int(horse_info.get('age', 0)),
            breed=horse_info.get('breed', 'Unknown'),
            color=horse_info.get('colour', 'Unknown'),
            jockey=jockey_info.get('name', 'Unknown'),
            trainer=trainer_info.get('name', 'Unknown'),
            owner=horse_info.get('owner', 'Unknown'),
            weight=float(runner_info.get('weight', 0)),
            odds=float(runner_info.get('odds', {}).get('decimal')) if runner_info.get('odds', {}).get('decimal') else None
        )
        
    def _load_config(self) -> Dict[str, Any]:
        """Load API configuration from environment or config file"""
        return {
            'timeout': int(os.getenv('API_TIMEOUT', '10')),  # Reduced from 30 to 10 seconds
            'max_retries': int(os.getenv('API_MAX_RETRIES', '1')),  # Reduced from 3 to 1 retry
            'base_urls': {
                'sample': 'https://api.example-racing.com/v1',
                'mock': 'http://localhost:8080/api'  # For testing
            },
            'api_keys': {
                'sample': os.getenv('RACING_API_KEY', ''),
                'mock': 'demo_key'
            }
        }
    
    def _make_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[Dict]:
        """Make HTTP request with error handling and retries"""
        for attempt in range(self.config['max_retries']):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.config['timeout'],
                    **kwargs
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config['max_retries'] - 1:
                    logger.error(f"All API request attempts failed for {url}")
                    return None
        return None

class SampleRacingAPI(HorseRacingAPIClient):
    """
    Sample implementation for a racing API
    This can be adapted for real APIs like:
    - The Odds API
    - RapidAPI Horse Racing
    - Custom racing data providers
    """
    
    def __init__(self, api_key: str = None):
        # Initialize with sample config instead of calling super().__init__()
        self.config = {
            'base_urls': {'sample': 'https://api.sample-racing.com'},
            'api_keys': {'sample': 'sample_key'},
            'rate_limits': {'sample': 60},
            'timeout': 30
        }
        self.session = requests.Session()
        self.api_key = api_key or self.config['api_keys']['sample']
        self.base_url = self.config['base_urls']['sample']
        
        if self.api_key:
            self.session.headers.update({'X-API-Key': self.api_key})
    
    def get_upcoming_races(self, days_ahead: int = 7) -> List[RaceData]:
        """Fetch upcoming races for the next N days"""
        end_date = datetime.now() + timedelta(days=days_ahead)
        
        params = {
            'start_date': datetime.now().strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'status': 'upcoming'
        }
        
        url = f"{self.base_url}/races"
        data = self._make_request(url, params=params)
        
        if not data:
            return []
        
        races = []
        for race_info in data.get('races', []):
            try:
                race = self._parse_race_data(race_info)
                races.append(race)
            except Exception as e:
                logger.error(f"Error parsing race data: {e}")
                continue
        
        return races
    
    def get_race_details(self, race_id: str) -> Optional[RaceData]:
        """Fetch detailed information for a specific race"""
        url = f"{self.base_url}/races/{race_id}"
        data = self._make_request(url)
        
        if not data:
            return None
        
        try:
            return self._parse_race_data(data)
        except Exception as e:
            logger.error(f"Error parsing race details: {e}")
            return None
    
    def get_race_horses(self, race_id: str) -> List[HorseData]:
        """Fetch horses participating in a specific race"""
        url = f"{self.base_url}/races/{race_id}/horses"
        data = self._make_request(url)
        
        if not data:
            return []
        
        horses = []
        for horse_info in data.get('horses', []):
            try:
                horse = self._parse_horse_data(horse_info)
                horses.append(horse)
            except Exception as e:
                logger.error(f"Error parsing horse data: {e}")
                continue
        
        return horses
    
    def _parse_race_data(self, race_info: Dict) -> RaceData:
        """Parse race data from API response"""
        return RaceData(
            race_id=str(race_info['id']),
            name=race_info['name'],
            date=datetime.fromisoformat(race_info['date'].replace('Z', '+00:00')),
            location=race_info['location'],
            distance=race_info['distance'],
            track_condition=race_info.get('track_condition', 'Good'),
            purse=float(race_info.get('purse', 0)),
            horses=race_info.get('horses', []),
            status=race_info.get('status', 'upcoming')
        )
    
    def _parse_horse_data(self, horse_info: Dict) -> HorseData:
        """Parse horse data from API response"""
        return HorseData(
            horse_id=str(horse_info['id']),
            name=horse_info['name'],
            age=int(horse_info.get('age', 0)),
            breed=horse_info.get('breed', 'Unknown'),
            color=horse_info.get('color', 'Unknown'),
            jockey=horse_info.get('jockey', 'Unknown'),
            trainer=horse_info.get('trainer', 'Unknown'),
            owner=horse_info.get('owner', 'Unknown'),
            weight=float(horse_info.get('weight', 0)),
            odds=float(horse_info['odds']) if horse_info.get('odds') else None
        )

class MockRacingAPI(HorseRacingAPIClient):
    """
    Mock API for testing purposes
    Generates sample race data
    """
    
    def __init__(self):
        # Initialize with mock config instead of calling super().__init__()
        self.config = {
            'base_urls': {'mock': 'http://localhost:mock'},
            'api_keys': {'mock': 'mock_key'},
            'rate_limits': {'mock': 60},
            'timeout': 30
        }
        self.base_url = self.config['base_urls']['mock']
    
    def get_upcoming_races(self, days_ahead: int = 7) -> List[RaceData]:
        """Generate mock upcoming races"""
        races = []
        
        for i in range(5):  # Generate 5 sample races
            race_date = datetime.now() + timedelta(days=i+1, hours=i*2)
            
            race = RaceData(
                race_id=f"mock_race_{i+1}",
                name=f"Sample Stakes Race {i+1}",
                date=race_date,
                location=f"Sample Track {i+1}",
                distance=f"{1200 + i*200}m",
                track_condition="Good",
                purse=50000.0 + i*10000,
                horses=self._generate_mock_horses(8),
                status='upcoming'
            )
            races.append(race)
        
        return races
    
    def get_race_details(self, race_id: str) -> Optional[RaceData]:
        """Get mock race details"""
        if not race_id.startswith('mock_race_'):
            return None
        
        race_num = int(race_id.split('_')[-1])
        race_date = datetime.now() + timedelta(days=race_num, hours=race_num*2)
        
        return RaceData(
            race_id=race_id,
            name=f"Sample Stakes Race {race_num}",
            date=race_date,
            location=f"Sample Track {race_num}",
            distance=f"{1200 + race_num*200}m",
            track_condition="Good",
            purse=50000.0 + race_num*10000,
            horses=self._generate_mock_horses(8),
            status='upcoming'
        )
    
    def get_race_horses(self, race_id: str) -> List[HorseData]:
        """Get mock horses for a race"""
        return self._generate_mock_horses(8)
    
    def _generate_mock_horses(self, count: int) -> List[Dict[str, Any]]:
        """Generate mock horse data"""
        horse_names = [
            "Thunder Bolt", "Lightning Strike", "Storm Chaser", "Wind Runner",
            "Fire Spirit", "Golden Arrow", "Silver Bullet", "Midnight Express",
            "Royal Crown", "Diamond Star", "Blazing Trail", "Swift Arrow"
        ]
        
        jockeys = [
            "J. Smith", "M. Johnson", "R. Williams", "S. Brown",
            "T. Davis", "K. Wilson", "L. Garcia", "P. Martinez"
        ]
        
        trainers = [
            "A. Thompson", "B. Anderson", "C. Taylor", "D. Moore",
            "E. Jackson", "F. White", "G. Harris", "H. Clark"
        ]
        
        horses = []
        for i in range(min(count, len(horse_names))):
            horse_data = {
                'id': f"mock_horse_{i+1}",
                'name': horse_names[i],
                'age': 3 + (i % 5),
                'breed': 'Thoroughbred',
                'color': ['Bay', 'Chestnut', 'Black', 'Gray'][i % 4],
                'jockey': jockeys[i % len(jockeys)],
                'trainer': trainers[i % len(trainers)],
                'owner': f"Owner {i+1}",
                'weight': 55.0 + i,
                'odds': 2.5 + i * 0.5
            }
            horses.append(horse_data)
        
        return horses

class APIManager:
    """
    Manager class to handle multiple API providers
    """
    
    def __init__(self):
        self.providers = {
            'mock': MockRacingAPI(),
            'sample': SampleRacingAPI(),
            'theracingapi': TheRacingAPI()
            # Add more providers as needed
            # 'sample': SampleRacingAPI(api_key=os.getenv('SAMPLE_API_KEY'))
        }
        self.default_provider = 'mock'
    
    def get_provider(self, provider_name: str = None) -> HorseRacingAPIClient:
        """Get API provider instance"""
        provider_name = provider_name or self.default_provider
        return self.providers.get(provider_name)
    
    def add_provider(self, name: str, provider: HorseRacingAPIClient):
        """Add a new API provider"""
        self.providers[name] = provider
    
    def get_upcoming_races(self, provider_name: str = None, days_ahead: int = 7) -> List[RaceData]:
        """Get upcoming races from specified provider"""
        provider = self.get_provider(provider_name)
        if provider:
            return provider.get_upcoming_races(days_ahead)
        return []
    
    def get_race_details(self, race_id: str, provider_name: str = None) -> Optional[RaceData]:
        """Get race details from specified provider"""
        provider = self.get_provider(provider_name)
        if provider:
            return provider.get_race_details(race_id)
        return None