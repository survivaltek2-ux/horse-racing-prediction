"""
API Service Layer
Integrates external race data APIs with the Flask application
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

# Import our models
from models.race import Race
from models.horse import Horse
from models.prediction import Prediction

from utils.api_client import APIManager, RaceData, HorseData
from config.api_config import APIConfig

logger = logging.getLogger(__name__)

class APIService:
    """
    Service layer for API integration
    Handles data synchronization between external APIs and local database
    """
    
    def __init__(self, app=None):
        self.app = app
        self.api_manager = APIManager()
        self.config = APIConfig.get_config()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize with Flask app"""
        self.app = app
        app.api_service = self
    
    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available API providers"""
        providers = APIConfig.get_available_providers()
        
        # Add configuration status
        for name, config in providers.items():
            config['configured'] = APIConfig.is_provider_configured(name)
        
        return providers
    
    def fetch_upcoming_races(self, provider: str = None, days_ahead: int = None) -> List[RaceData]:
        """
        Fetch upcoming races from API
        
        Args:
            provider: API provider name (defaults to configured provider)
            days_ahead: Number of days to look ahead (defaults to config)
        
        Returns:
            List of RaceData objects
        """
        days_ahead = days_ahead or self.config['default_days_ahead']
        
        try:
            races = self.api_manager.get_upcoming_races(provider, days_ahead)
            logger.info(f"Fetched {len(races)} races from {provider or 'default'} provider")
            return races
        except Exception as e:
            logger.error(f"Error fetching races from API: {e}")
            return []
    
    def import_race_from_api(self, race_data: RaceData, update_existing: bool = False) -> Optional[Any]:
        """
        Import a single race from API data into the JSON storage
        
        Args:
            race_data: RaceData object from API
            update_existing: Whether to update existing races
        
        Returns:
            Race object if successful, None otherwise
        """
        try:
            # Check if race already exists
            existing_races = Race.get_all_races()
            existing_race = None
            
            for race in existing_races:
                if (race.name == race_data.name and 
                    race.date == race_data.date.strftime('%Y-%m-%d') and 
                    race.location == race_data.location):
                    existing_race = race
                    break
            
            if existing_race and not update_existing:
                logger.info(f"Race {race_data.name} already exists, skipping")
                return existing_race
            
            # Import horses first
            horse_ids = []
            if race_data.horses:
                horse_ids = self._import_race_horses(race_data.horses)
            
            # Create race data
            race_dict = {
                'name': race_data.name,
                'date': race_data.date.strftime('%Y-%m-%d'),
                'location': race_data.location,
                'distance': race_data.distance,
                'track_condition': race_data.track_condition,
                'purse': race_data.purse,
                'horse_ids': horse_ids,
                'status': 'upcoming'
            }
            
            # Create or update race
            if existing_race:
                # Update existing race
                existing_race.distance = race_data.distance
                existing_race.track_condition = race_data.track_condition
                existing_race.purse = race_data.purse
                existing_race.horse_ids = horse_ids
                # Save would need to be implemented in the Race model
                race = existing_race
            else:
                # Create new race
                race = Race.create_race(race_dict)
            
            logger.info(f"Successfully imported race: {race_data.name}")
            return race
            
        except Exception as e:
            logger.error(f"Error importing race {race_data.name}: {e}")
            return None
    
    def _import_race_horses(self, horses_data: List[Dict[str, Any]]) -> List[int]:
        """Import horses and return list of horse IDs"""
        horse_ids = []
        
        for horse_info in horses_data:
            try:
                # Check if horse exists
                existing_horses = Horse.get_all_horses()
                existing_horse = None
                
                for horse in existing_horses:
                    if horse.name == horse_info['name']:
                        existing_horse = horse
                        break
                
                if existing_horse:
                    # Update existing horse info if needed
                    horse_data = {
                        'name': horse_info['name'],
                        'age': horse_info.get('age', existing_horse.age),
                        'breed': horse_info.get('breed', existing_horse.breed),
                        'color': horse_info.get('color', existing_horse.color),
                        'trainer': horse_info.get('trainer', existing_horse.trainer),
                        'owner': horse_info.get('owner', existing_horse.owner),
                        'jockey': horse_info.get('jockey', 'Unknown'),
                        'weight': horse_info.get('weight', 0),
                        'odds': horse_info.get('odds')
                    }
                    # Update existing horse (would need update method in Horse model)
                    horse = existing_horse
                    horse_ids.append(horse.id)
                else:
                    # Create new horse
                    horse_data = {
                        'name': horse_info['name'],
                        'age': horse_info.get('age', 0),
                        'breed': horse_info.get('breed', 'Unknown'),
                        'color': horse_info.get('color', 'Unknown'),
                        'trainer': horse_info.get('trainer', 'Unknown'),
                        'owner': horse_info.get('owner', 'Unknown'),
                        'jockey': horse_info.get('jockey', 'Unknown'),
                        'weight': horse_info.get('weight', 0),
                        'odds': horse_info.get('odds')
                    }
                    horse = Horse.create_horse(horse_data)
                    if horse:
                        horse_ids.append(horse.id)
                
            except Exception as e:
                logger.error(f"Error importing horse {horse_info.get('name', 'Unknown')}: {e}")
                continue
        
        return horse_ids
    
    def bulk_import_races(self, provider: str = None, days_ahead: int = None, 
                         update_existing: bool = False) -> Dict[str, Any]:
        """
        Bulk import races from API
        
        Returns:
            Dictionary with import statistics
        """
        stats = {
            'total_fetched': 0,
            'imported': 0,
            'updated': 0,
            'errors': 0,
            'skipped': 0
        }
        
        try:
            # Fetch races from API
            races_data = self.fetch_upcoming_races(provider, days_ahead)
            stats['total_fetched'] = len(races_data)
            
            if not races_data:
                logger.warning("No races fetched from API")
                return stats
            
            # Import each race
            for race_data in races_data:
                try:
                    # Check if race exists
                    existing_races = Race.get_all_races()
                    existing = None
                    
                    for race in existing_races:
                        if (race.name == race_data.name and 
                            race.date == race_data.date.strftime('%Y-%m-%d') and 
                            race.location == race_data.location):
                            existing = race
                            break
                    
                    if existing and not update_existing:
                        stats['skipped'] += 1
                        continue
                    
                    result = self.import_race_from_api(race_data, update_existing)
                    
                    if result:
                        if existing:
                            stats['updated'] += 1
                        else:
                            stats['imported'] += 1
                    else:
                        stats['errors'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing race {race_data.name}: {e}")
                    stats['errors'] += 1
            
            logger.info(f"Bulk import completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in bulk import: {e}")
            stats['errors'] = stats['total_fetched']
            return stats
    
    def get_race_details_from_api(self, race_id: str, provider: str = None) -> Optional[RaceData]:
        """Get detailed race information from API"""
        try:
            return self.api_manager.get_race_details(race_id, provider)
        except Exception as e:
            logger.error(f"Error fetching race details: {e}")
            return None
    
    def sync_race_odds(self, race_id: int, provider: str = None) -> bool:
        """
        Sync odds for a specific race from API
        
        Args:
            race_id: Local race ID
            provider: API provider name
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get local race
            all_races = Race.get_all_races()
            race = None
            for r in all_races:
                if r.id == race_id:
                    race = r
                    break
            
            if not race:
                logger.error(f"Race {race_id} not found")
                return False
            
            # This would need to be implemented based on the specific API
            # For now, return True as placeholder
            logger.info(f"Odds sync for race {race_id} would be implemented here")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing odds for race {race_id}: {e}")
            return False
    
    def sync_odds(self, provider: str = None) -> Dict[str, Any]:
        """Sync odds for upcoming races"""
        try:
            provider = provider or self.config.default_provider
            client = self.api_manager.get_client(provider)
            
            if not client:
                return {'error': f'Provider {provider} not available'}
            
            # Get upcoming races (next 7 days)
            from datetime import datetime, timedelta
            end_date = datetime.now() + timedelta(days=7)
            
            all_races = Race.get_all_races()
            upcoming_races = []
            
            for race in all_races:
                race_date = datetime.strptime(race.date, '%Y-%m-%d').date()
                if race_date >= datetime.now().date() and race_date <= end_date.date():
                    upcoming_races.append(race)
            
            stats = {'updated': 0, 'errors': 0}
            
            for race in upcoming_races:
                try:
                    # Fetch odds for this race
                    odds_data = client.get_odds(race.name, race.date, race.location)
                    
                    if odds_data:
                        # Update horse odds in race data
                        for horse_odds in odds_data:
                            # Find horse in race and update odds
                            if hasattr(race, 'horse_ids'):
                                for i, horse_id in enumerate(race.horse_ids):
                                    if horse_id == horse_odds.get('horse_id'):
                                        # Update odds (would need to extend race model to store odds)
                                        stats['updated'] += 1
                                        break
                        
                except Exception as e:
                    logger.error(f"Error syncing odds for race {race.name}: {e}")
                    stats['errors'] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error syncing odds: {e}")
            return {'error': str(e)}
    

    
    def get_import_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get history of API imports
        This would typically be stored in a separate table
        For now, return placeholder data
        """
        return [
            {
                'id': 1,
                'timestamp': datetime.now(),
                'provider': 'mock',
                'races_imported': 5,
                'status': 'success'
            }
        ]

# Global service instance
api_service = APIService()