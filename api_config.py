"""
API configuration module for the Flask application.
Provides API configuration and settings.
"""

import os
from typing import Dict, Any


class APIConfig:
    """API configuration class"""
    
    def __init__(self):
        self.api_key = os.getenv('API_KEY', '')
        self.api_secret = os.getenv('API_SECRET', '')
        self.api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
        self.api_timeout = int(os.getenv('API_TIMEOUT', '30'))
        self.api_retries = int(os.getenv('API_RETRIES', '3'))
        self.rate_limit = int(os.getenv('API_RATE_LIMIT', '100'))
        
    def get_config(self) -> Dict[str, Any]:
        """Get API configuration dictionary"""
        return {
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'base_url': self.api_base_url,
            'timeout': self.api_timeout,
            'retries': self.api_retries,
            'rate_limit': self.rate_limit
        }
    
    def get_headers(self) -> Dict[str, str]:
        """Get default API headers"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'HorseRacingPrediction/1.0'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        return headers
    
    def is_configured(self) -> bool:
        """Check if API is properly configured"""
        return bool(self.api_key and self.api_secret)
    
    @property
    def endpoints(self) -> Dict[str, str]:
        """Get API endpoints"""
        return {
            'races': f'{self.api_base_url}/api/races',
            'horses': f'{self.api_base_url}/api/horses',
            'predictions': f'{self.api_base_url}/api/predictions',
            'train': f'{self.api_base_url}/api/train',
            'status': f'{self.api_base_url}/api/status'
        }


# Create global API config instance
api_config = APIConfig()