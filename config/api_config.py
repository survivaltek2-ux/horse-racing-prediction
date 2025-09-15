"""
API Configuration for Horse Racing Data Integration
"""

import os
from typing import Dict, Any

class APIConfig:
    """Configuration class for API settings"""
    
    # Default configuration
    DEFAULT_CONFIG = {
        'timeout': 30,
        'max_retries': 3,
        'rate_limit_delay': 1.0,  # seconds between requests
        'cache_duration': 300,    # 5 minutes cache
        'max_races_per_request': 50,
        'default_days_ahead': 7
    }
    
    # API Provider configurations
    PROVIDERS = {
        'mock': {
            'name': 'Mock Racing API',
            'description': 'Test API for development',
            'base_url': 'http://localhost:8080/api',
            'api_key_required': False,
            'rate_limit': None,
            'features': ['races', 'horses', 'results']
        },
        'sample': {
            'name': 'Sample Racing API',
            'description': 'Example external API',
            'base_url': 'https://api.example-racing.com/v1',
            'api_key_required': True,
            'rate_limit': 100,  # requests per hour
            'features': ['races', 'horses', 'odds', 'results']
        },
        'odds_api': {
            'name': 'The Odds API',
            'description': 'Sports betting odds API',
            'base_url': 'https://api.the-odds-api.com/v4',
            'api_key_required': True,
            'rate_limit': 500,  # requests per month
            'features': ['odds', 'races']
        },
        'rapid_api': {
            'name': 'RapidAPI Horse Racing',
            'description': 'Horse racing data via RapidAPI',
            'base_url': 'https://horse-racing.p.rapidapi.com',
            'api_key_required': True,
            'rate_limit': 1000,  # requests per month
            'features': ['races', 'horses', 'results', 'statistics']
        },
        'theracingapi': {
            'name': 'The Racing API',
            'description': 'Comprehensive UK, Ireland & USA horse racing data',
            'base_url': 'https://api.theracingapi.com/v1',
            'api_key_required': False,
            'auth_type': 'basic',  # HTTP Basic Authentication
            'username_required': True,
            'password_required': True,
            'rate_limit': 10000,  # requests per month (varies by plan)
            'features': ['races', 'horses', 'results', 'odds', 'form', 'statistics'],
            'regions': ['GB', 'IE', 'US'],
            'data_points': ['racecards', 'results', 'form', 'odds', 'commentary']
        }
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get complete API configuration"""
        config = cls.DEFAULT_CONFIG.copy()
        
        # Override with environment variables if available
        config.update({
            'timeout': int(os.getenv('API_TIMEOUT', config['timeout'])),
            'max_retries': int(os.getenv('API_MAX_RETRIES', config['max_retries'])),
            'rate_limit_delay': float(os.getenv('API_RATE_LIMIT_DELAY', config['rate_limit_delay'])),
            'cache_duration': int(os.getenv('API_CACHE_DURATION', config['cache_duration'])),
            'max_races_per_request': int(os.getenv('API_MAX_RACES', config['max_races_per_request'])),
            'default_days_ahead': int(os.getenv('API_DEFAULT_DAYS', config['default_days_ahead']))
        })
        
        return config
    
    @classmethod
    def get_provider_config(cls, provider_name: str) -> Dict[str, Any]:
        """Get configuration for specific provider"""
        provider_config = cls.PROVIDERS.get(provider_name, {}).copy()
        
        if provider_config.get('api_key_required'):
            # Try to get API key from environment
            env_key = f"{provider_name.upper()}_API_KEY"
            api_key = os.getenv(env_key)
            provider_config['api_key'] = api_key
        
        return provider_config
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available API providers"""
        return cls.PROVIDERS.copy()
    
    @classmethod
    def is_provider_configured(cls, provider_name: str) -> bool:
        """Check if provider is properly configured"""
        provider_config = cls.get_provider_config(provider_name)
        
        if not provider_config:
            return False
        
        # Check if API key is required and available
        if provider_config.get('api_key_required'):
            return bool(provider_config.get('api_key'))
        
        # Check if HTTP Basic Auth is required and available
        if provider_config.get('auth_type') == 'basic':
            username = os.getenv(f"{provider_name.upper()}_USERNAME")
            password = os.getenv(f"{provider_name.upper()}_PASSWORD")
            return bool(username and password)
        
        return True

# Environment variable examples for .env file
ENV_TEMPLATE = """
# API Configuration
API_TIMEOUT=30
API_MAX_RETRIES=3
API_RATE_LIMIT_DELAY=1.0
API_CACHE_DURATION=300
API_MAX_RACES=50
API_DEFAULT_DAYS=7

# API Keys (replace with actual keys)
SAMPLE_API_KEY=your_sample_api_key_here
ODDS_API_API_KEY=your_odds_api_key_here
RAPID_API_API_KEY=your_rapidapi_key_here
THERACINGAPI_USERNAME=your_theracingapi_username_here
THERACINGAPI_PASSWORD=your_theracingapi_password_here

# Provider Selection
DEFAULT_API_PROVIDER=mock
FALLBACK_API_PROVIDER=sample
"""

def create_env_template(file_path: str = '.env.template'):
    """Create environment template file"""
    with open(file_path, 'w') as f:
        f.write(ENV_TEMPLATE)
    print(f"Environment template created at {file_path}")

if __name__ == "__main__":
    # Print current configuration
    print("Current API Configuration:")
    print(APIConfig.get_config())
    print("\nAvailable Providers:")
    for name, config in APIConfig.get_available_providers().items():
        status = "✓ Configured" if APIConfig.is_provider_configured(name) else "✗ Not configured"
        print(f"  {name}: {config['name']} - {status}")