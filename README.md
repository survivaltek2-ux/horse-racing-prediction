# Horse Racing Prediction System

A comprehensive web application for predicting horse racing outcomes using machine learning algorithms and external API integration.

## 🏇 Features

### Core Functionality
- **Race Management**: Add, view, and manage horse races
- **Horse Database**: Comprehensive horse profiles with performance history
- **Prediction Engine**: ML-powered race outcome predictions
- **Statistics Dashboard**: Performance analytics and insights
- **Prediction History**: Track and analyze past predictions

### API Integration
- **External Data Import**: Fetch race data from multiple API providers
- **Real-time Updates**: Sync odds and race information
- **Provider Management**: Support for multiple racing data sources
- **Mock Testing**: Built-in testing capabilities

### Web Interface
- **Responsive Design**: Modern, mobile-friendly interface
- **Interactive Forms**: Easy data entry and management
- **Real-time Feedback**: Live updates and notifications
- **Navigation**: Intuitive menu system

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Flask
- Required Python packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/horse-racing-prediction.git
   cd horse-racing-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python3 app.py
   ```

4. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:5000`

## 📁 Project Structure

```
APP/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── config/
│   └── api_config.py     # API configuration management
├── models/
│   ├── horse.py          # Horse data model
│   ├── race.py           # Race data model
│   └── prediction.py     # Prediction data model
├── services/
│   └── api_service.py    # API integration service
├── utils/
│   ├── api_client.py     # API client utilities
│   ├── data_processor.py # Data processing utilities
│   └── predictor.py      # ML prediction engine
├── templates/            # HTML templates
├── static/              # CSS, JS, and images
└── data/               # JSON data storage
```

## 🔧 Configuration

### API Setup
1. Copy the environment template:
   ```python
   from config.api_config import create_env_template
   create_env_template()
   ```

2. Edit the `.env` file with your API credentials:
   ```
   # API Configuration
   DEFAULT_PROVIDER=mock
   
   # Sample API
   SAMPLE_API_KEY=your_api_key_here
   SAMPLE_API_URL=https://api.example.com
   
   # The Odds API
   ODDS_API_KEY=your_odds_api_key
   ODDS_API_URL=https://api.the-odds-api.com
   ```

### Data Storage
The application uses JSON files for data persistence:
- `data/races.json` - Race information
- `data/horses.json` - Horse profiles and statistics

## 🎯 Usage

### Adding Races
1. Navigate to "Add Race" in the menu
2. Fill in race details (name, date, location, etc.)
3. Add participating horses
4. Save the race

### Making Predictions
1. Go to "Predict" page
2. Select a race from the dropdown
3. Click "Predict" to get ML-powered predictions
4. View confidence scores and recommended bets

### API Integration
1. Navigate to "Import from API"
2. Select an API provider
3. Test the connection
4. Fetch and import race data
5. Review imported races and horses

### Viewing Statistics
- Access comprehensive analytics via the "Statistics" menu
- View win rates, performance trends, and prediction accuracy
- Analyze horse and jockey performance

## 🤖 Machine Learning

The prediction engine uses various algorithms to analyze:
- Horse performance history
- Jockey statistics
- Track conditions
- Recent form
- Head-to-head comparisons

## 🔌 API Providers

### Supported Providers
- **Mock API**: For testing and development
- **Sample API**: Example external provider
- **The Odds API**: Real-time odds and race data
- **RapidAPI**: Various racing data sources

### Adding New Providers
1. Extend the `BaseAPIClient` class in `utils/api_client.py`
2. Add provider configuration in `config/api_config.py`
3. Update the API service to handle the new provider

## 🧪 Testing

### Mock API Testing
The application includes a built-in mock API for testing:
```python
# Test API connection
from services.api_service import APIService
api_service = APIService()
result = api_service.test_connection('mock')
```

### Running Tests
```bash
# Add your test commands here
python -m pytest tests/
```

## 📊 Data Models

### Race Model
- Race details (name, date, location)
- Participating horses
- Race conditions and track info
- Results and payouts

### Horse Model
- Basic information (name, age, breed)
- Performance statistics
- Racing history
- Trainer and jockey information

### Prediction Model
- Prediction results and confidence
- Historical accuracy tracking
- Model performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Flask framework for the web application
- Machine learning libraries for prediction algorithms
- External API providers for race data
- Open source community for various utilities

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the code comments for implementation details

---

**Note**: This application is for educational and entertainment purposes. Please gamble responsibly and be aware of local laws regarding betting and gambling.