# 🏇 Horse Racing Prediction App

A comprehensive machine learning-powered horse racing prediction system with both web application and static HTML interfaces.

## 🌟 Features

### 🤖 Machine Learning Predictions
- **Multiple ML Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM, Neural Networks, and Extra Trees
- **Advanced Feature Engineering**: Horse performance metrics, jockey/trainer analysis, track conditions
- **Probability Calibration**: Accurate win/place probability predictions
- **Model Performance**: Best model achieves 68.3% AUC for win predictions

### 📊 Comprehensive Analytics
- **Horse Performance Tracking**: Win rates, earnings, recent form analysis
- **Race Analysis**: Track conditions, distance preferences, historical performance
- **Statistical Insights**: Feature importance analysis and prediction confidence scores
- **Training Data Generation**: Synthetic data generation for model training

### 🎯 Prediction Capabilities
- **Win Predictions**: Probability of horse winning the race
- **Place Predictions**: Probability of horse finishing in top 3
- **Position Forecasting**: Expected finishing position
- **Confidence Scoring**: Model certainty indicators

### 🖥️ User Interfaces
- **Flask Web Application**: Full-featured web interface with user authentication
- **Static HTML Version**: GitHub Pages compatible interface
- **Admin Dashboard**: User management and system administration
- **API Integration**: External data source connectivity

## 🚀 Live Demo

**GitHub Pages Demo**: [https://yourusername.github.io/HorseRacingPrediction](https://yourusername.github.io/HorseRacingPrediction)

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **SQLAlchemy** - Database ORM
- **Firebase** - Cloud database and authentication

### Machine Learning
- **scikit-learn** - ML algorithms and preprocessing
- **NumPy & Pandas** - Data manipulation
- **Joblib** - Model serialization

### Frontend
- **Bootstrap 5** - Responsive UI framework
- **JavaScript** - Interactive functionality
- **Chart.js** - Data visualization
- **Font Awesome** - Icons

### Deployment
- **GitHub Pages** - Static site hosting
- **Heroku/Railway** - Web app deployment options

## 📁 Project Structure

```
HorseRacingPrediction/
├── app.py                          # Main Flask application
├── models/                         # Trained ML models
│   ├── enhanced_*.joblib          # Optimized models with hyperparameter tuning
│   ├── feature_importances.pkl   # Feature importance analysis
│   └── model_performance.pkl     # Performance metrics
├── data/                          # Training and sample data
│   ├── horses.json               # Horse database
│   ├── races.json                # Race database
│   └── training_*.json           # Generated training data
├── static-html/                   # GitHub Pages deployment
│   ├── index.html                # Main landing page
│   ├── css/style.css             # Custom styling
│   ├── js/app.js                 # Frontend logic
│   └── data/                     # Sample data for demo
├── templates/                     # Flask templates
├── utils/                         # Utility modules
│   ├── ai_predictor.py           # ML prediction engine
│   ├── data_processor.py         # Data preprocessing
│   └── predictor.py              # Legacy prediction logic
└── scripts/                       # Deployment and utility scripts
```

## 🔧 Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/HorseRacingPrediction.git
   cd HorseRacingPrediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Generate training data**
   ```bash
   python generate_training_data.py
   python data_preprocessing.py
   ```

5. **Train ML models**
   ```bash
   python enhanced_training_script.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

### GitHub Pages Deployment

The static HTML version is automatically deployed to GitHub Pages from the `static-html` directory.

1. **Enable GitHub Pages** in repository settings
2. **Set source** to `/ (root)` or `static-html` folder
3. **Access your site** at `https://yourusername.github.io/HorseRacingPrediction`

## 🤖 Machine Learning Pipeline

### 1. Data Generation
```bash
python generate_training_data.py  # Creates synthetic training data
```

### 2. Data Preprocessing
```bash
python data_preprocessing.py       # Feature engineering and data preparation
```

### 3. Model Training
```bash
python enhanced_training_script.py # Trains multiple models with hyperparameter tuning
```

### 4. Model Evaluation
```bash
python view_training_results.py    # View comprehensive results
```

## 📊 Model Performance

| Model | Win Prediction AUC | Place Prediction AUC | Accuracy |
|-------|-------------------|---------------------|----------|
| **Logistic Regression** | **0.683** | **0.519** | **92.9%** |
| Gradient Boosting | 0.578 | 0.479 | 92.9% |
| SVM | 0.525 | 0.479 | 92.9% |
| Random Forest | 0.520 | 0.483 | 92.9% |
| Extra Trees | 0.397 | 0.492 | 92.9% |
| Neural Network | 0.387 | 0.424 | 92.9% |

### Key Features (by importance)
1. **Horse Earnings** (15.5%)
2. **Horse Win Rate** (10.7%)
3. **Days Since Last Race** (9.4%)
4. **Recent Average Earnings** (7.3%)
5. **Horse Place Rate** (5.9%)

## 🔮 Making Predictions

### Web Interface
1. Navigate to the prediction page
2. Select horses and race conditions
3. View probability predictions and confidence scores

### API Usage
```python
from utils.ai_predictor import AIPredictor

predictor = AIPredictor()
prediction = predictor.predict_race_outcome(race_data, horses_data)
print(f"Win probabilities: {prediction['win_probabilities']}")
```

## 🛡️ Security Features

- **Firebase Authentication** - Secure user management
- **Environment Variables** - Sensitive data protection
- **Input Validation** - SQL injection prevention
- **CSRF Protection** - Cross-site request forgery protection

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **scikit-learn** community for excellent ML tools
- **Bootstrap** team for responsive UI components
- **Flask** developers for the lightweight web framework
- **Horse racing data providers** for inspiration

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/HorseRacingPrediction/issues)
- **Documentation**: [Wiki](https://github.com/yourusername/HorseRacingPrediction/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/HorseRacingPrediction/discussions)

---

**⭐ Star this repository if you find it helpful!**

Built with ❤️ for horse racing enthusiasts and data science practitioners.