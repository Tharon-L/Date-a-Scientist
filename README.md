# Dating Profile Analyzer 🔍💕

A comprehensive machine learning system for analyzing dating profiles and predicting user attractiveness and compatibility. This project leverages advanced NLP, feature engineering, and ensemble methods to provide insights into what makes dating profiles successful.

## 🎯 Project Overview

The Dating Profile Analyzer is a sophisticated data science project that:
- **Predicts profile attractiveness** using composite scoring and machine learning
- **Matches compatible users** based on multiple compatibility factors
- **Provides actionable insights** for profile optimization
- **Analyzes text sentiment and complexity** from profile essays

## 📊 Dataset

- **Size**: 59,946 dating profiles with 31 original features
- **Features**: Demographics, lifestyle choices, essays, preferences
- **Generated Features**: 18 additional engineered features
- **Target Variables**: 
  - Attractiveness: 30% high-attractiveness profiles (17,984 users)
  - Compatibility: Generated 2,000 user pairs with 31% compatibility rate

## 🔧 Key Features

### Advanced Feature Engineering
- **Profile Completeness**: Measures how complete a user's profile is
- **Text Analysis**: Essay length, word count, reading complexity, sentiment analysis
- **Personality Indicators**: Exclamation marks, questions, emoji usage
- **Lifestyle Consistency**: Completeness across lifestyle categories
- **Age Demographics**: Categorical age groupings

### Machine Learning Models
- **Attractiveness Prediction**: Random Forest with hyperparameter tuning
  - Features: TF-IDF text vectors, categorical variables, numerical features
  - Performance: ROC-AUC optimized with cross-validation
- **Compatibility Matching**: Multi-model comparison (RF, GB, Logistic Regression)
  - Features: Age differences, shared characteristics, attractiveness matching

### Natural Language Processing
- **Sentiment Analysis**: VADER sentiment intensity analyzer
- **Text Complexity**: Flesch reading ease scores
- **Content Analysis**: Essay length, vocabulary richness
- **Engagement Metrics**: Interactive elements in text

## 🚀 Installation

### Prerequisites
```bash
python >= 3.7
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
nltk >= 3.6
textstat >= 0.7.0
```

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/dating-profile-analyzer.git
cd dating-profile-analyzer

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

## 📈 Usage

### Basic Usage
```python
from dating_analyzer import DatingProfileAnalyzer

# Initialize analyzer
analyzer = DatingProfileAnalyzer()

# Load and process data
df = analyzer.load_data('profiles.csv')
analyzer.original_columns = list(df.columns)

# Feature engineering
analyzer.create_advanced_features()
analyzer.create_attractiveness_target()
analyzer.create_compatibility_pairs(n_pairs=2000)

# Build models
analyzer.build_attractiveness_model()
analyzer.build_compatibility_model()

# Generate insights
insights = analyzer.generate_insights()
print(insights)
```

### Making Predictions
```python
# Predict attractiveness for a new profile
profile_data = {
    'all_essays': 'I love hiking and reading books...',
    'age_group': '26-30',
    'body_type': 'average',
    # ... other features
}

result = analyzer.predict_attractiveness(profile_data)
print(f"High attractiveness probability: {result['probability_high']:.3f}")

# Predict compatibility between users
compatibility = analyzer.predict_compatibility(user1_id=100, user2_id=200)
print(f"Compatibility probability: {compatibility['probability_compatible']:.3f}")
```

## 📊 Model Performance

### Attractiveness Prediction Model
- **Algorithm**: Random Forest with Grid Search optimization
- **Features**: 500 TF-IDF features + categorical + numerical variables
- **Optimization**: ROC-AUC maximization with 3-fold cross-validation
- **Memory Efficient**: Reduced feature dimensionality for large datasets

### Compatibility Matching Model
- **Algorithm**: Best performing among Random Forest, Gradient Boosting, Logistic Regression
- **Features**: Age differences, shared characteristics, attractiveness compatibility
- **Evaluation**: F1-score optimization with stratified sampling
- **Feature Importance**: Automated ranking of compatibility factors

## 🎨 Visualizations

The system generates comprehensive visualizations:
- **ROC Curves**: Model performance assessment
- **Feature Importance**: Top predictive factors
- **Distribution Plots**: Attractiveness and compatibility patterns
- **Confusion Matrices**: Classification performance
- **Correlation Analysis**: Age vs. compatibility relationships

## 🔍 Key Insights

### Attractiveness Factors
1. **Profile Completeness** (25% weight): Complete profiles perform better
2. **Essay Quality** (35% weight): Longer, well-written essays increase attractiveness
3. **Engagement** (20% weight): Positive sentiment and interactive elements
4. **Lifestyle Consistency** (20% weight): Consistent lifestyle information

### Compatibility Factors
- **Age Compatibility**: Users within 10 years show higher compatibility
- **Shared Interests**: Common lifestyle choices improve matching
- **Attractiveness Balance**: Similar attractiveness scores predict compatibility
- **Orientation Matching**: Critical for successful matches

## 🛠️ Technical Architecture

### Data Pipeline
1. **Data Loading**: CSV ingestion with missing value analysis
2. **Feature Engineering**: 18 derived features from raw data
3. **Text Processing**: NLP pipeline with sentiment and complexity analysis
4. **Target Creation**: Composite scoring for attractiveness labeling
5. **Pair Generation**: Stratified sampling for compatibility modeling

### Model Pipeline
1. **Preprocessing**: Column transformers for different data types
2. **Text Vectorization**: TF-IDF with n-grams and stop word removal
3. **Categorical Encoding**: One-hot encoding with missing value handling
4. **Numerical Scaling**: StandardScaler with median imputation
5. **Model Training**: Ensemble methods with hyperparameter optimization

### Memory Optimization
- Limited TF-IDF features (500 max) for memory efficiency
- Stratified sampling for pair generation
- Single-threaded grid search to prevent memory issues
- Chunked processing for large datasets

## 📋 Project Structure

```
dating-profile-analyzer/
│
├── dating_analyzer.py          # Main analyzer class
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── data/
│   └── profiles.csv           # Dating profile dataset
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── feature_engineering.ipynb
│   └── model_evaluation.ipynb
├── models/
│   ├── attractiveness_model.pkl
│   └── compatibility_model.pkl
├── visualizations/
│   └── model_performance_plots.png
└── results/
    └── insights_summary.json
```

## 🔬 Research Applications

### Academic Use Cases
- **Social Psychology**: Understanding attraction patterns in digital dating
- **NLP Research**: Text analysis in social contexts
- **Recommendation Systems**: Compatibility matching algorithms
- **Behavioral Analysis**: Profile optimization strategies

### Industry Applications
- **Dating Platforms**: Profile recommendation engines
- **Marketing**: Demographic targeting and messaging
- **Social Media**: Content engagement prediction
- **User Experience**: Profile completion optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure backward compatibility

## 🙏 Acknowledgments

- **OkCupid Dataset**: Original dataset source
- **NLTK Team**: Natural language processing tools
- **Scikit-learn**: Machine learning framework
- **Dating Platform Research**: Inspiration from academic studies

## 📞 Contact

- **Author**: THARON Loudiern
- **Email**: loudiern.tharon.pro@gmail.com

## 🚀 Future Enhancements

### Planned Features
- [ ] Deep learning models for text analysis
- [ ] Image analysis for profile photos
- [ ] Real-time recommendation API
- [ ] A/B testing framework
- [ ] Multi-language support
- [ ] Demographic bias analysis

### Technical Improvements
- [ ] Distributed computing for large datasets
- [ ] Model interpretability with SHAP
- [ ] Automated model retraining pipeline
- [ ] REST API with Flask/FastAPI
- [ ] Docker containerization
- [ ] Cloud deployment options

---

*Made with ❤️ for better understanding of digital relationships*
