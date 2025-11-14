# F1 Race Prediction System

A machine learning system for predicting Formula 1 race results using the FastF1 API.

## Project Structure

```
F1/
├── data/
│   ├── raw/              # Raw data from FastF1 API
│   └── processed/        # Cleaned and processed data
├── models/
│   └── trained/          # Saved trained models
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_collector.py     # FastF1 API data collection
│   ├── preprocessor.py       # Data cleaning & feature engineering
│   ├── model.py              # ML model definitions
│   ├── predictor.py          # Prediction logic
│   └── utils.py              # Helper functions & visualization
├── tests/
├── requirements.txt
├── .gitignore
├── README.md
└── main.py               # CLI interface
```

## Features

- **Data Collection**: Fetch historical F1 data using FastF1 API
- **Feature Engineering**: Create meaningful features from race data
  - Driver performance metrics (rolling averages, podiums, wins)
  - Constructor/team performance
  - Qualifying position effects
  - Historical trends
- **Multiple ML Models**: Support for:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
- **Prediction Pipeline**: End-to-end pipeline for race predictions
- **Visualizations**: Interactive plots and charts for analysis
- **CLI Interface**: Easy-to-use command-line tools

## Installation

1. **Clone the repository**
```bash
cd /path/to/F1
```

2. **Create virtual environment**
```bash
python3 -m venv venv
```

3. **Activate virtual environment**
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect Historical Data

```bash
python main.py collect --start-year 2020 --end-year 2023 --output data/raw/historical_data.csv
```

### 2. Train a Model

```bash
python main.py train --data data/raw/historical_data.csv --model-type xgboost --output models/trained/model.pkl
```

Model types available:
- `random_forest`
- `gradient_boosting`
- `xgboost`
- `lightgbm`

### 3. Predict Race Results

```bash
python main.py predict --model models/trained/model.pkl --year 2024 --race "Monaco"
```

### 4. Compare Models

```bash
python main.py compare --data data/raw/historical_data.csv
```

## Python API Usage

### Collecting Data

```python
from src.data_collector import F1DataCollector

collector = F1DataCollector()

# Get season schedule
schedule = collector.get_season_schedule(2024)

# Get race session data
session = collector.get_session_data(2024, 'Monaco', 'R')

# Collect historical data
df = collector.collect_historical_data(2020, 2023, 'data/raw/historical.csv')
```

### Training a Model

```python
from src.preprocessor import F1DataPreprocessor
from src.model import F1PredictionModel
import pandas as pd

# Load data
df = pd.read_csv('data/raw/historical_data.csv')

# Preprocess
preprocessor = F1DataPreprocessor()
df_processed = preprocessor.create_all_features(df)
X, y = preprocessor.prepare_training_data(df_processed)

# Train model
model = F1PredictionModel(model_type='xgboost')
metrics = model.train(X, y)

# Save model
model.save_model('models/trained/my_model.pkl')
```

### Making Predictions

```python
from src.predictor import F1RacePredictor

predictor = F1RacePredictor(model_path='models/trained/my_model.pkl')

# Predict race results
predictions = predictor.predict_race(2024, 'Monaco')

# Get predicted podium
podium = predictor.get_podium_predictions(2024, 'Monaco')
print(f"Predicted podium: {podium}")

# Get predicted winner
winner = predictor.predict_winner(2024, 'Monaco')
print(f"Predicted winner: {winner}")
```

### Visualization

```python
from src.utils import Visualizer

viz = Visualizer()

# Plot feature importance
viz.plot_feature_importance(model.feature_importance)

# Plot predictions vs actual
viz.plot_prediction_vs_actual(y_test, y_pred)

# Plot driver performance
viz.plot_driver_performance(df, 'VER')

# Interactive race prediction plot
viz.plot_race_prediction(predictions)
```

## Jupyter Notebooks

Explore the data and models using the provided notebooks in the `notebooks/` directory:

1. `01_data_exploration.ipynb` - Explore historical F1 data
2. `02_feature_engineering.ipynb` - Create and analyze features
3. `03_model_training.ipynb` - Train and evaluate models

To start Jupyter:
```bash
jupyter notebook
```

## Model Performance Metrics

The system evaluates models using:
- **MAE** (Mean Absolute Error): Average position difference
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R²** (R-squared): Proportion of variance explained
- **Exact Predictions**: Percentage of exactly correct predictions
- **Within N Positions**: Percentage within N positions of actual

## Development Roadmap

### Phase 1: Core Functionality (Current)
- [x] Data collection from FastF1
- [x] Basic feature engineering
- [x] ML model training
- [x] Prediction pipeline

### Phase 2: Enhanced Features
- [ ] Weather data integration
- [ ] Tire strategy features
- [ ] Safety car probability
- [ ] Driver consistency metrics
- [ ] Track-specific features

### Phase 3: Advanced Models
- [ ] Neural networks
- [ ] Ensemble methods
- [ ] Time series models
- [ ] Real-time prediction updates

### Phase 4: Production
- [ ] Web API
- [ ] Web dashboard
- [ ] Automated data updates
- [ ] Prediction confidence intervals

## Data Sources

- **FastF1 API**: Official F1 timing data
- Historical race results (2018-present)
- Lap times, telemetry, weather conditions

## Requirements

- Python 3.12+
- FastF1 3.4.0
- pandas, numpy
- scikit-learn
- XGBoost, LightGBM
- matplotlib, seaborn, plotly
- Jupyter

## Contributing

Contributions are welcome! Areas for improvement:
- Additional feature engineering
- New model types
- Better visualization
- Documentation
- Unit tests

## License

This project is for educational and research purposes.

## Acknowledgments

- FastF1 library for providing F1 data access
- Formula 1 for the amazing sport

## Support

For issues or questions, please open an issue on the repository.

---

**Note**: This is a prediction system for entertainment and analysis purposes. Race outcomes depend on many unpredictable factors!
