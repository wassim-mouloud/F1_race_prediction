# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F1 Race Prediction System - A machine learning pipeline for predicting Formula 1 race results using the FastF1 API. The system collects historical F1 data, engineers features from driver/team performance, and trains ML models (Random Forest, Gradient Boosting, XGBoost, LightGBM) to predict race positions.

## Commands

### Environment Setup
```bash
# Activate virtual environment (REQUIRED before any Python commands)
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt
```

### Data Collection
```bash
# Collect historical data from FastF1 API
python main.py collect --start-year 2020 --end-year 2023 --output data/raw/historical_data.csv

# Test API connectivity
python test_api.py
```

### Model Training
```bash
# Train a model (available types: random_forest, gradient_boosting, xgboost, lightgbm)
python main.py train --data data/raw/historical_data.csv --model-type xgboost --output models/trained/model.pkl

# Compare all model types
python main.py compare --data data/raw/historical_data.csv
```

### Predictions
```bash
# Predict race results
python main.py predict --model models/trained/model.pkl --year 2024 --race "Monaco"
```

### Development
```bash
# Launch Jupyter for exploratory analysis
jupyter notebook

# No linting/testing infrastructure is currently set up
```

## Architecture

### Data Flow Pipeline

**Collection → Preprocessing → Feature Engineering → Model Training → Prediction**

1. **Data Collection** ([src/data_collector.py](src/data_collector.py))
   - `F1DataCollector` interfaces with FastF1 API
   - Caches data in `./cache` directory (managed by FastF1)
   - `collect_historical_data()` iterates through seasons/races, handling errors gracefully
   - Returns raw session results with metadata (Year, RaceName, Round)

2. **Preprocessing** ([src/preprocessor.py](src/preprocessor.py))
   - `F1DataPreprocessor` creates features from raw data
   - Feature generation is sequential: driver features → constructor features → qualifying features → encoding
   - **Important**: Features rely on historical data ordering (sorted by DriverNumber, Year, Round)
   - `prepare_training_data()` returns (X, y) tuple after NaN removal

3. **Feature Engineering** ([src/preprocessor.py](src/preprocessor.py))
   - **Driver features**: Rolling averages (5/10 races), cumulative podium/wins counts
   - **Constructor features**: Team average position, cumulative points
   - **Qualifying features**: Grid position, positions gained/lost
   - **Encoding**: Categorical encoding for DriverNumber and TeamName
   - Default feature set for training is defined in `prepare_training_data()`

4. **Model Training** ([src/model.py](src/model.py))
   - `F1PredictionModel` supports 4 model types via `model_type` parameter
   - Uses sklearn/xgboost/lightgbm with hardcoded hyperparameters
   - `train()` performs train/test split (80/20), 5-fold CV, returns metrics dict
   - Predictions are clipped to [1, 20] and rounded to integers
   - Models are persisted with joblib

5. **Prediction Pipeline** ([src/predictor.py](src/predictor.py))
   - `F1RacePredictor` orchestrates end-to-end prediction
   - Fetches qualifying data if not provided
   - **Limitation**: Prediction quality depends on having historical features for drivers
   - Returns sorted DataFrame with PredictedPosition

### Key Architectural Patterns

- **State Management**: Models track `is_trained` state; preprocessor stores `feature_columns`
- **Error Handling**: Data collection uses try/except per race; continues on failure
- **Feature Consistency**: Preprocessor stores feature_columns to ensure train/predict consistency
- **Model Persistence**: Uses joblib for saving/loading trained models

### Module Dependencies

```
main.py → all src modules (orchestration)
predictor.py → data_collector + preprocessor + model (full pipeline)
model.py → independent (sklearn/xgboost/lightgbm only)
preprocessor.py → independent (pandas/numpy only)
data_collector.py → independent (fastf1 only)
utils.py → independent (visualization utilities)
```

## Important Implementation Details

### FastF1 API Caching
- FastF1 automatically caches downloaded data in `./cache` directory
- Cache is persistent across runs (speeds up subsequent data requests)
- Cache directory is created by `F1DataCollector.__init__()`

### Model Type Selection
The `lightgbm` model type is available in code but commented out in requirements.txt due to system dependency issues (requires libomp). If enabling:
1. Install system dependencies first (e.g., `brew install libomp` on macOS)
2. Uncomment `lightgbm==4.3.0` in requirements.txt
3. Run `pip install -r requirements.txt`

### Feature Engineering Constraints
- Rolling averages require at least 1 race of history (min_periods=1)
- Cumulative features (podiums, wins) start at 0 for first race
- Features are calculated per driver, so new drivers without history will have limited features
- Grid position uses qualifying data; missing values are coerced to NaN

### Prediction Limitations
- Predictor expects historical data for feature generation
- For new drivers or first race of season, features may be incomplete
- Qualifying data is required for predictions (fetched automatically if not provided)

### Data Validation
- `prepare_training_data()` removes all rows with NaN in features or target
- `clean_data()` drops rows missing Position or DriverNumber
- Categorical encoding uses `pd.Categorical().codes` (assigns -1 to NaN)

## File Organization

```
src/
├── data_collector.py   # FastF1 API interface
├── preprocessor.py     # Feature engineering
├── model.py           # ML models + evaluation
├── predictor.py       # End-to-end prediction pipeline
└── utils.py           # Visualization + export utilities

data/
├── raw/               # Raw CSV from API (gitignored)
└── processed/         # Processed CSV (gitignored)

models/
└── trained/           # Saved .pkl models (gitignored)

cache/                 # FastF1 cache (gitignored)
notebooks/             # Jupyter notebooks for exploration
tests/                 # Empty (no test infrastructure)
```

## Python Version & Dependencies

- **Python 3.12+** required
- Core dependencies: fastf1, pandas, numpy, scikit-learn, xgboost
- Visualization: matplotlib, seaborn, plotly
- Virtual environment is in `venv/` (already created, activate before use)

## Development Notes

- No testing framework is set up (tests/ directory is nearly empty)
- No linting/formatting configuration
- Comments in code are in French (e.g., test_api.py)
- Main CLI uses argparse subcommands (collect, train, predict, compare)
- Jupyter notebooks are referenced but may not exist yet (01_data_exploration.ipynb, etc.)
