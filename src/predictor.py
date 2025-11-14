"""
Prediction Pipeline for F1 Races
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .data_collector import F1DataCollector
from .preprocessor import F1DataPreprocessor
from .model import F1PredictionModel


class F1RacePredictor:
    """
    Complete pipeline for predicting F1 race results.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the race predictor.

        Args:
            model_path: Optional path to a pre-trained model
        """
        self.collector = F1DataCollector()
        self.preprocessor = F1DataPreprocessor()
        self.model = F1PredictionModel()

        if model_path and Path(model_path).exists():
            self.model.load_model(model_path)

    def prepare_race_prediction(
        self,
        year: int,
        race: str,
        qualifying_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prepare data for predicting an upcoming race.

        Args:
            year: Season year
            race: Race name or round number
            qualifying_data: Optional qualifying results

        Returns:
            DataFrame ready for prediction
        """
        # Get qualifying session if not provided
        if qualifying_data is None:
            try:
                quali_session = self.collector.get_session_data(year, race, 'Q')
                qualifying_data = quali_session.results
            except Exception as e:
                print(f"Could not fetch qualifying data: {e}")
                return pd.DataFrame()

        # Get historical data for drivers
        # This would need historical performance data
        # For now, we'll use the qualifying data as base

        df = qualifying_data.copy()

        # Add year and race info
        df['Year'] = year
        df['RaceName'] = race

        return df

    def predict_race(
        self,
        year: int,
        race: str,
        qualifying_data: Optional[pd.DataFrame] = None,
        return_probabilities: bool = False
    ) -> pd.DataFrame:
        """
        Predict race results.

        Args:
            year: Season year
            race: Race name or round number
            qualifying_data: Optional qualifying results
            return_probabilities: Whether to return prediction confidence

        Returns:
            DataFrame with predictions
        """
        if not self.model.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare data
        race_data = self.prepare_race_prediction(year, race, qualifying_data)

        if race_data.empty:
            raise ValueError("Could not prepare race data for prediction")

        # Process features
        processed_data = self.preprocessor.create_all_features(race_data)

        # Prepare for prediction
        X, _ = self.preprocessor.prepare_training_data(
            processed_data,
            feature_cols=self.model.feature_importance['feature'].tolist()
            if self.model.feature_importance is not None
            else None
        )

        # Make predictions
        predictions = self.model.predict(X)

        # Create results DataFrame
        results = race_data[['DriverNumber', 'Abbreviation', 'TeamName']].copy()
        results['PredictedPosition'] = predictions.astype(int)
        results = results.sort_values('PredictedPosition')

        return results

    def get_podium_predictions(
        self,
        year: int,
        race: str,
        qualifying_data: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Get predicted podium finishers.

        Args:
            year: Season year
            race: Race name or round number
            qualifying_data: Optional qualifying results

        Returns:
            List of driver abbreviations for predicted top 3
        """
        results = self.predict_race(year, race, qualifying_data)
        top_3 = results.nsmallest(3, 'PredictedPosition')
        return top_3['Abbreviation'].tolist()

    def predict_winner(
        self,
        year: int,
        race: str,
        qualifying_data: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Predict race winner.

        Args:
            year: Season year
            race: Race name or round number
            qualifying_data: Optional qualifying results

        Returns:
            Driver abbreviation of predicted winner
        """
        results = self.predict_race(year, race, qualifying_data)
        winner = results.nsmallest(1, 'PredictedPosition')
        return winner['Abbreviation'].iloc[0]

    def get_driver_prediction(
        self,
        year: int,
        race: str,
        driver: str,
        qualifying_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Get prediction for a specific driver.

        Args:
            year: Season year
            race: Race name or round number
            driver: Driver abbreviation
            qualifying_data: Optional qualifying results

        Returns:
            Dictionary with driver prediction details
        """
        results = self.predict_race(year, race, qualifying_data)
        driver_result = results[results['Abbreviation'] == driver]

        if driver_result.empty:
            return {'error': f'Driver {driver} not found'}

        return {
            'driver': driver,
            'predicted_position': int(driver_result['PredictedPosition'].iloc[0]),
            'team': driver_result['TeamName'].iloc[0]
        }
