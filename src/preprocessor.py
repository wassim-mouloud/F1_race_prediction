"""
Data Preprocessing and Feature Engineering Module
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class F1DataPreprocessor:
    """
    Preprocesses F1 data and creates features for machine learning.
    """

    def __init__(self):
        """Initialize the preprocessor."""
        self.feature_columns = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        # Remove rows with missing critical values
        df_clean = df_clean.dropna(subset=['Position', 'DriverNumber'])

        # Convert data types
        if 'Position' in df_clean.columns:
            df_clean['Position'] = pd.to_numeric(df_clean['Position'], errors='coerce')

        return df_clean

    def create_driver_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create driver-specific features.

        Args:
            df: DataFrame with race data

        Returns:
            DataFrame with driver features
        """
        df_features = df.copy()

        # Sort by driver and race date
        df_features = df_features.sort_values(['DriverNumber', 'Year', 'Round'])

        # Calculate rolling averages for each driver
        df_features['avg_position_last_5'] = df_features.groupby('DriverNumber')['Position'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )

        df_features['avg_position_last_10'] = df_features.groupby('DriverNumber')['Position'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )

        # Podium count
        df_features['podium_count'] = df_features.groupby('DriverNumber')['Position'].transform(
            lambda x: (x <= 3).cumsum()
        )

        # Wins count
        df_features['wins_count'] = df_features.groupby('DriverNumber')['Position'].transform(
            lambda x: (x == 1).cumsum()
        )

        return df_features

    def create_constructor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create constructor (team) specific features.

        Args:
            df: DataFrame with race data

        Returns:
            DataFrame with constructor features
        """
        df_features = df.copy()

        if 'TeamName' in df_features.columns:
            # Constructor performance metrics
            df_features['constructor_avg_position'] = df_features.groupby(['TeamName', 'Year', 'Round'])['Position'].transform('mean')

            # Constructor points in season
            if 'Points' in df_features.columns:
                df_features['constructor_total_points'] = df_features.groupby(['TeamName', 'Year'])['Points'].transform('cumsum')

        return df_features

    def create_qualifying_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create qualifying-related features.

        Args:
            df: DataFrame with race data

        Returns:
            DataFrame with qualifying features
        """
        df_features = df.copy()

        if 'GridPosition' in df_features.columns:
            df_features['grid_position'] = pd.to_numeric(df_features['GridPosition'], errors='coerce')

            # Position change from grid
            df_features['positions_gained'] = df_features['grid_position'] - df_features['Position']

        return df_features

    def encode_categorical_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical features.

        Args:
            df: DataFrame
            columns: List of column names to encode

        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()

        for col in columns:
            if col in df_encoded.columns:
                df_encoded[f'{col}_encoded'] = pd.Categorical(df_encoded[col]).codes

        return df_encoded

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for model training.

        Args:
            df: Raw DataFrame

        Returns:
            DataFrame with all features
        """
        # Clean data
        df_processed = self.clean_data(df)

        # Create features
        df_processed = self.create_driver_features(df_processed)
        df_processed = self.create_constructor_features(df_processed)
        df_processed = self.create_qualifying_features(df_processed)

        # Encode categorical variables
        categorical_cols = ['DriverNumber', 'TeamName'] if 'TeamName' in df_processed.columns else ['DriverNumber']
        df_processed = self.encode_categorical_features(df_processed, categorical_cols)

        return df_processed

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'Position',
        feature_cols: Optional[List[str]] = None
    ) -> tuple:
        """
        Prepare data for model training.

        Args:
            df: Processed DataFrame
            target_col: Target variable column name
            feature_cols: List of feature column names

        Returns:
            Tuple of (X, y) for training
        """
        if feature_cols is None:
            # Default feature columns
            feature_cols = [
                'avg_position_last_5',
                'avg_position_last_10',
                'podium_count',
                'wins_count',
                'grid_position',
                'DriverNumber_encoded'
            ]

            if 'TeamName_encoded' in df.columns:
                feature_cols.append('TeamName_encoded')

        # Filter to only include available columns
        feature_cols = [col for col in feature_cols if col in df.columns]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Remove rows with NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        self.feature_columns = feature_cols

        return X, y
