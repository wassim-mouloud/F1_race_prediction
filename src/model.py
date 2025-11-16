"""
Machine Learning Models for F1 Race Prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from typing import Dict, Any, Optional, Tuple


class F1PredictionModel:
    """
    Machine learning model for predicting F1 race positions.
    """

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the prediction model.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm')
        """
        self.model_type = model_type
        self.model = self._create_model()
        self.is_trained = False
        self.feature_importance = None

    def _create_model(self):
        """Create the specified model type."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        validation: bool = True
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            validation: Whether to perform cross-validation

        Returns:
            Dictionary of evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        # Cross-validation
        if validation:
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=5, scoring='neg_mean_absolute_error'
            )
            metrics['cv_mae'] = -cv_scores.mean()
            metrics['cv_mae_std'] = cv_scores.std()

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted positions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = self.model.predict(X)

        # Round to nearest position and clip to valid range
        predictions = np.clip(np.round(predictions), 1, 20)

        return predictions

    def save_model(self, filepath: str):
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance.

        Returns:
            DataFrame with feature importances
        """
        return self.feature_importance


class ModelEvaluator:
    """
    Evaluate and compare different models.
    """

    @staticmethod
    def compare_models(
        X: pd.DataFrame,
        y: pd.Series,
        model_types: list = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
    ) -> pd.DataFrame:
        """
        Compare multiple model types.

        Args:
            X: Feature matrix
            y: Target variable
            model_types: List of model types to compare

        Returns:
            DataFrame with comparison results
        """
        results = []

        for model_type in model_types:
            try:
                print(f"Training {model_type}...")
                model = F1PredictionModel(model_type=model_type)
                metrics = model.train(X, y)
                metrics['model_type'] = model_type
                results.append(metrics)
            except Exception as e:
                print(f"Error training {model_type}: {e}")

        return pd.DataFrame(results)

    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate prediction quality.

        Args:
            y_true: True positions
            y_pred: Predicted positions

        Returns:
            Dictionary of metrics
        """
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'exact_predictions': np.sum(y_true == y_pred) / len(y_true),
            'within_1_position': np.sum(np.abs(y_true - y_pred) <= 1) / len(y_true),
            'within_3_positions': np.sum(np.abs(y_true - y_pred) <= 3) / len(y_true)
        }
