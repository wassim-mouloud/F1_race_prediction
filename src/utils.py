"""
Utility Functions for F1 Prediction Project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional
import numpy as np


class Visualizer:
    """
    Create visualizations for F1 data and predictions.
    """

    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10):
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with features and importance scores
            top_n: Number of top features to display
        """
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(top_n)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Plot predicted vs actual positions.

        Args:
            y_true: True positions
            y_pred: Predicted positions
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([0, 20], [0, 20], 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Position')
        plt.ylabel('Predicted Position')
        plt.title('Predicted vs Actual Positions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_driver_performance(df: pd.DataFrame, driver: str):
        """
        Plot driver performance over time.

        Args:
            df: DataFrame with race results
            driver: Driver abbreviation
        """
        driver_data = df[df['Abbreviation'] == driver].copy()

        if driver_data.empty:
            print(f"No data found for driver {driver}")
            return

        driver_data = driver_data.sort_values(['Year', 'Round'])

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(driver_data)), driver_data['Position'], marker='o')
        plt.gca().invert_yaxis()
        plt.xlabel('Race Number')
        plt.ylabel('Position')
        plt.title(f'{driver} - Race Positions Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_race_prediction(prediction_df: pd.DataFrame):
        """
        Create interactive visualization of race predictions.

        Args:
            prediction_df: DataFrame with race predictions
        """
        fig = go.Figure(data=[
            go.Bar(
                x=prediction_df['Abbreviation'],
                y=prediction_df['PredictedPosition'],
                text=prediction_df['TeamName'],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title='Predicted Race Positions',
            xaxis_title='Driver',
            yaxis_title='Predicted Position',
            yaxis=dict(autorange='reversed')
        )

        fig.show()

    @staticmethod
    def plot_model_comparison(comparison_df: pd.DataFrame):
        """
        Compare different models.

        Args:
            comparison_df: DataFrame with model comparison metrics
        """
        fig = go.Figure()

        metrics = ['mae', 'rmse', 'r2']

        for metric in metrics:
            if metric in comparison_df.columns:
                fig.add_trace(go.Bar(
                    name=metric.upper(),
                    x=comparison_df['model_type'],
                    y=comparison_df[metric]
                ))

        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model Type',
            yaxis_title='Score',
            barmode='group'
        )

        fig.show()


class DataExporter:
    """
    Export data and predictions to various formats.
    """

    @staticmethod
    def export_predictions(
        predictions: pd.DataFrame,
        filepath: str,
        format: str = 'csv'
    ):
        """
        Export predictions to file.

        Args:
            predictions: DataFrame with predictions
            filepath: Output file path
            format: Output format ('csv', 'json', 'excel')
        """
        if format == 'csv':
            predictions.to_csv(filepath, index=False)
        elif format == 'json':
            predictions.to_json(filepath, orient='records', indent=2)
        elif format == 'excel':
            predictions.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Predictions exported to {filepath}")

    @staticmethod
    def generate_report(
        predictions: pd.DataFrame,
        metrics: dict,
        output_path: str
    ):
        """
        Generate a prediction report.

        Args:
            predictions: DataFrame with predictions
            metrics: Dictionary of model metrics
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("# F1 Race Prediction Report\n\n")

            f.write("## Model Metrics\n")
            for metric, value in metrics.items():
                f.write(f"- {metric}: {value:.4f}\n")

            f.write("\n## Top 10 Predictions\n")
            top_10 = predictions.nsmallest(10, 'PredictedPosition')
            f.write(top_10.to_markdown(index=False))

        print(f"Report generated at {output_path}")


def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid, False otherwise
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        print(f"Missing required columns: {missing}")
        return False
    return True


def get_current_season() -> int:
    """
    Get current F1 season year.

    Returns:
        Current year
    """
    from datetime import datetime
    return datetime.now().year
