"""
Main entry point for F1 Race Prediction System
"""

import argparse
from src.data_collector import F1DataCollector
from src.preprocessor import F1DataPreprocessor
from src.model import F1PredictionModel, ModelEvaluator
from src.predictor import F1RacePredictor
from src.utils import Visualizer, DataExporter


def collect_data(start_year: int, end_year: int, output_path: str):
    """Collect historical F1 data."""
    print(f"Collecting data from {start_year} to {end_year}...")
    collector = F1DataCollector()
    df = collector.collect_historical_data(start_year, end_year, output_path)
    print(f"Collected {len(df)} race results")


def train_model(data_path: str, model_type: str, output_path: str):
    """Train prediction model."""
    print(f"Training {model_type} model...")

    # Load data
    import pandas as pd
    df = pd.read_csv(data_path)

    # Preprocess
    preprocessor = F1DataPreprocessor()
    df_processed = preprocessor.create_all_features(df)
    X, y = preprocessor.prepare_training_data(df_processed)

    # Train model
    model = F1PredictionModel(model_type=model_type)
    metrics = model.train(X, y)

    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save model
    model.save_model(output_path)

    # Show feature importance
    if model.feature_importance is not None:
        print("\nTop 5 Important Features:")
        print(model.feature_importance.head())


def predict_race(model_path: str, year: int, race: str):
    """Predict race results."""
    print(f"Predicting results for {year} - {race}...")

    predictor = F1RacePredictor(model_path=model_path)
    predictions = predictor.predict_race(year, race)

    print("\nPredicted Results:")
    print(predictions.to_string(index=False))

    # Get podium predictions
    podium = predictor.get_podium_predictions(year, race)
    print(f"\nPredicted Podium: {' | '.join(podium)}")


def compare_models(data_path: str):
    """Compare different model types."""
    print("Comparing models...")

    # Load data
    import pandas as pd
    df = pd.read_csv(data_path)

    # Preprocess
    preprocessor = F1DataPreprocessor()
    df_processed = preprocessor.create_all_features(df)
    X, y = preprocessor.prepare_training_data(df_processed)

    # Compare models
    comparison = ModelEvaluator.compare_models(X, y)
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='F1 Race Prediction System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Collect data command
    collect_parser = subparsers.add_parser('collect', help='Collect historical F1 data')
    collect_parser.add_argument('--start-year', type=int, required=True, help='Start year')
    collect_parser.add_argument('--end-year', type=int, required=True, help='End year')
    collect_parser.add_argument('--output', type=str, default='data/raw/historical_data.csv',
                                help='Output file path')

    # Train model command
    train_parser = subparsers.add_parser('train', help='Train prediction model')
    train_parser.add_argument('--data', type=str, required=True, help='Path to training data')
    train_parser.add_argument('--model-type', type=str, default='random_forest',
                              choices=['random_forest', 'gradient_boosting', 'xgboost'],
                              help='Model type')
    train_parser.add_argument('--output', type=str, default='models/trained/model.pkl',
                              help='Output model path')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict race results')
    predict_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    predict_parser.add_argument('--year', type=int, required=True, help='Season year')
    predict_parser.add_argument('--race', type=str, required=True, help='Race name or round number')

    # Compare models command
    compare_parser = subparsers.add_parser('compare', help='Compare different models')
    compare_parser.add_argument('--data', type=str, required=True, help='Path to training data')

    args = parser.parse_args()

    if args.command == 'collect':
        collect_data(args.start_year, args.end_year, args.output)
    elif args.command == 'train':
        train_model(args.data, args.model_type, args.output)
    elif args.command == 'predict':
        predict_race(args.model, args.year, args.race)
    elif args.command == 'compare':
        compare_models(args.data)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
