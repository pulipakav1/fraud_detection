"""
Baseline Model: Logistic Regression

Provides interpretable baseline with class weights to handle imbalance.
Often kept as fallback in production systems.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from pathlib import Path
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.evaluation.metrics import evaluate_model
from src.evaluation.cost_matrix import CostMatrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns (exclude metadata and target)."""
    exclude = [
        'transaction_id', 'user_id', 'timestamp', 'is_fraud',
        'merchant_category', 'device_id', 'latitude', 'longitude',
        'payment_method'
    ]
    return [col for col in df.columns if col not in exclude]


def train_baseline_model(
    data_path: str = "data/processed/features.csv",
    config_path: str = "config.yaml",
    model_save_path: str = "models/baseline_model.pkl"
):
    """
    Train baseline logistic regression model.
    
    Args:
        data_path: Path to processed features
        config_path: Path to config file
        model_save_path: Path to save trained model
    """
    logger.info("=" * 60)
    logger.info("TRAINING BASELINE MODEL: Logistic Regression")
    logger.info("=" * 60)
    
    # Load config
    config = load_config(config_path)
    model_config = config.get('models', {}).get('baseline', {})
    eval_config = config.get('evaluation', {})
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} transactions")
    logger.info(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    
    # Prepare features
    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")
    
    X = df[feature_cols].fillna(0)
    y = df['is_fraud'].values
    amounts = df['amount'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
        X, y, amounts,
        test_size=config.get('data', {}).get('train_test_split', 0.2),
        random_state=config.get('data', {}).get('random_seed', 42),
        stratify=y
    )
    
    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logger.info("\nTraining Logistic Regression...")
    model = LogisticRegression(
        class_weight=model_config.get('class_weight', 'balanced'),
        max_iter=model_config.get('max_iter', 1000),
        C=model_config.get('C', 1.0),
        random_state=config.get('data', {}).get('random_seed', 42),
        solver='lbfgs',
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    logger.info("✅ Model trained")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    logger.info("\nEvaluating model...")
    cost_matrix_config = eval_config.get('cost_matrix', {})
    metrics = evaluate_model(
        y_test, y_pred, y_proba, amounts_test,
        cost_matrix=cost_matrix_config
    )
    
    # Save model and scaler
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'metrics': metrics,
        'model_type': 'logistic_regression'
    }
    joblib.dump(model_data, model_save_path)
    logger.info(f"\n✅ Model saved to {model_save_path}")
    
    # Feature importance (coefficients)
    logger.info("\nTop 10 Most Important Features (by absolute coefficient):")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    print(feature_importance.head(10).to_string(index=False))
    
    return model, scaler, metrics


if __name__ == "__main__":
    train_baseline_model()
