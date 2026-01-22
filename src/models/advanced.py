"""
Advanced Model: XGBoost

Industry standard for fraud detection.
Handles non-linear patterns and feature interactions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
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


def train_advanced_model(
    data_path: str = "data/processed/features.csv",
    config_path: str = "config.yaml",
    model_save_path: str = "models/advanced_model.pkl"
):
    """
    Train advanced XGBoost model.
    
    Args:
        data_path: Path to processed features
        config_path: Path to config file
        model_save_path: Path to save trained model
    """
    logger.info("=" * 60)
    logger.info("TRAINING ADVANCED MODEL: XGBoost")
    logger.info("=" * 60)
    
    # Load config
    config = load_config(config_path)
    model_config = config.get('models', {}).get('advanced', {})
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
    
    # Calculate scale_pos_weight for class imbalance
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    scale_pos_weight = model_config.get('scale_pos_weight', fraud_ratio)
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Train model
    logger.info("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=model_config.get('n_estimators', 200),
        max_depth=model_config.get('max_depth', 6),
        learning_rate=model_config.get('learning_rate', 0.1),
        subsample=model_config.get('subsample', 0.8),
        colsample_bytree=model_config.get('colsample_bytree', 0.8),
        scale_pos_weight=scale_pos_weight,
        random_state=config.get('data', {}).get('random_seed', 42),
        eval_metric='aucpr',  # Precision-Recall AUC (fraud favorite)
        n_jobs=-1,
        tree_method='hist'  # Faster training
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    logger.info("✅ Model trained")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    logger.info("\nEvaluating model...")
    cost_matrix_config = eval_config.get('cost_matrix', {})
    metrics = evaluate_model(
        y_test, y_pred, y_proba, amounts_test,
        cost_matrix=cost_matrix_config
    )
    
    # Save model
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        'model': model,
        'feature_columns': feature_cols,
        'metrics': metrics,
        'model_type': 'xgboost'
    }
    joblib.dump(model_data, model_save_path)
    logger.info(f"\n✅ Model saved to {model_save_path}")
    
    # Feature importance
    logger.info("\nTop 20 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(20).to_string(index=False))
    
    return model, metrics


if __name__ == "__main__":
    train_advanced_model()
