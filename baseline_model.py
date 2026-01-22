# Baseline model: Logistic Regression
# Simple and interpretable, good for comparison

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from pathlib import Path
import logging

from evaluation_metrics import evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame):
    # Get feature columns, exclude metadata
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
    config = load_config(config_path)
    model_config = config.get('models', {}).get('baseline', {})
    eval_config = config.get('evaluation', {})
    
    df = pd.read_csv(data_path)
    feature_cols = get_feature_columns(df)
    
    X = df[feature_cols].fillna(0)
    y = df['is_fraud'].values
    amounts = df['amount'].values
    
    X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
        X, y, amounts,
        test_size=config.get('data', {}).get('train_test_split', 0.2),
        random_state=config.get('data', {}).get('random_seed', 42),
        stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(
        class_weight=model_config.get('class_weight', 'balanced'),
        max_iter=model_config.get('max_iter', 1000),
        C=model_config.get('C', 1.0),
        random_state=config.get('data', {}).get('random_seed', 42),
        solver='lbfgs',
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    cost_matrix_config = eval_config.get('cost_matrix', {})
    metrics = evaluate_model(
        y_test, y_pred, y_proba, amounts_test,
        cost_matrix=cost_matrix_config
    )
    
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'metrics': metrics,
        'model_type': 'logistic_regression'
    }
    joblib.dump(model_data, model_save_path)
    
    logger.info(f"Baseline model - PR-AUC: {metrics['pr_auc']:.4f}, Recall: {metrics['recall']:.4f}")
    
    return model, scaler, metrics


if __name__ == "__main__":
    train_baseline_model()
