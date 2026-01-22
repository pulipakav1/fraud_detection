"""
Real-Time Inference API

FastAPI-based API for fraud detection predictions.
Handles feature engineering and model inference in real-time.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict
import numpy as np
import pandas as pd
import joblib
import yaml
from pathlib import Path
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.features.engineering import FeatureEngineer
from src.utils.helpers import determine_risk_level, format_prediction_response
from src.models.registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection inference API",
    version="1.0.0"
)

# Global variables (would be loaded at startup in production)
model_data = None
feature_engineer = None
config = None
risk_thresholds = None


class TransactionRequest(BaseModel):
    """Transaction request schema."""
    transaction_id: str
    user_id: str
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: str
    timestamp: str  # ISO format datetime string
    device_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    payment_method: str = Field(..., pattern="^(credit|debit|digital_wallet)$")


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    fraud_probability: float
    risk_level: str
    recommended_action: str
    model_version: str


def load_model_and_config():
    """Load model and configuration at startup."""
    global model_data, feature_engineer, config, risk_thresholds
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model (try advanced first, fallback to baseline)
    model_path = config.get('inference', {}).get('model_path', 'models/advanced_model.pkl')
    
    if not Path(model_path).exists():
        # Try baseline
        model_path = "models/baseline_model.pkl"
        if not Path(model_path).exists():
            logger.warning("No model found. Please train a model first.")
            return
    
    logger.info(f"Loading model from {model_path}")
    model_data = joblib.load(model_path)
    logger.info("✅ Model loaded")
    
    # Initialize feature engineer
    # Note: For single-transaction inference, velocity features will be 0
    # (no history available). In production, you'd have a feature store.
    feature_engineer = FeatureEngineer(config_path="config.yaml")
    feature_engineer.is_training = False
    # Initialize empty stats for inference (velocity features will default to 0)
    feature_engineer.feature_stats = {
        'user_amount_mean': {},
        'user_amount_std': {},
        'global_amount_mean': 100.0,  # Default
        'global_amount_std': 50.0,  # Default
        'user_locations': {},
        'user_devices': {}
    }
    
    # Load risk thresholds
    risk_thresholds = config.get('evaluation', {}).get('risk_thresholds', {
        'low': 0.3,
        'medium': 0.7,
        'high': 1.0
    })


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Fraud Detection API...")
    load_model_and_config()
    logger.info("✅ API ready")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict fraud probability for a transaction.
    
    Args:
        transaction: Transaction data
    
    Returns:
        Fraud prediction with risk level and recommended action
    """
    if model_data is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    try:
        # Convert request to DataFrame (single row)
        df = pd.DataFrame([transaction.dict()])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Engineer features
        # Note: For single-transaction inference:
        # - Velocity features will be 0 (no history)
        # - Geographic features will default (new user assumed)
        # - Device features will default (new device assumed)
        # In production, you'd query a feature store for user history
        df_features = feature_engineer.transform(df)
        
        # Get feature columns
        feature_cols = model_data.get('feature_columns', [])
        
        # Select and align features
        X = df_features[feature_cols].fillna(0)
        
        # Handle missing features (set to 0)
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        X = X[feature_cols]  # Ensure correct order
        
        # Scale if scaler exists (baseline model)
        if 'scaler' in model_data:
            X = model_data['scaler'].transform(X)
        
        # Predict
        model = model_data['model']
        probability = model.predict_proba(X)[0, 1]
        
        # Determine risk level
        risk_level, action = determine_risk_level(probability, risk_thresholds)
        
        # Format response
        model_version = config.get('inference', {}).get('model_version', 'v1.0')
        response = format_prediction_response(
            probability, risk_level, action, model_version
        )
        
        logger.info(
            f"Prediction for {transaction.transaction_id}: "
            f"prob={probability:.4f}, risk={risk_level}, action={action}"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    metrics = model_data.get('metrics', {})
    return {
        "model_loaded": True,
        "metrics": metrics,
        "feature_count": len(model_data.get('feature_columns', []))
    }


if __name__ == "__main__":
    import uvicorn
    
    # Load config for port
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    inference_config = config.get('inference', {})
    host = inference_config.get('api_host', '0.0.0.0')
    port = inference_config.get('api_port', 8000)
    
    uvicorn.run(app, host=host, port=port)
