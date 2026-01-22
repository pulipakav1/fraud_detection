# FastAPI server for fraud detection
# Takes transaction data, returns fraud probability and risk level

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import yaml
from pathlib import Path
import logging

from feature_engineering import FeatureEngineer
from helpers import determine_risk_level, format_prediction_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection inference API",
    version="1.0.0"
)

# Global state
model_data = None
feature_engineer = None
config = None
risk_thresholds = None


class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float = Field(..., gt=0)
    merchant_category: str
    timestamp: str
    device_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    payment_method: str = Field(..., pattern="^(credit|debit|digital_wallet)$")


class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_level: str
    recommended_action: str
    model_version: str


def load_model_and_config():
    global model_data, feature_engineer, config, risk_thresholds
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Try advanced model first, fallback to baseline
    model_path = config.get('inference', {}).get('model_path', 'models/advanced_model.pkl')
    
    if not Path(model_path).exists():
        model_path = "models/baseline_model.pkl"
        if not Path(model_path).exists():
            logger.warning("No model found. Please train a model first.")
            return
    
    logger.info(f"Loading model from {model_path}")
    model_data = joblib.load(model_path)
    logger.info("Model loaded")
    
    # Initialize feature engineer
    # Note: velocity features will be 0 for single transactions (no history)
    feature_engineer = FeatureEngineer(config_path="config.yaml")
    feature_engineer.is_training = False
    feature_engineer.feature_stats = {
        'user_amount_mean': {},
        'user_amount_std': {},
        'global_amount_mean': 100.0,
        'global_amount_std': 50.0,
        'user_locations': {},
        'user_devices': {}
    }
    
    risk_thresholds = config.get('evaluation', {}).get('risk_thresholds', {
        'low': 0.3,
        'medium': 0.7,
        'high': 1.0
    })


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Fraud Detection API...")
    load_model_and_config()
    logger.info("API ready")


@app.get("/")
async def root():
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
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    if model_data is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )
    
    try:
        # Convert to dataframe
        df = pd.DataFrame([transaction.dict()])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create features
        df_features = feature_engineer.transform(df)
        
        feature_cols = model_data.get('feature_columns', [])
        X = df_features[feature_cols].fillna(0)
        
        # Handle missing features
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        X = X[feature_cols]
        
        # Scale if needed (baseline model)
        if 'scaler' in model_data:
            X = model_data['scaler'].transform(X)
        
        # Predict
        model = model_data['model']
        probability = model.predict_proba(X)[0, 1]
        
        risk_level, action = determine_risk_level(probability, risk_thresholds)
        
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
