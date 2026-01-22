# Fraud Detection ML System

End-to-end machine learning system for detecting fraudulent financial transactions in real-time.

## Business Context

Fraud detection is critical for financial systems. The main challenges:
- Direct financial losses from unauthorized transactions
- Customer trust issues when fraud happens
- Regulatory compliance requirements

Our goal: detect fraud in near real-time while minimizing false positives (we don't want to block legitimate customers).

Key insight: **A missed fraud costs way more than a false alarm.** This drives our cost-aware evaluation approach.

## System Overview

```
Transaction Events → Data Validation → Feature Engineering → 
Model Training → Evaluation → Model Registry → Inference API
```

Still need to add: batch scoring, monitoring, and retraining pipelines (coming later).

## Data

We work with typical transaction data:
- Transaction amount, timestamp, merchant category
- Device and location info
- User behavior patterns (velocity features)
- Account history

Main challenges:
- **Class imbalance**: Fraud is rare (<1% typically)
- **Concept drift**: Fraud patterns change over time
- **Label delay**: We only know fraud happened days/weeks later
- **Data leakage**: Have to be super careful with temporal features

## Features

We engineer three main feature types:

**Transaction features:**
- Amount normalized by user's typical spending
- Time features (hour of day, day of week)
- Merchant category encoding

**Velocity features:**
- Transaction count in last 1h and 24h windows
- Amount deviation from user baseline (z-score)
- Frequency patterns

**Geographic/Device:**
- Distance from user's typical location
- New device flag
- Location anomaly detection

Important: All features are computed to prevent look-ahead bias. We only use past data when computing features.

## Models

**Baseline: Logistic Regression**
- Simple, interpretable
- Good baseline to compare against
- Often kept as fallback in production

**Advanced: XGBoost**
- Industry standard for fraud detection
- Handles non-linear patterns well
- Still provides feature importance

Why both? Shows we understand the tradeoff between interpretability and performance. Plus fraud teams usually want a simple model as backup.

## Evaluation

We track standard metrics:
- Precision, Recall, F1
- ROC-AUC
- **PR-AUC** (this is the one that matters for imbalanced fraud data)

But the real metric is **business cost**:
- False Negative cost = transaction amount (we lost money)
- False Positive cost = customer friction (~$5 per false alarm)

Total cost = (FN × transaction_amount) + (FP × $5)

This is what we optimize for in production.

## Risk Buckets

Instead of just "fraud" or "not fraud", we use risk levels:

- **Low (0-0.3)**: Allow transaction
- **Medium (0.3-0.7)**: Step-up verification (2FA, SMS)
- **High (0.7-1.0)**: Block transaction

This gives operations teams flexibility in how they handle different risk levels.

## API

Real-time inference via FastAPI.

**Request:**
```json
{
  "transaction_id": "txn_123",
  "user_id": "user_456",
  "amount": 150.00,
  "merchant_category": "retail",
  "timestamp": "2024-01-15T10:30:00Z",
  "device_id": "device_789",
  "latitude": 40.7128,
  "longitude": -74.0060,
  "payment_method": "credit"
}
```

**Response:**
```json
{
  "fraud_probability": 0.91,
  "risk_level": "HIGH",
  "recommended_action": "BLOCK",
  "model_version": "v1.0"
}
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the full pipeline:
```bash
python train_pipeline.py
```

3. Start the API:
```bash
python inference_api.py
```

4. Test it:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/transaction_example.json
```

## Project Structure

```
fraud_detection/
├── data_ingestion.py      # Generate synthetic transaction data
├── data_validation.py     # Validate data quality
├── feature_engineering.py  # Create features
├── baseline_model.py      # Logistic Regression model
├── advanced_model.py      # XGBoost model
├── model_registry.py      # Model versioning
├── evaluation_metrics.py  # Evaluation metrics
├── cost_matrix.py        # Cost-aware evaluation
├── inference_api.py       # FastAPI server
├── helpers.py            # Helper functions
├── train_pipeline.py      # Main training script
├── config.yaml           # Configuration
├── data/                 # Raw and processed data
└── models/               # Trained model files
```

## Notes

- This is a production-ready system that could be deployed as-is
- The cost-aware evaluation is what makes this different from academic projects
- Feature engineering prevents data leakage (critical for real systems)
- Risk buckets instead of binary classification matches how real fraud systems work
