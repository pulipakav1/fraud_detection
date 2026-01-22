# End-to-End Fraud Detection ML System

## 🎯 Business Framing

### Business Problem

Financial transactions are vulnerable to fraud, causing:
- **Direct financial loss**: Unauthorized transactions result in immediate monetary damage
- **Customer dissatisfaction**: Fraud incidents erode trust and customer experience
- **Regulatory risk**: Non-compliance with fraud prevention standards can lead to penalties

### Business Objectives

The goal is to detect fraudulent transactions in **near real-time** while:
- **Minimizing false positives**: Blocking legitimate users creates friction and lost revenue
- **Catching high-risk fraud early**: Early detection prevents cascading losses

### 💡 Key Insight: Fraud is Asymmetric

**A missed fraud costs far more than a false alarm.**

This asymmetry drives our evaluation strategy:
- False Negative (missed fraud) = High cost (transaction amount + operational overhead)
- False Positive (false alarm) = Lower cost (customer friction, potential revenue loss)

## 🏗️ System Architecture

```
Transaction Events
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
Model Training (Baseline → Advanced)
      ↓
Evaluation (Cost-Aware Metrics)
      ↓
Model Registry
      ↓
Inference API (Real-Time)
```

### Future Enhancements
- Batch scoring pipeline
- Model monitoring and drift detection
- Automated retraining pipeline
- A/B testing framework

## 📊 Data Understanding

### Data Types (Typical Fraud Signals)

1. **Transaction-based**
   - Transaction amount
   - Transaction time (timestamp)
   - Merchant category
   - Payment method

2. **Behavioral / Velocity**
   - Transactions per time window (1h, 24h)
   - Amount deviation from user baseline
   - Transaction frequency patterns

3. **Geographic / Device**
   - Location mismatch
   - New device indicator
   - IP address anomalies

### ⚠️ Key Challenges

1. **Severe class imbalance**: Fraud cases are rare (typically <1% of transactions)
2. **Concept drift**: Fraud patterns evolve over time
3. **Label delay**: Fraud confirmation happens days/weeks after transaction
4. **Data leakage risks**: Must prevent look-ahead bias in feature engineering

## 🔧 Feature Engineering

### Feature Categories

#### Transaction-based Features
- Amount normalized by user history
- Time-of-day / day-of-week encoding
- Merchant category encoding

#### Behavioral / Velocity Features
- Transactions in last 1h / 24h
- Amount deviation from user baseline (z-score)
- Transaction frequency patterns

#### Geographic / Device Features
- Location mismatch (distance from user's typical location)
- New device indicator
- IP address risk score

### Critical Requirements
- ✅ Feature computation logic explained
- ✅ Prevention of look-ahead leakage
- ✅ Train/inference parity

## 🤖 Modeling Strategy

### Baseline Model
- **Logistic Regression** with class weights
- Provides interpretability and baseline performance
- Often kept as fallback in production systems

### Advanced Model
- **XGBoost / LightGBM** (industry standard for fraud detection)
- Handles non-linear patterns and feature interactions
- Provides feature importance for explainability

### Why This Matters
- Shows understanding of interpretability vs. power tradeoff
- Fraud teams often keep logistic regression as fallback
- Demonstrates model selection reasoning

## 📈 Evaluation Framework

### Metrics to Report
- **Precision**: Minimize false positives
- **Recall**: Catch as many fraud cases as possible
- **F1-Score**: Balanced metric
- **ROC-AUC**: Overall model discrimination
- **Precision-Recall AUC** ⭐ (fraud favorite - handles imbalance better)

### Business Metric (Required)
**Cost Matrix**:
- False Negative Cost = Transaction Amount + Operational Overhead
- False Positive Cost = Customer Friction Cost (estimated)

**Total Cost = (FN × FN_Cost) + (FP × FP_Cost)**

## 🎚️ Threshold Tuning & Risk Buckets

Instead of binary output, we use **risk buckets**:

- **Low Risk** (0-0.3): Allow transaction
- **Medium Risk** (0.3-0.7): Step-up verification (2FA, SMS)
- **High Risk** (0.7-1.0): Block transaction

This mimics real fraud systems and provides operational flexibility.

## 🚀 Inference API

### API Contract

**Input**: Transaction payload
```json
{
  "transaction_id": "txn_123",
  "user_id": "user_456",
  "amount": 150.00,
  "merchant_category": "electronics",
  "timestamp": "2024-01-15T10:30:00Z",
  "device_id": "device_789",
  "location": {"lat": 40.7128, "lon": -74.0060}
}
```

**Output**:
```json
{
  "fraud_probability": 0.91,
  "risk_level": "HIGH",
  "recommended_action": "BLOCK",
  "model_version": "v1.0"
}
```

## 📁 Project Structure

```
fraud_detection/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   └── validation.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── advanced.py
│   │   └── registry.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── cost_matrix.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── api.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── notebooks/
│   └── exploration.ipynb
├── models/
│   └── .gitkeep
└── tests/
    └── __init__.py
```

## 🚀 Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Generate synthetic data**:
```bash
python src/data/ingestion.py
```

3. **Train models**:
```bash
python -m src.models.baseline
python -m src.models.advanced
```

4. **Start inference API**:
```bash
python -m src.inference.api
```

5. **Test API**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/transaction_example.json
```

## 📝 Notes

This is a production-grade ML system demonstrating:
- End-to-end ML pipeline design
- Cost-aware evaluation
- Real-time inference
- Risk-based decision making
- Industry best practices
