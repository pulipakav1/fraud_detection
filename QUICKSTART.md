# Quick Start Guide

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

Train both baseline and advanced models:

```bash
python train_pipeline.py
```

This will:
- Generate synthetic transaction data
- Validate data quality
- Engineer features (transaction, velocity, geographic, device)
- Train baseline model (Logistic Regression)
- Train advanced model (XGBoost)
- Register models in the model registry
- Display evaluation metrics

### 3. Start the Inference API

```bash
python -m src.inference.api
```

The API will start on `http://localhost:8000`

### 4. Test the API

**Using curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/transaction_example.json
```

**Using Python:**
```python
import requests

transaction = {
    "transaction_id": "txn_test_001",
    "user_id": "user_12345",
    "amount": 250.00,
    "merchant_category": "electronics",
    "timestamp": "2024-01-15T14:30:00Z",
    "device_id": "device_789",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "payment_method": "credit"
}

response = requests.post("http://localhost:8000/predict", json=transaction)
print(response.json())
```

**Expected Response:**
```json
{
    "fraud_probability": 0.1234,
    "risk_level": "LOW",
    "recommended_action": "ALLOW",
    "model_version": "v1.0"
}
```

## 📊 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Fraud prediction
- `GET /model/info` - Model information and metrics

## 🔍 Understanding the Output

### Risk Levels

- **LOW** (probability < 0.3): Transaction allowed
- **MEDIUM** (0.3 ≤ probability < 0.7): Step-up verification required (2FA, SMS)
- **HIGH** (probability ≥ 0.7): Transaction blocked

### Recommended Actions

- **ALLOW**: Process transaction normally
- **STEP_UP_VERIFICATION**: Require additional authentication
- **BLOCK**: Reject transaction

## 📈 Model Performance

After training, you'll see metrics like:

- **Precision**: Minimize false positives
- **Recall**: Catch as many fraud cases as possible
- **F1-Score**: Balanced metric
- **ROC-AUC**: Overall discrimination
- **PR-AUC**: ⭐ Best for imbalanced fraud data
- **Total Cost**: Business cost using cost matrix

## 🛠️ Customization

### Adjust Cost Matrix

Edit `config.yaml`:
```yaml
evaluation:
  cost_matrix:
    false_negative_cost_multiplier: 1.0  # Transaction amount × multiplier
    false_positive_cost: 5.0  # Fixed cost per false positive
```

### Adjust Risk Thresholds

Edit `config.yaml`:
```yaml
evaluation:
  risk_thresholds:
    low: 0.3
    medium: 0.7
    high: 1.0
```

### Change Model Parameters

Edit `config.yaml` under `models.baseline` or `models.advanced`

## 📝 Next Steps

1. **Explore the code**: Check out `src/` directory for implementation details
2. **Modify features**: Add new features in `src/features/engineering.py`
3. **Experiment with models**: Try different algorithms or hyperparameters
4. **Add monitoring**: Implement model drift detection
5. **Scale up**: Deploy to production with proper infrastructure

## 🐛 Troubleshooting

**Model not found error:**
- Run `python train_pipeline.py` first to train models

**Import errors:**
- Make sure you're in the project root directory
- Install all dependencies: `pip install -r requirements.txt`

**API won't start:**
- Check if port 8000 is available
- Modify port in `config.yaml` if needed
