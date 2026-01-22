# Quick Start

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run Everything

Train both models:
```bash
python train_pipeline.py
```

This will:
1. Generate synthetic transaction data (100k transactions)
2. Validate the data
3. Engineer features
4. Train baseline model (Logistic Regression)
5. Train advanced model (XGBoost)
6. Register models and show metrics

## Start the API

```bash
python inference_api.py
```

API runs on `http://localhost:8000`

## Test the API

Using curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/transaction_example.json
```

Or Python:
```python
import requests

txn = {
    "transaction_id": "txn_test_001",
    "user_id": "user_12345",
    "amount": 250.00,
    "merchant_category": "retail",
    "timestamp": "2024-01-15T14:30:00Z",
    "device_id": "device_789",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "payment_method": "credit"
}

response = requests.post("http://localhost:8000/predict", json=txn)
print(response.json())
```

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Get fraud prediction
- `GET /model/info` - Model metrics

## Understanding the Output

**Risk Levels:**
- LOW (< 0.3): Transaction allowed
- MEDIUM (0.3-0.7): Step-up verification needed
- HIGH (≥ 0.7): Block transaction

**Actions:**
- ALLOW: Process normally
- STEP_UP_VERIFICATION: Require 2FA/SMS
- BLOCK: Reject transaction

## Customization

Edit `config.yaml` to change:
- Cost matrix (FN/FP costs)
- Risk thresholds
- Model hyperparameters
- API port/host

## Troubleshooting

**Model not found?**
- Run `python train_pipeline.py` first

**Import errors?**
- Make sure you're in the project root
- Install dependencies: `pip install -r requirements.txt`

**API won't start?**
- Check if port 8000 is available
- Change port in `config.yaml` if needed
