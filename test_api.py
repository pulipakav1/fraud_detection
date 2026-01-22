"""
Simple test script for the Fraud Detection API

Tests the API endpoints to ensure everything works correctly.
"""

import requests
import json
import time
from pathlib import Path


def test_api():
    """Test the fraud detection API."""
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("TESTING FRAUD DETECTION API")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API. Is it running?")
        print("   Start it with: python -m src.inference.api")
        return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Root endpoint works")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Model info
    print("\n3. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model/info", timeout=5)
        if response.status_code == 200:
            print("   ✅ Model info retrieved")
            info = response.json()
            print(f"   Model loaded: {info.get('model_loaded')}")
            print(f"   Feature count: {info.get('feature_count')}")
            if 'metrics' in info:
                metrics = info['metrics']
                print(f"   PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}")
        else:
            print(f"   ⚠️  Model info failed: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Test 4: Prediction - Low risk transaction
    print("\n4. Testing prediction - Low risk transaction...")
    low_risk_transaction = {
        "transaction_id": "txn_test_low_001",
        "user_id": "user_12345",
        "amount": 25.00,
        "merchant_category": "groceries",
        "timestamp": "2024-01-15T14:30:00Z",
        "device_id": "device_789",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "payment_method": "credit"
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=low_risk_transaction,
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Prediction successful")
            print(f"   Fraud Probability: {result['fraud_probability']:.4f}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Recommended Action: {result['recommended_action']}")
            print(f"   Model Version: {result['model_version']}")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Prediction - High risk transaction
    print("\n5. Testing prediction - High risk transaction...")
    high_risk_transaction = {
        "transaction_id": "txn_test_high_001",
        "user_id": "user_99999",
        "amount": 5000.00,
        "merchant_category": "online",
        "timestamp": "2024-01-15T02:30:00Z",  # Late night
        "device_id": "device_new_12345",  # New device
        "latitude": 50.0,  # Far from typical location
        "longitude": -100.0,
        "payment_method": "credit"
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=high_risk_transaction,
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Prediction successful")
            print(f"   Fraud Probability: {result['fraud_probability']:.4f}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Recommended Action: {result['recommended_action']}")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 6: Invalid request
    print("\n6. Testing error handling - Invalid request...")
    invalid_transaction = {
        "transaction_id": "txn_invalid",
        "user_id": "user_123",
        "amount": -10.0,  # Invalid: negative amount
        "merchant_category": "groceries",
        "timestamp": "2024-01-15T14:30:00Z",
        "device_id": "device_789",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "payment_method": "credit"
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=invalid_transaction,
            timeout=10
        )
        if response.status_code == 422:  # Validation error
            print("   ✅ Error handling works (validation error caught)")
        else:
            print(f"   ⚠️  Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_api()
