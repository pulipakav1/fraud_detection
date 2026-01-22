"""
Data Ingestion Module

Generates synthetic fraud detection dataset with realistic patterns.
Handles class imbalance, temporal patterns, and feature relationships.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
from pathlib import Path
from typing import Tuple


class FraudDataGenerator:
    """
    Generate synthetic transaction data with fraud patterns.
    
    Key characteristics:
    - Severe class imbalance (~1% fraud)
    - Temporal patterns (time-of-day, day-of-week effects)
    - Behavioral patterns (velocity features)
    - Geographic patterns (location anomalies)
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_transactions(
        self,
        n_transactions: int = 100000,
        fraud_rate: float = 0.01,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31"
    ) -> pd.DataFrame:
        """
        Generate synthetic transaction dataset.
        
        Args:
            n_transactions: Total number of transactions
            fraud_rate: Proportion of fraudulent transactions
            start_date: Start date for transactions
            end_date: End date for transactions
        
        Returns:
            DataFrame with transaction data
        """
        n_fraud = int(n_transactions * fraud_rate)
        n_legitimate = n_transactions - n_fraud
        
        # Generate legitimate transactions
        legitimate = self._generate_legitimate_transactions(n_legitimate, start_date, end_date)
        legitimate['is_fraud'] = 0
        
        # Generate fraudulent transactions
        fraudulent = self._generate_fraudulent_transactions(n_fraud, start_date, end_date)
        fraudulent['is_fraud'] = 1
        
        # Combine and shuffle
        df = pd.concat([legitimate, fraudulent], ignore_index=True)
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Add transaction IDs
        df['transaction_id'] = [f"txn_{i:08d}" for i in range(len(df))]
        
        return df
    
    def _generate_legitimate_transactions(
        self,
        n: int,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Generate legitimate transaction patterns."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # User IDs (smaller set for legitimate - repeat customers)
        n_users = int(n * 0.3)  # 30% of transactions are from repeat users
        user_ids = [f"user_{i:06d}" for i in range(n_users)]
        
        # Generate transactions
        transactions = []
        for i in range(n):
            # User selection (some users more active)
            user_id = np.random.choice(user_ids, p=self._user_activity_weights(n_users))
            
            # Timestamp (more during business hours)
            timestamp = self._generate_legitimate_timestamp(start, end)
            
            # Amount (log-normal distribution, most transactions small)
            amount = np.random.lognormal(mean=3.0, sigma=1.2)
            amount = min(amount, 10000)  # Cap at $10k
            
            # Merchant category
            merchant_categories = [
                "groceries", "gas", "restaurant", "retail", "online",
                "utilities", "entertainment", "travel", "healthcare"
            ]
            merchant_category = np.random.choice(merchant_categories)
            
            # Device (users typically use same device)
            device_id = f"device_{hash(user_id) % 1000:04d}"
            
            # Location (users typically in same region)
            base_lat = 40.0 + np.random.uniform(-5, 5)
            base_lon = -74.0 + np.random.uniform(-5, 5)
            lat = base_lat + np.random.normal(0, 0.1)
            lon = base_lon + np.random.normal(0, 0.1)
            
            transactions.append({
                'user_id': user_id,
                'timestamp': timestamp,
                'amount': round(amount, 2),
                'merchant_category': merchant_category,
                'device_id': device_id,
                'latitude': round(lat, 4),
                'longitude': round(lon, 4),
                'payment_method': np.random.choice(['credit', 'debit', 'digital_wallet'])
            })
        
        return pd.DataFrame(transactions)
    
    def _generate_fraudulent_transactions(
        self,
        n: int,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Generate fraudulent transaction patterns."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Fraud patterns: larger amounts, unusual times, new devices, location mismatches
        transactions = []
        for i in range(n):
            # User ID (could be legitimate user or new account)
            if np.random.random() < 0.3:
                # Stolen account
                user_id = f"user_{np.random.randint(0, 10000):06d}"
            else:
                # New fraudulent account
                user_id = f"user_{np.random.randint(50000, 60000):06d}"
            
            # Timestamp (more during off-hours)
            timestamp = self._generate_fraudulent_timestamp(start, end)
            
            # Amount (tends to be larger, but varies)
            if np.random.random() < 0.6:
                # Large transaction
                amount = np.random.lognormal(mean=5.0, sigma=1.5)
            else:
                # Small test transaction
                amount = np.random.lognormal(mean=2.0, sigma=0.8)
            amount = min(amount, 15000)  # Cap at $15k
            
            # Merchant category (fraudsters prefer certain categories)
            fraud_preferred = ["online", "travel", "electronics", "retail"]
            merchant_category = np.random.choice(
                fraud_preferred + ["groceries", "gas"],
                p=[0.7, 0.1, 0.1, 0.05, 0.03, 0.02]
            )
            
            # Device (often new device)
            if np.random.random() < 0.7:
                device_id = f"device_{np.random.randint(10000, 20000):04d}"  # New device
            else:
                device_id = f"device_{hash(user_id) % 1000:04d}"  # Stolen device
            
            # Location (often far from user's typical location)
            if np.random.random() < 0.6:
                # Unusual location
                lat = 40.0 + np.random.uniform(-20, 20)
                lon = -74.0 + np.random.uniform(-20, 20)
            else:
                # Normal location (sophisticated fraud)
                base_lat = 40.0 + np.random.uniform(-5, 5)
                base_lon = -74.0 + np.random.uniform(-5, 5)
                lat = base_lat + np.random.normal(0, 0.1)
                lon = base_lon + np.random.normal(0, 0.1)
            
            transactions.append({
                'user_id': user_id,
                'timestamp': timestamp,
                'amount': round(amount, 2),
                'merchant_category': merchant_category,
                'device_id': device_id,
                'latitude': round(lat, 4),
                'longitude': round(lon, 4),
                'payment_method': np.random.choice(['credit', 'debit', 'digital_wallet'])
            })
        
        return pd.DataFrame(transactions)
    
    def _user_activity_weights(self, n_users: int) -> np.ndarray:
        """Generate weights for user activity (power law distribution)."""
        weights = np.random.power(2, n_users)
        return weights / weights.sum()
    
    def _generate_legitimate_timestamp(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
        """Generate timestamp with business hours bias."""
        delta = end - start
        random_days = np.random.randint(0, delta.days)
        base_time = start + timedelta(days=random_days)
        
        # More activity during business hours (9 AM - 6 PM)
        # Probabilities: 0-8 (low), 9-17 (high), 18-23 (low)
        probs = [0.02]*9 + [0.08]*9 + [0.02]*6
        probs = np.array(probs) / np.sum(probs)  # Normalize to sum to 1.0
        hour = np.random.choice(list(range(24)), p=probs)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        return base_time.replace(hour=hour, minute=minute, second=second)
    
    def _generate_fraudulent_timestamp(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
        """Generate timestamp with off-hours bias."""
        delta = end - start
        random_days = np.random.randint(0, delta.days)
        base_time = start + timedelta(days=random_days)
        
        # More activity during off-hours (late night, early morning)
        # Probabilities: 0-5 (high), 6-14 (low), 15-23 (medium)
        probs = [0.1]*6 + [0.03]*9 + [0.05]*9
        probs = np.array(probs) / np.sum(probs)  # Normalize to sum to 1.0
        hour = np.random.choice(list(range(24)), p=probs)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        return base_time.replace(hour=hour, minute=minute, second=second)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_and_save_data(
    output_path: str = "data/raw/transactions.csv",
    n_transactions: int = 100000,
    config_path: str = "config.yaml"
):
    """
    Generate synthetic data and save to CSV.
    
    Args:
        output_path: Path to save the generated data
        n_transactions: Number of transactions to generate
        config_path: Path to config file
    """
    config = load_config(config_path)
    random_seed = config.get('data', {}).get('random_seed', 42)
    
    generator = FraudDataGenerator(random_seed=random_seed)
    df = generator.generate_transactions(n_transactions=n_transactions)
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df):,} transactions")
    print(f"   - Legitimate: {len(df[df['is_fraud']==0]):,}")
    print(f"   - Fraudulent: {len(df[df['is_fraud']==1]):,}")
    print(f"   - Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"   - Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    generate_and_save_data()
