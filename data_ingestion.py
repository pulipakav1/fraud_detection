# Generate synthetic transaction data for fraud detection
# In production you'd pull from your actual transaction database

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
from pathlib import Path


class FraudDataGenerator:
    # Generates fake transaction data that looks realistic
    # Fraud rate is ~1%, fraud happens more at weird hours, uses new devices
    
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
        # Generate the dataset
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
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        delta_days = (end - start).days
        
        # Smaller user set for legitimate transactions (repeat customers)
        n_users = int(n * 0.3)
        user_ids = [f"user_{i:06d}" for i in range(n_users)]
        user_weights = self._user_activity_weights(n_users)
        
        selected_user_indices = np.random.choice(n_users, size=n, p=user_weights)
        user_id_list = [user_ids[i] for i in selected_user_indices]
        
        # Generate timestamps
        random_days = np.random.randint(0, delta_days, size=n)
        base_times = [start + timedelta(days=int(d)) for d in random_days]
        
        # More activity during business hours
        probs = np.array([0.02]*9 + [0.08]*9 + [0.02]*6)
        probs = probs / probs.sum()
        hours = np.random.choice(24, size=n, p=probs)
        minutes = np.random.randint(0, 60, size=n)
        seconds = np.random.randint(0, 60, size=n)
        
        timestamps = [
            base_times[i].replace(hour=int(hours[i]), minute=int(minutes[i]), second=int(seconds[i]))
            for i in range(n)
        ]
        
        # Amounts
        amounts = np.random.lognormal(mean=3.0, sigma=1.2, size=n)
        amounts = np.clip(amounts, 0, 10000)
        amounts = np.round(amounts, 2)
        
        merchant_categories = [
            "groceries", "gas", "restaurant", "retail", "online",
            "utilities", "entertainment", "travel", "healthcare"
        ]
        merchant_category_list = np.random.choice(merchant_categories, size=n)
        
        device_ids = [f"device_{hash(user_id) % 1000:04d}" for user_id in user_id_list]
        
        # Locations - users typically in same region
        base_lats = 40.0 + np.random.uniform(-5, 5, size=n)
        base_lons = -74.0 + np.random.uniform(-5, 5, size=n)
        lats = np.round(base_lats + np.random.normal(0, 0.1, size=n), 4)
        lons = np.round(base_lons + np.random.normal(0, 0.1, size=n), 4)
        
        payment_methods = ['credit', 'debit', 'digital_wallet']
        payment_method_list = np.random.choice(payment_methods, size=n)
        
        df = pd.DataFrame({
            'user_id': user_id_list,
            'timestamp': timestamps,
            'amount': amounts,
            'merchant_category': merchant_category_list,
            'device_id': device_ids,
            'latitude': lats,
            'longitude': lons,
            'payment_method': payment_method_list
        })
        
        return df
    
    def _generate_fraudulent_transactions(
        self,
        n: int,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        delta_days = (end - start).days
        
        # Some fraud uses stolen accounts, some are new accounts
        is_stolen = np.random.random(n) < 0.3
        user_id_list = []
        for i in range(n):
            if is_stolen[i]:
                user_id_list.append(f"user_{np.random.randint(0, 10000):06d}")
            else:
                user_id_list.append(f"user_{np.random.randint(50000, 60000):06d}")
        
        random_days = np.random.randint(0, delta_days, size=n)
        base_times = [start + timedelta(days=int(d)) for d in random_days]
        
        # More fraud at off-hours
        probs = np.array([0.1]*6 + [0.03]*9 + [0.05]*9)
        probs = probs / probs.sum()
        hours = np.random.choice(24, size=n, p=probs)
        minutes = np.random.randint(0, 60, size=n)
        seconds = np.random.randint(0, 60, size=n)
        
        timestamps = [
            base_times[i].replace(hour=int(hours[i]), minute=int(minutes[i]), second=int(seconds[i]))
            for i in range(n)
        ]
        
        # Fraud amounts tend to be larger
        is_large = np.random.random(n) < 0.6
        amounts = np.where(
            is_large,
            np.random.lognormal(mean=5.0, sigma=1.5, size=n),
            np.random.lognormal(mean=2.0, sigma=0.8, size=n)
        )
        amounts = np.clip(amounts, 0, 15000)
        amounts = np.round(amounts, 2)
        
        # Fraud prefers certain categories
        fraud_preferred = ["online", "travel", "retail", "entertainment"]
        merchant_probs = np.array([0.7, 0.1, 0.1, 0.05, 0.03, 0.02])
        merchant_probs = merchant_probs / merchant_probs.sum()
        merchant_category_list = np.random.choice(
            fraud_preferred + ["groceries", "gas"],
            size=n,
            p=merchant_probs
        )
        
        # Fraud often uses new devices
        is_new_device = np.random.random(n) < 0.7
        device_ids = []
        for i in range(n):
            if is_new_device[i]:
                device_ids.append(f"device_{np.random.randint(10000, 20000):04d}")
            else:
                device_ids.append(f"device_{hash(user_id_list[i]) % 1000:04d}")
        
        # Fraud often from unusual locations
        is_unusual_location = np.random.random(n) < 0.6
        base_lats = np.where(
            is_unusual_location,
            40.0 + np.random.uniform(-20, 20, size=n),
            40.0 + np.random.uniform(-5, 5, size=n)
        )
        base_lons = np.where(
            is_unusual_location,
            -74.0 + np.random.uniform(-20, 20, size=n),
            -74.0 + np.random.uniform(-5, 5, size=n)
        )
        lats = np.round(
            base_lats + np.where(is_unusual_location, 0, np.random.normal(0, 0.1, size=n)),
            4
        )
        lons = np.round(
            base_lons + np.where(is_unusual_location, 0, np.random.normal(0, 0.1, size=n)),
            4
        )
        
        payment_methods = ['credit', 'debit', 'digital_wallet']
        payment_method_list = np.random.choice(payment_methods, size=n)
        
        df = pd.DataFrame({
            'user_id': user_id_list,
            'timestamp': timestamps,
            'amount': amounts,
            'merchant_category': merchant_category_list,
            'device_id': device_ids,
            'latitude': lats,
            'longitude': lons,
            'payment_method': payment_method_list
        })
        
        return df
    
    def _user_activity_weights(self, n_users: int) -> np.ndarray:
        # Some users are more active than others
        weights = np.random.power(2, n_users)
        return weights / weights.sum()


def load_config(config_path: str = "config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_and_save_data(
    output_path: str = "data/raw/transactions.csv",
    n_transactions: int = 100000,
    config_path: str = "config.yaml"
):
    # Generate data and save it
    config = load_config(config_path)
    random_seed = config.get('data', {}).get('random_seed', 42)
    
    generator = FraudDataGenerator(random_seed=random_seed)
    df = generator.generate_transactions(n_transactions=n_transactions)
    
    # Make sure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df


if __name__ == "__main__":
    generate_and_save_data()
