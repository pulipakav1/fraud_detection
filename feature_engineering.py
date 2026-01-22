# Feature engineering - create features from raw transaction data
# Important: only use past data to avoid leakage

import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    # Creates features: transaction, velocity, geographic, device
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config.get('features', {})
        self.velocity_windows = self.feature_config.get('velocity_windows', [1, 24])
        
        # Store feature statistics for inference (to prevent leakage)
        self.feature_stats = {}
        self.is_training = True
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fit on training data
        self.is_training = True
        df = df.copy()
        
        # Sort by time - needed for velocity features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Compute stats for inference
        self._compute_feature_stats(df)
        
        # Create features
        df = self._create_transaction_features(df)
        df = self._create_velocity_features(df)
        df = self._create_geographic_features(df)
        df = self._create_device_features(df)
        
        # Remember which columns are features
        exclude = [
            'transaction_id', 'user_id', 'timestamp', 'is_fraud',
            'merchant_category', 'device_id', 'latitude', 'longitude',
            'payment_method'
        ]
        self.feature_stats['feature_columns'] = [
            col for col in df.columns if col not in exclude
        ]
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transform new data
        self.is_training = False
        df = df.copy()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Use stored stats to create features
        df = self._create_transaction_features(df)
        df = self._create_velocity_features(df)
        df = self._create_geographic_features(df)
        df = self._create_device_features(df)
        
        return df
    
    def _compute_feature_stats(self, df: pd.DataFrame):
        # Compute stats we need for feature engineering
        user_stats = df.groupby('user_id').agg({
            'amount': ['mean', 'std', 'count']
        }).reset_index()
        user_stats.columns = ['user_id', 'user_amount_mean', 'user_amount_std', 'user_txn_count']
        
        self.feature_stats = {
            'user_amount_mean': user_stats.set_index('user_id')['user_amount_mean'].to_dict(),
            'user_amount_std': user_stats.set_index('user_id')['user_amount_std'].to_dict(),
            'global_amount_mean': df['amount'].mean(),
            'global_amount_std': df['amount'].std(),
            'user_locations': self._compute_user_location_baselines(df),
            'user_devices': self._compute_user_device_baselines(df)
        }
    
    def _compute_user_location_baselines(self, df: pd.DataFrame):
        # Where does each user usually transact?
        user_locations = {}
        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id]
            if len(user_df) > 0:
                user_locations[user_id] = {
                    'lat_mean': user_df['latitude'].mean(),
                    'lon_mean': user_df['longitude'].mean(),
                    'lat_std': user_df['latitude'].std() or 0.1,
                    'lon_std': user_df['longitude'].std() or 0.1
                }
        return user_locations
    
    def _compute_user_device_baselines(self, df: pd.DataFrame):
        # What devices has each user used?
        user_devices = {}
        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id]
            if len(user_df) > 0:
                user_devices[user_id] = set(user_df['device_id'].unique())
        return user_devices
    
    def _create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Time features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['timestamp'].dt.month
        
        # Amount features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_squared'] = df['amount'] ** 2
        
        # Normalize amount by user history
        if self.is_training:
            user_means = df.groupby('user_id')['amount'].transform('mean')
            user_stds = df.groupby('user_id')['amount'].transform('std')
            user_stds = user_stds.replace(0, self.feature_stats['global_amount_std'])
        else:
            # Use stored stats
            user_means = df['user_id'].map(
                self.feature_stats['user_amount_mean']
            ).fillna(self.feature_stats['global_amount_mean'])
            user_stds = df['user_id'].map(
                self.feature_stats['user_amount_std']
            ).fillna(self.feature_stats['global_amount_std'])
            user_stds = user_stds.replace(0, self.feature_stats['global_amount_std'])
        
        df['amount_z_score'] = (df['amount'] - user_means) / user_stds
        df['amount_deviation_from_user_mean'] = df['amount'] - user_means
        
        # One-hot encode merchant and payment method
        merchant_dummies = pd.get_dummies(df['merchant_category'], prefix='merchant')
        df = pd.concat([df, merchant_dummies], axis=1)
        
        payment_dummies = pd.get_dummies(df['payment_method'], prefix='payment')
        df = pd.concat([df, payment_dummies], axis=1)
        
        return df
    
    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Velocity = how many transactions in last X hours
        # Only use past data to avoid leakage
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        for window_hours in self.velocity_windows:
            window_name = f"velocity_{window_hours}h"
            df[window_name] = 0
            df[f"amount_sum_{window_hours}h"] = 0.0
            
            # For each user, count transactions in time window
            for user_id in df['user_id'].unique():
                user_mask = df['user_id'] == user_id
                user_indices = df[user_mask].index
                user_timestamps = df.loc[user_mask, 'timestamp'].values
                user_amounts = df.loc[user_mask, 'amount'].values
                
                for i, idx in enumerate(user_indices):
                    current_time = user_timestamps[i]
                    window_start = current_time - timedelta(hours=window_hours)
                    
                    # Only count past transactions
                    past_mask = (user_timestamps >= window_start) & (user_timestamps < current_time)
                    df.loc[idx, window_name] = past_mask.sum()
                    df.loc[idx, f"amount_sum_{window_hours}h"] = user_amounts[past_mask].sum()
        
        # Average amount per transaction in window
        for window_hours in self.velocity_windows:
            window_name = f"velocity_{window_hours}h"
            amount_sum_name = f"amount_sum_{window_hours}h"
            avg_amount_name = f"avg_amount_{window_hours}h"
            
            df[avg_amount_name] = np.where(
                df[window_name] > 0,
                df[amount_sum_name] / df[window_name],
                0
            )
        
        return df
    
    def _create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate distance between two lat/lon points
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # km
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat/2)**2 + 
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
                 np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c
        
        df['distance_from_user_baseline'] = 0.0
        
        for idx, row in df.iterrows():
            user_id = row['user_id']
            current_lat = row['latitude']
            current_lon = row['longitude']
            
            if self.is_training:
                # Use past transactions to compute baseline
                past_df = df[(df['user_id'] == user_id) & (df.index < idx)]
                if len(past_df) > 0:
                    baseline_lat = past_df['latitude'].mean()
                    baseline_lon = past_df['longitude'].mean()
                    distance = haversine_distance(
                        current_lat, current_lon,
                        baseline_lat, baseline_lon
                    )
                else:
                    distance = 0.0  # new user
            else:
                # Use stored baseline
                if user_id in self.feature_stats['user_locations']:
                    baseline = self.feature_stats['user_locations'][user_id]
                    distance = haversine_distance(
                        current_lat, current_lon,
                        baseline['lat_mean'], baseline['lon_mean']
                    )
                else:
                    distance = 0.0  # new user
            
            df.loc[idx, 'distance_from_user_baseline'] = distance
        
        # Flag if >100km from usual location
        df['is_unusual_location'] = (df['distance_from_user_baseline'] > 100).astype(int)
        
        return df
    
    def _create_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['is_new_device'] = 0
        
        for idx, row in df.iterrows():
            user_id = row['user_id']
            device_id = row['device_id']
            
            if self.is_training:
                # Check if this device was used before
                past_df = df[(df['user_id'] == user_id) & (df.index < idx)]
                if len(past_df) > 0:
                    is_new = device_id not in past_df['device_id'].values
                else:
                    is_new = True  # first transaction
            else:
                # Use stored device list
                if user_id in self.feature_stats['user_devices']:
                    is_new = device_id not in self.feature_stats['user_devices'][user_id]
                else:
                    is_new = True  # new user
            
            df.loc[idx, 'is_new_device'] = int(is_new)
        
        return df
    
    def get_feature_list(self):
        return self.feature_stats.get('feature_columns', [])


def engineer_features(
    input_path: str,
    output_path: str,
    config_path: str = "config.yaml",
    is_training: bool = True
):
    df = pd.read_csv(input_path)
    
    engineer = FeatureEngineer(config_path=config_path)
    
    if is_training:
        df_features = engineer.fit_transform(df)
    else:
        df_features = engineer.transform(df)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    
    return df_features


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/transactions.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/features.csv"
    engineer_features(input_path, output_path)
