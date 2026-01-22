# Data validation - check data quality before training

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    # Validates transaction data
    
    def __init__(self):
        self.validation_results = {}
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        results = {}
        
        # Check required columns
        results['required_columns'] = self._check_required_columns(df)
        
        # Check data types
        results['data_types'] = self._check_data_types(df)
        
        # Check missing values
        results['missing_values'] = self._check_missing_values(df)
        
        # Check value ranges
        results['value_ranges'] = self._check_value_ranges(df)
        
        # Check temporal consistency
        results['temporal'] = self._check_temporal_consistency(df)
        
        # Check business logic
        results['business_logic'] = self._check_business_logic(df)
        
        # Overall validation status
        is_valid = all(
            result.get('status') == 'pass' 
            for result in results.values()
        )
        
        self.validation_results = results
        return is_valid, results
    
    def _check_required_columns(self, df: pd.DataFrame) -> Dict:
        required = [
            'transaction_id', 'user_id', 'timestamp', 'amount',
            'merchant_category', 'device_id', 'latitude', 'longitude',
            'payment_method', 'is_fraud'
        ]
        
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            return {'status': 'fail', 'missing_columns': missing}
        
        return {'status': 'pass'}
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict:
        issues = []
        
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
            except:
                issues.append("timestamp cannot be converted to datetime")
        
        numeric_cols = ['amount', 'latitude', 'longitude']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"{col} should be numeric")
        
        if issues:
            logger.warning(f"Data type issues: {issues}")
            return {'status': 'fail', 'issues': issues}
        
        return {'status': 'pass'}
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        critical_missing = missing_pct[missing_pct > 5]
        
        if len(critical_missing) > 0:
            logger.warning(f"High missing value rates: {critical_missing.to_dict()}")
            return {
                'status': 'fail',
                'missing_counts': missing.to_dict(),
                'missing_percentages': missing_pct.to_dict()
            }
        
        if missing.sum() > 0:
            logger.info(f"Some missing values detected: {missing[missing > 0].to_dict()}")
            return {
                'status': 'warn',
                'missing_counts': missing.to_dict()
            }
        
        return {'status': 'pass'}
    
    def _check_value_ranges(self, df: pd.DataFrame) -> Dict:
        issues = []
        
        if 'amount' in df.columns:
            if (df['amount'] <= 0).any():
                issues.append("Some amounts are <= 0")
            if (df['amount'] > 50000).any():
                issues.append("Some amounts exceed $50,000 (unusual)")
        
        if 'latitude' in df.columns:
            if ((df['latitude'] < -90) | (df['latitude'] > 90)).any():
                issues.append("Latitude out of valid range [-90, 90]")
        
        if 'longitude' in df.columns:
            if ((df['longitude'] < -180) | (df['longitude'] > 180)).any():
                issues.append("Longitude out of valid range [-180, 180]")
        
        if issues:
            logger.warning(f"Value range issues: {issues}")
            return {'status': 'fail', 'issues': issues}
        
        return {'status': 'pass'}
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> Dict:
        issues = []
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if (df['timestamp'] > pd.Timestamp.now()).any():
                issues.append("Some timestamps are in the future")
            
            if (df['timestamp'] < pd.Timestamp('2000-01-01')).any():
                issues.append("Some timestamps are very old (< 2000)")
        
        if issues:
            logger.warning(f"Temporal issues: {issues}")
            return {'status': 'fail', 'issues': issues}
        
        return {'status': 'pass'}
    
    def _check_business_logic(self, df: pd.DataFrame) -> Dict:
        issues = []
        
        if 'transaction_id' in df.columns:
            if df['transaction_id'].duplicated().any():
                issues.append("Duplicate transaction IDs found")
        
        if 'merchant_category' in df.columns:
            valid_categories = [
                "groceries", "gas", "restaurant", "retail", "online",
                "utilities", "entertainment", "travel", "healthcare"
            ]
            invalid = df[~df['merchant_category'].isin(valid_categories)]
            if len(invalid) > 0:
                issues.append(f"Invalid merchant categories: {invalid['merchant_category'].unique().tolist()}")
        
        if 'payment_method' in df.columns:
            valid_methods = ['credit', 'debit', 'digital_wallet']
            invalid = df[~df['payment_method'].isin(valid_methods)]
            if len(invalid) > 0:
                issues.append(f"Invalid payment methods: {invalid['payment_method'].unique().tolist()}")
        
        if issues:
            logger.warning(f"Business logic issues: {issues}")
            return {'status': 'fail', 'issues': issues}
        
        return {'status': 'pass'}
    
def validate_data_file(file_path: str) -> Tuple[bool, Dict]:
    df = pd.read_csv(file_path)
    
    validator = DataValidator()
    is_valid, results = validator.validate(df)
    
    if is_valid:
        logger.info("Data validation passed")
    else:
        # Only show issues if validation failed
        for check_name, result in results.items():
            if result.get('status') != 'pass':
                issues = result.get('issues', [])
                if issues:
                    logger.warning(f"Validation issue in {check_name}: {issues}")
    
    return is_valid, results
