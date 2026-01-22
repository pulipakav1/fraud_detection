# Evaluation metrics for fraud detection

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    # Calculate metrics for fraud detection
    
    def __init__(self, cost_matrix: Dict = None):
        self.cost_matrix = cost_matrix or {
            'false_negative_cost_multiplier': 1.0,
            'false_positive_cost': 5.0
        }
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        transaction_amounts: np.ndarray = None
    ) -> Dict:
        metrics = {}
        
        # Standard classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        # Cost-aware metrics
        if transaction_amounts is not None:
            metrics['total_cost'] = self._calculate_total_cost(
                y_true, y_pred, transaction_amounts
            )
            metrics['cost_per_transaction'] = (
                metrics['total_cost'] / len(y_true)
            )
        
        return metrics
    
    def _calculate_total_cost(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        transaction_amounts: np.ndarray
    ) -> float:
        # Calculate cost: FN cost = transaction amount, FP cost = fixed
        fn_mask = (y_true == 1) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)
        
        fn_cost = (
            transaction_amounts[fn_mask].sum() * 
            self.cost_matrix['false_negative_cost_multiplier']
        )
        fp_cost = (
            len(transaction_amounts[fp_mask]) * 
            self.cost_matrix['false_positive_cost']
        )
        
        return fn_cost + fp_cost
    
    def print_metrics_report(self, metrics: Dict):
        # Only print if explicitly called, not automatically
        pass


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    transaction_amounts: np.ndarray = None,
    cost_matrix: Dict = None
) -> Dict:
    calculator = MetricsCalculator(cost_matrix=cost_matrix)
    metrics = calculator.calculate_all_metrics(
        y_true, y_pred, y_proba, transaction_amounts
    )
    
    return metrics
