"""
Evaluation Metrics Module

Computes standard ML metrics and cost-aware business metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics for fraud detection."""
    
    def __init__(self, cost_matrix: Dict = None):
        """
        Initialize metrics calculator.
        
        Args:
            cost_matrix: Dictionary with 'false_negative_cost_multiplier' 
                        and 'false_positive_cost'
        """
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
        """
        Calculate all metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            transaction_amounts: Transaction amounts (for cost calculation)
        
        Returns:
            Dictionary of metrics
        """
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
        """
        Calculate total cost using cost matrix.
        
        Cost = (FN × transaction_amount × multiplier) + (FP × fixed_cost)
        """
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
        """Print formatted metrics report."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION METRICS")
        print("=" * 60)
        
        print("\n📊 Classification Metrics:")
        print(f"   Precision:  {metrics['precision']:.4f}")
        print(f"   Recall:     {metrics['recall']:.4f}")
        print(f"   F1-Score:   {metrics['f1']:.4f}")
        print(f"   ROC-AUC:    {metrics['roc_auc']:.4f}")
        print(f"   PR-AUC:     {metrics['pr_auc']:.4f} ⭐ (fraud favorite)")
        
        print("\n📈 Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"   True Negatives:  {cm['true_negatives']:>8,}")
        print(f"   False Positives: {cm['false_positives']:>8,}")
        print(f"   False Negatives: {cm['false_negatives']:>8,}")
        print(f"   True Positives:  {cm['true_positives']:>8,}")
        
        if 'total_cost' in metrics:
            print("\n💰 Business Metrics:")
            print(f"   Total Cost:           ${metrics['total_cost']:>12,.2f}")
            print(f"   Cost per Transaction: ${metrics['cost_per_transaction']:>12,.4f}")
        
        print("\n" + "=" * 60)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    transaction_amounts: np.ndarray = None,
    cost_matrix: Dict = None
) -> Dict:
    """
    Evaluate model and return metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        transaction_amounts: Transaction amounts
        cost_matrix: Cost matrix configuration
    
    Returns:
        Dictionary of metrics
    """
    calculator = MetricsCalculator(cost_matrix=cost_matrix)
    metrics = calculator.calculate_all_metrics(
        y_true, y_pred, y_proba, transaction_amounts
    )
    calculator.print_metrics_report(metrics)
    
    return metrics
