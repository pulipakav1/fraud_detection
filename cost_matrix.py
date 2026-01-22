# Cost matrix for fraud detection
# False negatives cost way more than false positives

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class CostMatrix:
    # Cost matrix: FN = transaction amount, FP = fixed cost
    
    def __init__(
        self,
        false_negative_cost_multiplier: float = 1.0,
        false_positive_cost: float = 5.0
    ):
        self.fn_multiplier = false_negative_cost_multiplier
        self.fp_cost = false_positive_cost
    
    def calculate_cost(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        transaction_amounts: np.ndarray
    ) -> float:
        fn_mask = (y_true == 1) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)
        
        fn_cost = transaction_amounts[fn_mask].sum() * self.fn_multiplier
        fp_cost = len(transaction_amounts[fp_mask]) * self.fp_cost
        
        return fn_cost + fp_cost
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        transaction_amounts: np.ndarray,
        threshold_range: np.ndarray = None
    ) -> Tuple[float, float]:
        if threshold_range is None:
            threshold_range = np.arange(0.1, 0.95, 0.01)
        
        costs = []
        for threshold in threshold_range:
            y_pred = (y_proba >= threshold).astype(int)
            cost = self.calculate_cost(y_true, y_pred, transaction_amounts)
            costs.append(cost)
        
        min_cost_idx = np.argmin(costs)
        optimal_threshold = threshold_range[min_cost_idx]
        min_cost = costs[min_cost_idx]
        
        return optimal_threshold, min_cost
    
    def plot_cost_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        transaction_amounts: np.ndarray,
        threshold_range: np.ndarray = None,
        save_path: str = None
    ):
        if threshold_range is None:
            threshold_range = np.arange(0.1, 0.95, 0.01)
        
        costs = []
        for threshold in threshold_range:
            y_pred = (y_proba >= threshold).astype(int)
            cost = self.calculate_cost(y_true, y_pred, transaction_amounts)
            costs.append(cost)
        
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_range, costs, linewidth=2)
        plt.axvline(
            self.find_optimal_threshold(y_true, y_proba, transaction_amounts)[0],
            color='r', linestyle='--', label='Optimal Threshold'
        )
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Total Cost ($)', fontsize=12)
        plt.title('Cost vs Threshold', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved cost curve to {save_path}")
        else:
            plt.show()
