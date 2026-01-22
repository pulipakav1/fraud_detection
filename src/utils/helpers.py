"""
Helper Utilities

Common utility functions used across the system.
"""

import numpy as np
from typing import Dict, Tuple


def determine_risk_level(probability: float, thresholds: Dict) -> Tuple[str, str]:
    """
    Determine risk level and recommended action based on probability.
    
    Args:
        probability: Fraud probability (0-1)
        thresholds: Dictionary with 'low', 'medium', 'high' thresholds
    
    Returns:
        Tuple of (risk_level, recommended_action)
    """
    if probability < thresholds.get('low', 0.3):
        return "LOW", "ALLOW"
    elif probability < thresholds.get('medium', 0.7):
        return "MEDIUM", "STEP_UP_VERIFICATION"
    else:
        return "HIGH", "BLOCK"


def format_prediction_response(
    probability: float,
    risk_level: str,
    action: str,
    model_version: str = "v1.0"
) -> Dict:
    """
    Format prediction response for API.
    
    Args:
        probability: Fraud probability
        risk_level: Risk level (LOW/MEDIUM/HIGH)
        action: Recommended action
        model_version: Model version string
    
    Returns:
        Formatted response dictionary
    """
    return {
        "fraud_probability": round(float(probability), 4),
        "risk_level": risk_level,
        "recommended_action": action,
        "model_version": model_version
    }
