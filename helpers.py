# Helper functions

import numpy as np
from typing import Dict, Tuple


def determine_risk_level(probability: float, thresholds: Dict) -> Tuple[str, str]:
    # Convert probability to risk level
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
    return {
        "fraud_probability": round(float(probability), 4),
        "risk_level": risk_level,
        "recommended_action": action,
        "model_version": model_version
    }
