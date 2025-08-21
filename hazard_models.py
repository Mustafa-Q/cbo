import numpy as np
import math

def score_to_default_rate(score: float) -> float:
    """
    Logistic mapping from score to default probability.
    Calibrated so low-score borrowers have ~30% default rate.
    """
    if score is None:
        return 0.10

    s = max(300, min(score, 850))
    x = (s - 300) / (850 - 300)
    pd = 1.0 / (1.0 + math.exp(10 * (x - 0.35)))
    return float(np.clip(pd, 1e-4, 0.99))


def cox_hazard_rate(score: float, baseline_hazard: float = 0.05, beta: float = -3.0) -> float:
    """
    Simplified Cox proportional hazards hazard rate based on standardized credit score.
    """
    if score is None:
        return baseline_hazard

    s = max(300, min(score, 850))
    x = (s - 300) / (850 - 300)
    hazard = baseline_hazard * math.exp(beta * x)
    return float(np.clip(hazard, 1e-4, 0.99))
