"""Cost-sensitive evaluation and threshold optimization."""

import numpy as np
from sklearn.metrics import confusion_matrix


def cost_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray,
    threshold: float, fn_cost: float, fp_cost: float,
) -> dict:
    """Compute total cost and confusion matrix at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_cost = fn * fn_cost + fp * fp_cost
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": precision, "recall": recall, "f1": f1,
        "total_cost": total_cost,
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray,
    fn_cost: float, fp_cost: float, n_steps: int = 200,
) -> dict:
    """Search for the threshold that minimizes total cost."""
    best = None
    for t in np.linspace(0.01, 0.99, n_steps):
        result = cost_at_threshold(y_true, y_prob, t, fn_cost, fp_cost)
        if best is None or result["total_cost"] < best["total_cost"]:
            best = result
    return best


def threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray,
    fn_cost: float, fp_cost: float, n_steps: int = 100,
) -> list[dict]:
    """Sweep thresholds and return metrics at each point."""
    results = []
    for t in np.linspace(0.01, 0.99, n_steps):
        results.append(cost_at_threshold(y_true, y_prob, t, fn_cost, fp_cost))
    return results
