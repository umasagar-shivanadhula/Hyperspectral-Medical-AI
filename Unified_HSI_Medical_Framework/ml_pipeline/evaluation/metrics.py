"""
Model evaluation for perfusion and tumor detection.

Computes Accuracy, Precision, Recall, F1 Score, Confusion Matrix,
and classifier probabilities. Evaluation is performed separately for
perfusion detection and tumor detection.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


def compute_metrics_per_task(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for a single task (perfusion or tumor).

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        class_names: List of class names
        y_prob: Optional (N, C) predicted probabilities

    Returns:
        Dict with accuracy, precision, recall, f1, confusion_matrix,
        classification_report, and optionally classifier_probabilities.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )

    # Handle binary/multiclass: use zero_division=0 and average
    n_classes = len(class_names)
    if n_classes == 2:
        average = "binary"
        pos_label = 1
    else:
        average = "weighted"
        pos_label = 1

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(
        precision_score(y_true, y_pred, average=average, zero_division=0, pos_label=pos_label)
    )
    recall = float(
        recall_score(y_true, y_pred, average=average, zero_division=0, pos_label=pos_label)
    )
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0, pos_label=pos_label))

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] < n_classes:
        # Pad if some classes missing in y_true/y_pred
        full_cm = np.zeros((n_classes, n_classes), dtype=int)
        full_cm[: cm.shape[0], : cm.shape[1]] = cm
        cm = full_cm

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )

    out = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_names,
        "classification_report": report,
    }

    if y_prob is not None:
        out["classifier_probabilities"] = {
            "mean_confidence_per_class": _mean_confidence_per_class(y_true, y_prob, n_classes),
        }

    return out


def _mean_confidence_per_class(
    y_true: np.ndarray, y_prob: np.ndarray, n_classes: int
) -> List[float]:
    """Mean predicted probability for the true class, per class."""
    means = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            means.append(0.0)
        else:
            means.append(float(np.mean(y_prob[mask, c])))
    return [round(x, 4) for x in means]


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred_rf: np.ndarray,
    y_pred_svm: np.ndarray,
    y_pred_fusion: np.ndarray,
    class_names: List[str],
    prob_rf: Optional[np.ndarray] = None,
    prob_svm: Optional[np.ndarray] = None,
    prob_fusion: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Full evaluation for the dual-branch (RF + SVM) + fusion pipeline.

    Returns metrics for RF, SVM, and fusion separately, plus confusion matrices
    and classifier probabilities.
    """
    results = {
        "random_forest": compute_metrics_per_task(
            y_true, y_pred_rf, class_names, y_prob=prob_rf
        ),
        "svm": compute_metrics_per_task(
            y_true, y_pred_svm, class_names, y_prob=prob_svm
        ),
        "fusion": compute_metrics_per_task(
            y_true, y_pred_fusion, class_names, y_prob=prob_fusion
        ),
    }
    return results
