"""
Softmax Probability Fusion for Multi-Classifier HSI Analysis.

Implements late fusion of Random Forest and SVM probability outputs
using Softmax normalization for calibrated final predictions.
"""
import numpy as np
from typing import List, Optional


class SoftmaxFusion:
    """
    Fuses probability outputs from multiple classifiers using softmax weighting.

    Approach:
      1. Collect predict_proba() from each classifier
      2. Average probabilities across classifiers (equal weighting by default)
      3. Apply Softmax to re-calibrate the fused probabilities
      4. Return argmax as final prediction
    """

    def __init__(self, classifiers: List, weights: Optional[List[float]] = None):
        """
        Args:
            classifiers: list of fitted sklearn-compatible classifiers
                         each must have predict_proba() method
            weights:     per-classifier weights (default: equal weights)
        """
        self.classifiers = classifiers
        if weights is None:
            self.weights = [1.0 / len(classifiers)] * len(classifiers)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute fused probability matrix.

        Args:
            X: (N, D) feature matrix

        Returns:
            (N, C) fused probability matrix after softmax
        """
        n_samples = X.shape[0]
        all_probs = []

        for clf, w in zip(self.classifiers, self.weights):
            probs = clf.predict_proba(X)  # (N, C)
            all_probs.append(probs * w)

        # Weighted average
        fused = np.sum(all_probs, axis=0)  # (N, C)

        # Softmax calibration
        fused_softmax = softmax(fused, axis=1)
        return fused_softmax

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels from fused probabilities."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_single(self, x: np.ndarray) -> tuple:
        """
        Predict for a single sample.

        Args:
            x: (D,) feature vector

        Returns:
            (class_idx, confidence, prob_vector)
        """
        probs = self.predict_proba(x.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx]), probs

    def get_classifier_predictions(self, X: np.ndarray) -> dict:
        """
        Get individual predictions from each classifier alongside fusion result.

        Returns:
            dict with 'random_forest', 'svm', 'fusion' keys
        """
        names = ["random_forest", "svm"]
        results = {}

        for i, clf in enumerate(self.classifiers):
            probs = clf.predict_proba(X)
            results[names[i]] = {
                "predictions": np.argmax(probs, axis=1),
                "probabilities": probs,
                "confidence": probs.max(axis=1)
            }

        results["fusion"] = {
            "predictions": self.predict(X),
            "probabilities": self.predict_proba(X)
        }

        return results


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax over specified axis.

    Args:
        x: input array
        axis: axis along which to apply softmax

    Returns:
        softmax probabilities with same shape as x
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-10)


def temperature_scaling(probs: np.ndarray, temperature: float = 1.5) -> np.ndarray:
    """
    Apply temperature scaling to probability logits for calibration.

    Higher temperature → softer (more uncertain) predictions.
    Lower temperature → harder (more confident) predictions.

    Args:
        probs:       (N, C) probability matrix
        temperature: scaling factor

    Returns:
        calibrated probabilities
    """
    logits = np.log(probs + 1e-10)
    scaled_logits = logits / temperature
    return softmax(scaled_logits, axis=1)


def ensemble_vote(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Hard voting ensemble from multiple classifier prediction arrays.

    Args:
        predictions: list of (N,) integer prediction arrays

    Returns:
        (N,) majority vote predictions
    """
    preds_matrix = np.stack(predictions, axis=1)  # (N, K)
    n_samples, n_clf = preds_matrix.shape
    n_classes = preds_matrix.max() + 1

    votes = np.zeros((n_samples, n_classes), dtype=int)
    for k in range(n_clf):
        for c in range(n_classes):
            votes[:, c] += (preds_matrix[:, k] == c)

    return np.argmax(votes, axis=1)


def compute_fusion_confidence(rf_probs: np.ndarray, svm_probs: np.ndarray) -> dict:
    """
    Compute fusion confidence metrics for a single sample.

    Args:
        rf_probs:  (C,) RF probability vector
        svm_probs: (C,) SVM probability vector

    Returns:
        dict with fused probs, winner idx, confidence, agreement flag
    """
    fused = softmax((rf_probs + svm_probs) / 2.0)
    winner = int(np.argmax(fused))
    rf_winner = int(np.argmax(rf_probs))
    svm_winner = int(np.argmax(svm_probs))

    return {
        "fused_probabilities": fused,
        "winner_class": winner,
        "confidence": float(fused[winner]),
        "rf_prediction": rf_winner,
        "svm_prediction": svm_winner,
        "classifiers_agree": rf_winner == svm_winner
    }
