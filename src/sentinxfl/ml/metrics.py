"""
SentinXFL - Model Metrics
==========================

Comprehensive metrics for fraud detection model evaluation.
Focuses on imbalanced classification metrics relevant to fraud detection.

Author: Anshuman Bakshi
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from sentinxfl.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationMetrics:
    """
    Comprehensive classification metrics for fraud detection.
    
    Emphasizes metrics relevant to highly imbalanced datasets:
    - AUPRC (Average Precision) - more informative than AUC-ROC for imbalanced
    - F2-score - weighs recall higher than precision (catching fraud matters more)
    - Cost-based metrics - monetary impact of FP vs FN
    """
    
    # Basic metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0  # Also called sensitivity/TPR
    specificity: float = 0.0  # TNR
    f1_score: float = 0.0
    f2_score: float = 0.0  # Weighs recall 2x more than precision
    
    # ROC metrics
    auc_roc: float = 0.0
    fpr: NDArray[np.float32] | None = field(default=None)
    tpr: NDArray[np.float32] | None = field(default=None)
    roc_thresholds: NDArray[np.float32] | None = field(default=None)
    
    # Precision-Recall metrics (more important for imbalanced data)
    auc_pr: float = 0.0  # AUPRC - Average Precision
    precision_curve: NDArray[np.float32] | None = field(default=None)
    recall_curve: NDArray[np.float32] | None = field(default=None)
    pr_thresholds: NDArray[np.float32] | None = field(default=None)
    
    # Confusion matrix
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # Rates
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    
    # Optimal threshold (by F1)
    optimal_threshold: float = 0.5
    
    # Dataset info
    n_samples: int = 0
    n_positive: int = 0  # Fraud cases
    n_negative: int = 0  # Legitimate cases
    imbalance_ratio: float = 0.0
    
    def to_dict(self, include_curves: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_curves: Whether to include ROC/PR curve arrays
        """
        result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1_score": self.f1_score,
            "f2_score": self.f2_score,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "optimal_threshold": self.optimal_threshold,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "imbalance_ratio": self.imbalance_ratio,
        }
        
        if include_curves:
            result["fpr"] = self.fpr.tolist() if self.fpr is not None else None
            result["tpr"] = self.tpr.tolist() if self.tpr is not None else None
            result["precision_curve"] = self.precision_curve.tolist() if self.precision_curve is not None else None
            result["recall_curve"] = self.recall_curve.tolist() if self.recall_curve is not None else None
        
        return result
    
    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Metrics Summary:\n"
            f"  Samples: {self.n_samples:,} (Fraud: {self.n_positive:,}, Ratio: {self.imbalance_ratio:.4%})\n"
            f"  AUC-ROC: {self.auc_roc:.4f} | AUPRC: {self.auc_pr:.4f}\n"
            f"  Precision: {self.precision:.4f} | Recall: {self.recall:.4f}\n"
            f"  F1: {self.f1_score:.4f} | F2: {self.f2_score:.4f}\n"
            f"  FPR: {self.false_positive_rate:.4%} | FNR: {self.false_negative_rate:.4%}\n"
            f"  Confusion: TP={self.true_positives:,} FP={self.false_positives:,} "
            f"TN={self.true_negatives:,} FN={self.false_negatives:,}"
        )


class MetricsCalculator:
    """
    Calculate comprehensive metrics for fraud detection models.
    
    Usage:
        calculator = MetricsCalculator()
        metrics = calculator.calculate(y_true, y_pred, y_proba)
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize calculator.
        
        Args:
            threshold: Classification threshold (default 0.5)
        """
        self.threshold = threshold
    
    def calculate(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32] | None = None,
        y_proba: NDArray[np.float32] | None = None,
        compute_curves: bool = True,
        find_optimal_threshold: bool = True,
    ) -> ClassificationMetrics:
        """
        Calculate all metrics.
        
        Args:
            y_true: Ground truth labels (0 = legitimate, 1 = fraud)
            y_pred: Predicted labels (optional if y_proba provided)
            y_proba: Predicted probabilities (optional if y_pred provided)
            compute_curves: Whether to compute ROC/PR curves
            find_optimal_threshold: Whether to find optimal threshold by F1
            
        Returns:
            ClassificationMetrics with all computed metrics
        """
        y_true = np.asarray(y_true).astype(np.int32)
        
        # Get predictions from probabilities if needed
        if y_pred is None:
            if y_proba is None:
                raise ValueError("Either y_pred or y_proba must be provided")
            y_proba = np.asarray(y_proba).astype(np.float32)
            y_pred = (y_proba >= self.threshold).astype(np.int32)
        else:
            y_pred = np.asarray(y_pred).astype(np.int32)
        
        if y_proba is not None:
            y_proba = np.asarray(y_proba).astype(np.float32)
        
        metrics = ClassificationMetrics()
        
        # Dataset info
        metrics.n_samples = len(y_true)
        metrics.n_positive = int(y_true.sum())
        metrics.n_negative = metrics.n_samples - metrics.n_positive
        metrics.imbalance_ratio = metrics.n_positive / metrics.n_samples if metrics.n_samples > 0 else 0.0
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.true_positives = int(tp)
        metrics.false_positives = int(fp)
        metrics.true_negatives = int(tn)
        metrics.false_negatives = int(fn)
        
        # Basic metrics
        metrics.accuracy = float(accuracy_score(y_true, y_pred))
        metrics.precision = float(precision_score(y_true, y_pred, zero_division=0))
        metrics.recall = float(recall_score(y_true, y_pred, zero_division=0))
        metrics.f1_score = float(f1_score(y_true, y_pred, zero_division=0))
        metrics.f2_score = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
        
        # Specificity (TNR)
        metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Rates
        metrics.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics.false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Probability-based metrics
        if y_proba is not None:
            try:
                metrics.auc_roc = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                metrics.auc_roc = 0.0
            
            try:
                metrics.auc_pr = float(average_precision_score(y_true, y_proba))
            except ValueError:
                metrics.auc_pr = 0.0
            
            # Curves
            if compute_curves:
                try:
                    fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
                    metrics.fpr = fpr.astype(np.float32)
                    metrics.tpr = tpr.astype(np.float32)
                    metrics.roc_thresholds = roc_thresh.astype(np.float32)
                except ValueError:
                    pass
                
                try:
                    prec, rec, pr_thresh = precision_recall_curve(y_true, y_proba)
                    metrics.precision_curve = prec.astype(np.float32)
                    metrics.recall_curve = rec.astype(np.float32)
                    metrics.pr_thresholds = pr_thresh.astype(np.float32)
                except ValueError:
                    pass
            
            # Find optimal threshold
            if find_optimal_threshold and metrics.precision_curve is not None:
                metrics.optimal_threshold = self._find_optimal_threshold(
                    metrics.precision_curve,
                    metrics.recall_curve,
                    metrics.pr_thresholds,
                )
        
        return metrics
    
    def _find_optimal_threshold(
        self,
        precision: NDArray[np.float32],
        recall: NDArray[np.float32],
        thresholds: NDArray[np.float32],
    ) -> float:
        """
        Find optimal threshold by maximizing F1 score.
        
        Uses precision-recall curve to find the threshold that
        maximizes the F1 score.
        """
        # Compute F1 for each threshold
        # Note: precision and recall arrays are 1 longer than thresholds
        f1_scores = np.zeros_like(thresholds)
        for i, thresh in enumerate(thresholds):
            if precision[i] + recall[i] > 0:
                f1_scores[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        
        # Find best threshold
        if len(f1_scores) > 0:
            best_idx = np.argmax(f1_scores)
            return float(thresholds[best_idx])
        return 0.5
    
    def calculate_cost_metrics(
        self,
        y_true: NDArray[np.int32],
        y_pred: NDArray[np.int32],
        cost_fp: float = 1.0,
        cost_fn: float = 10.0,
    ) -> dict[str, float]:
        """
        Calculate cost-based metrics.
        
        In fraud detection, missing a fraud (FN) typically costs more
        than a false alarm (FP) due to monetary loss.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            cost_fp: Cost of false positive (false alarm)
            cost_fn: Cost of false negative (missed fraud)
            
        Returns:
            Dictionary with cost metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        max_cost = (len(y_true) * max(cost_fp, cost_fn))
        
        return {
            "total_cost": total_cost,
            "cost_per_sample": total_cost / len(y_true) if len(y_true) > 0 else 0.0,
            "cost_ratio": total_cost / max_cost if max_cost > 0 else 0.0,
            "fp_cost": fp * cost_fp,
            "fn_cost": fn * cost_fn,
        }


def compare_models(
    metrics_list: list[tuple[str, ClassificationMetrics]],
) -> dict[str, dict[str, float]]:
    """
    Compare multiple models.
    
    Args:
        metrics_list: List of (model_name, metrics) tuples
        
    Returns:
        Dictionary with comparison data for each metric
    """
    comparison = {
        "auc_roc": {},
        "auc_pr": {},
        "f1_score": {},
        "f2_score": {},
        "precision": {},
        "recall": {},
    }
    
    for name, metrics in metrics_list:
        comparison["auc_roc"][name] = metrics.auc_roc
        comparison["auc_pr"][name] = metrics.auc_pr
        comparison["f1_score"][name] = metrics.f1_score
        comparison["f2_score"][name] = metrics.f2_score
        comparison["precision"][name] = metrics.precision
        comparison["recall"][name] = metrics.recall
    
    # Add best model for each metric
    for metric_name in comparison:
        values = comparison[metric_name]
        if values:
            best_model = max(values.keys(), key=lambda k: values[k])
            comparison[metric_name]["_best"] = best_model
    
    return comparison
