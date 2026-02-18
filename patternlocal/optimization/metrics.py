"""
Evaluation metrics for PatternLocal explanation optimization.

This module provides metrics to evaluate the quality of explanations
during hyperparameter optimization.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class ExplanationMetrics:
    """Collection of explanation evaluation metrics."""

    @staticmethod
    def fidelity_error(
        explanation_weights: np.ndarray,
        instance: np.ndarray,
        ground_truth_mask: np.ndarray,
        predict_fn: Callable,
        X_train: np.ndarray,
        n_samples: int = 1000,
        **kwargs,
    ) -> float:
        """Compute fidelity error of explanation.

        Measures how well the explanation approximates the model's behavior
        in the local neighborhood of the instance.

        Args:
            explanation_weights: Explanation weights
            instance: Instance being explained
            ground_truth_mask: Ground truth importance mask (if available)
            predict_fn: Model prediction function
            X_train: Training data for generating neighborhood
            n_samples: Number of samples for fidelity estimation

        Returns:
            Fidelity error (lower is better)
        """
        try:
            # Generate neighborhood samples
            neighborhood = ExplanationMetrics._generate_neighborhood(
                instance, X_train, n_samples
            )

            # Get model predictions
            model_preds = predict_fn(neighborhood)
            if len(model_preds.shape) > 1:
                model_preds = model_preds[:, 1]  # Binary classification

            # Compute linear approximation predictions
            differences = neighborhood - instance
            linear_preds = np.dot(differences, explanation_weights)

            # Compute fidelity error (MSE)
            fidelity_error = np.mean((model_preds - linear_preds) ** 2)

            return float(fidelity_error)

        except Exception as e:
            logger.warning(f"Error computing fidelity: {e}")
            return float("inf")

    @staticmethod
    def stability_error(
        explainer_instance: Any,
        instance: np.ndarray,
        predict_fn: Callable,
        X_train: np.ndarray,
        n_runs: int = 10,
        **kwargs,
    ) -> float:
        """Compute stability error of explanations.

        Measures consistency of explanations across multiple runs.

        Args:
            explainer_instance: Fitted explainer instance
            instance: Instance being explained
            predict_fn: Model prediction function
            X_train: Training data
            n_runs: Number of runs for stability estimation

        Returns:
            Stability error (lower is better)
        """
        try:
            explanations = []

            for _ in range(n_runs):
                # Generate explanation
                explanation = explainer_instance.explain_instance(
                    instance, predict_fn, X_train, **kwargs
                )
                explanations.append(explanation["pattern_weights"])

            # Compute pairwise similarities
            similarities = []
            for i in range(len(explanations)):
                for j in range(i + 1, len(explanations)):
                    # Cosine similarity
                    sim = np.dot(explanations[i], explanations[j]) / (
                        np.linalg.norm(explanations[i])
                        * np.linalg.norm(explanations[j])
                    )
                    similarities.append(sim)

            # Stability error is 1 - average similarity
            stability_error = 1.0 - np.mean(similarities)

            return float(stability_error)

        except Exception as e:
            logger.warning(f"Error computing stability: {e}")
            return float("inf")

    @staticmethod
    def sparsity_score(
        explanation_weights: np.ndarray, threshold: float = 0.01, **kwargs
    ) -> float:
        """Compute sparsity score of explanation.

        Measures how sparse (few non-zero weights) the explanation is.

        Args:
            explanation_weights: Explanation weights
            threshold: Threshold for considering a weight as non-zero

        Returns:
            Sparsity score (proportion of weights below threshold)
        """
        try:
            abs_weights = np.abs(explanation_weights)
            sparse_count = np.sum(abs_weights < threshold)
            sparsity = sparse_count / len(explanation_weights)

            return float(sparsity)

        except Exception as e:
            logger.warning(f"Error computing sparsity: {e}")
            return 0.0

    @staticmethod
    def ground_truth_agreement(
        explanation_weights: np.ndarray,
        ground_truth_mask: np.ndarray,
        method: str = "jaccard",
        top_k: Optional[int] = None,
        **kwargs,
    ) -> float:
        """Compute agreement with ground truth importance.

        Args:
            explanation_weights: Explanation weights
            ground_truth_mask: Ground truth binary importance mask
            method: Agreement method ('jaccard', 'precision', 'recall', 'f1')
            top_k: Consider only top-k features (if None, use threshold)

        Returns:
            Agreement score (higher is better)
        """
        try:
            if top_k is not None:
                # Select top-k features by absolute weight
                top_indices = np.argsort(np.abs(explanation_weights))[-top_k:]
                explanation_mask = np.zeros_like(explanation_weights, dtype=bool)
                explanation_mask[top_indices] = True
            else:
                # Use threshold-based selection
                threshold = np.percentile(np.abs(explanation_weights), 75)
                explanation_mask = np.abs(explanation_weights) >= threshold

            # Convert to boolean if needed
            if ground_truth_mask.dtype != bool:
                ground_truth_mask = ground_truth_mask.astype(bool)

            # Compute agreement metrics
            if method == "jaccard":
                intersection = np.sum(explanation_mask & ground_truth_mask)
                union = np.sum(explanation_mask | ground_truth_mask)
                score = intersection / union if union > 0 else 0.0

            elif method == "precision":
                tp = np.sum(explanation_mask & ground_truth_mask)
                fp = np.sum(explanation_mask & ~ground_truth_mask)
                score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            elif method == "recall":
                tp = np.sum(explanation_mask & ground_truth_mask)
                fn = np.sum(~explanation_mask & ground_truth_mask)
                score = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            elif method == "f1":
                tp = np.sum(explanation_mask & ground_truth_mask)
                fp = np.sum(explanation_mask & ~ground_truth_mask)
                fn = np.sum(~explanation_mask & ground_truth_mask)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                score = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

            else:
                raise ValueError(f"Unknown agreement method: {method}")

            return float(score)

        except Exception as e:
            logger.warning(f"Error computing ground truth agreement: {e}")
            return 0.0

    @staticmethod
    def _generate_neighborhood(
        instance: np.ndarray, X_train: np.ndarray, n_samples: int, sigma: float = 0.1
    ) -> np.ndarray:
        """Generate neighborhood samples around an instance."""
        # Use empirical data distribution for more realistic samples
        if len(X_train) >= n_samples:
            # Sample from training data
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            neighborhood = X_train[indices]
        else:
            # Generate Gaussian perturbations
            noise = np.random.normal(0, sigma, (n_samples, len(instance)))
            neighborhood = instance + noise

        return neighborhood


def evaluate_explanations_parallel(
    explainer_instance: Any,
    instances: np.ndarray,
    masks: Optional[np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    predict_fn: Callable,
    metric_fn: Callable,
    n_jobs: int = -1,
    batch_size: int = 10,
    subset_size: Optional[int] = None,
    **kwargs,
) -> tuple:
    """Evaluate explanations in parallel.

    Args:
        explainer_instance: Fitted explainer instance
        instances: Instances to explain
        masks: Ground truth masks (optional)
        X_train: Training data
        y_train: Training labels
        predict_fn: Model prediction function
        metric_fn: Metric function to use
        n_jobs: Number of parallel jobs
        batch_size: Batch size for processing
        subset_size: Limit number of instances to evaluate
        **kwargs: Additional arguments for metric function

    Returns:
        Tuple of (errors, explanations, metadata)
    """
    if subset_size is not None and subset_size < len(instances):
        # Randomly sample subset
        indices = np.random.choice(len(instances), subset_size, replace=False)
        instances = instances[indices]
        if masks is not None:
            masks = masks[indices]

    def evaluate_single_instance(i: int) -> Dict[str, Any]:
        """Evaluate a single instance."""
        try:
            instance = instances[i]
            mask = masks[i] if masks is not None else None

            # Generate explanation
            explanation = explainer_instance.explain_instance(
                instance, predict_fn, X_train, **kwargs
            )

            # Compute metric
            error = metric_fn(
                explanation_weights=explanation["pattern_weights"],
                instance=instance,
                ground_truth_mask=mask,
                predict_fn=predict_fn,
                X_train=X_train,
                **kwargs,
            )

            return {
                "index": i,
                "error": error,
                "explanation": explanation,
                "status": "success",
            }

        except Exception as e:
            logger.warning(f"Failed to evaluate instance {i}: {e}")
            return {
                "index": i,
                "error": float("inf"),
                "explanation": None,
                "status": "failed",
            }

    # Run parallel evaluation
    if n_jobs == 1:
        # Sequential processing
        results = [evaluate_single_instance(i) for i in range(len(instances))]
    else:
        # Parallel processing
        results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(evaluate_single_instance)(i) for i in range(len(instances))
        )

    # Extract results
    errors = [r["error"] for r in results]
    explanations = [r["explanation"] for r in results if r["explanation"] is not None]
    metadata = {
        "n_instances": len(instances),
        "n_successful": sum(1 for r in results if r["status"] == "success"),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
        "mean_error": np.mean([e for e in errors if e != float("inf")]),
        "std_error": np.std([e for e in errors if e != float("inf")]),
    }

    return errors, explanations, metadata


# Predefined metric functions
def create_fidelity_metric(**kwargs) -> Callable:
    """Create a fidelity metric function."""

    def metric_fn(**args):
        return ExplanationMetrics.fidelity_error(**args, **kwargs)

    return metric_fn


def create_stability_metric(**kwargs) -> Callable:
    """Create a stability metric function."""

    def metric_fn(**args):
        return ExplanationMetrics.stability_error(**args, **kwargs)

    return metric_fn


def create_ground_truth_metric(method: str = "jaccard", **kwargs) -> Callable:
    """Create a ground truth agreement metric function."""

    def metric_fn(**args):
        return 1.0 - ExplanationMetrics.ground_truth_agreement(
            **args, method=method, **kwargs
        )

    return metric_fn


def create_combined_metric(
    metrics: List[Callable], weights: Optional[List[float]] = None
) -> Callable:
    """Create a combined metric function."""
    if weights is None:
        weights = [1.0] * len(metrics)

    def metric_fn(**args):
        scores = []
        for metric in metrics:
            try:
                score = metric(**args)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Metric evaluation failed: {e}")
                scores.append(float("inf"))

        # Weighted average
        weighted_score = sum(w * s for w, s in zip(weights, scores)) / sum(weights)
        return weighted_score

    return metric_fn
