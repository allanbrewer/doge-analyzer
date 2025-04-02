"""
Functions for evaluating anomaly detection models, particularly Isolation Forest.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # Keep TSNE import for potential future use
from typing import Optional, List

# Assuming ContractAnomalyDetector is importable or its relevant attributes are passed
from doge_analyzer.models.isolation_forest import ContractAnomalyDetector

# Try importing shap, handle gracefully if not installed
try:
    import shap

    SHAP_INSTALLED = True
except ImportError:
    SHAP_INSTALLED = False
    pass  # Keep running even if shap is not installed

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_pca_scatter_anomaly_scores(
    features: np.ndarray,
    scores: np.ndarray,
    output_dir: str,
    random_state: Optional[int] = None,
) -> None:
    """
    Generates a 2D PCA scatter plot of features, colored by anomaly scores.

    Args:
        features: The combined feature array with shape (n_samples, n_features).
        scores: Anomaly scores corresponding to the features.
        output_dir: Directory to save the scatter plot.
        random_state: Random seed for PCA reproducibility.
    """
    logger.info("Running PCA for 2D visualization...")
    if features.shape[0] < 2 or features.shape[1] < 2:
        logger.warning(
            "Not enough samples or features for 2D PCA visualization. Skipping."
        )
        return
    if len(scores) != features.shape[0]:
        logger.warning(
            f"Scores length ({len(scores)}) does not match features length ({features.shape[0]}). Skipping PCA scatter plot."
        )
        return

    try:
        pca_2d = PCA(n_components=2, random_state=random_state)
        features_2d = pca_2d.fit_transform(features)

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            features_2d[:, 0], features_2d[:, 1], c=scores, cmap="viridis", alpha=0.7
        )
        plt.colorbar(scatter, label="Anomaly Score (Higher = More Similar to Training)")
        plt.title("2D PCA of Features Colored by Anomaly Score")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True)
        pca_scatter_path = os.path.join(output_dir, "pca_scatter_anomaly_scores.png")
        plt.savefig(pca_scatter_path)
        plt.close()
        logger.info(f"PCA scatter plot saved to {pca_scatter_path}")

    except Exception as e:
        logger.error(f"PCA visualization failed: {e}", exc_info=True)


def plot_threshold_analysis(
    scores: np.ndarray, fitted_threshold: float, output_dir: str
) -> None:
    """
    Plots the number of flagged contracts vs. the anomaly score threshold.

    Args:
        scores: Anomaly scores for the data points.
        fitted_threshold: The threshold determined during model fitting (e.g., percentile).
        output_dir: Directory to save the threshold analysis plot.
    """
    logger.info("Running threshold analysis...")
    if len(scores) == 0:
        logger.warning("No scores provided for threshold analysis. Skipping.")
        return

    try:
        min_score, max_score = np.min(scores), np.max(scores)
        # Add logging for score range and threshold
        logger.info(
            f"Anomaly scores range for evaluation: [{min_score:.4f}, {max_score:.4f}]"
        )
        logger.info(f"Using fitted threshold: {fitted_threshold:.4f}")

        thresholds = np.linspace(min_score, max_score, 50)
        flagged_counts = []
        for thresh in thresholds:
            # Count how many scores are ABOVE the threshold (anomalous = similar to training)
            count = np.sum(scores > thresh)
            flagged_counts.append(count)

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, flagged_counts, marker=".")
        plt.axvline(
            x=fitted_threshold,
            color="r",
            linestyle="--",
            label=f"Fitted Threshold ({fitted_threshold:.3f})",
        )
        plt.title(
            "Threshold Analysis: Number of Flagged Contracts vs. Anomaly Score Threshold"
        )
        plt.xlabel("Anomaly Score Threshold")
        plt.ylabel("Number of Contracts Flagged (Score > Threshold)")
        plt.legend()
        plt.grid(True)
        threshold_plot_path = os.path.join(output_dir, "threshold_analysis.png")
        plt.savefig(threshold_plot_path)
        plt.close()
        logger.info(f"Threshold analysis plot saved to {threshold_plot_path}")

    except Exception as e:
        logger.error(f"Threshold analysis failed: {e}", exc_info=True)


def run_shap_analysis(
    detector: ContractAnomalyDetector,
    features: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    max_display: int = 20,
) -> None:
    """
    Performs SHAP analysis on the fitted Isolation Forest model.

    Args:
        detector: The fitted ContractAnomalyDetector instance.
        features: The combined feature array used for fitting/evaluation.
        feature_names: List of names corresponding to the feature columns.
        output_dir: Directory to save the SHAP summary plot.
        max_display: Max number of features to display on summary plot.
    """
    if not SHAP_INSTALLED:
        logger.warning("SHAP library not installed. Skipping SHAP analysis.")
        return
    if not detector.fitted:
        logger.warning("Anomaly detector not fitted. Skipping SHAP analysis.")
        return
    if len(feature_names) != features.shape[1]:
        logger.error(
            f"Feature name count ({len(feature_names)}) does not match feature columns ({features.shape[1]}). Skipping SHAP analysis."
        )
        return

    logger.info("Running SHAP analysis...")
    try:
        # Use TreeExplainer for Isolation Forest
        # Note: SHAP values for Isolation Forest represent contribution to the anomaly score.
        # Higher SHAP values for a feature indicate it pushed the score higher (more normal/similar).
        # Lower SHAP values indicate it pushed the score lower (more anomalous/dissimilar).
        explainer = shap.TreeExplainer(detector.model)
        shap_values = explainer.shap_values(
            features
        )  # Get SHAP values for each feature/instance

        # Create SHAP summary plot (bar plot of mean absolute SHAP values)
        os.makedirs(output_dir, exist_ok=True)
        plt.figure()  # Create a new figure explicitly for SHAP plot
        shap.summary_plot(
            shap_values,
            features,
            feature_names=feature_names,
            plot_type="bar",  # Shows mean absolute importance
            max_display=max_display,
            show=False,  # Prevent automatic display, we save manually
        )
        plt.title(f"SHAP Feature Importance (Top {max_display})")
        plt.tight_layout()  # Adjust layout
        shap_plot_path = os.path.join(output_dir, "shap_summary_bar.png")
        plt.savefig(shap_plot_path)
        plt.close()  # Close the plot figure
        logger.info(f"SHAP summary plot saved to {shap_plot_path}")

        # Optional: Add other SHAP plots like beeswarm if needed
        plt.figure()
        shap.summary_plot(
            shap_values,
            features,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title(f"SHAP Feature Summary (Top {max_display})")
        plt.tight_layout()
        shap_beeswarm_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
        plt.savefig(shap_beeswarm_path)
        plt.close()
        logger.info(f"SHAP beeswarm plot saved to {shap_beeswarm_path}")

    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}", exc_info=True)
