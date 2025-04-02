"""
Functions for analyzing feature spaces, particularly text embeddings.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import List, Optional

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pca_analysis(
    features: np.ndarray,
    output_dir: str,
    random_state: Optional[int] = None,
    prefix: str = "text_embedding",
) -> None:
    """
    Performs PCA on the given features and plots the cumulative explained variance.

    Args:
        features: The feature array (e.g., text embeddings) with shape (n_samples, n_features).
        output_dir: Directory to save the PCA plot.
        random_state: Random seed for PCA reproducibility.
        prefix: Prefix for plot filename.
    """
    logger.info(f"Running PCA analysis on features with shape {features.shape}...")
    if features.shape[0] < 2 or features.shape[1] < 2:
        logger.warning("Not enough samples or features for PCA analysis. Skipping.")
        return

    try:
        pca = PCA(random_state=random_state)
        pca.fit(features)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        logger.info(
            f"PCA ({prefix}): {n_components_95} components explain >= 95% variance."
        )

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            marker="o",
            linestyle="--",
        )
        plt.axhline(y=0.95, color="r", linestyle=":", label="95% Threshold")
        plt.axvline(
            x=n_components_95,
            color="g",
            linestyle=":",
            label=f"{n_components_95} Components",
        )
        plt.title(f"PCA Cumulative Explained Variance ({prefix})")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.legend()
        plt.grid(True)
        pca_plot_path = os.path.join(output_dir, f"{prefix}_pca_explained_variance.png")
        plt.savefig(pca_plot_path)
        plt.close()
        logger.info(f"PCA variance plot saved to {pca_plot_path}")

    except Exception as e:
        logger.error(f"PCA analysis failed for {prefix}: {e}", exc_info=True)


def run_correlation_analysis(
    features: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    prefix: str = "text_embedding",
) -> None:
    """
    Calculates and plots the correlation matrix for the given features.

    Args:
        features: The feature array (e.g., text embeddings) with shape (n_samples, n_features).
        feature_names: List of names corresponding to the feature columns.
        output_dir: Directory to save the correlation heatmap.
        prefix: Prefix for plot filename.
    """
    logger.info(
        f"Running correlation analysis on features with shape {features.shape}..."
    )
    if features.shape[0] < 2 or features.shape[1] < 2:
        logger.warning(
            "Not enough samples or features for correlation analysis. Skipping."
        )
        return
    if len(feature_names) != features.shape[1]:
        logger.error(
            f"Feature name count ({len(feature_names)}) does not match feature columns ({features.shape[1]}) for {prefix}. Skipping correlation."
        )
        return

    try:
        features_df = pd.DataFrame(features, columns=feature_names)
        correlation_matrix = features_df.corr()

        # Identify highly correlated pairs
        high_corr_threshold = 0.9  # Example threshold
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        high_corr_pairs_precise = []
        for i in range(len(upper_tri.columns)):
            for j in range(i):
                if abs(upper_tri.iloc[j, i]) > high_corr_threshold:
                    high_corr_pairs_precise.append(
                        (upper_tri.columns[i], upper_tri.index[j])
                    )

        logger.info(
            f"Correlation ({prefix}): Found {len(high_corr_pairs_precise)} feature pairs with absolute correlation > {high_corr_threshold}."
        )

        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix, cmap="coolwarm", center=0, annot=False
        )  # Annot=False for potentially large matrices
        plt.title(f"Feature Correlation Matrix ({prefix})")
        plt.tight_layout()
        corr_plot_path = os.path.join(output_dir, f"{prefix}_correlation_heatmap.png")
        plt.savefig(corr_plot_path)
        plt.close()
        logger.info(f"Correlation heatmap saved to {corr_plot_path}")

        # Save the list of highly correlated pairs
        import json

        corr_pairs_path = os.path.join(
            output_dir, f"{prefix}_high_correlation_pairs.json"
        )
        try:
            with open(corr_pairs_path, "w") as f:
                json.dump(high_corr_pairs_precise, f, indent=4)
            logger.info(
                f"Highly correlated pairs (>{high_corr_threshold}) saved to {corr_pairs_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save correlation pairs to {corr_pairs_path}: {e}")

    except Exception as e:
        logger.error(f"Correlation analysis failed for {prefix}: {e}", exc_info=True)
