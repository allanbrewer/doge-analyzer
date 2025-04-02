"""
Isolation Forest model for anomaly detection based on similarity to a training set.
This module implements an Isolation Forest model for detecting contracts anomalous
relative to a baseline (e.g., identifying active contracts similar to canceled ones).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ContractAnomalyDetector:
    """
    Anomaly detection model for government contracts using Isolation Forest.
    Flags contracts that are similar to the training data as anomalous.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str = "auto",
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize the anomaly detection model.

        Args:
            n_estimators: Number of base estimators in the ensemble
            max_samples: Number of samples to draw from X to train each base estimator
            contamination: Expected proportion of outliers in the data
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
        )
        self.threshold = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the anomaly detection model.

        Args:
            X: Training data with shape (n_samples, n_features)
        """
        logger.info(f"Fitting Isolation Forest model on data with shape {X.shape}")
        self.model.fit(X)
        self.fitted = True

        # Get decision function scores for training data
        scores = self.model.decision_function(X)

        # Set threshold to the 90th percentile of scores (higher scores indicate similarity to training data)
        # This flags the top 10% most similar contracts (relative to training data) as anomalous.
        self.threshold = np.percentile(scores, 75)

        logger.info(f"Model fitted. Threshold set to {self.threshold}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels (1: normal, -1: anomalous).
        Note: Uses the model's internal contamination parameter for prediction,
        which might differ from the custom threshold logic. Prefer `predict_with_threshold`.

        Args:
            X: Data with shape (n_samples, n_features)

        Returns:
            Array of predictions (1: normal, -1: anomalous) based on contamination.
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return np.ones(X.shape[0])  # Default to normal

        logger.info(
            f"Predicting anomalies using internal threshold for data with shape {X.shape}"
        )
        return self.model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for data points.
        Higher scores indicate similarity to the training data (inliers relative to training);
        Lower scores indicate dissimilarity (anomalies relative to training).

        Args:
            X: Data with shape (n_samples, n_features)

        Returns:
            Array of anomaly scores
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return np.zeros(X.shape[0])

        return self.model.decision_function(X)

    def predict_with_threshold(
        self, X: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict anomaly labels using a custom threshold.
        Flags contracts similar to the training data as anomalous.

        Args:
            X: Data with shape (n_samples, n_features)
            threshold: Custom threshold (if None, use the threshold set during fitting)

        Returns:
            Array of predictions (1: normal, -1: anomalous, similar to training data)
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return np.ones(X.shape[0])  # Default to normal

        # Get anomaly scores
        scores = self.decision_function(X)

        # Use custom threshold or the one set during fitting
        threshold = threshold if threshold is not None else self.threshold

        # Predict based on threshold
        predictions = np.ones_like(scores)
        # Flag scores *above* the threshold as anomalous (-1)
        predictions[scores > threshold] = -1

        return predictions

    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.

        Args:
            filepath: Path to save the model
        """
        if not self.fitted:
            logger.warning("Saving unfitted model.")

        model_data = {
            "model": self.model,
            "threshold": self.threshold,
            "fitted": self.fitted,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "ContractAnomalyDetector":
        """
        Load a model from a file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded ContractAnomalyDetector instance
        """
        model_data = joblib.load(filepath)

        detector = cls()
        detector.model = model_data["model"]
        detector.threshold = model_data["threshold"]
        detector.fitted = model_data["fitted"]

        logger.info(f"Model loaded from {filepath}")
        return detector

    def evaluate_model(
        self, X: np.ndarray, y_true: np.ndarray, threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model performance.

        Args:
            X: Data with shape (n_samples, n_features)
            y_true: True labels (1: normal, -1: anomalous)
            threshold: Custom threshold (if None, use the threshold set during fitting)

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return {}

        # Get predictions
        y_pred = self.predict_with_threshold(X, threshold)

        # Calculate metrics
        precision = precision_score(y_true, y_pred, pos_label=-1)
        recall = recall_score(y_true, y_pred, pos_label=-1)
        f1 = f1_score(y_true, y_pred, pos_label=-1)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        logger.info(f"Model evaluation: {metrics}")
        return metrics

    def plot_anomaly_scores(
        self, X: np.ndarray, df: pd.DataFrame, top_n: int = 20
    ) -> None:
        """
        Plot anomaly scores for the most anomalous (similar to training) contracts.

        Args:
            X: Data with shape (n_samples, n_features)
            df: DataFrame containing contract data
            top_n: Number of top similar contracts to plot
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return

        # Get anomaly scores
        scores = self.decision_function(X)

        # Create DataFrame with scores
        score_df = pd.DataFrame(
            {
                "index": range(len(scores)),
                "score": scores,
            }
        )

        # Sort by score (descending, as higher scores are more anomalous/similar to training)
        score_df = score_df.sort_values("score", ascending=False).reset_index(drop=True)

        # Get top N anomalous contracts
        top_anomalies = score_df.head(top_n)

        # Create plot
        plt.figure(figsize=(12, 8))

        # Plot scores
        ax = sns.barplot(x="index", y="score", data=top_anomalies)

        # Add threshold line
        if self.threshold is not None:
            plt.axhline(y=self.threshold, color="r", linestyle="--", label="Threshold")

        # Add labels
        plt.title(f"Top {top_n} Anomalous Contracts (Most Similar to Training Data)")
        plt.xlabel("Contract Index (from sorted list)")
        plt.ylabel("Anomaly Score (higher is more similar to training)")

        # Add contract details as annotations
        if "description" in df.columns and "value" in df.columns:
            for i, row in top_anomalies.iterrows():
                contract_idx = row["index"]
                contract_desc = df.iloc[contract_idx]["description"]
                contract_value = df.iloc[contract_idx]["value"]

                # Truncate description if too long
                if len(contract_desc) > 50:
                    contract_desc = contract_desc[:47] + "..."

                # Add annotation
                ax.annotate(
                    f"${contract_value:,.0f}: {contract_desc}",
                    xy=(i, row["score"]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=45,
                    fontsize=8,
                )

        plt.tight_layout()
        plt.show()
