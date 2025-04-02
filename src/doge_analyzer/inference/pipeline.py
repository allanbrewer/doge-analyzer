"""
Inference pipeline for contract anomaly detection.
This module handles the end-to-end process of detecting anomalous contracts
(those similar to a baseline training set, e.g., canceled contracts).
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
import glob
from tqdm import tqdm
from joblib import dump, load
import matplotlib.pyplot as plt  # Added
import seaborn as sns  # Added
from sklearn.decomposition import PCA  # Added
from sklearn.manifold import TSNE  # Added
from sklearn.ensemble import RandomForestClassifier  # Added

from doge_analyzer.data.load import load_labeled_data, load_multiple_unlabeled_files
from doge_analyzer.data.preprocess import (
    preprocess_labeled_data,
    preprocess_unlabeled_data,
    align_dataframes,
)
from doge_analyzer.features.text import BertFeatureExtractor
from doge_analyzer.features.fusion import FeatureFusion
from doge_analyzer.models.isolation_forest import ContractAnomalyDetector

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ContractAnomalyPipeline:
    """
    End-to-end pipeline for contract anomaly detection.
    Identifies contracts similar to the training data.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        n_estimators: int = 100,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize the pipeline.

        Args:
            bert_model_name: Name of the BERT model to use
            n_estimators: Number of base estimators for Isolation Forest
            contamination: Expected proportion of outliers in the data
            random_state: Random seed for reproducibility
        """
        self.bert_model_name = bert_model_name
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        # Initialize components
        self.text_extractor = None
        self.feature_fusion = None
        self.anomaly_detector = None

        # Track fitted state
        self.fitted = False

    def fit(
        self,
        labeled_data_path: str,
        batch_size: int = 8,
        output_dir: Optional[str] = None,  # Added output_dir for plots
    ) -> None:
        """
        Fit the pipeline on labeled data, including feature analysis and evaluation plots.

        Args:
            labeled_data_path: Path to the labeled data file
            batch_size: Batch size for BERT feature extraction
            output_dir: Optional directory to save analysis and evaluation plots.
        """
        logger.info(f"Fitting pipeline on labeled data from {labeled_data_path}")

        # --- Data Loading and Preprocessing ---
        labeled_df = load_labeled_data(labeled_data_path)
        processed_labeled_df = preprocess_labeled_data(labeled_df)

        if processed_labeled_df.empty:
            logger.error("Preprocessing resulted in an empty DataFrame. Cannot fit.")
            return

        # --- Text Feature Extraction ---
        logger.info(
            f"Initializing BERT feature extractor with model: {self.bert_model_name}"
        )
        self.text_extractor = BertFeatureExtractor(self.bert_model_name)
        logger.info("Extracting text features from labeled data")
        text_features = self.text_extractor.extract_features_from_df(
            processed_labeled_df, "clean_description", batch_size
        )

        # --- Feature Fusion ---
        logger.info("Initializing and fitting feature fusion")
        self.feature_fusion = FeatureFusion()
        self.feature_fusion.fit(processed_labeled_df)

        logger.info("Combining features")
        # Assuming 'is_canceled' is the label column if present, otherwise use all columns
        cols_to_drop = (
            ["is_canceled"] if "is_canceled" in processed_labeled_df.columns else []
        )
        X_df = processed_labeled_df.drop(columns=cols_to_drop, errors="ignore")

        # Get combined features and feature names
        combined_features, feature_names = self.feature_fusion.transform(
            X_df, text_features
        )

        if combined_features.shape[0] == 0 or combined_features.shape[1] == 0:
            logger.error("Combined features are empty. Cannot proceed with fitting.")
            return
        logger.info(f"Combined features shape: {combined_features.shape}")
        logger.info(f"Number of feature names: {len(feature_names)}")

        # Create analysis output directory if output_dir is provided
        analysis_output_dir = None
        if output_dir:
            analysis_output_dir = os.path.join(output_dir, "analysis")
            os.makedirs(analysis_output_dir, exist_ok=True)
            logger.info(f"Saving analysis plots to: {analysis_output_dir}")

        # --- 1. Feature Space Analysis (Before Fitting Model) ---
        logger.info("--- Starting Feature Space Analysis ---")

        # Ensure we have enough samples for analysis
        if combined_features.shape[0] < 2:
            logger.warning("Not enough samples for feature space analysis. Skipping.")
        else:
            # --- PCA ---
            logger.info("Running PCA analysis...")
            try:
                pca = PCA(random_state=self.random_state)
                pca.fit(combined_features)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
                logger.info(
                    f"PCA: {n_components_95} components explain >= 95% variance."
                )

                if analysis_output_dir:
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
                    plt.title("PCA Cumulative Explained Variance")
                    plt.xlabel("Number of Components")
                    plt.ylabel("Cumulative Explained Variance Ratio")
                    plt.legend()
                    plt.grid(True)
                    pca_plot_path = os.path.join(
                        analysis_output_dir, "pca_explained_variance.png"
                    )
                    plt.savefig(pca_plot_path)
                    plt.close()
                    logger.info(f"PCA variance plot saved to {pca_plot_path}")
            except Exception as e:
                logger.error(f"PCA analysis failed: {e}", exc_info=True)

            # --- Feature Importance (Proxy using RandomForest) ---
            logger.info("Calculating feature importance proxy using RandomForest...")
            try:
                # Create dummy labels (all ones, as it's unsupervised)
                dummy_y = np.ones(combined_features.shape[0])
                rf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=10,
                )  # Limit depth for speed
                rf.fit(combined_features, dummy_y)
                importances = rf.feature_importances_

                importance_df = pd.DataFrame(
                    {"feature": feature_names, "importance": importances}
                )
                importance_df = importance_df.sort_values(
                    "importance", ascending=False
                ).reset_index(drop=True)

                # Identify low importance features (threshold might need tuning)
                low_importance_threshold = 0.001  # Example threshold
                low_importance_features = importance_df[
                    importance_df["importance"] < low_importance_threshold
                ]["feature"].tolist()
                logger.info(
                    f"Found {len(low_importance_features)} features with importance < {low_importance_threshold} (proxy)."
                )
                # logger.debug(f"Low importance features (proxy): {low_importance_features}") # Potentially very long list

                if analysis_output_dir:
                    # Plot top N features
                    n_top_features = min(30, len(importance_df))
                    plt.figure(figsize=(12, 8))
                    sns.barplot(
                        x="importance",
                        y="feature",
                        data=importance_df.head(n_top_features),
                    )
                    plt.title(
                        f"Top {n_top_features} Feature Importances (RandomForest Proxy)"
                    )
                    plt.tight_layout()
                    importance_plot_path = os.path.join(
                        analysis_output_dir, "feature_importance_proxy.png"
                    )
                    plt.savefig(importance_plot_path)
                    plt.close()
                    logger.info(
                        f"Feature importance plot saved to {importance_plot_path}"
                    )
            except Exception as e:
                logger.error(
                    f"Feature importance proxy calculation failed: {e}", exc_info=True
                )

            # --- Correlation Analysis ---
            logger.info("Running correlation analysis...")
            try:
                # Create DataFrame for correlation calculation
                features_df = pd.DataFrame(combined_features, columns=feature_names)
                correlation_matrix = features_df.corr()

                # Identify highly correlated pairs
                high_corr_threshold = 0.9  # Example threshold
                upper_tri = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
                )
                high_corr_pairs = [
                    (column, upper_tri.index[row_idx])
                    for row_idx, column in enumerate(upper_tri.columns)
                    if any(abs(upper_tri[column]) > high_corr_threshold)
                    for val in upper_tri[column]
                    if abs(val) > high_corr_threshold  # Find specific pairs
                ]
                # More precise way to get pairs
                high_corr_pairs_precise = []
                for i in range(len(upper_tri.columns)):
                    for j in range(i):
                        if abs(upper_tri.iloc[j, i]) > high_corr_threshold:
                            high_corr_pairs_precise.append(
                                (upper_tri.columns[i], upper_tri.index[j])
                            )

                logger.info(
                    f"Found {len(high_corr_pairs_precise)} feature pairs with absolute correlation > {high_corr_threshold}."
                )
                # logger.debug(f"Highly correlated pairs (> {high_corr_threshold}): {high_corr_pairs_precise}") # Potentially long

                if analysis_output_dir:
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(
                        correlation_matrix, cmap="coolwarm", center=0, annot=False
                    )  # Annot=False for large matrices
                    plt.title("Feature Correlation Matrix")
                    plt.tight_layout()
                    corr_plot_path = os.path.join(
                        analysis_output_dir, "correlation_heatmap.png"
                    )
                    plt.savefig(corr_plot_path)
                    plt.close()
                    logger.info(f"Correlation heatmap saved to {corr_plot_path}")
            except Exception as e:
                logger.error(f"Correlation analysis failed: {e}", exc_info=True)

        logger.info("--- Feature Space Analysis Complete ---")
        # --- Decision: For now, we log the analysis but don't modify combined_features ---
        # Based on the logs and plots saved in analysis_output_dir, a user could decide
        # whether to apply PCA or select features in a subsequent step.

        # --- 2. Fit the Isolation Forest Model ---
        logger.info("Initializing and fitting anomaly detector (Isolation Forest)")
        self.anomaly_detector = ContractAnomalyDetector(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.anomaly_detector.fit(
            combined_features
        )  # Fit on the original combined features

        # --- 3. Model Evaluation ---
        logger.info("--- Starting Model Evaluation ---")
        # Create evaluation output directory
        evaluation_output_dir = None
        if output_dir:
            evaluation_output_dir = os.path.join(output_dir, "evaluation")
            os.makedirs(evaluation_output_dir, exist_ok=True)
            logger.info(f"Saving evaluation plots to: {evaluation_output_dir}")

        # Get anomaly scores from the fitted model
        anomaly_scores = self.anomaly_detector.decision_function(combined_features)
        min_score, max_score = np.min(anomaly_scores), np.max(anomaly_scores)
        logger.info(f"Anomaly scores range: [{min_score:.4f}, {max_score:.4f}]")
        logger.info(f"Fitted threshold: {self.anomaly_detector.threshold:.4f}")

        # --- Visual Inspection with Dimensionality Reduction (PCA) ---
        logger.info("Running PCA for 2D visualization...")
        try:
            if combined_features.shape[1] >= 2:  # Need at least 2 features for PCA 2D
                pca_2d = PCA(n_components=2, random_state=self.random_state)
                features_2d = pca_2d.fit_transform(combined_features)

                if evaluation_output_dir:
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(
                        features_2d[:, 0],
                        features_2d[:, 1],
                        c=anomaly_scores,
                        cmap="viridis",
                        alpha=0.7,
                    )
                    plt.colorbar(
                        scatter,
                        label="Anomaly Score (Higher = More Similar to Training)",
                    )
                    plt.title("2D PCA of Features Colored by Anomaly Score")
                    plt.xlabel("Principal Component 1")
                    plt.ylabel("Principal Component 2")
                    plt.grid(True)
                    pca_scatter_path = os.path.join(
                        evaluation_output_dir, "pca_scatter_anomaly_scores.png"
                    )
                    plt.savefig(pca_scatter_path)
                    plt.close()
                    logger.info(f"PCA scatter plot saved to {pca_scatter_path}")
            else:
                logger.warning("Not enough features for 2D PCA visualization.")
        except Exception as e:
            logger.error(f"PCA visualization failed: {e}", exc_info=True)

        # --- Threshold Analysis ---
        logger.info("Running threshold analysis...")
        try:
            thresholds = np.linspace(min_score, max_score, 50)
            flagged_counts = []
            for thresh in thresholds:
                # Count how many scores are ABOVE the threshold (anomalous = similar to training)
                count = np.sum(anomaly_scores > thresh)
                flagged_counts.append(count)

            if evaluation_output_dir:
                plt.figure(figsize=(10, 6))
                plt.plot(thresholds, flagged_counts, marker=".")
                plt.axvline(
                    x=self.anomaly_detector.threshold,
                    color="r",
                    linestyle="--",
                    label=f"Fitted Threshold ({self.anomaly_detector.threshold:.3f})",
                )
                plt.title(
                    "Threshold Analysis: Number of Flagged Contracts vs. Anomaly Score Threshold"
                )
                plt.xlabel("Anomaly Score Threshold")
                plt.ylabel("Number of Contracts Flagged (Score > Threshold)")
                plt.legend()
                plt.grid(True)
                threshold_plot_path = os.path.join(
                    evaluation_output_dir, "threshold_analysis.png"
                )
                plt.savefig(threshold_plot_path)
                plt.close()
                logger.info(f"Threshold analysis plot saved to {threshold_plot_path}")
        except Exception as e:
            logger.error(f"Threshold analysis failed: {e}", exc_info=True)

        logger.info("--- Model Evaluation Complete ---")

        self.fitted = True
        logger.info(
            "Pipeline fitting (including analysis and evaluation) completed successfully"
        )

    def predict(
        self,
        unlabeled_data_paths: Union[str, List[str]],
        output_dir: Optional[str] = None,
        batch_size: int = 8,
        threshold: Optional[float] = None,
        extract_dir: Optional[str] = None,
        sample_size: Optional[int] = None,
        department_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Predict anomalies in unlabeled data (flags contracts similar to training data).

        Args:
            unlabeled_data_paths: Path(s) to unlabeled data files or directories
            output_dir: Directory to save results
            batch_size: Batch size for BERT feature extraction
            threshold: Custom threshold for anomaly detection (overrides fitted threshold)
            extract_dir: Directory to extract zip files
            sample_size: Number of contracts to sample
            department_filter: Filter to only include files from a specific department

        Returns:
            DataFrame with anomaly predictions and scores
        """
        if not self.fitted:
            logger.error("Pipeline not fitted. Call fit() first.")
            return pd.DataFrame()

        logger.info(f"Predicting anomalies using input: {unlabeled_data_paths}")

        # Load unlabeled data using the updated function
        unlabeled_df = load_multiple_unlabeled_files(
            input_paths=unlabeled_data_paths,
            extract_dir=extract_dir,
            sample_size=sample_size,
            department_filter=department_filter,
        )

        if unlabeled_df.empty:
            logger.error("No unlabeled data loaded")
            return pd.DataFrame()

        # Preprocess unlabeled data
        processed_unlabeled_df = preprocess_unlabeled_data(
            unlabeled_df,
            self.feature_fusion.numerical_columns
            + self.feature_fusion.categorical_columns,
        )

        # Extract text features
        logger.info("Extracting text features from unlabeled data")
        text_features = self.text_extractor.extract_features_from_df(
            processed_unlabeled_df, "clean_description", batch_size
        )

        # Combine features for prediction
        logger.info("Combining features for prediction")
        if self.text_extractor is not None:
            # Unpack the tuple, only keep the features array
            combined_features, _ = self.feature_fusion.transform(
                processed_unlabeled_df, text_features
            )
        else:
            # Unpack the tuple, only keep the features array
            combined_features, _ = self.feature_fusion.transform(
                processed_unlabeled_df, np.array([])
            )

        # Predict anomalies
        logger.info("Predicting anomalies")
        # Get raw anomaly scores (higher means more similar to training data)
        anomaly_scores = self.anomaly_detector.decision_function(combined_features)
        # Get anomaly labels (-1 for anomalous/similar, 1 for normal) using the percentile threshold
        anomaly_labels_threshold = self.anomaly_detector.predict_with_threshold(
            combined_features, threshold  # Uses detector's fitted threshold if None
        )
        # Get anomaly labels (-1 for anomalous, 1 for normal) using the contamination parameter's internal threshold
        logger.info(
            f"Predicting anomalies using contamination factor ({self.contamination})"
        )
        anomaly_labels_contamination = self.anomaly_detector.predict(combined_features)

        # Add predictions to DataFrame
        result_df = processed_unlabeled_df.copy()
        # Store raw anomaly scores (higher indicates similarity)
        result_df["anomaly_score"] = anomaly_scores
        # Flag contracts predicted as inliers (1, similar to training) by standard predict() based on contamination
        result_df["is_anomalous_by_contamination"] = anomaly_labels_contamination == 1
        # Flag contracts predicted as anomalous (-1, high scores) by predict_with_threshold() based on percentile threshold for review
        result_df["for_review"] = anomaly_labels_threshold == -1

        # Sort by anomaly score (descending, higher scores are more similar/anomalous)
        result_df = result_df.sort_values("anomaly_score", ascending=False).reset_index(
            drop=True
        )

        # Save results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"anomaly_predictions_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv",
            )
            result_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

        # Log counts for both flags (both now indicate similarity to training data)
        count_anomalous_contamination = result_df["is_anomalous_by_contamination"].sum()
        count_anomalous_threshold = result_df["for_review"].sum()
        total_count = len(result_df)
        logger.info(f"Anomaly detection complete. Total contracts: {total_count}.")
        logger.info(
            f"Flagged as ANOMALOUS by contamination (inliers, predict() == 1): {count_anomalous_contamination}."
        )
        logger.info(
            f"Flagged for review as ANOMALOUS by threshold (score > {self.anomaly_detector.threshold:.4f}): {count_anomalous_threshold}."
        )

        return result_df

    def save_pipeline(self, output_dir: str) -> None:
        """
        Save the pipeline components to files.

        Args:
            output_dir: Directory to save the pipeline components
        """
        if not self.fitted:
            logger.warning("Saving unfitted pipeline")

        os.makedirs(output_dir, exist_ok=True)

        # Save anomaly detector
        model_path = os.path.join(output_dir, "anomaly_detector.joblib")
        if self.anomaly_detector:
            self.anomaly_detector.save_model(model_path)
        else:
            logger.warning("Anomaly detector not available to save.")

        # Save feature fusion model
        fusion_dir = os.path.join(output_dir, "feature_fusion")
        self.feature_fusion.save_model(fusion_dir)

        logger.info(f"Pipeline saved to {output_dir}")

    @classmethod
    def load_pipeline(
        cls,
        model_dir: str,
        # labeled_data_path: str, # No longer needed directly for loading
        bert_model_name: str = "bert-base-uncased",
    ) -> "ContractAnomalyPipeline":
        """
        Load a pipeline from files.

        Args:
            model_dir: Directory containing the saved pipeline components (detector and fusion)
            # labeled_data_path: Path to the labeled data file (not directly used in loading)
            bert_model_name: Name of the BERT model to use

        Returns:
            Loaded ContractAnomalyPipeline instance
        """
        # Initialize pipeline
        pipeline = cls(bert_model_name=bert_model_name)

        # Load anomaly detector
        model_path = os.path.join(model_dir, "anomaly_detector.joblib")
        if os.path.exists(model_path):
            pipeline.anomaly_detector = ContractAnomalyDetector.load_model(model_path)
        else:
            logger.error(f"Anomaly detector model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize text feature extractor
        pipeline.text_extractor = BertFeatureExtractor(bert_model_name)

        # Initialize feature fusion
        pipeline.feature_fusion = FeatureFusion()
        fusion_dir = os.path.join(model_dir, "feature_fusion")
        pipeline.feature_fusion = FeatureFusion.load_model(fusion_dir)

        pipeline.fitted = True

        logger.info(f"Pipeline loaded from {model_dir}")
        return pipeline


def run_pipeline(
    labeled_data_path: str,
    unlabeled_data_paths: Union[str, List[str]],
    output_dir: str,
    bert_model_name: str = "bert-base-uncased",
    n_estimators: int = 100,
    contamination: float = 0.1,
    batch_size: int = 8,
    threshold: Optional[float] = None,
    extract_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
    department_filter: Optional[str] = None,
    save_model: bool = True,
) -> pd.DataFrame:
    """
    Run the complete pipeline from data loading to anomaly prediction.

    Args:
        labeled_data_path: Path to the labeled data file
        unlabeled_data_paths: Path(s) to unlabeled data files or directories
        output_dir: Directory to save results and model
        bert_model_name: Name of the BERT model to use
        n_estimators: Number of base estimators for Isolation Forest
        contamination: Expected proportion of outliers in the data
        batch_size: Batch size for BERT feature extraction
        threshold: Custom threshold for anomaly detection
        extract_dir: Directory to extract zip files
        sample_size: Number of contracts to sample
        department_filter: Filter to only include files from a specific department
        save_model: Whether to save the trained model

    Returns:
        DataFrame with anomaly predictions
    """
    # Initialize pipeline
    pipeline = ContractAnomalyPipeline(
        bert_model_name=bert_model_name,
        n_estimators=n_estimators,
        contamination=contamination,  # Passed to detector, but threshold logic is primary
    )

    # Fit pipeline
    pipeline.fit(labeled_data_path, batch_size=batch_size)

    # Save pipeline if requested
    if save_model:
        model_dir = os.path.join(output_dir, "model")
        pipeline.save_pipeline(model_dir)

    # Predict anomalies
    result_df = pipeline.predict(
        unlabeled_data_paths,
        output_dir=output_dir,
        batch_size=batch_size,
        threshold=threshold,
        extract_dir=extract_dir,
        sample_size=sample_size,
        department_filter=department_filter,
    )

    return result_df
