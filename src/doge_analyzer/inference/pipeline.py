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

# Removed plt, sns, PCA, TSNE, RandomForestClassifier imports as they are now in separate modules

from doge_analyzer.data.load import load_labeled_data, load_multiple_unlabeled_files
from doge_analyzer.data.process import (  # Renamed module
    preprocess_labeled_data,
    preprocess_unlabeled_data,
    align_dataframes,
)
from doge_analyzer.features.text import BertFeatureExtractor
from doge_analyzer.features.fusion import FeatureFusion
from doge_analyzer.models.isolation_forest import ContractAnomalyDetector

# Added imports for new modules
from doge_analyzer.analysis.feature_analysis import (
    run_pca_analysis,
    run_correlation_analysis,
)
from doge_analyzer.evaluation.model_evaluation import (
    plot_pca_scatter_anomaly_scores,
    plot_threshold_analysis,
    run_shap_analysis,  # Added run_shap_analysis
)

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

        # Track fitted state and data from fit
        self.fitted = False
        self.labeled_piids: set = set()  # Store PIIDs from labeled data

    def fit(
        self,
        labeled_data_path: str,
        batch_size: int = 8,
        output_dir: Optional[str] = None,  # Keep output_dir for plots
    ) -> None:
        """
        Fit the pipeline on labeled data. Optionally performs analysis on text features
        and evaluation plotting if output_dir is provided.

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
            logger.error(
                "Preprocessing labeled data resulted in an empty DataFrame. Cannot fit."
            )
            return

        # Store PIIDs from the labeled data for filtering during prediction
        if "piid" in processed_labeled_df.columns:
            self.labeled_piids = set(processed_labeled_df["piid"].dropna().unique())
            logger.info(
                f"Stored {len(self.labeled_piids)} unique PIIDs from labeled data."
            )
        else:
            logger.warning(
                "Column 'piid' not found in labeled data. Cannot filter unlabeled data later."
            )
            self.labeled_piids = set()

        # --- Text Feature Extraction ---
        logger.info(
            f"Initializing BERT feature extractor with model: {self.bert_model_name}"
        )
        self.text_extractor = BertFeatureExtractor(self.bert_model_name)
        logger.info("Extracting text features from labeled data")
        text_features = self.text_extractor.extract_features_from_df(
            processed_labeled_df, "clean_description", batch_size
        )

        # --- Text Feature Analysis (Optional) ---
        if output_dir and text_features.shape[0] > 0 and text_features.shape[1] > 0:
            analysis_output_dir = os.path.join(output_dir, "analysis")
            logger.info(
                f"--- Running Text Feature Analysis (Saving to {analysis_output_dir}) ---"
            )
            # Generate generic names for text features for correlation plot
            num_text_dims = text_features.shape[1]
            text_feature_names = [f"text_{i}" for i in range(num_text_dims)]

            run_pca_analysis(
                features=text_features,
                output_dir=analysis_output_dir,
                random_state=self.random_state,
                prefix="text_embedding",
            )
            run_correlation_analysis(
                features=text_features,
                feature_names=text_feature_names,
                output_dir=analysis_output_dir,
                prefix="text_embedding",
            )
            logger.info("--- Text Feature Analysis Complete ---")
        elif output_dir:
            logger.warning(
                "Skipping text feature analysis due to empty or invalid text features."
            )

        # --- Feature Fusion ---
        logger.info("Initializing and fitting feature fusion")
        self.feature_fusion = FeatureFusion()
        self.feature_fusion.fit(processed_labeled_df)

        logger.info("Combining features")
        cols_to_drop = (
            ["is_canceled"] if "is_canceled" in processed_labeled_df.columns else []
        )
        X_df = processed_labeled_df.drop(columns=cols_to_drop, errors="ignore")

        # Get combined features AND feature names (needed for SHAP later)
        combined_features, feature_names = self.feature_fusion.transform(
            X_df, text_features
        )

        if combined_features.shape[0] == 0 or combined_features.shape[1] == 0:
            logger.error("Combined features are empty. Cannot proceed with fitting.")
            return
        logger.info(f"Combined features shape for fitting: {combined_features.shape}")
        logger.info(
            f"Number of feature names: {len(feature_names)}"
        )  # Log feature name count

        # --- Fit the Isolation Forest Model ---
        logger.info("Initializing and fitting anomaly detector (Isolation Forest)")
        self.anomaly_detector = ContractAnomalyDetector(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.anomaly_detector.fit(combined_features)

        # --- Model Evaluation (Optional) ---
        if output_dir:
            evaluation_output_dir = os.path.join(output_dir, "evaluation")
            logger.info(
                f"--- Running Model Evaluation (Saving to {evaluation_output_dir}) ---"
            )
            # Get anomaly scores from the fitted model
            anomaly_scores = self.anomaly_detector.decision_function(combined_features)
            logger.info(
                f"Anomaly scores range: [{np.min(anomaly_scores):.4f}, {np.max(anomaly_scores):.4f}]"
            )
            logger.info(f"Fitted threshold: {self.anomaly_detector.threshold:.4f}")

            plot_pca_scatter_anomaly_scores(
                features=combined_features,
                scores=anomaly_scores,
                output_dir=evaluation_output_dir,
                random_state=self.random_state,
            )
            plot_threshold_analysis(
                scores=anomaly_scores,
                fitted_threshold=self.anomaly_detector.threshold,
                output_dir=evaluation_output_dir,
            )
            # Add SHAP analysis call
            run_shap_analysis(
                detector=self.anomaly_detector,
                features=combined_features,
                feature_names=feature_names,  # Pass feature names
                output_dir=evaluation_output_dir,
            )
            logger.info("--- Model Evaluation Complete ---")

        self.fitted = True
        logger.info("Pipeline fitting completed successfully")

    def predict(
        self,
        unlabeled_data_paths: Union[str, List[str]],
        output_dir: Optional[str] = None,
        batch_size: int = 8,
        threshold: Optional[float] = None,
        sample_size: Optional[int] = None,
        # Removed unused parameters like extract_dir, department_filter
    ) -> pd.DataFrame:
        """
        Predict anomalies in unlabeled data (flags contracts similar to training data).

        Args:
            unlabeled_data_paths: Path(s) to unlabeled data files or directories
            output_dir: Directory to save results
            batch_size: Batch size for BERT feature extraction
            threshold: Custom threshold for anomaly detection (overrides fitted threshold)
            sample_size: Number of contracts to sample (applied after loading all files)

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
            sample_size=sample_size,  # Sampling happens after combining files in load_multiple_unlabeled_files
        )

        if unlabeled_df.empty:
            logger.error("No unlabeled data loaded")
            return pd.DataFrame()

        # Preprocess unlabeled data, passing labeled PIIDs for filtering
        logger.info(
            f"Preprocessing unlabeled data and filtering against {len(self.labeled_piids)} labeled PIIDs."
        )
        processed_unlabeled_df = preprocess_unlabeled_data(
            unlabeled_df,
            self.feature_fusion.numerical_columns
            + self.feature_fusion.categorical_columns,
            self.labeled_piids,  # Pass the stored PIIDs
        )

        if processed_unlabeled_df.empty:
            logger.error(
                "Preprocessing unlabeled data resulted in an empty DataFrame (possibly after filtering). Cannot predict."
            )
            return pd.DataFrame()

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
            logger.info(f"Full results saved to {output_path}")

            # --- Save filtered results ---
            try:
                logger.info("Filtering results for 'for_review' contracts...")
                filtered_df = result_df[result_df["for_review"] == True].copy()

                # Define desired columns and order
                final_columns = [
                    "piid",
                    "description",
                    "value",
                    "vendor",
                    "agency",
                    "end_date",  # This column might not exist if grants were processed
                    "anomaly_score",
                    "for_review",
                ]

                # Select only existing columns from the desired list
                existing_final_columns = [
                    col for col in final_columns if col in filtered_df.columns
                ]

                if not existing_final_columns:
                    logger.warning(
                        "No specified columns found in the filtered data. Cannot save filtered file."
                    )
                else:
                    # Reorder and select
                    filtered_df_final = filtered_df[existing_final_columns]

                    # Define final output path
                    final_output_path = os.path.join(
                        output_dir, "anomaly_predictions.csv"
                    )
                    filtered_df_final.to_csv(final_output_path, index=False)
                    logger.info(
                        f"Filtered results ('for_review' = True) saved to {final_output_path} with columns: {existing_final_columns}"
                    )

            except Exception as e:
                logger.error(f"Error saving filtered results: {e}")

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
    sample_size: Optional[int] = None,
    save_model: bool = True,
    # Removed unused parameters like extract_dir, department_filter
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
        sample_size: Number of contracts to sample
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
    pipeline.fit(labeled_data_path, batch_size=batch_size, output_dir=output_dir)

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
        sample_size=sample_size,
        # Pass other relevant args if needed, but extract_dir/dept_filter are handled upstream or removed
    )

    return result_df
