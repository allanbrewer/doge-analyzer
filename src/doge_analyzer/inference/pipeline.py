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
    ) -> None:
        """
        Fit the pipeline on labeled data.

        Args:
            labeled_data_path: Path to the labeled data file
            batch_size: Batch size for BERT feature extraction

        Args:
            labeled_data_path: Path to the labeled data file
            batch_size: Batch size for BERT feature extraction
        """
        logger.info(f"Fitting pipeline on labeled data from {labeled_data_path}")

        # Load labeled data
        labeled_df = load_labeled_data(labeled_data_path)

        # Preprocess labeled data
        processed_labeled_df = preprocess_labeled_data(labeled_df)

        # Initialize text feature extractor
        logger.info(
            f"Initializing BERT feature extractor with model: {self.bert_model_name}"
        )
        self.text_extractor = BertFeatureExtractor(self.bert_model_name)

        # Extract text features
        logger.info("Extracting text features from labeled data")
        text_features = self.text_extractor.extract_features_from_df(
            processed_labeled_df, "clean_description", batch_size
        )

        # Initialize feature fusion
        logger.info("Initializing feature fusion")
        self.feature_fusion = FeatureFusion()

        # Fit feature fusion
        logger.info("Fitting feature fusion")
        self.feature_fusion.fit(processed_labeled_df)

        # Combine features
        logger.info("Combining features")
        X = processed_labeled_df.drop("is_canceled", axis=1)
        combined_features = self.feature_fusion.transform(X, text_features)

        # Initialize and fit anomaly detector
        logger.info("Initializing and fitting anomaly detector")
        self.anomaly_detector = ContractAnomalyDetector(
            n_estimators=self.n_estimators,
            contamination=self.contamination,  # Note: contamination is used by predict(), not predict_with_threshold()
            random_state=self.random_state,
        )
        # Fit using the 90th percentile threshold defined in the detector's fit method
        self.anomaly_detector.fit(combined_features)

        self.fitted = True
        logger.info("Pipeline fitted successfully")

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

        # Combine features
        logger.info("Combining features")
        if self.text_extractor is not None:
            combined_features = self.feature_fusion.transform(
                processed_unlabeled_df, text_features
            )
        else:
            combined_features = self.feature_fusion.transform(
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
