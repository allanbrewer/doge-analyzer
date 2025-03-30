"""
Inference pipeline for contract similarity detection.
This module handles the end-to-end process of detecting similar contracts.
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
from doge_analyzer.models.isolation_forest import ContractSimilarityDetector

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ContractSimilarityPipeline:
    """
    End-to-end pipeline for contract similarity detection.
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
        self.similarity_detector = None

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

        # Initialize and fit similarity detector
        logger.info("Initializing and fitting similarity detector")
        self.similarity_detector = ContractSimilarityDetector(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.similarity_detector.fit(combined_features)

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
        Predict similarities in unlabeled data.

        Args:
            unlabeled_data_paths: Path(s) to unlabeled data files or directories
            output_dir: Directory to save results
            batch_size: Batch size for BERT feature extraction
            threshold: Custom threshold for similarity detection
            extract_dir: Directory to extract zip files
            sample_size: Number of contracts to sample
            department_filter: Filter to only include files from a specific department

        Returns:
            DataFrame with similarity predictions
        """
        if not self.fitted:
            logger.error("Pipeline not fitted. Call fit() first.")
            return pd.DataFrame()

        logger.info(f"Predicting similarities using input: {unlabeled_data_paths}")

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

        # Predict similarities
        logger.info("Predicting similarities")
        similarity_scores = self.similarity_detector.decision_function(
            combined_features
        )
        similarity_labels = self.similarity_detector.predict_with_threshold(
            combined_features, threshold
        )

        # Add predictions to DataFrame
        result_df = processed_unlabeled_df.copy()
        # Invert scores to represent similarity to training data
        result_df["similarity_score"] = -similarity_scores
        result_df["for_review"] = similarity_labels == -1

        # Sort by similarity score (descending, as higher scores are more similar)
        result_df = result_df.sort_values(
            "similarity_score", ascending=False
        ).reset_index(drop=True)

        # Save results if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"similarity_predictions_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv",
            )
            result_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

        logger.info(
            f"Found {result_df['for_review'].sum()} similar contracts out of {len(result_df)}"
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

        # Save similarity detector
        model_path = os.path.join(output_dir, "similarity_detector.joblib")
        self.similarity_detector.save_model(model_path)

        # Save feature fusion model
        fusion_dir = os.path.join(output_dir, "feature_fusion")
        self.feature_fusion.save_model(fusion_dir)

        logger.info(f"Pipeline saved to {output_dir}")

    @classmethod
    def load_pipeline(
        cls,
        model_dir: str,
        labeled_data_path: str,
        bert_model_name: str = "bert-base-uncased",
    ) -> "ContractSimilarityPipeline":
        """
        Load a pipeline from files.

        Args:
            model_dir: Directory containing the saved pipeline components
            labeled_data_path: Path to the labeled data file
            bert_model_name: Name of the BERT model to use

        Returns:
            Loaded ContractSimilarityPipeline instance
        """
        # Initialize pipeline
        pipeline = cls(bert_model_name=bert_model_name)

        # Load similarity detector
        model_path = os.path.join(model_dir, "similarity_detector.joblib")
        pipeline.similarity_detector = ContractSimilarityDetector.load_model(model_path)

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
    Run the complete pipeline from data loading to similarity prediction.

    Args:
        labeled_data_path: Path to the labeled data file
        unlabeled_data_paths: Path(s) to unlabeled data files or directories
        output_dir: Directory to save results and model
        bert_model_name: Name of the BERT model to use
        n_estimators: Number of base estimators for Isolation Forest
        contamination: Expected proportion of outliers in the data
        batch_size: Batch size for BERT feature extraction
        threshold: Custom threshold for similarity detection
        extract_dir: Directory to extract zip files
        sample_size: Number of contracts to sample
        department_filter: Filter to only include files from a specific department
        save_model: Whether to save the trained model

    Returns:
        DataFrame with similarity predictions
    """
    # Initialize pipeline
    pipeline = ContractSimilarityPipeline(
        bert_model_name=bert_model_name,
        n_estimators=n_estimators,
        contamination=contamination,
    )

    # Fit pipeline
    pipeline.fit(labeled_data_path, batch_size=batch_size)

    # Save pipeline if requested
    if save_model:
        model_dir = os.path.join(output_dir, "model")
        pipeline.save_pipeline(model_dir)

    # Predict similarities
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
