"""
Feature fusion module.
This module handles combining text features with numerical and categorical features.
"""

import os
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
from typing import Dict, List, Optional, Tuple, Union
from joblib import dump, load

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureFusion:
    """
    Feature fusion class for combining different types of features.
    """

    def __init__(self):
        """
        Initialize the feature fusion class.
        """
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.numerical_columns = ["normalized_value"]
        self.categorical_columns = ["agency"]
        self.fitted = False
        self.top_n_agencies = 25
        self.top_agencies = []

    def extract_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract numerical features from a DataFrame.

        Args:
            df: DataFrame containing contract data

        Returns:
            Array of numerical features
        """
        # Filter to only include columns that exist in the DataFrame
        available_columns = [col for col in self.numerical_columns if col in df.columns]

        if not available_columns:
            logger.warning("No numerical columns found in DataFrame")
            return np.zeros((len(df), 0))

        # Extract numerical features
        numerical_features = df[available_columns].fillna(0).values

        logger.info(
            f"Extracted numerical features with shape: {numerical_features.shape}"
        )

        return numerical_features

    def extract_categorical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract categorical features from a DataFrame.

        Args:
            df: DataFrame containing contract data

        Returns:
            Array of categorical features
        """
        # Filter to only include columns that exist in the DataFrame
        available_columns = [
            col for col in self.categorical_columns if col in df.columns
        ]

        if not available_columns:
            logger.warning("No categorical columns found in DataFrame")
            return np.zeros((len(df), 0))

        # Extract categorical features
        categorical_data = df[available_columns].fillna("Unknown").values

        logger.info(
            f"Extracted categorical features with shape: {categorical_data.shape}"
        )

        return categorical_data

    def fit(self, df: pd.DataFrame) -> None:
        # Get top N agencies
        agency_counts = df["agency"].value_counts()
        self.top_agencies = list(agency_counts.index[: self.top_n_agencies])

        # Replace rare agencies with "Unknown"
        df["agency"] = df["agency"].apply(
            lambda x: x if x in self.top_agencies else "Unknown"
        )

        # Extract numerical features
        numerical_features = self.extract_numerical_features(df)

        # Fit numerical scaler
        if numerical_features.shape[1] > 0:
            self.numerical_scaler.fit(numerical_features)
            logger.info("Fitted numerical scaler")

        # Extract categorical features
        categorical_data = self.extract_categorical_features(df)

        # Fit categorical encoder
        if categorical_data.shape[1] > 0:
            self.categorical_encoder.fit(categorical_data)
            logger.info("Fitted categorical encoder")

        self.fitted = True

    def transform(self, df: pd.DataFrame, text_features: np.ndarray) -> np.ndarray:
        """
        Transform data using the fitted feature fusion model.

        Args:
            df: DataFrame containing data to transform
            text_features: BERT text features

        Returns:
            Combined feature array
        """
        if not self.fitted:
            logger.warning("Feature fusion model not fitted. Call fit() first.")
            return text_features

        # Extract and scale numerical features
        numerical_features = self.extract_numerical_features(df)
        if numerical_features.shape[1] > 0:
            numerical_features = self.numerical_scaler.transform(numerical_features)

        # Replace rare agencies with "Unknown"
        df["agency"] = df["agency"].apply(
            lambda x: x if x in self.top_agencies else "Unknown"
        )

        # Extract and encode categorical features
        categorical_features = self.extract_categorical_features(df)
        categorical_features_encoded = None
        if categorical_features.shape[1] > 0:
            categorical_features_encoded = self.categorical_encoder.transform(
                categorical_features
            )

        features_to_combine = []
        if text_features.size > 0:  # Only include text features if they exist
            features_to_combine.append(text_features)
        if numerical_features.shape[1] > 0:
            features_to_combine.append(numerical_features)
        if categorical_features_encoded is not None:
            features_to_combine.append(categorical_features_encoded)

        # Combine features horizontally
        if features_to_combine:
            combined_features = np.hstack(features_to_combine)
        else:
            combined_features = np.array([])

        logger.info(f"Combined features with shape: {combined_features.shape}")

        return combined_features

    def fit_transform(self, df: pd.DataFrame, text_features: np.ndarray) -> np.ndarray:
        """
        Fit the feature fusion model and transform the data.

        Args:
            df: DataFrame containing data
            text_features: BERT text features

        Returns:
            Combined feature array
        """
        self.fit(df)
        return self.transform(df, text_features)

    def save_model(self, output_dir: str) -> None:
        """
        Save the feature fusion model components to files.

        Args:
            output_dir: Directory to save the pipeline components
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save numerical scaler
        numerical_path = os.path.join(output_dir, "numerical_scaler.joblib")
        dump(self.numerical_scaler, numerical_path)

        # Save categorical encoder
        categorical_path = os.path.join(output_dir, "categorical_encoder.joblib")
        dump(self.categorical_encoder, categorical_path)

        # Save top agencies
        top_agencies_path = os.path.join(output_dir, "top_agencies.joblib")
        dump(self.top_agencies, top_agencies_path)

        logger.info(f"Feature fusion model saved to {output_dir}")

    @classmethod
    def load_model(cls, model_dir: str) -> "FeatureFusion":
        """
        Load a feature fusion model from files.

        Args:
            model_dir: Directory containing the saved pipeline components

        Returns:
            Loaded FeatureFusion instance
        """
        # Initialize feature fusion
        feature_fusion = cls()

        # Load numerical scaler
        numerical_path = os.path.join(model_dir, "numerical_scaler.joblib")
        feature_fusion.numerical_scaler = load(numerical_path)

        # Load categorical encoder
        categorical_path = os.path.join(model_dir, "categorical_encoder.joblib")
        feature_fusion.categorical_encoder = load(categorical_path)

        # Load top agencies
        top_agencies_path = os.path.join(model_dir, "top_agencies.joblib")
        feature_fusion.top_agencies = load(top_agencies_path)

        feature_fusion.fitted = True

        logger.info(f"Feature fusion model loaded from {model_dir}")
        return feature_fusion
