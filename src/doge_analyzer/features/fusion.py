"""
Feature fusion module.
This module handles combining text features with numerical and categorical features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
from typing import Dict, List, Optional, Tuple, Union

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
        self.numerical_columns = ["normalized_value", "value_per_word"]
        self.categorical_columns = ["agency", "vendor"]
        self.fitted = False

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
        """
        Fit the feature fusion model on training data.

        Args:
            df: DataFrame containing training data
        """
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

        # Extract and encode categorical features
        categorical_features = self.extract_categorical_features(df)
        categorical_features_encoded = None
        if categorical_features.shape[1] > 0:
            categorical_features_encoded = self.categorical_encoder.transform(
                categorical_features
            )

        features_to_combine = [text_features]  # Include text features
        if numerical_features.shape[1] > 0:
            features_to_combine.append(numerical_features)
        if categorical_features_encoded is not None:
            features_to_combine.append(categorical_features_encoded)

        # Combine features horizontally
        combined_features = np.hstack(features_to_combine)

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
