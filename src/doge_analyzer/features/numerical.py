"""
Numerical feature processing module.
This module handles processing numerical features from contract data.
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


def extract_numerical_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract numerical features from a DataFrame.

    Args:
        df: DataFrame containing contract data

    Returns:
        Array of numerical features
    """
    # Define numerical columns to use
    numerical_columns = ["normalized_value", "value_per_word"]

    # Filter to only include columns that exist in the DataFrame
    available_columns = [col for col in numerical_columns if col in df.columns]

    if not available_columns:
        logger.warning("No numerical columns found in DataFrame")
        return np.zeros((len(df), 0))

    # Extract numerical features
    numerical_features = df[available_columns].fillna(0).values

    logger.info(f"Extracted numerical features with shape: {numerical_features.shape}")

    return numerical_features


def scale_numerical_features(
    train_features: np.ndarray, test_features: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """
    Scale numerical features using StandardScaler.

    Args:
        train_features: Training features to fit the scaler
        test_features: Test features to transform (optional)

    Returns:
        Tuple of (scaled_train_features, scaled_test_features, scaler)
    """
    # Initialize scaler
    scaler = StandardScaler()

    # Fit and transform training features
    scaled_train = scaler.fit_transform(train_features)

    # Transform test features if provided
    scaled_test = None
    if test_features is not None:
        scaled_test = scaler.transform(test_features)

    logger.info(f"Scaled numerical features with shape: {scaled_train.shape}")

    return scaled_train, scaled_test, scaler


def encode_categorical_features(
    df: pd.DataFrame, categorical_columns: List[str] = ["agency", "vendor"]
) -> np.ndarray:
    """
    Encode categorical features using one-hot encoding.

    Args:
        df: DataFrame containing contract data
        categorical_columns: List of categorical columns to encode

    Returns:
        Array of encoded categorical features
    """
    # Filter to only include columns that exist in the DataFrame
    available_columns = [col for col in categorical_columns if col in df.columns]

    if not available_columns:
        logger.warning("No categorical columns found in DataFrame")
        return np.zeros((len(df), 0))

    # Initialize encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # Extract and encode categorical features
    categorical_data = df[available_columns].fillna("Unknown").values
    encoded_features = encoder.fit_transform(categorical_data)

    logger.info(f"Encoded categorical features with shape: {encoded_features.shape}")

    return encoded_features


def combine_features(
    text_features: np.ndarray,
    numerical_features: np.ndarray,
    categorical_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine text, numerical, and categorical features.

    Args:
        text_features: BERT text features
        numerical_features: Numerical features
        categorical_features: Categorical features (optional)

    Returns:
        Combined feature array
    """
    # List to store features to combine
    features_to_combine = [text_features, numerical_features]

    # Add categorical features if provided
    if categorical_features is not None:
        features_to_combine.append(categorical_features)

    # Combine features horizontally
    combined_features = np.hstack(features_to_combine)

    logger.info(f"Combined features with shape: {combined_features.shape}")

    return combined_features
