"""
Data preprocessing module for contract data.
This module handles cleaning, normalizing, and preparing data for feature extraction.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Tuple, Union

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text data.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and extra whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def normalize_value(value: Union[str, float, int]) -> float:
    """
    Normalize contract value to a float.

    Args:
        value: Contract value (could be string with $ or numeric)

    Returns:
        Normalized value as float
    """
    if pd.isna(value):
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Remove $ and commas
        value = value.replace("$", "").replace(",", "")

        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            return 0.0

    return 0.0


def preprocess_labeled_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess labeled data (canceled contracts).

    Args:
        df: DataFrame containing labeled data

    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Clean description text
    if "description" in processed_df.columns:
        processed_df["clean_description"] = processed_df["description"].apply(
            clean_text
        )

    # Normalize contract value
    if "value" in processed_df.columns:
        processed_df["normalized_value"] = processed_df["value"].apply(normalize_value)

    # Savings processing removed as requested

    # Fill missing values
    processed_df = processed_df.fillna(
        {
            "agency": "Unknown",
            "vendor": "Unknown",
            "clean_description": "",
            "normalized_value": 0.0,
            # "normalized_savings": 0.0, # Removed
        }
    )

    # Add derived features
    if (
        "normalized_value" in processed_df.columns
        and "clean_description" in processed_df.columns
    ):
        # Value per word (to identify contracts with high value but little description)
        processed_df["value_per_word"] = processed_df.apply(
            lambda row: row["normalized_value"]
            / max(len(row["clean_description"].split()), 1),
            axis=1,
        )

    # Add is_canceled column (True for all labeled data)
    processed_df["is_canceled"] = True

    logger.info(f"Preprocessed {len(processed_df)} labeled contracts")

    return processed_df


from typing import Dict, List, Optional, Set, Tuple, Union


def preprocess_unlabeled_data(
    df: pd.DataFrame, labeled_columns: List[str], labeled_piids: Set[str]
) -> pd.DataFrame:
    """
    Preprocess unlabeled data to match the format of labeled data,
    filtering out any records present in the labeled set.

    Args:
        df: DataFrame containing unlabeled data
        labeled_columns: List of column names from the labeled data
        labeled_piids: A set of PIIDs from the labeled data to exclude

    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Column mapping/renaming removed - this is now handled upstream by transform_data.py
    # The input DataFrame 'df' is expected to already have the standard column names:
    # 'piid', 'description', 'value', 'agency', 'vendor', etc.

    # Clean description text
    if "description" in processed_df.columns:
        processed_df["clean_description"] = processed_df["description"].apply(
            clean_text
        )

    # Normalize contract value
    if "value" in processed_df.columns:
        processed_df["normalized_value"] = processed_df["value"].apply(normalize_value)

    # Fill missing values
    processed_df = processed_df.fillna(
        {
            "agency": "Unknown",
            "vendor": "Unknown",
            "clean_description": "",
            "normalized_value": 0.0,
        }
    )

    # Add derived features
    if (
        "normalized_value" in processed_df.columns
        and "clean_description" in processed_df.columns
    ):
        # Value per word (to identify contracts with high value but little description)
        processed_df["value_per_word"] = processed_df.apply(
            lambda row: row["normalized_value"]
            / max(len(row["clean_description"].split()), 1),
            axis=1,
        )

    # Placeholder for savings removed as requested
    # if "savings" not in processed_df.columns:
    #     processed_df["savings"] = np.nan
    #     processed_df["normalized_savings"] = 0.0

    # Ensure all required columns exist
    for col in labeled_columns:
        if col not in processed_df.columns:
            processed_df[col] = np.nan

    # Filter out records that are present in the labeled data
    initial_count = len(processed_df)
    if "piid" in processed_df.columns and labeled_piids:
        processed_df = processed_df[~processed_df["piid"].isin(labeled_piids)]
        removed_count = initial_count - len(processed_df)
        if removed_count > 0:
            logger.info(
                f"Removed {removed_count} unlabeled records found in the labeled dataset."
            )

    logger.info(
        f"Preprocessed {len(processed_df)} unlabeled contracts (after filtering)"
    )

    return processed_df


def align_dataframes(
    labeled_df: pd.DataFrame, unlabeled_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure labeled and unlabeled DataFrames have the same columns.

    Args:
        labeled_df: DataFrame containing labeled data
        unlabeled_df: DataFrame containing unlabeled data

    Returns:
        Tuple of (aligned_labeled_df, aligned_unlabeled_df)
    """
    # Get common columns
    common_columns = list(set(labeled_df.columns) & set(unlabeled_df.columns))

    # Ensure essential columns are included
    essential_columns = [
        "piid",
        "agency",
        "vendor",
        "value",
        "description",
        "clean_description",
        "normalized_value",
    ]

    for col in essential_columns:
        if col not in common_columns and col in labeled_df.columns:
            common_columns.append(col)

    # Select common columns
    aligned_labeled_df = labeled_df[common_columns].copy()
    aligned_unlabeled_df = unlabeled_df[common_columns].copy()

    logger.info(f"Aligned DataFrames with {len(common_columns)} common columns")

    return aligned_labeled_df, aligned_unlabeled_df
