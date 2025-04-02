"""
Data loading module for contract data.
This module handles loading both labeled data (canceled contracts) and unlabeled data.
"""

import json
import os
import zipfile
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import glob

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_labeled_data(file_path: str) -> pd.DataFrame:
    """
    Load labeled data (canceled contracts) from a JSON file.
    Args:
        file_path: Path to the JSON file containing labeled data
    Returns:
        DataFrame containing the labeled data
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract the data from the JSON structure
        if "data" in data:
            contracts = data["data"]
        else:
            contracts = data

        # Convert to DataFrame
        df = pd.DataFrame(contracts)

        initial_count = len(df)
        logger.info(
            f"Loaded {initial_count} labeled contracts initially from {file_path}"
        )

        # Filter out contracts with specific piid values
        if "piid" in df.columns:
            filter_values = ["Unavailable", "Charge Card Purchase"]
            original_count = len(df)
            df = df[~df["piid"].isin(filter_values)]
            removed_count = original_count - len(df)
            if removed_count > 0:
                logger.info(
                    f"Removed {removed_count} contracts with piid in {filter_values}."
                )
        else:
            logger.warning("Column 'piid' not found, skipping filtering.")

        logger.info(f"Returning {len(df)} labeled contracts after filtering.")
        return df
    except Exception as e:
        logger.error(f"Error loading labeled data from {file_path}: {e}")
        raise


# Removed extract_csv_from_zip function as extraction is handled upstream by the download orchestrator.
def load_unlabeled_data_from_file(
    file_path: str,
    # extract_dir parameter removed
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load unlabeled data from a single file (CSV or JSON).
    Args:
        file_path: Path to the file (CSV or JSON)
        sample_size: Number of contracts to sample (if None, load all)
    Returns:
        DataFrame containing the unlabeled data
    """
    # Handle CSV files
    if file_path.lower().endswith(".csv"):
        csv_files = [file_path]
    # Handle JSON files (assuming same structure as labeled data for simplicity)
    elif file_path.lower().endswith(".json"):
        try:
            df = load_json_data(file_path)
            logger.info(f"Loaded unlabeled data from JSON file: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return pd.DataFrame()
    else:
        logger.warning(f"Unsupported file type: {file_path}. Skipping.")
        return pd.DataFrame()

    # Load and combine CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            # Try reading with different encodings if default fails
            try:
                df = pd.read_csv(csv_file)
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for {csv_file}, trying latin1.")
                df = pd.read_csv(csv_file, encoding="latin1")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file}: {e}")

    if not dfs:
        logger.error(f"No valid data loaded from {file_path}")
        return pd.DataFrame()

    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sample if requested
    if sample_size is not None and sample_size < len(combined_df):
        combined_df = combined_df.sample(sample_size, random_state=42)

    logger.info(f"Loaded {len(combined_df)} unlabeled contracts from {file_path}")

    return combined_df


def load_multiple_unlabeled_files(
    input_paths: Union[str, List[str]],
    # extract_dir parameter removed
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load unlabeled data from multiple CSV/JSON files or a directory containing them.
    Args:
        input_paths: Path to a directory, a single file (CSV/JSON), or a list of file paths
        sample_size: Total number of contracts to sample across all files
    Returns:
        DataFrame containing the combined unlabeled data
    """
    file_paths_to_process = []

    # Handle single path (file or directory)
    if isinstance(input_paths, str):
        if os.path.isdir(input_paths):
            logger.info(f"Loading data from directory: {input_paths}")
            # Find all CSV and JSON files in the directory
            csv_files = glob.glob(os.path.join(input_paths, "*.csv"))
            json_files = glob.glob(
                os.path.join(input_paths, "*.json")
            )  # Also look for JSON
            file_paths_to_process.extend(csv_files)
            file_paths_to_process.extend(json_files)
            logger.info(
                f"Found {len(csv_files)} CSV files and {len(json_files)} JSON files."
            )
        elif os.path.isfile(input_paths):
            logger.info(f"Loading data from single file: {input_paths}")
            file_paths_to_process = [input_paths]
        else:
            logger.error(f"Input path not found or invalid: {input_paths}")
            return pd.DataFrame()
    # Handle list of paths
    elif isinstance(input_paths, list):
        file_paths_to_process = input_paths
    else:
        logger.error(f"Invalid input type for input_paths: {type(input_paths)}")
        return pd.DataFrame()

    # Load data from each file
    dfs = []
    for file_path in file_paths_to_process:
        try:
            # Pass sample_size=None here, sampling will happen after combining
            df = load_unlabeled_data_from_file(
                file_path, sample_size=None
            )  # extract_dir removed
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")

    if not dfs:
        logger.error("No valid data loaded from any file")
        return pd.DataFrame()

    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sample if requested (after combining all data)
    if sample_size is not None and sample_size < len(combined_df):
        combined_df = combined_df.sample(sample_size, random_state=42)
        logger.info(f"Sampled {len(combined_df)} contracts.")

    logger.info(
        f"Loaded a total of {len(combined_df)} unlabeled contracts from {len(file_paths_to_process)} files/sources"
    )

    return combined_df


def load_json_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a JSON file.
    Args:
        file_path: Path to the JSON file
    Returns:
        DataFrame containing the data
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract the data from the JSON structure
        if "data" in data:
            contracts = data["data"]
        else:
            contracts = data

        # Convert to DataFrame
        df = pd.DataFrame(contracts)

        logger.info(f"Loaded {len(df)} contracts from {file_path}")

        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise
