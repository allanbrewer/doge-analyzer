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

        logger.info(f"Loaded {len(df)} labeled contracts from {file_path}")

        return df
    except Exception as e:
        logger.error(f"Error loading labeled data from {file_path}: {e}")
        raise


def extract_csv_from_zip(
    zip_file_path: str, output_dir: Optional[str] = None
) -> List[str]:
    """
    Extract CSV files from a zip file.

    Args:
        zip_file_path: Path to the zip file
        output_dir: Directory to extract files to (defaults to same directory as zip)

    Returns:
        List of paths to extracted CSV files
    """
    if output_dir is None:
        output_dir = os.path.dirname(zip_file_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    extracted_files = []

    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            # Get list of CSV files in the zip
            csv_files = [f for f in zip_ref.namelist() if f.lower().endswith(".csv")]

            # Extract CSV files
            for csv_file in csv_files:
                zip_ref.extract(csv_file, output_dir)
                extracted_path = os.path.join(output_dir, csv_file)
                extracted_files.append(extracted_path)

        logger.info(f"Extracted {len(extracted_files)} CSV files from {zip_file_path}")
        return extracted_files

    except Exception as e:
        logger.error(f"Error extracting CSV files from {zip_file_path}: {e}")
        raise


def load_unlabeled_data_from_file(
    file_path: str,
    extract_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load unlabeled data from a single file (ZIP or CSV).

    Args:
        file_path: Path to the file (ZIP or CSV)
        extract_dir: Directory to extract files to (if ZIP)
        sample_size: Number of contracts to sample (if None, load all)

    Returns:
        DataFrame containing the unlabeled data
    """
    csv_files = []
    if file_path.lower().endswith(".zip"):
        # Extract CSV files from zip
        try:
            csv_files = extract_csv_from_zip(file_path, extract_dir)
        except Exception as e:
            logger.error(f"Error processing zip file {file_path}: {e}")
            return pd.DataFrame()
    elif file_path.lower().endswith(".csv"):
        csv_files = [file_path]
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
    extract_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
    department_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load unlabeled data from multiple files or a directory.

    Args:
        input_paths: Path to a directory, a single file, or a list of file paths
        extract_dir: Directory to extract zip files to
        sample_size: Total number of contracts to sample across all files
        department_filter: Filter to only include files from a specific department

    Returns:
        DataFrame containing the combined unlabeled data
    """
    file_paths_to_process = []

    # Handle single path (file or directory)
    if isinstance(input_paths, str):
        if os.path.isdir(input_paths):
            logger.info(f"Loading data from directory: {input_paths}")
            # Find all ZIP and CSV files in the directory
            zip_files = glob.glob(os.path.join(input_paths, "*.zip"))
            csv_files = glob.glob(os.path.join(input_paths, "*.csv"))
            file_paths_to_process.extend(zip_files)
            file_paths_to_process.extend(csv_files)
            logger.info(
                f"Found {len(zip_files)} ZIP files and {len(csv_files)} CSV files."
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

    # Filter files by department if requested
    if department_filter:
        original_count = len(file_paths_to_process)
        file_paths_to_process = [
            f
            for f in file_paths_to_process
            if department_filter.lower() in os.path.basename(f).lower()
        ]
        logger.info(
            f"Filtered files by department '{department_filter}'. Kept {len(file_paths_to_process)} out of {original_count}."
        )

    if not file_paths_to_process:
        logger.error("No files found to process after filtering.")
        return pd.DataFrame()

    # Load data from each file
    dfs = []
    for file_path in file_paths_to_process:
        try:
            # Pass sample_size=None here, sampling will happen after combining
            df = load_unlabeled_data_from_file(file_path, extract_dir, sample_size=None)
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
