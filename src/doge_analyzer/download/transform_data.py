import os
import pandas as pd
from datetime import datetime
import re
import logging
import argparse
import zipfile
import shutil
import sys

from doge_analyzer.download.keyword import keywords

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_keywords():
    """
    Define specific keywords for initial contract download (case-insensitive)
    Always use the main list which is defined as the sum of all lists.

    Returns:
        Compiled regex pattern for matching keywords
    """
    if not keywords:
        logger.error("Failed to import keywords from keywords.py")
        return None

    keywords_list = keywords["main"]
    logger.info(f"Using {len(keywords_list)} keywords from 'main' category")

    # Create a regex pattern to match whole words or phrases
    pattern = re.compile(
        r"\b" + "|".join([re.escape(kw) for kw in keywords_list]) + r"\b", re.IGNORECASE
    )
    return pattern


def extract_zip_file(zip_path, extract_dir):
    """Extract a zip file to the specified directory"""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        return True
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {str(e)}")
        return False


def find_all_csv_files(directory):
    """Recursively find all CSV files in a directory and its subdirectories"""
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def process_csv_file(
    csv_path, output_dir, pattern, today_date, sub_award_type, dept_acronym
):
    """Process a single CSV file and return path to flagged file"""
    csv_file = os.path.basename(csv_path)
    logger.info(f"Processing {csv_file}...")

    try:
        # Load and filter
        df = pd.read_csv(csv_path, low_memory=False)

        # Define columns to keep based on award type and map to target names
        # Target names align with src/doge_analyzer/data/process.py mapping
        if sub_award_type == "contract":
            column_map = {
                "award_id_piid": "piid",
                "prime_award_base_transaction_description": "description",
                "current_total_value_of_award": "value",
                "period_of_performance_current_end_date": "end_date",
                "recipient_name": "vendor",
                "awarding_agency_name": "agency",
            }
            id_column = "award_id_piid"  # Original ID column for processing
            desc_column = (
                "prime_award_base_transaction_description"  # Original desc column
            )
        else:  # grant
            column_map = {
                "award_id_fain": "piid",  # Use 'piid' as the standard ID column name
                "prime_award_base_transaction_description": "description",
                "total_obligated_amount": "value",  # Use 'value' as the standard value column name
                "period_of_performance_current_end_date": "end_date",
                "recipient_name": "vendor",
                "awarding_agency_name": "agency",
            }
            id_column = "award_id_fain"  # Original ID column for processing
            desc_column = (
                "prime_award_base_transaction_description"  # Original desc column
            )

        columns_to_keep_original = list(column_map.keys())

        # Check if date column exists, use alternative if needed
        date_column = "period_of_performance_current_end_date"
        if date_column not in df.columns:
            date_column = "period_of_performance_end_date"
            if date_column not in df.columns:
                logger.warning(f"No performance end date column found in {csv_file}")
                return None, None

        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        # Filter active contracts/grants (those that expire after today)
        active_df = df[df[date_column] > today_date].copy()

        if active_df.empty:
            logger.info("  No active rows found")
            return None, None

        # Filter to only include columns that exist in the dataframe
        existing_columns_original = [
            col for col in columns_to_keep_original if col in active_df.columns
        ]
        if len(existing_columns_original) < len(columns_to_keep_original):
            missing = set(columns_to_keep_original) - set(existing_columns_original)
            logger.warning(f"Missing original columns in CSV {csv_file}: {missing}")

        # Only keep the original columns we're interested in if they exist
        if existing_columns_original:
            active_df = active_df[existing_columns_original]
        else:
            logger.warning(f"No relevant columns found in {csv_file}")
            return None, None  # Skip if no relevant columns

        logger.info(f"  Total rows: {len(df)}, Active rows: {len(active_df)}")

        # Ensure description column exists
        if desc_column not in active_df.columns:
            # Try alternative column names
            alt_desc_columns = [
                "description",
                "award_description",
                "prime_award_project_description",
            ]
            for alt_col in alt_desc_columns:
                if alt_col in active_df.columns:
                    desc_column = alt_col
                    break
            else:
                logger.warning(f"No description column found in {csv_file}")
                return None, None

        if pattern:
            # Filter active contracts/grants with matching keywords in description
            flagged_df = active_df[
                active_df[desc_column].fillna("").str.contains(pattern, na=False)
            ]

            if flagged_df.empty:
                logger.info("  No flagged rows found")
                return None, None
        else:
            # If no keywords are used, just copy the active_df
            flagged_df = active_df.copy()

        # Save flagged file with department acronym and award type
        flagged_path = os.path.join(
            output_dir, f"{dept_acronym}_{sub_award_type}_{csv_file}_flagged.csv"
        )
        flagged_df.to_csv(flagged_path, index=False)
        logger.info(f"  Saved {len(flagged_df)} flagged rows to {flagged_path}")

        return flagged_path, column_map

    except Exception as e:
        logger.error(f"Error processing {csv_file}: {str(e)}")
        return None


def combine_csv_files(file_paths, output_file, sub_award_type, column_map):
    """Combine multiple CSV files into a single master file"""
    if not file_paths:
        logger.warning(f"No {sub_award_type} files to combine")
        return False

    valid_paths = [p for p in file_paths if p and os.path.exists(p)]
    if not valid_paths:
        logger.warning(f"No valid {sub_award_type} files found")
        return False

    logger.info(f"Joining {len(valid_paths)} {sub_award_type} files...")
    try:
        master_df = pd.concat(
            [pd.read_csv(f, low_memory=False) for f in valid_paths], ignore_index=True
        )

        # Define aggregation logic based on original column names
        if sub_award_type == "contract":
            logger.info("Aggregating combined contract files by award_id_piid...")
            agg_dict = {
                "current_total_value_of_award": "max",
                "prime_award_base_transaction_description": "first",
                "recipient_name": "first",
                "awarding_agency_name": "first",
                "period_of_performance_current_end_date": "max",
            }
            # Only include columns that actually exist in the dataframe
            agg_dict_filtered = {
                k: v for k, v in agg_dict.items() if k in master_df.columns
            }
            if "award_id_piid" in master_df.columns:
                master_df = (
                    master_df.groupby("award_id_piid")
                    .agg(agg_dict_filtered)
                    .reset_index()
                )
            else:
                logger.warning(
                    "Column 'award_id_piid' not found for grouping contracts."
                )

        elif sub_award_type == "grant":
            logger.info("Aggregating combined grant files by award_id_fain...")
            agg_dict = {
                "total_obligated_amount": "max",
                "prime_award_base_transaction_description": "first",
                "recipient_name": "first",
                "awarding_agency_name": "first",
                "period_of_performance_current_end_date": "max",
            }
            # Only include columns that actually exist in the dataframe
            agg_dict_filtered = {
                k: v for k, v in agg_dict.items() if k in master_df.columns
            }
            if "award_id_fain" in master_df.columns:
                master_df = (
                    master_df.groupby("award_id_fain")
                    .agg(agg_dict_filtered)
                    .reset_index()
                )
            else:
                logger.warning("Column 'award_id_fain' not found for grouping grants.")

        # Rename columns to standard names AFTER aggregation
        final_column_map = column_map  # Use the map defined earlier
        master_df = master_df.rename(columns=final_column_map)
        # Ensure all target columns exist, adding missing ones with NaN
        for target_col in final_column_map.values():
            if target_col not in master_df.columns:
                master_df[target_col] = pd.NA

        logger.info(f"Deduped rows: {len(master_df)} rows")

        master_df.to_csv(output_file, index=False)
        logger.info(f"{sub_award_type.capitalize()} dataset: saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error combining {sub_award_type} files: {str(e)}")
        return False


def process_zip_files(
    zip_files, dept_name, dept_acronym, sub_award_type, output_dir, use_keywords
):
    """Process all zip files for a department and award type"""
    # Create temporary directory for extraction
    temp_dir = os.path.join(output_dir, "temp_extract")
    os.makedirs(temp_dir, exist_ok=True)

    # Setup
    if use_keywords:
        pattern = setup_keywords()
    else:
        pattern = None
    today_date = datetime.now().strftime("%Y-%m-%d")

    # List to hold flagged file paths
    flagged_files = []

    try:
        # Process each zip file
        for zip_path in zip_files:
            if not os.path.exists(zip_path):
                logger.warning(f"Zip file not found: {zip_path}")
                continue

            # Extract zip file
            logger.info(f"Extracting {zip_path}...")
            if not extract_zip_file(zip_path, temp_dir):
                continue

            # Find all CSV files in the extraction directory (including subdirectories)
            csv_files = find_all_csv_files(temp_dir)

            if not csv_files:
                logger.warning(f"No CSV files found in {zip_path}")
                continue

            logger.info(f"Found {len(csv_files)} CSV files in {zip_path}")

            for csv_path in csv_files:
                flagged_path, column_map = process_csv_file(
                    csv_path,
                    output_dir,
                    pattern,
                    today_date,
                    sub_award_type,
                    dept_acronym,
                )

                if flagged_path:
                    flagged_files.append(flagged_path)

            # Clean up extracted files after processing each zip
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)

        # Combine all flagged files into a master file in the standard unlabeled data location
        if flagged_files:
            # Define standard output directory and filename
            standard_output_dir = os.path.join("data", "unlabeled", sub_award_type)
            os.makedirs(standard_output_dir, exist_ok=True)
            master_file_path = os.path.join(
                standard_output_dir, f"{dept_acronym}_{sub_award_type}_processed.csv"
            )

            success = combine_csv_files(
                flagged_files, master_file_path, sub_award_type, column_map
            )  # Pass sub_award_type

            # Delete individual flagged files after successful combination
            if success:
                logger.info(f"Successfully combined files into {master_file_path}")
                logger.info("Cleaning up temporary flagged files...")
                for temp_file in flagged_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except OSError as e:
                        logger.warning(
                            f"Could not remove temporary file {temp_file}: {e}"
                        )
                return master_file_path  # Return path on success
            else:
                # This block executes if combine_csv_files returned False
                logger.error(
                    f"combine_csv_files function failed for {master_file_path}"
                )
                return None  # Return None if combination failed
        else:
            logger.info(f"No flagged files found for {dept_name} ({sub_award_type})")
            return None  # Return None if no flagged files were generated

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main(
    zip_dir="data/raw_data",
    dept_name=None,
    dept_acronym=None,
    sub_award_type="contract",
    use_keywords=False,
):
    """Process zip files for a specific department and award type

    Args:
        zip_dir: Directory containing zip files
        dept_name: Department name as used in the API
        dept_acronym: Department acronym for file naming
        sub_award_type: Type of award to process
        use_keywords: Flag to use keywords for filtering

    Returns:
        Exit code (0 for success, 1 for error)
    """
    if not dept_name or not dept_acronym:
        logger.error("Department name and acronym must be provided")
        return 1

    # Define the temporary processing directory (within the script's execution context)
    # The final output goes to data/unlabeled/{sub_award_type}/
    # Use 'data' as the base for the temporary directory
    temp_processing_dir = os.path.join("data", "temp_processed_data", dept_acronym)

    # Create output directory
    os.makedirs(temp_processing_dir, exist_ok=True)

    # Find zip files for this department and award type
    dept_name_pattern = dept_name.replace(" ", "_").lower()

    zip_files = []
    for file in os.listdir(zip_dir):
        if file.startswith(f"{dept_name_pattern}_{sub_award_type}_") and file.endswith(
            ".zip"
        ):
            zip_files.append(os.path.join(zip_dir, file))

    if not zip_files:
        logger.warning(f"No zip files found for {dept_name} ({sub_award_type})")
        return 1

    logger.info(f"Found {len(zip_files)} zip files for {dept_name} ({sub_award_type})")

    # Process the zip files using the temporary directory for intermediate files
    master_file_path = process_zip_files(
        zip_files,
        dept_name,
        dept_acronym,
        sub_award_type,
        temp_processing_dir,
        use_keywords,
    )

    # Clean up the temporary processing directory if it exists
    if os.path.exists(temp_processing_dir):
        logger.info(
            f"Cleaning up temporary processing directory: {temp_processing_dir}"
        )
        shutil.rmtree(temp_processing_dir)

    if master_file_path:
        logger.info(f"Successfully created final processed file: {master_file_path}")
        return 0  # Success
    else:
        logger.warning(
            f"No final processed file was created for {dept_name} ({sub_award_type})"
        )
        return 1  # Indicate failure or no data processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform and filter contract data from USA Spending API"
    )
    parser.add_argument(
        "--zip-dir",
        default="data/raw_data",
        help="Directory containing zip files (default: raw_data)",
    )
    parser.add_argument(
        "--dept-name",
        required=True,
        help="Department name as used in the API",
    )
    parser.add_argument(
        "--dept-acronym",
        required=True,
        help="Department acronym for file naming",
    )
    parser.add_argument(
        "--sub-award-type",
        default="contract",
        choices=["contract", "grant"],
        help="Type of award to process (default: contract)",
    )
    parser.add_argument(
        "--use-keywords",
        action="store_true",
        help="Use keywords for filtering (default: False)",
    )

    args = parser.parse_args()

    sys.exit(
        main(
            args.zip_dir,
            args.dept_name,
            args.dept_acronym,
            args.sub_award_type,
            args.use_keywords,
        )
    )
