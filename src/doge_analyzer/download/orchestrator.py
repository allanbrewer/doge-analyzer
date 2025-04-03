#!/usr/bin/env python3
import os
import logging
import argparse
import sys
from datetime import datetime
import json
import glob
import time

from doge_analyzer.download.download_awards import main as download_awards
from doge_analyzer.download.transform_data import process_zip_files

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define department mapping (API name to acronym)
DEPARTMENTS = {
    "Department of Agriculture": "USDA",
    "Department of Commerce": "DOC",
    "Department of Defense": "DOD",
    "Department of Education": "ED",
    "Department of Energy": "DOE",
    "Department of Health and Human Services": "HHS",
    "Department of Homeland Security": "DHS",
    "Department of Housing and Urban Development": "HUD",
    "Department of Justice": "DOJ",
    "Department of Labor": "DOL",
    "Department of State": "DOS",
    "Department of the Interior": "DOI",
    "Department of the Treasury": "TREAS",
    "Department of Transportation": "DOT",
    "Department of Veterans Affairs": "VA",
    "Environmental Protection Agency": "EPA",
    "National Aeronautics and Space Administration": "NASA",
    "Small Business Administration": "SBA",
    "Office of Personnel Management": "OPM",
    "General Services Administration": "GSA",
    "Social Security Administration": "SSA",
}

# Award types to process
AWARD_TYPES = ["contract", "grant"]


def process_department(
    dept_name,
    dept_acronym,
    award_types,
    start_date=None,
    end_date=None,
    skip_download=False,
    use_keywords=False,
):
    """Process all award types for a single department"""
    results = {}

    # Output directory logic removed, handled by transform_data

    for award_type in award_types:
        logger.info(f"Processing {dept_name} ({dept_acronym}) - {award_type}")

        zip_files = []

        # Step 1: Download contract data (if not skipped)
        if not skip_download:
            logger.info(f"Attempting download for {dept_name} - {award_type}...")
            # download_awards returns an exit code (0 for success)
            download_status = download_awards(
                department=dept_name,
                sub_award_type=award_type,
                start_date=start_date,
                end_date=end_date,
            )
            if download_status != 0:
                logger.warning(
                    f"Download step failed or produced no files for {dept_name} ({award_type}). Skipping transform."
                )
                continue
            # If download succeeded, transform_data will find the files in data/raw_data
            logger.info(f"Download step completed for {dept_name} - {award_type}.")
        else:
            # Find existing zip files for this department and award type
            # Format: department_of_X_awardtype_date_to_date.zip
            dept_name_lower = dept_name.lower().replace(" ", "_")
            zip_pattern = os.path.join(
                "data/raw_data", f"{dept_name_lower}_{award_type}_*.zip"
            )
            zip_files = glob.glob(zip_pattern)

            if not zip_files:
                logger.warning(
                    f"No existing zip files found for {dept_name} ({award_type})"
                )
                continue

            logger.info(
                f"Found {len(zip_files)} existing zip files for {dept_name} ({award_type})"
            )

        # Step 2: Transform and filter data (transform_data finds zips in data/raw_data)
        logger.info(f"Attempting transform for {dept_name} - {award_type}...")
        # transform_data now handles its own temp dirs and outputs to data/unlabeled/
        # It returns the path to the final file on success, or None on failure
        # Call process_zip_files directly
        # It needs the list of zip files for the specific dept/type
        dept_name_lower = dept_name.lower().replace(" ", "_")
        zip_pattern = os.path.join(
            "data/raw_data", f"{dept_name_lower}_{award_type}_*.zip"
        )
        current_zip_files = glob.glob(zip_pattern)

        if not current_zip_files:
            logger.warning(
                f"No zip files found matching {zip_pattern} for transform step."
            )
            final_file_path = None
        else:
            # Define the temporary processing directory base path
            temp_processing_base = os.path.join("data", "temp_processed_data")
            temp_dept_dir = os.path.join(temp_processing_base, dept_acronym)

            final_file_path = process_zip_files(
                zip_files=current_zip_files,  # Pass the found zip files
                dept_name=dept_name,
                dept_acronym=dept_acronym,
                sub_award_type=award_type,
                output_dir=temp_dept_dir,
                use_keywords=use_keywords,
            )

        # Check if transform_data returned a valid path AND the file actually exists
        if final_file_path and os.path.exists(final_file_path):
            logger.info(
                f"Transform step successful for {dept_name} - {award_type}. Output: {final_file_path}"
            )
            results[award_type] = final_file_path  # Record the actual path
        else:
            # Log failure if transform_data returned None or the file doesn't exist
            logger.warning(
                f"Transform step failed or produced no output file for {dept_name} - {award_type}."
            )

        # Add a small delay between processing different award types
        time.sleep(5)

    return results


def process_all_existing_data(use_keywords=False):  # output_dir removed
    """Process all existing downloaded data in data/raw_data without re-downloading"""
    results = {}

    # Find all existing zip files
    zip_files = glob.glob(os.path.join("data", "raw_data", "*.zip"))

    if not zip_files:
        logger.warning("No existing zip files found in data/raw_data directory")
        return results

    logger.info(f"Found {len(zip_files)} existing zip files to process")

    # Extract unique department-award type combinations from filenames
    dept_award_combos = set()

    for zip_file in zip_files:
        filename = os.path.basename(zip_file)

        # Expected format: department_of_X_awardtype_date_to_date.zip
        parts = filename.split("_")

        if len(parts) >= 4:
            # Extract department name and award type
            dept_parts = []
            award_type = None

            for i, part in enumerate(parts):
                if part in AWARD_TYPES:
                    award_type = part
                    dept_parts = parts[:i]
                    break

            if not award_type or not dept_parts:
                logger.warning(
                    f"Could not parse department and award type from filename: {filename}"
                )
                continue

            # Reconstruct department name
            dept_name = " ".join(dept_parts).replace("department of ", "Department of ")

            # Find the department acronym
            dept_acronym = None
            for name, acronym in DEPARTMENTS.items():
                if name.lower() == dept_name.lower():
                    dept_acronym = acronym
                    dept_name = name  # Use the correct casing from the dictionary
                    break

            if not dept_acronym:
                logger.warning(f"Unknown department in filename: {dept_name}")
                continue

            if award_type in AWARD_TYPES:
                dept_award_combos.add((dept_name, dept_acronym, award_type))

    logger.info(
        f"Found {len(dept_award_combos)} unique department-award type combinations to process"
    )

    # Process each unique combination
    for dept_name, dept_acronym, award_type in dept_award_combos:
        logger.info(
            f"Processing existing data for {dept_name} ({dept_acronym}) - {award_type}"
        )

        # Output directory logic removed

        # Transform and filter data (transform_data finds zips in data/raw_data)
        # It returns the path to the final file on success, or None on failure
        # Call process_zip_files directly
        # It needs the list of zip files for the specific dept/type
        dept_name_lower = dept_name.lower().replace(" ", "_")
        zip_pattern = os.path.join(
            "data/raw_data", f"{dept_name_lower}_{award_type}_*.zip"
        )
        current_zip_files = glob.glob(zip_pattern)

        if not current_zip_files:
            logger.warning(
                f"No zip files found matching {zip_pattern} for transform step."
            )
            final_file_path = None
        else:
            # Define the temporary processing directory base path
            temp_processing_base = os.path.join("data", "temp_processed_data")
            temp_dept_dir = os.path.join(temp_processing_base, dept_acronym)

            final_file_path = process_zip_files(
                zip_files=current_zip_files,  # Pass the found zip files
                dept_name=dept_name,
                dept_acronym=dept_acronym,
                sub_award_type=award_type,
                output_dir=temp_dept_dir,  # Pass the temp dir for intermediate files
                use_keywords=use_keywords,
            )

        # Check if transform_data returned a valid path AND the file actually exists
        if final_file_path and os.path.exists(final_file_path):
            logger.info(
                f"Existing data transform successful for {dept_name} - {award_type}. Output: {final_file_path}"
            )
            if dept_acronym not in results:
                results[dept_acronym] = {}
            results[dept_acronym][
                award_type
            ] = final_file_path  # Record the actual path
        else:
            # Log failure if transform_data returned None or the file doesn't exist
            logger.warning(
                f"Existing data transform failed or produced no output file for {dept_name} - {award_type}."
            )

        # Add a small delay between processing
        time.sleep(1)

    return results


def main(
    departments=None,
    award_types=None,
    start_date=None,
    end_date=None,
    skip_download=False,
    process_existing=False,
    use_keywords=False,
):
    """
    Main orchestration function for the DOGE Analyzer data download service

    Args:
        departments: List of departments to process
        award_types: List of award types to process
        start_date: Start date for contracts
        end_date: End date for downloads
        skip_download: Skip downloading and use existing files in data/raw_data/
        process_existing: Process all existing downloaded data in data/raw_data/
        use_keywords: Use keywords to filter data (default: False)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Output directory creation removed

    # Process all existing data if requested
    if process_existing:
        logger.info(
            "Processing all existing downloaded data found in data/raw_data/..."
        )
        results = process_all_existing_data(use_keywords)
    else:
        # Use all departments if none specified
        if not departments:
            departments = list(DEPARTMENTS.keys())

        # Use all award types if none specified
        if not award_types:
            award_types = AWARD_TYPES

        # Create a results dictionary
        results = {}

        # Process each department
        for dept_name in departments:
            if dept_name not in DEPARTMENTS:
                logger.warning(f"Unknown department: {dept_name}")
                continue

            dept_acronym = DEPARTMENTS[dept_name]
            logger.info(f"Processing department: {dept_name} ({dept_acronym})")

            dept_results = process_department(
                dept_name,
                dept_acronym,
                award_types,
                start_date,
                end_date,
                skip_download,
                use_keywords,
            )

            if dept_results:
                results[dept_acronym] = dept_results

    # Save results summary in data/logs/
    summary_dir = os.path.join("data", "logs")
    os.makedirs(summary_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(
        summary_dir, f"download_processing_summary_{timestamp}.json"
    )

    try:
        # Filter results to only include entries where a file path was successfully recorded
        # Ensure nested dictionaries are handled correctly
        valid_results = {}
        for dept, types in results.items():
            if isinstance(types, dict):
                valid_types = {
                    k: v for k, v in types.items() if v and isinstance(v, str)
                }  # Check for non-empty string paths
                if valid_types:
                    valid_results[dept] = valid_types

        with open(summary_file, "w") as f:
            json.dump(valid_results, f, indent=2)
        logger.info(f"Processing complete! Summary saved to {summary_file}")
    except Exception as e:
        logger.error(f"Failed to save summary file {summary_file}: {e}")

    # Determine overall success based on whether any valid results were recorded
    success = bool(valid_results)  # Simpler check if the filtered dict is non-empty
    return (
        0 if success else 1
    )  # Return 0 if at least one file was successfully processed


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Orchestrate the DOGE Analyzer data download and processing service"
    )
    parser.add_argument(
        "--departments",
        nargs="+",
        help="List of departments to process (default: all departments)",
    )
    parser.add_argument(
        "--award-types",
        nargs="+",
        choices=AWARD_TYPES,
        help="Award types to process (default: all types)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date in YYYY-MM-DD format (default: 2024-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading and use existing files (must specify departments and award types)",
    )
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="Process all existing downloaded data without re-downloading (ignores departments and award types arguments)",
    )
    parser.add_argument(
        "--use-keywords",
        action="store_true",
        help="Use keywords to filter data (default: False)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.skip_download and args.process_existing:
        parser.error("Cannot use both --skip-download and --process-existing together")

    # Call main function and exit with its return code
    sys.exit(
        main(
            args.departments,
            args.award_types,
            args.start_date,
            args.end_date,
            args.skip_download,
            args.process_existing,
            args.use_keywords,
        )
    )
