#!/usr/bin/env python3
import os
import logging
import argparse
import sys
from datetime import datetime
import json
import glob
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import modules from different possible paths
# Use relative imports for modules within the same package
try:
    from .download_awards import main as download_awards
    from .transform_data import main as transform_data

    logger.info("Successfully imported download_awards and transform_data modules.")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    # Depending on severity, you might want to exit or raise the error
    raise ImportError(
        "Could not import required download modules (download_awards, transform_data). Check file structure."
    ) from e

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
    # output_base_dir removed, transform_data handles final output location
    skip_download=False,
):
    """Process all award types for a single department"""
    results = {}

    # Output directory logic removed, handled by transform_data

    for award_type in award_types:
        logger.info(f"Processing {dept_name} ({dept_acronym}) - {award_type}")

        zip_files = []

        # Step 1: Download contract data (if not skipped)
        # Step 1: Download data (if not skipped)
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
            # If download succeeded, transform_data will find the files in raw_data
            logger.info(f"Download step completed for {dept_name} - {award_type}.")
        else:
            # Find existing zip files for this department and award type
            # Format: department_of_X_awardtype_date_to_date.zip
            dept_name_lower = dept_name.lower().replace(" ", "_")
            zip_pattern = os.path.join(
                "raw_data", f"{dept_name_lower}_{award_type}_*.zip"
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

        # Step 2: Transform and filter data (transform_data finds zips in raw_data)
        logger.info(f"Attempting transform for {dept_name} - {award_type}...")
        # transform_data now handles its own temp dirs and outputs to data/unlabeled/
        # It returns the path to the final file or None/1 on failure
        transform_status = transform_data(
            zip_dir="raw_data",  # Still needed to know where to look for zips
            # output_dir is no longer needed here, transform_data manages temp/final paths
            dept_name=dept_name,
            dept_acronym=dept_acronym,
            sub_award_type=award_type,
        )

        # transform_data returns 0 on success, 1 on failure/no data
        if transform_status == 0:
            # Construct the expected final output path to record it
            final_output_dir = os.path.join("data", "unlabeled", award_type)
            final_file_path = os.path.join(
                final_output_dir, f"{dept_acronym}_{award_type}_processed.csv"
            )
            # Check if the file actually exists after transform_data reported success
            if os.path.exists(final_file_path):
                logger.info(
                    f"Transform step successful for {dept_name} - {award_type}. Output: {final_file_path}"
                )
                results[award_type] = final_file_path
            else:
                logger.warning(
                    f"Transform step reported success for {dept_name} - {award_type}, but output file {final_file_path} not found."
                )
        else:
            logger.warning(
                f"Transform step failed or produced no output for {dept_name} - {award_type}."
            )

        # Add a small delay between processing different award types
        time.sleep(5)

    return results


def process_all_existing_data():  # output_dir removed
    """Process all existing downloaded data in raw_data without re-downloading"""
    results = {}

    # Find all existing zip files
    zip_files = glob.glob(os.path.join("raw_data", "*.zip"))

    if not zip_files:
        logger.warning("No existing zip files found in raw_data directory")
        return results

    logger.info(f"Found {len(zip_files)} existing zip files to process")

    # Extract unique department-award type combinations from filenames
    dept_award_combos = set()

    for zip_file in zip_files:
        filename = os.path.basename(zip_file)

        # Expected format: department_of_X_awardtype_date_to_date.zip
        # Example: department_of_agriculture_grant_2024-01-01_to_2024-03-31.zip
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

        # Transform and filter data
        # Transform and filter data (transform_data finds zips in raw_data)
        transform_status = transform_data(
            zip_dir="raw_data",
            # output_dir removed
            dept_name=dept_name,
            dept_acronym=dept_acronym,
            sub_award_type=award_type,
        )

        if transform_status == 0:
            # Construct the expected final output path
            final_output_dir = os.path.join("data", "unlabeled", award_type)
            final_file_path = os.path.join(
                final_output_dir, f"{dept_acronym}_{award_type}_processed.csv"
            )
            if os.path.exists(final_file_path):
                logger.info(
                    f"Existing data transform successful for {dept_name} - {award_type}. Output: {final_file_path}"
                )
                if dept_acronym not in results:
                    results[dept_acronym] = {}
                results[dept_acronym][award_type] = final_file_path
            else:
                logger.warning(
                    f"Existing data transform reported success for {dept_name} - {award_type}, but output file {final_file_path} not found."
                )
        else:
            logger.warning(
                f"Existing data transform failed or produced no output for {dept_name} - {award_type}."
            )

        # Add a small delay between processing
        time.sleep(1)

    return results


def main(
    departments=None,
    award_types=None,
    start_date=None,
    end_date=None,
    # output_dir removed
    skip_download=False,
    process_existing=False,
):
    """
    Main orchestration function for the DOGE Analyzer data download service

    Args:
        departments: List of departments to process
        award_types: List of award types to process
        start_date: Start date for contracts
        end_date: End date for downloads
        # output_dir removed
        skip_download: Skip downloading and use existing files in raw_data/
        process_existing: Process all existing downloaded data in raw_data/

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Output directory creation removed

    # Process all existing data if requested
    if process_existing:
        logger.info("Processing all existing downloaded data found in raw_data/...")
        results = process_all_existing_data()
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
                # output_dir removed
                skip_download,
            )

            if dept_results:
                results[dept_acronym] = dept_results

    # Save results summary
    # Save results summary in the current directory or a dedicated logs/ dir
    summary_dir = "logs"
    os.makedirs(summary_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(
        summary_dir, f"download_processing_summary_{timestamp}.json"
    )

    try:
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Processing complete! Summary saved to {summary_file}")
    except Exception as e:
        logger.error(f"Failed to save summary file {summary_file}: {e}")

    # Determine overall success (e.g., if at least one file was processed)
    success = any(results.values())
    return 0 if success else 1  # Return 0 if something was processed, 1 otherwise


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
    # Output directory argument removed
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
            # args.output_dir removed
            args.skip_download,
            args.process_existing,
        )
    )
