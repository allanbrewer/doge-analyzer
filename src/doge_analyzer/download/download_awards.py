import requests
from datetime import datetime, UTC, timedelta
from dateutil.relativedelta import relativedelta
import os
import json
import time
import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def request_download(
    start_date: str, end_date: str, department: str, sub_award_type: str = "contract"
):
    """Request a bulk download of contract data from USAspending API"""
    url = "https://api.usaspending.gov/api/v2/bulk_download/awards/"

    # Define award types based on the sub_award_type parameter
    award_types = []
    if sub_award_type == "contract":
        award_types = [
            "A",
            "B",
            "C",
            "D",
            "IDV_A",
            "IDV_B",
            "IDV_B_A",
            "IDV_B_B",
            "IDV_B_C",
            "IDV_C",
            "IDV_D",
            "IDV_E",
        ]
    elif sub_award_type == "grant":
        award_types = [
            "02",
            "03",
            "04",
            "05",
            "06",
        ]  # Grant types

    payload = {
        "filters": {
            "prime_award_types": award_types,
            "date_type": "action_date",
            "date_range": {"start_date": start_date, "end_date": end_date},
            "def_codes": [],
            "agencies": [{"type": "awarding", "tier": "toptier", "name": department}],
        },
        "columns": [],
        "file_format": "csv",
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        file_url = data.get("file_url")
        file_name = data.get("file_name")
        logging.info(
            f"Download requested for {department} ({start_date} to {end_date}), award type: {sub_award_type}"
        )
        logging.info(f"Status URL: {data.get('status_url', 'N/A')}")
        logging.info(f"File URL: {file_url}")
        logging.info(f"File name: {file_name}")
        return file_url
    except requests.exceptions.RequestException as e:
        logging.error(f"Error requesting download: {str(e)}")
        return None


def check_file_status(file_url):
    """Check if a file is ready for download"""
    try:
        response = requests.head(file_url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def fetch_download(
    file_url, department, start_date, end_date, sub_award_type="contract"
):
    """Fetch a file from the provided URL and save it"""
    # Create a deterministic filename based on parameters
    dept_name = department.replace(" ", "_").lower()
    filename = f"{dept_name}_{sub_award_type}_{start_date}_to_{end_date}.zip"
    download_dir = "raw_data"
    os.makedirs(download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, filename)

    # Check if file already exists
    if os.path.exists(file_path):
        logging.info(f"File {filename} already exists, skipping download")
        return file_path

    # Wait for file to be ready
    max_attempts = 40
    attempts = 0
    while attempts < max_attempts:
        if check_file_status(file_url):
            break
        logging.info(f"File not ready yet, waiting... ({attempts+1}/{max_attempts})")
        time.sleep(15)
        attempts += 1

    if attempts == max_attempts:
        logging.error("File never became ready for download")
        return None

    # Download the file
    try:
        logging.info(f"Downloading {filename}...")
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Download complete: {filename}")
        return file_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading file: {str(e)}")
        # Clean up partial download
        if os.path.exists(file_path):
            os.remove(file_path)
        return None


def check_and_download_missing(file_urls, successful_downloads):
    """Check for missing downloads and try to recover them"""
    successful_filenames = [os.path.basename(f) for f in successful_downloads if f]
    missing_files = []

    for (
        department,
        start_date,
        end_date,
        sub_award_type,
    ), file_url in file_urls.items():
        dept_name = department.replace(" ", "_").lower()
        expected_filename = (
            f"{dept_name}_{sub_award_type}_{start_date}_to_{end_date}.zip"
        )

        if expected_filename not in successful_filenames:
            logging.info(f"Attempting to recover {expected_filename}...")
            filename = fetch_download(
                file_url, department, start_date, end_date, sub_award_type
            )
            if not filename:
                missing_files.append(expected_filename)

    return missing_files


def create_date_ranges(start_date_str, end_date_str, interval_months=3):
    """Create date ranges with specified interval in months"""
    # Convert string dates to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    date_ranges = []
    current_start = start_date

    while current_start < end_date:
        # Simply add interval_months and subtract 1 day
        current_end = (
            current_start + relativedelta(months=interval_months) - timedelta(days=1)
        )

        # Ensure we don't go beyond the end_date
        if current_end > end_date:
            current_end = end_date

        # Add the date range as strings
        date_ranges.append(
            (current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d"))
        )

        # Move to the next interval
        current_start = current_end + timedelta(days=1)

    return date_ranges


def main(
    department,
    sub_award_type="contract",
    start_date=None,
    end_date=None,
):
    """
    Main function to download contract data from USA Spending API

    Args:
        department: Department to download
        sub_award_type: Type of award to download (contract or grant)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Set default end date to today in UTC if not provided
    if end_date is None:
        end_date = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    # Set default start date if not provided
    if start_date is None:
        start_date = "2024-01-01"  # Default to Jan 1, 2024

    # Create date ranges in 3-month intervals
    date_ranges = create_date_ranges(start_date, end_date, interval_months=3)

    logging.info(f"Processing date ranges: {date_ranges}")
    logging.info(f"Department: {department}, Award Type: {sub_award_type}")

    # Step 1: Fire off all requests
    file_urls = {}
    logging.info("Initiating all download requests...")
    for start_date, end_date in date_ranges:
        file_url = request_download(start_date, end_date, department, sub_award_type)
        file_urls[(department, start_date, end_date, sub_award_type)] = file_url
        time.sleep(10)

    # Step 2: Fetch the files
    successful_downloads = []
    logging.info("\nFetching generated files...")
    for (
        department,
        start_date,
        end_date,
        sub_award_type,
    ), file_url in file_urls.items():
        if file_url:
            filename = fetch_download(
                file_url, department, start_date, end_date, sub_award_type
            )
            if filename:
                successful_downloads.append(filename)
        else:
            logging.warning(
                f"{department} ({start_date} to {end_date}): No file_url from initial request."
            )

    # Step 3: Cleanup - only try to download files that weren't successfully downloaded
    if len(successful_downloads) < len(date_ranges):
        logging.info("\nChecking for missing downloads...")
        still_missing = check_and_download_missing(file_urls, successful_downloads)
        if still_missing:
            logging.error(f"Couldn't recover these: {still_missing}")
            return 1
        else:
            logging.info("All missing files recovered!")
    else:
        logging.info("\nAll files downloaded successfully!")

    if successful_downloads:
        return 0
    else:
        logging.error("No files were successfully downloaded")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download contract data from USA Spending API"
    )
    parser.add_argument("--department", required=True, help="Department to download")
    parser.add_argument(
        "--sub-award-type",
        default="contract",
        choices=["contract", "grant"],
        help="Type of award to download (default: contract)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date in YYYY-MM-DD format (default: 2024-01-01)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date in YYYY-MM-DD format (default: today's date)",
    )

    args = parser.parse_args()

    sys.exit(main(args.department, args.sub_award_type, args.start_date, args.end_date))
