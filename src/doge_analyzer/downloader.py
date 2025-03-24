import json
import requests
import time
import argparse
import os
import logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def fetch_savings_data(endpoint="/savings/contracts", page=1, per_page=100):
    """
    Fetch savings data from the specified DOGE API endpoint.

    Args:
        endpoint (str): API endpoint, e.g., '/savings/contracts', '/savings/grants', '/savings/leases'
        page (int): Page number for pagination
        per_page (int): Number of records per page

    Returns:
        None (saves data to a JSON file)
    """

    logging.info(f"Fetching data from endpoint: {endpoint}")

    # Validate endpoint
    valid_endpoints = [
        "/savings/contracts",
        "/savings/grants",
        "/savings/leases",
        "/payments",
    ]
    if endpoint not in valid_endpoints:
        logging.info(f"Invalid endpoint: {endpoint}. Must be one of {valid_endpoints}")
        return

    # Base URL and full endpoint
    base_url = "https://api.doge.gov"
    url = f"{base_url}{endpoint}"

    # Headers to mimic a browser (add API key if required by docs)
    headers = {
        "Accept": "application/json",
    }

    # Store all savings data
    all_savings = []
    current_page = 1
    has_more = True

    # Check for data directory to save
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check for endpoint directory
    endpoint_dir = f"{data_dir}/{endpoint.split('/')[-1]}"
    if not os.path.exists(endpoint_dir):
        os.makedirs(endpoint_dir)

    while has_more:
        try:
            # Add pagination to the request
            params = {"page": current_page, "per_page": per_page}
            response = requests.get(url, headers=headers, params=params, timeout=10)

            # Check status code
            if response.status_code == 200:
                data = response.json()
                logging.info(f"Response data: {data.get('success', False)}")

                # Extract savings data (adjust key based on response structure)
                savings = data.get("result", {}).get(f"{endpoint.split('/')[-1]}", [])
                if not savings:  # No more data
                    has_more = False
                    logging.info("No more data found.")
                    break

                all_savings.extend(savings)
                logging.info(
                    f"Page {current_page}: Retrieved {len(savings)} records (Total: {len(all_savings)})"
                )

                # Get number of total pages
                total_pages = data.get("meta", {}).get("pages", None)

                # Check current page vs requested pages
                if current_page >= page:
                    has_more = False
                    logging.info(f"Reached the last page: {total_pages}")

                # Check total pages vs current page
                logging.info(f"Total pages: {total_pages}")
                if total_pages and current_page >= total_pages:
                    has_more = False
                    logging.info(f"Reached the last page: {total_pages}")
                else:
                    current_page += 1

            elif response.status_code == 403:
                logging.info("403 Forbidden - Check headers, cookies, or API key")
                logging.info(response.text)
                break
            elif response.status_code == 429:
                logging.info("429 Too Many Requests - Sleeping for 60 seconds")
                time.sleep(60)
            else:
                logging.info(f"Error {response.status_code}: {response.text}")
                break

        except Exception as e:
            logging.info(f"Request failed: {e}")
            break

        # Avoid rate limiting
        time.sleep(1)

    # Save to file
    if all_savings:
        filename = f"{endpoint_dir}/doge_{endpoint.split('/')[-1]}.json"  # e.g., doge_contracts.json
        with open(filename, "w") as f:
            json.dump(all_savings, f, indent=2)
        logging.info(f"Saved {len(all_savings)} records to {filename}")

        # Print sample of first record
        logging.info("Sample of first record:")
        for item in all_savings[:1]:
            logging.info(item)
    else:
        logging.info("No data retrieved")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fetch savings data from DOGE API")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/savings/contracts",
        choices=[
            "/savings/contracts",
            "/savings/grants",
            "/savings/leases",
            "/payments",
        ],
        help="API endpoint to fetch data from (default: /savings/contracts)",
    )
    parser.add_argument(
        "--per_page",
        type=int,
        default=100,
        help="Number of records per page (default: 100)",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Starting page (default: 1)",
    )
    args = parser.parse_args()

    # Run the fetch function with the specified endpoint
    fetch_savings_data(endpoint=args.endpoint, page=args.page, per_page=args.per_page)


if __name__ == "__main__":
    main()
