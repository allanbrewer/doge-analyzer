import cloudscraper
import json
import requests
import time
import argparse


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
    # Validate endpoint
    valid_endpoints = [
        "/savings/contracts",
        "/savings/grants",
        "/savings/leases",
        "/payments",
    ]
    if endpoint not in valid_endpoints:
        print(f"Invalid endpoint: {endpoint}. Must be one of {valid_endpoints}")
        return

    # Initialize cloudscraper to handle Cloudflare challenges
    scraper = cloudscraper.create_scraper()

    # Base URL and full endpoint
    base_url = "https://api.doge.gov"
    url = f"{base_url}{endpoint}"

    # Headers to mimic a browser (add API key if required by docs)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://doge.gov/savings",
    }

    # Store all savings data
    all_savings = []

    has_more = True

    while has_more:
        try:
            # Add pagination to the request
            params = {"page": page, "per_page": per_page}
            response = scraper.get(url, headers=headers, params=params, timeout=10)

            # Check status code
            if response.status_code == 200:
                data = response.json()

                # Extract savings data (adjust key based on response structure)
                savings = data.get("data", [])  # Could be "results", "contracts", etc.
                if not savings:  # No more data
                    has_more = False
                    break

                all_savings.extend(savings)
                print(
                    f"Page {page}: Retrieved {len(savings)} records (Total: {len(all_savings)})"
                )

                # Check pagination metadata
                total_pages = data.get("meta", {}).get("pages", None)
                if total_pages and page >= total_pages:
                    has_more = False
                else:
                    page += 1

            elif response.status_code == 403:
                print("403 Forbidden - Check headers, cookies, or API key")
                print(response.text)
                break
            elif response.status_code == 429:
                print("429 Too Many Requests - Sleeping for 60 seconds")
                time.sleep(60)
            else:
                print(f"Error {response.status_code}: {response.text}")
                break

        except Exception as e:
            print(f"Request failed: {e}")
            break

        # Avoid rate limiting
        time.sleep(1)

    # Save to file
    if all_savings:
        filename = f"doge_{endpoint.split('/')[-1]}.json"  # e.g., doge_contracts.json
        with open(filename, "w") as f:
            json.dump(all_savings, f, indent=2)
        print(f"Saved {len(all_savings)} records to {filename}")

        # Print sample of first few records
        print("Sample of first 3 records:")
        for item in all_savings[:3]:
            print(item)
    else:
        print("No data retrieved")


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
