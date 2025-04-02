# scripts/create_test_fixture.py
import json
import random
import os
import argparse


def create_fixture_subset(input_path, output_path, num_samples=200, random_seed=42):
    """
    Reads a JSON file containing contract data, extracts a random subset,
    and writes it to a new JSON file.

    Args:
        input_path (str): Path to the input JSON file (full dataset).
        output_path (str): Path to the output JSON file (subset fixture).
        num_samples (int): Number of contracts to randomly select.
        random_seed (int): Seed for the random number generator for reproducibility.
    """
    print(f"Reading full contract data from: {input_path}")
    try:
        with open(input_path, "r") as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the input file: {e}")
        return

    if "data" not in full_data or not isinstance(full_data["data"], list):
        print(
            "Error: Input JSON does not contain a 'data' key with a list of contracts."
        )
        return

    contracts_list = full_data["data"]
    initial_count = len(contracts_list)
    print(f"Found {initial_count} contracts initially in the input file.")

    # Filter out contracts with specific piid values before sampling
    filter_values = ["Unavailable", "Charge Card Purchase"]
    # Assuming contracts are dictionaries and checking if 'piid' key exists and matches filter values
    filtered_contracts = [
        contract
        for contract in contracts_list
        if contract.get("piid") not in filter_values
    ]
    removed_count = initial_count - len(filtered_contracts)
    if removed_count > 0:
        print(
            f"Filtered out {removed_count} contracts with piid in {filter_values} before sampling."
        )

    contracts = filtered_contracts  # Use the filtered list for sampling
    total_contracts = len(contracts)
    print(f"Proceeding with {total_contracts} contracts after filtering.")

    if total_contracts == 0:
        print("Error: No contracts remaining after filtering. Cannot create fixture.")
        return
    elif total_contracts < num_samples:
        print(
            f"Warning: Requested {num_samples} samples, but only {total_contracts} contracts are available after filtering. Using all remaining contracts."
        )
        sampled_contracts = contracts
    else:
        print(f"Randomly selecting {num_samples} contracts from the filtered list...")
        random.seed(random_seed)  # Set seed for reproducibility
        sampled_contracts = random.sample(contracts, num_samples)
        print(f"Selected {len(sampled_contracts)} contracts.")

    output_data = {"data": sampled_contracts}

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Check if output_dir is not empty (i.e., not saving in root)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

    print(f"Writing {len(sampled_contracts)} sampled contracts to: {output_path}")
    try:
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)  # Use indent for readability
        print("Successfully created the new test fixture.")
    except Exception as e:
        print(f"An unexpected error occurred while writing the output file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a random subset fixture from a larger JSON contract file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/contracts/doge_contracts_20250401170925.json",
        help="Path to the input JSON file containing all contracts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/fixtures/doge_contracts_test.json",
        help="Path to the output JSON file for the test fixture.",
    )
    parser.add_argument(
        "--samples", type=int, default=200, help="Number of random samples to extract."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    create_fixture_subset(args.input, args.output, args.samples, args.seed)
