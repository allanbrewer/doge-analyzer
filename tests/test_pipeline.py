import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from doge_analyzer.inference.pipeline import run_pipeline
from doge_analyzer.data.load import load_labeled_data
from doge_analyzer.data.preprocess import preprocess_labeled_data
import logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_pipeline():
    """
    Test the contract anomaly detection pipeline.
    Ensures the pipeline runs and produces expected outputs.
    Note: This test checks pipeline execution and output format,
    not the semantic correctness of anomaly scores, which requires curated test data.
    """
    # Set paths (Assuming test data exists here relative to project root)
    # Using a more standard location for test fixtures
    labeled_data_path = "tests/fixtures/doge_contracts_test.json"
    output_dir = "tests/results/test_pipeline"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load labeled data
    try:
        labeled_df = load_labeled_data(labeled_data_path)
    except FileNotFoundError:
        logger.warning(
            f"Labeled data file not found at {labeled_data_path}. Skipping test."
        )
        return

    # Preprocess labeled data
    processed_labeled_df = preprocess_labeled_data(labeled_df)

    # Split data into training and testing sets
    X = processed_labeled_df.drop("is_canceled", axis=1)
    y = processed_labeled_df["is_canceled"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save training and testing data to separate files
    train_data = X_train.copy()
    train_data["is_canceled"] = y_train
    test_data = X_test.copy()
    test_data["is_canceled"] = y_test

    train_file_path = os.path.join(output_dir, "train_data.json")
    test_file_path = os.path.join(output_dir, "test_data.json")

    os.makedirs(output_dir, exist_ok=True)
    train_data.to_json(train_file_path, orient="records")
    test_data.to_json(test_file_path, orient="records")

    # Run the pipeline: Train on train_file_path, predict on test_file_path
    result_df = run_pipeline(
        labeled_data_path=train_file_path,
        unlabeled_data_paths=test_file_path,
        output_dir=output_dir,
        save_model=False,
    )

    # --- Basic Checks ---
    logger.info(f"Pipeline executed. Output DataFrame shape: {result_df.shape}")

    # Check if the output DataFrame is not empty
    assert not result_df.empty, "Output DataFrame should not be empty"

    # Check if the number of rows matches the test input
    assert len(result_df) == len(
        test_data
    ), "Output DataFrame row count should match test data"

    # Check for expected columns
    # Add other essential columns from preprocess_unlabeled_data if needed for checks
    expected_columns = [
        "anomaly_score",
        "for_review",
        "piid",
        "clean_description",
    ]
    for col in expected_columns:
        assert col in result_df.columns, f"Expected column '{col}' not found in output"

    # Check data types
    assert pd.api.types.is_numeric_dtype(
        result_df["anomaly_score"]
    ), "'anomaly_score' should be numeric"
    assert pd.api.types.is_bool_dtype(
        result_df["for_review"]
    ), "'for_review' should be boolean"

    # Check if any contracts were flagged (optional, depends on data)
    num_flagged = result_df["for_review"].sum()
    logger.info(f"Number of contracts flagged for review: {num_flagged}")
    # Consider adding an assertion here if the test data guarantees some flags, e.g.:
    # assert num_flagged > 0, "Expected at least one contract to be flagged (adjust based on test data)"

    logger.info("Basic pipeline output checks passed.")


if __name__ == "__main__":
    test_pipeline()
