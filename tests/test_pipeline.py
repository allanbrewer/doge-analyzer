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
    Test the anomaly detection pipeline.
    """
    # Set paths
    labeled_data_path = "data/test/doge_contracts_test.json"
    output_dir = "results/test"

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

    # Convert to numpy arrays
    # Create dummy paths for unlabeled data (not used in this test)
    unlabeled_data_paths = "data/unlabeled"

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

    # Predict anomalies on testing data
    result_df = run_pipeline(
        labeled_data_path=train_file_path,
        unlabeled_data_paths=test_file_path,
        output_dir=output_dir,
        save_model=False,
    )

    # Evaluate performance
    y_pred = result_df["for_review"].values

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=True)

    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Precision: {precision}")


if __name__ == "__main__":
    test_pipeline()
