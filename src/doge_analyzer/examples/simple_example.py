"""
Simple example of using the contract anomaly detection pipeline.
Identifies contracts similar to the training data.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

from doge_analyzer.inference.pipeline import ContractAnomalyPipeline
from doge_analyzer.utils.visualization import (
    plot_anomaly_distribution,
    plot_top_anomalies,
    plot_agency_anomaly_counts,
    plot_value_vs_anomaly_score,
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_example():
    """Run a simple example of the pipeline."""
    # Set paths
    labeled_data_path = os.path.join(
        "data", "contracts", "doge_contracts_20250323222302.json"
    )
    unlabeled_data_path = os.path.join("data", "unlabeled", "contracts")
    output_dir = "results/example"

    # Create unlabeled data directory if it doesn't exist (for example purposes)
    os.makedirs(unlabeled_data_path, exist_ok=True)
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize anomaly pipeline - using BERT features again
    pipeline = ContractAnomalyPipeline(
        bert_model_name="bert-base-uncased",  # Specify BERT model name
        n_estimators=100,
        contamination=0.1,  # Affects internal predict(), not threshold logic primarily
        random_state=42,
    )

    # Load pipeline if it exists, otherwise fit and save
    model_dir = os.path.join(output_dir, "model")
    model_path = os.path.join(
        model_dir, "anomaly_detector.joblib"
    )  # Check for the correct model file

    if os.path.exists(model_path):
        logger.info(f"Loading pre-trained anomaly pipeline from {model_dir}")
        # labeled_data_path is not needed for loading anymore
        pipeline = ContractAnomalyPipeline.load_pipeline(
            model_dir, bert_model_name="bert-base-uncased"
        )
    else:
        # Fit pipeline on labeled data, including BERT text feature extraction
        logger.info("Fitting anomaly pipeline on labeled data with BERT features")
        pipeline.fit(labeled_data_path, batch_size=8)

        # Save pipeline
        os.makedirs(model_dir, exist_ok=True)
        pipeline.save_pipeline(model_dir)
        logger.info(f"Pipeline saved to {model_dir}")

    # Process a small sample of unlabeled data
    logger.info(
        "Processing unlabeled data to find anomalies (similar to training data)"
    )
    result_df = pipeline.predict(
        unlabeled_data_path,
        output_dir=output_dir,
        batch_size=8,
        sample_size=100,  # Only process 100 contracts for this example
    )

    # Generate visualizations
    logger.info("Generating visualizations")
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Plot anomaly score distribution
    # Pass the threshold used (either default from fit or custom if provided to predict)
    threshold_used = (
        pipeline.anomaly_detector.threshold
        if hasattr(pipeline, "anomaly_detector") and pipeline.anomaly_detector
        else None
    )
    plot_anomaly_distribution(result_df, threshold=threshold_used)
    plt.savefig(os.path.join(viz_dir, "anomaly_distribution.png"))
    plt.close()

    # Plot top anomalies (most similar)
    plot_top_anomalies(result_df)
    plt.savefig(os.path.join(viz_dir, "top_anomalies.png"))
    plt.close()

    # Plot agency anomaly counts
    plot_agency_anomaly_counts(result_df)
    plt.savefig(os.path.join(viz_dir, "agency_anomaly_counts.png"))
    plt.close()

    # Plot value vs anomaly score
    plot_value_vs_anomaly_score(result_df)
    plt.savefig(os.path.join(viz_dir, "value_vs_anomaly_score.png"))
    plt.close()

    logger.info(f"Visualizations saved to {viz_dir}")

    # Print summary of results
    anomaly_count = result_df["for_review"].sum()
    total_count = len(result_df)
    logger.info(
        f"Found {anomaly_count} anomalous contracts (flagged for review) out of {total_count}"
    )
    if total_count > 0:
        logger.info(f"Anomaly rate: {anomaly_count / total_count:.2%}")
    else:
        logger.info("Anomaly rate: N/A (no contracts processed)")

    # Print top 5 anomalous (most similar) contracts
    logger.info("Top 5 anomalous contracts (most similar to training data):")
    # Sort by anomaly_score descending (higher score = more anomalous/similar)
    top_5 = result_df.sort_values("anomaly_score", ascending=False).head(5)
    for i, row in top_5.iterrows():
        logger.info(
            f"  Rank {i+1}: Score={row['anomaly_score']:.4f} - ID={row['piid']} - Agency={row.get('agency', 'N/A')} - Value=${row.get('normalized_value', 'N/A'):,.2f}"
        )

    logger.info("Example completed successfully")


if __name__ == "__main__":
    run_example()
