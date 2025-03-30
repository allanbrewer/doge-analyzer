"""
Simple example of using the contract similarity detection pipeline.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

from doge_analyzer.inference.pipeline import ContractSimilarityPipeline
from doge_analyzer.utils.visualization import (
    plot_similarity_distribution,
    plot_top_similarities,
    plot_agency_similarity_counts,
    plot_value_vs_similarity_score,
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
    unlabeled_data_path = os.path.join("data", "unlabeled")
    output_dir = "results/example"

    # Create unlabeled data directory if it doesn't exist (for example purposes)
    os.makedirs(unlabeled_data_path, exist_ok=True)
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize pipeline - using BERT features again
    pipeline = ContractSimilarityPipeline(
        bert_model_name="bert-base-uncased",  # Specify BERT model name
        n_estimators=100,
        contamination=0.1,
        random_state=42,
    )

    # Load pipeline if it exists, otherwise fit and save
    model_dir = os.path.join(output_dir, "model")
    model_path = os.path.join(model_dir, "similarity_detector.joblib")

    if os.path.exists(model_path):
        logger.info(f"Loading pre-trained pipeline from {model_dir}")
        pipeline = ContractSimilarityPipeline.load_pipeline(
            model_dir, labeled_data_path, bert_model_name="bert-base-uncased"
        )
    else:
        # Fit pipeline on labeled data, including BERT text feature extraction
        logger.info("Fitting pipeline on labeled data with BERT features")
        pipeline.fit(labeled_data_path, batch_size=8)

        # Save pipeline
        os.makedirs(model_dir, exist_ok=True)
        pipeline.save_pipeline(model_dir)
        logger.info(f"Pipeline saved to {model_dir}")

    # Process a small sample of unlabeled data
    logger.info("Processing unlabeled data")
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

    # Plot similarity distribution
    plot_similarity_distribution(result_df)
    plt.savefig(os.path.join(viz_dir, "similarity_distribution.png"))
    plt.close()

    # Plot top similarities
    plot_top_similarities(result_df)
    plt.savefig(os.path.join(viz_dir, "top_similarities.png"))
    plt.close()

    # Plot agency similarity counts
    plot_agency_similarity_counts(result_df)
    plt.savefig(os.path.join(viz_dir, "agency_similarity_counts.png"))
    plt.close()

    # Plot value vs similarity score
    plot_value_vs_similarity_score(result_df)
    plt.savefig(os.path.join(viz_dir, "value_vs_similarity_score.png"))
    plt.close()

    logger.info(f"Visualizations saved to {viz_dir}")

    # Print summary of results
    similarity_count = result_df["for_review"].sum()
    total_count = len(result_df)
    logger.info(f"Found {similarity_count} similar contracts out of {total_count}")
    logger.info(f"Similarity rate: {similarity_count / total_count:.2%}")

    # Print top 5 similar contracts
    logger.info("Top 5 similar contracts:")
    top_5 = result_df.sort_values("similarity_score").head(5)
    for i, row in top_5.iterrows():
        logger.info(
            f"Contract {i+1}: {row['piid']} - ${row['normalized_value']:,.2f} - Agency: {row['agency']}"
        )

    logger.info("Example completed successfully")


if __name__ == "__main__":
    run_example()
