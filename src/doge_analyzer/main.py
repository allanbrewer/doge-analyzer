"""
Main script for running the contract anomaly detection pipeline.
Identifies active contracts similar to a baseline set (e.g., canceled contracts).
"""

import os
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt

# Import the main execution function and the pipeline class
from doge_analyzer.inference.pipeline import run_pipeline, ContractAnomalyPipeline
from doge_analyzer.utils.visualization import (
    plot_anomaly_distribution,  # Renamed function
    plot_top_anomalies,  # Renamed function
    plot_agency_anomaly_counts,  # Renamed function
    plot_value_vs_anomaly_score,  # Renamed function
)

# Removed duplicate import, now handled above

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Contract Anomaly Detection Pipeline (Flags contracts similar to training data)"
    )

    # Required arguments
    parser.add_argument(
        "--labeled_data",
        type=str,
        required=True,
        help="Path to the labeled data file (JSON, e.g., canceled contracts for training)",
    )
    parser.add_argument(
        "--unlabeled_data",
        type=str,
        required=True,
        help="Path to the unlabeled data directory or file (ZIP or CSV, e.g., active contracts)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results and model",
    )

    # Optional arguments
    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-uncased",
        help="Name of the BERT model to use",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of base estimators for Isolation Forest",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,  # Note: This affects model.predict(), not predict_with_threshold()
        help="Expected proportion of anomalies (used by Isolation Forest internal prediction)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for BERT feature extraction",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom threshold for anomaly detection (overrides model's fitted threshold; higher values flag more similar contracts)",
    )
    parser.add_argument(
        "--extract_dir",
        type=str,
        default=None,
        help="Directory to extract zip files",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of contracts to sample",
    )
    parser.add_argument(
        "--department",
        type=str,
        default=None,
        help="Filter to only include files from a specific department",
    )
    parser.add_argument(
        "--no_save_model",
        action="store_true",
        help="Do not save the trained model",
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Do not generate visualizations",
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default=None,
        help="Directory containing a pre-trained model to load (skips training)",
    )

    return parser.parse_args()


def main():
    """Main function to run the pipeline."""
    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.load_model_dir:
        # Load pre-trained pipeline
        logger.info(f"Loading pre-trained anomaly pipeline from {args.load_model_dir}")
        # Note: labeled_data_path is not needed for loading the pipeline itself anymore
        pipeline = ContractAnomalyPipeline.load_pipeline(
            args.load_model_dir,
            bert_model_name=args.bert_model,
        )
        # Predict anomalies using the loaded pipeline
        result_df = pipeline.predict(
            unlabeled_data_paths=args.unlabeled_data,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            threshold=args.threshold,
            extract_dir=args.extract_dir,
            sample_size=args.sample_size,
            department_filter=args.department,
        )
    else:
        # Run the full pipeline (fit and predict)
        result_df = run_pipeline(
            labeled_data_path=args.labeled_data,
            unlabeled_data_paths=args.unlabeled_data,
            output_dir=args.output_dir,
            bert_model_name=args.bert_model,
            n_estimators=args.n_estimators,
            contamination=args.contamination,
            batch_size=args.batch_size,
            threshold=args.threshold,
            extract_dir=args.extract_dir,
            sample_size=args.sample_size,
            department_filter=args.department,
            save_model=not args.no_save_model,
        )

    # Generate visualizations
    if not args.no_visualize and not result_df.empty:
        logger.info("Generating visualizations")

        # Create visualizations directory
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Plot anomaly score distribution
        plot_anomaly_distribution(
            result_df,
            threshold=(
                pipeline.anomaly_detector.threshold
                if args.threshold is None
                else args.threshold
            ),
        )
        plt.savefig(os.path.join(viz_dir, "anomaly_distribution.png"))

        # Plot top anomalies (most similar)
        plot_top_anomalies(result_df)
        plt.savefig(os.path.join(viz_dir, "top_anomalies.png"))

        # Plot agency anomaly counts
        plot_agency_anomaly_counts(result_df)
        plt.savefig(os.path.join(viz_dir, "agency_anomaly_counts.png"))

        # Plot value vs anomaly score
        plot_value_vs_anomaly_score(result_df)
        plt.savefig(os.path.join(viz_dir, "value_vs_anomaly_score.png"))

        logger.info(f"Visualizations saved to {viz_dir}")

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
