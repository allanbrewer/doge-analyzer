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
import glob  # Added for file searching
from pathlib import Path  # Added for finding latest file

# Import the main execution function and the pipeline class
from doge_analyzer.inference.pipeline import run_pipeline, ContractAnomalyPipeline
from doge_analyzer.utils.visualization import (
    plot_anomaly_distribution,  # Renamed function
    plot_top_anomalies,  # Renamed function
    plot_agency_anomaly_counts,  # Renamed function
    plot_value_vs_anomaly_score,
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_latest_json_file(directory: str) -> Optional[str]:
    """Finds the most recently modified JSON file in a directory."""
    try:
        data_dir = Path(directory)
        json_files = list(data_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in directory: {directory}")
            return None
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest data file: {latest_file}")
        return str(latest_file)
    except FileNotFoundError:
        logger.error(f"Data directory not found: {directory}")
        return None
    except Exception as e:
        logger.error(f"Error finding latest file in {directory}: {e}")
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Anomaly Detection Pipeline for Contracts or Grants"
    )

    # Required arguments
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["contracts", "grants"],
        help="Type of data to process ('contracts' or 'grants'). Determines data input and output subdirectories.",
    )
    # Labeled data arguments (either specify directory for latest JSON or a specific file)
    parser.add_argument(
        "--labeled_data_dir",
        type=str,
        default=None,
        help="Directory containing labeled data JSON files (e.g., './data/contracts'). If provided, the latest JSON file will be used. Overridden by --labeled_data_file.",
    )
    parser.add_argument(
        "--labeled_data_file",
        type=str,
        default=None,
        help="Path to a specific labeled data file (JSON). Overrides --labeled_data_dir.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory to save results and model (e.g., './results'). Subdirectories for data_type will be created.",
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
        help="Custom threshold for anomaly detection (overrides model's fitted threshold; higher values flag more similar items)",
    )
    parser.add_argument(
        "--extract_dir",
        type=str,
        default=None,
        help="Directory to extract zip files (if unlabeled data is zip)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of items to sample from unlabeled data",
    )
    parser.add_argument(
        "--department",
        type=str,
        default=None,
        help="Filter to only include files from a specific department (if applicable to data format)",
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
        "--load_model_base_dir",  # Renamed for clarity
        type=str,
        default=None,
        help="Base directory containing a pre-trained model to load (e.g., './results'). The specific model will be loaded from '{load_model_base_dir}/{data_type}/model'. Skips training.",
    )

    return parser.parse_args()


def main():
    """Main function to run the pipeline."""
    args = parse_args()

    # --- Determine Labeled Data Path ---
    labeled_data_path = None
    if args.labeled_data_file:
        labeled_data_path = Path(args.labeled_data_file)
        if not labeled_data_path.is_file():
            logger.error(
                f"Specified labeled data file not found: {labeled_data_path}. Exiting."
            )
            return
        logger.info(f"Using specified labeled data file: {labeled_data_path}")
    else:
        # Default or specified directory for labeled data
        labeled_dir_path_str = (
            args.labeled_data_dir
            if args.labeled_data_dir
            else f"./data/{args.data_type}"
        )
        labeled_dir = Path(labeled_dir_path_str)
        logger.info(f"Searching for latest labeled data JSON in: {labeled_dir}")
        latest_labeled_file = find_latest_json_file(str(labeled_dir))
        if not latest_labeled_file:
            logger.error(
                f"Could not find any labeled data JSON file in {labeled_dir}. Exiting."
            )
            return
        labeled_data_path = Path(latest_labeled_file)

    # --- Determine Unlabeled Data Path ---
    # Unlabeled data is expected in ./data/unlabeled/{data_type}/ (directory containing ZIP/CSV)
    unlabeled_data_dir = Path("./data/unlabeled") / args.data_type
    if not unlabeled_data_dir.is_dir():
        logger.error(
            f"Unlabeled data directory not found: {unlabeled_data_dir}. Please ensure it exists and contains data (ZIP/CSV). Exiting."
        )
        return  # Exit if unlabeled data directory doesn't exist
    logger.info(f"Using unlabeled data from directory: {unlabeled_data_dir}")

    # --- Determine Output Paths ---
    output_dir = Path(args.output_dir) / args.data_type
    model_dir = output_dir / "model"  # Used for saving or loading

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    # --- Handle model loading or training ---
    model_to_load = None
    if args.load_model_base_dir:
        potential_model_path = Path(args.load_model_base_dir) / args.data_type / "model"
        if potential_model_path.is_dir():
            model_to_load = str(potential_model_path)
            logger.info(f"Attempting to load pre-trained model from: {model_to_load}")
        else:
            logger.warning(
                f"Specified load directory {potential_model_path} not found. Proceeding with training."
            )

    if model_to_load:
        # Load pre-trained pipeline
        try:
            pipeline = ContractAnomalyPipeline.load_pipeline(
                model_to_load,
                bert_model_name=args.bert_model,  # Bert model name might still be relevant
            )
            logger.info(
                f"Successfully loaded pre-trained pipeline from {model_to_load}"
            )
            # Predict anomalies using the loaded pipeline
            result_df = pipeline.predict(
                unlabeled_data_paths=str(unlabeled_data_dir),  # Pass the directory path
                output_dir=str(output_dir),  # Use derived output dir
                batch_size=args.batch_size,
                threshold=args.threshold,
                extract_dir=args.extract_dir,  # Keep if needed
                sample_size=args.sample_size,
                department_filter=args.department,  # Keep if needed
            )
        except Exception as e:
            logger.error(
                f"Failed to load or predict with model from {model_to_load}: {e}"
            )
            logger.info("Proceeding with training a new model.")
            model_to_load = None  # Reset flag so we train below

    # Train a new model if not loaded or loading failed
    if not model_to_load:
        logger.info("Training new anomaly detection model.")
        # Run the full pipeline (fit and predict)
        result_df = run_pipeline(
            labeled_data_path=str(
                labeled_data_path
            ),  # Use determined labeled data path
            unlabeled_data_paths=str(unlabeled_data_dir),  # Pass the directory path
            output_dir=str(output_dir),  # Use derived output dir
            bert_model_name=args.bert_model,
            n_estimators=args.n_estimators,
            contamination=args.contamination,
            batch_size=args.batch_size,
            threshold=args.threshold,
            extract_dir=args.extract_dir,  # Keep if needed
            sample_size=args.sample_size,
            department_filter=args.department,  # Keep if needed
            save_model=not args.no_save_model,  # Save to derived model_dir path inside run_pipeline
        )

    # Generate visualizations
    # --- Generate visualizations ---
    if not args.no_visualize and result_df is not None and not result_df.empty:
        logger.info("Generating visualizations")

        # Create visualizations directory within the specific output folder
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Determine threshold used (either from loaded model or args)
        final_threshold = args.threshold
        if (
            final_threshold is None
            and "pipeline" in locals()
            and hasattr(pipeline, "anomaly_detector")
            and hasattr(pipeline.anomaly_detector, "threshold")
        ):
            final_threshold = pipeline.anomaly_detector.threshold
        elif final_threshold is None:
            logger.warning("Could not determine threshold for visualization.")
            # Provide a default or skip plots requiring threshold if necessary

        # Plot anomaly score distribution
        if final_threshold is not None:
            plot_anomaly_distribution(result_df, threshold=final_threshold)
            plt.savefig(viz_dir / "anomaly_distribution.png")
            plt.close()  # Close plot to free memory
        else:
            logger.warning(
                "Skipping anomaly distribution plot due to missing threshold."
            )

        # Plot top anomalies (most similar)
        # Plot top anomalies (most similar)
        plot_top_anomalies(result_df)
        plt.savefig(viz_dir / "top_anomalies.png")
        plt.close()

        # Plot agency anomaly counts
        plot_agency_anomaly_counts(result_df)
        plt.savefig(viz_dir / "agency_anomaly_counts.png")
        plt.close()

        # Plot value vs anomaly score
        plot_value_vs_anomaly_score(result_df)
        plt.savefig(viz_dir / "value_vs_anomaly_score.png")
        plt.close()

        logger.info(f"Visualizations saved to {viz_dir}")

    elif result_df is None:
        logger.warning("Skipping visualizations because pipeline execution failed.")

    logger.info(f"Pipeline completed for data type: {args.data_type}")


if __name__ == "__main__":
    main()
