"""
Visualization utilities for contract anomaly detection.
This module provides functions for visualizing anomaly detection results,
where anomalies represent contracts similar to the training data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple, Union

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_anomaly_distribution(
    df: pd.DataFrame,
    score_column: str = "anomaly_score",  # Updated default column name
    threshold: Optional[float] = None,
) -> None:
    """
    Plot the distribution of anomaly scores.

    Args:
        df: DataFrame containing anomaly scores
        score_column: Name of the column containing anomaly scores
        threshold: Anomaly detection threshold (optional, higher scores are anomalous)
    """
    if score_column not in df.columns:
        logger.error(f"Column '{score_column}' not found in DataFrame")
        return

    plt.figure(figsize=(12, 6))

    # Plot histogram
    sns.histplot(df[score_column], kde=True)

    # Add threshold line if provided
    if threshold is not None:
        # Note: Anomalies are typically *above* the threshold in our corrected logic
        plt.axvline(
            x=threshold,
            color="r",
            linestyle="--",
            label=f"Threshold (Anomalous >): {threshold:.3f}",
        )
        plt.legend()

    # Add labels
    plt.title("Distribution of Anomaly Scores")
    plt.xlabel("Anomaly Score (higher indicates similarity to training data)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_top_anomalies(
    df: pd.DataFrame,
    top_n: int = 20,
    score_column: str = "anomaly_score",  # Updated default column name
    description_column: str = "clean_description",  # Use cleaned description if available
    value_column: str = "normalized_value",  # Use normalized value if available
) -> None:
    """
    Plot the top N anomalous contracts (most similar to training data).

    Args:
        df: DataFrame containing anomaly scores and contract data
        top_n: Number of top anomalous contracts to plot
        score_column: Name of the column containing anomaly scores
        description_column: Name of the column containing contract descriptions
        value_column: Name of the column containing contract values
    """
    if score_column not in df.columns:
        logger.error(f"Column '{score_column}' not found in DataFrame")
        return

    # Sort by anomaly score (descending, as higher scores are more anomalous/similar)
    sorted_df = df.sort_values(score_column, ascending=False).reset_index(drop=True)

    # Get top N anomalous contracts
    top_anomalies = sorted_df.head(top_n)

    plt.figure(figsize=(12, 8))

    # Plot scores
    ax = sns.barplot(x=range(len(top_anomalies)), y=score_column, data=top_anomalies)

    # Add labels
    plt.title(f"Top {top_n} Anomalous Contracts (Most Similar to Training Data)")
    plt.xlabel("Contract Rank (Sorted by Anomaly Score)")
    plt.ylabel("Anomaly Score (higher is more similar)")

    # Add contract details as annotations
    if description_column in df.columns and value_column in df.columns:
        for i, (_, row) in enumerate(top_anomalies.iterrows()):
            contract_desc = row[description_column]
            contract_value = row[value_column]

            # Truncate description if too long
            if isinstance(contract_desc, str) and len(contract_desc) > 50:
                contract_desc = contract_desc[:47] + "..."

            # Format value
            if isinstance(contract_value, (int, float)):
                value_str = f"${contract_value:,.0f}"
            else:
                value_str = str(contract_value)

            # Add annotation
            ax.annotate(
                f"{value_str}: {contract_desc}",
                xy=(i, row[score_column]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()


def plot_agency_anomaly_counts(
    df: pd.DataFrame,
    is_anomaly_column: str = "for_review",  # Updated default column name
    agency_column: str = "agency",
) -> None:
    """
    Plot the count and percentage of anomalous contracts by agency.

    Args:
        df: DataFrame containing anomaly predictions and agency data
        is_anomaly_column: Name of the column indicating anomaly status (True/False)
        agency_column: Name of the column containing agency names
    """
    if is_anomaly_column not in df.columns or agency_column not in df.columns:
        logger.error(f"Required columns not found in DataFrame")
        return

    # Group by agency and count anomalies
    agency_counts = (
        df.groupby(agency_column)[is_anomaly_column].agg(["sum", "count"]).reset_index()
    )
    agency_counts.columns = [agency_column, "anomaly_count", "total_count"]
    # Handle division by zero if an agency has 0 total contracts (shouldn't happen with groupby)
    agency_counts["anomaly_percentage"] = (
        agency_counts["anomaly_count"]
        / agency_counts["total_count"].replace(0, np.nan)
        * 100
    ).fillna(0)

    # Sort by anomaly percentage
    agency_counts = agency_counts.sort_values(
        "anomaly_percentage", ascending=False
    ).reset_index(drop=True)

    # Take top 15 agencies
    top_agencies = agency_counts.head(15)

    plt.figure(figsize=(14, 8))

    # Plot anomaly percentage
    ax = sns.barplot(
        x="anomaly_percentage",
        y=agency_column,
        data=top_agencies,
        color="lightcoral",  # Changed color
    )

    # Add count annotations
    for i, row in enumerate(top_agencies.itertuples()):
        ax.text(
            row.anomaly_percentage + 0.5,
            i,
            f"{row.anomaly_count}/{row.total_count} ({row.anomaly_percentage:.1f}%)",
            va="center",
        )

    # Add labels
    plt.title("Percentage of Anomalous Contracts by Agency (Top 15)")
    plt.xlabel("Anomaly Percentage (%)")
    plt.ylabel("Agency")

    plt.tight_layout()
    plt.show()


def plot_value_vs_anomaly_score(
    df: pd.DataFrame,
    score_column: str = "anomaly_score",  # Updated default column name
    value_column: str = "normalized_value",  # Updated default column name
    is_anomaly_column: str = "for_review",  # Updated default column name
    log_scale: bool = True,
) -> None:
    """
    Plot contract value vs. anomaly score.

    Args:
        df: DataFrame containing anomaly scores and contract values
        score_column: Name of the column containing anomaly scores
        value_column: Name of the column containing contract values
        is_anomaly_column: Name of the column indicating anomaly status (True/False)
        log_scale: Whether to use log scale for contract values
    """
    if (
        score_column not in df.columns
        or value_column not in df.columns
        or is_anomaly_column not in df.columns
    ):
        logger.error(f"Required columns not found in DataFrame")
        return

    plt.figure(figsize=(12, 8))

    # Create scatter plot
    scatter = plt.scatter(
        df[score_column],
        df[value_column],
        c=df[is_anomaly_column].astype(int),  # Color by anomaly status
        cmap="coolwarm",
        alpha=0.6,
    )

    # Add legend
    legend1 = plt.legend(
        *scatter.legend_elements(),
        title="Anomaly Status",
        labels=["Normal", "Anomalous (Similar)"],
    )
    plt.gca().add_artist(legend1)

    # Set log scale if requested
    if log_scale and (df[value_column] > 0).all():
        plt.yscale("log")

    # Add labels
    plt.title("Contract Value vs. Anomaly Score")
    plt.xlabel("Anomaly Score (higher indicates similarity to training data)")
    plt.ylabel("Contract Value" + (" (log scale)" if log_scale else ""))

    plt.tight_layout()
    plt.show()


def plot_anomaly_metrics(  # Renamed function
    thresholds: List[float], metrics: List[Dict[str, float]]
) -> None:
    """
    Plot precision, recall, and F1 score for different anomaly thresholds.
    Assumes metrics are calculated with pos_label=-1 (anomalous).

    Args:
        thresholds: List of threshold values used for anomaly detection
        metrics: List of dictionaries containing metrics (precision, recall, f1_score) for each threshold
    """
    if not thresholds or not metrics or len(thresholds) != len(metrics):
        logger.error("Invalid input for plotting metrics")
        return

    # Extract metrics
    precision = [m.get("precision", 0) for m in metrics]
    recall = [m.get("recall", 0) for m in metrics]
    f1 = [m.get("f1_score", 0) for m in metrics]

    plt.figure(figsize=(12, 6))

    # Plot metrics
    plt.plot(thresholds, precision, "b-", label="Precision")
    plt.plot(thresholds, recall, "r-", label="Recall")
    plt.plot(thresholds, f1, "g-", label="F1 Score")

    # Add labels
    plt.title("Anomaly Detection Metrics vs. Threshold")
    plt.xlabel("Anomaly Score Threshold (Anomalous if Score > Threshold)")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
