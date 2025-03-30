"""
Visualization utilities for contract similarity detection.
This module provides functions for visualizing similarity detection results.
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


def plot_similarity_distribution(
    df: pd.DataFrame,
    score_column: str = "similarity_score",
    threshold: Optional[float] = None,
) -> None:
    """
    Plot the distribution of similarity scores.

    Args:
        df: DataFrame containing similarity scores
        score_column: Name of the column containing similarity scores
        threshold: Threshold for similarity detection (optional)
    """
    if score_column not in df.columns:
        logger.error(f"Column '{score_column}' not found in DataFrame")
        return

    plt.figure(figsize=(12, 6))

    # Plot histogram
    sns.histplot(df[score_column], kde=True)

    # Add threshold line if provided
    if threshold is not None:
        plt.axvline(
            x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold:.3f}"
        )
        plt.legend()

    # Add labels
    plt.title("Distribution of similarity Scores")
    plt.xlabel("Similarity Score (lower is more similar)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_top_similarities(
    df: pd.DataFrame,
    top_n: int = 20,
    score_column: str = "similarity_score",
    description_column: str = "description",
    value_column: str = "value",
) -> None:
    """
    Plot the top N similar contracts.

    Args:
        df: DataFrame containing similarity scores and contract data
        top_n: Number of top similar contracts to plot
        score_column: Name of the column containing similarity scores
        description_column: Name of the column containing contract descriptions
        value_column: Name of the column containing contract values
    """
    if score_column not in df.columns:
        logger.error(f"Column '{score_column}' not found in DataFrame")
        return

    # Sort by similarity score (ascending, as lower scores are more similar)
    sorted_df = df.sort_values(score_column).reset_index(drop=True)

    # Get top N similar contracts
    top_similarities = sorted_df.head(top_n)

    plt.figure(figsize=(12, 8))

    # Plot scores
    ax = sns.barplot(
        x=range(len(top_similarities)), y=score_column, data=top_similarities
    )

    # Add labels
    plt.title(f"Top {top_n} Similar Contracts")
    plt.xlabel("Contract Index")
    plt.ylabel("Similarity Score (lower is more similar)")

    # Add contract details as annotations
    if description_column in df.columns and value_column in df.columns:
        for i, (_, row) in enumerate(top_similarities.iterrows()):
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


def plot_agency_similarity_counts(
    df: pd.DataFrame,
    is_similarity_column: str = "for_review",
    agency_column: str = "agency",
) -> None:
    """
    Plot the count of similar contracts by agency.

    Args:
        df: DataFrame containing similarity predictions and agency data
        is_similarity_column: Name of the column indicating similarity status (True/False)
        agency_column: Name of the column containing agency names
    """
    if is_similarity_column not in df.columns or agency_column not in df.columns:
        logger.error(f"Required columns not found in DataFrame")
        return

    # Group by agency and count similarities
    agency_counts = (
        df.groupby(agency_column)[is_similarity_column]
        .agg(["sum", "count"])
        .reset_index()
    )
    agency_counts.columns = [agency_column, "similarity_count", "total_count"]
    agency_counts["similarity_percentage"] = (
        agency_counts["similarity_count"] / agency_counts["total_count"] * 100
    )

    # Sort by similarity percentage
    agency_counts = agency_counts.sort_values(
        "similarity_percentage", ascending=False
    ).reset_index(drop=True)

    # Take top 15 agencies
    top_agencies = agency_counts.head(15)

    plt.figure(figsize=(14, 8))

    # Plot similarity percentage
    ax = sns.barplot(
        x="similarity_percentage", y=agency_column, data=top_agencies, color="skyblue"
    )

    # Add count annotations
    for i, row in enumerate(top_agencies.itertuples()):
        ax.text(
            row.similarity_percentage + 0.5,
            i,
            f"{row.similarity_count}/{row.total_count} ({row.similarity_percentage:.1f}%)",
            va="center",
        )

    # Add labels
    plt.title("Percentage of similar Contracts by Agency (Top 15)")
    plt.xlabel("Similarity Percentage (%)")
    plt.ylabel("Agency")

    plt.tight_layout()
    plt.show()


def plot_value_vs_similarity_score(
    df: pd.DataFrame,
    score_column: str = "similarity_score",
    value_column: str = "normalized_value",
    is_similarity_column: str = "for_review",
    log_scale: bool = True,
) -> None:
    """
    Plot contract value vs. similarity score.

    Args:
        df: DataFrame containing similarity scores and contract values
        score_column: Name of the column containing similarity scores
        value_column: Name of the column containing contract values
        is_similarity_column: Name of the column indicating similarity status (True/False)
        log_scale: Whether to use log scale for contract values
    """
    if (
        score_column not in df.columns
        or value_column not in df.columns
        or is_similarity_column not in df.columns
    ):
        logger.error(f"Required columns not found in DataFrame")
        return

    plt.figure(figsize=(12, 8))

    # Create scatter plot
    scatter = plt.scatter(
        df[score_column],
        df[value_column],
        c=df[is_similarity_column].astype(int),
        cmap="coolwarm",
        alpha=0.6,
    )

    # Add legend
    legend1 = plt.legend(
        *scatter.legend_elements(), title="Similarity", labels=["Normal", "Similar"]
    )
    plt.gca().add_artist(legend1)

    # Set log scale if requested
    if log_scale and (df[value_column] > 0).all():
        plt.yscale("log")

    # Add labels
    plt.title("Contract Value vs. Similarity Score")
    plt.xlabel("Similarity Score (lower is more similar)")
    plt.ylabel("Contract Value" + (" (log scale)" if log_scale else ""))

    plt.tight_layout()
    plt.show()


def plot_similarity_metrics(
    thresholds: List[float], metrics: List[Dict[str, float]]
) -> None:
    """
    Plot precision, recall, and F1 score for different thresholds.

    Args:
        thresholds: List of threshold values
        metrics: List of dictionaries containing metrics for each threshold
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
    plt.title("Similarity Detection Metrics vs. Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
