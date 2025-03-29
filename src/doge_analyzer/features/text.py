"""
Text feature extraction module using BERT.
This module handles extracting features from contract descriptions using BERT.
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import logging
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default BERT model
DEFAULT_MODEL_NAME = "bert-base-uncased"


class BertFeatureExtractor:
    """
    Extract features from text using BERT.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initialize the BERT feature extractor.

        Args:
            model_name: Name of the BERT model to use
        """
        logger.info(f"Initializing BERT feature extractor with model: {model_name}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Move model to device
        self.model = self.model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

    def extract_features(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Extract features from a list of texts.

        Args:
            texts: List of texts to extract features from
            batch_size: Batch size for processing

        Returns:
            Array of features with shape (len(texts), embedding_dim)
        """
        # Handle empty input
        if not texts:
            return np.array([])

        # Process in batches
        all_embeddings = []

        for i in tqdm(
            range(0, len(texts), batch_size), desc="Extracting BERT features"
        ):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # Extract features
            with torch.no_grad():
                model_output = self.model(**encoded_input)

                # Use CLS token embedding as the sentence embedding
                # Shape: (batch_size, hidden_size)
                embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()

                all_embeddings.append(embeddings)

        # Combine all batches
        combined_embeddings = np.vstack(all_embeddings)

        logger.info(f"Extracted features with shape: {combined_embeddings.shape}")

        return combined_embeddings

    def extract_features_from_df(
        self,
        df: pd.DataFrame,
        text_column: str = "clean_description",
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Extract features from a DataFrame column.

        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text
            batch_size: Batch size for processing

        Returns:
            Array of features with shape (len(df), embedding_dim)
        """
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in DataFrame")
            return np.array([])

        # Get texts from DataFrame
        texts = df[text_column].fillna("").tolist()

        # Extract features
        return self.extract_features(texts, batch_size)


def extract_text_features(
    df: pd.DataFrame,
    text_column: str = "clean_description",
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Extract text features from a DataFrame using BERT.

    Args:
        df: DataFrame containing text data
        text_column: Name of the column containing text
        model_name: Name of the BERT model to use
        batch_size: Batch size for processing

    Returns:
        Array of features with shape (len(df), embedding_dim)
    """
    # Initialize feature extractor
    extractor = BertFeatureExtractor(model_name)

    # Extract features
    features = extractor.extract_features_from_df(df, text_column, batch_size)

    return features
