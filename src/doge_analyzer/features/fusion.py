"""
Feature fusion module.
This module handles combining text features with numerical and categorical features.
"""

import os
import re  # Added for standardization
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
from typing import Dict, List, Optional, Tuple, Union
from joblib import dump, load

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureFusion:
    """
    Feature fusion class for combining different types of features.
    """

    def __init__(self, top_category_percentage: float = 0.2):
        """
        Initialize the feature fusion class.
        Args:
            top_category_percentage (float): Percentage of top categories (agencies, vendors) to keep.
                                             Defaults to 0.2 (20%).
        """
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            # Ensure consistent feature names across fit/transform
            feature_name_combiner="concat",
        )
        # Removed 'savings' as requested (it wasn't here anyway)
        self.numerical_columns = ["normalized_value"]
        # Added 'vendor'
        self.categorical_columns = ["agency", "vendor"]
        self.fitted = False
        self.top_category_percentage = top_category_percentage
        self.top_agencies = []
        self.top_vendors = []  # Added for vendors
        # Store the categories used during fit for consistent encoding
        self._encoder_categories: Optional[List[np.ndarray]] = None

    def extract_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract numerical features from a DataFrame.
        Args:
            df: DataFrame containing contract data
        Returns:
            Array of numerical features
        """
        # Filter to only include columns that exist in the DataFrame
        available_columns = [col for col in self.numerical_columns if col in df.columns]

        if not available_columns:
            logger.warning("No numerical columns found in DataFrame")
            return np.zeros((len(df), 0))

        # Extract numerical features
        numerical_features = df[available_columns].fillna(0).values

        logger.info(
            f"Extracted numerical features with shape: {numerical_features.shape}"
        )

        return numerical_features

    def _standardize_name(self, name: str) -> str:
        """Standardizes agency/vendor names for comparison."""
        if not isinstance(name, str):
            return "Unknown"  # Handle non-string inputs
        name = name.lower()
        # Remove punctuation like .,
        name = re.sub(r"[.,]", "", name)
        # Remove common suffixes like inc, llc (as whole words)
        name = re.sub(r"\s+(inc|llc|ngo)$", "", name, flags=re.IGNORECASE)
        # Replace multiple spaces with single space and strip
        name = re.sub(r"\s+", " ", name).strip()
        return name if name else "Unknown"  # Return "Unknown" if empty after cleaning

    def extract_categorical_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:  # Return DataFrame for easier handling
        """
        Extract categorical features from a DataFrame. Ensures columns exist.
        Returns a DataFrame subset with only the expected categorical columns.
        """
        available_columns = [
            col for col in self.categorical_columns if col in df.columns
        ]

        if not available_columns:
            logger.warning("No categorical columns found in DataFrame for extraction.")
            # Return an empty DataFrame with expected columns if none are available
            return pd.DataFrame(columns=self.categorical_columns, index=df.index)

        # Return an explicit copy to avoid SettingWithCopyWarning later
        return df[available_columns].copy()

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the feature fusion model to the training data.
        Learns scaling parameters, top categories, and fits the encoder.
        """
        logger.info("Fitting FeatureFusion model...")
        df_processed = df.copy()

        # --- Numerical ---
        numerical_features = self.extract_numerical_features(df_processed)
        if numerical_features.shape[1] > 0:
            self.numerical_scaler.fit(numerical_features)
            logger.info(f"Fitted numerical scaler on columns: {self.numerical_columns}")
        else:
            logger.warning("No numerical features found to fit scaler.")

        # --- Categorical ---
        categorical_df = self.extract_categorical_features(df_processed)

        if not categorical_df.empty:
            # 1. Handle specific vendor mappings BEFORE standardization
            if "vendor" in categorical_df.columns:
                logger.info(
                    "Applying specific vendor mappings ('Unavailable', 'DOMESTIC AWARDEES (UNDISCLOSED)' -> 'Unknown')"
                )
                categorical_df["vendor"] = categorical_df["vendor"].replace(
                    ["Unavailable", "DOMESTIC AWARDEES (UNDISCLOSED)"], "Unknown"
                )
                # Remove rows where vendor became "Unknown" due to mapping, before calculating top %
                # We keep "Unknown" that might exist originally or result from standardization later
                original_unknown_mask = categorical_df["vendor"] != "Unknown"

            # 2. Standardize names
            logger.info("Standardizing categorical column names...")
            for col in categorical_df.columns:
                # Ensure column is string type before applying string methods
                categorical_df[col] = categorical_df[col].astype(str).fillna("Unknown")
                categorical_df[col] = categorical_df[col].apply(self._standardize_name)

            # 3. Determine and store top categories based on standardized names
            if "agency" in categorical_df.columns:
                agency_counts = categorical_df["agency"].value_counts()
                # Exclude "Unknown" from the count for determining top N
                agency_counts = agency_counts[agency_counts.index != "Unknown"]
                n_top_agencies = max(
                    1, int(len(agency_counts) * self.top_category_percentage)
                )
                self.top_agencies = list(agency_counts.index[:n_top_agencies])
                logger.info(
                    f"Determined top {len(self.top_agencies)} agencies (target {n_top_agencies})."
                )
                # Apply "Unknown" mapping based on standardized top list
                categorical_df["agency"] = categorical_df["agency"].apply(
                    lambda x: x if x in self.top_agencies else "Unknown"
                )

            if "vendor" in categorical_df.columns:
                # Use the mask to calculate counts only on originally valid vendors
                vendor_counts = categorical_df.loc[
                    original_unknown_mask, "vendor"
                ].value_counts()
                # Exclude "Unknown" from the count for determining top N
                vendor_counts = vendor_counts[vendor_counts.index != "Unknown"]
                n_top_vendors = max(
                    1, int(len(vendor_counts) * self.top_category_percentage)
                )
                self.top_vendors = list(vendor_counts.index[:n_top_vendors])
                logger.info(
                    f"Determined top {len(self.top_vendors)} vendors (target {n_top_vendors})."
                )
                # Apply "Unknown" mapping based on standardized top list
                categorical_df["vendor"] = categorical_df["vendor"].apply(
                    lambda x: x if x in self.top_vendors else "Unknown"
                )

            # 4. Fit encoder
            # Fill any remaining NaNs created during processing just before encoding
            categorical_data_to_encode = categorical_df.fillna("Unknown").values
            if categorical_data_to_encode.shape[1] > 0:
                logger.info(
                    f"Fitting OneHotEncoder on columns: {list(categorical_df.columns)}"
                )
                self.categorical_encoder.fit(categorical_data_to_encode)
                # Store categories for consistent transform
                self._encoder_categories = self.categorical_encoder.categories_
                logger.info("Fitted categorical encoder.")
                # Log the categories learned for each feature
                feature_names = self.categorical_encoder.get_feature_names_out(
                    categorical_df.columns
                )
                logger.debug(f"Encoder feature names out: {feature_names}")
                for i, col in enumerate(categorical_df.columns):
                    logger.debug(f"Categories for {col}: {self._encoder_categories[i]}")

            else:
                logger.warning("No categorical features found to fit encoder.")
        else:
            logger.warning("No categorical columns found in DataFrame to fit.")

        self.fitted = True
        logger.info("FeatureFusion fitting complete.")

    def transform(
        self, df: pd.DataFrame, text_features: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:  # Modified return type
        if not self.fitted:
            raise RuntimeError("Feature fusion model not fitted. Call fit() first.")

        logger.info("Transforming data using fitted FeatureFusion model...")
        df_processed = df.copy()
        feature_names = []  # Initialize list for feature names

        # --- Text Features ---
        # Assuming text_features are BERT embeddings (e.g., 768 dimensions)
        # Create generic names for text features
        num_text_features = 0
        processed_text_features = np.zeros((len(df_processed), 0))  # Default empty
        if text_features.ndim == 1:
            text_features = text_features.reshape(-1, 1)
        if text_features.size > 0 and text_features.shape[0] == len(df_processed):
            num_text_features = text_features.shape[1]
            feature_names.extend([f"text_{i}" for i in range(num_text_features)])
            processed_text_features = text_features
            logger.debug(f"Adding {num_text_features} text features.")
        elif text_features.size > 0:
            logger.warning(
                f"Text features shape {text_features.shape} mismatch with DataFrame length {len(df_processed)}. Skipping text features."
            )

        # --- Numerical ---
        numerical_features = self.extract_numerical_features(df_processed)
        scaled_numerical_features = np.zeros((len(df_processed), 0))  # Default empty
        if numerical_features.shape[1] > 0:
            try:
                scaled_numerical_features = self.numerical_scaler.transform(
                    numerical_features
                )
                # Add numerical feature names (use the ones defined in __init__)
                available_numerical_columns = [
                    col for col in self.numerical_columns if col in df_processed.columns
                ]
                feature_names.extend(
                    [f"num_{col}" for col in available_numerical_columns]
                )
                logger.info(
                    f"Transformed numerical features shape: {scaled_numerical_features.shape}"
                )
                logger.debug(
                    f"Added numerical feature names: {[f'num_{col}' for col in available_numerical_columns]}"
                )

            except Exception as e:
                logger.error(
                    f"Error transforming numerical features: {e}", exc_info=True
                )
                raise
        else:
            logger.info("No numerical features to transform.")

        # --- Categorical ---
        categorical_df = self.extract_categorical_features(df_processed)
        categorical_features_encoded = np.zeros((len(df_processed), 0))  # Default empty
        cat_feature_names = []

        if not categorical_df.empty and hasattr(
            self.categorical_encoder, "categories_"
        ):  # Check if encoder is fitted
            logger.info(
                "Standardizing and mapping categorical columns for transform..."
            )
            # Standardize names
            for col in categorical_df.columns:
                categorical_df[col] = categorical_df[col].astype(str).fillna("Unknown")
                categorical_df[col] = categorical_df[col].apply(self._standardize_name)

            # Map to 'Unknown' based on stored top lists
            if "agency" in categorical_df.columns and self.top_agencies:
                categorical_df["agency"] = categorical_df["agency"].apply(
                    lambda x: x if x in self.top_agencies else "Unknown"
                )
            if "vendor" in categorical_df.columns and self.top_vendors:
                categorical_df["vendor"] = categorical_df["vendor"].apply(
                    lambda x: x if x in self.top_vendors else "Unknown"
                )

            # Encode using fitted encoder
            categorical_data_to_encode = categorical_df.fillna("Unknown").values
            if categorical_data_to_encode.shape[1] > 0:
                try:
                    if self._encoder_categories is not None:
                        self.categorical_encoder.categories_ = self._encoder_categories

                    categorical_features_encoded = self.categorical_encoder.transform(
                        categorical_data_to_encode
                    )
                    # Get feature names from the encoder
                    cat_feature_names = list(
                        self.categorical_encoder.get_feature_names_out(
                            categorical_df.columns
                        )
                    )
                    feature_names.extend(
                        cat_feature_names
                    )  # Add categorical feature names
                    logger.info(
                        f"Transformed categorical features shape: {categorical_features_encoded.shape}"
                    )
                    logger.debug(
                        f"Added categorical feature names: {cat_feature_names}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error transforming categorical features: {e}", exc_info=True
                    )
                    raise
            else:
                logger.info("No categorical features to transform.")
        else:
            logger.info("No categorical columns found or encoder not fitted.")

        # --- Combine Features ---
        features_to_combine = []
        if processed_text_features.shape[1] > 0:  # Use processed_text_features
            features_to_combine.append(processed_text_features)
        if scaled_numerical_features.shape[1] > 0:
            features_to_combine.append(scaled_numerical_features)
        if categorical_features_encoded.shape[1] > 0:
            features_to_combine.append(categorical_features_encoded)

        if features_to_combine:
            try:
                num_rows = features_to_combine[0].shape[0]
                if not all(arr.shape[0] == num_rows for arr in features_to_combine):
                    shapes = [arr.shape for arr in features_to_combine]
                    logger.error(
                        f"Cannot hstack features: arrays have inconsistent number of rows. Shapes: {shapes}"
                    )
                    raise ValueError(
                        f"Inconsistent number of rows in features to combine. Shapes: {shapes}"
                    )

                combined_features = np.hstack(features_to_combine)
                logger.info(f"Combined features final shape: {combined_features.shape}")
                # Verify feature name count matches combined feature columns
                if len(feature_names) != combined_features.shape[1]:
                    logger.error(
                        f"Feature name count ({len(feature_names)}) does not match combined feature columns ({combined_features.shape[1]})!"
                    )
                    # Attempt to truncate or pad feature names? Or raise error? Raising error is safer.
                    raise ValueError(
                        "Mismatch between number of feature names and number of feature columns."
                    )

            except ValueError as e:
                logger.error(
                    f"Error during hstack or feature name verification: {e}",
                    exc_info=True,
                )
                raise
        else:
            logger.warning("No features available to combine.")
            combined_features = np.zeros((len(df_processed), 0))
            feature_names = (
                []
            )  # Ensure feature_names is empty if combined_features is empty

        # Return both combined features and their names
        return combined_features, feature_names  # Modified return value

    def fit_transform(self, df: pd.DataFrame, text_features: np.ndarray) -> np.ndarray:
        """
        Fit the feature fusion model and transform the data.
        Args:
            df: DataFrame containing data
            text_features: BERT text features
        Returns:
            Combined feature array
        """
        self.fit(df)
        return self.transform(df, text_features)

    def save_model(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving FeatureFusion model components to {output_dir}")

        # Save numerical scaler if it was fitted
        if hasattr(self.numerical_scaler, "scale_"):
            numerical_path = os.path.join(output_dir, "numerical_scaler.joblib")
            dump(self.numerical_scaler, numerical_path)
            logger.info(f"Saved numerical scaler to {numerical_path}")
        else:
            logger.warning("Numerical scaler not fitted, skipping save.")

        # Save categorical encoder if it was fitted
        if hasattr(self.categorical_encoder, "categories_"):
            categorical_path = os.path.join(output_dir, "categorical_encoder.joblib")
            # Save the encoder itself
            dump(self.categorical_encoder, categorical_path)
            logger.info(f"Saved categorical encoder to {categorical_path}")
            # Also save the explicit categories used during fit for robustness
            categories_path = os.path.join(output_dir, "encoder_categories.joblib")
            dump(self._encoder_categories, categories_path)
            logger.info(f"Saved encoder categories to {categories_path}")

        else:
            logger.warning("Categorical encoder not fitted, skipping save.")

        # Save top agencies list
        top_agencies_path = os.path.join(output_dir, "top_agencies.joblib")
        dump(self.top_agencies, top_agencies_path)
        logger.info(
            f"Saved top agencies list ({len(self.top_agencies)} items) to {top_agencies_path}"
        )

        # Save top vendors list
        top_vendors_path = os.path.join(output_dir, "top_vendors.joblib")  # Added
        dump(self.top_vendors, top_vendors_path)  # Added
        logger.info(
            f"Saved top vendors list ({len(self.top_vendors)} items) to {top_vendors_path}"
        )  # Added

    @classmethod
    def load_model(cls, model_dir: str) -> "FeatureFusion":
        # ... (keep docstring)
        logger.info(f"Loading FeatureFusion model components from {model_dir}")
        # Initialize feature fusion - pass default percentage, it's not saved/loaded
        feature_fusion = cls()

        # Load numerical scaler
        numerical_path = os.path.join(model_dir, "numerical_scaler.joblib")
        if os.path.exists(numerical_path):
            feature_fusion.numerical_scaler = load(numerical_path)
            logger.info(f"Loaded numerical scaler from {numerical_path}")
            # Infer numerical columns from scaler if possible (though usually fixed)
            if hasattr(feature_fusion.numerical_scaler, "n_features_in_"):
                # This doesn't give names, just count. Keep self.numerical_columns as defined in init.
                pass
        else:
            logger.warning(
                f"Numerical scaler file not found: {numerical_path}. Assuming no numerical scaling was fitted."
            )
            feature_fusion.numerical_scaler = StandardScaler()  # Initialize fresh

        # Load categorical encoder
        categorical_path = os.path.join(model_dir, "categorical_encoder.joblib")
        if os.path.exists(categorical_path):
            feature_fusion.categorical_encoder = load(categorical_path)
            logger.info(f"Loaded categorical encoder from {categorical_path}")
            # Infer categorical columns from encoder if possible
            if hasattr(feature_fusion.categorical_encoder, "feature_names_in_"):
                feature_fusion.categorical_columns = list(
                    feature_fusion.categorical_encoder.feature_names_in_
                )
                logger.info(
                    f"Inferred categorical columns from encoder: {feature_fusion.categorical_columns}"
                )
            # Load the explicit categories saved during fit for robustness
            categories_path = os.path.join(
                model_dir, "encoder_categories.joblib"
            )  # Use model_dir, not output_dir
            if os.path.exists(categories_path):
                feature_fusion._encoder_categories = load(categories_path)
                logger.info(
                    f"Loaded explicit encoder categories from {categories_path}"
                )
                # Optionally re-assign to the encoder instance if needed, depends on sklearn version behavior
                # feature_fusion.categorical_encoder.categories_ = feature_fusion._encoder_categories
            else:
                logger.warning(
                    f"Encoder categories file not found: {categories_path}. Relying on loaded encoder's state."
                )

        else:
            logger.warning(
                f"Categorical encoder file not found: {categorical_path}. Assuming no categorical encoding was fitted."
            )
            feature_fusion.categorical_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                feature_name_combiner="concat",
            )  # Initialize fresh

        # Load top agencies list
        top_agencies_path = os.path.join(model_dir, "top_agencies.joblib")
        if os.path.exists(top_agencies_path):
            feature_fusion.top_agencies = load(top_agencies_path)
            logger.info(
                f"Loaded top agencies list ({len(feature_fusion.top_agencies)} items) from {top_agencies_path}"
            )
        else:
            logger.warning(
                f"Top agencies file not found: {top_agencies_path}. Top agencies list will be empty."
            )
            feature_fusion.top_agencies = []

        # Load top vendors list
        top_vendors_path = os.path.join(model_dir, "top_vendors.joblib")  # Added
        if os.path.exists(top_vendors_path):  # Added
            feature_fusion.top_vendors = load(top_vendors_path)  # Added
            logger.info(
                f"Loaded top vendors list ({len(feature_fusion.top_vendors)} items) from {top_vendors_path}"
            )  # Added
        else:
            logger.warning(
                f"Top vendors file not found: {top_vendors_path}. Top vendors list will be empty."
            )  # Added
            feature_fusion.top_vendors = []  # Added

        # Mark as fitted if essential components were loaded
        # Check if both scaler and encoder have attributes indicating they were fitted
        scaler_fitted = hasattr(
            feature_fusion.numerical_scaler, "scale_"
        ) or not os.path.exists(
            numerical_path
        )  # Consider fitted if loaded or file didn't exist
        encoder_fitted = hasattr(
            feature_fusion.categorical_encoder, "categories_"
        ) or not os.path.exists(
            categorical_path
        )  # Consider fitted if loaded or file didn't exist

        if scaler_fitted and encoder_fitted:
            feature_fusion.fitted = True
            logger.info("FeatureFusion model marked as fitted after loading.")
        else:
            logger.warning(
                "FeatureFusion model loaded, but essential components appear unfitted."
            )
            feature_fusion.fitted = False

        return feature_fusion
