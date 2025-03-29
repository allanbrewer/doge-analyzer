"""
Basic tests for the contract anomaly detection pipeline.
"""

import os
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np

from doge_analyzer.features.text import BertFeatureExtractor
from doge_analyzer.features.fusion import FeatureFusion
from doge_analyzer.models.isolation_forest import ContractAnomalyDetector


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the pipeline components."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_bert_feature_extractor(self):
        """Test BERT feature extractor."""
        # Create a simple DataFrame with contract descriptions
        df = pd.DataFrame(
            {
                "description": [
                    "Office supplies for government agency",
                    "IT consulting services for cybersecurity",
                    "Construction of new government building",
                ],
                "clean_description": [
                    "office supplies for government agency",
                    "it consulting services for cybersecurity",
                    "construction of new government building",
                ],
            }
        )

        # Initialize feature extractor
        extractor = BertFeatureExtractor(model_name="bert-base-uncased")

        # Extract features
        features = extractor.extract_features_from_df(
            df, "clean_description", batch_size=2
        )

        # Check that features have the expected shape
        self.assertEqual(features.shape[0], len(df))
        self.assertEqual(features.shape[1], 768)  # BERT base has 768 dimensions

    def test_feature_fusion(self):
        """Test feature fusion."""
        # Create a simple DataFrame with contract data
        df = pd.DataFrame(
            {
                "agency": ["Agency A", "Agency B", "Agency C"],
                "vendor": ["Vendor X", "Vendor Y", "Vendor Z"],
                "normalized_value": [100000.0, 250000.0, 500000.0],
                "value_per_word": [10000.0, 25000.0, 50000.0],
            }
        )

        # Create dummy text features
        text_features = np.random.rand(3, 10)  # 3 samples, 10 features

        # Initialize feature fusion
        fusion = FeatureFusion()

        # Combine features
        combined_features = fusion.fit_transform(df, text_features)

        # Check that combined features have the expected shape
        self.assertEqual(combined_features.shape[0], len(df))
        self.assertTrue(combined_features.shape[1] > text_features.shape[1])

    def test_anomaly_detector(self):
        """Test anomaly detector."""
        # Create dummy features
        features = np.random.rand(100, 20)  # 100 samples, 20 features

        # Initialize anomaly detector
        detector = ContractAnomalyDetector(
            n_estimators=10,  # Use fewer estimators for testing
            contamination=0.1,
            random_state=42,
        )

        # Fit detector
        detector.fit(features)

        # Make predictions
        predictions = detector.predict(features)

        # Check that predictions have the expected shape and values
        self.assertEqual(predictions.shape[0], features.shape[0])
        self.assertTrue(np.all(np.isin(predictions, [-1, 1])))

        # Check that approximately 10% of samples are predicted as anomalies
        anomaly_rate = np.sum(predictions == -1) / len(predictions)
        self.assertAlmostEqual(anomaly_rate, 0.1, delta=0.05)

        # Test saving and loading
        model_path = os.path.join(self.test_dir, "model.joblib")
        detector.save_model(model_path)
        loaded_detector = ContractAnomalyDetector.load_model(model_path)

        # Check that loaded model makes the same predictions
        loaded_predictions = loaded_detector.predict(features)
        self.assertTrue(np.array_equal(predictions, loaded_predictions))


if __name__ == "__main__":
    unittest.main()
