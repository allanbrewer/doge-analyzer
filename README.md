# DOGE Contract Anomaly Detector

A machine learning system for identifying government contracts that should be reviewed for potential cancelation using an Isolation Forest model with BERT-processed text features.

## Overview

This project analyzes government contracts to identify anomalous ones that may be candidates for cancelation. It uses:

- **BERT Transformer** to extract features from contract descriptions
- **Isolation Forest** for unsupervised anomaly detection
- Feature fusion to combine text, numerical, and categorical features

The system is trained on a dataset of contracts that have already been canceled by the Department of Government Efficiency (DOGE), and can then be applied to other contracts to identify similar patterns.

## Features

- Text feature extraction using BERT
- Numerical and categorical feature processing
- Unsupervised anomaly detection using Isolation Forest
- Comprehensive visualization tools
- Batch processing of contract data
- Command-line interface for easy use

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/doge-analyzer.git
cd doge-analyzer

# Install dependencies using Poetry
poetry install
```

## Usage

### Basic Usage

```bash
# Activate the Poetry environment
poetry shell

# Run the pipeline
python -m doge_analyzer.main \
  --labeled_data data/contracts/doge_contracts_20250323222302.json \
  --unlabeled_data data/unlabeled \
  --output_dir results
```

### Command-Line Arguments

- `--labeled_data`: Path to the labeled data file (JSON)
- `--unlabeled_data`: Path to the unlabeled data directory or file
- `--output_dir`: Directory to save results and model
- `--bert_model`: Name of the BERT model to use (default: "bert-base-uncased")
- `--n_estimators`: Number of base estimators for Isolation Forest (default: 100)
- `--contamination`: Expected proportion of outliers in the data (default: 0.1)
- `--batch_size`: Batch size for BERT feature extraction (default: 8)
- `--threshold`: Custom threshold for anomaly detection
- `--extract_dir`: Directory to extract zip files
- `--sample_size`: Number of contracts to sample
- `--department`: Filter to only include files from a specific department
- `--no_save_model`: Do not save the trained model
- `--no_visualize`: Do not generate visualizations

### Example: Processing Contracts from a Specific Department

```bash
python -m doge_analyzer.main \
  --labeled_data data/contracts/doge_contracts_20250323222302.json \
  --unlabeled_data fraud-finder/contract_data \
  --output_dir results/defense \
  --department "defense" \
  --sample_size 1000
```

## Project Structure

```
doge-analyzer/
├── data/                      # Data directory
│   └── contracts/             # Contract data
├── src/                       # Source code
│   └── doge_analyzer/         # Main package
│       ├── data/              # Data loading and preprocessing
│       ├── features/          # Feature extraction and fusion
│       ├── models/            # Anomaly detection models
│       ├── inference/         # Inference pipeline
│       ├── utils/             # Utility functions
│       └── main.py            # Main script
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## Data Format

### Labeled Data (Canceled Contracts)

The labeled data should be in JSON format with the following structure:

```json
{
  "data": [
    {
      "piid": "47HAA024F0028",
      "agency": "General Services Administration",
      "vendor": "Vendor Name",
      "value": 13715913,
      "description": "Research Memberships",
      "fpds_status": "CHANGE ORDER",
      "fpds_link": "https://www.fpds.gov/...",
      "deleted_date": "1/31/2025 (contract expired)",
      "savings": 12508877
    },
    ...
  ]
}
```

### Unlabeled Data

The unlabeled data should be in CSV format (typically in ZIP files) with columns that can be mapped to the labeled data format.

## How It Works

1. **Data Loading**: Load labeled data (canceled contracts) and unlabeled data
2. **Preprocessing**: Clean and normalize the data
3. **Feature Extraction**:
   - Extract text features from contract descriptions using BERT
   - Process numerical features (contract value, etc.)
   - Encode categorical features (agency, vendor)
4. **Feature Fusion**: Combine all features
5. **Model Training**: Train an Isolation Forest model on the labeled data
6. **Inference**: Apply the model to unlabeled data to identify anomalous contracts
7. **Visualization**: Generate visualizations of the results

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
