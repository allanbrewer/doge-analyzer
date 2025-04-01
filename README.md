# DOGE Contract Anomaly Detector

A machine learning system for identifying government contracts that should be reviewed for potential cancelation using an Isolation Forest model with BERT-processed text features.

## Overview

This project provides a command-line tool and underlying Python modules to analyze government contracts or grants, identifying potentially anomalous entries based on similarity to a known baseline set (e.g., previously canceled contracts).

**Core Functionality:**

1.  **Data Ingestion:** Handles distinct sources for labeled training data (JSON) and unlabeled inference data (ZIP/CSV).
2.  **Feature Engineering:** Extracts meaningful features using:
    *   **BERT Transformer:** For understanding the semantics of text descriptions (`features/text.py`).
    *   **Numerical/Categorical Processing:** Standard scaling for numbers, one-hot encoding for categories (`features/fusion.py`).
3.  **Anomaly Detection:** Employs an **Isolation Forest** model (`models/isolation_forest.py`) to learn patterns from the labeled data and score unlabeled data based on similarity. Higher scores indicate greater similarity to the labeled set.
4.  **Pipeline Orchestration:** The `inference/pipeline.py` module encapsulates the steps of fitting the model and making predictions.
5.  **Command-Line Interface:** `main.py` provides a user-friendly CLI to configure and run the analysis.
6.  **Data Downloading:** A separate utility (`download/downloader.py`) fetches labeled data from a specified API.

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

### Downloading Data

The `doge_analyzer.download.downloader` module can be used to fetch labeled data (e.g., canceled contracts/grants) from the DOGE API. This data serves as the baseline for training the anomaly detection model.

```bash
# Activate the Poetry environment
poetry shell

# Example: Download the latest 5 pages of contract data
poetry run python -m doge_analyzer.download.downloader --award_type contracts --max_pages 5

# Example: Download the latest 10 pages of grant data
poetry run python -m doge_analyzer.download.downloader --award_type grants --max_pages 10
```
Downloaded data will be saved as timestamped JSON files in the corresponding `./data/{data_type}/` directory (e.g., `./data/contracts/`). The main pipeline will use the *most recent* JSON file in this directory for training by default.

### Running the Analysis Pipeline

The main pipeline script (`src/doge_analyzer/main.py`) orchestrates the model training and inference process. It uses the labeled data (JSON) to train the model and then applies the trained model to score the unlabeled data (ZIP/CSV).

```bash
# Activate the Poetry environment (if not already active)
poetry shell

# Run the pipeline for contracts:
# - Uses the latest labeled JSON from ./data/contracts/ for training
# - Uses unlabeled data from ./data/unlabeled/contracts/ for inference
# - Saves results to ./results/contracts/
poetry run python -m doge_analyzer.main \
  --data_type contracts \
  --labeled_data_dir ./data/contracts \
  --output_dir ./results

# Run the pipeline for grants using a specific labeled file:
# - Uses the specified labeled JSON file for training
# - Uses unlabeled data from ./data/unlabeled/grants/ for inference
# - Saves results to ./results/grants/
poetry run python -m doge_analyzer.main \
  --data_type grants \
  --labeled_data_file ./data/grants/specific_labeled_grants_20250101.json \
  --output_dir ./results
```
**Important:** Before running the pipeline for inference, ensure the unlabeled data directory (`./data/unlabeled/{data_type}/`) exists and contains the relevant ZIP or CSV files provided from the external source.

### Command-Line Arguments

- `--data_type`: (Required) Specifies the type of data ('contracts' or 'grants'). This determines:
    - The default directory for finding the latest labeled JSON data (`./data/{data_type}/`).
    - The required directory for unlabeled ZIP/CSV data (`./data/unlabeled/{data_type}/`).
    - The subdirectory within `--output_dir` where results and models are saved (`{output_dir}/{data_type}/`).
- `--labeled_data_dir`: Specifies the directory to search for the *latest* labeled data JSON file (used for training). If omitted, defaults to `./data/{data_type}/`. Ignored if `--labeled_data_file` is used.
- `--labeled_data_file`: Specifies the *exact* path to a labeled data JSON file to use for training. Overrides `--labeled_data_dir`.
- `--output_dir`: (Required) The base directory where results and models will be saved. A subdirectory named after the `data_type` will be created within this directory.
- `--bert_model`: Name of the BERT model to use (default: "bert-base-uncased")
- `--n_estimators`: Number of base estimators for Isolation Forest (default: 100)
- `--contamination`: Expected proportion of outliers in the data (default: 0.1)
- `--batch_size`: Batch size for BERT feature extraction (default: 8)
- `--threshold`: Custom threshold for anomaly detection (overrides model's fitted threshold).
- `--extract_dir`: Directory to extract zip files from the unlabeled data directory.
- `--sample_size`: Number of items (contracts/grants) to sample from the unlabeled data.
- `--department`: Filter to only include items from a specific department (if applicable).
- `--no_save_model`: Do not save the trained model to the output directory.
- `--no_visualize`: Do not generate visualizations.
- `--load_model_base_dir`: Base directory containing a pre-trained model (e.g., `./results`). The pipeline will look for the model in `{load_model_base_dir}/{data_type}/model`. Skips training if found.

### Example: Processing Contracts from a Specific Department

```bash
# Run analysis on contracts, using latest labeled data, filtering for 'Defense', saving to ./output/contracts
poetry run python -m doge_analyzer.main \
  --data_type contracts \
  --labeled_data_dir ./data/contracts \
  --output_dir ./output \
  --department "Defense" \
  --sample_size 1000

# Load a pre-trained grants model and run inference using a specific labeled file for context (if needed by model structure)
poetry run python -m doge_analyzer.main \
  --data_type grants \
  --labeled_data_file ./data/grants/specific_labeled_grants_20250101.json \
  --output_dir ./inference_results \
  --load_model_base_dir ./results \
  --no_visualize
```

## Project Structure

```
doge-analyzer/
├── data/                      # Root directory for input data
│   ├── contracts/             # Labeled contract data (downloaded JSONs used for training)
│   ├── grants/                # Labeled grant data (downloaded JSONs used for training)
│   └── unlabeled/             # Unlabeled data provided externally (used for inference)
│       ├── contracts/         # Unlabeled contract data (ZIP/CSV format)
│       └── grants/            # Unlabeled grant data (ZIP/CSV format)
├── results/                   # Root directory for output (configurable via --output_dir)
│   ├── contracts/             # Output for contract analysis runs
│   │   ├── model/             # Saved trained model components (detector, feature fusion)
│   │   └── visualizations/    # Generated plots (anomaly distribution, etc.)
│   └── grants/                # Output for grant analysis runs
│       ├── model/
│       └── visualizations/
├── src/                       # Source code directory
│   └── doge_analyzer/         # Main Python package
│       ├── __init__.py        # Makes 'doge_analyzer' a package
│       ├── data/              # Modules for data loading and preprocessing
│       │   ├── __init__.py
│       │   ├── load.py        # Functions to load labeled (JSON) and unlabeled (ZIP/CSV) data
│       │   └── preprocess.py  # Functions for cleaning and preparing data
│       ├── download/          # Module for fetching data from the DOGE API
│       │   ├── __init__.py
│       │   └── downloader.py  # Script to download and save labeled data as JSON
│       ├── features/          # Modules for feature engineering
│       │   ├── __init__.py
│       │   ├── text.py        # BERT feature extraction
│       │   ├── numerical.py   # (If specific numerical processing needed beyond scaling)
│       │   └── fusion.py      # Combines text, numerical, categorical features; handles scaling/encoding
│       ├── inference/         # Module defining the end-to-end analysis pipeline
│       │   ├── __init__.py
│       │   └── pipeline.py    # Contains ContractAnomalyPipeline class and run_pipeline function
│       ├── models/            # Module containing the anomaly detection model implementation
│       │   ├── __init__.py
│       │   └── isolation_forest.py # Implementation of ContractAnomalyDetector using Isolation Forest
│       ├── utils/             # Utility modules
│       │   ├── __init__.py
│       │   └── visualization.py # Functions for generating plots
│       └── main.py            # Main CLI script - entry point for running the analysis
├── tests/                     # Directory for automated tests
│   └── ...
├── .gitignore                 # Specifies intentionally untracked files
├── LICENSE                    # Project license file
├── poetry.lock                # Exact dependencies used
├── pyproject.toml             # Project metadata and dependencies (for Poetry)
└── README.md                  # This documentation file
```

## Data Format

### Labeled Data (Training Data)

The labeled data (used for training the anomaly detection model) is expected to be in JSON format, typically downloaded using the `downloader.py` script into the `./data/contracts/` or `./data/grants/` directories. The pipeline uses the latest JSON file found in the relevant directory (or a specific file if `--labeled_data_file` is used). The structure should be similar to:

```json
{
  "data": [
    {
      "piid": "...",
      "agency": "...",
      "vendor": "...",
      "value": ...,
      "description": "...",
      // ... other relevant fields used for training
    },
    // ... more records
  ]
}
```

### Unlabeled Data (Inference Data)

The unlabeled data (on which the trained model makes predictions) is expected to be located in the `./data/unlabeled/contracts/` or `./data/unlabeled/grants/` directories. This data typically comes from an external source and should be in **ZIP or CSV format**. The pipeline processes all compatible files found within the specified directory. The columns should align with the features expected by the model (e.g., `piid`, `agency`, `value`, `description`).

## Workflow & Component Interaction

This section details the typical execution flow when running the main analysis pipeline via `python -m doge_analyzer.main ...`.

1.  **Initialization (`main.py`)**:
    *   Parses command-line arguments (`argparse`).
    *   Determines the `data_type` (contracts/grants).
    *   Resolves the path to the labeled data JSON file (using `--labeled_data_file` or finding the latest in `--labeled_data_dir` / `./data/{data_type}/`).
    *   Identifies the unlabeled data directory (`./data/unlabeled/{data_type}/`).
    *   Constructs output paths (`{output_dir}/{data_type}/`).

2.  **Model Loading or Training (`main.py` -> `inference/pipeline.py`)**:
    *   **If `--load_model_base_dir` is provided:**
        *   Attempts to load a pre-trained `ContractAnomalyPipeline` instance using `ContractAnomalyPipeline.load_pipeline()` from the specified `{load_model_base_dir}/{data_type}/model` directory. This loads the saved Isolation Forest (`anomaly_detector.joblib`) and feature transformations (`feature_fusion/`).
    *   **If no model is loaded (or loading fails):**
        *   Instantiates a new `ContractAnomalyPipeline`.
        *   Calls the `pipeline.fit(labeled_data_path, ...)` method.

3.  **Pipeline Fitting (`inference/pipeline.py` -> `data/`, `features/`, `models/`)**:
    *   `pipeline.fit()` orchestrates the training process:
        *   Loads labeled data using `load_labeled_data()` (`data/load.py`).
        *   Preprocesses data using `preprocess_labeled_data()` (`data/preprocess.py`).
        *   Initializes `BertFeatureExtractor` (`features/text.py`) and extracts text embeddings.
        *   Initializes `FeatureFusion` (`features/fusion.py`), fits it to the labeled data (learning scaling parameters, top categories, etc.), and transforms the data, combining text, numerical, and categorical features.
        *   Initializes `ContractAnomalyDetector` (`models/isolation_forest.py`) and fits the Isolation Forest model to the combined features, calculating an anomaly threshold based on the scores of the training data.
    *   If `save_model` is true (default), `pipeline.save_pipeline()` is called to save the trained `ContractAnomalyDetector` and `FeatureFusion` components to the `{output_dir}/{data_type}/model/` directory.

4.  **Prediction (`main.py` -> `inference/pipeline.py` -> `data/`, `features/`, `models/`)**:
    *   Calls `pipeline.predict(unlabeled_data_paths, ...)` method, passing the *directory path* for unlabeled data.
    *   `pipeline.predict()` orchestrates the inference process:
        *   Loads unlabeled data using `load_multiple_unlabeled_files()` (`data/load.py`), which finds and processes ZIP/CSV files within the provided directory.
        *   Preprocesses data using `preprocess_unlabeled_data()` (`data/preprocess.py`).
        *   Extracts text features using the already initialized `BertFeatureExtractor`.
        *   Transforms the data using the *fitted* `FeatureFusion` instance (applying learned scaling/encoding).
        *   Uses the *fitted* `ContractAnomalyDetector` to get anomaly scores (`decision_function()`) and predictions (`predict_with_threshold()`) for the unlabeled data.
        *   Combines predictions with original data and saves results to a CSV file in `{output_dir}/{data_type}/`.

5.  **Visualization (`main.py` -> `utils/visualization.py`)**:
    *   If `--no_visualize` is not set, calls plotting functions from `utils/visualization.py` using the results DataFrame.
    *   Saves plots to the `{output_dir}/{data_type}/visualizations/` directory.

**Separate Downloader Workflow (`download/downloader.py`)**:

*   This script runs independently via `python -m doge_analyzer.download.downloader ...`.
*   It takes arguments like `--award_type` and `--max_pages`.
*   It calls the DOGE API, fetches data page by page.
*   Saves the combined results as a timestamped JSON file in the appropriate `./data/{award_type}/` directory. This JSON file then becomes a candidate for the *latest labeled data file* used by `main.py`.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
