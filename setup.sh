#!/bin/bash

# Setup script for doge-analyzer

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Please install it first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
poetry install

# Run tests in poetry shell
echo "Running tests..."
poetry run pytest -v

echo "Setup complete!"
echo "You can now use the package with 'poetry run doge-analyzer'"
echo "For example:"
echo "poetry run doge-analyzer --labeled_data data/contracts/doge_contracts_20250323222302.json --unlabeled_data data/unlabeled --output_dir results"
