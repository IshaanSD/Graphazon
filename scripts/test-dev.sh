#!/usr/bin/env bash
set -e

echo "Running Graphazon environment test..."

PYTHONPATH=. pixi run python scripts/preprocess_data.py \
  --dataset-path data/shopping-queries-dataset/raw/amazon-science/esci-data/shopping_queries_dataset \
  --locale us \
  --n-dev-queries 5

echo "Environment test completed successfully."