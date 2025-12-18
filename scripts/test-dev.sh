PYTHONPATH=. pixi run python scripts/preprocess_data.py \
  --dataset-path data/shopping-queries-dataset/raw/amazon-science/esci-data/shopping_queries_dataset \
  --out-dir data/shopping-queries-dataset/processed \
  --locale us \
  --use-small-version \
  --n-dev-queries 200 \
  --write-kg-features