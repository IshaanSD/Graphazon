# scripts/preprocess_data.py

import argparse
from pathlib import Path

from src.data.loaders import load_esci_dataset
from src.data.features import build_product_kg, generate_kg_features


def main():
    parser = argparse.ArgumentParser(description="Preprocess ESCI dataset and build KG features")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to shopping_queries_dataset directory containing parquet files",
    )
    parser.add_argument(
        "--locale",
        type=str,
        default="us",
        choices=["us", "es", "jp"],
        help="Locale to process",
    )
    parser.add_argument(
        "--n-dev-queries",
        type=int,
        default=200,
        help="Number of dev queries to sample",
    )
    parser.add_argument(
        "--kg-attributes",
        nargs="+",
        default=["product_brand", "product_color"],
        help="Product columns to use as KG attributes",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    print("Loading ESCI dataset...")
    train_dataloader, df_dev = load_esci_dataset(
        dataset_path=str(dataset_path),
        locale=args.locale,
        n_dev_queries=args.n_dev_queries,
    )

    print(f"Loaded dev set with {len(df_dev)} rows")

    print("Building Knowledge Graph...")
    G = build_product_kg(df_dev, col_attributes=args.kg_attributes)

    print("Generating KG features...")
    df_kg_features = generate_kg_features(G, df_dev)

    print("Sample KG features:")
    print(df_kg_features.head())

    print("Done.")


if __name__ == "__main__":
    main()