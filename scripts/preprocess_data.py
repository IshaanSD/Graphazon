# scripts/preprocess_data.py

import argparse
from pathlib import Path

from src.data.loaders import load_esci_frames
from src.data.features import build_product_kg, generate_kg_features


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ESCI dataset and (optionally) build KG features")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to shopping_queries_dataset directory containing parquet files")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for processed artifacts (e.g. data/.../processed)")
    parser.add_argument("--locale", type=str, default="us", choices=["us", "es", "jp"])
    parser.add_argument("--n-dev-queries", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-small-version", action="store_true",
                        help="If set, filter to small_version==1")
    parser.add_argument("--kg-attributes", nargs="+", default=["product_brand", "product_color"])
    parser.add_argument("--write-kg-features", action="store_true")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    out_dir = Path(args.out_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    merged_dir = out_dir / "merged"
    kg_dir = out_dir / "kg"
    _ensure_dir(merged_dir)
    _ensure_dir(kg_dir)

    print("Loading ESCI frames via src.data.loaders...")
    df_train, df_dev, df_test = load_esci_frames(
        dataset_path=str(dataset_path),
        locale=args.locale,
        n_dev_queries=args.n_dev_queries,
        random_state=args.random_state,
        use_small_version=args.use_small_version,
    )

    train_out = merged_dir / f"train_{args.locale}.parquet"
    dev_out = merged_dir / f"dev_{args.locale}.parquet"
    test_out = merged_dir / f"test_{args.locale}.parquet"

    print(f"Writing: {train_out}")
    df_train.to_parquet(train_out, index=False)

    print(f"Writing: {dev_out}")
    df_dev.to_parquet(dev_out, index=False)

    if len(df_test) > 0:
        print(f"Writing: {test_out}")
        df_test.to_parquet(test_out, index=False)
    else:
        print("WARNING: test split is empty after filters; skipping test parquet write.")

    if args.write_kg_features:
        print("Building KG on dev split...")
        G = build_product_kg(df_dev, col_attributes=args.kg_attributes)
        df_kg = generate_kg_features(G, df_dev)

        kg_out = kg_dir / f"kg_features_dev_{args.locale}.parquet"
        print(f"Writing KG features: {kg_out}")
        df_kg.to_parquet(kg_out, index=False)

        print(df_kg.head())

    print("Done.")


if __name__ == "__main__":
    main()