from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# def main() -> None:
if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Join split pairs + query understanding + (optional) KG features into a single split features table."
    )
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--locale", type=str, default="us", choices=["us", "es", "jp"])
    parser.add_argument("--include-kg", action="store_true", help="Join KG features if present.")
    args = parser.parse_args()

    processed = Path(args.processed_dir)

    pairs_fp = processed / "merged" / f"{args.split}_{args.locale}.parquet"
    qu_fp = processed / "queries" / f"query_understanding_{args.split}_{args.locale}.parquet"
    kg_fp = processed / "kg" / f"kg_features_{args.split}_{args.locale}.parquet" 

    if not pairs_fp.exists():
        raise FileNotFoundError(f"Missing split parquet: {pairs_fp} (run scripts/preprocess_data.py first)")
    if not qu_fp.exists():
        raise FileNotFoundError(f"Missing query understanding parquet: {qu_fp} (run run_query_understanding first)")

    logger.info("Reading split pairs: %s", pairs_fp)
    df_split = pd.read_parquet(pairs_fp)

    logger.info("Reading query understanding: %s", qu_fp)
    df_qu = pd.read_parquet(qu_fp)
    df_qu['query_id'] = df_qu['query_id'].astype(df_split['query_id'].dtype)

    # many split rows per query_id, one query-understanding row per query_id
    df = df_split.merge(df_qu, on="query_id", how="left", validate="many_to_one")

    if args.include_kg:
        if not kg_fp.exists():
            logger.warning("KG features not found (%s). Continuing without KG.", kg_fp)
        else:
            logger.info("Reading KG features: %s", kg_fp)
            df_kg = pd.read_parquet(kg_fp)
            if "product_id" not in df_kg.columns:
                raise RuntimeError(f"KG features file missing 'product_id' column: {kg_fp}")
            df = df.merge(df_kg, on="product_id", how="left", validate="many_to_one")

    out_dir = processed / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{args.split}_features_{args.locale}.parquet"

    logger.info("Writing: %s", out_fp)
    df.to_parquet(out_fp, index=False)

    logger.info("Done. rows=%d cols=%d saved at %s", len(df), df.shape[1], out_fp)

# if __name__ == "__main__":
#     main()