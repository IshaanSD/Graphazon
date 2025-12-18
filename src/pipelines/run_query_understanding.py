from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import os

from src.query_understanding.interface import understand_query

import logging
def _setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)

def main():
    _setup_logging()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Run LangChain query understanding on dev queries.")
    parser.add_argument("--processed-dir", type=str, required=True, help="Processed directory (e.g., data/.../processed)")
    parser.add_argument("--locale", type=str, default="us", choices=["us", "es", "jp"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-queries", type=int, default=0, help="0 means all dev queries; otherwise limit for testing.")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    pairs_fp = processed_dir / "merged" / f"{args.split}_{args.locale}.parquet"
    if not pairs_fp.exists():
        raise FileNotFoundError(f"Missing parquet: {pairs_fp}. Run scripts/preprocess_data.py first.")


    df_split = pd.read_parquet(pairs_fp)
    df_q = df_split[["query_id", "query"]].drop_duplicates().reset_index(drop=True)
    if args.max_queries and args.max_queries > 0:
        df_q = df_q.head(args.max_queries)

    outputs = []
    for row in tqdm(df_q.itertuples(index=False), total=len(df_q), desc="Query Understanding"):
        out = understand_query(
            query_id=str(row.query_id),
            query=str(row.query),
            locale=args.locale,
            model_name=args.model,
            temperature=args.temperature,
        )
        outputs.append(out.model_dump())

    out_dir = processed_dir / "queries"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fp = out_dir / f"query_understanding_{args.split}_{args.locale}.parquet"
    pd.DataFrame(outputs).to_parquet(out_fp, index=False)

    logger.info(f"Wrote: {out_fp}")
    logger.info(pd.DataFrame(outputs).head())

if __name__ == "__main__":
    main()