from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pickle
import pandas as pd

from src.knowledge_graph.graph_store import build_product_attribute_kg
from src.knowledge_graph.embeddings import (
    Node2VecConfig,
    embeddings_to_frame,
    filter_product_embeddings,
    train_node2vec_embeddings,
)

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Build product-attribute KG and train Node2Vec embeddings.")
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--locale", type=str, default="us", choices=["us", "es", "jp"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "dev", "test"])
    parser.add_argument("--attributes", nargs="+", default=["product_brand", "product_color"])
    parser.add_argument("--dims", type=int, default=64)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    merged_fp = processed / "merged" / f"{args.split}_{args.locale}.parquet"
    if not merged_fp.exists():
        raise FileNotFoundError(f"Missing merged parquet: {merged_fp} (run scripts/preprocess_data.py first)")

    kg_dir = processed / "kg"
    kg_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading pairs: %s", merged_fp)
    df = pd.read_parquet(merged_fp)

    # Build KG
    logger.info("Building KG (attributes=%s)", args.attributes)
    G = build_product_attribute_kg(
        df,
        product_id_col="product_id",
        brand_col="product_brand",
        color_col="product_color",
        type_col=None,
        allowed_attributes=set(args.attributes),
    )

    logger.info("KG built: nodes=%d edges=%d", G.number_of_nodes(), G.number_of_edges())

    # Save graph artifact
    graph_fp = kg_dir / f"graph_{args.split}_{args.locale}.gpickle"
    with open(graph_fp, "wb") as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

    logger.info("Wrote graph -> %s", graph_fp)

    # Train embeddings
    cfg = Node2VecConfig(
        dimensions=args.dims,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        p=args.p,
        q=args.q,
        window=args.window,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )

    logger.info(
        "Training Node2Vec (dims=%d walk_length=%d num_walks=%d p=%.2f q=%.2f window=%d epochs=%d)",
        cfg.dimensions,
        cfg.walk_length,
        cfg.num_walks,
        cfg.p,
        cfg.q,
        cfg.window,
        cfg.epochs,
    )

    emb = train_node2vec_embeddings(G, cfg=cfg)

    emb_df = embeddings_to_frame(emb)
    nodes_fp = kg_dir / f"node_embeddings_{args.split}_{args.locale}.parquet"
    emb_df.to_parquet(nodes_fp, index=False)
    logger.info("Wrote node embeddings -> %s", nodes_fp)

    prod_df = filter_product_embeddings(emb_df, product_prefix="p:")
    prod_fp = kg_dir / f"product_embeddings_{args.split}_{args.locale}.parquet"
    prod_df.to_parquet(prod_fp, index=False)
    logger.info("Wrote product embeddings -> %s", prod_fp)

    logger.info("Done.")


if __name__ == "__main__":
    main()