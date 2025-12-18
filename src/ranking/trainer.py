from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ranking.metrics import summarize_ranking

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_features(processed_dir: Path, split: str, locale: str) -> pd.DataFrame:
    fp = processed_dir / "features" / f"{split}_features_{locale}.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing features parquet: {fp} (run build_features first)")
    return pd.read_parquet(fp)


def _make_group_sizes(df: pd.DataFrame, group_col: str) -> np.ndarray:
    # LightGBM expects group sizes aligned with the row order used for training.
    # We'll sort by group_col to ensure contiguous groups.
    return df.groupby(group_col, sort=False).size().to_numpy(dtype=np.int32)


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Keep numeric, model-ready features.
    This is intentionally conservative and will automatically include emb_* columns.
    """
    # Candidate feature prefixes you already have / will have
    allowed_prefixes = ("emb_", "kg_", "feat_")

    cols = []
    for c in df.columns:
        if c.startswith(allowed_prefixes):
            cols.append(c)

    # Optional: include some simple numeric columns if present
    for extra in ["gain"]:  # gain is label, exclude below
        if extra in cols:
            cols.remove(extra)

    # Ensure numeric only
    numeric_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return numeric_cols


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Train a LightGBM LambdaMART ranker and evaluate on dev.")
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--locale", type=str, default="us", choices=["us", "es", "jp"])
    parser.add_argument("--group-col", type=str, default="query_id")
    parser.add_argument("--label-col", type=str, default="gain")
    parser.add_argument("--model-out", type=str, default="models/lgbm_ranker.txt")
    parser.add_argument("--pred-out", type=str, default="outputs/dev_predictions.parquet")

    # LightGBM hyperparams
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=50)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    processed = Path(args.processed_dir)

    logger.info("Loading train/dev features...")
    df_train = _load_features(processed, "train", args.locale)
    df_dev = _load_features(processed, "dev", args.locale)

    # Sort so groups are contiguous for LightGBM
    df_train = df_train.sort_values(args.group_col).reset_index(drop=True)
    df_dev = df_dev.sort_values(args.group_col).reset_index(drop=True)

    feature_cols = _select_feature_columns(df_train)
    if not feature_cols:
        raise RuntimeError(
            "No numeric feature columns found. Expected columns starting with emb_, kg_, or feat_. "
            "Did you join embeddings into build_features?"
        )

    # Fill missing numeric features (common if some products missing embeddings)
    X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = df_train[args.label_col].astype(float)

    X_dev = df_dev[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_dev = df_dev[args.label_col].astype(float)

    group_train = _make_group_sizes(df_train, args.group_col)
    group_dev = _make_group_sizes(df_dev, args.group_col)

    logger.info("Training rows=%d, dev rows=%d, features=%d", len(df_train), len(df_dev), len(feature_cols))

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )

    ranker.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_dev, y_dev)],
        eval_group=[group_dev],
        eval_at=[args.ndcg_k],
        verbose=50,
    )

    # Evaluate on dev
    logger.info("Scoring dev...")
    dev_scores = ranker.predict(X_dev)
    df_pred = df_dev[[args.group_col, "query", "product_id", "product_title", args.label_col]].copy()
    df_pred["score"] = dev_scores

    metrics = summarize_ranking(df_pred, group_col=args.group_col, label_col=args.label_col, score_col="score")
    logger.info("Dev metrics: %s", metrics)

    # Save model + predictions
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(str(model_out))
    logger.info("Saved model -> %s", model_out)

    pred_out = Path(args.pred_out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_parquet(pred_out, index=False)
    logger.info("Saved dev predictions -> %s", pred_out)


if __name__ == "__main__":
    main()