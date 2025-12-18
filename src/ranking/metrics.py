from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _dcg(rels: np.ndarray) -> float:
    # DCG with log2 discount (positions start at 1)
    rels = rels.astype(float)
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def ndcg_at_k(
    df: pd.DataFrame,
    *,
    group_col: str = "query_id",
    label_col: str = "gain",
    score_col: str = "score",
    k: int = 10,
) -> float:
    """
    Compute mean NDCG@k across groups.
    Assumes higher label is better.
    """
    ndcgs = []
    for _, g in df.groupby(group_col, sort=False):
        g_sorted = g.sort_values(score_col, ascending=False)
        rels = g_sorted[label_col].to_numpy()[:k]
        dcg = _dcg(rels)

        ideal = np.sort(g[label_col].to_numpy())[::-1][:k]
        idcg = _dcg(ideal)
        ndcgs.append(0.0 if idcg == 0.0 else dcg / idcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def mrr_at_k(
    df: pd.DataFrame,
    *,
    group_col: str = "query_id",
    label_col: str = "gain",
    score_col: str = "score",
    k: int = 10,
    relevant_threshold: float = 1.0,
) -> float:
    """
    Mean Reciprocal Rank@k where a doc is 'relevant' if label >= relevant_threshold.
    Default assumes gain mapping where E=3,S=2,C=1,I=0 (so C counts as relevant).
    """
    rr = []
    for _, g in df.groupby(group_col, sort=False):
        g_sorted = g.sort_values(score_col, ascending=False).head(k)
        rel = (g_sorted[label_col].to_numpy() >= relevant_threshold)
        if not rel.any():
            rr.append(0.0)
        else:
            rank = int(np.argmax(rel)) + 1  # 1-indexed
            rr.append(1.0 / rank)
    return float(np.mean(rr)) if rr else 0.0


def summarize_ranking(
    df: pd.DataFrame,
    *,
    group_col: str = "query_id",
    label_col: str = "gain",
    score_col: str = "score",
) -> Dict[str, float]:
    return {
        "ndcg@10": ndcg_at_k(df, group_col=group_col, label_col=label_col, score_col=score_col, k=10),
        "ndcg@20": ndcg_at_k(df, group_col=group_col, label_col=label_col, score_col=score_col, k=20),
        "mrr@10": mrr_at_k(df, group_col=group_col, label_col=label_col, score_col=score_col, k=10),
    }