from __future__ import annotations

from dataclasses import dataclass
from random import seed
from typing import Dict, Iterable, Optional

from networkx.generators import random_clustered

import numpy as np
import networkx as nx
import pandas as pd
from node2vec import Node2Vec


@dataclass(frozen=True)
class Node2VecConfig:
    dimensions: int = 64
    walk_length: int = 20
    num_walks: int = 10
    p: float = 1.0
    q: float = 1.0
    window: int = 10
    min_count: int = 1
    workers: int = 1
    epochs: int = 5
    seed: int = 42


def train_node2vec_embeddings(
    G: nx.Graph,
    *,
    cfg: Node2VecConfig,
) -> Dict[str, np.ndarray]:
    """
    Train Node2Vec embeddings for all nodes in the graph.
    Returns: {node_id: np.ndarray(dimensions)}
    """
    # node2vec package expects nodes to be str/int; ours are str already.
    n2v = Node2Vec(
        G,
        dimensions=cfg.dimensions,
        walk_length=cfg.walk_length,
        num_walks=cfg.num_walks,
        p=cfg.p,
        q=cfg.q,
        workers=cfg.workers,
        # TODO : check how to add seed withou dependency conflicts
    )

    w2v = n2v.fit(
        window=cfg.window,
        min_count=cfg.min_count,
        batch_words=128,
        epochs=cfg.epochs,
    )

    emb: Dict[str, np.ndarray] = {}
    for node in G.nodes():
        emb[str(node)] = w2v.wv[str(node)].astype(np.float32)

    return emb


def embeddings_to_frame(emb: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert embedding dict into a dataframe:
      node_id | emb_0 | emb_1 | ... | emb_{d-1}
    """
    if not emb:
        return pd.DataFrame(columns=["node_id"])

    any_vec = next(iter(emb.values()))
    d = int(any_vec.shape[0])

    rows = []
    for node_id, vec in emb.items():
        rows.append((node_id, *vec.tolist()))

    cols = ["node_id"] + [f"emb_{i}" for i in range(d)]
    return pd.DataFrame(rows, columns=cols)


def filter_product_embeddings(
    emb_df: pd.DataFrame,
    *,
    product_prefix: str = "p:",
) -> pd.DataFrame:
    """
    Keep only product nodes and add product_id column (without prefix).
    """
    df = emb_df[emb_df["node_id"].astype(str).str.startswith(product_prefix)].copy()
    df["product_id"] = df["node_id"].astype(str).str[len(product_prefix) :]
    return df