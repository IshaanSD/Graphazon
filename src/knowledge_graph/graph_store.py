from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class KGNodePrefix:
    product: str = "p:"
    brand: str = "brand:"
    color: str = "color:"
    type: str = "type:"


def _norm_str(x: object) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def build_product_attribute_kg(
    df: pd.DataFrame,
    *,
    product_id_col: str = "product_id",
    brand_col: str = "product_brand",
    color_col: str = "product_color",
    type_col: Optional[str] = None,
    allowed_attributes: Optional[Iterable[str]] = None,
) -> nx.Graph:
    """
    Build an undirected KG with product nodes connected to attribute nodes.
    Nodes are namespaced with prefixes:
      - p:<product_id>
      - brand:<brand>
      - color:<color>
      - type:<type>  (optional)
    """
    prefixes = KGNodePrefix()
    G = nx.Graph()

    if allowed_attributes is not None:
        allowed_attributes = set(allowed_attributes)

    def _add_edge(p_node: str, a_node: str, rel: str) -> None:
        if not G.has_node(p_node):
            G.add_node(p_node, kind="product")
        if not G.has_node(a_node):
            G.add_node(a_node, kind="attribute", rel=rel)
        if not G.has_edge(p_node, a_node):
            G.add_edge(p_node, a_node, rel=rel)

    # Only keep relevant columns to reduce memory
    cols: List[str] = [product_id_col, brand_col, color_col]
    if type_col:
        cols.append(type_col)
    df_small = df[cols].copy()

    for row in df_small.itertuples(index=False):
        product_id = _norm_str(getattr(row, product_id_col))
        if not product_id:
            continue

        p_node = prefixes.product + product_id

        # Brand
        if (allowed_attributes is None) or (brand_col in allowed_attributes):
            brand = _norm_str(getattr(row, brand_col))
            if brand:
                a_node = prefixes.brand + brand.lower()
                _add_edge(p_node, a_node, rel="brand")

        # Color
        if (allowed_attributes is None) or (color_col in allowed_attributes):
            color = _norm_str(getattr(row, color_col))
            if color:
                a_node = prefixes.color + color.lower()
                _add_edge(p_node, a_node, rel="color")

        # Type (optional)
        if type_col and ((allowed_attributes is None) or (type_col in allowed_attributes)):
            ptype = _norm_str(getattr(row, type_col))
            if ptype:
                a_node = prefixes.type + ptype.lower()
                _add_edge(p_node, a_node, rel="type")

    return G