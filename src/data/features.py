# src/data/features.py
"""
Generate Knowledge Graph features for products and queries
"""

import networkx as nx
import pandas as pd

def build_product_kg(df_products: pd.DataFrame, col_product_id="product_id", col_attributes=None):
    """
    Build a simple product Knowledge Graph using attributes.
    
    Args:
        df_products: DataFrame containing product metadata
        col_product_id: column name for product ids
        col_attributes: list of columns to use as nodes/edges in KG
    
    Returns:
        nx.Graph
    """
    if col_attributes is None:
        col_attributes = ["product_brand", "product_color"]
    
    G = nx.Graph()
    
    for _, row in df_products.iterrows():
        pid = row[col_product_id]
        G.add_node(pid, type="product")
        
        for attr in col_attributes:
            val = row.get(attr)
            if pd.notna(val):
                attr_node = f"{attr}:{val}"
                G.add_node(attr_node, type="attribute")
                G.add_edge(pid, attr_node, relation=attr)
    
    return G

def generate_kg_features(G: nx.Graph, df_queries: pd.DataFrame, col_product_id="product_id"):
    """
    Generate simple graph features for each product.
    
    Features:
        - degree centrality
        - number of attributes connected
    
    Returns:
        df_features: DataFrame mapping product_id -> feature vector
    """
    degree_centrality = nx.degree_centrality(G)
    
    data = []
    for pid in df_queries[col_product_id].unique():
        node_degree = degree_centrality.get(pid, 0)
        num_attributes = len([n for n in G.neighbors(pid) if G.nodes[n]['type'] == 'attribute'])
        data.append({"product_id": pid, "kg_degree": node_degree, "kg_num_attributes": num_attributes})
    
    df_features = pd.DataFrame(data)
    return df_features