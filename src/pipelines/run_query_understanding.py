from src.data.loaders import load_esci_dataset
from src.data.features import build_product_kg, generate_kg_features

# 1. Load dataset
train_dataloader, df_dev = load_esci_dataset("data/shopping-queries-dataset/raw/amazon-science/esci-data/shopping_queries_dataset")

# 2. Build Knowledge Graph
G = build_product_kg(df_dev, col_attributes=["product_brand", "product_color"])

# 3. Generate KG features
df_kg_features = generate_kg_features(G, df_dev)

print(df_kg_features.head())