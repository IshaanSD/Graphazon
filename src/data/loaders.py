# src/data/loaders.py
# Portions adapted from Amazon ESCI dataset examples:
# https://github.com/amazon-science/esci-data

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

def load_esci_dataset(dataset_path: str, locale="us", n_dev_queries=200, random_state=42, train_batch_size=32):
    """
    Loads ESCI Shopping Queries dataset, filters by locale, splits train/dev,
    and prepares DataLoader objects for training.
    
    Returns:
        train_dataloader: DataLoader for training
        df_dev: Pandas DataFrame for development/evaluation
    """
    col_query = "query"
    col_query_id = "query_id"
    col_product_id = "product_id" 
    col_product_title = "product_title"
    col_product_locale = "product_locale"
    col_esci_label = "esci_label" 
    col_small_version = "small_version"
    col_split = "split"
    col_gain = 'gain'
    
    esci_label2gain = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}
    
    # Load raw parquet files
    df_examples = pd.read_parquet(os.path.join(dataset_path, 'shopping_queries_dataset_examples.parquet'))
    df_products = pd.read_parquet(os.path.join(dataset_path, 'shopping_queries_dataset_products.parquet'))

    # Merge examples with product info
    df = pd.merge(df_examples, df_products, how='left',
                  left_on=[col_product_locale, col_product_id],
                  right_on=[col_product_locale, col_product_id])
    
    # Filter by small version, train split, and locale
    df = df[df[col_small_version] == 1]
    df = df[df[col_split] == "train"]
    df = df[df[col_product_locale] == locale]
    
    # Compute gain from ESCI labels
    df[col_gain] = df[col_esci_label].apply(lambda l: esci_label2gain[l])
    
    # Train/dev split
    list_query_id = df[col_query_id].unique()
    dev_size = n_dev_queries / len(list_query_id)
    list_query_id_train, list_query_id_dev = train_test_split(list_query_id, test_size=dev_size, random_state=random_state)
    
    df_train = df[df[col_query_id].isin(list_query_id_train)]
    df_dev = df[df[col_query_id].isin(list_query_id_dev)]
    
    # Prepare DataLoader for sentence-transformers
    train_samples = [InputExample(texts=[row[col_query], row[col_product_title]], label=float(row[col_gain]))
                     for _, row in df_train.iterrows()]
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
    
    return train_dataloader, df_dev