# src/data/loaders.py
# Portions adapted from Amazon ESCI dataset examples:
# https://github.com/amazon-science/esci-data

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

ESCI_LABEL2GAIN = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}


def load_esci_frames(
    dataset_path: str,
    locale: str = "us",
    n_dev_queries: int = 200,
    random_state: int = 42,
    use_small_version: bool = True,
):
    """
    Load + merge raw ESCI parquet and return (train, dev, test) dataframes.
    Splits dev from official train split by query_id.
    """
    dataset_path = Path(dataset_path)
    examples_fp = dataset_path / "shopping_queries_dataset_examples.parquet"
    products_fp = dataset_path / "shopping_queries_dataset_products.parquet"

    if not examples_fp.exists():
        raise FileNotFoundError(f"Missing: {examples_fp}")
    if not products_fp.exists():
        raise FileNotFoundError(f"Missing: {products_fp}")

    df_examples = pd.read_parquet(examples_fp)
    df_products = pd.read_parquet(products_fp)

    df = pd.merge(
        df_examples,
        df_products,
        how="left",
        on=["product_locale", "product_id"],
    )

    # Filters
    df = df[df["product_locale"] == locale]
    if use_small_version and "small_version" in df.columns:
        df = df[df["small_version"] == 1]

    # Gain label
    df["gain"] = df["esci_label"].map(ESCI_LABEL2GAIN).astype(float)

    # Official split
    if "split" not in df.columns:
        raise RuntimeError("Expected `split` column in dataset.")
    df_train_all = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()

    if len(df_train_all) == 0:
        raise RuntimeError("No rows found for split=='train' after filtering.")

    # Dev split by query_id
    qids = df_train_all["query_id"].unique()
    dev_size = min(1.0, n_dev_queries / max(1, len(qids)))
    qids_train, qids_dev = train_test_split(qids, test_size=dev_size, random_state=random_state)

    df_train = df_train_all[df_train_all["query_id"].isin(qids_train)].copy()
    df_dev = df_train_all[df_train_all["query_id"].isin(qids_dev)].copy()

    return df_train, df_dev, df_test


def load_esci_dataset(
    dataset_path: str,
    locale: str = "us",
    n_dev_queries: int = 200,
    random_state: int = 42,
    train_batch_size: int = 32,
    use_small_version: bool = True,
):
    """
    Backwards-compatible helper that returns:
      - train_dataloader (sentence-transformers InputExamples)
      - df_dev (dev dataframe)
    """
    df_train, df_dev, _df_test = load_esci_frames(
        dataset_path=dataset_path,
        locale=locale,
        n_dev_queries=n_dev_queries,
        random_state=random_state,
        use_small_version=use_small_version,
    )

    train_samples = [
        InputExample(texts=[row["query"], row["product_title"]], label=float(row["gain"]))
        for _, row in df_train.iterrows()
    ]
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
    return train_dataloader, df_dev