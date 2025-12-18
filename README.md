# Graphazon


[![PyPI](https://img.shields.io/pypi/v/graphazon)](https://pypi.org/project/graphazon/) 
[![ESCI Dataset](https://img.shields.io/badge/ESCI-Dataset-blue)](https://github.com/amazon-science/esci-data)


Graphazon is a research-focused project that combines **query understanding**, **knowledge graph construction**, and **learning-to-rank models** for Amazon-style product search. The system is designed to demonstrate key applied science competencies in **machine learning, NLP, knowledge extraction, and recommender systems**.

---

## Motivation

Product search is a complex problem that requires:

- Understanding user intent from ambiguous queries using **LLM reasoning (LangChain)**
- Mapping queries to relevant products, including substitutes and complements
- Leveraging structured and unstructured product information via a **Knowledge Graph**
- Improving ranking with graph-derived and semantic features

Graphazon addresses this by:

1. Using **LangChain** for **query understanding**, intent detection, and attribute extraction.
2. Building a **task-specific Knowledge Graph** from metadata and product text, including structured and LLM-assisted triples.
3. Incorporating **graph-derived features** into a **learning-to-rank model** trained on real-world data.

---

## Technologies & Features

- **LangChain**: LLM chains for query rewriting, intent detection, and attribute extraction.
- **Knowledge Graphs (KGs)**: Structured + LLM-assisted KG construction to capture product relationships (Exact, Substitute, Complement).
- **Embeddings & Features**: Combine KG, product metadata, and query embeddings for ranking.
- **Learning-to-Rank**: Neural / gradient-boosted ranking models trained on ESCI labels.
- **Pipelines**: Modular orchestration for end-to-end experiments.

---

## Dataset

Graphazon uses the **Amazon Shopping Queries Dataset (ESCI Benchmark)**:

- Provides queries, candidate products, and **ESCI labels** (Exact / Substitute / Complement / Irrelevant)
- Includes multilingual queries (English, Spanish, Japanese)
- Available as a **git submodule** in `data/shopping-queries-data-set/raw/esci-data/`

### Directory structure:
```
data/
└── shopping-queries-data-set/
├── raw/
│ └── esci-data/ # Git submodule from amazon-science
└── processed/ # cleaned queries, products, KG triples, splits
```

To initialize the dataset:

```bash
git submodule update --init --recursive
```

## Project Structure

```
Graphazon/
├── data/                  # datasets
├── src/                   # source code
│   ├── query_understanding/
│   ├── knowledge_graph/
│   ├── ranking/
│   ├── evaluation/
│   └── pipelines/
├── experiments/           
├── notebooks/             # EDA, visualization, shows usage
├── scripts/               # utility scripts
├── pixi.toml              # environment definition
└── README.md
```

* query_understanding/: LLM chains and prompts for query parsing, intent detection, attribute extraction

* knowledge_graph/: KG schema, structured and LLM-assisted extractors, graph feature generation

* ranking/: Learning-to-rank models using features from KG and embeddings

* evaluation/: Metrics, ablations, error analysis

* pipelines/: Orchestration scripts that run end-to-end processes

## Installation

Graphazon uses **pixi** for environment management.

```bash
pixi install
```

Then initialize the ESCI dataset:

```
git submodule update --init --recursive
```

## Running Experiments

```
# Build the knowledge graph
pixi run python -m src.pipelines.build_kg.py

# Run query understanding
pixi run python -m src.pipelines.run_query_understanding

# Train the learning-to-rank model
pixi run python -m src.pipelines.train_ranker.py

# Evaluate results
pixi run python -m src.pipelines.evaluate.py
```

## License

This project is for research and educational purposes. ESCI dataset is provided under its own license.