# Personalized recommendation methods for e-learning (MOOCCube)

Thesis I project: development and comparative evaluation of four recommendation
methods (popularity baseline, content-based, item-based collaborative filtering,
matrix factorization) on the MOOCCube dataset.

## Setup

1. Python 3.12 or newer (version used: 3.15.2).
2. Create and activate a virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

3. Install dependencies:

   pip install -r requirements.txt

## Data

Download MOOCCube (Yu et al., ACL 2020) from http://moocdata.cn/data/MOOCCube
and unzip it into `data/MOOCCube`. The raw data is not included in this archive
because of its size.

## Run

Run the steps in this order from the project root:

    python -m src.data_prep    # clean raw data -> data/processed/interactions.csv, course_concepts.csv
    python -m src.split        # leave-one-out split (seed 42) -> train.csv, test.csv
    python -m src.run_eval     # fit and evaluate all methods -> results/results.csv

Expected output of the data preparation: 469,631 interactions, 94,485 users,
678 courses.

## Project structure

    config.py        paths, random seed, K values
    src/data_prep.py data loading and cleaning
    src/split.py     leave-one-out train/test split
    src/models.py    Popularity, ContentBased, ItemCF, MatrixFactorization
    src/evaluate.py  Precision, Recall, F1, NDCG at K
    src/run_eval.py  runs all models and writes the results table
    notebooks/       data exploration
    results/         evaluation output

## Status

Work in progress (Thesis I). Implemented: data pipeline, four methods,
offline evaluation. Planned: tuning, cold-start evaluation, optional hybrid.
