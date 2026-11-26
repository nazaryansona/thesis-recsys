"""Evaluate all methods on the saved test set and store one results table."""
import pandas as pd

from config import DATA_PROCESSED, RESULTS, K_VALUES
from src.evaluate import precision_at_k, recall_at_k, f1_at_k, ndcg_at_k
from src.models import Popularity, ContentBased, ItemCF, MatrixFactorization

METRICS = {"Precision": precision_at_k, "Recall": recall_at_k,
           "F1": f1_at_k, "NDCG": ndcg_at_k}


def evaluate(model, test, ks=K_VALUES):
    truth = test.groupby("user")["course"].apply(list).to_dict()
    recs = model.recommend_batch(list(truth), max(ks))
    totals = {(m, k): 0.0 for m in METRICS for k in ks}
    for user, relevant in truth.items():
        for m, fn in METRICS.items():
            for k in ks:
                totals[(m, k)] += fn(recs[user], relevant, k)
    n = len(truth)
    row = {f"{m}@{k}": round(v / n, 4) for (m, k), v in totals.items()}
    for k in ks:  # catalog coverage: share of courses recommended to anyone
        shown = {c for r in recs.values() for c in r[:k]}
        row[f"Coverage@{k}"] = round(len(shown) / len(model.courses), 4)
    return row


def main():
    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    concepts = pd.read_csv(DATA_PROCESSED / "course_concepts.csv")
    models = {
        "Popularity": Popularity(),
        "Content-based": ContentBased(),
        "Item-based CF": ItemCF(),
        "Matrix factorization": MatrixFactorization(factors=50),
    }
    results = {}
    for name, model in models.items():
        print("running", name, "...")
        results[name] = evaluate(model.fit(train, concepts), test)
    table = pd.DataFrame(results).T
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "results.csv")
    print(table.T.to_string())


if __name__ == "__main__":
    main()