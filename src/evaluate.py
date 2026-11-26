"""Ranking metrics for one user. 'recommended' is a ranked list of course ids,
'relevant' is the list of hidden (test) course ids."""
import numpy as np


def precision_at_k(recommended, relevant, k):
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant)


def f1_at_k(recommended, relevant, k):
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def ndcg_at_k(recommended, relevant, k):
    relevant = set(relevant)
    dcg = sum(1 / np.log2(i + 2) for i, c in enumerate(recommended[:k]) if c in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0