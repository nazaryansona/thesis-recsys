"""Recommendation methods. Every method has fit(train, concepts) and
recommend_batch(users, k). All of them read the SAME train file, never the
test file."""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

from config import SEED


class Recommender:
    """Shared plumbing: builds the user x course matrix from the training data."""

    def fit(self, train, concepts):
        self.concepts = concepts.drop_duplicates()
        self.courses = np.array(sorted(self.concepts["course"].unique()))
        self.cidx = {c: i for i, c in enumerate(self.courses)}
        self.users = np.array(sorted(train["user"].unique()))
        self.uidx = {u: i for i, u in enumerate(self.users)}
        rows = train["user"].map(self.uidx).to_numpy()
        cols = train["course"].map(self.cidx).to_numpy()
        self.R = csr_matrix(
            (np.ones(len(train)), (rows, cols)),
            shape=(len(self.users), len(self.courses)))
        self.pop = np.asarray(self.R.sum(axis=0)).ravel()
        self._fit()
        return self

    def _fit(self):
        pass

    def _scores(self, rows):
        raise NotImplementedError

    def recommend_batch(self, users, k, chunk=5000):
        """Return {user: [top-k course ids]} without courses the user already took."""
        tie_break = 1e-9 * self.pop / self.pop.max()  # popularity breaks exact ties
        out = {}
        for start in range(0, len(users), chunk):
            batch = users[start:start + chunk]
            rows = np.array([self.uidx[u] for u in batch])
            s = np.asarray(self._scores(rows), dtype=np.float64) + tie_break
            s[self.R[rows].toarray() > 0] = -np.inf
            top = np.argpartition(-s, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
            top = np.take_along_axis(top, order, axis=1)
            for u, t in zip(batch, top):
                out[u] = self.courses[t].tolist()
        return out


class Popularity(Recommender):
    """Everyone gets the most enrolled courses they have not taken yet."""

    def _scores(self, rows):
        return np.tile(self.pop, (len(rows), 1))


class ContentBased(Recommender):
    """Courses are described by their concepts (TF-IDF). A learner gets courses
    whose concepts are similar to the courses they already took."""

    def _fit(self):
        vocab = {c: i for i, c in enumerate(sorted(self.concepts["concept"].unique()))}
        rows = self.concepts["course"].map(self.cidx).to_numpy()
        cols = self.concepts["concept"].map(vocab).to_numpy()
        m = csr_matrix((np.ones(len(rows)), (rows, cols)),
                       shape=(len(self.courses), len(vocab)))
        x = TfidfTransformer().fit_transform(m)       # rows are L2-normalised
        self.S = (x @ x.T).toarray()                  # course-course similarity
        np.fill_diagonal(self.S, 0)

    def _scores(self, rows):
        return self.R[rows] @ self.S


class ItemCF(Recommender):
    """Item-based collaborative filtering (Sarwar et al., 2001) on enrolments:
    two courses are similar if the same learners took both (cosine similarity)."""

    def _fit(self):
        items = self.R.T.tocsr()
        norms = np.sqrt(np.asarray(items.multiply(items).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        items = csr_matrix(items.multiply(1 / norms[:, None]))
        self.S = (items @ items.T).toarray()
        np.fill_diagonal(self.S, 0)

    def _scores(self, rows):
        return self.R[rows] @ self.S


class MatrixFactorization(Recommender):
    """Latent factors from a truncated SVD of the enrolment matrix. A simplified
    version of the model in Koren et al. (2009): no regularisation term and no
    confidence weights."""

    def __init__(self, factors=50):
        self.factors = factors

    def _fit(self):
        svd = TruncatedSVD(n_components=self.factors, random_state=SEED)
        self.U = svd.fit_transform(self.R)            # learners x factors
        self.V = svd.components_                      # factors x courses

    def _scores(self, rows):
        return self.U[rows] @ self.V