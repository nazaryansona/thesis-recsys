# Reading notes

## Sarwar et al. (2001), item-based collaborative filtering

**What it is:** A collaborative filtering method that compares items instead of users.

**How it works:** It finds which items are similar from the user-item matrix, then recommends items similar to the ones a user already liked. Item relationships change more slowly than the user population, so they can be prepared in advance.

**What I need to know for my data:** The paper uses ratings. I only have enrolments (yes or no), so I use cosine similarity on binary columns and sum the similarities to a learner's courses.

**Strength / weakness:** Strength: much faster than user-based methods, with equal or better quality according to the authors. Weakness: it needs enough shared learners between courses to give reliable similarities. In my setup, a course that no learner has taken has no similarity scores and therefore cannot be recommended.

**Use in my thesis:** Section 2.4, Section 3.4 (method 3).

## Koren, Bell & Volinsky (2009), matrix factorization

**What it is:** A collaborative filtering approach that describes users and items with a few hidden (latent) factors.

**How it works:** Each user and each item becomes a vector of f factors. The score of an item for a user is the inner product of the two vectors.

**What I need to know for my data:** The paper works with ratings and also covers implicit feedback with confidence levels. I have only enrolments, and I use a simplified truncated SVD without regularisation or confidence weights. This is a limitation to state.

**Strength / weakness:** Strength: it beat nearest-neighbour methods in the Netflix Prize. Weakness: it overfits easily when few entries are known (it needs regularisation), and users with few ratings are hard to model (cold start).

**Use in my thesis:** Section 2.4, Section 3.4 (method 4), cold-start discussion in Analysis.

## Herlocker et al. (2004), evaluating recommenders

**What it is:** A review of how to evaluate collaborative filtering: user tasks, datasets, accuracy metrics and metrics beyond accuracy.

**How it works:** It sorts metrics into predictive, classification and rank metrics, discusses their strengths and weaknesses, and tests how they agree on one dataset.

**What I need to know for my data:** My data is unary (an enrolment is a positive signal, a missing one is unknown). Recall on hidden items is only an approximation and is for comparing methods on the same dataset. Per-user averages and overall averages can rank methods differently. Catalog coverage is the share of items ever recommended.

**Strength / weakness:** Strength: clear guidance on metrics and datasets. Weakness: it is from 2004, built around ratings and movie data, and it does not cover NDCG.

**Use in my thesis:** Section 2.6, Section 3.5.
