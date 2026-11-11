"""VIBE adapter for Epzero (https://pypi.org/project/epzero/).

Copied into vibe/algorithms/epzero/ in the VIBE repository. Each index is configured by epzero.build:
8-bit codes, a walk on principal directions where the vectors' variance allows, the build passes and
the re-rank. VIBE then sweeps the beam width. On out-of-distribution datasets, fit_ood hands Epzero
VIBE's sample queries, which it calibrates its search on instead of rows of the data.
"""

import numpy as np

import epzero

from ..base.module import BaseANN

LEARN_SAMPLE = 1000


class Epzero(BaseANN):
    def __init__(self, metric, M, efConstruction):
        # "normalized" vectors are unit length, so the inner product ranks them as cosine does.
        self.metric = {"euclidean": "l2", "cosine": "cosine", "normalized": "ip", "ip": "ip"}[metric]
        self.M = M
        self.efConstruction = efConstruction
        self.ef = 10
        self.learn = None
        self.index = None

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        # VIBE builds every index on a single thread.
        self.index = epzero.build(X, metric=self.metric, M=self.M, ef_construction=self.efConstruction,
                                  threads=1, queries=self.learn)

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        rows = np.random.default_rng(0).choice(len(X_learn), min(len(X_learn), LEARN_SAMPLE), replace=False)
        self.learn = np.ascontiguousarray(X_learn[np.sort(rows)], dtype=np.float32)
        self.fit(X_train)

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, n):
        labels, _ = self.index.search(np.ascontiguousarray(v[None, :], dtype=np.float32), k=n, ef=max(self.ef, n))
        return labels[0][labels[0] >= 0]

    def __str__(self):
        plan = self.index.plan if self.index is not None else None
        chosen = f", nav_dim={plan.nav_dim}, rerank={plan.rerank}" if plan is not None else ""
        return f"Epzero(M={self.M}, efConstruction={self.efConstruction}, ef={self.ef}{chosen})"
