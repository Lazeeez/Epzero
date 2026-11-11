"""ann-benchmarks wrapper for Epzero (https://pypi.org/project/epzero/).

Copied into ann_benchmarks/algorithms/epzero/ in the ann-benchmarks repository. Each index is
configured by epzero.build (8-bit codes, a walk on principal directions where the data allows, the
build passes, the re-rank, and prefetching timed on the machine); ann-benchmarks then sweeps the
beam width.
"""

import os

import numpy as np
import epzero

from ..base.module import BaseANN


class Epzero(BaseANN):
    def __init__(self, metric, method_param):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError(f"Epzero does not support the {metric} metric")
        self.metric = "cosine" if metric == "angular" else "l2"
        self.M = method_param["M"]
        self.ef_construction = method_param["efConstruction"]
        self.ef = 10
        self.name = f"epzero(M={self.M}, efC={self.ef_construction})"

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.index = epzero.build(X, metric=self.metric, M=self.M, ef_construction=self.ef_construction,
                                  threads=os.cpu_count() or 1)

    def set_query_arguments(self, ef):
        self.ef = ef
        self.name = f"epzero(M={self.M}, efC={self.ef_construction}, ef={ef})"

    def query(self, v, n):
        labels, _ = self.index.search(np.ascontiguousarray(v[None, :], dtype=np.float32), k=n, ef=max(self.ef, n))
        return labels[0][labels[0] >= 0]

    def batch_query(self, X, n):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.batch_results, _ = self.index.search(X, k=n, ef=max(self.ef, n), threads=os.cpu_count() or 1)

    def get_batch_results(self):
        return [row[row >= 0] for row in self.batch_results]

    def __str__(self):
        return self.name
