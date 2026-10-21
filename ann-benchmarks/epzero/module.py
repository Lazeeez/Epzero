"""ann-benchmarks wrapper for Epzero (https://pypi.org/project/epzero/).

Copied into ann_benchmarks/algorithms/epzero/ in the ann-benchmarks repository.
"""

import os

import numpy as np
import epzero

from ..base.module import BaseANN


class Epzero(BaseANN):
    def __init__(self, metric, method_param, quantization="none"):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError(f"Epzero does not support the {metric} metric")
        self.metric = "cosine" if metric == "angular" else "l2"
        self.M = method_param["M"]
        self.ef_construction = method_param["efConstruction"]
        self.quantization = quantization
        self.ef = 10
        self.name = f"epzero(M={self.M}, efC={self.ef_construction}, {quantization})"

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.index = epzero.Index(
            X.shape[1], self.metric, len(X), self.M, self.ef_construction, quantization=self.quantization
        )
        if self.quantization != "none":
            self.index.train(X[: min(len(X), 200_000)])
        self.index.add(X, threads=os.cpu_count() or 1)
        # Measure the best memory-prefetch setting for this machine on a sample of the data.
        sample = np.random.default_rng(0).choice(len(X), min(len(X), 1000), replace=False)
        self.index.tune_prefetch(X[sample])

    def set_query_arguments(self, ef):
        self.ef = ef
        self.name = f"epzero(M={self.M}, efC={self.ef_construction}, {self.quantization}, ef={ef})"

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


class EpzeroSQ8(Epzero):
    def __init__(self, metric, method_param):
        super().__init__(metric, method_param, quantization="sq8")
