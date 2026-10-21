"""Converts an ann-benchmarks HDF5 file into the TEXMEX files the C++ benchmarks read:
<prefix>_base.fvecs, <prefix>_query.fvecs and <prefix>_groundtruth.ivecs (exact top 100).

    python scripts/hdf5_to_vecs.py data/ann/glove-100-angular.hdf5 data/ann/glove

Vectors are written as stored; for angular datasets pass --metric cosine to the benchmarks, which
normalize them.
"""

import sys

import h5py
import numpy as np


def write(path, rows, dtype):
    rows = np.ascontiguousarray(rows, dtype=dtype)
    n, d = rows.shape
    out = np.empty((n, d + 1), dtype=dtype)
    out[:, 0] = np.array([d], dtype=np.int32).view(dtype)[0]  # each row starts with its int32 length
    out[:, 1:] = rows
    out.tofile(path)


def main():
    source, prefix = sys.argv[1], sys.argv[2]
    with h5py.File(source, "r") as f:
        write(prefix + "_base.fvecs", f["train"][:], np.float32)
        write(prefix + "_query.fvecs", f["test"][:], np.float32)
        write(prefix + "_groundtruth.ivecs", f["neighbors"][:, :100], np.int32)
        print(f"{source}: {f['train'].shape[0]} x {f['train'].shape[1]} base, {f['test'].shape[0]} queries -> {prefix}_*")


if __name__ == "__main__":
    main()
