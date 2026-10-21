"""Writes the photo vectors so the site can search before they have all arrived.

The CLIP vectors are rotated onto their principal axes (an orthogonal rotation, so every inner
product, and with it every search result, is unchanged) and stored dimension-block by
dimension-block: first dimensions 0-63 of every photo, then 64-127, 128-255 and 256-511. The
leading dimensions carry the most variance, so a prefix of the file is a coarse but usable copy of
the whole dataset: the browser searches with the first 1.9 MB (64 dimensions) and sharpens as each
further block lands, ending at the full vectors.

Each block holds, for its dimensions [lo, hi):
  scales    float32[hi - lo]          per-dimension int8 scales (symmetric, as before)
  rotation  float16[(hi - lo) x dim]  the rotation's columns lo..hi, one row each, to rotate queries
  codes     int8[count x (hi - lo)]   the photos' rotated, quantized coordinates, row-major

Every part is a multiple of 4 bytes, so the blocks stay aligned for typed-array views.

The axes are the photos' principal components. Text queries sit apart from photos in CLIP's space
and need more of the dimensions: on 30 natural-language queries an exact search finds 29%, 52%,
73% and 99% of the true top 10 at 64, 128, 256 and 512 dimensions (photos: 61%, 79%, 91%, 99%).
Axes fitted to text prompts as well did worse for both. On a first visit that costs little: the
text model (64 MB) arrives well after the vectors, and the early stages serve photo searches.

    python pipeline/progressive_vectors.py --emb data/openimages/embeddings.npy --out web/public/data/openimages
"""

import argparse
import json
import os

import numpy as np

VECTORS_FILE = "vectors.v2.bin"
BLOCKS = (64, 128, 256, 512)


def rotation(emb):
    """Principal axes of the (centered) vectors, most variance first, with a fixed sign each."""
    centered = emb - emb.mean(axis=0)
    variance, axes = np.linalg.eigh(centered.T @ centered / (len(emb) - 1))
    axes = axes[:, np.argsort(-variance)]
    axes *= np.sign(axes[np.abs(axes).argmax(axis=0), np.arange(axes.shape[1])])
    return axes.astype(np.float32)


def quantize(x):
    """Symmetric per-dimension int8: q = round(x / scale * 127), scale = max |x| in that dimension."""
    scales = np.abs(x).max(axis=0).astype(np.float32)
    scales[scales == 0] = 1.0
    return np.clip(np.rint(x / scales * 127), -127, 127).astype(np.int8), scales


def prefix_recall(emb, codes, scales, dims, k=10, queries=1000, seed=0):
    """Recall@k of an exact search over the first `dims` dequantized dimensions vs. the float vectors."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(emb), size=min(queries, len(emb)), replace=False)
    deq = codes[:, :dims].astype(np.float32) * (scales[:dims] / 127)
    deq /= np.linalg.norm(deq, axis=1, keepdims=True)
    exact = np.argsort(-(emb[picks] @ emb.T), axis=1)[:, 1:k + 1]
    approx = np.argsort(-(deq[picks] @ deq.T), axis=1)[:, 1:k + 1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(exact, approx)]))


def export(emb, out):
    """Writes VECTORS_FILE and returns its manifest entry."""
    emb = np.asarray(emb, dtype=np.float32)
    dim = emb.shape[1]
    if BLOCKS[-1] != dim:
        raise ValueError(f"blocks end at {BLOCKS[-1]}, vectors have {dim} dimensions")
    axes = rotation(emb)
    codes, scales = quantize(emb @ axes)
    with open(os.path.join(out, VECTORS_FILE), "wb") as f:
        lo = 0
        for hi in BLOCKS:
            f.write(scales[lo:hi].tobytes())
            f.write(axes[:, lo:hi].T.astype(np.float16).tobytes())
            f.write(np.ascontiguousarray(codes[:, lo:hi]).tobytes())
            lo = hi
    stages = [{"dims": d, "recallAt10VsFloat": round(prefix_recall(emb, codes, scales, d), 4)} for d in BLOCKS]
    print("progressive vectors: " + ", ".join(f"{s['dims']} dims recall@10 {s['recallAt10VsFloat']}" for s in stages))
    return {"file": VECTORS_FILE, "layout": "pca-blocks", "stages": stages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True, help="float CLIP embeddings (.npy), in the dataset's photo order")
    parser.add_argument("--out", required=True, help="dataset directory holding manifest.json")
    args = parser.parse_args()
    entry = export(np.load(args.emb), args.out)
    path = os.path.join(args.out, "manifest.json")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["files"]["vectors"] = entry["file"]
    manifest["files"].pop("scales", None)
    manifest["vectors"] = {"layout": entry["layout"], "stages": entry["stages"]}
    manifest["quantization"]["recallAt10VsFloat"] = entry["stages"][-1]["recallAt10VsFloat"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
