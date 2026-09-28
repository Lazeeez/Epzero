"""Builds the demo dataset: Open Images V7 validation photos -> CLIP embeddings -> UMAP layout ->
compact binaries the browser loads.

Why Open Images: its images are listed under CC BY 2.0 with author/title metadata and its labels
under CC BY 4.0, so a public demo can show them with attribution. (Datasets whose terms forbid
redistribution, such as Unsplash Lite, cannot be used for a public site.)

Stages are cached under --work, so reruns only redo what changed:

  1. metadata     image list, licenses, authors, and human-verified labels
  2. thumbnails   320px Flickr thumbnails; photos that were deleted from Flickr are dropped
  3. embeddings   CLIP ViT-B/32 image embeddings (L2-normalized, 512-d), GPU if available
  4. check        text->image sanity check with the same model's text tower
  5. layout       UMAP to 2-D (cosine), used to place nodes in the 3D view
  6. export       int8 vectors (per-dimension symmetric scales), layout, metadata, manifest

The browser embeds text queries with the same model (Xenova/clip-vit-base-patch32 in
transformers.js is a conversion of openai/clip-vit-base-patch32), so text and image vectors
share one space.

    python pipeline/build_dataset.py --work data/openimages --out web/public/data/openimages
"""

import argparse
import collections
import concurrent.futures as futures
import csv
import json
import os
import time
import urllib.request

import numpy as np

MODEL = "openai/clip-vit-base-patch32"
META_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
LABELS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv"
CLASSES_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
FLICKR = "https://c{host}.staticflickr.com/{path}_{size}.jpg"


def projected(output):
    """CLIP get_*_features returns a tensor in transformers 4.x and an output object in 5.x."""
    features = output if hasattr(output, "shape") else output.pooler_output
    if features.shape[-1] != 512:
        raise RuntimeError(f"expected 512-d projected CLIP embeddings, got {tuple(features.shape)}")
    return features.float()


def fetch(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


def stage_metadata(work, limit):
    """Rows with a standard Flickr thumbnail, plus up to four human-verified labels each."""
    os.makedirs(work, exist_ok=True)
    rows = []
    with open(fetch(META_URL, os.path.join(work, "images.csv")), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            url = r["Thumbnail300KURL"]
            # e.g. https://c8.staticflickr.com/6/5612/15340259497_6b4323bab9_z.jpg
            if not (url.startswith("https://c") and ".staticflickr.com/" in url and url.endswith("_z.jpg")):
                continue
            host = url[len("https://c"):url.index(".staticflickr.com/")]
            path = url[url.index(".staticflickr.com/") + len(".staticflickr.com/"):-len("_z.jpg")]
            rows.append({"id": r["ImageID"], "host": host, "path": path, "author": r["Author"].strip(),
                         "title": r["Title"].strip(), "landing": r["OriginalLandingURL"], "license": r["License"]})
    if limit:
        rows = rows[:limit]

    names = {}
    with open(fetch(CLASSES_URL, os.path.join(work, "classes.csv")), newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            names[r[0]] = r[1]
    labels = collections.defaultdict(list)
    with open(fetch(LABELS_URL, os.path.join(work, "labels.csv")), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Confidence"] == "1" and r["LabelName"] in names:
                labels[r["ImageID"]].append(names[r["LabelName"]])
    frequency = collections.Counter(label for ls in labels.values() for label in ls)
    for row in rows:
        # Most specific (rarest) labels first: "Golden retriever" says more than "Dog" or "Mammal".
        row["labels"] = sorted(set(labels.get(row["id"], [])), key=lambda l: frequency[l])[:4]
    print(f"metadata: {len(rows)} images, {sum(1 for r in rows if r['labels'])} with verified labels")
    return rows


def stage_thumbnails(rows, work, threads):
    """Downloads 320px thumbnails; returns which rows are still available on Flickr."""
    folder = os.path.join(work, "thumbs")
    os.makedirs(folder, exist_ok=True)
    missing_log = os.path.join(work, "missing.txt")
    missing = set(open(missing_log).read().split()) if os.path.exists(missing_log) else set()

    def get(row):
        path = os.path.join(folder, row["id"] + ".jpg")
        if os.path.exists(path):
            return True
        if row["id"] in missing:
            return False
        url = FLICKR.format(host=row["host"], path=row["path"], size="n")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    # Deleted photos redirect to a "photo unavailable" placeholder image.
                    if "unavailable" in response.url or response.headers.get("Content-Type") != "image/jpeg":
                        return None
                    data = response.read()
                with open(path + ".tmp", "wb") as f:
                    f.write(data)
                os.replace(path + ".tmp", path)
                return True
            except urllib.error.HTTPError as e:
                if e.code in (404, 410):
                    return None
            except Exception:
                pass
            time.sleep(1 + 2 * attempt)
        return False  # transient failure: retried on the next run

    start = time.time()
    results = []
    with futures.ThreadPoolExecutor(threads) as pool:
        for i, result in enumerate(pool.map(get, rows), 1):
            results.append(result)
            if i % 5000 == 0:
                print(f"  thumbnails {i}/{len(rows)} ({time.time() - start:.0f}s)", flush=True)
    gone = {row["id"] for row, result in zip(rows, results) if result is None}
    with open(missing_log, "w") as f:
        f.write("\n".join(sorted(missing | gone)))
    available = np.array([r is True for r in results])
    print(f"thumbnails: {available.sum()}/{len(rows)} available ({len(missing | gone)} deleted from Flickr)")
    return available


def stage_embeddings(rows, work, batch):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    path = os.path.join(work, "embeddings.npy")
    ids_path = os.path.join(work, "embeddings.ids")
    ids = [r["id"] for r in rows]
    if os.path.exists(path) and os.path.exists(ids_path) and open(ids_path).read().split() == ids:
        return np.load(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    if device == "cuda":
        model = model.half()
    processor = CLIPProcessor.from_pretrained(MODEL)
    folder = os.path.join(work, "thumbs")

    def load(image_id):
        return Image.open(os.path.join(folder, image_id + ".jpg")).convert("RGB")

    out = np.zeros((len(rows), 512), dtype=np.float32)
    start = time.time()
    with futures.ThreadPoolExecutor(8) as pool, torch.inference_mode():
        for lo in range(0, len(rows), batch):
            images = list(pool.map(load, ids[lo:lo + batch]))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            if device == "cuda":
                pixels = pixels.half()
            features = projected(model.get_image_features(pixel_values=pixels))
            out[lo:lo + len(images)] = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
            if (lo // batch) % 25 == 0:
                print(f"  embedded {lo + len(images)}/{len(rows)} ({time.time() - start:.0f}s, {device})", flush=True)
    np.save(path, out)
    with open(ids_path, "w") as f:
        f.write("\n".join(ids))
    print(f"embeddings: {out.shape} in {time.time() - start:.0f}s on {device}")
    return out


def text_embeddings(texts):
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(MODEL).eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL)
    with torch.inference_mode():
        features = projected(model.get_text_features(**tokenizer(texts, padding=True, return_tensors="pt")))
    return torch.nn.functional.normalize(features, dim=-1).numpy()


def describe(row):
    return " · ".join(row["labels"]) or row["title"]


def stage_check(rows, emb):
    queries = ["a dog playing on the beach", "snowy mountains", "a city street at night",
               "a bowl of food on a table", "an old black and white photograph"]
    scores = text_embeddings(queries) @ emb.T
    report = {}
    for query, row_scores in zip(queries, scores):
        top = np.argsort(-row_scores)[:5]
        report[query] = [describe(rows[i]) for i in top]
        print(f"  '{query}':")
        for i in top:
            print(f"      {row_scores[i]:.3f}  {describe(rows[i])[:80]}")
    return report


def stage_layout(work, emb):
    import umap

    path = os.path.join(work, "layout.npy")
    if os.path.exists(path) and np.load(path).shape[0] == len(emb):
        return np.load(path)
    start = time.time()
    xy = umap.UMAP(n_neighbors=30, min_dist=0.08, metric="cosine", random_state=42).fit_transform(emb)
    xy = xy - xy.mean(0)
    xy = xy / np.abs(xy).max()
    np.save(path, xy.astype(np.float32))
    print(f"layout: UMAP in {time.time() - start:.0f}s")
    return xy


def quantize(emb):
    """Symmetric per-dimension int8: q = round(x / scale * 127), scale = max |x| in that dimension."""
    scales = np.abs(emb).max(axis=0).astype(np.float32)
    scales[scales == 0] = 1.0
    q = np.clip(np.rint(emb / scales * 127), -127, 127).astype(np.int8)
    return q, scales


def quantization_recall(emb, q, scales, k=10, queries=1000, seed=0):
    """Recall@k of exact search over dequantized vectors vs. the float vectors (same queries)."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(emb), size=min(queries, len(emb)), replace=False)
    deq = q.astype(np.float32) * (scales / 127)
    deq /= np.linalg.norm(deq, axis=1, keepdims=True)
    exact = np.argsort(-(emb[picks] @ emb.T), axis=1)[:, 1:k + 1]
    approx = np.argsort(-(emb[picks] @ deq.T), axis=1)[:, 1:k + 1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(exact, approx)]))


def stage_export(rows, emb, xy, out, check):
    os.makedirs(out, exist_ok=True)
    q, scales = quantize(emb)
    recall = quantization_recall(emb, q, scales)
    q.tofile(os.path.join(out, "vectors.i8"))
    scales.tofile(os.path.join(out, "scales.f32"))
    np.rint((xy + 1) / 2 * 65535).clip(0, 65535).astype(np.uint16).tofile(os.path.join(out, "layout.u16"))

    licenses = sorted({r["license"] for r in rows})
    meta = {
        # Column-oriented for compactness; thumbnail URL = FLICKR with host/path/size.
        "id": [r["id"] for r in rows],
        "host": [r["host"] for r in rows],
        "path": [r["path"] for r in rows],
        "labels": [r["labels"] for r in rows],
        "title": [r["title"][:80] for r in rows],
        "author": [r["author"][:60] for r in rows],
        "landing": [r["landing"] for r in rows],
        "license": [licenses.index(r["license"]) for r in rows],
        "licenses": licenses,
    }
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"), ensure_ascii=False)
    manifest = {
        "name": "Open Images V7 (validation) × CLIP ViT-B/32",
        "count": len(rows),
        "dim": int(emb.shape[1]),
        "model": MODEL,
        "browserTextModel": "Xenova/clip-vit-base-patch32",
        "metric": "cosine",
        "files": {"vectors": "vectors.i8", "scales": "scales.f32", "layout": "layout.u16", "meta": "meta.json"},
        "quantization": {"scheme": "int8 symmetric per-dimension", "recallAt10VsFloat": round(recall, 4)},
        "thumbnail": "https://c{host}.staticflickr.com/{path}_{size}.jpg",
        "attribution": ("Images from Open Images V7 (Google), hosted on Flickr under the licenses listed per image "
                        "(CC BY 2.0); image-level labels CC BY 4.0. Each image links to its author and source."),
        "sanityCheck": check,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    sizes = {name: os.path.getsize(os.path.join(out, name)) for name in os.listdir(out)}
    print(f"export: int8 recall@10 vs float = {recall:.4f}; licenses {licenses}; files: " +
          ", ".join(f"{k} {v / 1e6:.2f} MB" for k, v in sorted(sizes.items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, help="cache directory for metadata, thumbnails and intermediates")
    parser.add_argument("--out", required=True, help="output directory served by the web app")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    rows = stage_metadata(args.work, args.limit)
    available = stage_thumbnails(rows, args.work, args.threads)
    rows = [r for r, ok in zip(rows, available) if ok]
    emb = stage_embeddings(rows, args.work, args.batch)
    check = stage_check(rows, emb)
    xy = stage_layout(args.work, emb)
    stage_export(rows, emb, xy, args.out, check)


if __name__ == "__main__":
    main()
