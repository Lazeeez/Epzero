"""Converts a published dataset's photo metadata from the first format (meta.json) to the compact
one build_dataset.py now writes (meta.v2.json), without rebuilding the dataset, and points the
manifest at it.

    python pipeline/compact_meta.py web/public/data/openimages
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_dataset import META_FILE, compact_meta  # noqa: E402


def main():
    out = sys.argv[1]
    with open(os.path.join(out, "meta.json"), encoding="utf-8") as f:
        old = json.load(f)
    rows = [
        {"host": host, "path": path, "labels": labels, "title": title, "author": author, "landing": landing,
         "license": old["licenses"][license]}
        for host, path, labels, title, author, landing, license in zip(
            old["host"], old["path"], old["labels"], old["title"], old["author"], old["landing"], old["license"])
    ]
    meta = compact_meta(rows, old["licenses"])
    with open(os.path.join(out, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"), ensure_ascii=False)
    manifest_path = os.path.join(out, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["files"]["meta"] = META_FILE
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.remove(os.path.join(out, "meta.json"))
    print(f"{len(rows)} photos -> {META_FILE} ({os.path.getsize(os.path.join(out, META_FILE)) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
