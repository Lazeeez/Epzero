# File formats and compatibility

Epzero writes four kinds of file. Each starts with an 8-byte magic string and, except the text index, a
version number.

| File | Written by | Magic | Versions |
|---|---|---|---|
| Index snapshot | `Index.save`, `HnswIndex::save` | `EPZRHNSW` (`ORYXHNSW` before the rename, still read) | 1-5 |
| Metadata | `Metadata.save`, `MetadataIndex::save` | `EPZRMETA` | 1 |
| Text index | `TextIndex.save` | `EPZRTXT1`, `EPZRTXT2` (`ORYXTXT1` still read) | 1-2 |
| Database | `Database` directory: `wal.log`, `CHECKPOINT` and `snapshot-<lsn>.epzero` (an index snapshot) | per record CRC32C | 1 |

## The promise

1. **A newer Epzero reads every file an older one wrote.** Old versions are never dropped.
2. **A file is written in the oldest version that can hold it.** An index that uses nothing newer is still
   written as version 2, so older releases keep reading it. Version 3 is written only by indexes that walk
   on principal directions, and version 5 only by indexes with a saved profile (from `epzero.build`,
   `recalibrate`, or `set_profile`) or 16-bit vectors. Version 4 is read but no longer written.
3. **A file a release can't read is refused, never misread.** An unknown version, magic, storage flag or
   enum value is an error, and so is a header that claims more than the file holds.
4. **Loading untrusted files is safe.** Every count is checked against the file size before anything is
   allocated, every graph link is checked to name a node on that link's layer, and `max_bytes` caps what
   loading may allocate. The loaders are fuzzed in CI.

## Index snapshot versions

| Version | Since | Adds |
|---|---|---|
| 1 | before 0.2 | Full-precision vectors and the graph |
| 2 | 0.2 | Quantization (8-bit and RaBitQ) and tombstones |
| 3 | 0.4 | Principal directions (`nav_dim`) and their basis |
| 4 | 0.4 | The `IndexProfile`: the default search settings, how the index was built, the reason for each choice, and the prefetch setting with the machine it was timed on |
| 5 | 0.5 | A storage-flags byte (bit 0: 16-bit full vectors), and in the profile the size the search was calibrated at and the variance the principal directions held |

The layout is little-endian and doesn't depend on in-memory padding: a header (dimension, metric,
capacity, M, efConstruction, seed, quantization, whether full vectors are kept, then version 3's
`nav_dim`, version 5's flags, and the profile from version 4), the node count, the entry point and the tombstone count, the
quantizer's state, and then each node's record (codes or vector, layer-0 links, label, level, tombstone,
upper-layer links, and the full-precision copy when kept). Prefetch settings are machine-specific: a
loaded index applies its saved one only on the same kind of CPU (`epzero.machine()`).

## Text index versions

The text index's version is the last character of its magic string.

| Version | Since | Adds |
|---|---|---|
| 1 | before 0.5 | k1 and b, then each live document's label, length and (term, count) pairs |
| 2 | 0.5 | A stemmer byte after b (1: Porter). Written only by indexes that stem; version 1 files were written without stemming and load with `stemmer="none"`, so their terms still match their queries |

## Changing a format

A pull request that changes what is written must:

- bump the version, and write the new version only when the new content is present;
- keep reading every older version, with a test that loads a file a release actually wrote (as
  `engine/tests/data/v4-from-0.4.0.epzero`, written by 0.4.0 from PyPI);
- validate every new field on load (range, size against the file, enum values);
- extend the fuzz seed corpus (`engine/fuzz/make_seeds.cpp`) with a file in the new version;
- update this page.
