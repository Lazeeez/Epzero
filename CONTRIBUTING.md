# Contributing to Epzero

Thanks for helping. Epzero is a vector search engine where every change is measured, so this guide is
mostly about how to build, test and measure.

## Layout

| Directory | What's there |
|---|---|
| `engine/` | The C++20 library (`src/`, public headers in `include/epzero/`), its tests, benchmarks, fuzzers, the Python package (`python/`) and the WebAssembly bindings |
| `bench/` | Comparisons with other libraries (`compete.py`), the scoreboard generator and every measured result |
| `cluster/` | The Go router for the replicated cluster |
| `web/` | The site, epzero.dev |
| `docs/` | The performance log, the scoreboard, guides and design notes |

## Build and test

Linux, macOS or WSL2 with CMake 3.20+ and GCC 13+ or Clang 17+, or Windows with Visual Studio 2022.

```bash
cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

pip install "./engine[test]"          # the Python package with LangChain, LlamaIndex and Haystack
pytest -q engine/python/tests
```

Before a pull request touching the engine, also run the sanitizers (Clang):

```bash
cmake -S engine -B build-asan -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DEPZERO_SANITIZE=address,undefined
cmake -S engine -B build-tsan -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DEPZERO_SANITIZE=thread
```

Code that reads files (snapshots, the write-ahead log, metadata, the text index) has libFuzzer targets:
`-DEPZERO_BUILD_FUZZ=ON`, then `build/fuzz_<target>` for `snapshot`, `graph`, `wal`, `text` or
`metadata`. A crash it finds goes into `engine/fuzz/regressions/` with the fix.

## Performance changes: measure, and keep only gains

A change meant to make Epzero faster comes with a measurement, and so does any change to the search or
build path:

1. Build the baseline and your change side by side, and run them alternately on the same machine (at least
   two rounds each). `engine/bench/bench_recall` reports recall, queries per second and distances per query
   at each `ef`; `bench/compete.py` compares against other libraries.
2. Report recall and speed at equal recall, both before and after. A gain smaller than the spread between
   rounds isn't a gain.
3. If it doesn't help, say so in `docs/performance.md` under "Tried and removed": negative results are
   kept on purpose.

CI checks that recall and distance computations per query on a fixed synthetic dataset stay within their
recorded bounds (`engine/bench/regression.json`); update the bounds in the same pull request, with a
reason, if a change moves them on purpose.

## Style

- Match the code around you: C++20, `snake_case`, comments that say why rather than what, and the measured
  number behind any tuning constant.
- Public headers document thread safety and failure behaviour for every function.
- Errors are exceptions in C++ (`std::invalid_argument`, `std::runtime_error`, ...) and become Python
  exceptions through the bindings.
- Keep pull requests focused: one change, with its tests and its measurement.

## File formats

Saved indexes are a compatibility promise; see [docs/file-format.md](docs/file-format.md) before changing
anything that is written to disk.

## Reporting a bug

Use the issue templates. For a wrong or slow result, include `epzero.__version__`, `epzero.machine()`, the
index's `plan` (`print(index.plan)`), and a way to reproduce it.
