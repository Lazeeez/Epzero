## What and why

## Tests

- [ ] New behaviour is tested, and `ctest` and `pytest` pass
- [ ] Engine changes pass the address/undefined and thread sanitizers

## Performance (search or build changes)

Baseline and change run alternately on the same machine, at least two rounds each: recall and queries per
second at equal recall, before and after. A change that doesn't help goes in `docs/performance.md` under
"Tried and removed" instead.

## File formats

- [ ] Nothing written to disk changed, or `docs/file-format.md` and its rules are followed
