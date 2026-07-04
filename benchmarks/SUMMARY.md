# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 10.82 | 92 | 146,917 | `1447aa9` |
| sheet | python | 206 | 7.62 | 131 | 27,024 | `1447aa9` |
| sheet | rust | 206 | 1.41 | 709 | 146,097 | `1447aa9` |
| sheet-scan | python | 1482 | 11.34 | 88 | 130,633 | `1447aa9` |
| sheet-scan | rust | 1482 | 1.35 | 740 | 1,096,357 | `1447aa9` |
| vessel | python | 320 | 9.57 | 104 | 33,440 | `1447aa9` |
| vessel | rust | 320 | 2.45 | 408 | 130,662 | `1447aa9` |
