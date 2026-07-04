# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 12.00 | 83 | 132,450 | `0f2d2f4` |
| sheet | python | 206 | 8.79 | 114 | 23,442 | `0f2d2f4` |
| sheet | rust | 206 | 2.42 | 414 | 85,255 | `0f2d2f4` |
| sheet-scan | python | 1482 | 11.71 | 85 | 126,579 | `0f2d2f4` |
| sheet-scan | rust | 1482 | 3.53 | 283 | 419,649 | `0f2d2f4` |
| vessel | python | 320 | 10.89 | 92 | 29,380 | `0f2d2f4` |
| vessel | rust | 320 | 3.64 | 274 | 87,856 | `0f2d2f4` |
