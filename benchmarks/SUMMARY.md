# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 11.54 | 87 | 137,759 | `df905a5` |
| sheet | python | 206 | 7.91 | 126 | 26,043 | `df905a5` |
| sheet | rust | 206 | 3.29 | 304 | 62,659 | `df905a5` |
| sheet-scan | python | 1482 | 11.35 | 88 | 130,576 | `df905a5` |
| sheet-scan | rust | 1482 | 3.45 | 290 | 429,454 | `df905a5` |
| vessel | python | 320 | 10.06 | 99 | 31,810 | `df905a5` |
