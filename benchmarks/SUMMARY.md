# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 11.67 | 86 | 136,284 | `fa61b7a` |
| sheet | python | 206 | 8.39 | 119 | 24,547 | `fa61b7a` |
| sheet | rust | 206 | 2.99 | 335 | 68,978 | `fa61b7a` |
| sheet-scan | python | 1482 | 11.53 | 87 | 128,544 | `fa61b7a` |
| sheet-scan | rust | 1482 | 3.52 | 284 | 421,242 | `fa61b7a` |
| vessel | python | 320 | 10.23 | 98 | 31,267 | `fa61b7a` |
| vessel | rust | 320 | 4.40 | 227 | 72,748 | `fa61b7a` |
