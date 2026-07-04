# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 11.65 | 86 | 136,450 | `3e43287` |
| sheet | python | 206 | 8.39 | 119 | 24,556 | `3e43287` |
| sheet | rust | 206 | 3.55 | 281 | 57,976 | `3e43287` |
| sheet-scan | python | 1482 | 11.76 | 85 | 126,056 | `3e43287` |
| sheet-scan | rust | 1482 | 3.56 | 281 | 416,299 | `3e43287` |
| vessel | python | 320 | 10.47 | 96 | 30,570 | `3e43287` |
| vessel | rust | 320 | 4.97 | 201 | 64,415 | `3e43287` |
