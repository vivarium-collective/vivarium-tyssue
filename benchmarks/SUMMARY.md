# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 11.23 | 89 | 141,638 | `41ac2de+dirty` |
| sheet | python | 206 | 8.11 | 123 | 25,410 | `41ac2de+dirty` |
| sheet | rust | 206 | 3.39 | 295 | 60,718 | `41ac2de+dirty` |
| sheet-scan | python | 1482 | 11.51 | 87 | 128,717 | `41ac2de+dirty` |
| sheet-scan | rust | 1482 | 3.42 | 292 | 432,898 | `41ac2de+dirty` |
| vessel | python | 320 | 10.05 | 100 | 31,835 | `41ac2de+dirty` |
