# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 11.35 | 88 | 140,137 | `5a64e6e+dirty` |
| sheet | python | 206 | 8.20 | 122 | 25,116 | `5a64e6e+dirty` |
| sheet | rust | 206 | 3.55 | 282 | 58,098 | `5a64e6e+dirty` |
| sheet-scan | python | 1482 | 11.19 | 89 | 132,390 | `e52e876` |
| sheet-scan | rust | 1482 | 10.12 | 99 | 146,371 | `e52e876` |
| vessel | python | 320 | 10.47 | 96 | 30,557 | `5a64e6e+dirty` |
