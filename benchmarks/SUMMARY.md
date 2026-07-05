# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 10.83 | 92 | 146,813 | `b8c2b47+dirty` |
| monolayer | rust | 1590 | 5.49 | 182 | 289,764 | `b8c2b47+dirty` |
| sheet | python | 206 | 7.74 | 129 | 26,601 | `b8c2b47+dirty` |
| sheet | rust | 206 | 1.33 | 754 | 155,398 | `b8c2b47+dirty` |
| sheet-substep | rust | 206 | 0.30 | 3308 | 681,405 | `b8c2b47+dirty` |
| vessel | python | 320 | 9.52 | 105 | 33,603 | `b8c2b47+dirty` |
| vessel | rust | 320 | 2.23 | 448 | 143,415 | `b8c2b47+dirty` |
