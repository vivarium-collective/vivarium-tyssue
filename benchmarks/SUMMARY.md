# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 10.73 | 93 | 148,191 | `15ba2be+dirty` |
| monolayer | rust | 1590 | 5.40 | 185 | 294,675 | `15ba2be+dirty` |
| sheet | python | 206 | 7.59 | 132 | 27,126 | `15ba2be+dirty` |
| sheet | rust | 206 | 1.39 | 721 | 148,569 | `15ba2be+dirty` |
| sheet-substep | rust | 206 | 0.31 | 3276 | 674,894 | `15ba2be+dirty` |
| vessel | python | 320 | 9.41 | 106 | 34,017 | `15ba2be+dirty` |
| vessel | rust | 320 | 2.39 | 418 | 133,714 | `15ba2be+dirty` |
