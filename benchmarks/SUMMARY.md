# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 10.69 | 94 | 148,686 | `1631640+dirty` |
| sheet | python | 206 | 7.77 | 129 | 26,505 | `1631640+dirty` |
| sheet | rust | 206 | 1.42 | 706 | 145,425 | `1631640+dirty` |
| sheet-scan | python | 1482 | 11.77 | 85 | 125,866 | `1631640+dirty` |
| sheet-scan | rust | 1482 | 1.31 | 766 | 1,135,288 | `1631640+dirty` |
| sheet-substep | rust | 206 | 0.32 | 3149 | 648,632 | `1631640+dirty` |
| vessel | python | 320 | 9.32 | 107 | 34,337 | `1631640+dirty` |
| vessel | rust | 320 | 2.39 | 418 | 133,913 | `1631640+dirty` |
