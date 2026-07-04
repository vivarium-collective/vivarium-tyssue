# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 10.88 | 92 | 146,199 | `032bec8` |
| sheet | python | 206 | 7.96 | 126 | 25,875 | `032bec8` |
| sheet | rust | 206 | 6.35 | 157 | 32,427 | `032bec8` |
| vessel | python | 320 | 9.83 | 102 | 32,561 | `032bec8` |
