# Backend benchmark tracker

Per-`EulerSolver.update()` time and cell throughput by geometry/backend.
Higher `cell·updates/s` = more cells feasible per unit time. Appended by
`scripts/bench_backends.py`; full history in `results.jsonl`.

| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |
|---|---|--:|--:|--:|--:|---|
| monolayer | python | 1590 | 12.27 | 82 | 129,620 | `ec76db8` |
| sheet | python | 206 | 8.54 | 117 | 24,110 | `ec76db8` |
| sheet | rust | 206 | 3.54 | 282 | 58,177 | `ec76db8` |
| sheet-scan | python | 1482 | 12.06 | 83 | 122,870 | `ec76db8` |
| sheet-scan | rust | 1482 | 3.57 | 280 | 415,495 | `ec76db8` |
| vessel | python | 320 | 10.77 | 93 | 29,721 | `ec76db8` |
| vessel | rust | 320 | 6.86 | 146 | 46,622 | `ec76db8` |
