# Project guidance for AI agents

## Protected files

**Do not modify `vivarium_tyssue/processes/eulersolver.py` unless the user
explicitly asks for it in the current request.** EulerSolver is the load-bearing
integrator for every simulation; autonomous edits are not permitted.

This is enforced two ways:
- **Contract tests** — `tests/test_eulersolver_history_columns.py` pins the config
  schema and history-recording behaviour; a silent change breaks them.
- **Commit hook** — `.githooks/pre-commit` blocks any commit that stages
  `eulersolver.py` from inside an AI-agent session (it keys on the `CLAUDECODE` /
  `AI_AGENT` environment markers). Manual commits from a human terminal are never
  affected. Enable it once per clone with `git config core.hooksPath .githooks`.
  When the user has explicitly authorized an EulerSolver change, commit it with
  `ALLOW_EULERSOLVER_EDIT=1 git commit …`.

## Environment

- Run Python and tests in the **`vivarium-tyssue` conda env** (the `.venv`/`uv`
  setup is broken): e.g. `conda run -n vivarium-tyssue python -m pytest …`.
