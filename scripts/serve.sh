#!/usr/bin/env bash
# Thin shim around the vivarium-workbench pip package.
#
# The dashboard runtime (server + templates + assets + lib helpers) was
# extracted out of pbg-template into the standalone `vivarium-workbench`
# package. Workspaces now depend on it as a regular pip dep; this script
# is just a convenience wrapper so `bash scripts/serve.sh` keeps working.
set -euo pipefail

# COPASI/basico (pbg-copasi dep) resets LC_CTYPE to "C" on import; force Python
# UTF-8 mode so locale-default YAML reads (composite/study specs) don't crash on
# non-ASCII characters when the dashboard loads composites that import COPASI.
export PYTHONUTF8=1

WS_ROOT="$(pwd)"
[ -f "$WS_ROOT/workspace.yaml" ] || { echo "ERROR: run from workspace root" >&2; exit 1; }

# The dashboard CLI is named `vivarium-dashboard` at the generation pyproject.toml
# pins, and `vivarium-workbench` at the repo's HEAD (the dist was renamed — see the
# GENERATION PIN note in pyproject.toml). Try both, venv first (matches the
# pbg-template scaffolding flow), then a system-wide install.
DASH=""
for name in vivarium-dashboard vivarium-workbench; do
    if [ -x "$WS_ROOT/.venv/bin/$name" ]; then
        DASH="$WS_ROOT/.venv/bin/$name"; break
    fi
    if command -v "$name" >/dev/null 2>&1; then
        DASH="$(command -v "$name")"; break
    fi
done

if [ -z "$DASH" ]; then
    echo "ERROR: the dashboard is not installed (looked for vivarium-dashboard and" >&2
    echo "       vivarium-workbench, in .venv/bin and on PATH)." >&2
    echo "Build the environment first — see SETUP.md at the repo root." >&2
    exit 2
fi

exec "$DASH" serve --workspace "$WS_ROOT"
