#!/usr/bin/env bash
# Thin shim around the vivarium-dashboard pip package.
#
# The dashboard runtime (server + templates + assets + lib helpers) was
# extracted out of pbg-template into the standalone `vivarium-dashboard`
# package. Workspaces now depend on it as a regular pip dep; this script
# is just a convenience wrapper so `bash scripts/serve.sh` keeps working.
set -euo pipefail

# COPASI/basico (pbg-copasi dep) resets LC_CTYPE to "C" on import; force Python
# UTF-8 mode so locale-default YAML reads (composite/study specs) don't crash on
# non-ASCII characters when the dashboard loads composites that import COPASI.
export PYTHONUTF8=1

WS_ROOT="$(pwd)"
[ -f "$WS_ROOT/workspace.yaml" ] || { echo "ERROR: run from workspace root" >&2; exit 1; }

# Prefer the workspace venv (matches the pbg-template scaffolding flow);
# fall back to a system-wide install if the venv has no vivarium-dashboard.
if [ -x "$WS_ROOT/.venv/bin/vivarium-dashboard" ]; then
    DASH="$WS_ROOT/.venv/bin/vivarium-dashboard"
else
    DASH="$(command -v vivarium-dashboard 2>/dev/null || true)"
fi

if [ -z "$DASH" ]; then
    echo "ERROR: vivarium-dashboard is not installed." >&2
    echo "Install it into the workspace venv, e.g." >&2
    echo "    .venv/bin/pip install vivarium-dashboard" >&2
    echo "or, for local dev:" >&2
    echo "    .venv/bin/pip install -e /path/to/vivarium-dashboard" >&2
    exit 2
fi

exec "$DASH" serve --workspace "$WS_ROOT"
