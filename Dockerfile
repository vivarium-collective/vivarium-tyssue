FROM python:3.13-slim-bookworm AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openssh-client \
        git \
        build-essential \
        gcc \
        g++ \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# The vivatyssue fork (the tyssue source in pyproject) declares a docs-only
# submodule `doc/notebooks` via an SSH URL (git@github.com:damcb/tyssue-demo.git).
# uv initialises submodules when cloning a git source, which fails in CI with
# "Host key verification failed" (no SSH key in the build). Rewrite SSH GitHub
# URLs to anonymous HTTPS so the public submodule clones without credentials.
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

WORKDIR /app

COPY . /app

RUN mkdir -p /app/.results_cache

# `uv sync` resolves the workspace's deps. `--no-install-project` avoids
# building the workspace itself as a wheel (hatchling's bypass-selection
# would no-op anyway, but skipping the step is faster); the OR fallback
# covers the case where a future workspace removes that bypass.
# Run AFTER the full COPY so [tool.uv.sources] entries that point at
# paths inside the workspace (a vendored sibling, a local sub-checkout)
# resolve. Loses some layer cache on source-only edits — acceptable
# trade-off for robustness across source types.
RUN uv sync --no-install-project || uv sync

EXPOSE 9863

CMD ["uv", "run", "vivarium-dashboard", "serve", "--workspace", "/app", "--host", "0.0.0.0", "--port", "9863"]