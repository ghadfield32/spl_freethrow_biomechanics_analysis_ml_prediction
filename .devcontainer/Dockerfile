# ── Base with GPU drivers ──────────────────────────────────────────────────────
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ARG PYTHON_VER=3.10
ARG ENV_NAME=data_science_ft_bio_predictions
ENV DEBIAN_FRONTEND=noninteractive

# ---- 1. OS deps + real Python -------------------------------------------------
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt \
    --mount=type=cache,id=apt-lists,target=/var/lib/apt/lists \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential git \
        python3 python3-venv python3-pip procps htop util-linux && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- 2. uv binary -------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.5.7 /uv /uvx /bin/

# ---- 3. Project root is *always* /workspace -----------------------------------
WORKDIR /workspace
COPY pyproject.toml ./

# ---- 4. Create venv, regenerate lockfile, and sync deps ----------------------
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv venv .venv --python ${PYTHON_VER} --prompt "${ENV_NAME}" && \
    uv lock && \
    uv sync --no-dev --no-install-project

ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"

# ---- 5. Install GPU-enabled PyTorch and JAX wheels ---------------------------
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install \
        "jax[cuda12-local]==0.4.38" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 6) copy project source last (keeps rebuilds quick when code changes)
COPY . /workspace

# 7) configure library paths for GPU libraries
ENV LD_LIBRARY_PATH="/workspace/.venv/lib:${LD_LIBRARY_PATH}"

# 8) ensure PATH / prompt activation
RUN printf '%s\n' \
    'export VIRTUAL_ENV=/workspace/.venv' \
    'export PATH="$VIRTUAL_ENV/bin:$PATH"' \
    'export LD_LIBRARY_PATH="/workspace/.venv/lib:$LD_LIBRARY_PATH"' \
  > /etc/profile.d/00-uv.sh

# 9) working directory is already set to /workspace
CMD ["bash"]


