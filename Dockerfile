# Sealed training image — backend-agnostic remote execution (docs/remote_execution.md).
# Runs identically on SkyPilot (any cloud), a bare GPU box, or local docker.
# State NEVER lives in the container: checkpoints/metrics rendezvous through
# W&B; results are sha-scoped and reproducible from the code alone.
#
#   docker build -t mbrl-curvature:$(git rev-parse --short HEAD) .
#   docker run --gpus all -e WANDB_API_KEY mbrl-curvature:<sha> \
#       +experiment=champion env=halfcheetah seed=0
FROM python:3.11-slim

# mujoco needs libgl/osmesa for headless rendering; git for the sha provenance
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 libosmesa6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY .git ./.git
RUN pip install --no-cache-dir --no-deps -e .

# non-root; results/checkpoints live on a volume (or evaporate — W&B holds
# the durable copies via checkpoint.push_wandb)
RUN useradd -m runner && chown -R runner /app
USER runner
ENV MUJOCO_GL=osmesa PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "scripts/train.py", "checkpoint.resume=auto"]
