#!/bin/bash
set -eu
export PYTHONPATH=/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002:/home/barberb/lift_coding/.worktrees/ir-release-datasets-20261002
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /usr/bin/python3.12 -m benchmarks.agent_supervisor.container_coding.native_repository_finite_supervision \
  --output /home/barberb/lift_coding/artifacts/repository-preparation-profile-20261002/native-pinned-parent-06 \
  --python /usr/bin/python3.12 \
  --lean /home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean \
  --preparation-profile /home/barberb/lift_coding/artifacts/repository-preparation-profile-20261002/pinned-parent-profile.json
