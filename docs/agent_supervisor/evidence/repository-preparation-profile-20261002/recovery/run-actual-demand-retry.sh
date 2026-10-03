#!/bin/bash
set -eu
export PYTHONPATH=/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002:/home/barberb/lift_coding/.worktrees/ir-release-datasets-20261002
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CODEBASE384_CHECKPOINT=/home/barberb/lift_coding/artifacts/distributed384-20261001/run-01/security_ir/coordinator/checkpoints/2ca38dfcc05536315fc3e2c0647b710b930ef4066b474061a7b4e5bfb9a258c5.json
export CODEBASE384_EMBEDDING_SNAPSHOT=/home/barberb/.cache/huggingface/hub/models--thenlper--gte-small/snapshots/17e1f347d17fe144873b1201da91788898c639cd
export PATH=/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin:$PATH
export RPI_PROFILE_EVIDENCE=/home/barberb/lift_coding/artifacts/repository-preparation-recovery-20261002/actual-demand-retry
export IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB=/home/barberb/lift_coding/artifacts/repository-preparation-recovery-20261002/actual-demand-retry-seal.duckdb
exec /usr/bin/python3.12 -m pytest -q test/integration/test_repository_preparation_profile.py::test_independent_index_contract_and_demanded_proof_preserve_inventory \
 --basetemp=/home/barberb/lift_coding/artifacts/repository-preparation-recovery-20261002/actual-demand-retry-fixture \
 --junitxml=/home/barberb/lift_coding/artifacts/repository-preparation-recovery-20261002/actual-demand-retry.xml
