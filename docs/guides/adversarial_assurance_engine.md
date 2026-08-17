# Adversarial Assurance Engine

Operator guide for the current-tree Adversarial Assurance Engine (AAE) on
`agent/adversarial-assurance-engine-v1`.

## What it is

AAE is a sealed 64-task program (`AAE-000`–`AAE-063`) that uses targeted
counterfactual mutations to test whether declared tests, proofs, policies,
semantic summaries, and incremental seals reject important incorrect
behavior. It does **not** prove product correctness.

Trust model and limitations: see
[`../architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md`](../architecture/ADVERSARIAL_ASSURANCE_ENGINE_REPORT.md).

## Pins

Released planning gitlinks (AAE-006 operator gate, pin generation 1):

- datasets `38cfb624e617fc878e627c3ef66d92a4d8817e59`
- kit `2066e6fe671e89be4ae5e5172d055c937ad02135`
- MCP++ `96238cc9a86e69d224ab7b52d211a79ecf27b382`

Do not rewrite sealed SCG plan pins. Do not treat planning revisions as
runtime completion evidence.

## Operator gate

`AAE-006` is operator-only. Workers cannot complete it. After the
prerequisite receipt is `completed`, launch still requires a single-use
admission file outside the repository:

```bash
export IPFS_ACCELERATE_AAE_LAUNCH_ADMISSION_PATH=/absolute/path/to/admission.json
python3 scripts/ops/agent_supervisor/adversarial_assurance_engine_scheduler.py \
  --repo-root . --config config/adversarial_assurance_engine_scheduler.json \
  preflight
```

The admission must bind exact controller HEAD, receipt CID, pin generation,
gitlinks, and a strictly increasing launch generation.

## Validate

```bash
python3 scripts/validate_adversarial_assurance_engine_board.py --check-all
python3 -m pytest -q test/api/adversarial_assurance/test_current_tree_conformance.py
```

Full closeout also reruns focused baselines, the four MCP++ harnesses, and
the ZK/seal campaign. Failures are gaps, not permission to weaken tests.

## Production

AAE runtime authorization is not production deployment. Promotion needs a
new incremental seal, held-out evaluation, and a separate signed
authorization. Zero unauthorized policy changes.
