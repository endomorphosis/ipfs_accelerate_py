# PGIR-110 R1-R6 controlled campaign

This directory is the immutable `IRControlledCampaign@1` produced for
PGIR-110. It binds every R1-R6 arm and seed to the PGIR-014
`IRCampaignInputRoot@1` freeze, the same `RESULT(PGIR-012)` heldouts, and one
closed seed policy.

The campaign verifies successfully but its decision is deliberately `no_go`:
the freeze reports zero rights-admitted training rows, an unmaterialized
corpus, incomplete required holdouts, and a historical semantic baseline that
is not currently qualified. Consequently every arm/seed lease remains
ungranted, no weights are written, hidden tests stay sealed, and no candidate
is promoted. This is an integrity success and an execution denial, not a
fabricated training result.

Run the independent verifier from the repository root:

```bash
python data/agent_supervisor/proof_grounded_ir_learning/experiments/verify_campaign.py
```

Rebuild replay is write-once by default:

```bash
python data/agent_supervisor/proof_grounded_ir_learning/experiments/build_campaign.py
```

The builder refuses to replace different bytes. Any admitted replacement must
use a new task revision and a separately located campaign whose freeze
`previous_root_cid` points at a superseding PGIR-014 root.

Arm meanings:

- `R1`: deterministic compiler/decompiler baseline. Historical
  `RESULT(PGIR-023)` fixture metrics are referenced by CID only and remain
  not currently qualified.
- `R2`: token-class cross-entropy only.
- `R3`: token-class CE plus normalized cosine.
- `R4`: supervised contrastive with false-negative filtering.
- `R5`: full multi-task `IRLossConfiguration@1` mix.
- `R6`: proof-grounded curriculum. Proof stays a nondifferentiable
  label/ranking/curriculum signal.

Learned arms use seeds `32`, `33`, and `34`. R1 uses the deterministic seed
`0`. Every N1-N8, latency, resource, and target-attainment metric is reported
with an explicit unavailable/unsupported reason, zero denominators, and no
invented confidence interval.

Key artifacts:

- `recipe.json`: compact closed campaign recipe.
- `heldouts.json` and `seeds.json`: identical heldout and seed identities.
- `checkpoints/`: one content-addressed manifest per arm/seed.
- `evaluations/`: actual metric statuses, reasons, and failures.
- `comparison.json`: paired statistical report with no winner.
- `receipts/`: admission, lease, resource, proof, training, evaluation, and
  reducer CAS evidence.
- `result.json`: `RESULT(PGIR-110)`.
