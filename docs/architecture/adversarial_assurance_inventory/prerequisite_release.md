# AAE-006 prerequisite release

Operator release of the Adversarial Assurance Engine prerequisite gate.

## Upstream programs

- Semantic Compression Governor: program-complete on
  `485edc087` (GitHub PR #185, already on `origin/main` for accelerate,
  datasets, and kit). A genuine three-lane drained terminal receipt was
  captured at launch commit
  `9b7cf7f0447b0d6a85bfadd2d854dff9709d2b7c` without rewriting sealed
  SCG plan pins. Official SCG worktree remains at `485edc087`.
- IncrementalProofSealer: 64/64 complete. Canonical release receipt is
  `artifacts/agent_supervisor/incremental_proof_sealer/release_validation.json`.
  Public full-checkpoint and delta APIs import from the current AAE tree.
  IPS release commit `5edc45694` is an ancestor of this controller.

## Pins

Released planning gitlinks equal nested HEADs:

- datasets `38cfb624e617fc878e627c3ef66d92a4d8817e59`
- kit `2066e6fe671e89be4ae5e5172d055c937ad02135`
- MCP++ `96238cc9a86e69d224ab7b52d211a79ecf27b382`

## Evidence

Copied under `docs/architecture/adversarial_assurance_inventory/prerequisite_evidence`
plus the canonical IPS release receipt. Focused baselines were executed
on this tree with network disabled.

## Authorization

Only `did:key:z6Mku1TT7TcoD2VksFwNmYGNpE1zprQMmXsT3tz39BzhVdsy` may complete
AAE-006. A separate single-use launch admission is required after this
gate commit before the supervisor may spawn runtime lanes.
