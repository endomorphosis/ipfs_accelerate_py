# Adversarial Assurance Engine — Final Report (AAE-063)

**Status:** terminal current-tree qualification report (non-authoritative)  
**Board namespace:** `adversarial-assurance-engine-v1`  
**Task / goal:** `AAE-063` / `AAE-G090`  
**Evidence:** `aae/final-report@1`, `aae/current-tree-conformance@1`  
**Interface:** `AdversarialAssuranceEngine` current-tree qualification  
**Authority:** this report is **not** production-authoritative. It records
current-tree commits, reused modules, campaign surfaces, seal bindings, and
remaining limits. A high mutation score is not presented as proof of
correctness.

Related:

- Plan: [`ADVERSARIAL_ASSURANCE_ENGINE_PLAN.md`](./ADVERSARIAL_ASSURANCE_ENGINE_PLAN.md)
- Objectives: [`adversarial_assurance_engine.objectives.md`](./adversarial_assurance_engine.objectives.md)
- Operator guide: [`../guides/adversarial_assurance_engine.md`](../guides/adversarial_assurance_engine.md)
- Prerequisite release: [`adversarial_assurance_inventory/prerequisite_release.md`](./adversarial_assurance_inventory/prerequisite_release.md)

---

## 1. Bounded concluding claim

The system used semantically targeted counterfactual mutations to test whether
declared tests, proofs, policies, semantic summaries, and incremental seals
reject important incorrect behavior. Surviving mutants were classified as
assurance gaps, candidate remediations were evaluated against held-out
mutations, and accepted assurance-policy changes were promoted through a
reproducible, content-addressed qualification process.

**Explicit non-claim:** mutation score, detector agreement, and cache reuse do
not prove product correctness. Proof claims stop at properties actually encoded
and verified. Heuristic evidence is never treated as exact.

---

## 2. Exact commits

| Tree | Commit | Role |
| --- | --- | --- |
| controller | `4b4f57143cc6c0869e10316ad5a9f6927f39e624` | AAE-000–062 complete on `agent/adversarial-assurance-engine-v1` |
| `ipfs_datasets_py` | `38cfb624e617fc878e627c3ef66d92a4d8817e59` | released planning gitlink (AAE + IPS merge) |
| `ipfs_kit_py` | `2066e6fe671e89be4ae5e5172d055c937ad02135` | released planning gitlink (AAE + IPS merge) |
| `ipfs_accelerate_py/mcplusplus` | `96238cc9a86e69d224ab7b52d211a79ecf27b382` | released MCP++ pin |
| SCG program-complete | `485edc0871c55b0e2ef21d83bece9fa12c2c8d84` | PR #185, ancestor of this controller |
| SCG terminal launch | `9b7cf7f0447b0d6a85bfadd2d854dff9709d2b7c` | drained 3-lane receipt |
| IncrementalProofSealer | `5edc4569424183b37fd3b3e58aad2b22088a76a6` | 64/64 release, ancestor of this controller |

Inspected commits:

```text
4b4f57143cc6c0869e10316ad5a9f6927f39e624
38cfb624e617fc878e627c3ef66d92a4d8817e59
2066e6fe671e89be4ae5e5172d055c937ad02135
96238cc9a86e69d224ab7b52d211a79ecf27b382
485edc0871c55b0e2ef21d83bece9fa12c2c8d84
9b7cf7f0447b0d6a85bfadd2d854dff9709d2b7c
5edc4569424183b37fd3b3e58aad2b22088a76a6
```

---

## 3. Reused modules and operators

Reused modules (no new identity, sealer, scheduler, or MCP++ profile):

- canonical identity: `ipfs_datasets_py.logic.software_contracts.content`
- incremental sealing: `ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing`
- durable store: `ipfs_kit_py` DurableCoordinationStore / semantic-state roots
- SCG calibration and history stores already released on `origin/main`

Operators: the eleven deterministic bounded operator classes from AAE-014–020
(requirement, policy, proof, receipt, fixture, environment, semantic,
incremental, distributed-storage, compression, GUI/accessibility). CLI
operators from AAE-056–058 remain the only public command surface.

---

## 4. Counts, scores, survivors, vacuity, and gaps

Board closeout counts on this tree:

| Quantity | Value |
| --- | --- |
| generated tasks | 64 (`AAE-000`–`AAE-063`) |
| admitted / completed tasks | 63 before this fan-in; `AAE-063` is the terminal report |
| killed / survived campaign mutants | recorded in campaign fixtures under `test/fixtures/adversarial_assurance`; this report does not invent a composite kill rate |
| equivalent / invalid / inconclusive | classified per AAE-025–031 receipts; survivors remain assurance gaps, not silent passes |
| risk-weighted score | campaign-local; no board-wide score is promoted as correctness |
| high-risk survivors | must appear in survivor reports from AAE-030–031; zero is a target, not assumed |
| vacuity categories | unsatisfiable, unreachable, unconstrained, shadowed (AAE-026–029, AAE-055) |
| gap categories | product gap vs assurance gap, held-out residual, unspecified behavior |

Detection rates (selected/full-suite and proof/policy) are campaign-scoped.
This closeout does not collapse them into one percentage.

---

## 5. Incremental savings, cache reuse, cost

Incremental execution (AAE-044) and sealer reuse (AAE-062 / IPS release) are
the cost and cache path. Incremental savings and cache reuse are evidenced by
the IPS release receipt
`artifacts/agent_supervisor/incremental_proof_sealer/release_validation.json`
and focused AAE baselines under
`docs/architecture/adversarial_assurance_inventory/prerequisite_evidence`.
Cost numbers that were not re-measured on this exact HEAD are not restated as
current.

Focused baseline passes at AAE-006 release (`6de16ae56`):

| Suite | passed | failed | skipped |
| --- | ---: | ---: | ---: |
| datasets | 503 | 0 | 0 |
| accelerate | 796 | 0 | 0 |
| ipfs_kit_py | 284 | 0 | 0 |
| mcp_plus_plus | 73 | 0 | 0 |

---

## 6. Remediation, promotion, regression, overconstraint

Proposed remediations (AAE-032) are evaluated (AAE-033, AAE-046) against
held-out mutations. Promoted remediations require CAS policy compare-and-swap
(AAE-047). Rejected remediations stay in campaign history. Regressions and
overconstraint are first-class outcomes: a policy that blocks correct behavior
cannot be promoted. Unauthorized production policy change is forbidden (count
target: 0).

---

## 7. Seal, improvement, limits

Seal results: IncrementalProofSealer public APIs
(`IncrementalProofSealer`, `FullCheckpointSeal`, `DeltaSeal`,
`create_full_checkpoint`, `publish_full_checkpoint`, `build_delta_seal`,
`publish_delta_seal`) import from this tree. SCG terminal receipt is drained
(`drained=true`, three lanes, `completed_count=49`). Improved assurance is
limited to content-addressed, held-out, operator-authorized promotions.

Limits / remaining unspecified or unverified behavior:

- host isolation is not a full production sandbox
- ZK campaigns use released sealer APIs; missing circuits stay typed-unavailable
- GUI mutation is excluded; accessibility fixtures only
- enterprise/production next steps require a separate promotion authorization,
  a new incremental seal, and held-out evaluation on the promoted tree

---

## 8. Next steps

Exact enterprise/production next steps:

1. Re-run the focused pre-change matrix and four MCP++ conformance harnesses on
   the post-`AAE-063` commit.
2. Issue a new incremental seal over that commit; do not reuse the AAE-006
   receipt as production promotion evidence.
3. Require a separate signed promotion authorization; the AAE-006 operator
   gate authorizes runtime work, not production deployment.
4. Record remaining unspecified behavior as open gaps, not as silent passes.

Improvement of the assurance engine itself is out of scope for this closeout
except as bounded successor tasks after AAE-063.
