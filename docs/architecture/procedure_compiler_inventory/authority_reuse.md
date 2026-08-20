# Authority reuse and compatibility decisions

PCPC adapters preserve these canonical owners:

| Concern | Canonical owner | PCPC role |
|---|---|---|
| Serialization and CID | `proof.formal_verification_contracts` | Import directly; add bounded PCPC schemas. |
| Authorization/effect admission | `control.control_contracts` and control service | Resolve declared effects into existing expected-effect/claim checks. |
| Semantic world state | `semantic_state.world_snapshot_contracts` and builder | Project bounded planning references; never replace. |
| Git repository state | Git and `analysis.repository_snapshot` | Bind exact commit/tree and observed deltas. |
| AST/dependency/semantic query | Existing analysis providers | Use allowlisted read operations. |
| Context/VOI/retry | `ContextCompiler` | Select/expand named content references. |
| Proof/test/adversarial evidence | Existing validation, verification, proof, and assurance packages | Consume independently admitted receipts. |
| Patch repair | Existing repair operator registries and autonomous repair engine | Invoke only approved bounded templates/typed holes. |
| Worktree/lease/fence/merge | Existing merge/coordination packages | Acquire and verify, never reimplement. |
| Task/intent state | DuckDB `IntentRepository`/`DatabaseTaskSource` through Quack | Store goals, subgoals, tasks, attempts, evidence, and current registry metadata. |
| History analytics | DuckLake projection | Non-authoritative append/replay only. |

Typed compatibility decisions:

- Adapt `AssuranceCampaignApi` to the requested adversarial-assurance role.
- Add a narrow incremental-sealer import facade only if direct leaf imports are
  unsuitable at an integration boundary.
- Do not integrate with `AdaptivePlanner` until its clean committed import is
  restored and focused tests pass.
- Treat the absent autonomous-meta-controller comparison baseline as
  unevaluable for dependent promotion metrics.
- Do not inherit or copy unrelated uncommitted route/lease/merge/catalog
  overlays from another checkout.
