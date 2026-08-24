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
- Bind that adapter to `AssuranceCampaignApi@1`,
  `AdversarialAssurancePublicApi@1`, its public schema, the exact baseline
  source/test blobs, and the pinned `ipfs_datasets_py` gitlink. The absence of
  an exact `AdversarialAssuranceEngine` class is re-probed; a similar name alone
  cannot silently change the disposition.
- Add a narrow incremental-sealer import facade only if direct leaf imports are
  unsuitable at an integration boundary. The leaf owner and its full/delta
  surfaces are separately blob- and symbol-bound, including both required
  sibling releases.
- Do not integrate with `AdaptivePlanner` until its clean committed import is
  restored and focused tests pass. The MCP contract catalog is now committed,
  so the former missing-catalog blocker is resolved. The current typed
  collection failure instead binds the undefined `HAMMER_TRACE_SCHEMA` reached
  in `multi_prover_router.py`; that failure is not a passing qualification.
- Keep the autonomous-meta-controller comparison baseline historically
  `missing`. The later current tree contains an exact source, symbol,
  interface/schema, and passing producer qualification, but that evidence is
  explicitly not admitted as the comparison baseline.
- Preserve the autonomy package, cognitive scheduler, and experience ledger as
  historically `missing`. The later current tree exactly qualifies the
  experience-ledger interface/schema and records the package and scheduler as
  source-present with typed interface gaps because neither declares a
  standalone versioned interface/schema identity. Keep the policy-distillation
  subsystem currently `missing`. Autonomous repair, bounded self-improvement,
  the supervisor token ledger, and formal-assurance CEGIS remain recorded only
  as non-equivalent neighboring mechanisms.
- Preserve the delta-retry daemon integration failure, default-provider route
  failures, two worktree test-import failures, process-fencing stub-contract
  drift, and legacy merge-queue failures as typed caveats. The two added
  worktree recovery tests pass, increasing that producer to 51 collected / 49
  passed / 2 failed without erasing either retained failure.
- Do not inherit or copy unrelated uncommitted route/lease/merge/catalog
  overlays from another checkout.

The materialization gate resolves these decisions mechanically. For each row it
checks the baseline commit/tree, historical and explicitly rebound current
source/test blobs, current `HEAD`, working-tree blob identity, declared
classes/functions, literal interface/schema constants, package manifest,
gitlinks, historical and current probes, and test-producer linkage. It
additionally requires each declared sibling checkout to be populated, clean,
and at the exact
current-tree gitlink, then executes all 22 producer commands. Exact current
pass/fail/error counts and literal typed failure fragments are sealed into
producer receipts and referenced by a per-authority receipt. A typed expected
failure preserves a caveat; it never converts that producer into passing
evidence. P0 admission therefore depends on fresh current-tree execution, not
the task board, generic test success, historical counts, or an inventory CID.
