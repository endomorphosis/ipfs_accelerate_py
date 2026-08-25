# CASF final qualification and residual-gap report

Machine report: `final_qualification_report.json`
Schema: `casf/qualification-report@1`
Report ID: `sha256:aef8c9036eb09c42e04269723265fb9fb869ffbec9091150fcd8ca12512c0237`
Machine-report raw SHA-256: `efbfe15e327ad44d2f6b97e351e5a190a1a5904b612aa7daf554dab71ba4fd46`

## Disposition

**Not qualified — quarantine recommended, not applied.** This repository-only
report is non-authoritative. It is not a qualification identity, completion
receipt, accepted gate decision, release authority, or production-state change.

The DuckDB + Quack core profile and the DuckLake + Quack profile are both
`not_qualified`. DuckLake remains non-authoritative and does not block the core
profile, but its missing receipts do block DuckLake promotion. Contention-free
operation is not qualified because no current-generation, fence-bound,
single-state-owner population attestation was admitted.

## Identity boundary

| Boundary | Revision | Tree | Status |
|---|---|---|---|
| Sealed program baseline | `84a056e41e48a81d4484be43840196578d6c87da` | `40f0771e77d394ac91d92cc1edb02f7860f6131b` | historical baseline |
| Qualification input | `5796d3f78b77b2b6c1c59a2b74c86020a0b141ae` | `14b36ca1f21bfd03dd4b88a7866a0c1a40059249` | pre-report component snapshot |
| Final merged tree | `null` | `null` | pending post-merge state-owner acceptance |

The input snapshot is an ancestor, not a claim about the report commit or the
eventual landing tree. The final revision, tree, generation, schema fingerprint,
qualification identity, validation receipt, acceptance receipt, and result
identity all remain null until accepted producers and an independent verifier
create them.

## Exact repository source bindings

| Path | Git blob | Raw SHA-256 |
|---|---|---|
| `docs/architecture/causal_event_federation_inventory/starting_tree.json` | `6f41ac8b1933b120cd3222bbd29668e48eb67194` | `8a4e2a658eb8c019e315afd28ea56ff956432e62dcc51fdf6e17a1c5239b83ed` |
| `docs/architecture/causal_event_federation_inventory/authorities.json` | `949044ab598f07ec11a0416dca8f5739f8e975fb` | `e787ce33bf647b5545548545c2a8f2d57de3be6ebb7478bb920490baa532a879` |
| `docs/architecture/causal_event_federation_inventory/capability_snapshot.json` | `0f8012a9dfb77bf9630df925b672f5a61a04eab3` | `48474ed9f70bc033968512cf23406f5679618f7655d26eb9e8dc447f1de1525e` |
| `docs/architecture/causal_event_federation_inventory/README.md` | `581bc6f9ed59fad19d900fcc030d8dcbd129f0b8` | `6749ac9d8fc46c2a388a4b8fe543ae630bfb1d24d717a0659b7b6b9e6a9d638b` |
| `docs/architecture/agent_supervisor_causal_event_federation.objectives.md` | `b4adc69f3f3dae288ae102daf6efbaeafd8a642f` | `ed6b2df4386c1ee52c90b228e3cfb96a059affbcd2235f3dc27e22be7011de08` |
| `docs/architecture/agent_supervisor_causal_event_federation.todo.md` | `4a8524af600c0db1c58dee5c44a5e25944582b07` | `a7fa2702e975714d4432c633ea9307c97d987db8b5683fb50af0f565f3b57408` |
| `scripts/validate_agent_supervisor_causal_event_federation_board.py` | `a685af0e15b7a7333fdd4ecd531478a4360bf633` | `1747b00a3e0844f52007cf544b7ab88ef13c06426d892c1d8ffb424df43503d4` |
| `config/agent_supervisor_causal_event_federation_scheduler.json` | `1ee6f72095ab171f37b655fc6bd89dd7f41d354a` | `708f8b00fbd5343fc7e1ca9eaf668e553992ebe89f9d6e8cbe2f1bdb58cf426f` |
| `ipfs_accelerate_py/agent_supervisor/federation/promotion.py` | `17ea2beb7e9a7f54ddf703a04beecd13bcdf8405` | `d2258f9685959ea2d6a89e03bb809c0876cbff4fd669169398c76498ed5128c3` |
| `test/api/causal_federation/test_promotion.py` | `36d293be2c7d3781237fb042e3f31790c778fe1a` | `953aab7f37ea49cb2ac687023e48489a517c1199a3122ec78fcd96bba1e68831` |
| `benchmarks/agent_supervisor/causal_event_federation/manifest.json` | `ed4682d69c60509f960f71023c6a38838fdf88f4` | `0d6280f6dc982dae824f24fe6de4ef245afa4f9e30b7ad18ceb48bea71ff3646` |

These bindings describe repository objects at the qualification input. They
prove byte identity and lineage only; they are not live or execution evidence.

## Benchmark suite

Suite ID: `sha256:aa916a4418b4345e90c75b7955f5d0bdefde657158c5fb4f2834ae6cfea3eb0b`

Suite projection: `edeb276d83713315864d38eb81dc09b519f5360e` / `1853e8297c44b7b5ee67f8a015fd67107670eb2a`

Component snapshot: `75f9487ff051ce5defd6171d7b41dd8127a0d59f` / `4e31236de005816686e68a336adb1a7fe679e6fa`

Suite manifest blob/raw SHA-256: `ed4682d69c60509f960f71023c6a38838fdf88f4` / `0d6280f6dc982dae824f24fe6de4ef245afa4f9e30b7ad18ceb48bea71ff3646`

Both suite snapshots are ancestors of the qualification input, and every bound
matrix, manifest, and runner blob is byte-identical there. This establishes a
current specification lineage, not benchmark execution or qualification.

| Task | Result schema | Availability | Execution | Metrics omitted | Result ref |
|---|---|---|---|---|---|
| CASF-038 | `casf/idle-benchmark@1` | unavailable | not_run | true | null |
| CASF-039 | `casf/parallel-benchmark@1` | unavailable | not_run | true | null |
| CASF-040 | `casf/load-benchmark@1` | unavailable | not_run | true | null |
| CASF-041 | `casf/token-benchmark@1` | unavailable | not_run | true | null |

The suite has no result artifacts, no metrics, and no scheduling,
qualification, completion, promotion, release, or DuckLake authority.

## Evidence disposition

- Verified: qualification-input repository bindings and benchmark specification bindings.
- Failed: none recorded; unexecuted gates are not reported as failures.
- Skipped: live runtime/database observation and the optional DuckLake live profile.
- Not run: idle, parallel, load, and token benchmark result schemas.
- Missing or unaccepted: current qualification identity, evidence bundle,
  promotion decision and validation, completion receipt, accepted CASF-030 and
  CASF-033 producer provenance, and CASF-035/CASF-036 decoders.

Every product result area is `not_qualified`, and every non-compensable safety
gate is `unverified` for the qualification input and pending final-tree
identity. This is evidence absence, not evidence that a gate failed.

## Residual gaps

1. **CASF-043-FINAL-TREE-ACCEPTANCE** (blocking_release): A registered state-owner producer and independent verifier must bind the actual merged revision and tree, schema, generation, policy, capability, task, attempt, assignment, worktree, lease, and fence to accepted validation and result identities. Reason codes: `accepted_current_generation_qualification_identity_unavailable`.
2. **CASF-043-LIVE-TYPED-QUACK-STATE-OWNER** (blocking_core): Admit and independently verify one authenticated typed Quack state owner, a current generation and schema fingerprint, fence-bound population, and a remote no-lost-wakeup event-wait path without direct DuckDB or file fallback. Reason codes: `accepted_current_tree_state_owner_attestation_unavailable, remote_no_lost_wakeup_qualification_unavailable`.
3. **CASF-043-CASF-030-PROVENANCE** (blocking_core): Produce a current-tree fixed-point receipt through an accepted producer with the full qualification identity and state-owner provenance. Reason codes: `missing:casf_030_accepted_producer_provenance, missing:casf_030_full_qualification_identity_binding, missing:casf_030_state_owner_provenance`.
4. **CASF-043-CASF-032-DUCKLAKE-PROVENANCE** (blocking_ducklake_profile): Produce current projection and recovery receipts through accepted producers with full qualification identity and state-owner provenance while retaining DuckLake non-authority. Reason codes: `missing:casf_032_accepted_producer_provenance, missing:casf_032_full_qualification_identity_binding, missing:casf_032_state_owner_provenance`.
5. **CASF-043-CASF-033-PROVENANCE** (blocking_core): Produce a current-tree drift report through an accepted producer with the full qualification identity and state-owner provenance. Reason codes: `missing:casf_033_accepted_producer_provenance, missing:casf_033_full_qualification_identity_binding, missing:casf_033_state_owner_provenance`.
6. **CASF-043-CASF-034-STATE-OWNER-CAPABILITY** (blocking_core): Attest the current typed state-owner capability and bind the accepted control audit to the exact qualification identity. Reason codes: `blocked:casf_034_current_state_owner_capability_unattested`.
7. **CASF-043-CASF-035-CONTROL-PARITY-DECODER** (blocking_core): Implement and independently validate the canonical control-parity report decoder before admitting CASF-035 evidence. Reason codes: `missing:casf_035_control_parity_report_decoder`.
8. **CASF-043-CASF-036-FORMAL-DECODER** (blocking_core): Implement and independently validate the canonical formal-model report decoder before admitting CASF-036 evidence. Reason codes: `missing:casf_036_formal_report_decoder`.
9. **CASF-043-CASF-037-LOCAL-QUALIFICATION** (blocking_core): Run and admit the upstream-reverified current-tree adversarial qualification; the repository-only unavailable report cannot substitute for it. Reason codes: `blocked:casf_037_local_qualification_unavailable`.
10. **CASF-043-CASF-038-IDLE-NOT-RUN** (blocking_core): Run the frozen idle profile through the admitted typed Quack live path and retain a current-tree content-addressed result. Reason codes: `unavailable:casf_038_live_not_run`.
11. **CASF-043-CASF-039-PARALLEL-NOT-RUN** (blocking_core): Run the frozen real-process twelve-supervisor profile and retain a current-tree content-addressed result satisfying every zero-tolerance gate. Reason codes: `unavailable:casf_039_live_not_run`.
12. **CASF-043-CASF-040-LOAD-NOT-RUN** (blocking_core): Run the frozen bounded 256-agent profile and retain a current-tree content-addressed result satisfying every zero-tolerance gate. Reason codes: `unavailable:casf_040_live_not_run`.
13. **CASF-043-CASF-041-TOKEN-NOT-RUN** (blocking_core): Run the frozen same-population baseline and twelve-supervisor comparison and retain current-tree result receipts meeting every target without reduced assurance. Reason codes: `unavailable:casf_041_live_not_run`.
14. **CASF-043-BENCHMARK-RESULT-ARTIFACTS** (blocking_core): Retain exact content-addressed result artifacts with non-null result references; metrics remain omitted until those runs occur. Reason codes: `benchmark_result_artifacts_absent`.
15. **CASF-043-CONJUNCTIVE-PROMOTION-DECISION** (blocking_release): Submit the complete exact-identity evidence bundle to the registered gate and independently validate an accepted decision; this report cannot create, apply, or authorize it. Reason codes: `accepted_conjunctive_promotion_decision_unavailable`.
16. **CASF-043-ROLLBACK-VERIFICATION** (blocking_release): Verify a fenced rollback path for only the CASF-043-owned artifacts against the pre-report component snapshot without rewriting history or applying production effects. Reason codes: `fenced_rollback_verification_unavailable`.

## Rollback boundary

Rollback is not authorized, executable, or applied. Its scope is only the
three CASF-043 owned artifacts, and its target is the pre-report input
`5796d3f78b77b2b6c1c59a2b74c86020a0b141ae` / `14b36ca1f21bfd03dd4b88a7866a0c1a40059249`. A verified, current-fence typed state-owner
decision is required; history rewriting is prohibited.

## Nonclaims

- This report is not a federation-completion receipt, accepted qualification identity, policy decision, release authority, or promotion decision.
- Repository bindings verify bytes and lineage only; they do not verify a live control plane, population, event transport, safety gate, benchmark result, or provider execution.
- No task-board status, source module, test name, process exit, quiet queue, model statement, historical receipt, metric, or DuckLake projection is current-tree qualification evidence.
- No direct DuckDB access, Quack-to-file fallback, DuckLake scheduling authority, model-created authority, model-created policy permission, or model-created completion is authorized.
- Not qualified and quarantine recommended do not assert that a safety gate failed, that quarantine was applied, or that production state changed.
- Exactly-once network delivery is not claimed; only separately verified fenced idempotent authoritative effects could have an exactly-once scope.
