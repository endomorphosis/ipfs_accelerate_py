# ASI-079 Integrated Analysis Completion-Gate Evidence Map

- Date: 2026-07-24
- Task: ASI-079
- Goal: ASI-G020 — Integrated analysis, caching, and ipfs_datasets_py offload
- Parent: ASI-G000
- Requirements: `189057730455837902155591890661235220962`, `184801846437522667882915494501685213497`, `206259342916458424196977899134352826879`
- Source gap fingerprint: `922f6a6ca41bed7be0423a4b89f18c8af7106437`
- Evidence obligation: `objective-work/v1/b22b3881227ebd6dc6aa41980459b59382f5561d`
- Todo vector: `b22b3881227ebd6d`
- Merge role: completion_gate
- Lifecycle: provisionally complete and supervisor-actionable

## Gap disposition

1. Producing-task closure is explicit. The parent evaluator requires the exact ASI-003, ASI-004, and ASI-007 population with successful terminal states in addition to `tasks_complete=True`.
2. Every submitted validation is authoritative to the decision. One failed, stale, malformed, or foreign-bound sibling invalidates the submission, and every one of the five literal criteria must have a fresh passing receipt.
3. Coverage is exact and current-tree-bound. Every criterion has one concrete implementation path, validation path, and `validation_receipt_id` matching that criterion's submitted receipt.
4. Live integration is operationally witnessed. The content-addressed cohort binds a live objective/planning analyzer result with a non-empty AST index identity and bounded retrieval response, then independently binds exact-tree reuse, keyed miss collapse, active-policy provider degradation, and measured reuse/stale-authority thresholds.
5. Analyzer health is explicit and separate. A provider capability, provider result, pipeline packet, cache entry, objective document, or operational cohort cannot stand in for a `healthy=True`, `safe_for_completion_reasoning=True` analyzer record bound to the exact repository, tree, objective, analyzer, and configuration.
6. Exhaustion is configured and independent. The parent fixes the required count at two and requires unique member, channel, and receipt identities; fresh exhaustive mode; explicit member health and completion safety; and the exact parent binding. A caller cannot lower the trusted count.

## Mandatory criterion map

| Mandatory acceptance criterion | Implementation witness | Fresh validation route |
| --- | --- | --- |
| Existing analysis cache, AST index, and retrieval contracts are used in the live objective/planning path | `AnalysisPipeline._build_context`, `AnalysisCacheCoordinator`, `run_low_backlog_analysis`, and the nomination-only `ast_index_id`/`retrieval_response_id` projection | `test_incremental_ast_index_is_projected_into_live_retrieval`, `test_live_objective_planner_receives_ast_index_and_retrieval_cache_context`, and `test_g020_integrated_completion_requires_live_producer_cohort_and_closed_gate` |
| expensive identical misses collapse across lanes | `SingleFlightCollapseEvidence` and the keyed sync/async/mixed coordinator | pipeline and coordinator identical-miss, unrelated-key, failure-cleanup, and mixed-facade tests |
| stale or negative records never become completion evidence | exact seven-dimension lookup, packet-artifact rebinding, negative TTL/no-store policy, and follower validation | authority-dimension, expiry, corruption, missing-artifact, negative-record, and joined-validator tests |
| optional datasets capabilities degrade explicitly | `IpfsDatasetsProviderDegradationEvidence.proved_requirement_ids_for` on the active request and full policy; provider outputs remain completion-unsafe | missing/disabled/unsupported/unhealthy/timeout/failure/malformed/cancelled provider tests and the G020 provider-boundary test |
| repeated fixtures achieve at least 70 percent cache reuse with zero stale authoritative hits. | `AnalysisPipelineMetrics` plus the threshold checks in `IntegratedAnalysisCompletionEvidence` | repeated 9/10 reuse fixture and the full 8/10 G020 producer cohort |

The G020 completion-matrix test proves the mandatory active-to-provisional transition and a separate hypothetical provisional-to-verified evaluation only after every input is valid. It fails closed for an incomplete producing-task population, missing required child, provider output substituted for local analyzer health, any extra failed validation, duplicate exhaustive receipt, caller-lowered quorum count, foreign binding, and content-identity tampering. Existing G094/G095/G096 tests retain their independent child proof populations.

## Validation observation

The required current-tree command is:

```text
python -m pytest test/api/test_agent_supervisor_analysis_pipeline.py test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py test/api/test_agent_supervisor_cache_coordinator.py -q
```

Fresh ASI-079 candidate-tree observation on 2026-07-24: **91 passed,
0 failed**. The same command is the mandatory submission validation and must
remain passing after this audit-index update; the submitting runner's
post-update result is the current-tree receipt.

This file is an audit index, not a completion receipt. It claims no final post-change tree, analyzer-health, or exhaustion-quorum execution. ASI-G020 and ASI-G000 remain supervisor-actionable until the supervisor ingests fresh passing current-tree receipts for all five criteria, fully bound safe analyzer health, the configured two-member independent healthy exhaustive quorum, and verified fresh G094/G095/G096 descendants, then performs a separate evaluation after provisional completion.
