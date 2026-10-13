# Legal autoformalization bounded inventory bootstrap

This is an execution specialization of the full plan. Exactly three evidence-only tasks may run. Completing this board does not mark the original 60-task plan or unresolved corpus identity work complete. Existing source repositories are read-only.

All three inventory tasks use the native deterministic-only declared-validation-plan. Their commands create missing evidence once and independently validate retained evidence on every later invocation. No model provider is dispatched by these tasks.

## LAC-001 Inventory current source authorities

- Status: completed
- Completion: auto
- Is schedulable: true
- Priority: P0
- Goal id: LAC-G013
- Depends on:
- Track: inventory
- Provider role: deterministic-only
- Context budget tokens: 0
- Objective: Read the exact required repositories and current effective source files; record before/after identity, dirty overlays and stale/current distinctions. The orchestrator already created the isolated branch. Do not create branches or modify source. Run python3 scripts/legal_autoformalization_bootstrap.py materialize --task LAC-001; preserve existing evidence and validate it without rewriting.
- Outputs: data/legal_autoformalization_convergence/LAC-001/result.json
- Predicted files: data/legal_autoformalization_convergence/LAC-001/result.json
- Validation: python3 scripts/legal_autoformalization_bootstrap.py materialize --task LAC-001
- Acceptance: Validator returns valid true; immutable input bindings and numeric facts independently verify; all missing evidence and original acceptance limits stay explicit.
- Existing source policy: Read-only source roots from config/legal_autoformalization_bootstrap_sources.json; protected scripts/config/full plan cannot be edited.
- Allowed effects: Write only data/legal_autoformalization_convergence/LAC-001/result.json and supervisor-owned attempt records. No model/training/provider calls beyond the supervisor implementation role; no nested agents, external publication, source edits or lease manipulation.
- Resource class: inventory-cpu-1-memory-1024
- Board namespace: legal-autoformalization-convergence-bootstrap-20261013
- Bundle: lac/bootstrap/lac-001
- Parallel lane: lac-001
- Conflict policy: Disjoint result paths; existing code immutable; serialized supervisor merge queue.

## LAC-002 Reconcile retained training lineage

- Status: todo
- Completion: auto
- Is schedulable: true
- Priority: P0
- Goal id: LAC-G012
- Depends on:
- Track: inventory
- Provider role: deterministic-only
- Context budget tokens: 0
- Objective: Read historical and recent run indexes and newer retained evidence. Preserve executed/replayed/rejected/selected/resume/exposure distinctions. Missing identities remain null plus gaps. Do not load checkpoints or execute models. Run python3 scripts/legal_autoformalization_bootstrap.py materialize --task LAC-002; preserve existing evidence and validate it without rewriting.
- Outputs: data/legal_autoformalization_convergence/LAC-002/result.json
- Predicted files: data/legal_autoformalization_convergence/LAC-002/result.json
- Validation: python3 scripts/legal_autoformalization_bootstrap.py materialize --task LAC-002
- Acceptance: Validator returns valid true; immutable input bindings and numeric facts independently verify; all missing evidence and original acceptance limits stay explicit.
- Existing source policy: Read-only source roots from config/legal_autoformalization_bootstrap_sources.json; protected scripts/config/full plan cannot be edited.
- Allowed effects: Write only data/legal_autoformalization_convergence/LAC-002/result.json and supervisor-owned attempt records. No model/training/provider calls beyond the supervisor implementation role; no nested agents, external publication, source edits or lease manipulation.
- Resource class: inventory-cpu-1-memory-1024
- Board namespace: legal-autoformalization-convergence-bootstrap-20261013
- Bundle: lac/bootstrap/lac-002
- Parallel lane: lac-002
- Conflict policy: Disjoint result paths; existing code immutable; serialized supervisor merge queue.

## LAC-003 Reconcile span and entity cache populations

- Status: todo
- Completion: auto
- Is schedulable: true
- Priority: P0
- Goal id: LAC-G011
- Depends on:
- Track: inventory
- Provider role: deterministic-only
- Context budget tokens: 0
- Objective: Verify machine-source count equations and pinned cache references. Full outside-target identities and ordinal recovery remain incomplete; report original task acceptance partial. Do not rescan the corpus, promote aliases or download weights. Run python3 scripts/legal_autoformalization_bootstrap.py materialize --task LAC-003; preserve existing evidence and validate it without rewriting.
- Outputs: data/legal_autoformalization_convergence/LAC-003/result.json
- Predicted files: data/legal_autoformalization_convergence/LAC-003/result.json
- Validation: python3 scripts/legal_autoformalization_bootstrap.py materialize --task LAC-003
- Acceptance: Validator returns valid true; immutable input bindings and numeric facts independently verify; all missing evidence and original acceptance limits stay explicit.
- Existing source policy: Read-only source roots from config/legal_autoformalization_bootstrap_sources.json; protected scripts/config/full plan cannot be edited.
- Allowed effects: Write only data/legal_autoformalization_convergence/LAC-003/result.json and supervisor-owned attempt records. No model/training/provider calls beyond the supervisor implementation role; no nested agents, external publication, source edits or lease manipulation.
- Resource class: inventory-cpu-1-memory-1024
- Board namespace: legal-autoformalization-convergence-bootstrap-20261013
- Bundle: lac/bootstrap/lac-003
- Parallel lane: lac-003
- Conflict policy: Disjoint result paths; existing code immutable; serialized supervisor merge queue.
