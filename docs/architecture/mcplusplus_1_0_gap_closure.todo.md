# MCP++ 1.0 Gap-Closure Taskboard

Consumable by `ipfs_accelerate_py.agent_supervisor` with task prefix `MCPP-`.

Companion artifacts:

- plan: `docs/architecture/MCPPLUSPLUS_1_0_GAP_CLOSURE_PLAN.md`
- objective heap: `docs/architecture/mcplusplus_1_0_gap_closure.objectives.md`
- scheduler: `config/agent_supervisor_mcplusplus_1_0_gap_closure_scheduler.json`
- validator: `scripts/validate_mcplusplus_1_0_gap_closure_board.py`

This is a sealed first projection with 84 append-only tasks (`MCPP-000`
through `MCPP-083`) over 18 goals. `MCPP-000` is the completed planning
seal. Exactly `MCPP-001` is initially dependency-ready.

Implementation tasks may change nested `ipfs_accelerate_py/mcplusplus`,
`ipfs_datasets_py`, and `ipfs_kit_py` only from isolated worktrees. SwissKnife
work uses the checkout discovered in MCPP-001. Never discard operator
overlays. Never force-push.

## MCPP-000 Seal the MCP++ 1.0 gap-closure control program

- Status: completed
- Completion: manual
- Completion evidence: plan, objective heap, taskboard, scheduler, validator, and parser test present
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: control
- Depends on: 
- Goal id: MCPP-G000
- Outputs: docs/architecture/MCPPLUSPLUS_1_0_GAP_CLOSURE_PLAN.md, docs/architecture/mcplusplus_1_0_gap_closure.objectives.md, docs/architecture/mcplusplus_1_0_gap_closure.todo.md, config/agent_supervisor_mcplusplus_1_0_gap_closure_scheduler.json, scripts/validate_mcplusplus_1_0_gap_closure_board.py, test/api/test_mcplusplus_1_0_gap_closure_board.py
- Validation: python scripts/validate_mcplusplus_1_0_gap_closure_board.py --check-all && python -m pytest -q test/api/test_mcplusplus_1_0_gap_closure_board.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/control
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 24000
- Implementation timeout seconds: 1800
- Predicted files: docs/architecture/MCPPLUSPLUS_1_0_GAP_CLOSURE_PLAN.md, docs/architecture/mcplusplus_1_0_gap_closure.objectives.md, docs/architecture/mcplusplus_1_0_gap_closure.todo.md, config/agent_supervisor_mcplusplus_1_0_gap_closure_scheduler.json, scripts/validate_mcplusplus_1_0_gap_closure_board.py, test/api/test_mcplusplus_1_0_gap_closure_board.py
- Interfaces: MCPPlusPlusGapClosurePlan@1
- Allow concurrent with: 
- Conflict policy: These control artifacts are protected after this task; later tasks may not rewrite task identities, dependencies, or acceptance.
- Preconditions: The accelerate branch is isolated enough to add new control files without discarding dirty overlays.
- Effects: A parseable, acyclic, six-lane, fail-closed implementation program is available.
- Evidence subset: plan, goals, tasks, scheduler, validator, parser test
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Validator proves 84 unique tasks, 18 unique goals, acyclicity, protected ownership, scheduler consistency, and exactly MCPP-001 ready after this completion.
- Embedding query: seal mcp plus plus supervisor goals tasks parallel scheduler

## MCPP-001 Preserve dirty overlays and create codex/mcplusplus-1.0-gap-closure branches

- Status: completed
- Completion: manual
- Completion evidence: docs/reports/mcplusplus-1.0-gap-closure/baseline/repository-forest.json and dirty-overlay.md; json.tool validation passed; operator overlays preserved; SwissKnife origin discovered via git remote -v
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-forest
- Depends on: MCPP-000
- Goal id: MCPP-G010
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/repository-forest.json, docs/reports/mcplusplus-1.0-gap-closure/baseline/dirty-overlay.md
- Validation: python -m json.tool docs/reports/mcplusplus-1.0-gap-closure/baseline/repository-forest.json > /dev/null
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-forest
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-small
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 3600
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline/repository-forest.json, docs/reports/mcplusplus-1.0-gap-closure/baseline/dirty-overlay.md
- Interfaces: RepositoryForestDescriptor@1, DirtyOverlayInventory@1
- Allow concurrent with: 
- Conflict policy: Record only; do not reset, stash-drop, checkout, or force-push any operator worktree.
- Preconditions: Operator checkouts may be dirty. Isolated worktrees will be used for later implementation.
- Effects: Every in-scope checkout has recorded remotes, HEAD, dirty paths, and a created or already-present program branch. The newer Mcp-Plus-Plus tree is bound as spec authority.
- Evidence subset: SHAs, remotes, dirty paths, branch names, submodule drift between lift_coding/Mcp-Plus-Plus and ipfs_accelerate_py/mcplusplus
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: No uncommitted file is deleted or overwritten. Forest JSON lists lift_coding, Mcp-Plus-Plus, accelerate, datasets, kit, SwissKnife remotes and SHAs. SwissKnife remote is discovered, not invented. Program branch exists or is recorded as already present.
- Embedding query: preserve dirty overlay create branch git forest submodule drift

## MCPP-002 Baseline Mcp-Plus-Plus Python tests and coverage

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-python
- Depends on: MCPP-001
- Goal id: MCPP-G010
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/mcpplusplus-python.json
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py --maxfail=1
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-python
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-medium
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline/mcpplusplus-python.json
- Interfaces: BaselineReceipt@1
- Allow concurrent with: MCPP-003, MCPP-004, MCPP-005, MCPP-006, MCPP-007, MCPP-008, MCPP-009
- Conflict policy: Write only the new receipt file; do not change validators to make the baseline green.
- Preconditions: Python 3 and pytest from tests-py/requirements.txt are available in the worktree.
- Effects: Exact command, SHA, pass/fail, count, coverage, skips, and environment are recorded.
- Evidence subset: pytest output, coverage if collectible, skipped tests
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Receipt contains command, env, commit SHA, pass/fail, test count, coverage or explicit unavailable, and skipped names. Failures are recorded, not hidden.
- Embedding query: pytest python mcpplusplus baseline coverage

## MCPP-003 Baseline Mcp-Plus-Plus TypeScript tests and coverage

- Status: completed
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-typescript
- Depends on: MCPP-001
- Goal id: MCPP-G010
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/mcpplusplus-typescript.json
- Validation: cd ipfs_accelerate_py/mcplusplus/tests-ts && npm test
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-typescript
- Parallel lane: mcpp-lane-state
- Resource class: cpu-medium
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline/mcpplusplus-typescript.json
- Interfaces: BaselineReceipt@1
- Allow concurrent with: MCPP-002, MCPP-004, MCPP-005, MCPP-006, MCPP-007, MCPP-008, MCPP-009
- Conflict policy: Write only the new receipt file.
- Preconditions: Node and the tests-ts package manifest are available.
- Effects: Exact vitest command, SHA, pass/fail, count, coverage, skips recorded.
- Evidence subset: vitest output, coverage-summary if present
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Receipt binds SHA and command. Disabled tests such as comprehensive.test.ts.disabled are listed as skipped, not passing.
- Embedding query: vitest typescript mcpplusplus baseline

## MCPP-004 Baseline Mcp-Plus-Plus Go tests

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-go
- Depends on: MCPP-001
- Goal id: MCPP-G010
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/mcpplusplus-go.json
- Validation: cd ipfs_accelerate_py/mcplusplus/tests-go && go test ./...
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-go
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-medium
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline/mcpplusplus-go.json
- Interfaces: BaselineReceipt@1
- Allow concurrent with: MCPP-002, MCPP-003, MCPP-005, MCPP-006, MCPP-007, MCPP-008, MCPP-009
- Conflict policy: Write only the new receipt file.
- Preconditions: Go toolchain is available or the receipt records the exact blocker.
- Effects: Exact go test command, SHA, pass/fail, count recorded.
- Evidence subset: go test output, coverage.out if produced
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Receipt is honest about missing toolchain. Existing coverage.html is not treated as current evidence.
- Embedding query: go test mcpplusplus baseline

## MCPP-005 Baseline Mcp-Plus-Plus Rust tests

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-rust
- Depends on: MCPP-001
- Goal id: MCPP-G010
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/mcpplusplus-rust.json
- Validation: cd ipfs_accelerate_py/mcplusplus/tests-rs && cargo test
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-rust
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-medium
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline/mcpplusplus-rust.json
- Interfaces: BaselineReceipt@1
- Allow concurrent with: MCPP-002, MCPP-003, MCPP-004, MCPP-006, MCPP-007, MCPP-008, MCPP-009
- Conflict policy: Write only the new receipt file.
- Preconditions: Cargo is available or the receipt records the exact blocker.
- Effects: Exact cargo test command, SHA, pass/fail, count recorded.
- Evidence subset: cargo test output, lcov if produced
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: COVERAGE_100_PERCENT_ACHIEVED.md is cited only as a stale document, never as the measured result.
- Embedding query: cargo test rust mcpplusplus baseline

## MCPP-006 Baseline ipfs_accelerate MCP++ runtime tests

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-accelerate
- Depends on: MCPP-001
- Goal id: MCPP-G010
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/ipfs-accelerate-mcplusplus.json
- Validation: python -m pytest -q ipfs_accelerate_py/mcp/tests test/api -k mcplusplus --maxfail=1
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-accelerate
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-large
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline/ipfs-accelerate-mcplusplus.json
- Interfaces: BaselineReceipt@1
- Allow concurrent with: MCPP-002, MCPP-003, MCPP-004, MCPP-005, MCPP-007, MCPP-008, MCPP-009
- Conflict policy: Do not modify dirty supervisor files to obtain a greener baseline.
- Preconditions: Accelerate worktree includes current dirty overlay as read-only observation plus HEAD SHA.
- Effects: Runtime MCP++ tests are inventoried with pass/fail and exclusions.
- Evidence subset: pytest collection and results for mcpplusplus and mcp_server
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Receipt lists collected tests, outcomes, and that operator dirty files were not discarded.
- Embedding query: ipfs accelerate mcpplusplus runtime baseline

## MCPP-007 Baseline ipfs_datasets MCP++ runtime tests

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-datasets
- Depends on: MCPP-001
- Goal id: MCPP-G010
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/ipfs-datasets-mcplusplus.json
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/mcp_server -k mcplusplus --maxfail=1
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-datasets
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-medium
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline/ipfs-datasets-mcplusplus.json
- Interfaces: BaselineReceipt@1
- Allow concurrent with: MCPP-002, MCPP-003, MCPP-004, MCPP-005, MCPP-006, MCPP-008, MCPP-009
- Conflict policy: Do not delete untracked datasets MCP++ P2P or UI-IR files.
- Preconditions: Datasets gitlink is initialized.
- Effects: Datasets MCP++ tests and untracked candidate files are inventoried.
- Evidence subset: pytest results, untracked mcplusplus paths
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Receipt records SHA, dirty overlay, and test outcomes without promoting untracked files to supported API.
- Embedding query: ipfs datasets mcplusplus p2p baseline

## MCPP-008 Baseline ipfs_kit MCP++ and UCAN admission tests

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-kit
- Depends on: MCPP-001
- Goal id: MCPP-G010
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/ipfs-kit-mcplusplus.json
- Validation: cd ipfs_kit_py && python -m pytest -q tests -k 'ucan or mcplusplus or profile' --maxfail=1
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-kit
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-medium
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline/ipfs-kit-mcplusplus.json
- Interfaces: BaselineReceipt@1
- Allow concurrent with: MCPP-002, MCPP-003, MCPP-004, MCPP-005, MCPP-006, MCPP-007, MCPP-009
- Conflict policy: Kit checkout is clean; do not pull origin/main as a side effect of baselining.
- Preconditions: Kit gitlink is initialized at the bound SHA.
- Effects: Kit MCP++/UCAN/Profile D tests are inventoried.
- Evidence subset: pytest results, KITA-G080 related tests
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Receipt records SHA 5a7a2df8 or the live bound SHA, command, and outcomes. Behind-origin status is noted, not auto-fast-forwarded.
- Embedding query: ipfs kit ucan profile d baseline

## MCPP-009 Baseline SwissKnife MCP++ tests from the discovered checkout

- Status: completed
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-swissknife
- Depends on: MCPP-001
- Goal id: MCPP-G010
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/swissknife-mcplusplus.json
- Validation: test -f docs/reports/mcplusplus-1.0-gap-closure/baseline/swissknife-mcplusplus.json
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-swissknife
- Parallel lane: mcpp-lane-state
- Resource class: cpu-large
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline/swissknife-mcplusplus.json
- Interfaces: BaselineReceipt@1
- Allow concurrent with: MCPP-002, MCPP-003, MCPP-004, MCPP-005, MCPP-006, MCPP-007, MCPP-008
- Conflict policy: Use the checkout discovered by git remote -v. Do not invent a URL. Do not discard the two dirty SwissKnife paths.
- Preconditions: MCPP-001 recorded the SwissKnife path and origin https://github.com/endomorphosis/swissknife.
- Effects: SwissKnife MCP++ unit/integration commands and results are recorded.
- Evidence subset: package.json test scripts, test/mcp-plus-plus, src/auth/ucan-auth.ts
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Receipt includes discovered remote, SHA 26f06277 or live bound SHA, command, and dirty-path preservation.
- Embedding query: swissknife mcp-plus-plus ucan baseline

## MCPP-010 Verify official MCP 2026-07-28 and A2A extension conventions from primary sources

- Status: completed
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-official-specs
- Depends on: MCPP-001
- Goal id: MCPP-G010
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/official-mcp-a2a.md
- Validation: test -s docs/reports/mcplusplus-1.0-gap-closure/baseline/official-mcp-a2a.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-official-specs
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-small
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline/official-mcp-a2a.md
- Interfaces: OfficialSpecNote@1
- Allow concurrent with: MCPP-002, MCPP-003, MCPP-004, MCPP-005, MCPP-011
- Conflict policy: Cite primary URLs only. Do not change bindings in this task.
- Preconditions: Network access to modelcontextprotocol.io and a2a-protocol.org or a recorded airgap blocker.
- Effects: A dated note records current MCP lifecycle, Tasks, extensions, initialize deprecation, and A2A extension URI convention.
- Evidence subset: MCP 2026-07-28 spec, A2A specification extensions section
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Note states that current MCP is not initialize-based, quotes the official A2A extension identifier rule, and either confirms https://mcplusplus.io/extensions/execution/v1 or records the verified substitute. Alias io.mcplusplus.execution@1 is documented.
- Embedding query: mcp 2026-07-28 initialize removed a2a extension uri

## MCPP-011 Inventory Profiles A-H normative versus implemented versus structural-only

- Status: completed
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-profiles
- Depends on: MCPP-001
- Goal id: MCPP-G010
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/profiles-a-h-inventory.md
- Validation: test -s docs/reports/mcplusplus-1.0-gap-closure/baseline/profiles-a-h-inventory.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-profiles
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/baseline/profiles-a-h-inventory.md
- Interfaces: ProfileInventory@1
- Allow concurrent with: MCPP-002, MCPP-010
- Conflict policy: Classification must not use schema field presence as implemented.
- Preconditions: Profile specs and validators are readable.
- Effects: Each profile is split into normative, guidance, implemented, structural-only, cryptographic, runtime-specific, missing, and contradictory docs.
- Evidence subset: mcp++-profiles-draft.md and chapter specs, validators, runtime adapters
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Profile C cryptographic enforcement is classified structural-only unless a real verifier is found. Contradictory 100-percent coverage docs are listed.
- Embedding query: profile a b c d e f g h inventory structural cryptographic

## MCPP-012 Create the requirement-to-evidence traceability matrix

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline-matrix
- Depends on: MCPP-002, MCPP-003, MCPP-004, MCPP-005, MCPP-006, MCPP-007, MCPP-008, MCPP-009, MCPP-010, MCPP-011
- Goal id: MCPP-G010
- Outputs: ipfs_accelerate_py/mcplusplus/docs/roadmap/mcplusplus-1.0-gap-closure.md
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/roadmap/mcplusplus-1.0-gap-closure.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/baseline-matrix
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-small
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/roadmap/mcplusplus-1.0-gap-closure.md
- Interfaces: RequirementTraceRow@1
- Allow concurrent with: 
- Conflict policy: Own the new roadmap file only.
- Preconditions: Baseline receipts and profile inventory exist.
- Effects: Every normative requirement maps to spec section, schema, validator, positive vector, negative vector, runtime, integration test, status, and evidence path.
- Evidence subset: baseline receipts, profile inventory, official spec note
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: No row is implemented merely because a schema accepts its fields. Status values are exactly implemented, partial, structural-only, missing, or blocked.
- Embedding query: traceability matrix normative requirement schema validator vector

## MCPP-013 Record ADR for spec versus runtime ownership

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adr-spec-runtime-ownership
- Depends on: MCPP-012
- Goal id: MCPP-G020
- Outputs: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0001-spec-runtime-ownership.md
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0001-spec-runtime-ownership.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/adr-spec-runtime-ownership
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 22000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0001-spec-runtime-ownership.md
- Interfaces: SpecRuntimeOwnership@1
- Allow concurrent with: MCPP-014, MCPP-015, MCPP-016, MCPP-017, MCPP-018
- Conflict policy: New ADR file only; do not edit the sealed plan identity.
- Preconditions: Traceability matrix and official spec note exist.
- Effects: An Accepted ADR records the default, rejected alternatives, and implementation consequences.
- Evidence subset: sealed plan Key Decisions, official spec note
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Mcp-Plus-Plus owns schemas, vectors, validators, matrices, and release bundles. Runtimes own adapters only.
- Embedding query: architecture decision spec-runtime-ownership

## MCPP-014 Record ADR for mandatory crypto suite and mcpp-jcs-v1

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adr-crypto-canonical
- Depends on: MCPP-012
- Goal id: MCPP-G020
- Outputs: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0002-crypto-canonical.md
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0002-crypto-canonical.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/adr-crypto-canonical
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 22000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0002-crypto-canonical.md
- Interfaces: CryptoSuiteDecision@1
- Allow concurrent with: MCPP-013, MCPP-015, MCPP-016, MCPP-017, MCPP-018
- Conflict policy: New ADR file only; do not edit the sealed plan identity.
- Preconditions: Traceability matrix and official spec note exist.
- Effects: An Accepted ADR records the default, rejected alternatives, and implementation consequences.
- Evidence subset: sealed plan Key Decisions, official spec note
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Ed25519, explicit key ids, DID-compatible identities, signatures over canonical bytes, CIDv1 raw+sha2-256, RFC 8785 JCS as mcpp-jcs-v1, historical algorithms remain readable.
- Embedding query: architecture decision crypto-canonical

## MCPP-015 Record ADR for conformance levels

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adr-conformance-levels
- Depends on: MCPP-012
- Goal id: MCPP-G020
- Outputs: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0003-conformance-levels.md
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0003-conformance-levels.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/adr-conformance-levels
- Parallel lane: mcpp-lane-state
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 22000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0003-conformance-levels.md
- Interfaces: ConformanceLevelDecision@1
- Allow concurrent with: MCPP-013, MCPP-014, MCPP-016, MCPP-017, MCPP-018
- Conflict policy: New ADR file only; do not edit the sealed plan identity.
- Preconditions: Traceability matrix and official spec note exist.
- Effects: An Accepted ADR records the default, rejected alternatives, and implementation consequences.
- Evidence subset: sealed plan Key Decisions, official spec note
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Levels are structural, canonical, cryptographic, policy-enforced, receipt-signed, and proof-verified. Promotion requires tests at that level.
- Embedding query: architecture decision conformance-levels

## MCPP-016 Record ADR for state modes and Automerge CRDT

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adr-state-modes
- Depends on: MCPP-012
- Goal id: MCPP-G020
- Outputs: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0004-state-modes.md
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0004-state-modes.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/adr-state-modes
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 22000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0004-state-modes.md
- Interfaces: StateModeDecision@1
- Allow concurrent with: MCPP-013, MCPP-014, MCPP-015, MCPP-017, MCPP-018
- Conflict policy: New ADR file only; do not edit the sealed plan identity.
- Preconditions: Traceability matrix and official spec note exist.
- Effects: An Accepted ADR records the default, rejected alternatives, and implementation consequences.
- Evidence subset: sealed plan Key Decisions, official spec note
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Mandatory modes immutable/single_authority/causal/crdt/consensus. SQLite is the single-authority backend. Automerge is the CRDT. Profile G is not BFT.
- Embedding query: architecture decision state-modes

## MCPP-017 Record ADR for DurableExecutor and local SQLite journal

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adr-durable-executor
- Depends on: MCPP-012
- Goal id: MCPP-G020
- Outputs: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0005-durable-executor.md
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0005-durable-executor.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/adr-durable-executor
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 22000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0005-durable-executor.md
- Interfaces: DurableExecutorDecision@1
- Allow concurrent with: MCPP-013, MCPP-014, MCPP-015, MCPP-016, MCPP-018
- Conflict policy: New ADR file only; do not edit the sealed plan identity.
- Preconditions: Traceability matrix and official spec note exist.
- Effects: An Accepted ADR records the default, rejected alternatives, and implementation consequences.
- Evidence subset: sealed plan Key Decisions, official spec note
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Evaluate Restate and Dapr. Mandatory adapter is a SQLite journaled executor testable locally. Second adapter only with repeatable local compose.
- Embedding query: architecture decision durable-executor

## MCPP-018 Record ADR for dual MCP bindings and A2A extension identifier

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: adr-bindings-a2a
- Depends on: MCPP-012
- Goal id: MCPP-G020
- Outputs: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0006-bindings-a2a.md
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0006-bindings-a2a.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/adr-bindings-a2a
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 22000
- Implementation timeout seconds: 3600
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0006-bindings-a2a.md
- Interfaces: BindingAndA2ADecision@1
- Allow concurrent with: MCPP-013, MCPP-014, MCPP-015, MCPP-016, MCPP-017
- Conflict policy: New ADR file only; do not edit the sealed plan identity.
- Preconditions: Traceability matrix and official spec note exist.
- Effects: An Accepted ADR records the default, rejected alternatives, and implementation consequences.
- Evidence subset: sealed plan Key Decisions, official spec note
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Legacy 2024-11-05 and current 2026-07-28 bindings. A2A URI from MCPP-010. No competing public task lifecycle.
- Embedding query: architecture decision bindings-a2a

## MCPP-019 Refactor Profiles A-H to be MCP-version and transport independent

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bindings-abstract
- Depends on: MCPP-013, MCPP-018
- Goal id: MCPP-G030
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/mcp++-profiles-draft.md, ipfs_accelerate_py/mcplusplus/docs/spec/bindings/README.md
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_cross_cutting.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/bindings-abstract
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/mcp++-profiles-draft.md, ipfs_accelerate_py/mcplusplus/docs/spec/bindings/README.md
- Interfaces: ProfileRegistry@1
- Allow concurrent with: MCPP-024
- Conflict policy: Edit profile registry and add bindings README; do not delete legacy initialize vectors.
- Preconditions: ADRs 013 and 018 are Accepted.
- Effects: Profile chapters define semantics without requiring initialize as current behavior.
- Evidence subset: profile specs, official MCP note
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Profiles A-H no longer treat initialize as the only negotiation path. Binding documents are referenced, not inlined as the only lifecycle.
- Embedding query: abstract profile semantics mcp version independent

## MCPP-020 Specify and test the legacy MCP 2024-11-05 binding

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bindings-legacy
- Depends on: MCPP-019
- Goal id: MCPP-G030
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/bindings/mcp-legacy-2024-11-05.md, ipfs_accelerate_py/mcplusplus/tests-py/integration/test_mcp_binding_legacy.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_mcp_binding_legacy.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/bindings-legacy
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/bindings/mcp-legacy-2024-11-05.md, ipfs_accelerate_py/mcplusplus/tests-py/integration/test_mcp_binding_legacy.py
- Interfaces: McpBindingLegacy20241105@1
- Allow concurrent with: MCPP-021
- Conflict policy: New binding module and tests; keep initialize_result.json readable.
- Preconditions: Abstract profiles are separated.
- Effects: Legacy initialize/initialized exchange is explicitly named and tested.
- Evidence subset: initialize_result.json, existing MCP clients
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Legacy tests pass including 2024-11-05 initialize. The binding name is mandatory in capability advertisement.
- Embedding query: legacy mcp 2024-11-05 initialize binding

## MCPP-021 Specify and test the current MCP 2026-07-28 binding

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bindings-current
- Depends on: MCPP-019, MCPP-010
- Goal id: MCPP-G030
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/bindings/mcp-2026-07-28.md, ipfs_accelerate_py/mcplusplus/tests-py/integration/test_mcp_binding_current.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_mcp_binding_current.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/bindings-current
- Parallel lane: mcpp-lane-state
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/bindings/mcp-2026-07-28.md, ipfs_accelerate_py/mcplusplus/tests-py/integration/test_mcp_binding_current.py
- Interfaces: McpBinding20260728@1
- Allow concurrent with: MCPP-020
- Conflict policy: New current-binding files only.
- Preconditions: Official spec note exists and abstract profiles are separated.
- Effects: Current binding uses MCP extensions, discovery, and Tasks. It does not require initialize. libp2p handshake is not MCP initialize. State handles map to StateRef. MCP Tasks map to MCP++ artifacts. Profile support is advertised per request or discovery.
- Evidence subset: official MCP 2026-07-28, Tasks, extensions
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Tests prove a current client works without initialize and that initialize-as-current is rejected.
- Embedding query: mcp 2026-07-28 tasks extensions discovery binding

## MCPP-022 Prove dual-binding peers and reject protocol-version downgrade or forgery

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bindings-compat
- Depends on: MCPP-020, MCPP-021
- Goal id: MCPP-G030
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/bindings/compatibility-matrix.md, ipfs_accelerate_py/mcplusplus/tests-py/integration/test_mcp_binding_compat.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_mcp_binding_compat.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/bindings-compat
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/bindings/compatibility-matrix.md, ipfs_accelerate_py/mcplusplus/tests-py/integration/test_mcp_binding_compat.py
- Interfaces: BindingCompatibilityMatrix@1
- Allow concurrent with: 
- Conflict policy: New matrix and tests only.
- Preconditions: Both bindings exist.
- Effects: A peer can support both bindings without ambiguity. Downgrade and version forgery fail closed.
- Evidence subset: legacy and current binding tests
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Matrix and tests cover legacy-only, current-only, dual, forged version, and downgrade. All negatives fail closed.
- Embedding query: compatibility matrix dual binding downgrade forgery

## MCPP-023 Adapt accelerate and datasets runtimes to the dual MCP bindings

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: bindings-runtime
- Depends on: MCPP-022
- Goal id: MCPP-G030
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/bindings.py, ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/bindings.py, ipfs_accelerate_py/mcp/tests/test_mcplusplus_bindings.py
- Validation: python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_bindings.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/bindings-runtime
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/bindings.py, ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/bindings.py, ipfs_accelerate_py/mcp/tests/test_mcplusplus_bindings.py
- Interfaces: RuntimeBindingAdapter@1
- Allow concurrent with: 
- Conflict policy: Runtime adapter files only; do not duplicate spec text.
- Preconditions: Compatibility matrix exists.
- Effects: Accelerate and datasets advertise the bindings they actually implement and reject the others fail-closed.
- Evidence subset: binding specs, existing mcp_server initialize paths
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Runtime tests cover legacy client, current client, and dual peer. No silent initialize on the current path.
- Embedding query: accelerate datasets mcp binding adapter

## MCPP-024 Specify mcpp-jcs-v1 without changing historical artifact CIDs

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: canonical-spec
- Depends on: MCPP-014
- Goal id: MCPP-G040
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/canonicalization-mcpp-jcs-v1.md, ipfs_accelerate_py/mcplusplus/schemas/canonicalization/mcpp-jcs-v1.schema.json
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/spec/canonicalization-mcpp-jcs-v1.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/canonical-spec
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-small
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/canonicalization-mcpp-jcs-v1.md, ipfs_accelerate_py/mcplusplus/schemas/canonicalization/mcpp-jcs-v1.schema.json
- Interfaces: McppJcsV1@1
- Allow concurrent with: MCPP-019
- Conflict policy: New spec and schema; do not rewrite existing vector CIDs.
- Preconditions: Crypto/canonical ADR is Accepted.
- Effects: UTF-8, key order, Unicode, numbers, -0, null, arrays, duplicate keys, unsupported values, and version markers are specified.
- Evidence subset: RFC 8785, existing cid-native-artifacts.md
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Document names mcpp-jcs-v1, describes migration, and forbids silent CID changes.
- Embedding query: rfc 8785 jcs mcpp-jcs-v1 canonicalization

## MCPP-025 Publish golden canonicalization vectors

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: canonical-vectors
- Depends on: MCPP-024
- Goal id: MCPP-G040
- Outputs: ipfs_accelerate_py/mcplusplus/conformance/vectors/mcpp-jcs-v1
- Validation: python -c "import json,pathlib; p=pathlib.Path('ipfs_accelerate_py/mcplusplus/conformance/vectors/mcpp-jcs-v1'); assert any(p.glob('*.json'))"
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/canonical-vectors
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/conformance/vectors/mcpp-jcs-v1
- Interfaces: GoldenVector@1
- Allow concurrent with: MCPP-029
- Conflict policy: New vector directory only.
- Preconditions: Canonicalization spec exists.
- Effects: Vectors include source object, canonical bytes hex/base64, SHA-256, CID, signature input, signature placeholder, expected validator result.
- Evidence subset: mcpp-jcs-v1 spec
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: At least one positive and one negative vector exist for numbers, Unicode, null, empty object, nested keys, and duplicate-key rejection.
- Embedding query: golden vector canonical bytes cid sha256

## MCPP-026 Implement mcpp-jcs-v1 in Python and TypeScript

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: canonical-py-ts
- Depends on: MCPP-025
- Goal id: MCPP-G040
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/validators/canonical_jcs.py, ipfs_accelerate_py/mcplusplus/tests-ts/src/validators/canonicalJcs.ts
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration -k jcs && cd tests-ts && npm test -- canonicalJcs
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/canonical-py-ts
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/validators/canonical_jcs.py, ipfs_accelerate_py/mcplusplus/tests-ts/src/validators/canonicalJcs.ts
- Interfaces: McppJcsV1@1
- Allow concurrent with: MCPP-027
- Conflict policy: New canonical modules plus tests; do not change historical codec defaults.
- Preconditions: Golden vectors exist.
- Effects: Python and TypeScript produce identical canonical bytes on the golden set.
- Evidence subset: golden vectors
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Python and TypeScript pass the same golden vectors. Historical vectors still parse under their recorded algorithm.
- Embedding query: python typescript jcs canonical implementation

## MCPP-027 Implement mcpp-jcs-v1 in Go and Rust

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: canonical-go-rs
- Depends on: MCPP-025
- Goal id: MCPP-G040
- Outputs: ipfs_accelerate_py/mcplusplus/tests-go/canonical_jcs.go, ipfs_accelerate_py/mcplusplus/tests-rs/src/canonical_jcs.rs
- Validation: cd ipfs_accelerate_py/mcplusplus/tests-go && go test ./... -count=1 && cd ../tests-rs && cargo test canonical_jcs
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/canonical-go-rs
- Parallel lane: mcpp-lane-state
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-go/canonical_jcs.go, ipfs_accelerate_py/mcplusplus/tests-rs/src/canonical_jcs.rs
- Interfaces: McppJcsV1@1
- Allow concurrent with: MCPP-026
- Conflict policy: New Go/Rust canonical modules only.
- Preconditions: Golden vectors exist.
- Effects: Go and Rust produce identical canonical bytes on the golden set.
- Evidence subset: golden vectors
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Go and Rust pass the same golden vectors as Python/TypeScript will in MCPP-028.
- Embedding query: go rust jcs canonical implementation

## MCPP-028 Prove four-language canonical, CID, and signature-input identity

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: canonical-identity
- Depends on: MCPP-026, MCPP-027
- Goal id: MCPP-G040
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_cross_language_jcs.py, docs/reports/mcplusplus-1.0-gap-closure/canonical/four-language.json
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_cross_language_jcs.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/canonical-identity
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_cross_language_jcs.py, docs/reports/mcplusplus-1.0-gap-closure/canonical/four-language.json
- Interfaces: CrossLanguageIdentityReceipt@1
- Allow concurrent with: 
- Conflict policy: New identity test and receipt only.
- Preconditions: All four implementations exist.
- Effects: Property, mutation, unknown-field, version-mismatch, malformed Unicode, numeric edge, and signature-input tests exist.
- Evidence subset: four language implementations, golden vectors
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Identical canonical bytes, SHA-256, and CID across Python, TypeScript, Go, and Rust. Mismatches fail the job.
- Embedding query: cross language canonical cid signature identity

## MCPP-029 Generate schemas from one canonical source

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: canonical-schemas
- Depends on: MCPP-024, MCPP-015
- Goal id: MCPP-G040
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/README.md, ipfs_accelerate_py/mcplusplus/scripts/generate_schemas.py
- Validation: python ipfs_accelerate_py/mcplusplus/scripts/generate_schemas.py --check
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/canonical-schemas
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/README.md, ipfs_accelerate_py/mcplusplus/scripts/generate_schemas.py
- Interfaces: SchemaId@1
- Allow concurrent with: MCPP-025
- Conflict policy: Own schema generation scripts and the schemas tree layout; do not hand-edit generated outputs after this lands.
- Preconditions: Canonicalization spec and conformance-level ADR exist.
- Effects: One source generates language bindings or at least JSON Schema plus a drift check.
- Evidence subset: existing profile-h schemas as the generation pattern
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: A drift check fails if a language model disagrees with the canonical schema source.
- Embedding query: schema generation single source drift check

## MCPP-030 Create ExecutionEnvelope@1, ExecutionResult@1, ExecutionReceipt@1, and PortableError@1

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: envelope-schemas
- Depends on: MCPP-028, MCPP-029
- Goal id: MCPP-G050
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/execution/execution-envelope-1.schema.json, ipfs_accelerate_py/mcplusplus/schemas/execution/execution-result-1.schema.json, ipfs_accelerate_py/mcplusplus/schemas/execution/execution-receipt-1.schema.json, ipfs_accelerate_py/mcplusplus/schemas/execution/portable-error-1.schema.json, ipfs_accelerate_py/mcplusplus/docs/spec/execution-envelope.md
- Validation: python -m json.tool ipfs_accelerate_py/mcplusplus/schemas/execution/execution-envelope-1.schema.json > /dev/null
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/envelope-schemas
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/execution/execution-envelope-1.schema.json, ipfs_accelerate_py/mcplusplus/schemas/execution/execution-result-1.schema.json, ipfs_accelerate_py/mcplusplus/schemas/execution/execution-receipt-1.schema.json, ipfs_accelerate_py/mcplusplus/schemas/execution/portable-error-1.schema.json, ipfs_accelerate_py/mcplusplus/docs/spec/execution-envelope.md
- Interfaces: ExecutionEnvelope@1, ExecutionResult@1, ExecutionReceipt@1, PortableError@1
- Allow concurrent with: 
- Conflict policy: New schema family and spec chapter.
- Preconditions: Four-language canonical identity and schema generation exist.
- Effects: Envelope covers identity, authority, intent, IO, constraints, state, policy, and provenance listed in the assignment.
- Evidence subset: cid-native-artifacts.md, Profile B objects
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: All required fields or references exist. Result/receipt include status, output CIDs, state transitions, side effects, decision/delegation refs, executor identity, retry, duration, portable errors, proofs, signature, event_cid.
- Embedding query: execution envelope result receipt portable error schema

## MCPP-031 Adapt existing Profile B artifacts to ExecutionEnvelope@1

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: envelope-profile-b
- Depends on: MCPP-030
- Goal id: MCPP-G050
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/validators/envelope_profile_b.py, ipfs_accelerate_py/mcplusplus/conformance/vectors/envelope/profile-b-adapter.json
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration -k profile_b_adapter
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/envelope-profile-b
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/validators/envelope_profile_b.py, ipfs_accelerate_py/mcplusplus/conformance/vectors/envelope/profile-b-adapter.json
- Interfaces: ProfileBAdapter@1
- Allow concurrent with: MCPP-032
- Conflict policy: Adapter and new vectors only; do not mutate historical B fixtures.
- Preconditions: Envelope schemas exist.
- Effects: Existing valid Profile B artifacts remain readable and adapt to Envelope@1 without CID change.
- Evidence subset: execution_receipt.json and cid artifact fixtures
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Historical B CIDs still verify. Adapter output validates as Envelope@1.
- Embedding query: profile b adapter execution envelope cid compatibility

## MCPP-032 Adapt existing Profile G artifacts to ExecutionEnvelope@1

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: envelope-profile-g
- Depends on: MCPP-030
- Goal id: MCPP-G050
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/validators/envelope_profile_g.py, ipfs_accelerate_py/mcplusplus/conformance/vectors/envelope/profile-g-adapter.json
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration -k profile_g_adapter
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/envelope-profile-g
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/validators/envelope_profile_g.py, ipfs_accelerate_py/mcplusplus/conformance/vectors/envelope/profile-g-adapter.json
- Interfaces: ProfileGAdapter@1
- Allow concurrent with: MCPP-031
- Conflict policy: Adapter and new vectors only; do not mutate historical G fixtures.
- Preconditions: Envelope schemas exist.
- Effects: Existing valid Profile G artifacts remain readable and adapt without CID change.
- Evidence subset: profile_g_artifacts_valid.json
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Historical G CIDs still verify. Adapter output validates as Envelope@1.
- Embedding query: profile g adapter execution envelope cid compatibility

## MCPP-033 Add four-language envelope validators and vectors

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: envelope-validators
- Depends on: MCPP-031, MCPP-032
- Goal id: MCPP-G050
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_execution_envelope.py, ipfs_accelerate_py/mcplusplus/tests-ts/src/__tests__/execution-envelope.test.ts, ipfs_accelerate_py/mcplusplus/tests-go/execution_envelope_test.go, ipfs_accelerate_py/mcplusplus/tests-rs/tests/execution_envelope_test.rs
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_execution_envelope.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/envelope-validators
- Parallel lane: mcpp-lane-state
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_execution_envelope.py, ipfs_accelerate_py/mcplusplus/tests-ts/src/__tests__/execution-envelope.test.ts, ipfs_accelerate_py/mcplusplus/tests-go/execution_envelope_test.go, ipfs_accelerate_py/mcplusplus/tests-rs/tests/execution_envelope_test.rs
- Interfaces: ExecutionEnvelopeValidator@1
- Allow concurrent with: 
- Conflict policy: New validator tests in all four languages.
- Preconditions: Both adapters exist.
- Effects: Positive and negative envelope/result/receipt/error vectors pass identically.
- Evidence subset: envelope schemas and adapters
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Four languages accept the same positives and reject the same negatives.
- Embedding query: execution envelope validator python typescript go rust

## MCPP-034 Emit and verify envelopes from the accelerate runtime

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: envelope-accelerate
- Depends on: MCPP-033
- Goal id: MCPP-G050
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/envelope.py, ipfs_accelerate_py/mcp/tests/test_mcplusplus_envelope.py
- Validation: python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_envelope.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/envelope-accelerate
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/envelope.py, ipfs_accelerate_py/mcp/tests/test_mcplusplus_envelope.py
- Interfaces: RuntimeEnvelopeAdapter@1
- Allow concurrent with: 
- Conflict policy: Accelerate runtime files only.
- Preconditions: Four-language validators exist.
- Effects: Accelerate can create, persist by CID, and verify an envelope.
- Evidence subset: mcp_server/mcplusplus artifacts module
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Runtime test creates an envelope, computes CID with mcpp-jcs-v1, and verifies it. No large spec reimplementation.
- Embedding query: accelerate execution envelope runtime adapter

## MCPP-035 Specify StateRef@1 with mandatory consistency modes

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-schema
- Depends on: MCPP-028, MCPP-016
- Goal id: MCPP-G060
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/state/state-ref-1.schema.json, ipfs_accelerate_py/mcplusplus/docs/spec/state-ref.md
- Validation: python -m json.tool ipfs_accelerate_py/mcplusplus/schemas/state/state-ref-1.schema.json > /dev/null
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/state-schema
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/state/state-ref-1.schema.json, ipfs_accelerate_py/mcplusplus/docs/spec/state-ref.md
- Interfaces: StateRef@1
- Allow concurrent with: MCPP-030
- Conflict policy: New StateRef schema and spec.
- Preconditions: Canonical identity and state-mode ADR exist.
- Effects: StateRef includes id, schema CID, root CID, mode, authority, version, epoch, clocks, read/write caps, lease, fence, merge/retention/confidentiality/consensus policy CIDs, parents.
- Evidence subset: state-mode ADR
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Missing or unknown mode is invalid. Allowed modes are exactly immutable, single_authority, causal, crdt, consensus.
- Embedding query: stateref consistency mode schema

## MCPP-036 Implement the StateProvider interface and immutable CID state

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-immutable
- Depends on: MCPP-035
- Goal id: MCPP-G060
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/state/provider.py, ipfs_accelerate_py/mcp_server/mcplusplus/state/immutable_cid.py, test/api/test_mcplusplus_state_immutable.py
- Validation: python -m pytest -q test/api/test_mcplusplus_state_immutable.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/state-immutable
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/state/provider.py, ipfs_accelerate_py/mcp_server/mcplusplus/state/immutable_cid.py, test/api/test_mcplusplus_state_immutable.py
- Interfaces: StateProvider@1, ImmutableCidState@1
- Allow concurrent with: MCPP-037, MCPP-038, MCPP-039
- Conflict policy: New state package files only.
- Preconditions: StateRef schema exists.
- Effects: Immutable provider is append-only, fetchable, and CID-verifiable using existing IPFS artifact persistence.
- Evidence subset: existing artifact persistence
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Writes are append-only. Fetch verifies CID. Mutation of an existing CID is rejected.
- Embedding query: immutable cid state provider append only

## MCPP-037 Implement durable single-authority SQLite state with CAS, leases, and restart

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-sqlite
- Depends on: MCPP-035
- Goal id: MCPP-G060
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/state/sqlite_authority.py, test/api/test_mcplusplus_state_sqlite_restart.py
- Validation: python -m pytest -q test/api/test_mcplusplus_state_sqlite_restart.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/state-sqlite
- Parallel lane: mcpp-lane-schema
- Resource class: io-large
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/state/sqlite_authority.py, test/api/test_mcplusplus_state_sqlite_restart.py
- Interfaces: SqliteAuthorityState@1
- Allow concurrent with: MCPP-036, MCPP-038, MCPP-039
- Conflict policy: New SQLite provider and restart test.
- Preconditions: StateRef schema exists.
- Effects: CAS/version preconditions, restart recovery, leases, and fencing are enforced.
- Evidence subset: SQLite WAL, StateRef
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Restart test recovers committed state, rejects stale fence, and fails a CAS mismatch. No acknowledged committed write is lost in the declared crash matrix.
- Embedding query: sqlite cas lease fencing restart state

## MCPP-038 Implement a real Automerge CRDT state adapter

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-crdt
- Depends on: MCPP-035
- Goal id: MCPP-G060
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/state/automerge_crdt.py, test/api/test_mcplusplus_state_automerge.py
- Validation: python -m pytest -q test/api/test_mcplusplus_state_automerge.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/state-crdt
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/state/automerge_crdt.py, test/api/test_mcplusplus_state_automerge.py
- Interfaces: AutomergeCrdtState@1
- Allow concurrent with: MCPP-036, MCPP-037, MCPP-039
- Conflict policy: New Automerge adapter and tests. Do not invent LWW.
- Preconditions: StateRef schema exists. Automerge is a maintained permissive dependency.
- Effects: Concurrent offline updates, reordering, duplication, partition healing, and convergence are tested.
- Evidence subset: Automerge library
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Two isolated replicas converge after partition heal. Duplicates are idempotent. The implementation is Automerge, not informal LWW.
- Embedding query: automerge crdt convergence partition

## MCPP-039 Define the consensus plugin contract with honest guarantee labels

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-consensus
- Depends on: MCPP-035
- Goal id: MCPP-G060
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/state/consensus_plugin.py, ipfs_accelerate_py/mcplusplus/docs/spec/consensus-plugin.md, test/api/test_mcplusplus_state_consensus_labels.py
- Validation: python -m pytest -q test/api/test_mcplusplus_state_consensus_labels.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/state-consensus
- Parallel lane: mcpp-lane-state
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/state/consensus_plugin.py, ipfs_accelerate_py/mcplusplus/docs/spec/consensus-plugin.md, test/api/test_mcplusplus_state_consensus_labels.py
- Interfaces: ConsensusPlugin@1
- Allow concurrent with: MCPP-036, MCPP-037, MCPP-038
- Conflict policy: New plugin contract and deterministic test adapter. Do not claim BFT.
- Preconditions: StateRef schema exists.
- Effects: Plugin evidence format exists. Profile G majority/neighborhood is wired only for the behavior it guarantees. Labels distinguish coordination, majority approval, crash consensus, and BFT.
- Evidence subset: Profile G neighborhood records
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Tests fail if a neighborhood result is labeled BFT. Deterministic test adapter is supplied.
- Embedding query: consensus plugin coordination majority bft label

## MCPP-040 Prove Event DAG branches do not silently merge mutable state

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-nonmerge
- Depends on: MCPP-037, MCPP-038, MCPP-039
- Goal id: MCPP-G060
- Outputs: test/api/test_mcplusplus_state_event_dag_nonmerge.py
- Validation: python -m pytest -q test/api/test_mcplusplus_state_event_dag_nonmerge.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/state-nonmerge
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: test/api/test_mcplusplus_state_event_dag_nonmerge.py
- Interfaces: StateNonMergeReceipt@1
- Allow concurrent with: 
- Conflict policy: New test only plus any tiny helper required.
- Preconditions: SQLite, Automerge, and consensus plugin exist.
- Effects: Observing two concurrent Event DAG branches does not merge single_authority values. CRDT merge happens only in crdt mode.
- Evidence subset: Event DAG validators, state providers
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Single-authority conflict is explicit. CRDT mode converges. Consensus mode requires plugin evidence. Silent merge fails the test.
- Embedding query: event dag branch nonmerge mutable state

## MCPP-041 Implement real Ed25519 delegation verification over canonical bytes

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: crypto-verify
- Depends on: MCPP-014, MCPP-028
- Goal id: MCPP-G070
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/validators/ucan_delegation.py, ipfs_accelerate_py/mcplusplus/tests-ts/src/validators/ucanDelegation.ts, ipfs_accelerate_py/mcplusplus/tests-go/validators/ucan_delegation.go, ipfs_accelerate_py/mcplusplus/tests-rs/src/validators/ucan_delegation.rs
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_ucan_delegation.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/crypto-verify
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-security
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/validators/ucan_delegation.py, ipfs_accelerate_py/mcplusplus/tests-ts/src/validators/ucanDelegation.ts, ipfs_accelerate_py/mcplusplus/tests-go/validators/ucan_delegation.go, ipfs_accelerate_py/mcplusplus/tests-rs/src/validators/ucan_delegation.rs
- Interfaces: DelegationProof@1
- Allow concurrent with: 
- Conflict policy: Replace structural-only verification with cryptographic verification. Keep fixtures; change expected results where they were false positives.
- Preconditions: Crypto ADR and canonical identity exist.
- Effects: Signatures over mcpp-jcs-v1 bytes verify. Missing or forged signatures fail closed.
- Evidence subset: existing UCAN fixtures, Ed25519 libraries
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: A token with required fields but an invalid signature is invalid at cryptographic level. Structural level may still be reported separately.
- Embedding query: ed25519 ucan canonical signature verification

## MCPP-042 Enforce attenuation, audience, time, depth, replay, and executor binding

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: crypto-attenuation
- Depends on: MCPP-041
- Goal id: MCPP-G070
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_ucan_attenuation.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_ucan_attenuation.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/crypto-attenuation
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-security
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_ucan_attenuation.py
- Interfaces: AttenuationPolicy@1
- Allow concurrent with: MCPP-043
- Conflict policy: New attenuation tests and verifier checks.
- Preconditions: Real signature verification exists.
- Effects: Issuer/audience continuity, capability/resource/method/budget attenuation, nbf/exp, depth, redelegation permission, nonce/replay, required policy CID, and executor binding are enforced.
- Evidence subset: delegation verifier
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Each listed check has a failing negative test. Expansion of capabilities or resources is deny.
- Embedding query: capability resource method budget attenuation replay

## MCPP-043 Specify and implement RevocationRecord@1

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: crypto-revocation
- Depends on: MCPP-041
- Goal id: MCPP-G070
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/delegation/revocation-record-1.schema.json, ipfs_accelerate_py/mcplusplus/tests-py/validators/revocation.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration -k revocation
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/crypto-revocation
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-security
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/delegation/revocation-record-1.schema.json, ipfs_accelerate_py/mcplusplus/tests-py/validators/revocation.py
- Interfaces: RevocationRecord@1
- Allow concurrent with: MCPP-042
- Conflict policy: New revocation schema and verifier.
- Preconditions: Real signature verification exists.
- Effects: Record includes issuer, revoked delegation CID, effective time, reason, optional replacement, signature, and discovery semantics.
- Evidence subset: SwissKnife UCAN revocation registry as observation
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Revoked delegations fail closed even if the signature on the original token is valid.
- Embedding query: revocation record delegation cid discovery

## MCPP-044 Add adversarial cryptographic negative vectors in all four languages

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: crypto-adversarial
- Depends on: MCPP-042, MCPP-043
- Goal id: MCPP-G070
- Outputs: ipfs_accelerate_py/mcplusplus/conformance/vectors/crypto/adversarial
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration -k adversarial_ucan
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/crypto-adversarial
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-security
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/conformance/vectors/crypto/adversarial
- Interfaces: AdversarialVector@1
- Allow concurrent with: 
- Conflict policy: New vector directory and tests. No mocks for verification.
- Preconditions: Attenuation and revocation exist.
- Effects: Vectors cover forged signature, altered bytes, wrong audience, expanded capabilities/resources, expired, future nbf, revoked, missing proof, replay, wrong executor, wrong policy CID, valid PeerID invalid UCAN.
- Evidence subset: verifier, revocation, canonical bytes
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Every listed case fails closed in Python, TypeScript, Go, and Rust.
- Embedding query: adversarial ucan negative vectors four languages

## MCPP-045 Require signed cross-trust-domain receipts and separate transport identity from authority

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: crypto-receipts
- Depends on: MCPP-044, MCPP-033
- Goal id: MCPP-G070
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_signed_receipts.py, ipfs_accelerate_py/mcplusplus/docs/spec/cid-native-artifacts.md
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_signed_receipts.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/crypto-receipts
- Parallel lane: mcpp-lane-state
- Resource class: cpu-security
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_signed_receipts.py, ipfs_accelerate_py/mcplusplus/docs/spec/cid-native-artifacts.md
- Interfaces: ReceiptVerifier@1
- Allow concurrent with: 
- Conflict policy: Receipt verification tests and spec clarifications.
- Preconditions: Adversarial vectors and envelope validators exist.
- Effects: Cross-trust-domain execution requires a signed receipt. Transport identity cannot satisfy execution authority.
- Evidence subset: ExecutionReceipt@1, UCAN verifier
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Unsigned cross-domain receipt is deny. Valid TLS/PeerID with invalid UCAN is deny. Independent verifier process can validate a receipt by CID.
- Embedding query: signed receipt trust domain transport identity

## MCPP-046 Implement the deterministic policy evaluator interface

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: policy-evaluator
- Depends on: MCPP-015, MCPP-045
- Goal id: MCPP-G080
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/validators/policy_evaluation.py, ipfs_accelerate_py/mcplusplus/docs/spec/temporal-deontic-policy.md
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_policy_evaluation.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/policy-evaluator
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/validators/policy_evaluation.py, ipfs_accelerate_py/mcplusplus/docs/spec/temporal-deontic-policy.md
- Interfaces: PolicyEvaluator@1, PolicyDecision@1
- Allow concurrent with: 
- Conflict policy: Evaluator implementation and spec update. Do not treat statement commitments as proofs.
- Preconditions: Conformance-level ADR and signed receipts exist.
- Effects: Evaluator accepts intent, delegation, policy, context roots, logical time, prior events and returns allow/deny/allow_with_obligations plus fired rules, facts, obligations, deadlines, compensation, human-approval, decision commitment, decision_cid, optional signature.
- Evidence subset: existing policy_evaluation validator
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Outputs are deterministic for the same inputs. Missing context or stale root is deny.
- Embedding query: policy evaluator allow deny obligations decision_cid

## MCPP-047 Implement obligation lifecycle events

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: policy-obligations
- Depends on: MCPP-046
- Goal id: MCPP-G080
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/policy/obligation-event-1.schema.json, ipfs_accelerate_py/mcplusplus/tests-py/integration/test_obligation_lifecycle.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_obligation_lifecycle.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/policy-obligations
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/policy/obligation-event-1.schema.json, ipfs_accelerate_py/mcplusplus/tests-py/integration/test_obligation_lifecycle.py
- Interfaces: ObligationEvent@1
- Allow concurrent with: 
- Conflict policy: New obligation schema and tests.
- Preconditions: Evaluator exists.
- Effects: Events obligation_created, obligation_satisfied, obligation_violated, compensation_required, compensation_completed, compensation_failed are emitted and content-addressed.
- Evidence subset: PolicyDecision obligations
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Each event type has a positive test. Deadline passage emits violated. Compensation paths are tested.
- Embedding query: obligation lifecycle deadline compensation events

## MCPP-048 Add policy version, stale context, revocation-before-execution, and conflict tests

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: policy-tests
- Depends on: MCPP-047
- Goal id: MCPP-G080
- Outputs: ipfs_accelerate_py/mcplusplus/conformance/vectors/policy
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration -k policy_negative
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/policy-tests
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/conformance/vectors/policy
- Interfaces: PolicyVector@1
- Allow concurrent with: 
- Conflict policy: New vectors and tests.
- Preconditions: Obligation lifecycle exists.
- Effects: Tests cover policy version mismatch, missing context, stale root, deadline, revocation after planning before execution, allowed-with-obligations, unsatisfied obligation, compensating action, conflicting policies with deterministic resolution.
- Evidence subset: evaluator, revocation, StateRef
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Every listed case has an expected decision. Conflicting policies resolve by a documented deterministic rule.
- Embedding query: policy negative stale revocation conflict deadline

## MCPP-049 Wire datasets Profile D to the real evaluator and fail closed

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: policy-datasets
- Depends on: MCPP-048
- Goal id: MCPP-G080
- Outputs: ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/policy.py, ipfs_datasets_py/tests/unit/mcp_server/test_mcplusplus_policy_evaluator.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/mcp_server/test_mcplusplus_policy_evaluator.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/policy-datasets
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/policy.py, ipfs_datasets_py/tests/unit/mcp_server/test_mcplusplus_policy_evaluator.py
- Interfaces: ProfileDPolicyProvider@1
- Allow concurrent with: 
- Conflict policy: Datasets policy wiring only. Do not reimplement the evaluator.
- Preconditions: Policy negatives exist.
- Effects: Datasets dispatch consults the evaluator before side effects. Unavailable evaluator is deny.
- Evidence subset: datasets mcplusplus tools, PolicyEvaluator
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: A deny decision never dispatches. Missing evaluator never degrades to allow.
- Embedding query: datasets profile d policy evaluator fail closed

## MCPP-050 Define the DurableExecutor interface

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: durable-interface
- Depends on: MCPP-017, MCPP-033, MCPP-036
- Goal id: MCPP-G090
- Outputs: ipfs_accelerate_py/mcplusplus/docs/architecture/durable-execution.md, ipfs_accelerate_py/mcplusplus/schemas/durable/durable-executor-1.schema.json
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/architecture/durable-execution.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/durable-interface
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-small
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/architecture/durable-execution.md, ipfs_accelerate_py/mcplusplus/schemas/durable/durable-executor-1.schema.json
- Interfaces: DurableExecutor@1
- Allow concurrent with: 
- Conflict policy: New interface spec and schema.
- Preconditions: Durable ADR, envelope validators, and state provider interface exist.
- Effects: Interface supports start, resume, signal, cancel, checkpoint, retry, durable timer, compensation, inspect, recover, finalize.
- Evidence subset: durable ADR
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Every method has a typed request/result. Journaled transitions map to Event DAG events. Final outputs bind to signed receipts.
- Embedding query: durable executor interface journal checkpoint

## MCPP-051 Implement the SQLite journaled DurableExecutor adapter

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: durable-sqlite
- Depends on: MCPP-050
- Goal id: MCPP-G090
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/durable/sqlite_executor.py, ipfs_accelerate_py/mcp_server/mcplusplus/durable/journal.py
- Validation: python -m pytest -q test/api/test_mcplusplus_durable_sqlite.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/durable-sqlite
- Parallel lane: mcpp-lane-state
- Resource class: io-large
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/durable/sqlite_executor.py, ipfs_accelerate_py/mcp_server/mcplusplus/durable/journal.py
- Interfaces: SqliteDurableExecutor@1
- Allow concurrent with: 
- Conflict policy: New durable package. Optional Restate/Dapr adapter only if local compose works.
- Preconditions: DurableExecutor interface exists.
- Effects: Adapter journals externally visible steps, uses idempotency keys, persists task/state refs, preserves cancel/obligations/timers, rejects stale fences.
- Evidence subset: SQLite, StateProvider, Envelope
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Unit tests cover journal replay, idempotent retry, cancel persistence, and stale fence reject. Restate/Dapr absence is a documented non-blocker.
- Embedding query: sqlite durable journal idempotency fencing

## MCPP-052 Add the crash-recovery integration test

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: durable-crash
- Depends on: MCPP-051
- Goal id: MCPP-G090
- Outputs: test/api/test_mcplusplus_durable_crash_recovery.py
- Validation: python -m pytest -q test/api/test_mcplusplus_durable_crash_recovery.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/durable-crash
- Parallel lane: mcpp-lane-transport
- Resource class: io-large
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: test/api/test_mcplusplus_durable_crash_recovery.py
- Interfaces: CrashRecoveryReceipt@1
- Allow concurrent with: 
- Conflict policy: New crash-recovery test. Kill a real subprocess, do not mock recover().
- Preconditions: SQLite durable adapter exists.
- Effects: Starts multi-step task, commits one side effect, terminates process, restarts, resumes, does not repeat the effect, emits one receipt.
- Evidence subset: sqlite executor
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Exactly one side-effect observation and one authoritative completion receipt after restart.
- Embedding query: crash recovery no duplicate side effect

## MCPP-053 Bind the accelerate runtime to DurableExecutor

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: durable-accelerate
- Depends on: MCPP-052
- Goal id: MCPP-G090
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/durable/runtime.py, ipfs_accelerate_py/mcp/tests/test_mcplusplus_durable_runtime.py
- Validation: python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_durable_runtime.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/durable-accelerate
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/durable/runtime.py, ipfs_accelerate_py/mcp/tests/test_mcplusplus_durable_runtime.py
- Interfaces: RuntimeDurableAdapter@1
- Allow concurrent with: 
- Conflict policy: Accelerate wiring only.
- Preconditions: Crash-recovery test exists.
- Effects: Accelerate task dispatch can start/resume/cancel through DurableExecutor and emit Event DAG events.
- Evidence subset: mcp_server workflow tools, DurableExecutor
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Runtime test exercises start and resume after a controlled restart hook. No second journal implementation.
- Embedding query: accelerate durable executor runtime

## MCPP-054 Specify the A2A execution extension and mappings

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: a2a-spec
- Depends on: MCPP-018, MCPP-010, MCPP-033
- Goal id: MCPP-G100
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/a2a-extension.md
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/spec/a2a-extension.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/a2a-spec
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-small
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/a2a-extension.md
- Interfaces: A2AExecutionExtension@1
- Allow concurrent with: 
- Conflict policy: New spec chapter. Use the verified URI from MCPP-010.
- Preconditions: Binding/A2A ADR, official spec note, and envelope validators exist.
- Effects: Mappings cover Agent Card to MCP-IDL, interface_cid on the card, Task to TaskSpec, context to StateRef, Message/Part to CID inputs, Artifact to output_cid, status to events, cancel to durable cancel, streaming to progress, push to receipt notifications, auth to delegation challenges, finals to receipt_cid/event_cid/proof.
- Evidence subset: official A2A spec, envelope schemas
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Spec cites the official extension identifier rule and does not invent a competing task lifecycle.
- Embedding query: a2a extension agent card task artifact mapping

## MCPP-055 Add A2A extension schemas and positive/negative vectors

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: a2a-schemas
- Depends on: MCPP-054
- Goal id: MCPP-G100
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/a2a, ipfs_accelerate_py/mcplusplus/conformance/vectors/a2a
- Validation: python -c "import pathlib; assert any(pathlib.Path('ipfs_accelerate_py/mcplusplus/schemas/a2a').glob('*.json'))"
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/a2a-schemas
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/a2a, ipfs_accelerate_py/mcplusplus/conformance/vectors/a2a
- Interfaces: A2AExtensionSchema@1
- Allow concurrent with: 
- Conflict policy: New A2A schema and vector directories.
- Preconditions: A2A spec exists.
- Effects: Positive and negative vectors cover well-formed extension, malformed extension, and unsupported profile.
- Evidence subset: a2a spec
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Malformed extension and unsupported profile vectors are expected failures.
- Embedding query: a2a schema vectors malformed unsupported

## MCPP-056 Implement the A2A reference adapter and two-agent handoff test

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: a2a-adapter
- Depends on: MCPP-055
- Goal id: MCPP-G100
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/a2a_adapter.py, test/api/test_mcplusplus_a2a_handoff.py
- Validation: python -m pytest -q test/api/test_mcplusplus_a2a_handoff.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/a2a-adapter
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/a2a_adapter.py, test/api/test_mcplusplus_a2a_handoff.py
- Interfaces: A2ATaskAdapter@1
- Allow concurrent with: 
- Conflict policy: New adapter and e2e test with two independently instantiated agents.
- Preconditions: A2A schemas exist.
- Effects: Handoff, cancel, retry, streaming, malformed-extension, and unsupported-profile tests exist.
- Evidence subset: A2A vectors, envelope, durable cancel
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Two agents complete a handoff. Cancel writes Event DAG records. Malformed extension fails closed.
- Embedding query: a2a handoff cancel retry streaming adapter

## MCPP-057 Adapt SwissKnife to the A2A execution extension

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: a2a-swissknife
- Depends on: MCPP-056
- Goal id: MCPP-G100
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/runtime/swissknife-a2a.md
- Validation: test -s docs/reports/mcplusplus-1.0-gap-closure/runtime/swissknife-a2a.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/a2a-swissknife
- Parallel lane: mcpp-lane-state
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/runtime/swissknife-a2a.md
- Interfaces: SwissKnifeA2AAdapter@1
- Allow concurrent with: 
- Conflict policy: Modify only the bound SwissKnife checkout discovered in MCPP-001. Record the nested commit SHA in the receipt.
- Preconditions: Reference adapter exists. SwissKnife checkout is bound.
- Effects: SwissKnife can present the extension on its Agent Card equivalent and complete a handoff test.
- Evidence subset: swissknife src/services and mcp-plus-plus adapters
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Receipt lists SwissKnife files changed and the handoff command. Missing checkout is blocked, not faked.
- Embedding query: swissknife a2a extension adapter

## MCPP-058 Specify the agent advertisement schema

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: discovery-schema
- Depends on: MCPP-054
- Goal id: MCPP-G110
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/discovery/agent-advertisement-1.schema.json, ipfs_accelerate_py/mcplusplus/docs/spec/mcp-idl.md
- Validation: python -m json.tool ipfs_accelerate_py/mcplusplus/schemas/discovery/agent-advertisement-1.schema.json > /dev/null
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/discovery-schema
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/discovery/agent-advertisement-1.schema.json, ipfs_accelerate_py/mcplusplus/docs/spec/mcp-idl.md
- Interfaces: AgentAdvertisement@1
- Allow concurrent with: MCPP-055
- Conflict policy: New advertisement schema and MCP-IDL additions.
- Preconditions: A2A spec exists so the advertisement can map to Agent Cards.
- Effects: Advertisement describes identity, endpoints, transports, MCP versions, A2A version, MCP++ profiles, interface CIDs, policy languages, proof systems, runtimes, accelerators, locality, price, health, load, TTL, trust domain, residency, confidentiality.
- Evidence subset: MCP-IDL, A2A Agent Card
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Schema rejects missing identity, missing TTL, and missing interface CID list. Mapping to Agent Card is documented.
- Embedding query: agent advertisement mcp-idl a2a card

## MCPP-059 Implement the registry interface and static test registry

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: discovery-static
- Depends on: MCPP-058
- Goal id: MCPP-G110
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/registry/interface.py, ipfs_accelerate_py/mcp_server/mcplusplus/registry/static.py, test/api/test_mcplusplus_registry_static.py
- Validation: python -m pytest -q test/api/test_mcplusplus_registry_static.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/discovery-static
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/registry/interface.py, ipfs_accelerate_py/mcp_server/mcplusplus/registry/static.py, test/api/test_mcplusplus_registry_static.py
- Interfaces: Registry@1, StaticRegistry@1
- Allow concurrent with: 
- Conflict policy: New registry package.
- Preconditions: Advertisement schema exists.
- Effects: Publish, refresh, withdraw, lookup by identity/interface CID/semantic capability/policy or proof, health-aware selection, deterministic tie-break, stale rejection.
- Evidence subset: AgentAdvertisement
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Static registry tests cover the listed operations. Selection is deterministic for equal health.
- Embedding query: static registry publish withdraw lookup

## MCPP-060 Add libp2p discovery and an isolated AGNTCY adapter package

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: discovery-libp2p
- Depends on: MCPP-059
- Goal id: MCPP-G110
- Outputs: ipfs_accelerate_py/mcp_server/mcplusplus/registry/libp2p.py, ipfs_accelerate_py/mcp_server/mcplusplus/registry/agntcy.py
- Validation: python -m pytest -q test/api/test_mcplusplus_registry_libp2p.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/discovery-libp2p
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcp_server/mcplusplus/registry/libp2p.py, ipfs_accelerate_py/mcp_server/mcplusplus/registry/agntcy.py
- Interfaces: Libp2pDiscovery@1, AgntcyAdapter@1
- Allow concurrent with: 
- Conflict policy: New adapters. If official AGNTCY SDK is unusable, keep the package isolated and document the blocker.
- Preconditions: Static registry exists.
- Effects: libp2p adapter publishes and looks up advertisements. AGNTCY is either working or clearly isolated.
- Evidence subset: existing p2p bootstrap, advertisement schema
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: libp2p adapter has a hermetic test. AGNTCY either passes a live-optional test or is marked unsupported with a typed reject.
- Embedding query: libp2p discovery agntcy adapter

## MCPP-061 Test registration expiry, withdrawal, duplicates, compromise, and selection

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: discovery-tests
- Depends on: MCPP-060
- Goal id: MCPP-G110
- Outputs: test/api/test_mcplusplus_registry.py
- Validation: python -m pytest -q test/api/test_mcplusplus_registry.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/discovery-tests
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: test/api/test_mcplusplus_registry.py
- Interfaces: RegistryAbuseVector@1
- Allow concurrent with: 
- Conflict policy: New adversarial registry tests.
- Preconditions: libp2p and static registries exist.
- Effects: Expiry, withdrawal, duplicates, compromised/unsigned records, and deterministic selection are tested. Registry presence is not execution authority.
- Evidence subset: Registry, UCAN verifier
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Compromised records are rejected. A matching advertisement without a valid delegation cannot dispatch.
- Embedding query: registry expiry compromise selection authority

## MCPP-062 Split Profile E into transport negotiation, MCP semantics, and execution authority

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: p2p-spec
- Depends on: MCPP-019, MCPP-021
- Goal id: MCPP-G120
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/transport-mcp-p2p.md
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_transport.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/p2p-spec
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/transport-mcp-p2p.md
- Interfaces: McpP2pBinding@1
- Allow concurrent with: 
- Conflict policy: Update the transport spec only.
- Preconditions: Abstract profiles and current MCP binding exist.
- Effects: Spec defines versioned libp2p protocol IDs, length-prefixed framing, max frame, backpressure, stream/peer quotas, correlation, in-flight requests, cancel, retry safety, replay, timeouts, rate limit, NAT/relay guidance, discovery, encrypted channels, capability advertisement, sessionless current MCP, legacy sessions only under legacy binding.
- Evidence subset: current transport spec, current MCP binding
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Readers can distinguish transport handshake from MCP application messages from UCAN authority. Current path is sessionless.
- Embedding query: profile e transport handshake mcp authority split

## MCPP-063 Implement deterministic framing, quotas, correlation, and replay detection

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: p2p-framing
- Depends on: MCPP-062
- Goal id: MCPP-G120
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/validators/transport.py, ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/p2p_framing.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_transport.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/p2p-framing
- Parallel lane: mcpp-lane-state
- Resource class: cpu-security
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/validators/transport.py, ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/p2p_framing.py
- Interfaces: LengthPrefixedFrame@1, TransportQuota@1
- Allow concurrent with: 
- Conflict policy: Framing implementation in spec validators and datasets transport helper.
- Preconditions: Updated Profile E spec exists.
- Effects: Length-prefixed framing, max size, quotas, correlation, multiple in-flight, cancel, replay detection, idle timeouts, and rate limiting are implemented.
- Evidence subset: transport spec
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Unit tests cover frame encode/decode, max size, and replay window.
- Embedding query: length prefixed frame quota replay correlation

## MCPP-064 Add adversarial P2P framing and abuse tests

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: p2p-abuse
- Depends on: MCPP-063
- Goal id: MCPP-G120
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_transport_abuse.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_transport_abuse.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/p2p-abuse
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-security
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_transport_abuse.py
- Interfaces: P2pAbuseVector@1
- Allow concurrent with: 
- Conflict policy: New abuse tests. Do not weaken quotas to pass.
- Preconditions: Framing implementation exists.
- Effects: Tests cover oversized, truncated, invalid length, request-before-negotiation, forged version, unknown method, empty success on transport failure, replay, flood, excessive streams, valid PeerID invalid UCAN, stale fence, duplicate response, wrong correlation id.
- Evidence subset: framing, UCAN verifier, fencing
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Every listed case fails closed. Empty success on transport failure is treated as failure.
- Embedding query: p2p abuse oversized replay flood peerid ucan

## MCPP-065 Bind datasets and kit P2P runtimes to the hardened Profile E

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: p2p-runtime
- Depends on: MCPP-064
- Goal id: MCPP-G120
- Outputs: ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/p2p_libp2p_transport.py, ipfs_kit_py/ipfs_kit_py/mcp_server, ipfs_datasets_py/tests/unit/mcp_server/test_mcplusplus_p2p_framing.py
- Validation: cd ipfs_datasets_py && python -m pytest -q tests/unit/mcp_server/test_mcplusplus_p2p_framing.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/p2p-runtime
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/p2p_libp2p_transport.py, ipfs_kit_py/ipfs_kit_py/mcp_server, ipfs_datasets_py/tests/unit/mcp_server/test_mcplusplus_p2p_framing.py
- Interfaces: RuntimeP2pAdapter@1
- Allow concurrent with: 
- Conflict policy: Runtime transport files only. Preserve untracked datasets P2P work by integrating or recording it, not deleting it.
- Preconditions: Abuse tests exist.
- Effects: Datasets and kit use versioned protocol IDs and the shared framing rules.
- Evidence subset: datasets p2p_libp2p_transport.py, kit mcp_server
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Runtime tests reuse the abuse vectors. Untracked datasets files are dispositioned, not discarded.
- Embedding query: datasets kit p2p profile e runtime

## MCPP-066 Publish one normative Profile G coordination specification

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: profile-g-spec
- Depends on: MCPP-011, MCPP-033, MCPP-035
- Goal id: MCPP-G130
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/risk-scheduling.md
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/spec/risk-scheduling.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/profile-g-spec
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/risk-scheduling.md
- Interfaces: ProfileGNormative@1
- Allow concurrent with: 
- Conflict policy: Reconcile registry, risk-scheduling.md, validators, and harnesses into one normative text.
- Preconditions: Profile inventory, envelope, and StateRef exist.
- Effects: Spec defines TaskSpec, ScheduleProposal, TaskClaim, ClaimResolution, Lease, LogicalEpoch, FencingToken, TaskAttempt, TaskReceipt, RiskEvidence, NeighborhoodRecord, Attestation plus conflict ordering, fitness, majority requirements, fail-closed, expiry, takeover, fencing, completion, duplicate suppression, reconciliation, restart, starvation, fairness, escalation.
- Evidence subset: existing risk-scheduling.md and profile G vectors
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Local coordination is explicitly not global consensus. Contradictory earlier text is marked superseded.
- Embedding query: profile g normative claim lease epoch fencing

## MCPP-067 Implement claims, leases, epochs, and fencing tokens

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: profile-g-fencing
- Depends on: MCPP-066
- Goal id: MCPP-G130
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/validators/profile_g.py, ipfs_accelerate_py/mcp_server/mcplusplus/profile_g.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_g_codec.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/profile-g-fencing
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/validators/profile_g.py, ipfs_accelerate_py/mcp_server/mcplusplus/profile_g.py
- Interfaces: TaskClaim@1, Lease@1, FencingToken@1
- Allow concurrent with: 
- Conflict policy: Validator and accelerate coordination helper.
- Preconditions: Normative Profile G spec exists.
- Effects: Claims, leases, epochs, and fencing are validated and enforced. Stale tokens cannot complete.
- Evidence subset: Profile G spec, StateRef leases
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Unit tests reject expired leases and stale fencing tokens.
- Embedding query: task claim lease epoch fencing token

## MCPP-068 Expand the three-peer Profile G harness

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: profile-g-harness
- Depends on: MCPP-067, MCPP-040
- Goal id: MCPP-G130
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/harness/profile_g_three_peer.py, ipfs_accelerate_py/mcplusplus/tests-py/integration/test_profile_g_three_peer.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_g_three_peer.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/profile-g-harness
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-large
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/harness/profile_g_three_peer.py, ipfs_accelerate_py/mcplusplus/tests-py/integration/test_profile_g_three_peer.py
- Interfaces: ThreePeerHarness@1
- Allow concurrent with: 
- Conflict policy: Expand the existing harness; do not create a second unofficial one.
- Preconditions: Fencing implementation and state non-merge tests exist.
- Effects: Harness covers simultaneous claims, partitions, duplicates, reorder, restart, expiry, takeover, late/alternate completion, malicious stale publisher, frontier reconciliation, state-root convergence, one authoritative completion, no policy bypass.
- Evidence subset: existing profile_g_three_peer harness
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Partition heal converges. Exactly one authoritative completion for an exclusive task. Stale publisher is rejected.
- Embedding query: three peer partition takeover stale publisher

## MCPP-069 Bind kit and accelerate Profile G runtimes and reject stale fenced completion

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: profile-g-runtime
- Depends on: MCPP-068
- Goal id: MCPP-G130
- Outputs: ipfs_kit_py/ipfs_kit_py/mcp_server, ipfs_accelerate_py/mcp/tests/test_mcplusplus_profile_g_fence.py
- Validation: python -m pytest -q ipfs_accelerate_py/mcp/tests/test_mcplusplus_profile_g_fence.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/profile-g-runtime
- Parallel lane: mcpp-lane-state
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_kit_py/ipfs_kit_py/mcp_server, ipfs_accelerate_py/mcp/tests/test_mcplusplus_profile_g_fence.py
- Interfaces: RuntimeProfileG@1
- Allow concurrent with: 
- Conflict policy: Runtime fencing enforcement only.
- Preconditions: Three-peer harness exists.
- Effects: Kit and accelerate reject stale fenced completion and do not fork Profile G fields.
- Evidence subset: harness, kit mcp_server, accelerate mcplusplus
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Runtime test publishes a stale fence and is denied. Field names match the normative spec.
- Embedding query: kit accelerate profile g stale fence

## MCPP-070 Separate Profile H payment success from execution authorization

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: profile-h-boundary
- Depends on: MCPP-011, MCPP-045
- Goal id: MCPP-G140
- Outputs: ipfs_accelerate_py/mcplusplus/docs/spec/x402-payments.md
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_h_codec.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/profile-h-boundary
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-security
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/spec/x402-payments.md
- Interfaces: PaymentAuthorizationBoundary@1
- Allow concurrent with: 
- Conflict policy: Spec clarification and validator checks. Do not grant capabilities from payment code.
- Preconditions: Profile inventory and signed receipts exist.
- Effects: A successful payment never grants execution authority. Authorized-but-unpaid is also deny when the profile is required.
- Evidence subset: existing profile-h schemas
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Validators reject paid-but-unauthorized and authorized-but-unpaid invocations.
- Embedding query: profile h payment not authorization

## MCPP-071 Harden quotes, settlement, entitlements, refunds, and replay protection

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: profile-h-settlement
- Depends on: MCPP-070
- Goal id: MCPP-G140
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/profile-h/1.0, ipfs_accelerate_py/mcplusplus/tests-py/validators/profile_h.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_h_codec.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/profile-h-settlement
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-security
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/profile-h/1.0, ipfs_accelerate_py/mcplusplus/tests-py/validators/profile_h.py
- Interfaces: PaymentQuote@1, SettlementReceipt@1, Entitlement@1
- Allow concurrent with: 
- Conflict policy: Harden existing Profile H schemas and validators.
- Preconditions: Payment/authorization split exists.
- Effects: Quotes, payment authorization, settlement, entitlements, usage, refunds, reconciliation, signed access receipts, idempotent settlement, replay protection, expiration, price-version binding, and required policy/delegation are implemented.
- Evidence subset: profile-h 1.0 schemas, x402 v2
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Idempotent settlement does not double-entitle. Expired quotes cannot settle. Price-version mismatch fails closed.
- Embedding query: x402 quote settlement entitlement refund replay

## MCPP-072 Add Profile H negative tests and SwissKnife/accelerate adapters

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: profile-h-adapters
- Depends on: MCPP-071
- Goal id: MCPP-G140
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_profile_h_negatives.py, docs/reports/mcplusplus-1.0-gap-closure/runtime/profile-h-adapters.md
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_profile_h_negatives.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/profile-h-adapters
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-security
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_profile_h_negatives.py, docs/reports/mcplusplus-1.0-gap-closure/runtime/profile-h-adapters.md
- Interfaces: ProfileHNegativeVector@1
- Allow concurrent with: 
- Conflict policy: New negatives plus adapter receipt. SwissKnife edits only in the bound checkout.
- Preconditions: Hardened settlement exists.
- Effects: Negatives cover paid-unauthorized, authorized-unpaid, replay, price mismatch, wrong recipient, duplicate settlement, expired quote, refund-after-consumed, forged settlement. x402 HTTP and libp2p carriage remain distinct.
- Evidence subset: Profile H schemas, SwissKnife profile-h adapter, accelerate
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: All negatives fail closed. Adapter receipt lists files and commands. Transport split is tested.
- Embedding query: profile h negatives swissknife accelerate x402 libp2p

## MCPP-073 Specify EncryptedArtifactRef and key-envelope semantics

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: confidential-schema
- Depends on: MCPP-030, MCPP-015
- Goal id: MCPP-G150
- Outputs: ipfs_accelerate_py/mcplusplus/schemas/confidential/encrypted-artifact-ref-1.schema.json, ipfs_accelerate_py/mcplusplus/docs/spec/cid-native-artifacts.md
- Validation: python -m json.tool ipfs_accelerate_py/mcplusplus/schemas/confidential/encrypted-artifact-ref-1.schema.json > /dev/null
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/confidential-schema
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-small
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/schemas/confidential/encrypted-artifact-ref-1.schema.json, ipfs_accelerate_py/mcplusplus/docs/spec/cid-native-artifacts.md
- Interfaces: EncryptedArtifactRef@1, KeyEnvelope@1
- Allow concurrent with: 
- Conflict policy: New confidential schema and spec additions.
- Preconditions: Envelope schemas and conformance-level ADR exist.
- Effects: Reference includes ciphertext CID, algorithm, key-envelope, recipients/capability, plaintext schema CID, optional protected digest, disclosure and retention policy, redaction metadata.
- Evidence subset: cid-native-artifacts.md
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Ciphertext is verifiable. Receipts can attest use without disclosure. Revocation behavior is documented honestly.
- Embedding query: encrypted artifact ref ciphertext key envelope

## MCPP-074 Prevent plaintext leaks in logs, Event DAG metadata, and persistence fallbacks

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: confidential-tests
- Depends on: MCPP-073
- Goal id: MCPP-G150
- Outputs: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_confidential_artifacts.py, ipfs_kit_py/tests/test_mcplusplus_confidential_persistence.py
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m pytest -q tests-py/integration/test_confidential_artifacts.py
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/confidential-tests
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-security
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/tests-py/integration/test_confidential_artifacts.py, ipfs_kit_py/tests/test_mcplusplus_confidential_persistence.py
- Interfaces: ConfidentialPersistenceReceipt@1
- Allow concurrent with: 
- Conflict policy: New tests and kit persistence hooks. Do not log plaintext in the test harness either.
- Preconditions: EncryptedArtifactRef exists.
- Effects: Tests cover unauthorized read, altered ciphertext, wrong recipient, revoked key access, and accidental plaintext persistence including cache/local fallback.
- Evidence subset: kit artifact store, Event DAG metadata
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: No tested persistence path writes plaintext. Altered ciphertext fails verify. Revoked key access fails closed.
- Embedding query: confidential plaintext leak cache fallback

## MCPP-075 Create the installable mcpp CLI

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: cli
- Depends on: MCPP-033, MCPP-045, MCPP-048
- Goal id: MCPP-G160
- Outputs: ipfs_accelerate_py/mcplusplus/cli/mcpp.py, ipfs_accelerate_py/mcplusplus/pyproject.toml
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m mcpp doctor
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/cli
- Parallel lane: mcpp-lane-state
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/cli/mcpp.py, ipfs_accelerate_py/mcplusplus/pyproject.toml
- Interfaces: McppCli@1
- Allow concurrent with: 
- Conflict policy: New CLI package in the spec repo. Do not reimplement validators.
- Preconditions: Envelope, receipt, and policy validators exist.
- Effects: Commands: inspect, validate, conformance, envelope create, envelope verify, receipt verify, artifact get, peer list, demo, doctor.
- Evidence subset: validators, schemas
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Each command has a --help path and a focused test. doctor reports binding, schema, and crypto suite versions.
- Embedding query: mcpp cli inspect validate envelope receipt doctor

## MCPP-076 Create the three-peer Docker Compose demonstration

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: demo-compose
- Depends on: MCPP-068, MCPP-056, MCPP-052, MCPP-075
- Goal id: MCPP-G160
- Outputs: ipfs_accelerate_py/mcplusplus/demo/docker-compose.yml, ipfs_accelerate_py/mcplusplus/demo/README.md
- Validation: cd ipfs_accelerate_py/mcplusplus && python -m mcpp demo --help
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/demo-compose
- Parallel lane: mcpp-lane-transport
- Resource class: io-large
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/demo/docker-compose.yml, ipfs_accelerate_py/mcplusplus/demo/README.md
- Interfaces: ThreePeerDemo@1
- Allow concurrent with: 
- Conflict policy: New demo compose files. No cloud-only dependency for the happy path.
- Preconditions: Three-peer harness, A2A handoff, crash recovery, and CLI exist.
- Effects: Compose brings up at least three peers and the demo script covers the 16 required demonstration steps.
- Evidence subset: CLI, harness, A2A adapter, durable adapter
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: README lists the 16 steps and the one-command invocation. Missing optional relays are documented, not silently skipped.
- Embedding query: docker compose three peer mcpp demo

## MCPP-077 Emit a machine-readable evidence bundle and independent verifier

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: demo-verifier
- Depends on: MCPP-076
- Goal id: MCPP-G160
- Outputs: ipfs_accelerate_py/mcplusplus/cli/verify_bundle.py, docs/reports/mcplusplus-1.0-gap-closure/demo/evidence-bundle.schema.json
- Validation: python -c "import pathlib; assert pathlib.Path('ipfs_accelerate_py/mcplusplus/cli/verify_bundle.py').exists()"
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/demo-verifier
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-medium
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/cli/verify_bundle.py, docs/reports/mcplusplus-1.0-gap-closure/demo/evidence-bundle.schema.json
- Interfaces: DemoEvidenceBundle@1, IndependentVerifier@1
- Allow concurrent with: 
- Conflict policy: New verifier and bundle schema.
- Preconditions: Demo compose exists.
- Effects: Bundle contains commit SHAs, schema versions, interface/envelope/policy/proof/decision/state/output/receipt/event CIDs, signatures, test results, external dependencies.
- Evidence subset: demo, receipts, Event DAG
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Verifier exits 0 on a valid bundle and nonzero if any CID or signature is tampered. It is a separate process from the demo peers.
- Embedding query: evidence bundle independent verifier cid signature

## MCPP-078 Write architecture, threat-model, trust-boundary, and migration docs

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: docs-architecture
- Depends on: MCPP-018, MCPP-077
- Goal id: MCPP-G170
- Outputs: ipfs_accelerate_py/mcplusplus/docs/architecture/overview.md, ipfs_accelerate_py/mcplusplus/docs/architecture/threat-model.md, ipfs_accelerate_py/mcplusplus/docs/architecture/trust-boundaries.md, ipfs_accelerate_py/mcplusplus/docs/architecture/state-model.md, ipfs_accelerate_py/mcplusplus/docs/architecture/durable-execution.md
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/architecture/threat-model.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/docs-architecture
- Parallel lane: mcpp-lane-spec
- Resource class: cpu-small
- Resource stage: documentation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/architecture/overview.md, ipfs_accelerate_py/mcplusplus/docs/architecture/threat-model.md, ipfs_accelerate_py/mcplusplus/docs/architecture/trust-boundaries.md, ipfs_accelerate_py/mcplusplus/docs/architecture/state-model.md, ipfs_accelerate_py/mcplusplus/docs/architecture/durable-execution.md
- Interfaces: ArchitectureGuide@1, ThreatModel@1
- Allow concurrent with: MCPP-079
- Conflict policy: Documentation files. Mark every section normative, non-normative, experimental, reference, production-admitted, or not production-admitted.
- Preconditions: ADRs and demo verifier exist.
- Effects: Overview, threat model, trust boundaries, state model, durable execution, execution envelope, state-ref, a2a extension, versioned MCP bindings, migration, compatibility matrix, conformance-level, deployment, three-peer, security disclosure, and release checklist exist or are linked.
- Evidence subset: implemented specs
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: No section uses forbidden over-claim phrases. Profile bundles Evidence Core, Secure Delegation, Federated Mesh, Commerce, and Verified Execution are defined.
- Embedding query: architecture threat model trust boundary migration

## MCPP-079 Rewrite stale coverage and production-readiness claims to match CI evidence

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: docs-stale-claims
- Depends on: MCPP-012, MCPP-078
- Goal id: MCPP-G170
- Outputs: ipfs_accelerate_py/mcplusplus/README.md, ipfs_accelerate_py/mcplusplus/docs/testing
- Validation: rg -n "production-ready|zero vulnerabilities|fully conformant|100 percent coverage" ipfs_accelerate_py/mcplusplus/README.md docs/testing || true
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/docs-stale-claims
- Parallel lane: mcpp-lane-schema
- Resource class: cpu-small
- Resource stage: documentation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/README.md, ipfs_accelerate_py/mcplusplus/docs/testing
- Interfaces: DocumentationHonestyReceipt@1
- Allow concurrent with: MCPP-078
- Conflict policy: Rewrite stale testing docs. Do not delete historical files without marking them historical.
- Preconditions: Traceability matrix and architecture docs exist.
- Effects: Stale 100-percent and production-ready documents are marked historical or rewritten to cite generated CI artifacts.
- Evidence subset: baseline receipts, CI (may still be pending)
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Current README does not claim 100 percent coverage or production readiness unless a CI artifact proves that exact claim.
- Embedding query: stale coverage production ready documentation rewrite

## MCPP-080 Add CI for all languages, vectors, crypto negatives, P2P abuse, demo, and scans

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: ci
- Depends on: MCPP-028, MCPP-044, MCPP-064, MCPP-077
- Goal id: MCPP-G170
- Outputs: ipfs_accelerate_py/mcplusplus/.github/workflows/mcplusplus-1.0.yml, .github/workflows/mcplusplus-1.0-gap-closure.yml
- Validation: python -c "import pathlib; assert pathlib.Path('ipfs_accelerate_py/mcplusplus/.github/workflows/mcplusplus-1.0.yml').exists() or pathlib.Path('.github/workflows/mcplusplus-1.0-gap-closure.yml').exists()"
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/ci
- Parallel lane: mcpp-lane-crypto
- Resource class: cpu-large
- Resource stage: validation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/.github/workflows/mcplusplus-1.0.yml, .github/workflows/mcplusplus-1.0-gap-closure.yml
- Interfaces: CiWorkflow@1
- Allow concurrent with: 
- Conflict policy: New workflow files. Do not lower existing coverage thresholds to go green.
- Preconditions: Cross-language identity, adversarial crypto, P2P abuse, and demo verifier exist.
- Effects: CI runs Python, TypeScript, Go, Rust, vectors, schema validation, canonicalization, crypto negatives, P2P abuse, three-peer, crash recovery, A2A, current+legacy bindings, vuln scan, secret scan, static analysis, fuzz smoke, doc/schema checks. Produces coverage, test artifacts, conformance matrix, SBOM, license inventory, release manifest, checksums.
- Evidence subset: existing workflows as observation
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Workflow is present and the report later cites actual run results. Unsigned release artifacts are allowed if signing credentials are absent and documented.
- Embedding query: github actions ci sbom fuzz secret scan

## MCPP-081 Close remaining runtime adapter and documentation gaps

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runtime-closeout
- Depends on: MCPP-023, MCPP-034, MCPP-049, MCPP-053, MCPP-057, MCPP-065, MCPP-069, MCPP-072
- Goal id: MCPP-G170
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/runtime/adapter-matrix.md
- Validation: test -s docs/reports/mcplusplus-1.0-gap-closure/runtime/adapter-matrix.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/runtime-closeout
- Parallel lane: mcpp-lane-state
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/runtime/adapter-matrix.md
- Interfaces: RuntimeAdapterMatrix@1
- Allow concurrent with: 
- Conflict policy: Fill only remaining runtime gaps listed by the matrix. Do not duplicate spec implementations.
- Preconditions: Binding, envelope, policy, durable, A2A, P2P, Profile G, and Profile H runtime tasks have landed or recorded blockers.
- Effects: Accelerate, datasets, kit, and SwissKnife each have a current adapter disposition: implemented, partial, or blocked with evidence.
- Evidence subset: prior runtime tasks
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Matrix lists each runtime and profile. No runtime is marked implemented without a command. SwissKnife remote remains the discovered origin.
- Embedding query: runtime adapter matrix accelerate datasets kit swissknife

## MCPP-082 Write the implementation report and draft pull-request descriptions

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: report
- Depends on: MCPP-079, MCPP-080, MCPP-081
- Goal id: MCPP-G170
- Outputs: ipfs_accelerate_py/mcplusplus/docs/reports/MCPPLUSPLUS_1_0_IMPLEMENTATION_REPORT.md, docs/reports/mcplusplus-1.0-gap-closure/DRAFT_PULL_REQUESTS.md
- Validation: test -s ipfs_accelerate_py/mcplusplus/docs/reports/MCPPLUSPLUS_1_0_IMPLEMENTATION_REPORT.md
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/report
- Parallel lane: mcpp-lane-transport
- Resource class: cpu-medium
- Resource stage: documentation
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/mcplusplus/docs/reports/MCPPLUSPLUS_1_0_IMPLEMENTATION_REPORT.md, docs/reports/mcplusplus-1.0-gap-closure/DRAFT_PULL_REQUESTS.md
- Interfaces: ImplementationReport@1
- Allow concurrent with: 
- Conflict policy: New report files only. If GitHub credentials exist, open draft PRs without merging or force-pushing.
- Preconditions: Docs honesty, CI, and runtime matrix exist.
- Effects: Report includes every required section. Draft PR text covers problem, architecture, major changes, compatibility, security, tests, deploy, remaining risks.
- Evidence subset: all prior evidence
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Recommendation is NO-GO, RC, or GO with evidence. Items not claimed are explicit. Migration instructions exist. User overlays remain intact.
- Embedding query: implementation report draft pull request release recommendation

## MCPP-083 Publish the joined release-candidate receipt

- Status: todo
- Completion: authoritative
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: terminal
- Depends on: MCPP-082
- Goal id: MCPP-G170
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/MCPPLUSPLUS_1_0_RELEASE_RECEIPT.json
- Validation: python scripts/validate_mcplusplus_1_0_gap_closure_board.py --check-all
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Bundle: mcplusplus/1.0/terminal
- Parallel lane: mcpp-lane-runtime
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 22000
- Implementation timeout seconds: 3600
- Predicted files: docs/reports/mcplusplus-1.0-gap-closure/MCPPLUSPLUS_1_0_RELEASE_RECEIPT.json
- Interfaces: MCPPlusPlusReleaseReceipt@1
- Allow concurrent with: 
- Conflict policy: Review-only joined receipt. Do not implement new subsystem behavior.
- Preconditions: Implementation report exists and all prior tasks are terminal or explicitly blocked with evidence.
- Effects: Receipt binds child goal revisions, schema versions, validation CIDs, gate results, and the NO-GO/RC/GO decision.
- Evidence subset: implementation report, CI artifacts, demo bundle
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: All 28 acceptance gates are listed with command, result, and artifact path. No gate is passed without evidence. This task does not merge PRs.
- Embedding query: joined release receipt acceptance gates no-go rc go

## MCPP-084 Resolve validation retry-budget failure for MCPP-001

- Status: completed
- Completion: manual
- Completion evidence: Provider route required fallback_reasoning_effort=high; scheduler updated and MCPP-001 released from strategy blocked_tasks
- Canonical board task: false
- Is schedulable: true
- Review only: true
- Priority: P1
- Track: ops
- Depends on: MCPP-000
- Goal id: MCPP-G010
- Board namespace: mcplusplus-1-0-gap-closure-v1
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/repository-forest.json, docs/reports/mcplusplus-1.0-gap-closure/baseline/dirty-overlay.md, data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery
- Validation: test -f /home/barberb/lift_coding/.worktrees/ipfs-accelerate-mcplusplus-1.0-gap-closure/data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery/2026-08-15-mcpp-084-mcpp-001-retry-budget.md

- Acceptance: Retry-budget guardrail filed this from repeated validation failures in MCPP-001. Use evidence in /home/barberb/lift_coding/.worktrees/ipfs-accelerate-mcplusplus-1.0-gap-closure/data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery/2026-08-15-mcpp-084-mcpp-001-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release MCPP-001 from strategy blocked_tasks.

## MCPP-085 Resolve validation retry-budget failure for MCPP-007

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: MCPP-001
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/ipfs-datasets-mcplusplus.json, data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery
- Validation: export PYTHONPATH="$PWD"/ipfs_accelerate_py/mcplusplus:"$PWD"/ipfs_datasets_py:"$PWD"/ipfs_kit_py; cd ipfs_datasets_py && python -m pytest -q tests/unit/mcp_server -k mcplusplus --maxfail=1

- Acceptance: Retry-budget guardrail filed this from repeated validation failures in MCPP-007. Use evidence in /home/barberb/lift_coding/.worktrees/ipfs-accelerate-mcplusplus-1.0-gap-closure/data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery/2026-08-15-mcpp-085-mcpp-007-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release MCPP-007 from strategy blocked_tasks.

## MCPP-086 Resolve validation retry-budget failure for MCPP-005

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: MCPP-001
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/mcpplusplus-rust.json, data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery
- Validation: cd ipfs_accelerate_py/mcplusplus/tests-rs && cargo test

- Acceptance: Retry-budget guardrail filed this from repeated validation failures in MCPP-005. Use evidence in /home/barberb/lift_coding/.worktrees/ipfs-accelerate-mcplusplus-1.0-gap-closure/data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery/2026-08-15-mcpp-086-mcpp-005-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release MCPP-005 from strategy blocked_tasks.

## MCPP-087 Resolve validation retry-budget failure for MCPP-006

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: MCPP-001
- Outputs: docs/reports/mcplusplus-1.0-gap-closure/baseline/ipfs-accelerate-mcplusplus.json, data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery
- Validation: export PYTHONPATH="$PWD"/ipfs_accelerate_py/mcplusplus:"$PWD"/ipfs_datasets_py:"$PWD"/ipfs_kit_py; python -m pytest -q ipfs_accelerate_py/mcp/tests test/api -k mcplusplus --maxfail=1

- Acceptance: Retry-budget guardrail filed this from repeated validation failures in MCPP-006. Use evidence in /home/barberb/lift_coding/.worktrees/ipfs-accelerate-mcplusplus-1.0-gap-closure/data/agent_supervisor/mcplusplus_1_0_gap_closure/state/discovery/2026-08-15-mcpp-087-mcpp-006-retry-budget.md to fix the validation blocker, then mark this repair task completed so the supervisor can release MCPP-006 from strategy blocked_tasks. The declared validation target paths (ipfs_accelerate_py/mcp/tests, test/api) are bounded diagnostic and repair scope: change them only when evidence proves inherited validation debt, and do not weaken correct assertions or policy.
