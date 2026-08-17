# LPC-002: Datasets public logic APIs and compatibility shims

**Task:** LPC-002  
**Goal:** LPC-G010  
**Schema:** `logic-platform-canonicalization/datasets-public-api-inventory@1`  
**Machine-readable companion:** [`datasets_public_api.json`](./datasets_public_api.json)

## Purpose

Classify every public import from:

- `ipfs_datasets_py.logic` (`logic.__init__`)
- `ipfs_datasets_py.logic.api`
- `ipfs_datasets_py.logic.verification_api`
- datasets logic CLI (`logic.cli`)
- datasets logic MCP (`logic_verification`, `logic_tools`, related dataset conversion tools)

Classification vocabulary (LPC-G010): `canonical`, `canonical_component`,
`compatibility_facade`, `legacy`, `experimental`, `declaration_only`,
`generated`, `duplicate`, `obsolete`, `unresolved`.

## Source authority and inspection limits

| Field | Value |
| --- | --- |
| Repository | `ipfs_datasets_py` |
| Reviewed baseline | `ac82107e246b30e35a2bbdcf75e01370d22350c6` |
| Current head (implementation authority) | `ac82107e246b30e35a2bbdcf75e01370d22350c6` |
| Intended external checkout | `/home/barberb/lift_coding/external/ipfs_datasets` |
| Nested submodule path | `ipfs_datasets_py/` |

**Inspection note:** In this worktree the nested `ipfs_datasets_py` checkout is
empty and the external datasets path is unavailable. Classifications are bound
to the reviewed datasets revision through accelerate-side parity contracts,
the offline toolchain lock `stable_python_operations` list, LFV completion
receipts, and documented public surfaces that require those exact exports.
LPC-001 owns revision deltas; this slice owns public API classification.

Primary evidence anchors:

- `config/formal_verification_toolchains.lock.json` (`stable_python_operations`)
- `docs/formal_verification_tactician.md` (public API + CLI/MCP maps)
- `test/api/test_root_mcp_formal_verification_parity.py`
- `test/api/test_goal_tactician_cli_mcp_parity.py`
- `test/api/test_logic_verification_api_install_provider.py`
- `test/packaging/test_logic_verification_clean_install.py`
- LFV G070/G071 objectives and completion receipt
- parent MCP delegates under `ipfs_accelerate_py/mcp_server/tools/logic_tools/`

## Summary

| Metric | Count |
| --- | --- |
| Inventory items | 59 |
| Canonical | 29 |
| Canonical component | 9 |
| Compatibility facade | 4 |
| Legacy | 11 |
| Declaration-only | 4 |
| Duplicate | 1 |
| Obsolete | 1 |

By surface:

| Surface | Items |
| --- | --- |
| `logic.verification_api` | 33 |
| MCP | 17 |
| `logic.__init__` | 3 |
| `logic.api` | 3 |
| CLI | 3 |

By inventory category (for LPC-008 composition):

| Category | Items |
| --- | --- |
| `public_logic_api` | 34 |
| `compatibility_shim` | 14 |
| `mcp_cli_python_exposure` | 10 |
| `deprecated_module` | 1 |

## Interfaces

| Interface | Role |
| --- | --- |
| `LogicVerificationAPI@1` | Stable Python software-verification facade |
| `LogicVerificationCLI@1` | CLI channel for the same operations |
| `LogicVerificationMCP@1` | MCP channel for the same operations |
| `GoalTacticianAPI@1` | Public goal-tactician Python operations |
| `GoalTacticianCLIMCP@1` | CLI/MCP parity for goal tactician |
| `LogicAPICompatibility@1` | Frozen legacy FOL/deontic/family surface |
| `FormalVerificationMCPParity@1` | Python / datasets MCP / root MCP envelope parity |
| `ExecutableProviderMatrix@1` | Declarative executable provider matrix identity |
| `LogicVerificationLazyInstaller@1` | Explicit `install_provider` mutation boundary |

## `logic.__init__`

| Symbol | Classification | Role |
| --- | --- | --- |
| `ipfs_datasets_py.logic` | compatibility facade | Package root; thin lazy re-export surface; pure-data import must not probe or install |
| `logic.submodule_registry` | canonical component | Declarative lazy submodule catalog (`logic_submodule_names/specs`, integration manifest helpers) |
| `logic.__getattr__` | compatibility facade | Lazy attribute resolution without eager optional imports |

LPC-G060 keeps `logic.__init__` as a thin adapter over future `logic/platform/*`
services. It is not a second implementation.

## `logic.api` (legacy / compatibility)

| Symbol | Classification | Role |
| --- | --- | --- |
| `ipfs_datasets_py.logic.api` | compatibility facade | Frozen LogicAPICompatibility@1 module |
| `convert_text_to_fol` | legacy | NL → FOL conversion; also MCP `text_to_fol` |
| `convert_legal_text_to_deontic` | legacy | Legal text → deontic; also MCP `legal_text_to_deontic` |

Related implementation modules remain family-local
(`logic.fol`, `logic.deontic`) and are compatibility targets, not the new-write
path. New software-verification writes go through `verification_api`.

## `logic.verification_api` (canonical)

### Module and types

| Symbol | Classification | Role |
| --- | --- | --- |
| `logic.verification_api` | canonical | Side-effect-free public facade module |
| `LogicVerificationAPI` | canonical | Primary class API |
| `get_verification_api` | canonical component | Factory/accessor (`reset=` for hermetic tests) |
| `VerificationStatus` | canonical component | Operation-status axis |
| `VerificationAuthority` | canonical component | Semantic authority ceiling |

Importing the module never probes the environment, installs tools, opens the
network, or mutates disk.

### Stable operations (`STABLE_OPERATIONS`)

Pinned by `config/formal_verification_toolchains.lock.json` and packaging gate
tests:

| Operation | Classification | Authority ceiling | MCP tool |
| --- | --- | --- | --- |
| `list_logic_families` | declaration_only | declarative | `verification_list_logic_families` |
| `list_providers` | declaration_only | declarative | `verification_list_providers` |
| `provider_capabilities` | declaration_only | declarative | `verification_provider_capabilities` |
| `compile_verification_artifact` | canonical | source translation | `verification_compile_verification_artifact` |
| `check` | canonical | as returned | `verification_check` |
| `monitor` | canonical | monitor | `verification_monitor` |
| `run_portfolio` | canonical | per provider | `verification_portfolio` |
| `explain_counterexample` | canonical | none / bounded witness | `verification_explain_counterexample` |
| `verify_receipt` | canonical | structure attestation | `verification_verify_receipt` |
| `attest_receipt` | canonical | attestation (no raise) | `verification_attest_receipt` |
| `advise` | canonical | advisory | `verification_advise` |
| `probe_provider` | canonical | capability health | `verification_probe_provider` |
| `install_provider` | canonical | capability health / installer | `verification_install_provider` |

Additional public discovery operation (not in lock `stable_python_operations`):

| Operation | Classification | MCP tool |
| --- | --- | --- |
| `list_features` | declaration_only | `verification_list_features` |

`list_features` must return a superset of `STABLE_OPERATIONS`.

### Goal tactician operations (`GoalTacticianAPI@1`)

Channel-neutral map (Python = CLI = MCP):

| Python | CLI | MCP tool | Classification |
| --- | --- | --- | --- |
| `formalize_goal` | `goal-formalize` | `goal_tactician_formalize_goal` | canonical |
| `compare_interpretations` | `goal-compare-interpretations` | `goal_tactician_compare_interpretations` | canonical |
| `discover_missing_proofs` | `goal-discover-missing-proofs` | `goal_tactician_discover_missing_proofs` | canonical |
| `plan_proof` | `goal-plan-proof` | `goal_tactician_plan_proof` | canonical |
| `validate_proof_candidate` | `goal-validate-candidate` | `goal_tactician_validate_proof_candidate` | canonical |
| `execute_proof_plan` | `goal-execute-plan` | `goal_tactician_execute_proof_plan` | canonical |
| `proof_status` | `goal-proof-status` | `goal_tactician_proof_status` | canonical |
| `minimize_counterexample` | `goal-minimize-counterexample` | `goal_tactician_minimize_counterexample` | canonical |
| `explain_counterexample_causal` | `goal-explain-counterexample` | `goal_tactician_explain_counterexample_causal` | canonical |
| `replay_counterexample` | `goal-replay-counterexample` | `goal_tactician_replay_counterexample` | canonical |
| `list_goal_tactician_operations` | `goal-list-operations` | `goal_tactician_list_operations` | canonical |

Supporting public helpers (canonical components):

- `list_goal_tactician_cli_mcp_surface`
- `goal_tactician_tool_schemas`
- `invoke_goal_tactician`
- `invoke_goal_tactician_cli`
- `invoke_goal_tactician_mcp_tool`

Supervisor-only controls are rejected on all public channels:
`admit_goal`, `close_plan`, `mutate_supervisor`, `force_complete`,
`lease_steal`, `rewrite_event_log`, `bypass_resource_policy`,
`promote_proof_authority`, `supervisor_mutate`, `supervisor_only`.

Transport success never implies proof success. `list_goal_tactician_cli_mcp_surface`
preserves `STABLE_OPERATIONS` under `legacy_operations_preserved`.

## CLI (`logic.cli`)

| Item | Classification | Interface |
| --- | --- | --- |
| `ipfs_datasets_py.logic.cli` | canonical | `LogicVerificationCLI@1` |
| `goal-*` command group | canonical | `GoalTacticianCLIMCP@1` |
| stable verification command group | canonical | `LogicVerificationCLI@1` |

Invocation pattern documented for operators:

```bash
ipfs-datasets logic goal-list-operations
ipfs-datasets logic goal-formalize --request goal.json
ipfs-datasets logic goal-replay-counterexample --request cex.json
```

LFV-G071 requires CLI coverage for list, capability, compile, check, monitor,
portfolio, counterexample, receipt, advisor, and attestation operations with
the same envelope as Python.

## MCP

### Canonical formal verification (`logic_verification`)

| Item | Classification | Interface |
| --- | --- | --- |
| `mcp_server.tools.logic_verification` | canonical | `LogicVerificationMCP@1` |
| `verification_*` tools | canonical | `LogicVerificationMCP@1` |
| `goal_tactician_*` tools | canonical | `GoalTacticianCLIMCP@1` |
| `logic_tools` re-export package | compatibility facade | — |

Required tool presence (parity tests):

- `verification_list_providers`
- `verification_provider_capabilities`
- `verification_check`
- `verification_portfolio` → operation `run_portfolio`
- `verification_probe_provider`
- `verification_list_features`

Envelope keys: `status`, `authority`, `operation`, `result`, `assumptions`,
`bounds`, `translations`, `witnesses`, `unsupported_features`, `diagnostics`,
`cache`, `interface` (`LogicVerificationAPI@1`).

### Legacy family tools (`logic_tools` and related)

| Tool group | Classification | Notes |
| --- | --- | --- |
| `logic_capabilities`, `logic_health` | legacy | Family-weighted processor health/capabilities |
| `tdfol_*` | legacy | TDFOL parse/convert/prove/KB/visualize; not FOL |
| `cec_*` | legacy | CEC/DCEC prove/parse/analysis; not generic deontic |
| `flogic_*` | legacy | F-logic assert/query/normalize; not object framing |
| `temporal_deontic_logic_tools.*` | legacy | Caselaw temporal-deontic helpers |
| `logic_graphrag_tool.*` | legacy | GraphRAG build/verify helpers |
| `text_to_fol` (dataset_tools) | legacy | MCP mirror of `logic.api.convert_text_to_fol` |
| `legal_text_to_deontic` (dataset_tools) | legacy | MCP mirror of `logic.api.convert_legal_text_to_deontic` |

### Deprecated / duplicate shims

| Item | Classification | Role |
| --- | --- | --- |
| `mcp_server.tools.legacy_mcp_tools` temporal-deontic routes | obsolete | Routes into canonical `logic_tools` |
| `ipfs_accelerate_py...native_logic_tools` | duplicate | Parent delegate/fallback; not a second semantic authority |

## Compatibility and migration posture

1. **New writes** use `LogicVerificationAPI@1` / CLI / MCP verification tools.
2. **Legacy family tools and `logic.api` converters** remain available as thin
   adapters under LogicAPICompatibility@1.
3. **Goal tactician** is additive: it must not remove STABLE_OPERATIONS coverage.
4. **LPC-G060** will split `verification_api` into internal `logic/platform/*`
   services while preserving these public imports as facades.
5. **LPC-062** owns further thinning of `logic.api` and `logic.__init__`
   compatibility facades after this inventory closes.

## Gaps recorded (not blocking classification)

| Gap | Status |
| --- | --- |
| Nested datasets sources not populated in this worktree | Classified via binding contracts; re-verify against live AST when checkout is present |
| Exact kebab-case CLI tokens for every non-goal verification subcommand | Classified as operation group under LogicVerificationCLI@1; tokens are operation-derived and LFV-G071-covered |
| Exhaustive `__all__` dump of every lazy family submodule export | Covered by `submodule_registry` classification; family modules themselves belong to LPC-003/LPC-004/LPC-007 |

## Acceptance

Every public import from `logic.__init__`, `logic.api`, `logic.verification_api`,
CLI, and MCP is classified in this document and in
`datasets_public_api.json`.
