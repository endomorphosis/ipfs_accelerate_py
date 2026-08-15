# LPC-130 LogicOperationCatalog@1 — Python, CLI, and MCP parity

**Task:** LPC-130 — Python, CLI, and MCP parity tests  
**Goal:** LPC-G130  
**Depends on:** LPC-062 (outer facades), LPC-110 (`SupervisorLogicPlatformClient@1`)  
**Interface:** `LogicOperationCatalog@1`  
**Canonical service / facade:** `ipfs_datasets_py.logic.verification_api`  
  (`LogicVerificationAPI@1`, `GoalTacticianAPI@1`, `GoalTacticianCLIMCP@1`)  
**Channel modules:**  
- Python: `ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py`  
- CLI: `ipfs_datasets_py/ipfs_datasets_py/logic/cli.py` (`LogicVerificationCLI@1`)  
- MCP: `ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_verification.py`  
  (`LogicVerificationMCP@1`)  
**Acceptance:** Channels agree on names, schemas, status, authority, failure
codes, and opt-in. Installation is not an ordinary verify operation.
Supervisor-only mutation controls are not exposed from datasets logic.  
**Conflict policy:** Own catalog projection and parity tests. Do not
add a new MCP++ profile.  
**Validation:**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/test_channel_parity.py test/api/test_logic_channel_parity.py -q`

## Purpose

Every public channel that can invoke datasets logic verification must agree on
one operation catalog derived from the canonical service. Python, CLI, and MCP
are **projections** of that catalog, not independent inventories.

`LogicOperationCatalog@1` freezes:

1. Closed operation names for stable verification and goal-tactician surfaces.
2. Channel name maps (Python method / MCP tool / CLI command) onto those names.
3. Shared response envelope, status vocabulary, authority ceilings, and
   request/response schemas.
4. Opt-in boundaries (`probe_provider`, `install_provider`, `attest_receipt`).
5. Forbidden supervisor-only mutation controls that datasets surfaces must never
   expose.
6. Non-goals: installation is not ordinary verify; transport success is not
   proof success; catalog presence is not provider availability.

This note is the durable LPC-130 / LPC-G130 evidence artifact. Automated
channel closure is enforced by the unit and API parity suites listed under
Validation.

## Canonical projection path

```text
LogicVerificationAPI@1 / GoalTacticianAPI@1
  (STABLE_OPERATIONS ∪ GOAL_TACTICIAN_OPERATIONS + shared envelope)
        │
        ├─ Python  verification_api methods / module wrappers
        │
        ├─ MCP     LogicVerificationMCP@1  TOOL_TO_OPERATION
        │          GoalTacticianCLIMCP@1   GOAL_TACTICIAN_TOOL_TO_OPERATION
        │
        └─ CLI     LogicVerificationCLI@1  command → MCP/Python dispatch
                   GoalTacticianCLIMCP@1   GOAL_TACTICIAN_CLI_TO_OPERATION
```

Discovery helpers that project the catalog without probing:

| Helper | Role |
| --- | --- |
| `LogicVerificationAPI.list_features` | Declares `STABLE_OPERATIONS` + feature descriptors |
| `verification_capabilities` (MCP/CLI) | Returns tools, operations, `tool_to_operation`, bounds |
| `list_goal_tactician_cli_mcp_surface` | Returns goal-tactician ops, tools, CLI map, forbidden controls |
| Supervisor client `catalog(...)` (LPC-110) | Sealed catalog root/digest via handshake-gated client |

## Shared response envelope

All channels return the same public envelope keys (channel markers may add
`mcp_interface`, `cli_interface`, `channel`, `python_operation` without
changing semantic identity):

| Field | Role |
| --- | --- |
| `status` | Terminal status (`VerificationStatus`) |
| `authority` | Authority ceiling (`VerificationAuthority`); never silently upgraded |
| `operation` | Canonical Python operation name |
| `result` | Operation payload (mapping) |
| `assumptions` | Assumption ids / residual assumptions |
| `bounds` | Resource / size / step bounds |
| `translations` | Translation residual / loss records |
| `witnesses` | Counterexample / model witnesses (public fields only) |
| `unsupported_features` | Explicit unsupported set |
| `diagnostics` | Secret-safe diagnostics |
| `cache` | Cache provenance |
| `interface` | `LogicVerificationAPI@1` (goal-tactician responses may also set goal-tactician interface fields) |

Schema constants:

* Response: `logic-verification-response/v1` (`LOGIC_VERIFICATION_RESPONSE_SCHEMA` /
  `GOAL_TACTICIAN_RESPONSE_SCHEMA`)
* Request: `logic-verification-request/v1`, `goal-tactician-request/v1`
* MCP surface: `logic-verification-mcp/v1`
* Parity interface marker: `FormalVerificationMCPParity@1`

### Status vocabulary (`VerificationStatus`)

| Status | Meaning |
| --- | --- |
| `succeeded` | Operation completed under declared authority |
| `partial` | Bounded partial result |
| `unsupported` | Feature/family explicitly unsupported |
| `unavailable` | Provider/tool unavailable (not success by silence) |
| `invalid` | Malformed request / admission failure |
| `error` | Execution error without authority upgrade |
| `declarative` | Catalog/discovery only; no live probe |

### Authority vocabulary (`VerificationAuthority`)

Closed ceilings include at least: `none`, `advisory`, `bounded`,
`satisfiability`, `model_check`, `monitor`, `authorization`, `protocol`,
`hyperproperty`, `candidate`, `reconstruction`, `attestation`, `theorem`,
`declarative`. Discovery defaults to `declarative`; success never upgrades
authority beyond what the residual and evidence admit (LPC-032).

### Failure / opt-in codes (channel-neutral)

| Class | Behavior |
| --- | --- |
| Malformed request | `VerificationAPIError` / status `invalid`; envelope diagnostics |
| Unsupported / unavailable | Explicit status; never silent success |
| Opt-in probe | `probe_provider` only; discovery never probes |
| Opt-in install | `install_provider` requires `allow_install=True` (CLI `--allow-install`); dry-run / offline stay non-mutating |
| Opt-in attest | `attest_receipt` is explicit; not ordinary verify |
| Supervisor-only controls | Rejected on goal-tactician public surfaces (`_GOAL_TACTICIAN_FORBIDDEN_CONTROLS`) |

## Catalog A — Stable verification operations (`STABLE_OPERATIONS`)

Interface: `LogicVerificationAPI@1` / `LogicVerificationMCP@1` /
`LogicVerificationCLI@1`.

| Python operation | MCP tool | CLI command | Kind | Opt-in |
| --- | --- | --- | --- | --- |
| `list_logic_families` | `verification_list_logic_families` | `list-families` | discovery | no |
| `list_providers` | `verification_list_providers` | `list-providers` | discovery | no |
| `provider_capabilities` | `verification_provider_capabilities` | `provider-capabilities` | discovery | no |
| `compile_verification_artifact` | `verification_compile` | `compile` | compile | no |
| `check` | `verification_check` | `check` | verify | no |
| `monitor` | `verification_monitor` | `monitor` | verify/monitor | no |
| `run_portfolio` | `verification_portfolio` | `portfolio` | plan | no |
| `explain_counterexample` | `verification_explain_counterexample` | `counterexample` | explain | no |
| `verify_receipt` | `verification_verify_receipt` | `verify-receipt` | receipt | no |
| `attest_receipt` | `verification_attest_receipt` | `attest-receipt` | attest | **yes** |
| `advise` | `verification_advise` | `advise` | advisory | no |
| `probe_provider` | `verification_probe_provider` | `probe-provider` | probe | **yes** |
| `install_provider` | `verification_install_provider` | `install-provider` | **install mutation** | **yes** |

### Discovery helpers (not merged into `STABLE_OPERATIONS` identity set)

| Python / role | MCP tool | CLI command | Notes |
| --- | --- | --- | --- |
| `list_features` | `verification_list_features` | `list-features` | Declares stable ops + features |
| `list_features` (alias) | `verification_capabilities` | `verification-capabilities` | Surface/bounds projection; also lists `STABLE_OPERATIONS` |

MCP `TOOL_TO_OPERATION` maps both `verification_list_features` and
`verification_capabilities` to `list_features`. CLI verification commands
dispatch through the MCP tool layer except `install-provider`, which calls the
Python facade directly with the local operator boundary (`allow_install`).

### Installation is not ordinary verify

`install_provider` is the sole public mutation boundary for reviewed prover
installation (`LogicVerificationLazyInstaller@1`). Rules retained:

1. Not treated as an ordinary verification success path.
2. Requires explicit opt-in (`allow_install=True` / CLI `--allow-install`).
3. Inventory, dry-run, offline, and probe paths never import a plugin or open
   the network as a side effect of discovery.
4. Install success never promotes capability or semantic/proof authority.
5. Without opt-in, channels return status `unsupported` with
   `authorization_required` / `install_without_opt_in` (never `succeeded`).
6. Supervisor-only mutation controls remain outside this catalog.

## Catalog B — Goal tactician operations (`GOAL_TACTICIAN_OPERATIONS`)

Interface: `GoalTacticianAPI@1` / `GoalTacticianCLIMCP@1`.  
**Not** merged into `STABLE_OPERATIONS` so legacy `LogicVerificationMCP@1`
mappings stay intact.

| Python operation | MCP tool | CLI command |
| --- | --- | --- |
| `formalize_goal` | `goal_tactician_formalize_goal` | `goal-formalize` |
| `compare_interpretations` | `goal_tactician_compare_interpretations` | `goal-compare-interpretations` |
| `discover_missing_proofs` | `goal_tactician_discover_missing_proofs` | `goal-discover-missing-proofs` |
| `plan_proof` | `goal_tactician_plan_proof` | `goal-plan-proof` |
| `validate_proof_candidate` | `goal_tactician_validate_proof_candidate` | `goal-validate-candidate` |
| `execute_proof_plan` | `goal_tactician_execute_proof_plan` | `goal-execute-plan` |
| `proof_status` | `goal_tactician_proof_status` | `goal-proof-status` |
| `minimize_counterexample` | `goal_tactician_minimize_counterexample` | `goal-minimize-counterexample` |
| `explain_counterexample_causal` | `goal_tactician_explain_counterexample_causal` | `goal-explain-counterexample` |
| `replay_counterexample` | `goal_tactician_replay_counterexample` | `goal-replay-counterexample` |
| `list_goal_tactician_operations` | `goal_tactician_list_operations` | `goal-list-operations` |

Maps are sealed as `GOAL_TACTICIAN_TOOL_TO_OPERATION` and
`GOAL_TACTICIAN_CLI_TO_OPERATION` in `verification_api.py`. Projection helper:
`list_goal_tactician_cli_mcp_surface()`.

### Forbidden supervisor-only controls

Datasets goal-tactician public surfaces reject these controls (closed set
`_GOAL_TACTICIAN_FORBIDDEN_CONTROLS`):

`admit_goal`, `close_plan`, `mutate_supervisor`, `force_complete`,
`lease_steal`, `rewrite_event_log`, `bypass_resource_policy`,
`promote_proof_authority`, `supervisor_mutate`, `supervisor_only`.

They must never appear as ordinary verification or goal-tactician operations
on Python, CLI, or MCP channels.

## Additive surfaces (declared, not stable MCP identity)

These remain separate closed vocabularies so they do not break
`LogicVerificationMCP@1` / `STABLE_OPERATIONS` parity:

| Surface | Operations (representative) | Rule |
| --- | --- | --- |
| `VerificationAPI@2` / `CanonicalLogicDiscovery@1` | `list_namespaces`, `list_namespace_identities`, `dual_read_label`, `canonical_write_label`, `migrate_artifact`, `inspect_translation_loss`, `inspect_provider_authority` | Migration dual-read; not merged into `STABLE_OPERATIONS` |
| `LogicVerificationProviderRoleClosure@1` | `list_provider_roles`, `provider_role`, SecPAL intake/lookup | Role closure; not ordinary verify |
| Accelerate native `logic_tools` (TDFOL/CEC) | family tools | **Compatibility / legacy**; not `LogicVerificationMCP@1` |

## Parity invariants

1. **One name per operation.** Channel renames are projection labels only;
   `operation` in the envelope is the Python canonical name.
2. **Schema agreement.** Request/response schemas and envelope keys match
   across channels for the same operation.
3. **Status and authority agreement.** Same status/authority vocabularies;
   no channel-local silent upgrade.
4. **Opt-in agreement.** Probe, install, and attest remain opt-in on every
   channel.
5. **Install ≠ verify.** Installation evidence is mutation evidence, not proof.
6. **No supervisor mutation bleed.** Forbidden controls stay out of datasets
   logic channels.
7. **Transport success ≠ proof success.** MCP/CLI HTTP-style success never
   implies kernel proof authority (LPC-032).
8. **Presence ≠ availability.** Catalog and capability declarations never
   claim live prover availability without probe.

## Automated parity evidence

| Suite | Role |
| --- | --- |
| `ipfs_datasets_py/tests/unit/logic/test_channel_parity.py` | Datasets-side channel closure: names, schemas, status/authority vocabularies, opt-in, install≠verify, forbidden controls |
| `test/api/test_logic_channel_parity.py` | Accelerate-side live Python/CLI/MCP agreement on discovery, failure codes, install boundary, goal-tactician refusal |

Related production suites (must not be weakened):

| Suite | What it proves |
| --- | --- |
| `test/api/test_root_mcp_formal_verification_parity.py` | `STABLE_OPERATIONS` ↔ datasets MCP `TOOL_TO_OPERATION`; shared envelope; `FormalVerificationMCPParity@1` |
| `test/api/test_goal_tactician_cli_mcp_parity.py` | Goal-tactician Python / MCP / CLI closed maps, envelope, forbidden controls, legacy `STABLE_OPERATIONS` preserved |
| `ipfs_datasets_py/tests/unit/logic/test_verification_api.py` | Python facade contracts for stable operations |

Direct-versus-supervisor semantic parity remains LPC-141
(`notes/direct_supervisor_parity.md`); this catalog is the shared vocabulary
those tests consume.

## File ownership

| Path | Role |
| --- | --- |
| `data/agent_supervisor/logic_platform_canonicalization/notes/operation_catalog.md` | This catalog projection note (declared output for LPC-130) |
| `ipfs_datasets_py/tests/unit/logic/test_channel_parity.py` | Unit parity gate (implied validation output) |
| `test/api/test_logic_channel_parity.py` | API parity gate (implied validation output) |
| `ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py` | Canonical operation names, envelopes, goal-tactician maps |
| `ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/logic_verification.py` | MCP tool names, schemas, `TOOL_TO_OPERATION` |
| `ipfs_datasets_py/ipfs_datasets_py/logic/cli.py` | CLI commands dispatching to MCP/Python |

Task-owned proposal envelope for LPC-130 (fail closed):

* **Declared Outputs:** this note.
* **Implied validation outputs:** the two parity suites above.
* Paths outside allowed edit paths (daemon code, protected plan/board
  validators, MCP++ profiles) are **out of scope**.
* LPC-141 owns direct-vs-supervisor parity notes/tests; do not absorb that
  scope into this catalog note.

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-062 | Outer `logic.api` / `logic.__init__` facades stay thin; verification ops live on `verification_api` |
| LPC-060 | Internal `LogicPlatformService@1` decomposition behind the facade |
| LPC-110 | Supervisor client handshakes then invokes typed ops; consumes catalog projection |
| LPC-100 | Manifest handshake precedes supervisor semantic work |
| LPC-032 | Success is not proof; authority never upgrades from transport |
| LPC-141 | Direct-versus-supervisor parity tests over this catalog |
| LPC-140 | Mandatory test matrix lists hermetic parity requirements |

## What this task does **not** do

* Does not add a new MCP++ profile or a second operation inventory.
* Does not expose supervisor-only mutation controls on datasets channels.
* Does not treat `install_provider` as ordinary verification.
* Does not claim live prover availability from catalog presence.
* Does not implement LPC-141 direct-vs-supervisor parity.
* Does not edit protected board/plan/validator files.

## Acceptance

- One operation catalog projects from `LogicVerificationAPI@1` /
  `GoalTacticianAPI@1` to Python, CLI, and MCP without a second hand inventory.
- Channel maps agree on operation names, request/response schemas, status,
  authority, failure/opt-in behavior.
- `install_provider` remains opt-in mutation, not ordinary verify.
- Supervisor-only mutation controls are not exposed from datasets logic.
- Transport success never implies proof authority; catalog presence never
  implies provider availability.
- Validation:  
  `python -m pytest ipfs_datasets_py/tests/unit/logic/test_channel_parity.py test/api/test_logic_channel_parity.py -q`
