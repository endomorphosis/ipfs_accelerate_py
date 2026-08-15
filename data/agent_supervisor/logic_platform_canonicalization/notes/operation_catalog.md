# LPC-130 LogicOperationCatalog@1 — Python, CLI, and MCP parity

**Task:** LPC-130  
**Goal:** LPC-G130  
**Interface:** `LogicOperationCatalog@1`  
**Validation:**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/test_channel_parity.py test/api/test_logic_channel_parity.py -q`

## Purpose

Derive **one operation catalog** from the canonical datasets logic service
(`LogicVerificationAPI@1` / related public facades). Every channel —

| Channel | Interface | Source |
| --- | --- | --- |
| Python | `LogicVerificationAPI@1` | `ipfs_datasets_py.logic.verification_api` |
| CLI | `LogicVerificationCLI@1` | `ipfs_datasets_py.logic.cli` |
| MCP | `LogicVerificationMCP@1` | `ipfs_datasets_py.mcp_server.tools.logic_verification` |

— must agree on:

1. **Operation names** (closed Python vocabulary)
2. **Request / response schemas** (shared envelope + per-tool parameter shapes)
3. **Status** vocabulary (`VerificationStatus`)
4. **Authority** ceilings (`VerificationAuthority`)
5. **Failure codes** (`unsupported_features` + structured diagnostics)
6. **Opt-in requirements** (`FeatureAvailability.OPT_IN`, `requires_opt_in`)

Additive surfaces (goal tactician, migration, provider-role closure) keep their
own closed maps and **must not** erase or re-map `STABLE_OPERATIONS`.

## Non-goals

* No new MCP++ profile.
* No supervisor mutation controls on datasets channels.
* Installation is **not** an ordinary verification / check / receipt-verify
  operation (see [Installation boundary](#installation-boundary)).
* Transport success never implies proof success.

## Shared response envelope

All three channels return the same envelope identity:

* Schema: `logic-verification-response/v1`
* Interface field: `LogicVerificationAPI@1`

| Field | Type | Agreement rule |
| --- | --- | --- |
| `status` | string enum | Same closed `VerificationStatus` values |
| `authority` | string enum | Same closed `VerificationAuthority` values; never silently upgraded |
| `operation` | string | Canonical Python operation name |
| `result` | object | Channel-neutral payload; install receipts stay installer-scoped |
| `assumptions` | list | Same assumption ids when present |
| `bounds` | object | Same bound keys when present |
| `translations` | list | Same translation descriptors |
| `witnesses` | list | Public witnesses only |
| `unsupported_features` | list | Shared failure-code channel |
| `diagnostics` | list | Secret-safe strings |
| `cache` | object | Provenance only; not authority |
| `interface` | string | Always `LogicVerificationAPI@1` |

MCP tool schemas advertise `returns.envelope = logic-verification-response/v1`
and `python_operation` equal to the catalog name.

## Closed status and authority vocabularies

### Status (`VerificationStatus`)

| Value | Meaning |
| --- | --- |
| `succeeded` | Operation completed under its authority ceiling |
| `partial` | Partial result; never silent success |
| `unsupported` | Closed vocabulary rejection or missing opt-in |
| `unavailable` | Declared but not usable (offline, missing tool, policy block) |
| `invalid` | Malformed request / type error |
| `error` | Unexpected failure; still non-proof |
| `declarative` | Discovery / plan-only response (no execution claim) |

### Authority (`VerificationAuthority`)

| Value | Typical use |
| --- | --- |
| `none` | Installer / probe health; no semantic claim |
| `advisory` | Advisor proposals |
| `bounded` | Check / portfolio / compile / counterexample projections |
| `satisfiability` / `model_check` / `monitor` / `authorization` / `protocol` / `hyperproperty` | Provider-class ceilings when earned |
| `candidate` / `reconstruction` | Candidate evidence only |
| `attestation` | Receipt attestation surface |
| `theorem` | Kernel theorem authority only when independently established |
| `declarative` | Catalog / capability listings |

Channels must return the **same** status and authority strings for the same
canonical request. A CLI or MCP adapter may add transport metadata
(`mcp_interface`, `cli_interface`, `channel`) but must not change status,
authority, operation, or failure codes.

## Failure codes

Failure codes travel primarily in `unsupported_features` (string tags) with
human diagnostics in `diagnostics`. Codes that every channel must preserve:

| Code | When |
| --- | --- |
| `install_without_opt_in` | `install_provider` without `allow_install=True` (non dry-run) |
| `mcp_provider_install_operator_policy` | MCP live install denied by host env gate |
| `offline_install` | Install refused under offline policy |
| `provider_installer:<id>` | Installer unavailable for provider |
| `provider:<id>` | Unknown / unsupported provider capability lookup |
| `compile_target:<name>` | Unknown compile target |
| `advisor:<name>` | Unknown advisor provider |
| `attestation_backend` | Attestation backend disabled / unavailable |
| `receipt` | Missing receipt input |
| `supervisor_only_control` | Goal-tactician refusal of supervisor mutation controls |

Unknown MCP tools / CLI commands use stable tags of the form
`mcp_tool:<name>` / `cli_command:<name>` on the goal-tactician surface.

## Opt-in requirements

From `list_stable_features()` / `FeatureAvailability`:

| Operation | Availability | Opt-in |
| --- | --- | --- |
| Discovery: `list_logic_families`, `list_providers`, `provider_capabilities` | `declared` | no |
| Execute: `compile_verification_artifact`, `check`, `monitor`, `run_portfolio`, `explain_counterexample`, `verify_receipt`, `advise` | `declared` | no |
| `probe_provider` | `opt_in` | yes — explicit call only |
| `install_provider` | `opt_in` | yes — `allow_install=True` for mutation |
| `attest_receipt` | `opt_in` | yes — explicit attestation backend selection |

Discovery never probes, installs, opens the network, or starts processes.
Import of `verification_api` remains side-effect free.

## Stable verification operations (`STABLE_OPERATIONS`)

Canonical Python names and channel projections:

| Python operation | CLI command | MCP tool | Authority ceiling | Opt-in |
| --- | --- | --- | --- | --- |
| `list_logic_families` | `list-families` | `verification_list_logic_families` | declarative | no |
| `list_providers` | `list-providers` | `verification_list_providers` | declarative | no |
| `provider_capabilities` | `provider-capabilities` | `verification_provider_capabilities` | declarative | no |
| `compile_verification_artifact` | `compile` | `verification_compile` | bounded | no |
| `check` | `check` | `verification_check` | bounded (provider may refine) | no |
| `monitor` | `monitor` | `verification_monitor` | monitor when earned | no |
| `run_portfolio` | `portfolio` | `verification_portfolio` | bounded / provider | no |
| `explain_counterexample` | `counterexample` | `verification_explain_counterexample` | bounded | no |
| `verify_receipt` | `verify-receipt` | `verification_verify_receipt` | bounded structure check | no |
| `attest_receipt` | `attest-receipt` | `verification_attest_receipt` | attestation | yes |
| `advise` | `advise` | `verification_advise` | advisory | no |
| `probe_provider` | `probe-provider` | `verification_probe_provider` | none (health only) | yes |
| `install_provider` | `install-provider` | `verification_install_provider` | none (installer only) | yes |

### Discovery helpers (not members of `STABLE_OPERATIONS`, still channel-shared)

| Python operation | CLI command | MCP tool |
| --- | --- | --- |
| `list_features` | `list-features` | `verification_list_features` |
| *(meta surface)* | `verification-capabilities` | `verification_capabilities` |

`list_features` must return a **superset** of `STABLE_OPERATIONS`.  
`verification_capabilities` reports tool names, operation maps, and bounds
without probing providers.

### MCP mapping invariant

```text
∀ op ∈ STABLE_OPERATIONS:  op ∈ values(TOOL_TO_OPERATION)
```

`TOOL_TO_OPERATION` is defined in `logic_verification.py`. Parent accelerate
MCP re-exports (when present) must preserve the same tool→operation pairs for
the datasets formal-verification tools.

### CLI mapping invariant

Every CLI command in the stable verification group dispatches to the same
Python operation name returned in the envelope `operation` field. CLI
`install-provider` requires `--allow-install` for mutation; `--dry-run` and
`--offline` remain non-mutating.

## Installation boundary

**Installation is not an ordinary verify operation.**

| Concern | `check` / `verify_receipt` / `run_portfolio` | `install_provider` |
| --- | --- | --- |
| Purpose | Semantic / structural verification | Toolchain mutation / planning |
| Authority | Bounded or earned provider authority | Always `none` on the install receipt |
| Default call | May run under ordinary request | Requires explicit `allow_install=True` |
| Dry-run | N/A as installer plan | Returns `status=declarative`, `install_attempted=false` |
| Denied opt-in | N/A | `status=unsupported`, code `install_without_opt_in` |
| MCP host gate | N/A | Additional `IPFS_DATASETS_PY_MCP_ALLOW_PROVIDER_INSTALLS` for live mutation |
| Success meaning | Result under authority ceiling | Installer receipt only — **never** proof success |

Rules:

1. A successful install does **not** mint satisfiability, theorem, or attestation
   authority for subsequent checks.
2. `probe_provider` reports availability only; it never installs.
3. `verify_receipt` validates receipt structure / bindings; it never installs.
4. Channels must not alias `install_provider` onto `check` or `verify_receipt`.

## Goal tactician channel maps (additive)

Interface: `GoalTacticianCLIMCP@1` over `GoalTacticianAPI@1`.

| Python | CLI | MCP tool |
| --- | --- | --- |
| `formalize_goal` | `goal-formalize` | `goal_tactician_formalize_goal` |
| `compare_interpretations` | `goal-compare-interpretations` | `goal_tactician_compare_interpretations` |
| `discover_missing_proofs` | `goal-discover-missing-proofs` | `goal_tactician_discover_missing_proofs` |
| `plan_proof` | `goal-plan-proof` | `goal_tactician_plan_proof` |
| `validate_proof_candidate` | `goal-validate-candidate` | `goal_tactician_validate_proof_candidate` |
| `execute_proof_plan` | `goal-execute-plan` | `goal_tactician_execute_proof_plan` |
| `proof_status` | `goal-proof-status` | `goal_tactician_proof_status` |
| `minimize_counterexample` | `goal-minimize-counterexample` | `goal_tactician_minimize_counterexample` |
| `explain_counterexample_causal` | `goal-explain-counterexample` | `goal_tactician_explain_counterexample_causal` |
| `replay_counterexample` | `goal-replay-counterexample` | `goal_tactician_replay_counterexample` |
| `list_goal_tactician_operations` | `goal-list-operations` | `goal_tactician_list_operations` |

These maps are closed 1:1. Goal tactician wiring is additive: it preserves
`STABLE_OPERATIONS` under `legacy_operations_preserved` and must not remove
LogicVerificationMCP@1 coverage.

### Supervisor-only controls (forbidden on all public channels)

Datasets logic channels refuse:

`admit_goal`, `close_plan`, `mutate_supervisor`, `force_complete`,
`lease_steal`, `rewrite_event_log`, `bypass_resource_policy`,
`promote_proof_authority`, `supervisor_mutate`, `supervisor_only`.

Refusal is channel-neutral: `status=invalid` with failure code
`supervisor_only_control`.

## Schema identity summary

| Identity | Value |
| --- | --- |
| Catalog interface | `LogicOperationCatalog@1` |
| Python API | `LogicVerificationAPI@1` |
| CLI | `LogicVerificationCLI@1` |
| MCP | `LogicVerificationMCP@1` |
| Response schema | `logic-verification-response/v1` |
| Request schema | `logic-verification-request/v1` |
| Feature schema | `logic-verification-feature/v1` |
| Installer | `LogicVerificationLazyInstaller@1` |
| Goal tactician API | `GoalTacticianAPI@1` |
| Goal tactician CLI/MCP | `GoalTacticianCLIMCP@1` |
| MCP parity evidence | `FormalVerificationMCPParity@1` |

## Derivation rule

The catalog is a **projection** of the canonical service, not a hand-maintained
second list:

1. Python `STABLE_OPERATIONS` is the closed name authority for ordinary
   verification.
2. MCP `TOOL_TO_OPERATION` and CLI command dispatch must cover that set.
3. `list_stable_features()` is the authority for opt-in / availability /
   authority ceilings advertised to callers.
4. Status and authority enums live in `verification_api` and are shared
   verbatim in JSON envelopes.
5. Failure codes are the `unsupported_features` strings returned by the
   facade; adapters may only add transport-policy codes (for example MCP host
   install policy) without renaming facade codes.

## Acceptance (LPC-130 / LPC-G130)

* Channels agree on names, schemas, status, authority, failure codes, and opt-in.
* Installation is not an ordinary verify operation.
* Supervisor-only mutation controls are not exposed from datasets logic.
* No new MCP++ profile is introduced.

## Related evidence

* Unit: `ipfs_datasets_py/tests/unit/logic/test_channel_parity.py`
* API: `test/api/test_logic_channel_parity.py`
* Prior interim surfaces: `test/api/test_goal_tactician_cli_mcp_parity.py`,
  `test/api/test_root_mcp_formal_verification_parity.py`,
  `ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py`
