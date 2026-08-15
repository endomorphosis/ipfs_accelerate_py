# LPC-130 LogicOperationCatalog@1 — Python, CLI, and MCP parity

**Task:** LPC-130  
**Goal:** LPC-G130  
**Interface:** `LogicOperationCatalog@1`  
**Owned output (this task):**  
`data/agent_supervisor/logic_platform_canonicalization/notes/operation_catalog.md`  
**Board validation command (predicted companions):**  
`python -m pytest ipfs_datasets_py/tests/unit/logic/test_channel_parity.py test/api/test_logic_channel_parity.py -q`

## Purpose

Derive **one operation catalog** from the canonical datasets logic service
(`LogicVerificationAPI@1` / related public facades). Every channel —

| Channel | Interface | Source module |
| --- | --- | --- |
| Python | `LogicVerificationAPI@1` | `ipfs_datasets_py.logic.verification_api` |
| CLI | `LogicVerificationCLI@1` | `ipfs_datasets_py.logic.cli` |
| MCP | `LogicVerificationMCP@1` | `ipfs_datasets_py.mcp_server.tools.logic_verification` |

— must agree on:

1. **Operation names** (closed Python vocabulary `STABLE_OPERATIONS`)
2. **Request / response schemas** (shared envelope + per-tool parameter shapes)
3. **Status** vocabulary (`VerificationStatus`)
4. **Authority** ceilings (`VerificationAuthority`)
5. **Failure codes** (`unsupported_features` + structured diagnostics)
6. **Opt-in requirements** (`FeatureAvailability.OPT_IN`, `requires_opt_in`)

This note is the **catalog projection**: a readable, closed map of what the
canonical service already exposes. It is not a second hand-maintained authority
and must not invent new operation names, tools, or MCP++ profiles.

Additive surfaces (goal tactician, migration, provider-role closure, production
authorization) keep their own closed maps and **must not** erase or re-map
`STABLE_OPERATIONS`.

## Non-goals

* No new MCP++ profile.
* No supervisor mutation controls on datasets channels.
* Installation is **not** an ordinary verification / check / receipt-verify
  operation (see [Installation boundary](#installation-boundary)).
* Transport success never implies proof success.
* No new prover, family, registry, or second semantic authority.
* This LPC-130 deliverable owns the catalog note only. Predicted automated
  gate modules remain board companions; they must not widen scope beyond the
  declared output when adjudication denies those paths.

## Source-of-truth derivation

| Catalog claim | Authoritative source |
| --- | --- |
| Closed operation names | `STABLE_OPERATIONS` in `verification_api.py` |
| Feature availability / opt-in / authority ceilings | `list_stable_features()` |
| Status / authority enums | `VerificationStatus`, `VerificationAuthority` |
| Response envelope schema | `LOGIC_VERIFICATION_RESPONSE_SCHEMA` = `logic-verification-response/v1` |
| Request schema id | `LOGIC_VERIFICATION_REQUEST_SCHEMA` = `logic-verification-request/v1` |
| Feature schema id | `LOGIC_VERIFICATION_FEATURE_SCHEMA` = `logic-verification-feature/v1` |
| MCP tool → Python operation | `TOOL_TO_OPERATION` in `logic_verification.py` |
| MCP parameter shapes | `TOOL_SCHEMAS` in `logic_verification.py` |
| CLI command → Python operation | `LogicVerificationCLI@1` dispatch in `logic/cli.py` (`_run_verification_command`) |
| Goal-tactician closed maps | `GOAL_TACTICIAN_*` constants in `verification_api.py` |
| Supervisor-only refusals | `_GOAL_TACTICIAN_FORBIDDEN_CONTROLS` |

## Shared response envelope

All three channels return the same facade envelope identity for stable
verification operations:

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
and `python_operation` equal to the catalog name. CLI verification commands
dispatch through the MCP tool layer (or the same Python facade for install) and
return the same envelope fields.

Transport adapters may add metadata (`tool`, `mcp_interface`, `cli_interface`,
`channel`, `success`, `python_operation`) without renaming facade `status`,
`authority`, `operation`, or failure codes.

### MCP transport bounds (channel-local, non-semantic)

Declared on `verification_capabilities` and applied by the MCP wrapper:

| Bound | Value |
| --- | --- |
| `max_json_bytes` | 256_000 |
| `max_string_chars` | 64_000 |
| `max_diagnostic_chars` | 2_000 |
| `max_result_depth` | 12 |
| `max_collection_items` | 500 |

Bounds constrain transport size only. They do not change status, authority, or
operation identity.

## Closed status, availability, and authority vocabularies

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

### Feature availability (`FeatureAvailability`)

| Value | Meaning |
| --- | --- |
| `declared` | Present in the closed catalog; may still need a live provider |
| `available` | Probe/runtime reported usable |
| `unavailable` | Declared but not usable now |
| `unsupported` | Explicitly not supported |
| `absent` | Not present |
| `opt_in` | Requires explicit caller consent (`requires_opt_in=true`) |

### Authority (`VerificationAuthority`)

| Value | Typical use |
| --- | --- |
| `none` | Installer / probe health; no semantic claim |
| `advisory` | Advisor proposals |
| `bounded` | Check / portfolio / compile / counterexample projections |
| `satisfiability` | Provider-class ceiling when earned |
| `model_check` | Provider-class ceiling when earned |
| `monitor` | Runtime monitor ceiling when earned |
| `authorization` | Authorization-class ceiling when earned |
| `protocol` | Protocol-class ceiling when earned |
| `hyperproperty` | Hyperproperty-class ceiling when earned |
| `candidate` | Candidate evidence only |
| `reconstruction` | Reconstruction evidence only |
| `attestation` | Receipt attestation surface |
| `theorem` | Kernel theorem authority only when independently established |
| `declarative` | Catalog / capability listings |

Channels must return the **same** status and authority strings for the same
canonical request. A CLI or MCP adapter may add transport metadata but must not
change status, authority, operation, or failure codes.

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

From `list_stable_features()` / `FeatureAvailability` for the stable surface:

| Operation group | Operations | Availability | Opt-in |
| --- | --- | --- | --- |
| Discovery | `list_logic_families`, `list_providers`, `provider_capabilities` | `declared` | no |
| Execute / structure | `compile_verification_artifact`, `check`, `monitor`, `run_portfolio`, `explain_counterexample`, `verify_receipt`, `advise` | `declared` | no |
| Probe | `probe_provider` | `opt_in` | yes — explicit call only; authority ceiling `none` |
| Install | `install_provider` | `opt_in` | yes — `allow_install=True` for mutation; authority ceiling `none` |
| Attest | `attest_receipt` | `opt_in` | yes — explicit attestation backend selection; ceiling `attestation` |

Authority ceilings advertised by `list_stable_features()` for declared execute
ops:

| Operation | Authority ceiling |
| --- | --- |
| `list_logic_families`, `list_providers`, `provider_capabilities` | `declarative` |
| `compile_verification_artifact`, `check`, `monitor`, `run_portfolio`, `explain_counterexample`, `verify_receipt` | `bounded` |
| `advise` | `advisory` |
| `probe_provider`, `install_provider` | `none` |
| `attest_receipt` | `attestation` |

Discovery never probes, installs, opens the network, or starts processes.
Import of `verification_api` remains side-effect free.

## Stable verification operations (`STABLE_OPERATIONS`)

Canonical Python names (closed tuple order from `verification_api.py`):

1. `list_logic_families`
2. `list_providers`
3. `provider_capabilities`
4. `compile_verification_artifact`
5. `check`
6. `monitor`
7. `run_portfolio`
8. `explain_counterexample`
9. `verify_receipt`
10. `attest_receipt`
11. `advise`
12. `probe_provider`
13. `install_provider`

### Channel projection table

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

| Python operation | CLI command | MCP tool | Notes |
| --- | --- | --- | --- |
| `list_features` | `list-features` | `verification_list_features` | Superset of `STABLE_OPERATIONS` (plus additive feature rows) |
| `list_features` (meta surface) | `verification-capabilities` | `verification_capabilities` | Reports tools, operation map, bounds; no provider probes |

`list_features` must return a **superset** of `STABLE_OPERATIONS`.  
`verification_capabilities` is a discovery meta-tool: it maps to Python
operation name `list_features` in `TOOL_TO_OPERATION` but returns a capability
summary (`tools`, `operations`, `tool_to_operation`, `bounds`) rather than the
ordinary feature-list envelope payload.

### Closed CLI projection (`LogicVerificationCLI@1`)

```text
list-features            → list_features
list-families            → list_logic_families
list-providers           → list_providers
provider-capabilities    → provider_capabilities
compile                  → compile_verification_artifact
check                    → check
monitor                  → monitor
portfolio                → run_portfolio
counterexample           → explain_counterexample
verify-receipt           → verify_receipt
advise                   → advise
attest-receipt           → attest_receipt
probe-provider           → probe_provider
install-provider         → install_provider
verification-capabilities→ list_features (meta surface)
```

Every CLI command in the stable verification group dispatches to the same
Python operation name returned in the envelope `operation` field (install is
dispatched directly to the Python facade; other commands go through the MCP
tool wrappers that call the same facade). CLI `install-provider` requires
`--allow-install` for mutation; `--dry-run` and `--offline` remain non-mutating.

### Closed MCP projection (`LogicVerificationMCP@1`)

```text
∀ op ∈ STABLE_OPERATIONS:  op ∈ values(TOOL_TO_OPERATION)
```

Full `TOOL_TO_OPERATION` map (datasets MCP):

```text
verification_list_features            → list_features
verification_list_logic_families      → list_logic_families
verification_list_providers           → list_providers
verification_provider_capabilities    → provider_capabilities
verification_compile                  → compile_verification_artifact
verification_check                    → check
verification_monitor                  → monitor
verification_portfolio                → run_portfolio
verification_explain_counterexample   → explain_counterexample
verification_verify_receipt           → verify_receipt
verification_advise                   → advise
verification_attest_receipt           → attest_receipt
verification_probe_provider           → probe_provider
verification_install_provider         → install_provider
verification_capabilities             → list_features
```

`TOOL_TO_OPERATION` is defined in
`ipfs_datasets_py/mcp_server/tools/logic_verification.py`. Parent accelerate
MCP re-exports (when present) must preserve the same tool→operation pairs for
the datasets formal-verification tools.

Each `TOOL_SCHEMAS[tool]["python_operation"]` equals `TOOL_TO_OPERATION[tool]`,
and `returns.envelope` is always `logic-verification-response/v1` for stable
verification tools (capability meta-tool may advertise a summary shape).

## Request / parameter agreement

Channels accept the same logical parameters for each operation. Names below are
Python / MCP keyword names; CLI uses kebab-case flags of the same meaning.

| Python operation | Required params | Optional params |
| --- | --- | --- |
| `list_features` | — | — |
| `list_logic_families` | — | — |
| `list_providers` | — | — |
| `provider_capabilities` | — | `provider_id` |
| `compile_verification_artifact` | `artifact` | `target` (default `smtlib2`), `request_id` |
| `check` | `request` | `backend_id`, `request_id` |
| `monitor` | `formula`, `observations` | `request_id` |
| `run_portfolio` | `obligation` | `capabilities`, `resource_policy`, `request_id` |
| `explain_counterexample` | `witness` | `request_id` |
| `verify_receipt` | `receipt` | `expectation`, `request_id` |
| `advise` | `request` | `provider` (default `static`), `request_id` |
| `attest_receipt` | `receipt` | `backend_mode` (default `disabled`), `backend_policy`, `witness`, `issued_at`, `expires_at`, `request_id` |
| `probe_provider` | `provider_id` | `request_id` |
| `install_provider` | `provider_id` | `allow_install` (default false), `dry_run`, `offline`, `force`, `request_id` |

CLI flag mapping notes:

* JSON-bearing values (`--artifact`, `--request`, `--formula`, `--observations`,
  `--obligation`, `--witness`, `--receipt`, …) are JSON object mappings.
* `install-provider` uses positional `provider_id` plus `--allow-install`,
  `--dry-run`, `--offline`, `--force`.
* `probe-provider` uses positional `provider_id`.

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
5. `install_provider` remains in `STABLE_OPERATIONS` as an explicit opt-in
   mutation boundary, but is classified separately from ordinary verify ops in
   feature availability, authority ceiling, and failure codes.
6. MCP live install without the host env gate returns
   `mcp_provider_install_operator_policy` while preserving
   `status=unsupported` and authority `none`.

## Additive closed maps (must not rewrite `STABLE_OPERATIONS`)

### Goal tactician (`GoalTacticianCLIMCP@1` over `GoalTacticianAPI@1`)

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

### Migration operations (`MIGRATION_OPERATIONS`, VerificationAPI@2 discovery)

Not merged into `STABLE_OPERATIONS`:

`list_namespaces`, `list_namespace_identities`, `dual_read_label`,
`canonical_write_label`, `migrate_artifact`, `inspect_translation_loss`,
`inspect_provider_authority`.

### Provider role closure (`PROVIDER_ROLE_CLOSURE_OPERATIONS`)

Not merged into `STABLE_OPERATIONS`:

`list_provider_roles`, `provider_role`, `secpal_artifact_intake`,
`secpal_compatibility_lookup`.

`secpal_artifact_intake` and `secpal_compatibility_lookup` are opt-in.

### Production authorization (`PRODUCTION_AUTHORIZATION_OPERATIONS`)

Not merged into `STABLE_OPERATIONS`:

`production_authorization_identity`, `production_authorization_check`,
`production_authorization_receipt`.

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
6. Additive maps stay additive: goal tactician, migration, role-closure, and
   production-authorization operations never rewrite `STABLE_OPERATIONS`.

## Channel agreement invariants (testable)

For any closed request that all three channels accept:

1. `operation` is the same Python name from `STABLE_OPERATIONS` (or the shared
   discovery helper `list_features`).
2. `status` ∈ `VerificationStatus` and is identical across Python / CLI / MCP.
3. `authority` ∈ `VerificationAuthority` and is identical (never silently
   upgraded by a transport adapter).
4. `unsupported_features` sets are identical for facade-owned failure codes.
   MCP may add host-policy codes only for live install mutation
   (`mcp_provider_install_operator_policy`).
5. Opt-in flags from `list_stable_features()` match tool schemas and CLI flags
   (`allow_install` default false; probe / attest remain explicit opt-in).
6. Installation is not an ordinary verify operation: `install_provider` keeps
   authority `none`, separate failure codes, and is never aliased to `check` or
   `verify_receipt`.
7. Supervisor-only mutation controls are never exposed as callable operations
   from datasets Python, CLI, or MCP surfaces.
8. `∀ op ∈ STABLE_OPERATIONS: op ∈ values(TOOL_TO_OPERATION)` and every stable
   CLI verification command dispatches to that same Python name.

Sealed validation environments may lack external provers on `PATH`. Probe and
portfolio responses remain channel-neutral under that constraint: missing tools
surface as `unavailable` / `partial` with authority ceilings preserved, never
as silent proof success.

Authoritative validation environment (fail-closed):

* `PATH` is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`
* Canonical interpreter target is `/usr/bin/python3.12`
* Fresh private `HOME` with standard XDG paths only
* External-tool availability assertions must match this environment; absence of
  a prover is a capability gap or `unavailable`, never silent success
* Operator profile toolchains (`~/.elan`, provider-only PATH entries) are
  unavailable unless injected through the sealed validation PATH replacement

## Parity evidence and gate paths

### Existing interim channel evidence (already in tree)

| Layer | Path | Role |
| --- | --- | --- |
| Integration | `ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py` | CLI + MCP + Python envelope, install opt-in, MCP host install policy |
| API | `test/api/test_root_mcp_formal_verification_parity.py` | Datasets/root MCP share tools and cover `STABLE_OPERATIONS` |
| API | `test/api/test_goal_tactician_cli_mcp_parity.py` | Goal-tactician closed maps; preserves `STABLE_OPERATIONS` |
| Unit | `ipfs_datasets_py/tests/unit/logic/test_verification_api.py` | Facade status/authority/install opt-in behavior |

### Board-predicted dedicated gate modules (companions)

| Layer | Path | Role |
| --- | --- | --- |
| Unit | `ipfs_datasets_py/tests/unit/logic/test_channel_parity.py` | Closed map, feature opt-in, install boundary, supervisor refusal, enum identity |
| API | `test/api/test_logic_channel_parity.py` | Live Python / CLI / MCP envelope agreement on shared requests |

Those predicted modules are the board validation command targets. They are
**not** part of this note's owned-output authority. LPC-130 completes when this
catalog projection is accurate against the live sources above. Automated gate
modules, when admitted by a later proposal, must assert the catalog projection
rather than transport-specific success and must not weaken install / supervisor
boundaries.

## Acceptance (LPC-130 / LPC-G130)

* Channels agree on names, schemas, status, authority, failure codes, and opt-in.
* Installation is not an ordinary verify operation.
* Supervisor-only mutation controls are not exposed from datasets logic.
* No new MCP++ profile is introduced.
* Catalog is a projection of `STABLE_OPERATIONS` + channel maps, not a second list.

## Related evidence

* Owned output: `data/agent_supervisor/logic_platform_canonicalization/notes/operation_catalog.md`
* Interim: `ipfs_datasets_py/tests/integration/test_logic_verification_cli_mcp.py`
* Interim: `test/api/test_root_mcp_formal_verification_parity.py`
* Interim: `test/api/test_goal_tactician_cli_mcp_parity.py`
* Sources: `ipfs_datasets_py/logic/verification_api.py`,
  `ipfs_datasets_py/logic/cli.py`,
  `ipfs_datasets_py/mcp_server/tools/logic_verification.py`
