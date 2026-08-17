# LPC-060 Decompose verification_api into internal platform services

**Task:** LPC-060 — Decompose `verification_api` into internal platform services  
**Goal:** LPC-G060  
**Depends on:** LPC-051 (v1 payloads cannot bypass `BackendRequest@2`)  
**Interface:** `LogicPlatformService@1`  
**Public facade (retained):** `ipfs_datasets_py.logic.verification_api`  
**Internal package:** `ipfs_datasets_py.logic.platform`  
**Evidence module (current):** `ipfs_datasets_py/logic/verification_api.py`  
**Predicted composition root:** `ipfs_datasets_py/logic/platform/service.py`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/test_verification_api.py -q`

## Purpose

`LogicVerificationAPI@1` (`verification_api.py`) is the stable public Python
software-verification facade. It currently owns response contracts, discovery,
migration dual-read, formalization/goal-tactician entry points, portfolio and
check execution, receipt verification/attestation, counterexample projection,
and the sole public install/probe mutation boundary in one large module.

LPC-060 freezes the **internal service layout** that subsequent implementation
moves into `logic/platform/*` while preserving every public import path. The
facade stays; the second independent implementation does not appear. No new
top-level namespace is introduced.

This note is the authoritative decomposition contract for LPC-G060. LPC-061
owns pure-data import hermeticity tests against the resulting import graph.
LPC-062 owns further thinning of `logic.api` and `logic.__init__` as
compatibility facades over the same service composition.

## Decision (authoritative)

Internal layout under `ipfs_datasets_py/ipfs_datasets_py/logic/platform/` is
exactly this closed set of service modules (plus the already-shipped
`manifest.py` handshake surface from LPC-100):

| Module | Path | Pure-data import? | Role |
| --- | --- | --- | --- |
| **contracts** | `logic/platform/contracts.py` | **yes** | Response/request envelopes, status/authority enums, feature descriptors, cache provenance, schema constants |
| **catalog** | `logic/platform/catalog.py` | **yes** | Declarative family/provider/feature catalog reads over sealed snapshots and registries |
| **discovery** | `logic/platform/discovery.py` | **yes** | `CanonicalLogicDiscovery@1`, namespace dual-read / canonical-write, migration label inspection |
| **formalization** | `logic/platform/formalization.py` | **yes** (lazy runtime) | Goal formalization, interpretation compare, advisor proposals (`advise`, `formalize_goal`, …) |
| **obligations** | `logic/platform/obligations.py` | **yes** (lazy runtime) | Obligation/IR admission for compile and check inputs; SMT obligation shaping |
| **translations** | `logic/platform/translations.py` | **yes** | Translation-loss inspection, translation receipt identity, preservation reporting |
| **providers** | `logic/platform/providers.py` | **yes** (lazy runtime) | Provider role closure, capability declarations, production-authorization identity, SecPAL archival surfaces |
| **planning** | `logic/platform/planning.py` | **yes** (lazy runtime) | Goal-tactician planning: missing proofs, plan ranking, proof status, candidate validation |
| **execution** | `logic/platform/execution.py` | **no** (explicit only) | Live `check`, `monitor`, `run_portfolio`, `execute_proof_plan` under typed bounds |
| **evidence** | `logic/platform/evidence.py` | **yes** | Authority mapping, evidence kind/ceiling projection into public responses (never upgrades) |
| **receipts** | `logic/platform/receipts.py` | **yes** (lazy runtime) | Closed `verify_receipt` / `attest_receipt` dispatch (`VerifiedReceiptDispatch@2`) |
| **counterexamples** | `logic/platform/counterexamples.py` | **yes** (lazy runtime) | Public counterexample boundary, minimize/explain/replay projections |
| **installation** | `logic/platform/installation.py` | **no** (explicit only) | Sole mutation boundary: `probe_provider`, `install_provider` |
| **compatibility** | `logic/platform/compatibility.py` | **yes** | Legacy alias maps, migration ops, production-auth aliases, SecPAL compatibility lookup |
| **service** | `logic/platform/service.py` | **yes** (composition only) | `LogicPlatformService@1` composition root; wires services; no second semantics |
| **manifest** | `logic/platform/manifest.py` | **yes** | `LogicPlatformManifest@1` handshake (LPC-100; already present) |

Acceptance layout phrase (must match exactly as the internal service set):

```
contracts / catalog / discovery / formalization / obligations / translations /
providers / planning / execution / evidence / receipts / counterexamples /
installation / compatibility / service
```

`manifest.py` is adjacent package infrastructure, not a substitute for any of
the fifteen services above.

## Interface: LogicPlatformService@1

| Field | Value |
| --- | --- |
| Interface id | `LogicPlatformService@1` |
| Schema | `logic-platform-service/v1` |
| Composition module | `ipfs_datasets_py.logic.platform.service` |
| Public adapter | `ipfs_datasets_py.logic.verification_api.LogicVerificationAPI` |
| Factory | `get_platform_service(...)` (internal) / `get_verification_api(...)` (public) |

`LogicPlatformService@1` is the **single internal composition object**. Public
callers continue to use `LogicVerificationAPI@1` / module-level wrappers. The
service does not invent a parallel public API surface and does not re-implement
backends, catalogs, or receipt verifiers—it composes existing canonical
modules under typed ownership.

### Composition invariants

1. **One semantic authority.** Datasets `logic/*` ownership stays as today:
   families, formalization artifacts, provider protocol, evidence axes, and
   receipts. The platform package is an internal packaging of the facade, not a
   rival domain model.
2. **Facade stays thin.** After extraction, `verification_api.py` re-exports
   contracts for stable imports and delegates methods to
   `LogicPlatformService@1` (or service-local helpers). No second check/prove
   path may live only in the facade.
3. **Pure-data vs explicit runtime.** Importing contracts, catalog, discovery,
   translations, evidence, compatibility, service, and manifest must not probe
   the environment, install packages, open the network, start processes, mutate
   files, probe hardware, or change environment variables (LPC-061 gate).
4. **Runtime is opt-in.** Execution and installation services may load solvers,
   installers, and process runners only when their public operations are
   invoked with explicit caller intent.
5. **Success is not proof.** Service boundaries preserve LPC-032: operation
   status, semantic verdict, evidence kind, and evidence authority remain
   orthogonal. Provider success never upgrades authority.
6. **v1 cannot bypass @2.** Execution admission routes executable work through
   `BackendRequest@2` / `LogicProviderProtocol@2` and the LPC-051 v1 adapter
   dispositions (`parsed` / `rejected` / `advisory`). Free-form payload keys
   never mint a backend request.

## Current → target ownership map

### contracts

**Source today:** top of `verification_api.py` — enums, dataclasses, response
builders, schema constants.

| Symbol / constant | Destination |
| --- | --- |
| `LOGIC_VERIFICATION_API_INTERFACE`, `LOGIC_VERIFICATION_*_SCHEMA` | `contracts` |
| `VERIFICATION_API_V2_INTERFACE`, discovery/tactician/receipt interface ids | `contracts` |
| `STABLE_OPERATIONS`, `MIGRATION_OPERATIONS`, `GOAL_TACTICIAN_OPERATIONS`, `PROVIDER_ROLE_CLOSURE_OPERATIONS`, `PRODUCTION_AUTHORIZATION_OPERATIONS` | `contracts` |
| `VerificationStatus`, `FeatureAvailability`, `VerificationAuthority` | `contracts` |
| `VerificationAPIError`, `CacheProvenance`, `FeatureDescriptor`, `ProviderDescriptor`, `VerificationResponse` | `contracts` |
| `_response`, `_empty_cache`, `_text`, `_mapping`, `_string_tuple` | `contracts` (private helpers) |

`verification_api` re-exports these symbols so existing
`from ipfs_datasets_py.logic.verification_api import VerificationResponse`
imports remain green.

### catalog

**Source today:** `list_logic_families`, `list_providers`, `list_features`,
`list_stable_features`, declarative portions of `provider_capabilities`.

| Public operation | Destination service method |
| --- | --- |
| `list_logic_families` | `PlatformCatalog.list_logic_families` |
| `list_providers` | `PlatformCatalog.list_providers` |
| `list_features` / `list_stable_features` | `PlatformCatalog.list_features` |
| `provider_capabilities` (declarative only) | `PlatformCatalog.provider_capabilities` |

Delegates to sealed sources only:

* `logic.families` registry / `CanonicalLogicCatalogSnapshot@1`
* declared backend catalog (`backends.registry` declarative views)
* production-authorization descriptor (declarative identity; no live probe)

Discovery payloads always force `FeatureAvailability.DECLARED` until
`installation.probe_provider` is called.

### discovery

**Source today:** `CanonicalLogicDiscovery`, namespace dual-read/write,
migration artifact helpers.

| Public operation | Destination |
| --- | --- |
| `list_namespaces` | `PlatformDiscovery.list_namespaces` |
| `list_namespace_identities` | `PlatformDiscovery.list_identities` |
| `dual_read_label` | `PlatformDiscovery.dual_read_label` |
| `canonical_write_label` | `PlatformDiscovery.canonical_write_label` |
| `migrate_artifact` | `PlatformDiscovery.migrate_artifact` |
| `inspect_provider_authority` | `PlatformDiscovery.inspect_provider_authority` |

`CanonicalLogicDiscovery@1` moves intact into `discovery.py`. Writers emit only
canonical labels; dual-read remains the sole legacy alias resolution path.

### formalization

**Source today:** `advise`, `formalize_goal`, `compare_interpretations`.

| Public operation | Destination |
| --- | --- |
| `advise` | `PlatformFormalization.advise` |
| `formalize_goal` | `PlatformFormalization.formalize_goal` |
| `compare_interpretations` | `PlatformFormalization.compare_interpretations` |

Lazy-imports existing `logic.formalization` / advisor modules. Advisors propose
only; they never mark themselves proved or raise evidence authority
(LPC-G070 / plan §F).

### obligations

**Source today:** obligation shaping inside `compile_verification_artifact` and
check request construction.

| Public operation | Destination |
| --- | --- |
| `compile_verification_artifact` | `PlatformObligations.compile_verification_artifact` |
| check/portfolio input normalization (shared) | `PlatformObligations.admit_obligation` / `shape_backend_request` |

Owns mapping facade payloads → typed obligations / `BackendRequest@2` seeds
without executing providers. Executable bounds stay positive and finite.

### translations

**Source today:** `inspect_translation_loss`, translation fields on compile and
receipt paths.

| Public operation | Destination |
| --- | --- |
| `inspect_translation_loss` | `PlatformTranslations.inspect_translation_loss` |
| translation receipt schema identity helpers | `PlatformTranslations` constants / projectors |

Preserves translation-preservation as an independent axis (LPC-030/032). Does
not invent preservation claims from successful compile or check status.

### providers

**Source today:** role closure, provider role listing, production-authorization
identity, SecPAL archival intake (non-executable), refuse-non-verification
dispatch.

| Public operation | Destination |
| --- | --- |
| `list_provider_roles` | `PlatformProviders.list_provider_roles` |
| `provider_role` | `PlatformProviders.provider_role` |
| `production_authorization_identity` | `PlatformProviders.production_authorization_identity` |
| `secpal_artifact_intake` | `PlatformProviders.secpal_artifact_intake` |
| `_refuse_non_verification_dispatch` | `PlatformProviders.refuse_non_verification_dispatch` |

Classifies public roles (`runnable`, `support`, `advisor`, `archival_intake`)
without claiming live availability. Support/advisor/archival tools cannot be
used as live verification providers.

### planning

**Source today:** goal-tactician planning and status ops.

| Public operation | Destination |
| --- | --- |
| `discover_missing_proofs` | `PlatformPlanning.discover_missing_proofs` |
| `plan_proof` | `PlatformPlanning.plan_proof` |
| `validate_proof_candidate` | `PlatformPlanning.validate_proof_candidate` |
| `proof_status` | `PlatformPlanning.proof_status` |
| `list_goal_tactician_operations` | `PlatformPlanning.list_operations` |

Plans remain candidate-producing. Supervisor may reorder lanes later; this
service does not rewrite semantic meaning of obligations or receipts.

### execution

**Source today:** `check`, `monitor`, `run_portfolio`, `execute_proof_plan`,
portfolio attempt loop.

| Public operation | Destination |
| --- | --- |
| `check` | `PlatformExecution.check` |
| `monitor` | `PlatformExecution.monitor` |
| `run_portfolio` | `PlatformExecution.run_portfolio` |
| `execute_proof_plan` | `PlatformExecution.execute_proof_plan` |

Explicit runtime only. Admission path:

1. Normalize via `obligations` → typed request / bounds.
2. Elevate any v1 leaf through LPC-051 adapter (no free-form bypass).
3. Dispatch through backend registry / provider protocol.
4. Project outcome through `evidence` (authority ceiling) before response.

Cancellation, deadlines, and resource bounds remain hard ceilings.

### evidence

**Source today:** `_evidence_to_verification_authority`,
`_result_to_verification_authority`, authority fields on every response.

| Responsibility | Destination |
| --- | --- |
| Map provider/evidence axes → `VerificationAuthority` | `PlatformEvidence` |
| Enforce non-upgrade of authority ceilings | `PlatformEvidence.project_authority` |
| Attach assumptions / bounds / unsupported features consistently | `PlatformEvidence.project_response_axes` |

Does not store proofs. Does not promote `succeeded` into theorem/kernel
authority. Consumed by execution, receipts, counterexamples, and formalization
response builders.

### receipts

**Source today:** `verify_receipt`, `attest_receipt`, closed schema dispatch.

| Public operation | Destination |
| --- | --- |
| `verify_receipt` | `PlatformReceipts.verify_receipt` |
| `attest_receipt` | `PlatformReceipts.attest_receipt` |

Closed schemas only (`trusted-proof-receipt/v1`,
`logic-translation-receipt/v1`). Empty, unknown-schema, forged-kernel, stale,
and cross-authority inputs reject with `authority=none`. Attestation never
exceeds the receipt’s declared ceiling
(`AttestationAuthorityBoundary@2`).

### counterexamples

**Source today:** `explain_counterexample`, minimize/causal/replay tactician
ops, local secret-safe projection helpers.

| Public operation | Destination |
| --- | --- |
| `explain_counterexample` | `PlatformCounterexamples.explain` |
| `minimize_counterexample` | `PlatformCounterexamples.minimize` |
| `explain_counterexample_causal` | `PlatformCounterexamples.explain_causal` |
| `replay_counterexample` | `PlatformCounterexamples.replay` |

Public boundary is `CounterexampleEnvelope@2` only. Private channels,
credentials, and raw provider blobs never appear in `result` or `witnesses`.

### installation

**Source today:** `probe_provider`, `install_provider`.

| Public operation | Destination |
| --- | --- |
| `probe_provider` | `PlatformInstallation.probe_provider` |
| `install_provider` | `PlatformInstallation.install_provider` |

Sole public mutation boundary (`LogicVerificationLazyInstaller@1`). Rules:

* Import of `installation` may register lazy hooks but must not install.
* Live mutation requires `allow_install=True`.
* Probe/inventory/dry-run/offline paths never open the network or import
  installer plugins unless the operation is explicit.
* Install success never becomes semantic or capability authority.

### compatibility

**Source today:** alias maps, migration label field table, production-auth
aliases, SecPAL compatibility lookup, goal-tactician invoke helpers used as
channel adapters.

| Public operation / concern | Destination |
| --- | --- |
| `_PROVIDER_ROLE_ALIASES`, `_PRODUCTION_AUTHORIZATION_ALIASES` | `PlatformCompatibility` |
| `_MIGRATION_LABEL_FIELDS` | `PlatformCompatibility` |
| `secpal_compatibility_lookup` | `PlatformCompatibility.secpal_compatibility_lookup` |
| MCP/CLI invoke bridges that only rename channels | `PlatformCompatibility` (thin) |

Compatibility never implements a second prover. It normalizes names and routes
to the owning service.

### service

**Source today:** `LogicVerificationAPI` method body orchestration,
`get_verification_api`, module-level wrappers, `invoke_goal_tactician*`.

| Responsibility | Destination |
| --- | --- |
| Compose all services above | `LogicPlatformService` |
| Operation dispatch table | `LogicPlatformService.invoke` / method forwards |
| Process-local default instance | `get_platform_service` |
| Public class `LogicVerificationAPI` | Thin adapter over service (stays in `verification_api.py`) |
| Module-level `list_*` / `check` / … wrappers | Stay in `verification_api.py`, call facade |

`LogicPlatformService.to_dict()` advertises the same interface inventory the
facade exposes today (`STABLE_OPERATIONS`, goal-tactician ops, migration ops,
role-closure ops).

## Full public operation → service routing

| Operation group | Operations | Service |
| --- | --- | --- |
| Stable discovery | `list_logic_families`, `list_providers`, `provider_capabilities` | catalog (+ providers for role metadata) |
| Stable compile | `compile_verification_artifact` | obligations |
| Stable execute | `check`, `monitor`, `run_portfolio` | execution |
| Stable cex | `explain_counterexample` | counterexamples |
| Stable receipts | `verify_receipt`, `attest_receipt` | receipts |
| Stable advisor | `advise` | formalization |
| Stable opt-in | `probe_provider`, `install_provider` | installation |
| Migration | `list_namespaces`, `list_namespace_identities`, `dual_read_label`, `canonical_write_label`, `migrate_artifact`, `inspect_translation_loss`, `inspect_provider_authority` | discovery / translations |
| Role closure | `list_provider_roles`, `provider_role`, `secpal_artifact_intake`, `secpal_compatibility_lookup` | providers / compatibility |
| Production auth | `production_authorization_identity`, `production_authorization_check`, `production_authorization_receipt` | providers / execution (check only) |
| Goal tactician | `formalize_goal`, `compare_interpretations`, `discover_missing_proofs`, `plan_proof`, `validate_proof_candidate`, `execute_proof_plan`, `proof_status`, `minimize_counterexample`, `explain_counterexample_causal`, `replay_counterexample`, `list_goal_tactician_operations` | formalization / planning / execution / counterexamples |

Every operation continues to return `VerificationResponse` from `contracts`.

## Dependency graph (import direction)

```
contracts          (no platform deps)
   ↑
catalog, discovery, translations, evidence, compatibility, manifest
   ↑
obligations, providers, formalization, planning, receipts, counterexamples
   ↑
execution, installation          ← explicit runtime only
   ↑
service                          ← composes all
   ↑
verification_api.py              ← public facade / re-exports
   ↑
logic.api / logic.__init__       ← LPC-062 thin adapters
```

Rules:

* Arrows mean “may import”. Cycles are forbidden.
* `execution` and `installation` may import heavier backend/installer modules
  lazily inside methods, never at module import time for pure-data modules.
* `service` imports sibling services; sibling services do not import `service`.
* Downstream family modules (`formalization/*`, `backends/*`, `tactician/*`,
  `software_verification/*`) remain the implementation owners; platform modules
  are facades over them, not copies.

## Facade contract (`verification_api.py` after decomposition)

Retained in the public module:

1. Re-exports of all previously public types, constants, and helpers required by
   tests and external importers.
2. `LogicVerificationAPI` class as a thin method-forwarding adapter.
3. Module-level convenience wrappers (`list_providers`, `check`, …).
4. `get_verification_api` process-local factory.
5. Docstring invariants (quiet import, declarative discovery, explicit
   probe/install, typed responses, no silent success).

Moved out of the public module body into `logic/platform/*`:

* Discovery/migration implementation classes.
* Check/portfolio/execution loops.
* Receipt verification and attestation bodies.
* Counterexample projection bodies.
* Installer/probe mutation bodies.
* Goal-tactician planning/execution bodies.

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-051 | Execution admission depends on v1 adapter non-bypass |
| LPC-032 / LPC-030 | Evidence service preserves orthogonal axes |
| LPC-020..023 | Catalog service reads sealed snapshot / registries |
| LPC-040..043 | Formalization/obligations consume artifacts_v3 / domain slices |
| LPC-050 | Provider protocol @2 is the executable request vocabulary |
| LPC-061 | Pure-data import test against this graph |
| LPC-062 | `logic.api` / `logic.__init__` thin over `LogicPlatformService@1` |
| LPC-070 | Planning defers to canonical proof-plan model; does not fork it |
| LPC-100 | `manifest.py` already ships; service advertises handshake compatibility |
| LPC-110 | Supervisor client handshakes then invokes typed operations through facades |

## What this task does **not** do

* Does not create a second top-level package or a parallel public API name.
* Does not delete or rename public `verification_api` imports.
* Does not implement LPC-061 hermetic import tests (follow-on).
* Does not rewrite `logic.api` / `logic.__init__` facades (LPC-062).
* Does not claim provider availability, proof authority, or production
  readiness from catalog presence or install success.
* Does not replace `BackendRequest@2`, catalog snapshots, or receipt verifiers
  with free-form dictionaries.
* Does not move family-local provers (TDFOL, DCEC, hammers) into `platform/`;
  those remain family adapters behind providers/execution.

## Extraction order (implementation recipe)

When code moves land under `logic/platform/*`, extract in this order so imports
stay fail-closed and tests stay green:

1. **contracts** — pure types; re-export from facade immediately.
2. **evidence**, **translations**, **compatibility** — pure projectors.
3. **catalog**, **discovery** — declarative reads.
4. **providers**, **obligations**, **receipts**, **counterexamples** — lazy
   runtime helpers without process spawn on import.
5. **formalization**, **planning** — advisor/tactician lazy paths.
6. **execution**, **installation** — explicit mutation/runtime last.
7. **service** — wire composition; switch `LogicVerificationAPI` methods to
   forwards.
8. Keep `test_verification_api.py` green after each step; public symbols and
   quiet-import assertions must not regress.

## File ownership

| Path | Role |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/contracts.py` | Shared envelopes and constants |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/catalog.py` | Declarative catalog reads |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/discovery.py` | Canonical discovery + migration dual-read |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/formalization.py` | Advisor / goal formalization entry |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/obligations.py` | Obligation and compile admission |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/translations.py` | Translation inspection / receipts identity |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/providers.py` | Provider roles and non-verification refusal |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/planning.py` | Goal-tactician planning surface |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/execution.py` | Live check / monitor / portfolio / plan execute |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/evidence.py` | Authority and axis projection |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/receipts.py` | Closed receipt verify/attest |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/counterexamples.py` | Public counterexample boundary |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/installation.py` | Probe / install mutation boundary |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/compatibility.py` | Alias and legacy routing |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/service.py` | `LogicPlatformService@1` composition root |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/manifest.py` | Handshake (LPC-100; pre-existing) |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/__init__.py` | Package exports (expand as modules land) |
| `ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py` | Public compatibility facade |
| `data/agent_supervisor/logic_platform_canonicalization/notes/api_decomposition.md` | This note (LPC-060 declared output) |

## Acceptance matrix

| Check | Fail-closed behavior | Owner |
| --- | --- | --- |
| Internal layout matches the fifteen services listed above | Names and roles are exact; no alternate taxonomy | this note |
| Public imports preserved | `logic.verification_api` symbols and module wrappers remain | facade |
| No second implementation | Services compose existing backends/catalogs/receipts | service |
| Pure-data imports quiet | contracts/catalog/discovery/… do not probe/install/network | LPC-061 |
| Runtime explicit | execution + installation only on named ops | execution / installation |
| Authority non-upgrade | evidence + receipts refuse forged/stale/unknown schema | evidence / receipts |
| v1 non-bypass | executable path requires BackendRequest@2 admission | execution + LPC-051 |
| Channel parity unchanged | Python = CLI = MCP operation sets | contracts operation tables |
| Validation | `pytest …/test_verification_api.py -q` stays green | CI / this task gate |

## Status

LPC-060 freezes the internal platform service decomposition and operation
routing for `LogicPlatformService@1`. The public `verification_api` surface
remains the compatibility facade. Follow-on LPC-061/LPC-062 enforce hermetic
imports and thin `logic.api` / `logic.__init__` adapters against this layout.
