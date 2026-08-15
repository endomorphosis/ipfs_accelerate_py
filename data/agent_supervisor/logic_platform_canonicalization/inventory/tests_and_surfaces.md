# LPC-007 Inventory: tests, MCP, CLI, and deprecated modules

**Task:** LPC-007  
**Goal:** LPC-G010  
**Interface:** `LogicPlatformInventory@1`  
**Machine-readable companion:** `tests_and_surfaces.json`  
**Observed:** 2026-08-15

## Scope

This slice classifies:

| Category | Meaning |
| --- | --- |
| `mcp_cli_python_exposure` | Public Python, CLI, and MCP channels for logic/verification |
| `compatibility_shim` | Facades and adapters that preserve old imports or wire shapes |
| `deprecated_module` | Surfaces marked deprecated, legacy, or superseded |
| `test_and_conformance_corpus` | Relevant unit, integration, packaging, and parity test suites |

Classification vocabulary (LPC-G010): `canonical`, `canonical_component`, `compatibility_facade`, `legacy`, `experimental`, `declaration_only`, `generated`, `duplicate`, `obsolete`, `unresolved`.

Related slices own other inventory categories:

- **LPC-002** — public API export census (`logic.__init__`, `logic.api`, CLI/MCP symbol lists)
- **LPC-006** — supervisor semantic types and datasets imports
- **LPC-008** — composed `inventory.json` / `INDEX.md`

## Scan authority

| Source | Observation |
| --- | --- |
| Accelerate implementation head | `ea11293bb996f052d620eae989f5377a956764b1` (plan authority) |
| Accelerate reviewed baseline | `485edc0871c55b0e2ef21d83bece9fa12c2c8d84` |
| Datasets reviewed baseline | `ac82107e246b30e35a2bbdcf75e01370d22350c6` |
| Datasets external path | `/home/barberb/lift_coding/external/ipfs_datasets` — **not present** in this scan environment |
| Nested gitlink `ipfs_datasets_py/` | **empty** (submodule not initialized in this worktree) |

Rule: current heads remain implementation authority. Datasets-only paths are still inventoried from plans, tests, docs, and completion receipts, and are marked `unresolved` when the filesystem is absent.

## Summary

| Category | Count |
| --- | --- |
| MCP/CLI/Python exposures | 15 |
| Compatibility shims | 8 |
| Deprecated / legacy modules | 6 |
| Test / conformance corpora | 20 |
| **Total items** | **49** |
| Explicit unresolved notes | 3 |

**Parity status:** partial existing evidence; full `LogicOperationCatalog@1` gate is planned under LPC-G130 / LPC-130.

---

## 1. MCP / CLI / Python exposures

### 1.1 Canonical Python surfaces (datasets-owned)

| ID | Surface | Path | Classification |
| --- | --- | --- | --- |
| `surface.python.logic_verification_api` | `LogicVerificationAPI@1` | `ipfs_datasets_py/logic/verification_api.py` | **canonical** |
| `surface.python.goal_tactician_api` | `GoalTacticianAPI@1` | same module | **canonical** |
| `surface.python.logic_api_v1` | `logic.api` public import | `ipfs_datasets_py/logic/api.py` | **compatibility_facade** |
| `surface.python.logic_package_init` | `logic.__init__` re-exports | `ipfs_datasets_py/logic/__init__.py` | **compatibility_facade** |

**Stable verification operations (`STABLE_OPERATIONS`):**

`list_logic_families`, `list_providers`, `provider_capabilities`, `compile_verification_artifact`, `check`, `monitor`, `run_portfolio`, `explain_counterexample`, `verify_receipt`, `attest_receipt`, `advise`, `probe_provider`, `install_provider`

Notes:

- Import of `verification_api` must remain side-effect free (no probes, installs, network).
- `install_provider` is opt-in mutation, **not** ordinary verify.
- Goal-tactician operations share the module but must **not** merge into `STABLE_OPERATIONS`.

**Goal tactician operations:**

`formalize_goal`, `compare_interpretations`, `discover_missing_proofs`, `plan_proof`, `validate_proof_candidate`, `execute_proof_plan`, `proof_status`, `minimize_counterexample`, `explain_counterexample_causal`, `replay_counterexample`, `list_goal_tactician_operations`

### 1.2 CLI surfaces

| ID | Surface | Classification | Notes |
| --- | --- | --- | --- |
| `surface.cli.logic_verification_cli` | `LogicVerificationCLI@1` (`logic/cli.py`) | **canonical** | Datasets-owned; example entry `ipfs-datasets logic …` |
| `surface.cli.goal_tactician_commands` | `GOAL_TACTICIAN_CLI_TO_OPERATION` | **canonical_component** | Closed CLI vocabulary 1:1 with Python/MCP |
| `surface.cli.accelerate_console_scripts` | Root `setup.py` entry points | **declaration_only** | No logic CLI registered on accelerate |

CLI command names (goal tactician):

| Python operation | CLI command |
| --- | --- |
| `formalize_goal` | `goal-formalize` |
| `compare_interpretations` | `goal-compare-interpretations` |
| `discover_missing_proofs` | `goal-discover-missing-proofs` |
| `plan_proof` | `goal-plan-proof` |
| `validate_proof_candidate` | `goal-validate-candidate` |
| `execute_proof_plan` | `goal-execute-plan` |
| `proof_status` | `goal-proof-status` |
| `minimize_counterexample` | `goal-minimize-counterexample` |
| `explain_counterexample_causal` | `goal-explain-counterexample` |
| `replay_counterexample` | `goal-replay-counterexample` |
| `list_goal_tactician_operations` | `goal-list-operations` |

### 1.3 MCP surfaces

| ID | Surface | Path | Classification |
| --- | --- | --- | --- |
| `surface.mcp.logic_verification` | `LogicVerificationMCP@1` | `ipfs_datasets_py/mcp_server/tools/logic_verification.py` | **canonical** |
| `surface.mcp.goal_tactician_tools` | `GOAL_TACTICIAN_TOOL_TO_OPERATION` | `verification_api.py` maps | **canonical_component** |
| `surface.mcp.accelerate_native_logic_tools` | mcp_server `logic_tools` category | `ipfs_accelerate_py/mcp_server/tools/logic_tools/native_logic_tools.py` | **compatibility_facade** |
| `surface.mcp.accelerate_legacy_mcp_package` | `ipfs_accelerate_py.mcp` | `ipfs_accelerate_py/mcp/` | **compatibility_facade** |

**Canonical MCP verification tools (representative):**  
`verification_list_providers`, `verification_provider_capabilities`, `verification_check`, `verification_portfolio`, `verification_probe_provider`

**Goal tactician MCP tools:**  
`goal_tactician_*` names map 1:1 to operations (see JSON).

**Accelerate native `logic_tools` registered tools (legacy family surface):**

`logic_capabilities`, `logic_health`, `tdfol_parse`, `tdfol_convert`, `tdfol_prove`, `tdfol_kb_add_axiom`, `tdfol_kb_add_theorem`, `tdfol_kb_query`, `tdfol_kb_export`, `cec_prove`, `cec_check_theorem`, `cec_parse`, `cec_validate_formula`, `cec_analyze_formula`, `cec_formula_complexity`

These prefer datasets `mcp_server.tools.logic_tools.*` delegates and fall back to fail-closed local stubs when datasets is unavailable. They are **not** the modern `LogicVerificationMCP@1` channel.

### 1.4 Supervisor Python exposures (accelerate-owned)

| ID | Surface | Path | Classification |
| --- | --- | --- | --- |
| `surface.python.supervisor_canonical_logic_adapter` | `SupervisorCanonicalLogicAdapter@1` | `proof/canonical_logic_adapter.py` | **canonical** |
| `surface.python.supervisor_logic_provider_contract` | `SupervisorLogicProviderFacade` | `proof/logic_provider_contract.py` | **canonical_component** |
| `surface.python.supervisor_hammer_adapter` | Hammer portfolio adapter | `integrations/ipfs_datasets_logic_provider.py` | **canonical_component** |
| `surface.python.supervisor_formal_verification_provider` | Operational provider API | `proof/formal_verification_provider.py` | **canonical** |

Supervisor owns scheduling, isolation, resources, and candidacy. Datasets owns semantic identity. Transport/facade success is never proof success.

---

## 2. Compatibility shims

| ID | Shim | Classification | Role |
| --- | --- | --- | --- |
| `shim.logic_api_facade` | `logic.api` / `logic.__init__` | compatibility_facade | Thin public imports over platform services (LPC-062) |
| `shim.verification_api_facade` | `verification_api.py` | compatibility_facade | Public facade while internals move to `logic/platform/*` (LPC-060) |
| `shim.provider_protocol_v1_adapter` | `protocol_v1_adapter.py` | compatibility_facade | v1 payloads cannot bypass BackendRequest@2 |
| `shim.accelerate_native_logic_tools_delegate` | native_logic_tools fallback | compatibility_facade | Datasets delegate + local non-proving stubs |
| `shim.supervisor_logic_provider_facade` | provider contract facade | compatibility_facade | Lazy supervisor ↔ datasets wire conversion |
| `shim.canonical_logic_adapter_facades` | adapter singletons / facades | compatibility_facade | Existing consumer compatibility |
| `shim.mcp_package_facade` | `ipfs_accelerate_py.mcp` | compatibility_facade | Alternate MCP package; not preferred entry |
| `shim.goal_tactician_legacy_aliases` | migration aliases doc | compatibility_facade | Preserves legacy public names |

---

## 3. Deprecated and legacy modules

| ID | Module / surface | Classification | Disposition |
| --- | --- | --- | --- |
| `deprecated.legacy_cec_tdfol_mcp_tools` | TDFOL/CEC MCP tools as primary API | **legacy** | Keep as family adapters; prefer `verification_*` / `goal_tactician_*` |
| `deprecated.legacy_mcp_tool_register_helpers` | `mcp/tools/*` `register_*` helpers | **legacy** | Emit DeprecationWarning; use mcp_server natives |
| `deprecated.legacy_mcp_facade_runtime` | `mcp/server.py` D2 opt-in path | **legacy** | Canonical runtime is `mcp_server` |
| `deprecated.plat2_holdout_alias` | `PLAT2_HOLDOUT_PROMOTION_GATE` | **legacy** | Explicit deprecated alias of SRT gate |
| `deprecated.legacy_logic_api_name` | `logic.api` as primary verify name | **legacy** | Migration preserves name; modern path is `verification_api` |
| `deprecated.legacy_domain_runners` | CEC/TDFOL/external runners as authorities | **legacy** | Become adapters to ir_core/backends (LFV plan) |

Exhaustive datasets-internal deprecation list remains **unresolved** until the datasets checkout is available.

---

## 4. Test and conformance corpora

### 4.1 Accelerate — channel and contract (present)

| ID | Path | Lane | Classification |
| --- | --- | --- | --- |
| `corpus.accelerate.goal_tactician_cli_mcp_parity` | `test/api/test_goal_tactician_cli_mcp_parity.py` | hermetic when datasets importable | **canonical** |
| `corpus.accelerate.root_mcp_formal_verification_parity` | `test/api/test_root_mcp_formal_verification_parity.py` | hermetic when datasets importable | **canonical** |
| `corpus.accelerate.logic_provider_contract` | `test/api/test_logic_provider_contract.py` | hermetic when datasets importable | **canonical** |
| `corpus.accelerate.canonical_logic_adapter` | `test/api/test_agent_supervisor_canonical_logic.py` | hermetic | **canonical** |
| `corpus.accelerate.logic_receipt_authority` | `test/api/test_logic_receipt_authority_boundary.py` | hermetic when datasets importable | **canonical** |
| `corpus.accelerate.logic_verification_install_provider` | `test/api/test_logic_verification_api_install_provider.py` | hermetic when datasets importable | **canonical** |
| `corpus.accelerate.logic_submodule_alignment` | `test/api/test_logic_submodule_alignment.py` | hermetic | **canonical_component** |
| `corpus.accelerate.prover_conformance` | `test/api/test_agent_supervisor_prover_conformance.py` | hermetic | **canonical** |
| `corpus.accelerate.ipfs_datasets_logic_conformance` | `test/api/test_agent_supervisor_ipfs_datasets_logic_conformance.py` | installed-provider optional | **canonical** |

### 4.2 Accelerate — program-logic / hammer / tactician suites (present)

| ID | Members (representative) | Classification |
| --- | --- | --- |
| `corpus.accelerate.program_logic_prediction` | program_logic_* contracts, gap slicer, CEGIS, repair fixed-point, live repair | **canonical_component** |
| `corpus.accelerate.tactician_hammer_logic_repair` | hammer capabilities, coordinator, rollout, benchmark, e2e, logic provider | **canonical_component** |
| `corpus.accelerate.formal_verification_tactician` | tactician docs, corpus contract, provider-role, FV capabilities/contracts, CEGIS tactician | **canonical_component** |
| `corpus.accelerate.logic_formal_verification_completion` | `test_logic_formal_verification_completion.py` + LFV receipt | **canonical_component** |
| `corpus.accelerate.mcp_uni_logic_tools` | UNI-105/161/221/303 logic_tools tests under `mcp/tests/` | **canonical_component** |

### 4.3 Accelerate — packaging and integration (present)

| ID | Path / members | Lane | Classification |
| --- | --- | --- | --- |
| `corpus.accelerate.packaging_logic_clean_install` | `test/packaging/test_logic_verification_clean_install.py` (+ distribution contract) | hermetic packaging | **canonical** |
| `corpus.accelerate.integration_formal_verification` | `test/integration/**` formal verification / toolchain matrix | network/heavy opt-in | **canonical_component** |

### 4.4 Datasets corpora (documented; checkout absent → unresolved)

| ID | Expected root | Classification |
| --- | --- | --- |
| `corpus.datasets.unit_logic` | `ipfs_datasets_py/tests/unit/logic/` (catalog, axes, formalization, syntax, protocol v2, pure import, channel parity, tactician, cache key, manifest, …) | **unresolved** |
| `corpus.datasets.integration_logic` | CLI/MCP integration, goal tactician public API, logic_pipeline matrix, capability/conformance fixtures | **unresolved** |

Expected members are listed in `tests_and_surfaces.json` from LPC objectives/todo and the LFV completion receipt.

### 4.5 Planned LPC matrix (declaration-only)

| ID | Future artifact | Classification |
| --- | --- | --- |
| `corpus.planned.lpc_mandatory_matrix` | `notes/test_matrix.md` (LPC-140) | **declaration_only** |
| `corpus.planned.channel_parity_tests` | `test_channel_parity.py`, `test_logic_channel_parity.py`, `test_direct_vs_supervisor_logic_parity.py` | **declaration_only** |

Required matrix buckets (LPC-G140): pure import, catalog, syntax/formalization, typed request admission, translation safety, provider protocol, evidence/receipts, supervisor parity, real-provider smoke, bounded adversarial.

---

## 5. Channel parity picture

```
Python LogicVerificationAPI@1 / GoalTacticianAPI@1
        │
        ├─ CLI  LogicVerificationCLI@1  (datasets logic/cli.py)
        │         goal-* commands 1:1
        │
        ├─ MCP  LogicVerificationMCP@1 + goal_tactician_* tools
        │         (datasets; parent may re-export)
        │
        └─ Accelerate native logic_tools (TDFOL/CEC)
                  compatibility / legacy family surface
```

Existing evidence:

- `test_goal_tactician_cli_mcp_parity.py` — GoalTactician channel closure
- `test_root_mcp_formal_verification_parity.py` — STABLE_OPERATIONS MCP/Python envelope
- Full operation catalog + direct-vs-supervisor parity remain **planned** (LPC-130 / LPC-141)

Rule retained: **installation is not verify**; transport success is not proof success; mocks cannot satisfy real-provider gates.

---

## 6. Unresolved items (not silently dropped)

1. **Datasets checkout absent** — cannot file-list unit/integration corpora or public modules in this worktree.
2. **Full logic API export census** — owned primarily by LPC-002 once datasets is readable.
3. **Exhaustive datasets deprecated-module list** — requires datasets source scan at implementation head.

LPC-008 must incorporate these unresolved entries rather than treating silence as completeness.

---

## 7. Cross-references

| Artifact | Role |
| --- | --- |
| `tests_and_surfaces.json` | Machine-readable slice inventory |
| `docs/architecture/logic_platform_canonicalization.objectives.md` | LPC-G010 vocabulary and acceptance |
| `docs/architecture/logic_platform_canonicalization.todo.md` | LPC-007 acceptance |
| `docs/formal_verification_tactician.md` | Channel tables and operations |
| `docs/architecture/logic_formal_verification_expansion_completion_receipt.json` | Datasets corpus content identities |
| Future `inventory/INDEX.md` + `inventory.json` | LPC-008 composition |

## Acceptance (LPC-007)

Relevant test/conformance corpora, MCP/CLI/Python exposures, compatibility shims, and deprecated modules are classified above and in `tests_and_surfaces.json`. Unresolved items are listed explicitly.
