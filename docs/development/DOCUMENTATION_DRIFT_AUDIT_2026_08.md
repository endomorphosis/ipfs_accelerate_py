# Documentation drift audit (2026-08)

**Status:** frozen current-tree inventory (Wave 0 evidence)
**Program:** `ipfs-accelerate-documentation-refresh-v1`
**Task:** `DOC-001` / goal `DOC-G011`
**Audit date (UTC):** 2026-08-03
**Tree under audit (HEAD):** `a9efbc9355174a6f23feb78715355f66b8fce9e3`
**Plan baseline commit:** `d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15`
  (`Fix scoped bundle retry and receipt draining`, 2026-08-03)
**Prior freeze HEADs (superseded by this pin):** `df284b01dc66e448aaef3fae66930681ddd8e689`
  (first DOC-001 merge into the refresh branch)
**Published documentation baselines named in navigation:**

| Marker | Source |
| --- | --- |
| 2026-07-24 | `docs/README.md` (“Current documentation baseline”) |
| 2026-07-28 | `docs/INDEX.md` (“Documentation baseline”) |

This file is the reproducible drift inventory for the documentation refresh.
It records what the checked-out tree *actually contains*, which maintained
pages disagree with that tree, and which later `DOC-*` tasks own the fix.
It is **not** an architecture guide and does **not** mark any subsystem
“done.” Findings were re-verified against the HEAD pin above before this
freeze; prior freeze HEADs are historical only.

**Required coverage (acceptance map):** prompt-entrypoint primitives; contract
assurance; **merge-versus-acceptance**; model catalog; endpoint usage;
CID/backend semantics; MCP compatibility; cross-repository changes;
incorrect module/test paths; installation case collision. Old board status
is never treated as current behavior authority.

## Authority and non-authority

| Claim family | Authoritative for this audit | Non-authoritative |
| --- | --- | --- |
| Present module layout, exports, scripts | Files under `ipfs_accelerate_py/`, `pyproject.toml`, `setup.py` | Historical summaries, completion reports, board `- Status:` fields |
| CLI and module entrypoints | Installed console scripts and `python -m … --help` | Stale command examples in guides that disagree with live help |
| Test layout | Paths that exist under `test/` on this tree | Documented pytest paths that do not resolve |
| Task/board completion | Post-merge authoritative completion evidence and present code | ASE / AICAT / GOOSE / ASREF board status tokens, PIDs, merged branches alone |
| Package version string | Must be read from metadata *and* runtime export separately when they disagree | A single version quoted without saying which file it came from |

**Hard rule used by this audit:** old board status is only a *drift signal*.
A task marked `completed` on an ASE or catalog board is never treated as
current behavior authority. Landed modules, tests, and executable help win.

## Reproducible comparison method

Run from the repository root of this worktree:

```bash
# Pin the audit baseline and head
git rev-parse HEAD
git show -s --format='%H %ci %s' d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15

# Published doc baselines still advertise July dates
rg -n 'Documentation baseline|Current documentation baseline' docs/INDEX.md docs/README.md

# Case-fold collision on installation guides (two distinct blobs)
git ls-files -s docs/guides/getting-started/
git hash-object docs/guides/getting-started/installation.md \
  docs/guides/getting-started/INSTALLATION.md

# Version-source disagreement
rg -n 'version\s*=' pyproject.toml setup.py
rg -n '__version__\s*=' ipfs_accelerate_py/__init__.py

# Canonical MCP vs compatibility facade
test -d ipfs_accelerate_py/mcp_server && test -d ipfs_accelerate_py/mcp
head -20 ipfs_accelerate_py/mcp_server/README.md

# Prompt-entrypoint package exists; many board-predicted facades do not
ls ipfs_accelerate_py/agent_supervisor/entrypoints/
test ! -e ipfs_accelerate_py/agent_supervisor/entrypoints/runtime_factory.py

# Incorrect maintained test path
test ! -e test/api/test_unified_cli_integration.py
test -f test/test_unified_cli_integration.py

# Merge ≠ acceptance (fail-closed completion gate)
test -f ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py

git diff --check
```

Between `d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15` and this HEAD the
documentation-program commits include the refresh plan/objectives/board
(`c30aa28fc`), Wave 0 guide conventions (`DOC-003` / `GUIDE_CONVENTIONS.md`),
lifecycle policy (`DOC-002` / `DOCUMENTATION_LIFECYCLE.md`), and successive
freezes of this audit (`DOC-001`). Peer Wave 0 files are **not** behavior
authority for product runtime claims. The behavioral drift described below is
against the **July published baselines** and the **current source tree**, not
against a fictional clean pre-supervisor tree and not against ASE/AICAT board
status rows.

## Tree shape at audit time (inventory)

Approximate maintained Markdown volume on this checkout:

| Area | Role | ~`.md` count |
| --- | --- | --- |
| `docs/guides/` | Task-oriented user/operator journeys | 121 |
| `docs/architecture/` | Plans, hubs, objective heaps, boards, some current architecture | 100 |
| `docs/archive/` | Explicitly historical | 67 |
| `docs/development_history/` | Delivery/session history | 60 |
| `docs/features/`, `docs/api/`, `docs/project/`, top-level `docs/*.md` | Mixed current / historical / migration | remainder of ~470 total |

Code surfaces that grew after the July documentation baselines and still lack
matching *maintained architecture guides* (plans exist; product guides do not):

| Subsystem | Present source anchors | Maintained product guide today |
| --- | --- | --- |
| Prompt-first entrypoints | `ipfs_accelerate_py/agent_supervisor/entrypoints/` (14 Python modules + README) | Plan only: `AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md` |
| Contract assurance refill | `agent_supervisor/objectives/contract_assurance_refill.py`, `runtime_contract_assurance_refill.py`, mismatch refineries | Embedded in operator guide fragments; no dedicated assurance architecture page |
| Merge vs authoritative acceptance | `todo_daemon/authoritative_completion.py`, `merge/*`, `validation/*` | Partially in `AGENT_SUPERVISOR_GUIDE.md`; needs `EXECUTION_AND_RECOVERY` guide (`DOC-013`) |
| Model / AI service catalog | `ipfs_accelerate_py/model_catalog/` (17 modules) | Architecture page `AI_SERVICE_CATALOG.md` exists; not wired as a full routing journey |
| Endpoint usage plane | `ipfs_accelerate_py/endpoint_usage/` (13 modules) + `test/test_endpoint_usage_*.py` | Plan only: `ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md` |
| CID / IPLD / backend semantics | `entrypoints/verified_ipld_backend.py`, `ipfs_backend_router.py`, `ipfs_multiformats.py`, `mcp_server/cid_artifacts.py`, `mcplusplus_module/cid_ucan.py` | Scattered; no single `DISTRIBUTED_RUNTIME` guide yet (`DOC-009`) |
| MCP / MCP++ compatibility | Canonical `mcp_server/` (~228 `.py`), facade `mcp/` (~362 `.py`), `mcplusplus_module/` (~38 `.py`), empty `mcpplusplus/` submodule checkout | Mixed: package README current; many guides still lead with `mcp` alone |
| Cross-repository / nested products | `.gitmodules` pins for `ipfs_kit_py`, `ipfs_datasets_py`, `ipfs_transformers_py`, `ipfs_model_manager_py`, `docs/fastmcp`, `docs/mcp-python-sdk`, `ipfs_accelerate_py/mcplusplus` | `docs/NESTED_PACKAGES.md` present; nested dirs are empty without submodule init |

## Priority legend

| Priority | Meaning |
| --- | --- |
| **P0** | Breaks install, wrong entrypoint, wrong test path in a maintained page, or authority confusion that can mark work complete incorrectly |
| **P1** | Material architecture/user journey missing or contradictory; blocks integrators |
| **P2** | Historical/plan noise, secondary guides, archive labeling |

Owners below name the **documentation refresh task** that should absorb the
fix. They do not claim that task has run.

---

## Finding 1 — Installation case collision (P0)

**Symptom.** Two distinct tracked files differ only by case (plus a third
compatibility path):

| Path | Blob (this tree) | Lines | Content role |
| --- | ---: | ---: | --- |
| `docs/guides/getting-started/installation.md` | `8e18aa3693445b8c986ee31fb8fc773c5e1856ec` | 172 | Maintained installation guide (canonical) |
| `docs/guides/getting-started/INSTALLATION.md` | `321fea6a90680d466beeed5dc4d2e1d833af8b6e` | 736 | Four-line “This document moved” notice **followed by the full legacy installation body** (still present; not a short stub) |
| `docs/guides/getting-started/INSTALLATION_GUIDE.md` | `0ce77664ad1324c93fe81ef165a79803c098c779` | 9 | Compatibility pointer to `installation.md` |

On case-insensitive filesystems (macOS default, Windows), Git cannot
faithfully materialize both `installation.md` and `INSTALLATION.md`. Checkout
behavior is platform-dependent; agents and humans can silently edit the wrong
file or see only one of the two blobs. The uppercase file’s retained legacy
body also still cites paths such as `python -m pytest tests/` (wrong suite
root; live suites are under `test/`).

**Evidence that the program already expects removal of the uppercase file:**
`DOC-021` validation in the operator board includes
`test ! -e docs/guides/getting-started/INSTALLATION.md`.

**Canonical path (this audit):** `docs/guides/getting-started/installation.md`
(also linked from `docs/INDEX.md`, `docs/README.md`, and
`DOCUMENTATION_CURRENT_STATE.md`).

**Owner:** `DOC-021` (installation/quickstart refresh).
**Do not** treat either filename as proof of install correctness; verify
against `pyproject.toml` extras and console scripts.

---

## Finding 2 — Incorrect module and test paths in maintained docs (P0)

These paths are cited by **current** guides/overview pages and **do not
exist** on this tree (or point at the wrong package layout).

### 2.1 Wrong pytest path for unified CLI

| Document | Cited path | Actual path |
| --- | --- | --- |
| `docs/development/testing.md` | `test/api/test_unified_cli_integration.py` | `test/test_unified_cli_integration.py` |
| `docs/architecture/overview.md` (verification recipes if present) | same wrong path | same actual path |
| `docs/development/DOCUMENTATION_CURRENT_STATE.md` | correctly uses `test/test_unified_cli_integration.py` | — |

Reproduce:

```bash
test ! -e test/api/test_unified_cli_integration.py
test -f test/test_unified_cli_integration.py
```

### 2.2 Supervisor modules cited without package directories

`docs/architecture/overview.md` and `docs/agent_supervisor_objective_graph.md`
still present flat basenames that used to live at the package root. After the
layout refactor they live under domain packages. Concrete overview basenames
still used as if they were package-root modules (examples from the Intent /
Projection / Execution tables):

- `objective_graph.py`, `objective_tracker.py`
- `objective_daemon.py`, `backlog_refinery.py`, `taskboard_store.py`
- `bundle_supervisor.py` (alongside correct `todo_daemon/implementation_daemon.py`)

| Documented / implied path | Actual path on this tree |
| --- | --- |
| `agent_supervisor/objective_graph.py` | `agent_supervisor/objectives/objective_graph.py` |
| `agent_supervisor/objective_daemon.py` | `agent_supervisor/objectives/objective_daemon.py` |
| `agent_supervisor/bundle_supervisor.py` | `agent_supervisor/objectives/bundle_supervisor.py` |
| `objective_tracker.py` (flat) | `objectives/objective_tracker.py` |
| `analysis_ast_index.py` / `analysis_retrieval.py` | `analysis/…` |
| `taskboard_store.py` / `todo_vector_index.py` | `task_sources/…` |
| `lease_coordination.py` | `merge/lease_coordination.py` |
| `resource_scheduler.py` | `runtime/resource_scheduler.py` |
| `conflict_graph.py` | `core/conflict_graph.py` |
| `prover_conformance.py` / multi-prover | `proof/…` |
| `formal_plan_*` (flat) | `planning/formal_plan_*.py` (no `formal_plan_parser.py`) |

Objective-graph notes still cite absolute-style anchors under the **old** flat
layout, for example
`ipfs_accelerate_py/agent_supervisor/objective_graph.py` and
`…/bundle_supervisor.py`. Those files are not at package root on this tree;
use `objectives/` as above. The matching test
`test/api/test_agent_supervisor_objective_graph.py` **does** exist (do not
confuse with the unified-CLI path bug in §2.1).

### 2.3 Board-predicted entrypoint and test paths that are not on disk

`docs/architecture/agent_supervisor_prompt_only_entrypoints.todo.md` predicts
facades and tests that **are not present** on this tree, including (sample):

- `entrypoints/runtime_factory.py`, `intent_service.py`, `plan_materializer.py`,
  `launch_profile.py`, `lease_backend.py`, `python_api.py`, `cli.py`,
  `steering_runtime.py`, …
- matching `test/api/test_agent_supervisor_*` files for those facades
- `mcp_server/tools/agent_supervisor_tools/prompt_entrypoint_tools.py`

**What *is* present under `entrypoints/`:** contracts and resolvers
(`contracts.py`, `target_resolver.py`, `state_resolver.py`,
`objective_resolver.py`, `authority_resolver.py`, `capability_resolver.py`,
`profile_resolver.py`, `prompt_broker.py`, `run_registry.py`,
`inference_explain.py`, `plan_lint.py`, `steering_contracts.py`,
`verified_ipld_backend.py`) plus package README describing cold-import rules
and empty `ENTRYPOINT_LAZY_FACADE_EXPORTS`.

Treat board “Predicted files” as a *plan of record*, not as a file listing.

### 2.4 Additional broken paths in secondary guides (P1 sample)

Automated path scan of guides found many missing references (examples):

| Cited | In | Note |
| --- | --- | --- |
| `ipfs_accelerate_py/mcp/mcp_server.py` | docker cache guides | Not a current module |
| `tests/test_comprehensive_validation.py` | integration/testing guides | `tests/` is nearly empty; real suites live under `test/` |
| `ipfs_accelerate_py/scripts/…` | infrastructure guides | Scripts live at repo-root `scripts/` |
| `test/run_tests.py` | runner guides | Path absent |

Full enumeration is not frozen here; re-run the path scan when closing
`DOC-027`/`DOC-028`.

**Owners:** path fixes in maintained overview/testing → `DOC-005`, `DOC-026`;
supervisor path vocabulary → `DOC-010`–`DOC-014`; secondary guides → `DOC-021`–`DOC-026`.

---

## Finding 3 — Package version source disagreement (P0)

| Source | Value on this tree |
| --- | --- |
| `pyproject.toml` `[project].version` | `0.0.45` |
| `setup.py` `version=` | `0.0.45` |
| `ipfs_accelerate_py/__init__.py` `__version__` | `0.4.0` |

Runtime docs that print `ipfs_accelerate_py.__version__` (README, installation
guide, API overview) will disagree with packaging metadata. This is a
**code-owned blocker**, not something documentation should paper over with a
single invented version.

**Owner:** record in lifecycle policy (`DOC-002`); do not “fix” by picking a
version in prose. Packaging/code reconciliation is outside this documentation
program’s edit scope unless a later task explicitly owns it.

---

## Finding 4 — Prompt-entrypoint primitives (P0/P1)

### Landed (source of truth)

- Package: `ipfs_accelerate_py/agent_supervisor/entrypoints/`
- Cold-import contract and composition boundary documented in package README
- Resolvers for target, state, objective, authority, capability, profile
- Prompt body brokering (`prompt_broker.py`)
- Run registry (`run_registry.py`)
- Steering **contracts** (`steering_contracts.py`) without a full steering runtime facade
- Verified CIDv1/IPLD adapter (`verified_ipld_backend.py`)

### Planned / board-incomplete (not behavior authority)

- ASE prompt-only board on this tree: status mix includes many `todo` rows and
  only a small number of `completed` rows; dozens of predicted facade files
  are missing (Finding 2.3).
- Index still advertises
  `AGENT_SUPERVISOR_PROMPT_ONLY_ENTRYPOINTS_PLAN.md` as a plan, which is
  correct labeling — keep it that way until facades land.

### Documentation gap

| Gap | Impact | Owner |
| --- | --- | --- |
| No maintained architecture guide for prompt-first runtime, run registry, or steering | Integrators read the plan as if facades exist | `DOC-014` |
| Operator guide does not yet center `entrypoints` as the composition root | Agents import lower packages incorrectly | `DOC-014`, `DOC-022` |
| Board `completed` rows without files | Risk of treating ASE status as shipped | **Audit rule:** ignore board status; use directory listing |

---

## Finding 5 — Contract assurance (P1)

**Present code:**

- `objectives/contract_assurance_refill.py` — bounded refill from analyzer
  results; generated tasks are evidence, never completion authority
- `objectives/contract_mismatch_refinery.py` and runtime variants
- Analysis contracts under `agent_supervisor/analysis/`
- MCP contract edit packets under `proof/`
- Related **persistence** surfaces (not acceptance authority):
  `entrypoints/run_registry.py`, `entrypoints/state_resolver.py`,
  `test/api/test_agent_supervisor_bounded_persistence.py`, and merge/lease
  stores under `merge/` — durable run/state/evidence retention is present in
  code but lacks a single maintained “what is authoritative state” guide

**Doc drift:**

- No dedicated maintained page explaining the assurance loop (scan → packet →
  refinery → task → validation → acceptance).
- Plans and objective heaps (`agent_supervisor_analysis_*.todo.md`, codebase
  proof boards) mix completed markers with predicted modules such as
  `proof_attestation.py` at wrong paths (actual:
  `proof/proof_attestation.py`).
- Persistence and registry semantics are scattered across operator fragments
  and package READMEs; integrators cannot see which stores are cold-import
  safe, which are CID-backed, and which are ephemeral without reading code.

**Owner:** `DOC-011` / `DOC-012` (planning and assurance architecture), with
lifecycle rules in `DOC-002`; persistence narrative also feeds `DOC-013` and
`DOC-019`/`DOC-020` when ADRs land.

---

## Finding 6 — Merge-versus-acceptance (P0)

**Topic name (acceptance):** merge-versus-acceptance — merge evidence is not
acceptance authority.

**Source of truth:**
`ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py`

The module docstring states the binding invariant:

- An implementation commit, merge-queue status, or Git ancestry relationship
  is evidence that **code landed**.
- **None** of those facts authorizes task-board completion.
- Acceptance states (constants on this tree):
  - `implemented_merged_but_pending` (`ACCEPTANCE_STATE_MERGED_PENDING`)
  - `authoritatively_completed` (`ACCEPTANCE_STATE_AUTHORITATIVE`)
  - `acceptance_reopened` (`ACCEPTANCE_STATE_REOPENED`)
- Gate kinds (`AUTHORITATIVE_COMPLETION_GATE_KINDS`): `merge`, `freshness`,
  `semantic`, `proof`, `provider_review`, `deterministic_only`.

Supporting surfaces: `merge/merge_resolver.py`, `merge/merge_queue.py`,
`validation/proposal_validation.py`, tests such as
`test/api/test_agent_supervisor_authoritative_task_completion.py`.

**Doc drift:**

- Architecture overview still collapses “merge/completion receipts” into one
  arrow, which under-sells the pending-acceptance state.
- Historical ASE boards marked `completed` must not be read as
  authoritative completion of those tasks on a later tree.
- This documentation program itself carries
  `completion_authoritative: false` on implementation packets — consistent
  with the code contract.

**Owner:** `DOC-013` (execution and recovery), `DOC-015`/`DOC-016` (ADRs for
intent/trust/evidence).

---

## Finding 7 — Model catalog (P1)

**Source of truth:** `ipfs_accelerate_py/model_catalog/`

Cold, side-effect-free catalog contracts: identities, schema, registry,
resolver, snapshot, security, receipts. Public architecture narrative exists
at `docs/architecture/AI_SERVICE_CATALOG.md` and correctly states that the
catalog is **not** a fifth inference router.

**Doc drift:**

| Gap | Detail |
| --- | --- |
| Package path under-documented in journeys | Guides still lead with `ModelManager` alone; few point at `model_catalog` imports |
| Naming split | Prose says “AI Service Catalog”; package directory is `model_catalog/` |
| Index baseline | July baseline predates catalog conformance and security rollout commits |
| Plan vs landed | `ai_service_catalog.todo.md` status is not completion authority |

**Owner:** `DOC-007` (model/service routing architecture), `DOC-023` (MCP/catalog
operator journeys as applicable).

---

## Finding 8 — Endpoint usage (P1)

**Source of truth:** `ipfs_accelerate_py/endpoint_usage/`

Contracts for endpoint/account-scoped limits, atomic reservation ledger,
routing helpers, controls, receipts, observability. Conformance and fault
tests exist as `test/test_endpoint_usage_*.py`.

**Doc drift:**

- Primary narrative is still the plan
  `docs/architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md` (opt-in `off` /
  `shadow` promotion language).
- No maintained “current behavior” page separates plan defaults from landed
  modules.
- Overview and quickstart do not mention usage-aware admission.

**Owner:** `DOC-007` (compose with catalog), `DOC-013` (supervisor capacity),
user-facing notes in `DOC-023`/`DOC-025` as needed.

---

## Finding 9 — CID / backend semantics (P1)

**Source anchors on this tree** (paths under `ipfs_accelerate_py/` unless noted):

| Concern | Path |
| --- | --- |
| Supervisor coordination CIDs | `agent_supervisor/entrypoints/verified_ipld_backend.py` (strict CIDv1, sha2-256, raw/dag-json; HF synthetic CIDs not admitted) |
| Multiformats helpers | `agent_supervisor/multiformats_identity.py`, `ipfs_multiformats.py` |
| Backend router roles | `ipfs_backend_router.py` (filesystem, HF cache, `ipfs_kit_py`, Kubo-compatible) |
| MCP artifacts | `mcp_server/cid_artifacts.py` |
| MCP++ CID/UCAN | `mcplusplus_module/cid_ucan.py` |
| Catalog/usage content IDs | `model_catalog/identity.py`, `endpoint_usage/identity.py` (`canonical_cid`, `content_cid`) |
| Run/state persistence (related) | `agent_supervisor/entrypoints/run_registry.py`, state resolvers; not a substitute for a distributed-runtime guide |

**Doc drift:**

- IPFS feature guides describe kit integration at a high level but do not
  document the supervisor’s fail-closed CID profile or the distinction
  between coordination CIDs, catalog snapshot revisions, and MCP++ interface
  CIDs (the catalog architecture page does distinguish catalog versions).
- Nested `ipfs_kit_py` checkout is **empty** without submodule init; docs that
  assume kit symbols are importable without stating the submodule prerequisite
  over-claim.

**Owner:** `DOC-009` (distributed runtime), `DOC-008` (MCP runtime CID
artifacts), `DOC-010` (integration boundaries).

---

## Finding 10 — MCP compatibility and dual trees (P0/P1)

**Canonical runtime (package README, this tree):**
`ipfs_accelerate_py.mcp_server`

**Compatibility facade:** `ipfs_accelerate_py.mcp`

| Surface | Approx. Python files | Role |
| --- | --- | --- |
| `mcp_server/` | ~228 | Canonical unified runtime, FastAPI service, CID artifacts, compatibility bridges |
| `mcp/` | ~362 | Compatibility facade; still hosts `cli.py` used by operators |
| `mcplusplus_module/` | ~38 | MCP++ descriptors, P2P transport, CID/UCAN, topology |
| `ipfs_accelerate_py/mcplusplus/` | 0 files (empty submodule) | Declared in `.gitmodules`; not populated in this worktree |

**Doc drift:**

| Issue | Examples |
| --- | --- |
| Guides lead with facade module only | `docs/guides/MCP_SETUP_GUIDE.md`, `QUICKSTART.md`, `features/auto-healing/overview.md` use `python -m ipfs_accelerate_py.mcp.cli` or `.mcp.server` without always stating `mcp_server` is canonical |
| Package `mcp/README.md` still reads like a primary product README | Contradicts `mcp_server/README.md` cutover language |
| Empty `mcplusplus` submodule | Cross-repo docs that assume upstream MCP++ tree contents will fail offline |
| Vendored doc submodules | `docs/fastmcp`, `docs/mcp-python-sdk` are external clones; not project contracts |

Maintained pages that already get this right (use as templates): root
`README.md`, `docs/api/overview.md`, `docs/architecture/overview.md`,
`DOCUMENTATION_CURRENT_STATE.md`.

**Owner:** `DOC-008` (MCP runtime architecture), `DOC-022` (MCP operator
journey), `DOC-010` (integration boundaries).

---

## Finding 11 — Cross-repository and nested package state (P1)

From `.gitmodules` and `docs/NESTED_PACKAGES.md`:

| Nested path | Remote (declared) | This worktree |
| --- | --- | --- |
| `ipfs_kit_py` | `endomorphosis/ipfs_kit_py` | Empty directory |
| `ipfs_datasets_py` | `endomorphosis/ipfs_datasets_py` | Empty directory |
| `ipfs_transformers_py` | `endomorphosis/ipfs_transformers_py` | Empty directory |
| `ipfs_model_manager_py` | `endomorphosis/ipfs_model_manager_py` | Empty directory |
| `ipfs_accelerate_py/mcplusplus` | `endomorphosis/Mcp-Plus-Plus` | Empty directory |
| `docs/fastmcp`, `docs/mcp-python-sdk` | jlowin upstreams | May be unpopulated |
| `test/huggingface_transformers`, doc-builder | Hugging Face | Optional test assets |

`ipfs_accelerate_py/__init__.py` optionally prepends `external/<package>` and
nested checkouts onto `sys.path`. Documentation that says “import ipfs_kit_py”
without submodule or extras prerequisites is incomplete.

**Owner:** `DOC-010` (integration boundaries), install notes in `DOC-021`.
This program does **not** update sibling repositories’ documentation.

---

## Finding 12 — Stale ASE / program board status (P0 policy)

Boards under `docs/architecture/*.todo.md` contain many `- Status: completed`
rows (codebase proof, goose integration, VFS assurance, partial ASE prompt
entrypoints, etc.). Those statuses are **point-in-time delivery signals**.

For documentation and agent behavior:

1. Never promote board status to “current API exists.”
2. Verify with `test -f` / imports / focused pytest.
3. Prefer package READMEs and maintained guides once refresh tasks land.
4. The documentation refresh board itself (`documentation_refresh.todo.md`)
   is operator-owned input; implementation agents must not rewrite it.

This audit intentionally records ASE completion marks only as a warning that
**predicted files and completed status diverge** (Finding 2.3 / 4).

---

## Finding 13 — July documentation baselines vs code delta (P1)

Navigation still freezes reader expectation at 2026-07-24 / 2026-07-28 while
the tree already contains post-baseline systems (prompt entrypoints, catalog,
endpoint usage, authoritative completion, MCP cutover defaults, verified IPLD
adapter). The refresh plan at
`docs/architecture/DOCUMENTATION_REFRESH_PLAN_2026_08.md` pins baseline
`d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15` for this program.

**Owner:** `DOC-027` / `DOC-028` update index/current-state baseline markers
after architecture and journey tasks land. Until then, treat July dates as
**stale navigation metadata**, not as a claim that the code stopped changing.

---

## Source-to-document ownership matrix (routing)

| Claim area | Primary code anchors | Best current doc (may be partial) | Refresh owner |
| --- | --- | --- | --- |
| Package metadata / extras / scripts | `pyproject.toml`, `setup.py` | `guides/getting-started/installation.md` | `DOC-021` |
| Python exports | `__init__.py`, `ipfs_accelerate.py` | `api/overview.md` | `DOC-021` / API refresh |
| Unified CLI vs AI CLI | `cli_entry.py` / `cli.py` vs `ai_inference_cli.py` | CLI guides + current-state note on hyphen vs underscore | `DOC-021` |
| Prompt entrypoints | `agent_supervisor/entrypoints/` | Package README; plan only at architecture level | `DOC-014` |
| Contract assurance | `objectives/contract_*`, analysis contracts | Operator guide fragments | `DOC-011`/`DOC-012` |
| Merge vs acceptance | `todo_daemon/authoritative_completion.py`, `merge/`, `validation/` | Guide partial | `DOC-013` |
| Model catalog | `model_catalog/` | `AI_SERVICE_CATALOG.md` | `DOC-007` |
| Endpoint usage | `endpoint_usage/` | Plan only | `DOC-007` |
| CID / backends | `verified_ipld_backend.py`, `ipfs_backend_router.py`, `cid_artifacts.py` | Scattered IPFS guides | `DOC-009` |
| MCP canonical runtime | `mcp_server/` | Package README, API overview | `DOC-008`/`DOC-022` |
| MCP facade | `mcp/` | Too many guides treat as primary | `DOC-008`/`DOC-022` |
| Nested / sibling repos | `.gitmodules`, empty nested dirs | `NESTED_PACKAGES.md` | `DOC-010` |
| Lifecycle / authority | (policy) | `DOCUMENTATION_CURRENT_STATE.md` (short) | `DOC-002` |
| Persistence / run-state registries | `entrypoints/run_registry.py`, `state_resolver.py`, merge stores | Package README fragments | `DOC-013` / `DOC-019` |
| Drift inventory | (this file) | **this file** | `DOC-001` (freeze re-verified; board status is not authority) |

---

## Prioritized gap list (for later writers)

### P0 — fix or label before trusting journeys

1. Resolve installation case collision (`installation.md` vs `INSTALLATION.md`).
2. Fix `test/api/test_unified_cli_integration.py` → `test/test_unified_cli_integration.py` in testing docs.
3. Replace flat supervisor module basenames with package-qualified paths in
   overview and objective-graph notes.
4. Always state MCP canonical (`mcp_server`) vs facade (`mcp`) in operator docs.
5. Never use ASE/board `completed` as proof of presence; check the tree
   (board status is not current behavior authority).
6. Enforce merge-versus-acceptance: do not collapse merge into acceptance;
   cite authoritative completion states and gate kinds.
7. Disclose version disagreement (`0.0.45` packaging vs `0.4.0` `__version__`)
   instead of quoting one number as universal.

### P1 — architecture and integrator gaps

1. Maintained guides for prompt entrypoints, catalog+usage routing, distributed
   CID/backends, MCP runtime, integration boundaries (Wave 1 tasks).
2. Document submodule requirements for nested packages and empty checkouts.
3. Align secondary guides that still reference `tests/…` or
   `ipfs_accelerate_py/mcp/mcp_server.py`.
4. Advance documentation baseline dates only after content catches up
   (`DOC-027`/`DOC-028`).

### P2 — hygiene

1. Label archive and development_history documents when linked from indexes.
2. Keep plans under architecture clearly marked planned vs landed.
3. Defer mass deletion; prefer status labels (program non-goal).

---

## Explicit non-findings (avoid false drift)

- `ipfs_accelerate_py.mcp.cli` **does exist** and remains a supported process
  entry; the drift is **primacy/canonical labeling**, not total removal.
- `test/test_unified_cli_agent_supervisor.py` and
  `test/mcp_server/test_agent_supervisor_tools.py` **do exist** as cited in the
  operator guide’s self-improvement suite.
- `docs/architecture/AI_SERVICE_CATALOG.md` is real maintained architecture
  prose for the catalog, not merely a stub plan.
- Empty nested product directories are an **environment/submodule** condition of
  this worktree, not proof that upstream projects were deleted.

---

## Verification recipe (rerun on a later checkout)

```bash
test -f docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md
rg -q 'd7da3d6bf8ca2f7ec870d03742b09f26e3e16d15' \
  docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md

# Spot-check the highest-risk drifts still hold or were fixed:
git ls-files docs/guides/getting-started/ | rg -i installation
test -f test/test_unified_cli_integration.py
test ! -e test/api/test_unified_cli_integration.py || \
  echo "NOTE: unified CLI test path may have been relocated"
test -d ipfs_accelerate_py/agent_supervisor/entrypoints
test -d ipfs_accelerate_py/model_catalog
test -d ipfs_accelerate_py/endpoint_usage
test -f ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py
rg -n '__version__|version' ipfs_accelerate_py/__init__.py pyproject.toml setup.py

git diff --check
```

If a later tree fixes a finding, update this audit’s “fixed on” notes in a
new dated audit or in `DOCUMENTATION_CURRENT_STATE.md` — do not silently
rewrite history without a new baseline commit marker.

---

## Document control

| Field | Value |
| --- | --- |
| Interfaces | `DocumentationDriftAudit@1`, `SourceDocOwnershipFinding@1` |
| Allowed edit path for producing task | `docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md` only |
| Protected inputs (do not modify in implementation) | `DOCUMENTATION_REFRESH_PLAN_2026_08.md`, `documentation_refresh.objectives.md`, `documentation_refresh.todo.md` |
| Baseline commit pin | `d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15` |
| Tree under audit (HEAD) | `a9efbc9355174a6f23feb78715355f66b8fce9e3` |
| Evidence sources used | Git history and current tree under `ipfs_accelerate_py/`, `docs/`, `test/`, `pyproject.toml`, `setup.py`, `.gitmodules`; executable path existence checks only (no board status as authority) |
| Next consumers | `DOC-002` lifecycle policy; Wave 1 architecture tasks `DOC-005`–`DOC-014` |
