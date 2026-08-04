# Documentation Current State

**Status:** Current
**Owner:** documentation-governance
**Audience:** maintainers, documentation authors, implementation agents, and
readers who need to know which surfaces are normative for the checked-out tree
**Scope:** Maintained-surface matrix, source-of-truth map, code-owned blockers,
offline audit checklist, and next-audit triggers after the documentation-refresh
closeout (DOC-028).
**Non-goals:** Restating leaf guide content; mass-moving archives; resolving
code-owned packaging or CLI contradictions in prose; inventing CI gates that
do not exist.
**Sources:** [DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md);
[DOCUMENTATION_MANIFEST.md](DOCUMENTATION_MANIFEST.md);
[DOCUMENTATION_MAINTENANCE.md](DOCUMENTATION_MAINTENANCE.md);
[DOCUMENTATION_VALIDATION_POST_MERGE_2026_08_03.md](DOCUMENTATION_VALIDATION_POST_MERGE_2026_08_03.md);
[DOCUMENTATION_VALIDATION_2026_08.md](DOCUMENTATION_VALIDATION_2026_08.md);
`docs/README.md`; `docs/INDEX.md`; `pyproject.toml`; live package layout.
**Last verified:** 2026-08-04; navigation and maintained-surface inventory
revalidated after integrating `origin/main`, plus post-closeout revalidation of
the AI Service Catalog and HF model-server Reference pages, CLI help and
version-pin fixes, and an allowlisted offline link checker for maintained
surfaces. Exact merge identity for the earlier integrated matrix:
[DOCUMENTATION_VALIDATION_POST_MERGE_2026_08_03.md](DOCUMENTATION_VALIDATION_POST_MERGE_2026_08_03.md).
**Freshness triggers:** new Current guide; ADR accept/supersede; package or
entrypoint renames; top-level navigation changes; packaging version
reconciliation; new doc checkers under `scripts/docs/`.
**Interface:** DocumentationBaseline@2
**Supersedes:** pre-DOC-028 short inventory that lacked status labels and a
reproducible closeout receipt
**Superseded-by:** none

This page records which documents are normative for the checked-out code and
how to audit documentation drift. The source code and executable help remain
authoritative for behavior. Full classification inventory:
[DOCUMENTATION_MANIFEST.md](DOCUMENTATION_MANIFEST.md).

The DOC-027 manifest and architecture hub froze this file as a pre-DOC-028
**Reference** snapshot pending revalidation. This page plus the exact closeout
receipt complete that revalidation and supersede only those two pending
markers; their remaining inventory and routing classifications are unchanged.

---

## Sources of truth

| Surface | Source of truth | Documentation route |
| --- | --- | --- |
| Package metadata and extras | `pyproject.toml`, `setup.py` | [Installation](../guides/getting-started/installation.md) |
| Python exports | `ipfs_accelerate_py/__init__.py`, `ipfs_accelerate_py/ipfs_accelerate.py` | [API overview](../api/overview.md) |
| Unified CLI (`ipfs-accelerate`) | `ipfs_accelerate_py/cli.py`, `cli_entry.py` | [CLI guide](../guides/cli/README_CLI.md) |
| Direct AI CLI (`ipfs_accelerate`) | `ipfs_accelerate_py/ai_inference_cli.py` | Inspect `ipfs_accelerate --help` (separate parser) |
| Canonical MCP runtime | `ipfs_accelerate_py/mcp_server/` | [MCP setup](../guides/MCP_SETUP_GUIDE.md), [MCP runtime](../architecture/MCP_RUNTIME.md) |
| MCP compatibility facade | `ipfs_accelerate_py/mcp/` | Labelled in MCP guides; not the primary operator entry |
| HF model server | `ipfs_accelerate_py/hf_model_server/` | [HF model server](../features/hf-model-server/README.md) (**Reference**; revalidated 2026-08-03) |
| Model catalog / routing | `model_catalog/`, `endpoint_usage/`, routers | [MODEL_SERVICE_ROUTING.md](../architecture/MODEL_SERVICE_ROUTING.md) |
| Agent supervisor | `ipfs_accelerate_py/agent_supervisor/` | [Supervisor guide](../guides/AGENT_SUPERVISOR_GUIDE.md), [hub](../architecture/agent_supervisor/README.md) |
| IPFS / P2P | `ipfs_backend_router.py`, `p2p_tasks/`, workflow modules | [DISTRIBUTED_RUNTIME.md](../architecture/DISTRIBUTED_RUNTIME.md), [P2P guide](../guides/p2p/README.md) |
| Test contracts | `test/`, `test/api/`, optional integration suites | [Testing](testing.md) |
| Doc status / authority | lifecycle policy + live tree | [DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md) |

Authority order (summary): package metadata and live code → executable help →
tests → **Current** docs → **Reference** docs → **Plan** → **Historical** /
**Generated** / **Vendored**. Full matrix:
[DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md).

The hyphenated `ipfs-accelerate` command and underscore `ipfs_accelerate`
command are separate installed scripts. Their parsers and capabilities are not
interchangeable. Use the command's own `--help` output when working with either
entry point.

---

## Maintained navigation and closeout evidence

Canonical entrypoints:

- [Root README](../../README.md) — project entrypoint
- [Documentation index](../INDEX.md) — **Current** topic and audience index
- [Documentation orientation](../README.md) — **Current** choose-a-path page
- [Architecture hub](../architecture/README.md) — **Reference** concern router
- [Documentation manifest](DOCUMENTATION_MANIFEST.md) — **Current** status inventory
- [Post-merge validation](DOCUMENTATION_VALIDATION_POST_MERGE_2026_08_03.md) — latest **Reference** integrated receipt
- [DOC-028 validation closeout](DOCUMENTATION_VALIDATION_2026_08.md) — historical **Reference** offline receipt

These pages should describe the current package boundaries, supported commands,
optional dependency behavior, and reproducible validation commands. When a
maintained page and a historical report disagree, the maintained page and live
code win.

### Current product and operator surfaces

| Area | Paths |
| --- | --- |
| Install / first run | [getting-started/README.md](../guides/getting-started/README.md), [installation.md](../guides/getting-started/installation.md), [QUICKSTART.md](../guides/QUICKSTART.md) |
| API / CLI | [api/overview.md](../api/overview.md), [guides/cli/README_CLI.md](../guides/cli/README_CLI.md) |
| MCP | [MCP_SETUP_GUIDE.md](../guides/MCP_SETUP_GUIDE.md), [MCP_SERVER.md](../MCP_SERVER.md), [MCP_RUNTIME.md](../architecture/MCP_RUNTIME.md) |
| Architecture | [overview](../architecture/overview.md), [SYSTEM_CONTEXT](../architecture/SYSTEM_CONTEXT.md), [INFERENCE_RUNTIME](../architecture/INFERENCE_RUNTIME.md), [MODEL_SERVICE_ROUTING](../architecture/MODEL_SERVICE_ROUTING.md), [DISTRIBUTED_RUNTIME](../architecture/DISTRIBUTED_RUNTIME.md), [INTEGRATION_BOUNDARIES](../architecture/INTEGRATION_BOUNDARIES.md) |
| Supervisor | [AGENT_SUPERVISOR_GUIDE.md](../guides/AGENT_SUPERVISOR_GUIDE.md), [agent_supervisor/README.md](../architecture/agent_supervisor/README.md), [CONTROL_PLANE](../architecture/agent_supervisor/CONTROL_PLANE.md), [PLANNING_AND_ASSURANCE](../architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md), [EXECUTION_AND_RECOVERY](../architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md), [PROMPT_FIRST_RUNTIME](../architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md), [DEVELOPER_GUIDE](../architecture/agent_supervisor/DEVELOPER_GUIDE.md), [FOR_AGENTS](../architecture/agent_supervisor/FOR_AGENTS.md) |
| Ops journeys | [deployment](../guides/deployment/README.md), [hardware](../guides/hardware/overview.md), [p2p](../guides/p2p/README.md), [troubleshooting](../guides/troubleshooting/faq.md), [testing](testing.md) |
| Governance | [DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md), [DOCUMENTATION_MAINTENANCE.md](DOCUMENTATION_MAINTENANCE.md), [GUIDE_CONVENTIONS.md](../architecture/GUIDE_CONVENTIONS.md) |

### Reference surfaces (partial authority)

| Path | Role |
| --- | --- |
| [GLOSSARY.md](../architecture/GLOSSARY.md) | Shared product vocabulary |
| [AGENT_SUPERVISOR_PHILOSOPHY.md](../architecture/AGENT_SUPERVISOR_PHILOSOPHY.md), [PACKAGE_MAP.md](../architecture/agent_supervisor/PACKAGE_MAP.md), [PROGRAMS.md](../architecture/agent_supervisor/PROGRAMS.md) | Supervisor orientation |
| [AGENT_SUPERVISOR_ARCHITECTURE.md](../architecture/AGENT_SUPERVISOR_ARCHITECTURE.md) | Deep architecture map |
| [AI_SERVICE_CATALOG.md](../architecture/AI_SERVICE_CATALOG.md) | Catalog identity plane; schema/precedence revalidated 2026-08-03 |
| [decisions/](../architecture/decisions/README.md) | Accepted ADRs (*why*, not sole *what*) |
| [NESTED_PACKAGES.md](../NESTED_PACKAGES.md) | Gitlink inventory |

### Plan and Historical (non-normative)

| Class | Examples | Rule |
| --- | --- | --- |
| **Plan** | `*_PLAN*.md`, `*.objectives.md`, `*.todo.md`, documentation refresh program inputs | Sequencing only; never Current API |
| **Historical** | `docs/archive/`, `docs/development_history/`, `docs/summaries/`, `docs/project/status/`, completion summaries, [DOCUMENTATION_DRIFT_AUDIT_2026_08.md](DOCUMENTATION_DRIFT_AUDIT_2026_08.md) | Context only; label when linked from navigation |

Operator-protected program inputs (read-only for implementation agents):

- `docs/architecture/documentation_refresh.todo.md`
- `docs/architecture/documentation_refresh.objectives.md`
- `docs/architecture/DOCUMENTATION_REFRESH_PLAN_2026_08.md`

---

## Code-owned blockers (do not paper over in prose)

These disagreements live in **code or packaging**, not documentation. Current
pages must **name both sides** until a code/test task reconciles them.

| Blocker | Evidence | Owner |
| --- | --- | --- |
| Optional capability stacks | CUDA, IPFS, P2P, external LLMs, provers may be absent after a successful import | subsystem maintainers; docs use capability language |
| Nested product gitlinks | Submodule / gitlink trees may be empty offline | integration maintainers; [NESTED_PACKAGES.md](../NESTED_PACKAGES.md) |
| (none currently for docs CI wiring) | Branch protection on `main` requires status check **documentation-gates** (`enforce_admins` off) | documentation governance |

**Resolved (2026-08-03 / 2026-08-04 revalidation):**

- Unified CLI module docstring and parser epilog no longer advertise
  unregistered `inference` / `queue` / `network` examples. Historical docs and
  completion reports may still mention those strings; the live `choices=` set
  rejects them.
- Package version pin is `0.0.45` in `pyproject.toml` / `setup.py`, and
  `ipfs_accelerate_py.__version__` uses the same `_PACKAGING_VERSION` pin.
- Allowlisted offline relative-link checker for maintained surfaces:
  `python scripts/docs/check_current_docs_links.py`
- GitHub Actions workflow
  [documentation-gates.yml](../../.github/workflows/documentation-gates.yml)
  runs vocabulary, link, version-pin, and closeout-marker gates on every PR
  and push to `main`; branch protection requires status **documentation-gates**

The second console script, `ipfs_accelerate`, intentionally uses the separate
`ai_inference_cli:main` parser. That split is a documented interface boundary.
Documentation tasks must not merge the two CLI surfaces in prose.

---

## Review checklist

Before merging a documentation change:

1. Confirm imports and public names in the live module.
2. Confirm CLI flags with `--help` for the **named** entry point; do not infer
   a command from an old example.
3. Confirm package extras and console scripts in `pyproject.toml`.
4. Confirm relative links from the document's actual directory.
5. Use capability language for optional hardware, providers, IPFS, P2P, and
   formal provers.
6. Avoid fixed model counts, benchmark numbers, or test totals without a date,
   commit, hardware, and reproducible command.
7. Label **Plan** / **Historical** destinations when linking them from Current
   navigation.
8. Run the smallest relevant deterministic test and the offline gates below.
9. Complete the non-suppressing checklist in
   [DOCUMENTATION_MAINTENANCE.md](DOCUMENTATION_MAINTENANCE.md).

### Offline-safe local checks

```bash
# Diff hygiene
git diff --check

# Primary agent-supervisor docs: no board-prefix ticket leakage
python scripts/docs/check_agent_supervisor_docs.py

# Maintained-surface relative links (allowlisted; not full-tree)
python scripts/docs/check_current_docs_links.py

# Same offline gates as .github/workflows/documentation-gates.yml (local)
# (vocabulary + links above; version pin via test_package_version_alignment)

# Navigation baseline markers (closeout contract)
rg -q 'Documentation baseline' docs/INDEX.md
rg -q 'Last verified' docs/development/DOCUMENTATION_CURRENT_STATE.md
test -f docs/development/DOCUMENTATION_VALIDATION_2026_08.md

# Packaging / entrypoint sanity when install or CLI claims change
rg -n '\[project\.scripts\]|^\[project\.optional-dependencies\]' pyproject.toml

# Correct unified-CLI test path (not under test/api/)
test -f test/test_unified_cli_integration.py
test ! -e test/api/test_unified_cli_integration.py

# Optional behavioral smoke when CLI/API claims change
python -m pytest test/test_unified_cli_integration.py -q

# Public agent-supervisor surface when supervisor navigation changes
python -m pytest \
  test/api/test_agent_supervisor_v2_public_api.py \
  test/api/test_agent_supervisor_semantic_layout_exports.py \
  test/api/test_agent_supervisor_entrypoint_package.py -q
```

Current integrated command matrix and measured results:
[DOCUMENTATION_VALIDATION_POST_MERGE_2026_08_03.md](DOCUMENTATION_VALIDATION_POST_MERGE_2026_08_03.md).
The original DOC-028 closeout remains preserved in
[DOCUMENTATION_VALIDATION_2026_08.md](DOCUMENTATION_VALIDATION_2026_08.md).

---

## Known optionality

The base package is deliberately defensive. CUDA, Transformers, MCP, IPFS,
libp2p, browser runtimes, external LLMs, and theorem provers may be absent or
unhealthy even when `import ipfs_accelerate_py` succeeds. Documentation should
show how to discover and report that state instead of claiming universal
availability. Import success is **not** a capability signal.

---

## Next audit triggers

Re-run the offline closeout (or a successor validation receipt) when any of
the following occur:

1. A Current guide or top-level navigation page (`docs/README.md`,
   `docs/INDEX.md`) gains or loses a primary entrypoint.
2. Package scripts, extras, or public exports change in `pyproject.toml` or
   package `__init__` files.
3. Supervisor domain packages move or the primary-doc ticket-ID checker path
   list changes.
4. The packaging version pin changes in `pyproject.toml` / `setup.py` (keep
   `_PACKAGING_VERSION` and Current install pages aligned).
5. The allowlisted link checker path set grows, or branch protection marks
   **documentation-gates** required on `main`.
6. Nested product submodules become required for a Current journey.

---

## Related documents

| Document | Role |
| --- | --- |
| [DOCUMENTATION_MANIFEST.md](DOCUMENTATION_MANIFEST.md) | Full status inventory |
| [DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md) | Status vocabulary and authority |
| [DOCUMENTATION_MAINTENANCE.md](DOCUMENTATION_MAINTENANCE.md) | PR review without suppressing drift |
| [DOCUMENTATION_VALIDATION_POST_MERGE_2026_08_03.md](DOCUMENTATION_VALIDATION_POST_MERGE_2026_08_03.md) | Latest integrated validation receipt |
| [DOCUMENTATION_VALIDATION_2026_08.md](DOCUMENTATION_VALIDATION_2026_08.md) | Historical DOC-028 closeout receipt |
| [DOCUMENTATION_DRIFT_AUDIT_2026_08.md](DOCUMENTATION_DRIFT_AUDIT_2026_08.md) | Frozen Wave-0 drift inventory (**Historical**) |
| [testing.md](testing.md) | Test selection guide |
| [docs/INDEX.md](../INDEX.md) | Canonical navigation |
| [architecture/README.md](../architecture/README.md) | Architecture audience router |
