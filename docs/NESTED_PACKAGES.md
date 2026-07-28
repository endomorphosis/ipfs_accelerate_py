# Nested packages and product trees

This monorepo (`ipfs_accelerate_py`) keeps several nested product trees and git
submodules next to the installable Python package. They are intentional
checkouts or sibling products, **not** disposable clutter.

**Do not delete, force-reinit, or mass-rewrite these trees without ownership
review.** Root-hygiene work (ASREF-006 / **ASREF-G090** cutover) may document,
ignore ephemera, or relocate *misplaced root plans/tests* only. It must not
move `agent_supervisor` packages into nested product trees or rewrite submodule
history as part of package-layout moves.

Cutover packet: `goal_packet/cutover/ipfs_accelerate_py/090ea2138c6f`
(tasks **ASREF-012**, **ASREF-013**, **ASREF-014**; goal **ASREF-G090**).
Package layout evidence for parent cutover remains **ASREF-G020**,
**ASREF-G030**, **ASREF-G040**, **ASREF-G050**, **ASREF-G060**, **ASREF-G070**,
and **ASREF-G080** under `ipfs_accelerate_py/agent_supervisor/` — not under
nested product trees.

**ASREF-013** closes the evidence cluster for **ASREF-G020** (core),
**ASREF-G030** (control), **ASREF-G040** (task_sources), and **ASREF-G050**
(context/prompt) in the package README and root `__init__.py` layout
constants (`AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS`,
`AGENT_SUPERVISOR_CORE_STEMS`, `AGENT_SUPERVISOR_CONTROL_STEMS`,
`AGENT_SUPERVISOR_TASK_SOURCES_STEMS`,
`AGENT_SUPERVISOR_CONTEXT_PROMPT_PLANNED_MODULES`). Those packages never relocate
into nested product trees.

**ASREF-014** closes the evidence cluster for **ASREF-G060** (analysis/proof),
**ASREF-G070** (objectives/planning/validation/merge/rescue/runtime/
self_improvement), and **ASREF-G080** (todo_daemon/integrations) in the
package README and root `__init__.py` layout constants. Those packages never
relocate into nested product trees.

| Document role | Path |
| --- | --- |
| This inventory | `docs/NESTED_PACKAGES.md` |
| Process-junk ignore rules | `.gitignore` (ASREF-006 / ASREF-G090 section) |
| Agent supervisor public API map | `ipfs_accelerate_py/agent_supervisor/README.md` |
| MCP unification plan (canonical) | `docs/architecture/MCP_SERVER_UNIFICATION_PLAN.md` |
| Module refactor program plan | `docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md` |

## Ownership policy

| Rule | Detail |
| --- | --- |
| No silent deletion | Nested checkouts and submodules stay until an explicit ownership decision |
| No agent_supervisor moves into nested products | Package layout work stays under `ipfs_accelerate_py/agent_supervisor/` |
| Package goals stay in-tree | **ASREF-G020**–**ASREF-G080** packages (`core`, `control`, `task_sources`, `context`, `analysis`, `proof`, `objectives`, `planning`, `prompt`, `validation`, `merge`, `rescue`, `runtime`, `self_improvement`, `todo_daemon`, `integrations`) never relocate into nested products |
| Submodule pin changes | Require review; prefer recording intended branch in `.gitmodules` |
| Empty checkout dirs | Common when submodules are not initialized; still reserved paths |
| Docs vs products | Architecture and operator docs live under `docs/`; nested trees keep their own READMEs |
| Import surface | Root hygiene and nested-package docs must not break `ipfs_accelerate_py.agent_supervisor` imports |
| Cutover owns final gate | **ASREF-G090** owns public API map, no-old-import gate, and root hygiene docs; child package goals own physical module moves |

## Nested product trees (first-party / sibling)

| Path | Role | Notes |
| --- | --- | --- |
| `ipfs_accelerate_py/` | Installable primary package | Canonical Python library, MCP runtime, agent supervisor |
| `ipfs_accelerate_js/` | Browser / JS accelerate surface | Sibling product tree; not a git submodule in this layout |
| `ipfs_datasets_py/` | Datasets / GraphRAG / MCP source parity | Git submodule (`endomorphosis/ipfs_datasets_py`) |
| `ipfs_kit_py/` | IPFS kit integration surface | Git submodule (`endomorphosis/ipfs_kit_py`) |
| `ipfs_model_manager_py/` | Model manager product tree | Git submodule (`endomorphosis/ipfs_model_manager_py`) |
| `ipfs_transformers_py/` | Transformers + IPFS helpers | Git submodule (`endomorphosis/ipfs_transformers_py`) |
| `mcpplusplus/` | MCP++ conformance artifacts at repo root | Planning/conformance docs; related to `ipfs_accelerate_py/mcplusplus` |

Empty directories at the nested submodule paths are normal when
`git submodule update --init` has not been run. Treat them as reserved product
slots, not cleanup candidates.

## Git submodules (`.gitmodules`)

| Path | Remote (summary) | Typical use |
| --- | --- | --- |
| `ipfs_datasets_py` | `endomorphosis/ipfs_datasets_py` | Source parity and datasets tooling |
| `ipfs_kit_py` | `endomorphosis/ipfs_kit_py` | Kit / IPFS operations |
| `ipfs_model_manager_py` | `endomorphosis/ipfs_model_manager_py` | Model catalog / manager |
| `ipfs_transformers_py` | `endomorphosis/ipfs_transformers_py` | Transformers integration |
| `ipfs_accelerate_py/mcplusplus` | `endomorphosis/Mcp-Plus-Plus` | MCP++ spec and checklist source |
| `docs/fastmcp` | `jlowin/fastmcp` | Upstream FastMCP reference docs |
| `docs/mcp-python-sdk` | `jlowin/mcp-python-sdk` | Upstream MCP Python SDK reference |
| `test/huggingface_transformers` | `huggingface/transformers` | Upstream transformers for tests |
| `test/doc-builder` | `huggingface/doc-builder` | Doc build tooling |
| `test/huggingface_doc_builder` | `huggingface/doc-builder` | Alternate doc-builder pin path |

Initialize when needed:

```bash
git submodule update --init --recursive
# or a single path:
git submodule update --init ipfs_kit_py
```

## Related monorepo paths (not nested products)

| Path | Role |
| --- | --- |
| `docs/` | Operator and architecture documentation (this file lives here) |
| `test/` | Canonical test tree — prefer for all new tests |
| `tests/` | Legacy or secondary test path; prefer `test/` for new work |
| `scripts/`, `deployments/`, `install/`, `examples/`, `config/` | Operational and packaging support |
| `data/` | Mixed fixtures and local runtime data; many entries gitignored |
| `state/` | Local/runtime state; broadly gitignored via `state/*` |
| `ipfs_accelerate_py/agent_supervisor/` | ASREF package layout (**ASREF-G020**–**ASREF-G080**); public map in package README |

## Root hygiene (ASREF-006 / ASREF-G090)

### Process junk (ignored, not nested products)

Ephemeral process files at the monorepo root must not be treated as product
artifacts. `.gitignore` excludes at least:

| Pattern / path | Why ignored |
| --- | --- |
| `dashboard.out`, `dashboard.pid` | Dashboard process stdout / PID |
| `err.txt` | Ad-hoc error capture |
| `*.pid`, `nohup.out`, `/*.out` | Generic process / nohup residue |
| `.DS_Store`, `Thumbs.db`, `*.swp`, `*~` | OS / editor junk |
| `data/*.db`, `data/**/*.duckdb.wal` | Local DB / WAL noise under `data/` |

**Note:** Paths that were committed before the ignore rules still appear in
`git ls-files` until an operator (or a later cutover task with broader path
scope) runs `git rm --cached` on them. Ignore rules prevent *new* tracking.
Cutover acceptance for **ASREF-G090** requires these process files to be
untracked on the merge-ready branch; when this task's edit scope is limited to
`.gitignore` and docs, the ignore rules land here and untracking is the
operator follow-up.

Suggested untrack (outside narrow cutover path scope when needed):

```bash
git rm --cached --ignore-unmatch dashboard.out dashboard.pid err.txt
```

### Misplaced root plans and one-offs

| Legacy / root path | Canonical home or disposition |
| --- | --- |
| `MCP_SERVER_UNIFICATION_PLAN.md` | `docs/architecture/MCP_SERVER_UNIFICATION_PLAN.md` (canonical) |
| `SDK_PLAYGROUND_PREVIEW.html` | Prefer `docs/` or `docs/exports/` in a later hygiene/cutover step |
| Root `test_*.py` (e.g. `test_dashboard_sdk.py`, `test_mcp_jsonrpc_conformance.py`) | Prefer `test/` in a hygiene or cutover task; not nested-product tests |

ASREF-006 **documents** and **ignores** within its allowed path set; it does
not delete nested checkouts, untrack every historical file, or move tests when
those paths are outside the task edit scope. **ASREF-G090** cutover still
documents the preferred end state: no root process junk tracked, root tests
under `test/`, nested trees inventoried.

### What root hygiene must not do

- Delete nested product checkouts or submodules without ownership review
- Move `agent_supervisor` packages into nested product trees
- Rewrite submodule history as part of package-layout moves
- Break `ipfs_accelerate_py.agent_supervisor` import paths (no package moves in hygiene-only tasks)
- Relocate **ASREF-G020**–**ASREF-G080** domain packages outside
  `ipfs_accelerate_py/agent_supervisor/`

## Agent supervisor refactor boundary

The agent-supervisor module refactor (`ASREF-*`) may:

- document nested packages (this file);
- ignore or remove root process junk via `.gitignore` (and later untrack steps);
- relocate root-level plans into `docs/` (MCP unification plan lives under
  `docs/architecture/`);
- publish the package public API map and cutover evidence for **ASREF-G090**
  in `ipfs_accelerate_py/agent_supervisor/README.md` and `__init__.py`.

It must **not**:

- delete nested product checkouts without ownership review;
- move `agent_supervisor` packages into nested product trees;
- rewrite submodule history as part of package-layout moves.

### Package goals vs nested products

| Goal | Packages (under `agent_supervisor/`) | Nested-product interaction |
| --- | --- | --- |
| **ASREF-G020** | `core/` (landed) | none — stay in primary package |
| **ASREF-G030** | `control/` (landed) | none |
| **ASREF-G040** | `task_sources/` (landed) | none |
| **ASREF-G050** | `context/`, `prompt/` (planned; flat modules until move) | none |
| **ASREF-G060** | `analysis/`, `proof/` (planned dirs; flat modules until move) | none — analysis/proof modules stay in primary package |
| **ASREF-G070** | `objectives/`, `planning/`, `validation/`, `merge/`, `rescue/`, `runtime/`, `self_improvement/` (landed scaffolds) | none |
| **ASREF-G080** | `todo_daemon/` (package-native), `integrations/` (planned) | none — external tool bridges stay in primary package |
| **ASREF-G090** | public API, root hygiene, cutover | documents nested trees; does not absorb them |

ASREF-013 cutover witness for **ASREF-G020** / **ASREF-G030** / **ASREF-G040** /
**ASREF-G050** lives in `ipfs_accelerate_py/agent_supervisor/README.md` and
`AGENT_SUPERVISOR_FOUNDATION_LAYOUT_GOAL_IDS` /
`AGENT_SUPERVISOR_CORE_STEMS` /
`AGENT_SUPERVISOR_CONTROL_STEMS` /
`AGENT_SUPERVISOR_TASK_SOURCES_STEMS` /
`AGENT_SUPERVISOR_CONTEXT_PROMPT_PLANNED_MODULES` in the package `__init__.py`.

ASREF-014 cutover witness for **ASREF-G060** / **ASREF-G070** / **ASREF-G080**
lives in `ipfs_accelerate_py/agent_supervisor/README.md` and
`AGENT_SUPERVISOR_OPERATIONS_LAYOUT_GOAL_IDS` /
`AGENT_SUPERVISOR_PLANNED_MODULE_OWNERS` in the package `__init__.py`.

## See also

- [Agent Supervisor Module Refactor Plan](architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md) — root hygiene targets
- [Agent supervisor package README](../ipfs_accelerate_py/agent_supervisor/README.md) — public API and package-goal evidence
- [MCP Server Unification Master Plan](architecture/MCP_SERVER_UNIFICATION_PLAN.md) — canonical MCP runtime plan
- [Documentation index](INDEX.md)
- Root [README.md](../README.md) and `.gitmodules`
