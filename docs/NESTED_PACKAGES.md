# Nested packages and product trees

This monorepo (`ipfs_accelerate_py`) keeps several nested product trees and git
submodules next to the installable Python package. They are intentional
checkouts or sibling products, **not** disposable clutter.

**Do not delete, force-reinit, or mass-rewrite these trees without ownership
review.** Root-hygiene work (ASREF-006 / ASREF-G090) may document, ignore
ephemera, or relocate *misplaced root plans/tests* only. It must not move
`agent_supervisor` packages into nested product trees or rewrite submodule
history as part of package-layout moves.

| Document role | Path |
| --- | --- |
| This inventory | `docs/NESTED_PACKAGES.md` |
| Process-junk ignore rules | `.gitignore` (ASREF-006 section) |
| MCP unification plan (canonical) | `docs/architecture/MCP_SERVER_UNIFICATION_PLAN.md` |
| Module refactor program plan | `docs/architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md` |

## Ownership policy

| Rule | Detail |
| --- | --- |
| No silent deletion | Nested checkouts and submodules stay until an explicit ownership decision |
| No agent_supervisor moves into nested products | Package layout work stays under `ipfs_accelerate_py/agent_supervisor/` |
| Submodule pin changes | Require review; prefer recording intended branch in `.gitmodules` |
| Empty checkout dirs | Common when submodules are not initialized; still reserved paths |
| Docs vs products | Architecture and operator docs live under `docs/`; nested trees keep their own READMEs |
| Import surface | Root hygiene and nested-package docs must not break `ipfs_accelerate_py.agent_supervisor` imports |

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

## Root hygiene (ASREF-006)

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

### Misplaced root plans and one-offs

| Legacy / root path | Canonical home or disposition |
| --- | --- |
| `MCP_SERVER_UNIFICATION_PLAN.md` | `docs/architecture/MCP_SERVER_UNIFICATION_PLAN.md` (canonical) |
| `SDK_PLAYGROUND_PREVIEW.html` | Prefer `docs/` or `docs/exports/` in a later hygiene/cutover step |
| Root `test_*.py` (e.g. `test_dashboard_sdk.py`, `test_mcp_jsonrpc_conformance.py`) | Prefer `test/` in a hygiene or cutover task; not nested-product tests |

ASREF-006 **documents** and **ignores** within its allowed path set; it does
not delete nested checkouts, untrack every historical file, or move tests when
those paths are outside the task edit scope.

### What ASREF-006 must not do

- Delete nested product checkouts or submodules without ownership review
- Move `agent_supervisor` packages into nested product trees
- Rewrite submodule history as part of package-layout moves
- Break `ipfs_accelerate_py.agent_supervisor` import paths (no package moves in this task)

## Agent supervisor refactor boundary

The agent-supervisor module refactor (`ASREF-*`) may:

- document nested packages (this file);
- ignore or remove root process junk via `.gitignore` (and later untrack steps);
- relocate root-level plans into `docs/` (MCP unification plan lives under
  `docs/architecture/`).

It must **not**:

- delete nested product checkouts without ownership review;
- move `agent_supervisor` packages into nested product trees;
- rewrite submodule history as part of package-layout moves.

## See also

- [Agent Supervisor Module Refactor Plan](architecture/AGENT_SUPERVISOR_MODULE_REFACTOR_PLAN.md) — root hygiene targets
- [MCP Server Unification Master Plan](architecture/MCP_SERVER_UNIFICATION_PLAN.md) — canonical MCP runtime plan
- [Documentation index](INDEX.md)
- Root [README.md](../README.md) and `.gitmodules`
