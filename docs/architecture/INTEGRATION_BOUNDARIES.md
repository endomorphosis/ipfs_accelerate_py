# Cross-repository and nested-package integration boundaries

**Status:** Current
**Audience:** Integrators, maintainers, security reviewers, and implementation
agents placing code or evidence across sibling products
**Scope:** Ownership, dependency direction, gitlink pins, discovery, capability
probes, graceful fallback, fail-closed assurance, and independent Git authority
for `ipfs_accelerate_py`, sibling nested packages (`ipfs_datasets_py`,
`ipfs_kit_py`, `ipfs_model_manager_py`, `ipfs_transformers_py`), MCP++ surfaces,
and related reference checkouts
**Non-goals:** Backend CID/P2P runtime semantics (see
[distributed runtime](DISTRIBUTED_RUNTIME.md) / existing IPFS guides); MCP transport policy and tool
registry (see [MCP runtime](MCP_RUNTIME.md)); agent-supervisor intent/control plane beyond
repository-forest and optional-provider boundaries; installation UX copy
(DOC-021); initialization or mutation of any submodule gitlink
**Last verified:** `e559ff0046c639ba1dadabe02ea0ea91d9877e20` (2026-08-03);
`.gitmodules`, adapter modules, forest/manifest code, and focused integration
tests inspected in this worktree (nested product directories present as empty
gitlink slots unless initialized)

---

## Source anchors

| Concern | Primary path / symbol | Notes |
| --- | --- | --- |
| Gitlink inventory | `.gitmodules` | Declared submodule paths and remotes |
| Nested product policy | `docs/NESTED_PACKAGES.md` | Inventory and hygiene rules (not authority) |
| Package path discovery | `ipfs_accelerate_py/__init__.py` (`_add_external_package`) | Optional `external/<pkg>` prepend only |
| Kit integration | `ipfs_accelerate_py/ipfs_kit_integration.py` (`IPFSKitStorage`, `_resolve_ipfs_kit_source_path`) | Compatibility adapter with explicit `using_fallback`; skips empty gitlinks; local writes return synthetic CID-like keys |
| Backend roles | `ipfs_accelerate_py/ipfs_backend_router.py` (`BackendRole`, `ENABLE_IPFS_KIT`, `IPFS_KIT_DISABLE`) | Preferred `ipfs_kit_py` when enabled and available |
| Kit fallback cache | `ipfs_accelerate_py/common/ipfs_kit_fallback.py` (`IPFSKitFallbackStore`) | Best-effort CID retrieval |
| In-tree kit modules | `ipfs_accelerate_py/kit/` | **Not** the external `ipfs_kit_py` package |
| Datasets integration | `ipfs_accelerate_py/datasets_integration/` (`is_datasets_available`, `_resolve_datasets_source_path`) | Env `IPFS_DATASETS_ENABLED`, `IPFS_DATASETS_PATH` |
| Router deps | `ipfs_accelerate_py/router_deps.py` (`RouterDeps.get_accelerate_manager`) | Optional `ipfs_datasets_py.ml.accelerate_integration` |
| Forest authority | `ipfs_accelerate_py/agent_supervisor/repository_forest.py` (`RepositoryAuthority`, `RepositoryForest`, `AuthorityMode`) | Co-location ≠ shared Git authority |
| Forest manifests | `ipfs_accelerate_py/agent_supervisor/repository_forest_manifest.py` | Sole-write alias; observed commits are observational |
| Corpus inventory | `ipfs_accelerate_py/agent_supervisor/repository_corpus_index.py` | Git-object-first exhaustive index |
| Optional analysis provider | `ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_analysis_provider.py` | Nominates evidence; never completion-authoritative |
| Optional logic/proof provider | `ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py` | Hammer portfolio; ATP/SMT untrusted for kernel assurance |
| Analysis pipeline | `ipfs_accelerate_py/agent_supervisor/analysis/analysis_pipeline.py` | Advisory claims vs exact-tree evidence |
| MCP++ in-tree module | `ipfs_accelerate_py/mcplusplus_module/` | Runtime Trio/P2P implementation under accelerate |
| MCP++ server surfaces | `ipfs_accelerate_py/mcp_server/mcplusplus/` | Canonical MCP server MCP++ features |
| MCP++ git submodule | `ipfs_accelerate_py/mcplusplus` (gitlink) | Spec/source pin; often empty without `submodule update` |
| Root MCP++ artifacts | `mcpplusplus/` | Conformance/planning docs in this monorepo (not the gitlink) |
| Transformers path | `ipfs_accelerate_py/transformers_integration.py`, `auto_patch_transformers` | Prefer in-tree patch; sibling `ipfs_transformers_py` optional/legacy |
| Primary package | `ipfs_accelerate_py/` | Installable library; owns adapters and kit modules |

### Focused tests (current paths)

| Claim family | Test path |
| --- | --- |
| Datasets discovery / empty gitlink | `test/test_datasets_integration.py` |
| Kit storage / empty gitlink | `test/test_ipfs_kit_integration.py` |
| Repository forest authority | `test/api/test_agent_supervisor_repository_forest.py` |
| Forest manifest replay | `test/api/test_agent_supervisor_repository_forest_manifest.py` |
| Corpus index | `test/api/test_agent_supervisor_repository_corpus_index.py` |
| Optional datasets analysis (non-authoritative) | `test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py` |
| Analysis pipeline advisory fallback | `test/api/test_agent_supervisor_analysis_pipeline.py` |
| Datasets logic provider | `test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py` |
| MCP++ interop smoke | `test/integration/test_mcp_mcplusplus_interop_smoke.py` |

---

## 1. Context and component map

```text
                    Independent remotes / Git histories
         ┌──────────────────┬──────────────────┬──────────────────┐
         │  ipfs_kit_py     │ ipfs_datasets_py │  MCP++ / others  │
         │  (submodule pin) │ (submodule pin)  │  (submodule pin) │
         └────────┬─────────┴────────┬─────────┴────────┬─────────┘
                  │ optional import  │ optional import  │ pin / docs
                  ▼                  ▼                  ▼
         ┌────────────────────────────────────────────────────────┐
         │              ipfs_accelerate_py (this repo)            │
         │  adapters: ipfs_kit_integration, datasets_integration  │
         │  routers:  ipfs_backend_router, router_deps            │
         │  kit/*     (in-tree modules; not external kit package) │
         │  mcp_server + mcplusplus_module (runtime)              │
         │  agent_supervisor: forest, corpus, integrations/*      │
         └────────────────────────────────────────────────────────┘
                  │
                  │ sole write authority when this root is sole_write_alias
                  ▼
         Host checkout / worktree (local locator; not portable identity)
```

**Rule of co-location:** Paths such as `./ipfs_kit_py` and `./ipfs_datasets_py`
under the accelerator monorepo are **discovery slots and gitlink pins**, not a
shared Git repository and not shared mutation authority. Each nested product
keeps its own remote, commit, tree, dirty overlay, and (when used in a forest)
`RepositoryAuthority`. Sibling roots **never** inherit write rights from
sharing a parent directory.

| Container | Live package / path | Role |
| --- | --- | --- |
| Accelerator monorepo | this checkout | Primary product Git authority for accelerate code and docs |
| Nested product slots | `ipfs_kit_py/`, `ipfs_datasets_py/`, … | Reserved gitlink paths; empty until initialized |
| In-tree kit | `ipfs_accelerate_py/kit/` | Accelerate-owned CLI/MCP core modules |
| External kit | importable `ipfs_kit_py` | Optional distributed storage provider |
| Datasets product | importable `ipfs_datasets_py` | Optional datasets, GraphRAG, analysis, Hammer surfaces |
| MCP++ runtime | `mcplusplus_module/`, `mcp_server/mcplusplus/` | Accelerate-owned runtime surfaces |
| MCP++ pin | `ipfs_accelerate_py/mcplusplus` gitlink | Spec/source pin for Mcp-Plus-Plus |
| Root MCP++ docs | `mcpplusplus/` | Conformance matrices and plans in monorepo history |
| Reference docs | `docs/fastmcp`, `docs/mcp-python-sdk` | Upstream reference only |
| Test assets | `test/huggingface_*`, `test/doc-builder` | Optional test tooling pins |

---

## 2. Repository matrix (RepositoryAuthorityBoundary@1)

### 2.1 Declared gitlinks (GitlinkPin@1)

From `.gitmodules` (branch `main` unless noted). Pins are **observed** at the
parent commit as mode-`160000` gitlinks; recorded SHAs are not trusted as
identity without live Git re-derivation in the forest path.

| Path | Remote (summary) | Typical use | Mutation authority in accelerate work |
| --- | --- | --- | --- |
| `ipfs_kit_py` | `endomorphosis/ipfs_kit_py` | Distributed kit / IPFS operations | **Read-only** relative to accelerate tasks unless a forest grants write on that alias |
| `ipfs_datasets_py` | `endomorphosis/ipfs_datasets_py` | Datasets, GraphRAG, optional analysis/proof | **Read-only** for accelerate completion; optional provider only |
| `ipfs_model_manager_py` | `endomorphosis/ipfs_model_manager_py` | Model manager product tree | Separate product; not required for core inference |
| `ipfs_transformers_py` | `endomorphosis/ipfs_transformers_py` | Historical transformers helpers | Prefer in-tree `auto_patch_transformers` |
| `ipfs_accelerate_py/mcplusplus` | `endomorphosis/Mcp-Plus-Plus` | MCP++ specification pin | Spec pin; runtime lives under accelerate modules |
| `docs/fastmcp` | `jlowin/fastmcp` | Upstream FastMCP reference | Docs/reference only |
| `docs/mcp-python-sdk` | `jlowin/mcp-python-sdk` | Upstream MCP Python SDK reference | Docs/reference only |
| `test/huggingface_transformers` | `huggingface/transformers` | Test asset | Tests only |
| `test/doc-builder`, `test/huggingface_doc_builder` | `huggingface/doc-builder` | Doc build tooling | Tests/docs tooling only |

**Pinned SHAs observed at last verification** (parent index; not a promise that
worktrees are populated):

| Path | Gitlink SHA (parent index) |
| --- | --- |
| `ipfs_kit_py` | `f6a574375febbcf9a46fcd24bbc7bc5cfb551de5` |
| `ipfs_datasets_py` | `d144be65ffe4c6423e4e1c30cd692812607343eb` |
| `ipfs_model_manager_py` | `f6151d2113f42e75ea7d83a1b2362fc97e55e44d` |
| `ipfs_transformers_py` | `b397988ed9e3e656475c1cf4417b84efdb95daf3` |
| `ipfs_accelerate_py/mcplusplus` | `15c1816d6c63a2b11edd505704f6a04a9abc6167` |

### 2.2 Default forest aliases

`repository_forest.py` and the reviewed four-root manifest use:

| Alias | Default product | Typical authority in reviewed launch |
| --- | --- | --- |
| `swissknife` | External SwissKnife root | Read-only observation |
| `ipfs_accelerate_py` | This monorepo | **Sole write** in reviewed VFS/assurance forests |
| `ipfs_kit_py` | Kit checkout | Read-only |
| `ipfs_datasets_py` | Datasets checkout | Read-only |

`RepositoryAuthority.mode` is only `read_only` or `read_write`. Read-only
authority **must not** carry a write-path allowlist. Manifests require
**exactly one** sole-write alias with `read_write` authority.

Manifest fields that record historical commits are **observational only**.
Materialization always re-derives commit/tree from live Git
(`repository_forest_manifest.py`).

### 2.3 What co-location is not

| Observation | Does **not** imply |
| --- | --- |
| Directory exists under monorepo root | Submodule is initialized |
| Path appears in `.gitmodules` | Package is importable |
| Import succeeds | Capability, health, or proof |
| Shared parent path | Shared Git history or shared write authority |
| Optional analysis success | Completion or merge authority |
| Board / plan status | Current API or completed integration |

---

## 3. Dependency direction and import arrows

```text
Application / CLI / MCP / supervisor
        |
        v
ipfs_accelerate_py  ──(owns)──>  adapters, routers, kit/*, mcp_server,
        |                         mcplusplus_module, agent_supervisor
        |
        | optional, lazy, capability-gated
        +── import ──>  ipfs_kit_py          (storage / distributed FS)
        +── import ──>  ipfs_datasets_py     (datasets, ML accelerate mgr,
        |                                    analysis nomination, Hammer)
        +── import ──>  ipfs_model_manager_py / ipfs_transformers_py
        |               (optional product trees; transformers prefer in-tree)
        |
        x  never: sibling packages mutate accelerate Git or claim sole_write
        x  never: optional provider result upgrades to completion_authoritative
```

**Dependency rule:** Accelerate **consumes** optional sibling packages through
in-tree adapters. Sibling packages do not become part of the trusted completion
or merge boundary for accelerator tasks.

### 3.1 Discovery order (editable / umbrella)

Adapters resolve source checkouts before import when needed. Empty gitlink
directories are **skipped** so they cannot mask a valid sibling.

**`ipfs_kit_py`** (`_resolve_ipfs_kit_source_path`):

1. `<repo_root>/ipfs_kit_py` with nested `ipfs_kit_py/__init__.py`
2. `<repo_root>/external/ipfs_kit_py` with nested package
3. `<repo_root.parent>/ipfs_kit` (umbrella sibling name)
4. `<repo_root.parent>/ipfs_kit_py`

**`ipfs_datasets_py`** (`_resolve_datasets_source_path`):

1. `IPFS_DATASETS_PATH` if set and is a directory
2. `<repo_root>/ipfs_datasets_py`
3. `<repo_root>/external/ipfs_datasets_py`
4. `<repo_root.parent>/ipfs_datasets` (umbrella)
5. `<repo_root.parent>/ipfs_datasets_py`

Package root also offers `_add_external_package` for
`external/{ipfs_kit_py,ipfs_model_manager_py,ipfs_transformers_py}` when core
import is not skipped (`IPFS_ACCEL_SKIP_CORE=1` avoids heavy core import).

---

## 4. Optional integration capabilities (OptionalIntegrationCapability@1)

### 4.1 Capability language

| Signal | Means | Does not mean |
| --- | --- | --- |
| Discovery / path found | A candidate source or package layout exists | Backend works |
| Import success | Module loads in-process | Health, SLA, or proof |
| Capability probe | Configured path may **attempt** work | Completion authority |
| Typed degradation evidence | Explicit local fallback with bound witness | Successful optional op |
| Validation / focused tests | Claim family checked for a revision | Universal install guarantee |

**Import success is never a capability signal. Capability is never completion.**

### 4.2 `ipfs_kit_py` (storage / backend)

| Item | Detail |
| --- | --- |
| Adapter | `IPFSKitStorage` in `ipfs_kit_integration.py`; independently falls back to its local cache and exposes `using_fallback` |
| Router | `ipfs_backend_router.py`; separately selects an `IPFSKitBackend`, HF-cache, or Kubo role and emits `BackendSelectionReceipt` |
| Env | `ENABLE_IPFS_KIT` (default true), `IPFS_KIT_DISABLE`, `IPFS_BACKEND`, `IPFS_KIT_CACHE_DIR` |
| Happy path | Router preference is `ipfs_kit_py` when enabled and available; adapter health still requires inspection rather than import inference |
| Degradation | `IPFSKitStorage` can use its own local fallback (`using_fallback=true`); independently, the router can select HF cache or Kubo and records that selection |
| Fail-closed notes | The adapter's local `bafy…` keys and synthetic HF `bafy…` keys are not verified multiformats CIDs; codec/CAR claims require verification, not a CID-like prefix |
| Confusion to avoid | `ipfs_accelerate_py.kit` modules wrap accelerate concerns; they are not the external package |

### 4.3 `ipfs_datasets_py` (datasets and supervisor providers)

| Item | Detail |
| --- | --- |
| Runtime adapter | `datasets_integration/` (`DatasetsManager`, provenance, workflow, filesystem) |
| Env | `IPFS_DATASETS_ENABLED` (`auto` / enable / disable), `IPFS_DATASETS_PATH` |
| Router consumption | `RouterDeps.get_accelerate_manager` may load `ipfs_datasets_py.ml.accelerate_integration` |
| Supervisor analysis | `integrations/ipfs_datasets_analysis_provider.py` — closed operation vocabulary, bounds, lazy import |
| Supervisor logic | `integrations/ipfs_datasets_logic_provider.py` — Hammer portfolio; candidates remain untrusted for kernel proof |
| Happy path | Distributed dataset ops or advisory analysis when package is healthy |
| Degradation | Local fallbacks in datasets_integration; typed `AnalysisProviderResult` / degradation evidence for supervisor |
| **Completion rule** | Optional providers **nominate analysis/evidence** only. They **cannot manufacture completion**. Pipeline exposes qualifying optional IDs only via `advisory_evidence_claim_references`, separate from completion-authoritative exact-tree evidence |

Construction and capability declaration of the analysis provider **must not**
import `ipfs_datasets_py` (enforced in
`test_agent_supervisor_ipfs_datasets_analysis_provider.py`). Missing, disabled,
or unhealthy optional capabilities produce typed local-fallback results, not
silent success.

### 4.4 MCP++

| Layer | Path | Authority |
| --- | --- | --- |
| Spec / pin | gitlink `ipfs_accelerate_py/mcplusplus` | Independent remote; empty without init |
| Runtime module | `ipfs_accelerate_py.mcplusplus_module` | Accelerate-owned implementation |
| Server features | `ipfs_accelerate_py/mcp_server/mcplusplus/` | Canonical MCP server plane |
| Conformance docs | root `mcpplusplus/` | Monorepo-owned planning/conformance artifacts |

Presence of the empty gitlink directory does not imply MCP++ runtime is
unavailable: the installable package ships `mcplusplus_module` and
`mcp_server/mcplusplus` independently of submodule init.

### 4.5 Other nested products

| Product | Boundary |
| --- | --- |
| `ipfs_model_manager_py` | Optional product tree / git pin; model catalog and endpoint ownership remain accelerate surfaces (`model_catalog`, `endpoint_usage`) |
| `ipfs_transformers_py` | Historical; current path prefers `transformers_integration` + `auto_patch_transformers` |
| `ipfs_accelerate_js` | Sibling product tree in monorepo; not a git submodule in this layout |
| Upstream docs submodules | Reference only; never product API |

---

## 5. Control and data flows

### 5.1 Optional storage / datasets call

```text
Caller (inference, MCP tool, CLI)
        |
   adapter / router entry
        |
   env gate + source resolution
        |                      \
   import optional pkg          empty gitlink / missing import
        |                              |
   attempt operation            explicit fallback / role report
        |                              |
   success payload              local path / disabled feature
```

### 5.2 Supervisor optional analysis (advisory only)

```text
AnalysisPipeline
        |
   build bounded request (no heavy smuggling)
        |
   IpfsDatasetsAnalysisProvider (lazy)
        |                    \
   backend analyze            missing/unhealthy/unsupported
        |                              |
   AnalysisProviderResult       typed degradation evidence
   (advisory)                          |
        \______________________________/
                        |
   advisory_evidence_claim_references
                        |
   exact-tree / local analysis remains completion-authoritative path
                        |
   x dashed edge: optional success --X--> completion_authoritative
```

### 5.3 Multi-repository forest observation

```text
ReviewedForestManifest (aliases, sole_write, policies)
        |
   live Git materialization per root
        |
   RepositoryDescriptor (commit, tree, gitlink closure,
                         dirty overlay, authority, ignore policy)
        |
   RepositoryForest portable identity (no host credentials)
        |
   repository_corpus_index (Git objects first; bounded overlay)
        |
   fail-closed reject on escape, missing root, or authority mismatch
```

---

## 6. State and identity

| Kind | Owned by | Notes |
| --- | --- | --- |
| Parent monorepo commit | this repository | Includes gitlink SHAs as tree entries |
| Nested product commit | each submodule remote | Independent history |
| Portable forest identity | `RepositoryForest` | Commit/tree/gitlinks/dirty/authority/policies; not host path |
| Local locator | `LocalLocator` | Host path only; never folded into portable CIDs |
| Dirty overlay digest | forest descriptor | Bound into identity; dirty ≠ clean |
| Capability / health | provider adapters | Process-local, non-authoritative for completion |
| Completion / merge | supervisor validation + merge train | Requires exact allowed paths and receipts; optional providers excluded |

---

## 7. Trust, authorization, and failure semantics

### 7.1 Fail-closed conditions

- Forest path resolution: missing roots, path escape, symlink escape →
  `RepositoryForestError` (reject, do not broaden scope).
- Authority mismatch on manifest replay → reject.
- Read-only root with write allowlist → reject at authority construction.
- Optional analysis payloads over bounds / forbidden fields → fail closed.
- Fabricated or detached degradation evidence cannot claim requirement IDs.
- Optional provider claiming completion or rebinding authority → rejected as
  advisory / non-qualifying (pipeline tests).

### 7.2 Degradation (optional absent)

| Integration | When missing / disabled | Behavior |
| --- | --- | --- |
| `ipfs_kit_py` | Adapter constructor disables it, `IPFS_KIT_DISABLE`, or adapter import/init fails | `IPFSKitStorage` reports local `using_fallback`; `ENABLE_IPFS_KIT=false` is instead a router selection input, and the router independently reports HF cache or Kubo when its preferred candidate is unavailable |
| `ipfs_datasets_py` (runtime) | auto mode import fail or explicit disable | Local datasets_integration fallbacks |
| `ipfs_datasets_py` (analysis) | missing module, unhealthy, unsupported op | Typed degradation; local analysis continues |
| MCP++ gitlink | uninitialized empty dir | Spec pin absent; runtime modules still present in package |

### 7.3 Recovery

- Initialize a **single** needed submodule when a feature requires source:
  `git submodule update --init ipfs_kit_py` (or datasets path). Do not batch
  mutate gitlinks as part of documentation or accelerate code tasks.
- Prefer env disable in CI rather than pretending optional packages are present.
- Re-derive forest descriptors from live Git after any checkout change;
  do not trust stale manifest commit observations.

### 7.4 Non-authoritative signals

- Nested directory presence, board completion marks, import success alone,
  optional analysis health, Hammer/ATP candidates, and chat/model prose
  **do not** authorize mutation or completion of accelerator work.

---

## 8. Rationale

1. **Independent Git authority** keeps product release and review boundaries
   honest: a pin records a dependency, not a merged monorepo.
2. **Adapter ownership in accelerate** lets CI and minimal installs run without
   sibling packages while remaining explicit about capability.
3. **Empty-gitlink skipping** prevents false “found package” paths that only
   contain a reserved directory shell.
4. **Advisory optional analysis** allows rich nomination from
   `ipfs_datasets_py` without letting optional code forge completion receipts.
5. **Forest sole-write** encodes the multi-repo invariant that co-location is
   observation topology, not write sharing.

---

## 9. Alternatives considered

| Alternative | Breakage if adopted |
| --- | --- |
| Vendor siblings into one Git history | Loses independent release/review; confuses pin vs mutation |
| Treat nested path as automatic install | CI and sparse checkouts falsely claim capability |
| Eager import of kit/datasets at package import | Heavy/slow startup; CI fragility (hence `IPFS_ACCEL_SKIP_CORE` and lazy adapters) |
| Promote optional analysis to completion authority | Optional remote/package code manufactures acceptance |
| Share write authority across forest roots by co-location | Cross-repo mutation without sole-write control |
| Use in-tree `kit/` as if it were `ipfs_kit_py` | Wrong package; broken external storage integration |

---

## 10. Consequences

**Positive**

- Clear ownership: accelerate code and docs under this repo; siblings via pin.
- CI-friendly degradation with tests for both present and absent optionals.
- Forest identity portable across host paths.
- Supervisor completion remains fail-closed against optional evidence.

**Negative / costs**

- Empty nested directories confuse newcomers (“is the package missing?”).
- Dual naming: `ipfs_accelerate_py.kit` vs external `ipfs_kit_py`.
- Docs and install notes must restate init/extras prerequisites (DOC-021).
- Maintainers must not casually `submodule update` or commit gitlink changes
  inside unrelated tasks.
- Umbrella vs standalone discovery order requires careful test coverage.

---

## 11. Extension and compatibility

### 11.1 Adding an optional sibling

1. Prefer a new **in-tree adapter** under `ipfs_accelerate_py/` with lazy import.
2. Define env enable/disable and a capability probe that does not import until
   dispatch when construction must stay pure.
3. Document discovery candidates; skip empty gitlink shells.
4. Add focused tests for: disabled, missing, success, and fail-closed bounds.
5. If the sibling is a git submodule, record it in `.gitmodules` only under an
   ownership-reviewed change — **not** as a side effect of feature docs.

### 11.2 Compatibility notes

| Surface | Status |
| --- | --- |
| `ipfs_accelerate_py.mcp` | Compatibility facade; canonical server is `mcp_server` |
| `ipfs_transformers_py` dependency | Legacy path; prefer auto-patch |
| Root `mcpplusplus/` docs | Monorepo artifacts; not the nested gitlink |
| `docs/NESTED_PACKAGES.md` | Inventory/hygiene; architecture authority is this guide |

### 11.3 Conflict policy for this document’s task family

Tasks that own only `INTEGRATION_BOUNDARIES.md` **must not** initialize, update,
or commit submodule gitlinks. Operators update pins deliberately.

---

## 12. Operational signals

| Signal | Where | Operator use |
| --- | --- | --- |
| `get_datasets_status()` | `datasets_integration` | available / path / mode / reason |
| Backend selection receipt | `ipfs_backend_router` | which router role is active (`ipfs_kit_py` / kubo / cache) and why it degraded |
| `IPFSKitStorage.get_backend_status()` / logs | kit integration | adapter availability and `using_fallback`, independent of router selection |
| Analysis degradation evidence | supervisor integrations | typed reason + bound requirement IDs |
| Forest / manifest replay reasons | `repository_forest_manifest` | authority_mismatch, missing root, etc. |
| `git submodule status` | worktree | `-` prefix = not initialized |

Dirty or uninitialized checkouts are **normal** in documentation and CI
worktrees. Treat empty nested product dirs as **reserved slots**, not cleanup
candidates (`docs/NESTED_PACKAGES.md`).

---

## 13. Verification recipe

Re-run against a later checkout. These commands do not require submodules to be
initialized.

```bash
# Document and required name anchors
test -f docs/architecture/INTEGRATION_BOUNDARIES.md
rg -q 'ipfs_datasets_py' docs/architecture/INTEGRATION_BOUNDARIES.md
rg -q 'ipfs_kit_py' docs/architecture/INTEGRATION_BOUNDARIES.md
git diff --check

# Gitlink inventory still matches declared paths
test -f .gitmodules
rg -q 'path = ipfs_kit_py' .gitmodules
rg -q 'path = ipfs_datasets_py' .gitmodules
rg -q 'path = ipfs_accelerate_py/mcplusplus' .gitmodules

# Empty-gitlink discovery invariants (no optional package install required)
python - <<'PY'
from pathlib import Path
from ipfs_accelerate_py.datasets_integration import _resolve_datasets_source_path
from ipfs_accelerate_py.ipfs_kit_integration import _resolve_ipfs_kit_source_path
root = Path('.').resolve()
# Empty reserved dirs must not be treated as initialized sources:
for name, resolver in (
    ('ipfs_datasets_py', lambda: _resolve_datasets_source_path(root)),
    ('ipfs_kit_py', lambda: _resolve_ipfs_kit_source_path(root)),
):
    slot = root / name
    if slot.is_dir() and not (slot / name / '__init__.py').is_file():
        resolved = resolver()
        assert resolved is None or (resolved / name / '__init__.py').is_file(), (
            f'{name}: resolver returned non-package path {resolved}'
        )
print('discovery invariants ok')
PY

# Focused authority / optional-provider tests (when test deps installed)
# pytest -q test/test_datasets_integration.py test/test_ipfs_kit_integration.py
# pytest -q test/api/test_agent_supervisor_repository_forest.py \
#          test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py
```

**Review checks**

1. Co-location is never described as shared Git or mutation authority.
2. Integration paths and test paths above still exist.
3. Optional providers are described as nominating evidence only.
4. No claim that empty nested directories mean packages are installed.

---

## Related guides and records

| Document | Relationship |
| --- | --- |
| `docs/NESTED_PACKAGES.md` | Nested inventory and ASREF hygiene policy |
| `docs/architecture/overview.md` | One-screen runtime layers |
| `docs/architecture/IPFS_KIT_INTEGRATION.md` | Kit-oriented feature notes (verify against current adapters) |
| `docs/development/DOCUMENTATION_DRIFT_AUDIT_2026_08.md` | Finding 11: empty nested dirs / import prerequisites |
| `docs/architecture/GUIDE_CONVENTIONS.md` | ArchitectureGuideContract@1 |
| [Distributed runtime](DISTRIBUTED_RUNTIME.md) | CID/backend/P2P runtime (DOC-009); complements this ownership guide |
| [MCP runtime](MCP_RUNTIME.md) | Transport/tool planes (DOC-008) |

---

## Interfaces (documentation contracts)

| Interface | Version | Summary |
| --- | --- | --- |
| `RepositoryAuthorityBoundary` | 1 | Per-root Git identity + `read_only` / `read_write` with sole-write forest policy; co-location never grants authority |
| `OptionalIntegrationCapability` | 1 | Discovery → import → probe → typed degrade; optional evidence is advisory only |
| `GitlinkPin` | 1 | Parent tree records mode-160000 SHA for a path; pin is observational until live materialization |
