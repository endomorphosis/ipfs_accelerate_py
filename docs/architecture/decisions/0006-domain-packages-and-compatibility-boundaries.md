# ADR-0006: Domain packages own code; compatibility must converge on exact delegation

- **Status:** Accepted
- **Date:** 2026-08-03
- **Last verified:** 2026-08-03 @ b128cceef
- **Owner:** architecture maintainers
- **Deciders:** architecture maintainers; agent-supervisor package owners; MCP runtime maintainers
- **Scope:** How production code is owned and laid out under semantic domain packages
  (primarily `ipfs_accelerate_py.agent_supervisor`), how delivery **programs**
  relate to that ownership, how highest-layer composition (`entrypoints`) sits
  above the domain DAG, how independent Git repository authority is preserved
  across co-located products, and how **compatibility** surfaces (package-root
  aliases, stable export manifests, MCP/MCP++ facades and shims) must preserve
  exact object identity, behavior, and mutable-state ownership without becoming
  a second implementation or a second authority plane. This Accepted record
  also identifies the remaining cross-tier import and legacy MCP exceptions;
  it does not certify that the full target DAG or MCP cutover is complete.
- **Non-goals:** Individual control-operation semantics (see `CONTROL_PLANE.md`);
  worktree/lease fencing protocol (ADR-0004); mutable coordination store versus
  immutable replication (ADR-0005); objective-versus-taskboard
  projection design (ADR-0001); catalog/usage/routing plane separation
  (ADR-0003); sealing delivery boards or marking program tasks complete.
- **Supersedes:** none
- **Superseded-by:** none
- **Related guides:**
  - [`docs/architecture/agent_supervisor/PACKAGE_MAP.md`](../agent_supervisor/PACKAGE_MAP.md)
  - [`docs/architecture/agent_supervisor/PROGRAMS.md`](../agent_supervisor/PROGRAMS.md)
  - [`docs/architecture/agent_supervisor/packages/entrypoints.md`](../agent_supervisor/packages/entrypoints.md)
  - [`docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md`](../AGENT_SUPERVISOR_PHILOSOPHY.md)
  - [`docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md`](../AGENT_SUPERVISOR_ARCHITECTURE.md)
  - [`docs/architecture/INTEGRATION_BOUNDARIES.md`](../INTEGRATION_BOUNDARIES.md)
  - [`docs/architecture/MCP_RUNTIME.md`](../MCP_RUNTIME.md)
- **Source anchors:**
  - `ipfs_accelerate_py/agent_supervisor/__init__.py` —
    `AGENT_SUPERVISOR_DOMAIN_PACKAGES`,
    `AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE`,
    `AGENT_SUPERVISOR_PUBLIC_API_EXPORTS` /
    `AGENT_SUPERVISOR_V2_STABLE_EXPORTS`, package-root lazy resolution
  - Domain packages under `ipfs_accelerate_py/agent_supervisor/{core,control,
    task_sources,context,analysis,proof,objectives,planning,validation,prompt,
    merge,rescue,runtime,self_improvement,integrations,todo_daemon,entrypoints}/`
    (each with `README.md`)
  - `ipfs_accelerate_py/agent_supervisor/entrypoints/` —
    `ENTRYPOINT_CONTRACT_EXPORTS`, `ENTRYPOINT_LOWER_DOMAIN_PACKAGES`,
    identity-preserving contract re-exports
  - `ipfs_accelerate_py/agent_supervisor/repository_forest.py` —
    `RepositoryAuthority`, `RepositoryForest`, `sole_write_alias`
  - `ipfs_accelerate_py/mcp/` (compatibility facade),
    `ipfs_accelerate_py/mcp_server/` (canonical runtime),
    `ipfs_accelerate_py/mcp_server/compatibility.py` (shared shim helpers),
    `ipfs_accelerate_py/mcp/unified_registry.py` (legacy mutable registry debt),
    `ipfs_accelerate_py/mcp/server.py` (unified-default bridge and legacy
    rollback/error fallback),
    `ipfs_accelerate_py/mcplusplus_module/` (Trio-first surface that must
    delegate, not fork, tool authority)
  - `test/api/test_agent_supervisor_v2_public_api.py` (manifest exactness and
    object-identity checks)
  - `test/api/test_agent_supervisor_entrypoint_package.py` (cold import,
    contract identity, and no lower-package import of `entrypoints`)
  - Integration boundary tests under `test/api/test_agent_supervisor_repository_forest*.py`
    and kit/datasets integration tests cited by `INTEGRATION_BOUNDARIES.md`

## Status meanings (do not invent new values)

| Value | Use when |
| --- | --- |
| Proposed | Decision is under review; **not** yet evidenced current design |
| Accepted | Decision matches current code/tests/ops practice and is normative for Scope |
| Deprecated | Still historical; prefer another practice for new work |
| Superseded | Replaced by the ADR in Superseded-by |
| Rejected | Considered and not adopted; retained to document the negative choice |

Only **Accepted** records are current design authority. **Proposed** records
must not be treated as implemented system law.

## Context

The monorepo and the agent supervisor evolved under several pressures that pull
layout, delivery, and compatibility in incompatible directions:

1. **Feature delivery uses programs.** Self-improvement, domain layout,
   codebase-proof, catalog work, and similar efforts are scheduled as
   **programs**: objective heaps plus Markdown taskboards with stable board
   prefixes (`ASI-`, `ASREF-`, `CBP-`, …). Operators and agents naturally talk
   about “the ASREF package” or “the CBP modules.” That vocabulary is correct
   for **boards and evidence**, but dangerous if it becomes the **Python
   package tree**. Board IDs renumber, complete, archive, and restart; code
   ownership must outlive any one board.

2. **Flat module sprawl does not scale.** A single package root with dozens of
   production modules made ownership, dependency direction, cold-import policy,
   and public-API review unenforceable. Reviewers could not tell which concern
   owned a symbol; agents duplicated logic next to the nearest taskboard name.

3. **Callers need stable import paths during migration.** Historical flat
   stems, package-root re-exports, MCP import paths under `ipfs_accelerate_py.mcp`,
   and MCP++ Trio modules still have live consumers. Compatibility that
   re-implements behavior, maintains a second mutable registry, or clones
   classes so `is` identity breaks creates **two systems** that drift under
   partial migration.

4. **Co-located products are not one repository.** Nested paths such as
   `./ipfs_kit_py` or `./ipfs_datasets_py` are discovery slots and gitlink pins.
   Sharing a parent directory does not share Git history, dirty overlay, or
   write authority. Collapse of repository authority produces false completion
   evidence and unsafe cross-product mutation.

5. **Composition is not ownership.** Product edges (Python embeddings, CLI,
   MCP/MCP++) need a highest-layer composition surface, but that surface must
   not re-own control operations, daemons, or proof. Lower packages must never
   import upward into composition.

Who is affected: library consumers and MCP clients (import stability),
implementation agents (where to place code), operators (which path is
canonical), and maintainers of domain packages, entrypoints, and MCP runtime.

What fails if deferred: programs invent parallel supervisors named after board
prefixes; compatibility facades silently become second runtimes with divergent
state; package cycles and upward entrypoint imports reappear; co-located
siblings inherit write rights they do not own.

## Decision

**Semantic domain packages define code ownership. Delivery programs define
scheduled work, not the package tree. Compatibility surfaces must preserve
exact behavior, object identity, and single-writer state by delegating to the
canonical owner; existing exceptions are migration debt, not an alternate
architecture or evidence that the cutover is complete.**

### 1. Domain packages encode ownership; the complete tier DAG is a target

Production modules under `ipfs_accelerate_py.agent_supervisor` live in
**semantic domain packages** named by product role, not by taskboard prefix:

| Target layer (dependencies flow toward lower layers) | Packages |
| --- | --- |
| Foundation | `core`, `control`, `task_sources`, `context`, `analysis`, `proof` |
| Mid | `objectives`, `planning`, `validation`, `prompt` |
| Ops | `merge`, `rescue`, `runtime`, `self_improvement` |
| Edge | `todo_daemon`, `integrations` |
| Facade (highest composition) | `entrypoints` |

The package inventory, landed-module map, semantic names, package directories,
and ownership READMEs are current. The table's complete cross-package import
direction is the accepted convergence target, not a claim that every current
module already obeys it.

Normative layout and convergence rules:

1. **Target acyclic DAG.** New and migrated code must not introduce package
   cycles. Higher layers may import lower; lower-to-higher imports must be
   removed rather than treated as precedent.
2. **Semantic names are public inventory.**
   `AGENT_SUPERVISOR_DOMAIN_PACKAGES`,
   `AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE`,
   `AGENT_SUPERVISOR_*_PACKAGES` / `*_STEMS`, and related layout constants use
   product-role names. Board-prefix constant spellings
   (`AGENT_SUPERVISOR_G0xx_*`, `AGENT_SUPERVISOR_EVIDENCE_CLUSTER_*`,
   `AGENT_SUPERVISOR_LANDED_MODULE_OWNERS`) remain only as **compatibility
   aliases** of the semantic names. Board IDs may appear as **string values**
   for scanners and historical receipts; they are not package names.
3. **Package README is the ownership contract.** Every domain package ships
   `README.md` describing purpose and bounds. New features land in the package
   that owns the concern (see `PACKAGE_MAP.md` “Where new work goes”).
4. **Prefer domain imports for new code.**
   `from ipfs_accelerate_py.agent_supervisor.<pkg>.<mod> import …`. Do not
   reintroduce long-lived flat root production modules or root-level re-export
   stubs as the primary home of behavior.
5. **No new daemon / runtime dependencies in foundation.** `core` must not
   import `todo_daemon`, `runtime`, `merge`, or `rescue`. Lower packages must
   not import `entrypoints`.

The highest-layer `entrypoints` boundary is landed and tested: its initializer
imports contracts only, its eager exports preserve contract object identity,
and the lower domain packages are AST-scanned to reject imports of
`entrypoints`. The broader tier ordering is not yet fully enforced. Current
cross-tier debt includes:

- `analysis/cache_coordinator.py` importing `merge` and `runtime`;
- `objectives/objective_daemon.py` importing `runtime`;
- `objectives/bundle_supervisor.py` importing `merge`, `runtime`, and
  `todo_daemon`; and
- `task_sources/task_source.py` importing `runtime` and `prompt`.

These exceptions do not change which semantic package owns each module. They
do mean that “full acyclic tier DAG implemented” is not valid completion
evidence until a source-level dependency check passes for the entire package
tree.

### 2. Delivery programs do not define code ownership

A **program** is a delivery construct: boards + objectives (+ optional modules
placed under the domain package that already owns the concern). Programs layer
on the shared control plane; they do not fork a second supervisor.

| Construct | Answers | Does not answer |
| --- | --- | --- |
| Program / board prefix (`ASREF-`, `CBP-`, …) | What work is scheduled; which evidence cluster; operator drain order | Python package name; import DAG; long-term module home |
| Domain package (`proof/`, `control/`, …) | Who owns code, tests, and cold-import policy for a concern | Whether a particular board task is complete |
| Objective heap | Durable intent and acceptance | Package tree shape |

Therefore:

- **Do not** create packages, public APIs, or primary module trees named only
  after board prefixes or program slogans.
- **Do** place program-specific modules under the semantic owner (for example
  codebase-proof logic under `proof/` and `context/`; implementation loops under
  `todo_daemon/`).
- **Do** keep board IDs in task text, goal packets, and evidence footers—not as
  the only public symbol names for durable product surfaces.
- Completing a program does not move ownership; abandoned programs must not
  leave orphan “program packages” that bypass the domain DAG.

### 3. `entrypoints` is highest-layer composition, not a second control plane

`agent_supervisor.entrypoints` is the sole highest-layer composition package:

- Eager public inventory is `ENTRYPOINT_CONTRACT_EXPORTS` (provider-free
  contracts). Package exports assign the **exact objects** from
  `entrypoints.contracts` so package-root and module imports preserve `is`
  identity.
- Runtime/service facades, when added, must be lazy and listed in
  `ENTRYPOINT_LAZY_FACADE_EXPORTS`. Importing `entrypoints` must not scan a
  repository, resolve a provider/service, open DuckDB/Parquet/IPLD/IPFS, or
  start a process.
- Lower domain packages **never** import `entrypoints`. Composition flows one
  way: product edge → `entrypoints` → domain packages.
- Domain behavior (control ops, daemons, merge, proof) stays in lower packages.
  Entrypoints **compose**; they do not re-implement the control plane.

### 4. Independent repository authority

Co-location is not shared Git authority. Nested product slots and sibling
checkouts keep independent remotes, commits, trees, dirty overlays, and
`RepositoryAuthority` records. `RepositoryForest` may observe many roots, but
**sole write authority** is explicit (`sole_write_alias`); sibling roots never
inherit mutation rights from sharing a parent directory. Optional integrations
nominate evidence; they are not completion-authoritative for another product’s
tree. See `INTEGRATION_BOUNDARIES.md`.

### 5. Compatibility must preserve exact behavior and state—without a permanent second implementation

Compatibility exists so callers can migrate without dual ownership. The
accepted end state is bounded by exactness; the legacy exceptions below remain
explicit migration debt:

1. **Single canonical owner per concern.** Behavior, validation, registries,
   and mutable state live in one authoritative package/module. Facades, shims,
   and package-root aliases **delegate** or **re-export** that owner.
2. **Object identity.** Reviewed public exports resolve to the same objects as
   the owner module (`x is owner.x`). Package import, owner-module import, CLI
   adapter, and MCP adapter must not acquire distinct class, enum, catalog, or
   operation identities for the same contract member
   (`AGENT_SUPERVISOR_PUBLIC_API_EXPORTS` / v2 stable manifests;
   `ENTRYPOINT_CONTRACT_EXPORTS`; discovery publications that bind the same
   catalog and dispatcher identity).
3. **No permanent second mutable registry or lifecycle.** Compatibility layers
   must converge away from parallel tool registries, lease tables, objective
   heaps, or completion authorities. Shared helpers (for example
   `mcp_server.compatibility` resolvers and stubs) are utilities, not a second
   registry.
4. **MCP / MCP++ facade rule.** `ipfs_accelerate_py.mcp` is a **compatibility
   facade** that bridges to the unified `mcp_server` runtime by default.
   `mcplusplus_module` may host Trio-first transport surfaces but must
   **delegate** tool registration and business logic to canonical
   `mcp_server` paths rather than grow unique scheduler or policy behavior.
   Deferred modules may exist as thin adapters; they are not the target
   authority plane (`MCP_RUNTIME.md`).
5. **Historical path aliases are temporary and exact.** Flat stem resolution
   via package-root aliasing may preserve import spells for landed modules
   mapped in `AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE`. New callers must use
   domain paths. Do not add long-lived stub modules that copy implementation
   bodies.
6. **Exact compatibility ≠ soft approximation.** Compatibility means same
   symbols, same identities, same admission and state effects (or an explicit
   fail-closed stub that refuses work). It does not mean “roughly equivalent
   reimplementation,” partial shadow state, or best-effort divergence under
   feature flags that permanently fork the product.
7. **Business logic stays out of shims.** Extension adds tools and behavior in
   canonical packages; shims only adapt import paths and transport.

The MCP cutover is incomplete at this verification point. The `mcp` facade
selects `mcp_server` by default, but `mcp/unified_registry.py` still owns a
module-global mutable legacy tool registry and call path. In addition,
`mcp/server.py` can construct the legacy `MCPServerWrapper` after a unified
bridge error, during cutover dry-run, or under explicit rollback. Those paths
are compatibility escape hatches, not evidence that duplicate mutable state
has already been eliminated. New behavior belongs in `mcp_server`; the legacy
registry and fallback must not acquire new authority while callers migrate.

### Ownership summary

| Concern | Authoritative home | Compatibility form allowed |
| --- | --- | --- |
| Domain module behavior | Owning domain package | Package-root / historical stem alias to same module |
| Control ops, catalogs, discovery | `control/` + reviewed public manifests | Transport adapters bind same identities |
| Prompt-first contracts | `entrypoints.contracts` | Package re-export with object identity |
| MCP tool registry / dispatch | `mcp_server` (target and unified default) | `mcp` bridge; legacy mutable registry/wrapper remain bounded migration exceptions |
| Shared MCP shim helpers | `mcp_server.compatibility` | Thin aliases from historical modules |
| Git mutation authority | Explicit `RepositoryAuthority` / sole-write alias | Observation and evidence nomination only |
| Delivery schedule | Program boards + objectives | Never a package ownership signal |

## Alternatives

### Alternative A: Board-ID / program-named packages

- **Summary:** Organize code as `asref/`, `cbp/`, `asi/`, or similarly
  prefix-named packages so each delivery program owns a tree matching its
  taskboard. Domain concerns (proof, merge, control) scatter under whichever
  program last touched them.
- **Expected benefits:** Agents find “their” code next to the board; program
  completion looks like a self-contained package delete; simple mental model for
  one-off programs.
- **Why not chosen:** Delivery programs do **not** define durable ownership.
  - Board prefixes renumber, archive, and restart; package names would thrash or
    leave permanent fossil trees.
  - Cross-program reuse (leases, validation, control ops) forces either
    duplication or illegal upward imports across program silos.
  - Public APIs named only after board IDs train callers on ephemeral delivery
    labels instead of product roles (`PROGRAMS.md`, philosophy §6–7).
  - Completing a program would tempt deletion of shared infrastructure that
    still serves other programs.

### Alternative B: Compatibility as a second full implementation

- **Summary:** Keep legacy packages (`mcp`, flat `agent_supervisor` modules,
  `mcplusplus_module` tool stacks) as complete parallel runtimes with their own
  registries, state, and behavior, evolving independently for “stability.”
- **Expected benefits:** Zero migration risk for old callers; teams can ship
  features on either stack; no need for identity-preserving re-exports.
- **Why not chosen:** Compatibility that becomes a second implementation
  permanently forks the product.
  - Divergent validation, UCAN, and dispatch paths mean the same logical tool
    call has two admission stories.
  - Duplicate mutable state (two registries, two completion views) produces
    split-brain evidence and “green on the facade, failed on canonical” races.
  - Identity breaks (`is` false across imports) break isinstance checks,
    singleton catalogs, and transport parity tests already enforced for v2
    public API and entrypoint contracts.
  - Migration never ends; every fix ships twice. The adopted rule is
    **compatibility without dual ownership**: facades default to canonical
    behavior and shared helpers, not twin systems (`MCP_RUNTIME.md` rationale).

### Alternative C: Soft / approximate compatibility (best-effort wrappers)

- **Summary:** Allow facades to reimplement “close enough” behavior—different
  class objects, partial method sets, alternate storage, or optional shadow
  caches—as long as demos pass.
- **Expected benefits:** Faster shims; less discipline on export manifests;
  easier one-off customer patches.
- **Why not chosen:** Approximate compatibility is still a second
  implementation under another name. Exact behavior and state are the contract:
  same owner objects, same dispatch, same fail-closed stubs when optional
  symbols are missing. Soft wrappers hide drift until multi-lane automation or
  identity-sensitive callers fail in production.

### Alternative D: Single flat package root (no domain packages)

- **Summary:** Keep all production modules at `agent_supervisor/*.py` with
  informal ownership comments and no enforced DAG.
- **Expected benefits:** Fewer directories; shorter import paths; less layout
  machinery.
- **Why not chosen:** Flat sprawl is what the domain-layout cutover removed.
  Without package boundaries, cold-import policy, “no daemon in core,” and
  entrypoint one-way composition cannot be enforced. Agents reintroduce cycles
  and misplace features beside the last edited file.

### Alternative E: Shared Git authority for co-located products

- **Summary:** Treat nested product directories under one monorepo checkout as
  one repository for mutation, completion, and evidence—path co-location
  implies write rights.
- **Expected benefits:** Simpler forest model; one dirty overlay; fewer
  authority records.
- **Why not chosen:** Nested products have independent remotes and histories.
  Shared authority would let accelerate work corrupt sibling trees, treat empty
  gitlinks as owned content, and accept completion evidence against the wrong
  Git identity. Co-location remains discovery only (`INTEGRATION_BOUNDARIES.md`,
  `RepositoryAuthority`).

## Consequences

### Positive

- Code ownership is stable across program lifecycles; boards can regenerate
  without renaming packages.
- Semantic ownership gives agents a deterministic “where new work goes” map;
  the target tier DAG gives dependency cleanup an auditable end state.
- Highest-layer `entrypoints` composition stays one-way and import-safe.
- Compatibility has an exact, testable convergence contract for public object
  identity and delegated state across Python/CLI/MCP surfaces.
- Independent repository authority prevents co-location from becoming false
  write rights or false completion evidence.
- Rejecting board-ID package naming keeps public APIs product-shaped rather
  than taskboard-shaped.

### Negative

- Contributors must learn the domain map and program glossary instead of
  “package == board prefix.”
- Compatibility layers add indirection (aliases, facades, lazy resolution) and
  must be reviewed so they never accumulate unique business logic.
- Enforcing object identity and exact export manifests increases test and
  release discipline when expanding public APIs.
- Migration from historical flat or `mcp` import paths requires explicit caller
  updates over time; long-lived dual import spells are technical debt even when
  behavior is correct.
- Nested products need forest/authority plumbing rather than a single root
  checkout mental model.

### Neutral / residual risks

- The complete package tier DAG is not yet landed. The cross-tier imports named
  in Decision §1 must be removed or explicitly redesigned before a full-DAG
  completion claim is valid.
- `mcp/unified_registry.py` and the fallback `MCPServerWrapper` remain mutable
  legacy runtime debt. The unified bridge is the default, but error/dry-run/
  rollback paths can still select the legacy implementation.
- Some Trio-first or deferred MCP modules may still exist as migration debt;
  they must continue to shrink toward delegation.
- Planned entrypoint facades (`ENTRYPOINT_LAZY_FACADE_EXPORTS`) are empty today;
  future facades must obey the same cold-import and non-ownership rules.
- Board-prefix aliases remain for scanners; documentation and new code must
  prefer semantic names so aliases do not re-become the primary vocabulary.
- Exact compatibility does not replace lease/fence isolation, validation, or
  authoritative completion gates (other ADRs and guides).

## Evidence

| Claim in Decision | Evidence (path, test, or operational check) | Notes |
| --- | --- | --- |
| Semantic domain package inventory | `AGENT_SUPERVISOR_DOMAIN_PACKAGES` in `agent_supervisor/__init__.py`; package dirs + READMEs | Product-role names |
| Landed stem → package map | `AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE`; deprecated `LANDED_MODULE_OWNERS` alias | Board spellings are aliases |
| Programs ≠ packages | `PROGRAMS.md`; philosophy §6–7; `PACKAGE_MAP.md` “new *program*” row | Code still in domain packages |
| Target tier DAG | `PACKAGE_MAP.md` rules plus the current exceptions listed in Decision §1 | Accepted direction; not yet a full-tree pass |
| No upward import of `entrypoints` | `ENTRYPOINT_LOWER_DOMAIN_PACKAGES`; `test/api/test_agent_supervisor_entrypoint_package.py` | Landed one-way composition boundary |
| Entrypoint contract object identity | `entrypoints/__init__.py` assigns `getattr(_contracts, name)`; `test/api/test_agent_supervisor_entrypoint_package.py` | Landed eager contracts only |
| v2 public API exact identity | `test/api/test_agent_supervisor_v2_public_api.py` (`is` checks across owner, package, CLI, MCP) | Manifest immutable |
| MCP unified-default compatibility and remaining dual-state debt | `mcp/server.py`; `mcp/unified_registry.py`; `MCP_RUNTIME.md` compatibility rationale | Default delegates; legacy registry/fallback still exist |
| Independent repository authority | `repository_forest.py` (`RepositoryAuthority`, `sole_write_alias`); `INTEGRATION_BOUNDARIES.md` | Co-location ≠ authority |
| Domain layout architecture narrative | `AGENT_SUPERVISOR_ARCHITECTURE.md` domain package layout section | Matches package map |

## Verification

From the repository root:

```text
# ADR structure and acceptance keywords
test -f docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md
rg -q '^## Alternatives' docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md
rg -qi 'compatib' docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md

# Domain inventory and semantic names still present
rg -n 'AGENT_SUPERVISOR_DOMAIN_PACKAGES|AGENT_SUPERVISOR_LANDED_MODULE_TO_PACKAGE' \
  ipfs_accelerate_py/agent_supervisor/__init__.py

# Entrypoint one-way boundary, cold import and identity-preserving exports
rg -n 'ENTRYPOINT_CONTRACT_EXPORTS|ENTRYPOINT_LOWER_DOMAIN_PACKAGES' \
  ipfs_accelerate_py/agent_supervisor/entrypoints
python -m pytest test/api/test_agent_supervisor_entrypoint_package.py -q

# Public API identity tests
python -m pytest test/api/test_agent_supervisor_v2_public_api.py -q

# Unified MCP is the default; legacy mutable registry/fallback debt is visible
rg -n 'create_unified_server|bridge_error_fallback|MCPServerWrapper' \
  ipfs_accelerate_py/mcp/server.py
rg -n '_global_registry|def get_global_registry|def call_tool' \
  ipfs_accelerate_py/mcp/unified_registry.py

# Forest authority symbols
rg -n 'class RepositoryAuthority|sole_write_alias' \
  ipfs_accelerate_py/agent_supervisor/repository_forest.py
```

Pass signals: domain packages remain the ownership map; board-prefix names stay
aliases/values only; the tested `entrypoints` boundary remains cold and one-way;
reviewed public identities hold; unified MCP remains the default while the
named legacy exceptions remain bounded and visible; forest authority remains
per-root.

Fail signals (ADR stale): new primary packages named only after board prefixes;
new compatibility business logic or mutable authority expands beyond the
disclosed legacy exceptions; public export `is` identity breaks; lower packages
import `entrypoints`; the tier-DAG debt is claimed complete without a full-tree
check; co-located siblings are treated as shared write authority without
explicit sole-write binding.

## Review triggers

- [ ] Source anchors no longer match the Decision statement
- [ ] A recorded negative consequence becomes unacceptable
- [ ] A rejected alternative becomes viable without those costs
- [ ] Domain package DAG is restructured or cycles are introduced
- [ ] Compatibility facade gains unique business logic or a second registry
- [ ] Public export or entrypoint contract object identity is intentionally broken
- [ ] Repository forest authority model changes
- [ ] Superseding design is Accepted under a new ADR number

When superseding: create a new ADR number; set this file to **Superseded** with
`Superseded-by`; set the successor’s `Supersedes`; do not delete this file.

## Notes (optional)

- Program reservation for this ADR is DOC-020 in the documentation-refresh
  decision wave (IDs 0001–0006). Index table refresh is owned by a later
  closeout task; this file’s Status field is authoritative for the decision.
- Historical domain-layout cutover evidence (`LAYOUT_CUTOVER_EVIDENCE.md`,
  ASREF goal tables) is scanner-oriented; day-to-day ownership is
  `PACKAGE_MAP.md` and domain package READMEs.
- Related non-goals intentionally defer lease fencing, coordination-store
  mutability, and intent-versus-projection authority to sibling ADRs so this
  record stays focused on package ownership and compatibility exactness.
