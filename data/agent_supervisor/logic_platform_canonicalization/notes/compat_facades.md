# LPC-062 Thin compatibility facades for logic.api and logic.__init__

**Task:** LPC-062 — Thin compatibility facades for `logic.api` and `logic.__init__`  
**Goal:** LPC-G060  
**Depends on:** LPC-060 (internal `LogicPlatformService@1` decomposition)  
**Interfaces:** `LogicAPICompatibility@1` (frozen public surface); adapter over `LogicPlatformService@1` / `LogicVerificationAPI@1`  
**Facade modules:**  
- `ipfs_datasets_py/ipfs_datasets_py/logic/api.py`  
- `ipfs_datasets_py/ipfs_datasets_py/logic/__init__.py`  
**Executable contract:** `ipfs_datasets_py/tests/fixtures/logic/api_v1/manifest.json`  
**Compatibility doc:** `ipfs_datasets_py/docs/logic/logic_api_v1_compatibility.md`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/test_logic_api_v1_compatibility.py -q`

## Purpose

LPC-060 freezes the internal platform service layout under
`logic/platform/*` while keeping `logic.verification_api` as the public
software-verification facade. LPC-062 freezes the **outer** package facades
that external callers already import:

| Facade | Path | Role |
| --- | --- | --- |
| Package root | `ipfs_datasets_py.logic` (`logic/__init__.py`) | Thin lazy re-export surface for submodules, registry helpers, and profile symbols |
| Legacy Python API | `ipfs_datasets_py.logic.api` (`logic/api.py`) | Frozen `LogicAPICompatibility@1` import surface for FOL/deontic/types/bridge converters |

These modules remain **compatibility facades**, not a second implementation of
families, providers, receipts, or provers. Semantic ownership stays in
family modules (`fol`, `deontic`, `TDFOL`, …), bridge/common/types leaves, and
the internal platform composition (via `verification_api` after LPC-060).

## Decision (authoritative)

1. **Public import paths stay stable.** Every name in the frozen
   `python_api.exact_exports` list of the v1 manifest continues to resolve
   from `ipfs_datasets_py.logic.api`. Family modules listed under
   `family_contracts` continue to resolve their declared symbols.
2. **No second implementation.** Facades re-export or lazily forward. They
   do not host independent converter, prover, receipt, or catalog logic.
3. **New software-verification writes use `verification_api`.** `logic.api`
   remains the reviewed pre-verification surface (`LogicAPICompatibility@1`).
   Verification symbols may be reached from `logic.api` only via **lazy**
   `__getattr__` so the frozen exact-export list is unchanged.
4. **Package root stays lazy.** `logic.__init__` must not eagerly import
   optional, heavy, or runtime modules. Attribute access loads declared
   submodules and helpers on demand.
5. **Deprecation diagnostics emit where appropriate.** Superseded import
   paths raise `DeprecationWarning` (or equivalent) on use; quiet import of
   the supported facades remains required.
6. **Import hermeticity is preserved.** Importing `ipfs_datasets_py.logic` or
   `ipfs_datasets_py.logic.api` must be quiet: no install, network, process
   spawn, environment mutation, filesystem write, or non-empty
   stdout/stderr/warnings (LPC-061 / v1 lazy-import contract).

## Facade roles

### `logic.__init__` — package root adapter

| Concern | Behavior |
| --- | --- |
| Classification | `compatibility_facade` (`shim.logic_api_facade` / package root) |
| Export sets | `_SUBMODULE_EXPORTS`, `_REGISTRY_EXPORTS`, `_PROFILE_D_EXPORTS`, `_PROFILE_G_EXPORTS` |
| Resolution | `__getattr__` → `importlib.import_module` / attribute bind into `globals()` |
| Quiet import | Silences only non-actionable Beartype PEP585 deprecations; emits no other import-time warnings |
| Deprecated alias | `tools` → redirects to `logic.integration` **with** `DeprecationWarning` |
| Platform relation | Does not re-implement services; exposes `verification_api` as a lazy submodule name when requested |

Declared submodule names include (non-exhaustive relative to implementation):
`api`, `bridge`, `cli`, `common`, `deontic`, `fol`, `flogic`, `modal`,
`TDFOL`, `verification_api`, `zkp`, `integration`, `integrations`, `tools`
(deprecated), and the remaining package catalog entries.

### `logic.api` — `LogicAPICompatibility@1`

| Concern | Behavior |
| --- | --- |
| Classification | `compatibility_facade` / interface `LogicAPICompatibility@1` |
| Frozen `__all__` | Exact ordered list from `tests/fixtures/logic/api_v1/manifest.json` |
| Eager re-exports | FOL/deontic converters, common errors/caches/monitors, shared types, TDFOL core formula constructors, bridge registry helpers, submodule-registry helpers, NL→UCAN convenience wrappers that do not require optional deps at import |
| Lazy via `__getattr__` | Optional NL→UCAN classes, BW133 delegation/conflict symbols, **and** the entire `LogicVerificationAPI@1` symbol set (not in exact_exports) |
| Formula identity | `Formula` / `Predicate` / `Variable` / `Constant` / `ProofResult` / `ProofStatus` / `ProofStep` are the TDFOL core objects (identity equality with `logic.TDFOL.tdfol_core`) |
| Thin constructors | `And` / `Or` / `Not` / `Implies` / `Forall` / `Exists` wrap TDFOL creators only |

#### Lazy verification bridge (additive, non-frozen)

`_VERIFICATION_API_EXPORT_NAMES` is resolved through `_lazy_verification_api()`
so callers may write:

```python
from ipfs_datasets_py.logic.api import get_verification_api, check
# or
from ipfs_datasets_py.logic import verification_api
```

without expanding the frozen exact-export contract. The implementation lives
only in `logic.verification_api` (and, after extraction, in
`LogicPlatformService@1`). The facade does not copy check/prove/install bodies.

## Dependency graph (import direction)

```
family leaves / common / types / bridge / TDFOL core
   ↑
logic.api  (LogicAPICompatibility@1 re-exports + lazy optional paths)
   ↑
logic.__init__  (lazy submodule / registry / profile surface)

logic/platform/* services  (LPC-060)
   ↑
verification_api.py  (public verification facade)
   ↑
logic.api.__getattr__ / logic.__init__["verification_api"]  (lazy only)
```

Rules:

* Arrows mean “may import”. Facades never become the semantic owner of a
  family, provider protocol, receipt verifier, or catalog snapshot.
* Cycles between facades and platform services are forbidden: services must
  not import `logic.api` or package-root `__getattr__` paths to obtain their
  own implementations.
* Heavy modules listed in the v1 manifest `lazy_imports.forbidden_after_import`
  (`logic.integration`, `logic.external_provers`, `logic.modal`, `logic.zkp`)
  must not load merely because `logic` or `logic.api` was imported.

## Deprecation diagnostics

Deprecation is **use-time**, not import-time of the supported facades.
Importing `ipfs_datasets_py.logic` or `ipfs_datasets_py.logic.api` must remain
quiet under the v1 contract (`observable_side_effects.warnings == []`).

| Surface | When | Diagnostic | Migration target |
| --- | --- | --- | --- |
| `logic.tools` (package attribute) | First access via `logic.__getattr__("tools")` | `DeprecationWarning`: removed in v2.0; use `logic.integration` or module-specific imports | `logic.integration`, `logic.fol`, `logic.deontic`, … |
| `logic.api_server` | `create_app()` / REST usage | `DeprecationWarning`: use MCP `logic_tools` | `ipfs_datasets_py.mcp_server.tools.logic_tools` |
| `logic.api` as primary **verify** name | Documented legacy posture | Prefer `logic.verification_api` / `LogicVerificationAPI@1` for new writes; name retained for converters | `verification_api` STABLE_OPERATIONS |
| Beartype PEP585 hints in dependency modules | Import of facades | Filtered (not user-facing deprecation of logic APIs) | Upstream typing cleanup; not an LPC-062 product signal |

Required properties of diagnostics:

1. **Category** is `DeprecationWarning` (or a documented subclass) so
   `-W default::DeprecationWarning` surfaces them.
2. **stacklevel** points at the caller, not the facade helper.
3. **Supported quiet imports stay quiet.** Deprecation for `tools` /
   `api_server` must not fire merely because `logic` or `logic.api` was
   imported.
4. **No authority upgrade.** A deprecation message never implies that a
   legacy path is still a preferred proof authority.

## Public import preservation matrix

| Import | Must resolve | Notes |
| --- | --- | --- |
| `from ipfs_datasets_py.logic import api` | yes | Package root lazy submodule |
| `from ipfs_datasets_py.logic.api import Formula, Predicate, …` | yes | Exact identity with TDFOL core |
| `from ipfs_datasets_py.logic.api import convert_text_to_fol` | yes | Legacy converter; also MCP `text_to_fol` |
| `from ipfs_datasets_py.logic.api import convert_legal_text_to_deontic` | yes | Legacy converter; also MCP `legal_text_to_deontic` |
| `from ipfs_datasets_py.logic.api import DeonticOperator, TemporalOperator` | yes | Enum values frozen in manifest |
| `import ipfs_datasets_py.logic.fol` / `.deontic` / `.TDFOL` / … | yes | Family contracts in manifest |
| `from ipfs_datasets_py.logic.api import get_verification_api` | yes (lazy) | Not in exact_exports; additive |
| `from ipfs_datasets_py.logic import verification_api` | yes (lazy) | Canonical verification facade module |
| `from ipfs_datasets_py.logic import tools` | yes (deprecated) | Redirect + `DeprecationWarning` |

Changing the frozen exact-export order or removing a family contract symbol is
a deliberate `LogicAPICompatibility@1` change and requires updating the
manifest, compatibility doc, and executable test together.

## Authority and fail-closed semantics (facade-level)

Facades must not reinterpret proof outcomes:

| Observation | Facade obligation |
| --- | --- |
| Optional tool / prover missing | Propagate unavailable / error envelopes; never invent success |
| Cache hit | Preserve cached authority only; never raise it |
| Bridge gate unavailable | `compiles is False`; not soft success |
| Simulated ZKP | Remain simulation-class; not cryptographic theorem proof |
| Verification operation success | Status and authority come from `verification_api` / platform services only |

These rules are enforced by the v1 compatibility test suite and by LPC-032 /
LPC-060 service boundaries. Facades do not add a parallel success path.

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-060 | Owns `logic/platform/*` service decomposition and `verification_api` thinning |
| LPC-061 | Owns pure-data import hermeticity tests against the platform import graph |
| LPC-002 / LPC-007 | Classified public exports, shims, and deprecated modules; this note implements the LPC-062 facade contract over that inventory |
| LFV-G011 / LFV-G070 | Froze `LogicAPICompatibility@1` and additive verification facade rules |
| LPC-100 | Platform manifest handshake is internal; facades do not require Git layout |
| LPC-110 | Supervisor clients handshake then call typed operations through facades |

## What this task does **not** do

* Does not create a second top-level package or parallel public API name.
* Does not delete or rename frozen `logic.api` exact exports.
* Does not move family provers into the facade modules.
* Does not implement the full `logic/platform/*` service extraction (LPC-060).
* Does not replace CLI / MCP contracts; those remain under the same v1
  manifest and channel-parity gates.
* Does not mark deprecation complete for every historical datasets path; only
  the package-root and documented REST/tools paths listed above are in scope
  for LPC-062 diagnostics.
* Does not claim that presence of a re-export implies provider availability or
  production readiness.

## Implementation ownership (current modules)

| Path | Ownership under LPC-062 |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/api.py` | Thin re-export + lazy optional/verification bridge; no second check/prove body |
| `ipfs_datasets_py/ipfs_datasets_py/logic/__init__.py` | Lazy package surface + `tools` deprecation redirect |
| `ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py` | Public verification facade (LPC-060); target of lazy api bridge |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/*` | Internal composition (LPC-060/LPC-100); not imported eagerly by facades |
| `ipfs_datasets_py/tests/unit/logic/test_logic_api_v1_compatibility.py` | Executable acceptance for public imports, payloads, lazy quiet import |
| `ipfs_datasets_py/tests/fixtures/logic/api_v1/manifest.json` | Frozen exact exports, family contracts, lazy-import forbidden set |
| `data/agent_supervisor/logic_platform_canonicalization/notes/compat_facades.md` | This note (LPC-062 declared output) |

## Acceptance matrix

| Check | Fail-closed behavior | Evidence |
| --- | --- | --- |
| Existing public imports resolve | Manifest exact_exports and family symbols import successfully | `test_stable_api_exports_and_family_imports` |
| Formula identity | `logic.api` TDFOL types are the core objects | same test |
| Quiet lazy import | No forbidden modules, warnings, or stderr on `logic` / `logic.api` import | `test_imports_are_quiet_lazy_and_side_effect_free` |
| Deprecation diagnostics | `logic.tools` access emits `DeprecationWarning`; `api_server` path warns on use | package `__getattr__` / `api_server.create_app` |
| No second implementation | Facades re-export or lazy-forward only | this note + source layout |
| Verification remains additive | Verification symbols via `__getattr__`; exact_exports frozen | `api.py` `_VERIFICATION_API_EXPORT_NAMES` |
| Authority non-upgrade | Unavailable/cache/ZKP/bridge semantics unchanged | v1 authority tests |
| Validation green | `pytest …/test_logic_api_v1_compatibility.py -q` | CI / this task gate |

## Status

LPC-062 freezes `logic.api` and `logic.__init__` as thin compatibility
facades over family leaves and the LPC-060 platform / verification composition.
Existing public imports remain resolvable under `LogicAPICompatibility@1`.
Deprecation diagnostics emit for superseded `tools` and REST (`api_server`)
entry points without breaking the quiet-import contract for supported facades.
