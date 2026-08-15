# LPC-061 Keep pure-data imports side-effect free

**Task:** LPC-061 — Keep pure-data imports side-effect free  
**Goal:** LPC-G060  
**Depends on:** LPC-060 (`notes/api_decomposition.md`)  
**Interface gate:** pure-data import hermeticity for the LPC-G060 graph  
**Enforcement test:** `ipfs_datasets_py/tests/unit/logic/test_pure_data_import.py`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/test_pure_data_import.py -q`

## Purpose

LPC-060 freezes the internal `logic/platform/*` service layout and the pure-data
versus explicit-runtime split. LPC-061 owns the **import hermeticity gate**:
importing contracts, catalog, syntax, formalization, provider protocol, and the
supervisor adapter must remain a pure data load.

Runtime discovery, install, probe, and execution stay explicit operations on
named public APIs. Catalog presence never implies executability. Provider
success never implies proof authority (LPC-032).

## Pure-data import inventory

| Role | Module | Interface / surface | Notes |
| --- | --- | --- | --- |
| **contracts** | `ipfs_datasets_py.logic.syntax_core.contracts` | `SourceDocument@1` / parse contracts | Stdlib + identity namespaces only |
| **catalog** | `ipfs_datasets_py.logic.families.canonical_catalog` | `CanonicalLogicCatalogSnapshot@1` | Sealed composition root; no probe |
| **syntax** | `ipfs_datasets_py.logic.syntax_core` | `LogicSyntaxCore@1` | Lazy package root; leaf modules load on attribute access |
| **formalization** | `ipfs_datasets_py.logic.formalization` | domain-neutral formalization contracts | Lazy package root; advisors stay out of root export |
| **provider protocol** | `ipfs_datasets_py.logic.backends.protocol_v2` | `LogicProviderProtocol@2` | Typed requests only; no solver/backend runner |
| **provider protocol (v1 wire)** | `ipfs_datasets_py.logic.backends.provider` | `LogicProvider@1` | Portable JSON envelope; stdlib leaf |
| **provider protocol (v1 adapter)** | `ipfs_datasets_py.logic.backends.protocol_v1_adapter` | `LogicProviderProtocolV1Adapter@1` | Dual-read disposition only |
| **verification contracts facade** | `ipfs_datasets_py.logic.verification_api` | `LogicVerificationAPI@1` | Public facade contracts; install/probe are methods, not import |
| **platform handshake** | `ipfs_datasets_py.logic.platform.manifest` | `LogicPlatformManifest@1` | Declaration maps only |
| **supervisor adapter** | `ipfs_accelerate_py.agent_supervisor.proof.canonical_logic_adapter` | `SupervisorCanonicalLogicAdapter@1` | Lazy datasets import; no top-level datasets load |
| **supervisor provider facade** | `ipfs_accelerate_py.agent_supervisor.proof.logic_provider_contract` | `SupervisorLogicProviderFacade@1` | Lazy datasets load on conversion only |

Planned platform service modules from LPC-060 (`logic/platform/contracts.py`,
`catalog.py`, `discovery.py`, …) inherit the same pure-data rule when extracted.
Until extraction, the table above is the live enforcement surface.

## Forbidden import side effects (fail-closed)

Importing any pure-data module above **must not**:

| Forbidden effect | How the gate detects it |
| --- | --- |
| Import solvers | `z3`, `cvc5`, `pysmt`, `z3py` absent from `sys.modules` after import |
| Install packages | no `pip`/`ensurepip` load; no installer registry/plugin execution |
| Open the network | audit hook rejects `socket.connect` |
| Start processes | audit hook rejects `subprocess.Popen`; `os.system` / `posix_spawn*` patched |
| Mutate files | audit hook rejects write opens and `os.mkdir` / `remove` / `rename` / `replace` |
| Probe hardware | no `psutil`, `pynvml`, `GPUtil`, `cpuinfo` load; no NVML/CUDA probe modules |
| Change environment variables | `os.environ` snapshot equal before and after import |

## Hermetic import policy (test harness)

The enforcement test imports each pure-data module in a **fresh interpreter**
with:

| Control | Value / behavior |
| --- | --- |
| `IPFS_DATASETS_AUTO_INSTALL` | `0` (opt out of package-root auto-install enablement) |
| `IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS` | `0` |
| `IPFS_DATASETS_PY_MINIMAL_IMPORTS` | `1` |
| `IPFS_KIT_AUTO_INSTALL_DEPS` | `0` |
| `PYTHONDONTWRITEBYTECODE` | `1` |
| Audit hooks | fail on write opens, `socket.connect`, `subprocess.Popen`, path mutation |
| `sys.modules` denylist | solvers, pip/ensurepip, installer registry, hardware probes |

Under those opt-outs, `os.environ` must be bit-identical before and after import.
Pure-data modules must not add further environment mutations of their own.

## Allowed on import

* Import of the standard library and pure data contracts.
* Import of other pure-data logic modules (identity namespaces, sealed catalog
  layers, typed request/response dataclasses).
* Construction of in-memory constants, enums, frozen maps, and sealed snapshots.
* Lazy attribute binding that still only loads pure-data leaves.

## Explicit runtime (not pure-data)

These surfaces may perform probes, installs, process starts, or network only
when a named operation is invoked with explicit caller intent:

| Surface | Module / entry | Opt-in operations |
| --- | --- | --- |
| Installation boundary | `verification_api.install_provider` / `probe_provider` | `allow_install=True`, dry-run/offline stay non-mutating |
| Toolchain registry | `logic.backends.toolchains` | install authorization helpers; import itself is pure |
| Lazy installers | `logic.backends.installers.*`, `external_provers.lazy_installer` | explicit ensure/install entrypoints |
| Solver backends | `logic.backends.z3`, `cvc5`, SMT/ATP runners | `check` / `prove` execution paths |
| Process runner | `logic.backends.process` | bounded tool execution |
| Platform execution | planned `logic/platform/execution.py` | `check`, `monitor`, `run_portfolio`, `execute_proof_plan` |
| Platform installation | planned `logic/platform/installation.py` | sole mutation boundary after LPC-060 extraction |

`install_is_forbidden_on_import()` and
`registry_side_effect_free_on_import()` on the toolchain registry remain
true under pure-data import.

## Supervisor adapter rules

`SupervisorCanonicalLogicAdapter@1` and `SupervisorLogicProviderFacade@1`:

1. **Never** top-level import `ipfs_datasets_py`.
2. Load datasets packages only for explicit conversion, registry discovery, or
   revision check.
3. Importing the adapter must not start processes, open sockets, install
   packages, mutate files, probe hardware, or change environment variables.
4. Supervisor retains scheduling, isolation, leases, placement, and
   single-flight; datasets retains family, protocol, evidence, and receipt
   identity.

## Composition with LPC-060

| LPC-060 pure-data service | Live pure-data stand-in until extraction |
| --- | --- |
| `platform/contracts` | `verification_api` contract types + `syntax_core.contracts` |
| `platform/catalog` | `families.canonical_catalog` |
| `platform/discovery` | declarative `list_*` methods on the verification facade |
| `platform/formalization` | `logic.formalization` package root |
| `platform/providers` (declarative) | `protocol_v2` + `provider` + toolchain descriptors |
| `platform/manifest` | `platform.manifest` (already shipped) |
| `platform/service` composition import | must stay composition-only (no probe) |

LPC-060 invariant 3 (pure-data vs explicit runtime) is enforced by this task's
test, not by documentation alone.

## Non-goals

* Does not extract platform service modules (LPC-060 layout note only).
* Does not thin `logic.api` / `logic.__init__` facades (LPC-062).
* Does not rewrite installers, solvers, or execution backends.
* Does not claim provider availability or proof authority from successful
  pure-data import.

## Acceptance check

Importing contracts, catalog, syntax, formalization, provider protocol, and
supervisor adapter does **not**:

* import solvers
* install packages
* open the network
* start processes
* mutate files
* probe hardware
* change environment variables

Evidence: `python -m pytest ipfs_datasets_py/tests/unit/logic/test_pure_data_import.py -q`
