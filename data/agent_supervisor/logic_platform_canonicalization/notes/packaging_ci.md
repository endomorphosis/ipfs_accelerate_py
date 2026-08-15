# LPC-150 Clean-install and no-sibling packaging tests

**Task:** LPC-150 — Clean-install and no-sibling packaging tests  
**Goal:** LPC-G150 (`LogicPlatformPackaging@1`)  
**Depends on:** LPC-141 (direct-vs-supervisor parity)  
**Track:** packaging  
**Parallel lane:** `lpc-packaging`  
**Resource class:** `cpu-validation`  
**Interface:** `LogicPlatformPackaging@1`  
**Declared output:** `data/agent_supervisor/logic_platform_canonicalization/notes/packaging_ci.md`  
**Board validation:**  
`python scripts/validate_logic_platform_canonicalization_board.py --check-ci`  
**Acceptance:** Alone-datasets, alone-accelerate, compatible together, incompatible
together, no sibling, no Git, no optional solver, and one local solver scenarios
are specified and the hermetic subset passes.

## Purpose

LPC-G150 makes **both** packages testable as independently installed
distributions. LPC-150 owns the **scenario contract and hermetic packaging
gate**: every clean-install / co-install / layout-independence case below is
named, mapped to executable suites, and classified as **hermetic required** or
**opt-in / heavy**. LPC-151 owns the CI job wiring that must fail on failure
(`notes/ci_lanes.md`); this note does not define job YAML.

Both packages under test:

| Package | Distribution name | Primary roots |
| --- | --- | --- |
| datasets | `ipfs_datasets_py` | `ipfs_datasets_py/` (setup + `logic/platform/*`) |
| accelerate | `ipfs_accelerate_py` | repo root `setup.py` / `ipfs_accelerate_py/` |

Semantic compatibility authority is **`LogicPlatformManifest@1`** (LPC-100), not
checkout adjacency, Git provenance, or monorepo layout.

## Lane vocabulary

| Lane | Meaning | CI disposition |
| --- | --- | --- |
| **hermetic packaging required** | In-process / offline / fixture-only; no network; no live wheel publish; no sibling checkout; sealed validation PATH | required; failure blocks LPC-150 |
| **offline wheel (heavy)** | Local `bdist_wheel` + isolated `--target` install without index | required when build toolchain present; not a silent skip |
| **one-local-solver smoke** | One already-supported local solver on PATH (typically `z3`) | required when present; unavailable ≠ packaging pass |
| **opt-in / network** | Index installs, multi-solver portfolio, live prover install | never silent-pass |

Authoritative validation environment (fail-closed):

* `PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`
* Python target `/usr/bin/python3.12`
* Private validation `HOME` (`ipfs-accelerate-validation-home-*`) with sealed XDG paths
* No operator `~/.elan`, shell startup, or provider-only toolchains

## 1. Scenario matrix (LPC-150 acceptance)

Every row is **specified**. Rows marked **hermetic** are the subset that must
pass under the sealed validation environment without network, sibling
checkouts, Git, or optional native solvers.

| # | Scenario token | Hermetic? | Expected outcome | Primary suites |
| --- | --- | --- | --- | --- |
| 1 | **alone-datasets** | yes | `ipfs_datasets_py` pure-data / platform / verification surfaces import and inventory without accelerate source tree, sibling checkout, Git, network, or solver install | `ipfs_datasets_py/tests/unit/logic/test_pure_data_import.py`, `ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py`, `test/packaging/test_logic_verification_clean_install.py` |
| 2 | **alone-accelerate** | yes | Importing supervisor logic adapters/client modules does not load `ipfs_datasets_py`; package discovery of accelerate packaging surfaces succeeds without datasets sibling | `test/api/test_supervisor_logic_platform_client.py` (quiet import), `test/packaging/test_ipfs_accelerate_supervisor_packaging.py` |
| 3 | **compatible-together** | yes | Matching package + interface + catalog + adapter pins produce `HandshakeResult.compatible is True` when both packages are present in-process | `ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py`, `test/api/test_supervisor_logic_platform_client.py` |
| 4 | **incompatible-together** | yes | Version / interface / catalog / adapter mismatches return typed `HandshakeResult(compatible=False, incompatibilities=...)`; structural errors still fail closed | `ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py` |
| 5 | **no-sibling** | yes | Default manifest and handshake succeed with `requires_sibling_repos() is False` and `requires_repository_layout() is False`; layout adjacency is not compatibility authority | `test_manifest.py` (`test_manifest_does_not_require_git_or_siblings`, `test_repository_layout_is_not_compatibility_authority`), clean-install empty-env import |
| 6 | **no-Git** | yes | Default handshake succeeds with `source_commit is None`; provenance is env/metadata only; never shells out to `git` or walks `.git` | `test_manifest.py` (`test_handshake_succeeds_without_source_commit`, `test_optional_source_commit_ignores_git_and_reads_env_only`, `test_resolve_package_version_does_not_need_git`) |
| 7 | **no-optional-solver** | yes | Heavyweight native/external provers are **not** mandatory pip deps; base install / inventory / pure import succeed without them; install-on-import remains forbidden | `test/packaging/test_formal_verification_distribution_contract.py` (`test_optional_native_provers_are_not_mandatory_pip_dependencies`), `test_logic_verification_clean_install.py` (`test_clean_install_does_not_trigger_provider_install`), pure-data import |
| 8 | **one-local-solver** | partial | With one already-supported local solver present (primary: `z3`), offline probe / smoke may exercise it; when absent, status is **unavailable** (not a packaging pass, not a packaging failure) | `test_logic_verification_clean_install.py` (`test_offline_toolchain_probes_respect_lock_and_detect_mismatches`), LPC-142 real-provider smoke note |

### 1.1 Scenario recipes (compact; no golden dumps)

#### S1 — alone-datasets

```text
env: isolated interpreter; PYTHONPATH limited to datasets package root OR
     wheel install target; no monorepo accelerate path; no network; no user site
steps:
  1. import pure-data modules (contracts, catalog, syntax, formalization,
     provider protocol, platform.manifest, verification_api)
  2. build_logic_platform_manifest(include_source_commit=False, environ={})
  3. handshake() → compatible
  4. LogicVerificationAPI declarative inventory (list_providers / capabilities)
assert:
  - no solver module load on pure-data import
  - no pip/ensurepip/subprocess install
  - no socket.connect / network
  - requires_git / requires_sibling_repos / requires_repository_layout all False
  - unavailable optional provers do not break import or inventory
```

#### S2 — alone-accelerate

```text
env: accelerate package importable; ipfs_datasets_py deliberately absent from
     sys.modules and not required for adapter/client import
steps:
  1. import SupervisorCanonicalLogicAdapter / logic_platform_client modules
  2. construct default adapter (lazy datasets boundary)
  3. discover accelerate packaging surfaces (setuptools package list)
assert:
  - no ipfs_datasets_py.* modules appear in sys.modules after quiet import
  - no Git / sibling checkout required to import supervisor packaging surface
  - datasets is loaded only on explicit handshake/catalog/invoke paths
```

#### S3 — compatible-together

```text
env: both packages importable in-process (source tree or co-installed wheels)
steps:
  1. handshake(HandshakeRequirements matching installed manifest pins)
  2. optional: SupervisorLogicPlatformClient handshake then catalog
assert:
  - HandshakeResult.compatible is True
  - incompatibilities == ()
  - catalog_root / interface_versions / compatible_adapter_versions align
  - co-install does not require sibling layout or Git
```

#### S4 — incompatible-together

```text
env: hermetic in-process (no install required)
cases (recipe generator; closed IncompatibilityCode vocabulary):
  - required_manifest_interface = LogicPlatformManifest@99
  - min_package_version / exact_package_version mismatch
  - missing interface_version / schema_root / operation_version
  - required adapter not in compatible_adapter_versions
  - required_catalog_root / required_catalog_digest mismatch
  - require_source_commit without provenance; wrong source_commit
assert:
  - compatible is False
  - each finding has typed code + expected/actual
  - no exception for semantic mismatches (structural errors still raise)
```

#### S5 — no-sibling

```text
env: no sibling repo directories required; empty environ; no layout claims
steps:
  1. build_logic_platform_manifest(...); inspect safety floors
  2. empty-env / wheel-target import that strips monorepo source from sys.path
assert:
  - requires_sibling_repos() is False
  - requires_repository_layout() is False
  - handshake does not consult checkout adjacency
  - installed module file paths resolve under install target, not sibling trees
```

#### S6 — no-Git

```text
env: no .git, no git CLI, empty provenance environ
steps:
  1. optional_source_commit(environ={}) → None
  2. build_logic_platform_manifest(include_source_commit=False, environ={})
  3. handshake() default path
assert:
  - source_commit is None; git_provenance_available is False
  - requires_git() is False
  - default handshake remains compatible
  - resolver never walks .git or shells out to git
```

#### S7 — no-optional-solver

```text
env: base dependency inventory only; no tamarin/lean/coq/isabelle/apalache/…
steps:
  1. machine-check requirements / setup / pyproject inventories
  2. pure-data import + verification API declarative inventory
  3. authorize_provider_install without consent / on import → ToolchainError
assert:
  - FORBIDDEN_MANDATORY_PROVER_DISTRIBUTIONS ∩ inventories == ∅
  - install_is_forbidden_on_import() is True
  - missing optional solvers surface as unavailable, not import failure
  - Python SMT bindings (z3-solver/cvc5) are inventory-declared when required;
    heavyweight native binaries remain lazy/user-local installers
```

#### S8 — one-local-solver

```text
env: sealed PATH; at most one already-supported local solver exercised (z3)
steps:
  1. offline lock probe for tool_id=z3 (no network, no install)
  2. if executable present: version banner vs lock pin; mismatch is detectable
  3. if absent: status=unavailable (does not greenwash packaging or real-provider)
assert:
  - probe.network is False; no download/install during verification
  - unavailable ≠ hermetic packaging failure
  - unavailable ≠ real-provider smoke satisfaction (LPC-142)
  - mocks/fixtures never satisfy the real-provider gate
```

## 2. Hermetic packaging subset (must pass)

### 2.1 Required command (single invocation)

```bash
python -m pytest -q \
  ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py \
  ipfs_datasets_py/tests/unit/logic/test_pure_data_import.py \
  test/packaging/test_logic_verification_clean_install.py \
  test/packaging/test_ipfs_accelerate_supervisor_packaging.py \
  test/packaging/test_formal_verification_distribution_contract.py \
    -k "not clean_isolated_wheel_install and not distribution_build_uses_disposable"
```

Rationale for the `-k` filter: full offline wheel build/install is **offline
heavy** (local `bdist_wheel` + pip `--target`). It remains a first-class LPC-G150
scenario (see §3) but is not required for the hermetic subset that must pass
on every sealed `cpu-validation` runner without a full packaging toolchain
warmup. All eight scenarios above still have hermetic coverage via manifest,
pure-import, inventory, and install-authorization gates.

Board gate for this note:

```bash
python scripts/validate_logic_platform_canonicalization_board.py --check-ci
```

### 2.2 Per-scenario focused validation

| Scenario | Focused validation |
| --- | --- |
| alone-datasets | `python -m pytest ipfs_datasets_py/tests/unit/logic/test_pure_data_import.py ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py test/packaging/test_logic_verification_clean_install.py -q -k "not offline_toolchain_probes"` |
| alone-accelerate | `python -m pytest test/api/test_supervisor_logic_platform_client.py test/packaging/test_ipfs_accelerate_supervisor_packaging.py -q -k "quiet or packaging or import or adapter or client"` |
| compatible-together | `python -m pytest ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py -q -k "compatible or default_handshake or matching"` |
| incompatible-together | `python -m pytest ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py -q -k "incompatible or mismatch or missing"` |
| no-sibling | `python -m pytest ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py -q -k "sibling or repository_layout or does_not_require"` |
| no-Git | `python -m pytest ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py -q -k "git or source_commit or without_source"` |
| no-optional-solver | `python -m pytest test/packaging/test_formal_verification_distribution_contract.py test/packaging/test_logic_verification_clean_install.py -q -k "optional_native or clean_install_does_not_trigger or install_authorization or offline_toolchain_lock"` |
| one-local-solver | `python -m pytest test/packaging/test_logic_verification_clean_install.py -q -k "offline_toolchain_probes or shim_and_version"` |

### 2.3 Hermetic invariants (fail-closed)

| Invariant | Required behavior |
| --- | --- |
| No network | `PIP_NO_INDEX=1`, offline env; no `socket.connect` during pure import / inventory |
| No install-on-import | `install_is_forbidden_on_import()`; installer plugins not executed at import |
| No sibling authority | Manifest floors; handshake ignores monorepo adjacency |
| No Git authority | Default handshake with `source_commit is None` |
| No layout authority | `requires_repository_layout() is False` |
| No optional-solver mandate | Heavyweight provers absent from mandatory pip inventories |
| Typed incompatibility | Semantic mismatches return `HandshakeResult`, do not raise |
| Unavailable ≠ pass | Missing solver does not satisfy real-provider or one-local-solver smoke claims |
| No PYTHONPATH hide | Wheel/install probes must not paper over missing wheel members with source tree paths (distribution contract) |

## 3. Offline wheel / heavy packaging surface (still owned)

These remain part of LPC-G150 evidence but are heavier than the hermetic subset:

| Check | Suite | Notes |
| --- | --- | --- |
| Disposable source stage without mutating checkout egg-info | `test_distribution_build_uses_disposable_source_without_egg_metadata` | build isolation |
| Clean isolated wheel install + Logic API inventory | `test_clean_isolated_wheel_install_imports_and_inventories_logic_api` | offline `bdist_wheel` + `pip install --no-index --target`; strips monorepo from `sys.path` |
| Namespace packages ship | `test_namespace_package_discovery_includes_*` | `find_namespace_packages` includes logic backends / software verification |
| Runtime assets / installer plugins present in wheel | distribution contract member checks | no silent omission |
| Root + datasets dependency inventory machine-checked | `test_root_and_datasets_dependency_inventory_is_machine_checked` | single inventory authority |

Full heavy command (when packaging toolchain is available):

```bash
python -m pytest -q \
  test/packaging/test_formal_verification_distribution_contract.py \
  test/packaging/test_logic_verification_clean_install.py
```

## 4. Package independence rules

| Rule | datasets (`ipfs_datasets_py`) | accelerate (`ipfs_accelerate_py`) |
| --- | --- | --- |
| Alone install | Pure-data + platform + verification API usable without accelerate source | Adapter/client importable without datasets loaded |
| Co-install compatible | Manifest pins match client adapter ids | Client handshakes then invokes through lazy datasets boundary |
| Co-install incompatible | Typed handshake findings | Client must not proceed on incompatible handshake (fail closed) |
| Sibling checkout | Never required for semantic compatibility | Never required for quiet import |
| Git | Optional provenance only | Optional provenance only |
| Optional native solvers | Lazy installers / unavailable | Must not become mandatory root deps for packaging gate |
| One local solver | Offline probe / LPC-142 smoke when present | Supervisor may consume datasets evidence; does not invent solver authority |

## 5. Interface and evidence map

| Interface / lock | Role in LPC-150 |
| --- | --- |
| `LogicPlatformPackaging@1` | This goal's packaging scenario interface |
| `LogicPlatformManifest@1` | Package-neutral identity + handshake (LPC-100) |
| `FormalVerificationPackagingGate@1` | Offline clean-install / empty-env gate |
| `OfflineToolchainLock@1` | Pin + probe policy (`config/formal_verification_toolchains.lock.json`) |
| `FormalVerificationDistributionContract@1` | Wheel/inventory/optional-prover distribution gate |
| `SupervisorLogicPlatformClient@1` | Alone-accelerate quiet import + co-install handshake consumer |
| `LogicRealProviderSmoke@1` (LPC-142) | One-local-solver real path; mocks labeled, never substitute |

### Primary evidence modules

| Path | Scenario coverage |
| --- | --- |
| `ipfs_datasets_py/ipfs_datasets_py/logic/platform/manifest.py` | no-Git, no-sibling, compatible, incompatible |
| `ipfs_datasets_py/tests/unit/logic/platform/test_manifest.py` | hermetic handshake matrix |
| `ipfs_datasets_py/tests/unit/logic/test_pure_data_import.py` | alone-datasets pure import |
| `test/packaging/test_logic_verification_clean_install.py` | alone-datasets empty-env, no-optional-solver install forbid, one-local-solver probes |
| `test/packaging/test_formal_verification_distribution_contract.py` | no-optional-solver inventory, offline wheel, namespace shipping |
| `test/packaging/test_ipfs_accelerate_supervisor_packaging.py` | alone-accelerate package discovery |
| `test/api/test_supervisor_logic_platform_client.py` | alone-accelerate quiet import; co-install client handshake |
| `notes/manifest_handshake.md` | LPC-100 normative handshake rules |
| `notes/real_provider_smoke.md` | LPC-142 one-local-solver real-provider boundary |

## 6. Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-100 | Manifest / handshake is the co-install compatibility authority |
| LPC-061 | Pure-data import hermeticity feeds alone-datasets |
| LPC-110 | Supervisor client quiet import feeds alone-accelerate |
| LPC-140 / LPC-141 | Hermetic conformance + parity remain separate; packaging does not weaken them |
| LPC-142 | Owns real-provider smoke labeling for one-local-solver |
| **LPC-151** | Owns required CI lane definitions and fail-on-failure wiring (`notes/ci_lanes.md`) |

```text
LPC-100 manifest ──┐
LPC-061 pure import ┼──► LPC-150 packaging scenarios (this note)
LPC-110 client ─────┘         │
                              └──► LPC-151 CI lanes fail on failure
```

## 7. What this task does **not** do

* Does not define CI job YAML or `continue-on-error` policy (LPC-151).
* Does not add a new prover, provider identity, or mandatory native solver pip dep.
* Does not treat mocks, metadata-only receipts, or unavailable solvers as real-provider success.
* Does not use `PYTHONPATH` monorepo leakage to hide missing wheel content.
* Does not require Git, sibling checkouts, or monorepo layout for semantic compatibility.
* Does not claim production readiness, coverage percentages, or hardcoded test counts.
* Does not mark LPC-151 complete or edit protected plan/todo/validator files.

## 8. Acceptance (LPC-150)

| Criterion | Where specified |
| --- | --- |
| **alone-datasets** scenario specified | §1 row 1, §1.1 S1 |
| **alone-accelerate** scenario specified | §1 row 2, §1.1 S2 |
| **compatible-together** scenario specified | §1 row 3, §1.1 S3 |
| **incompatible-together** scenario specified | §1 row 4, §1.1 S4 |
| **no-sibling** scenario specified | §1 row 5, §1.1 S5 |
| **no-Git** scenario specified | §1 row 6, §1.1 S6 |
| **no-optional-solver** scenario specified | §1 row 7, §1.1 S7 |
| **one-local-solver** scenario specified | §1 row 8, §1.1 S8 |
| Hermetic subset command listed | §2.1 |
| Hermetic subset passes under focused pytest | §2 / executable suites above |
| Board check admits this note | `--check-ci` (packaging_ci.md present and non-empty) |
| CI lane fail-on-failure deferred correctly | LPC-151 / §6 |

## File ownership

| Path | Role |
| --- | --- |
| `data/agent_supervisor/logic_platform_canonicalization/notes/packaging_ci.md` | This note (LPC-150 sole declared output) |
| Suites in §2 / §5 | Executable packaging coverage (owned by originating packaging/manifest/import tasks; exercised here as evidence) |
| `data/agent_supervisor/logic_platform_canonicalization/notes/ci_lanes.md` | LPC-151 only — required CI lanes fail on failure |
