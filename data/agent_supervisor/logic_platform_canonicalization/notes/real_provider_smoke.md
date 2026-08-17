# LPC-142 Real Local Provider Smoke Path

**Task:** LPC-142 — Real local provider smoke path  
**Goal:** LPC-G140 (`LogicConformanceMatrix@1`)  
**Depends on:** LPC-140 (hermetic required matrix)  
**Track:** tests  
**Parallel lane:** `lpc-tests-smoke`  
**Resource class:** `cpu-proof-solver`  
**Interface:** `LogicRealProviderSmoke@1`  
**Declared output:** `data/agent_supervisor/logic_platform_canonicalization/notes/real_provider_smoke.md`  
**Validation:** `test -f data/agent_supervisor/logic_platform_canonicalization/notes/real_provider_smoke.md`

## Purpose

LPC-140 freezes the **hermetic required** conformance matrix. That matrix must
pass without network, live prover install, or PATH-dependent toolchains. LPC-142
owns the complementary **real local provider smoke** lane:

1. Exercise **at least one already-supported local provider** end-to-end through
   existing production adapters (no new prover, no new provider identity).
2. Keep **mocks, fixtures, and metadata-only** surfaces **explicitly labeled**.
3. Enforce a fail-closed **real-provider gate**: mock / hermetic-fixture /
   metadata-only / unavailable receipts **never** satisfy the real-provider
   smoke claim, even when they report success-shaped outcomes.

This note is the durable smoke contract for LPC-G140. It does **not** add a
prover, expand the catalog, or weaken hermetic required suites.

## Relationship to LPC-140

| Lane | Task | CI disposition |
| --- | --- | --- |
| Hermetic required | LPC-140 (`notes/test_matrix.md`) | Must pass on sealed validation PATH; no live prover |
| Direct-vs-supervisor parity | LPC-141 | Separate note/suite; semantic identity agreement |
| **Real local provider smoke** | **LPC-142 (this note)** | **Required when the provider is present; skip/unavailable is not a pass** |
| Network / heavy opt-in | outside LPC-G140 gate | Never silent-pass |

LPC-140 non-goals already state that mocks cannot satisfy real-provider gates.
This note is the owned evidence path for that rule.

## Already-supported local provider (primary smoke)

**Primary provider:** `z3`  
**Family:** SMT / software-verification  
**Catalog / toolchain id:** `z3`  
**Production interfaces:**

| Surface | Module / path | Role |
| --- | --- | --- |
| Software-verification backend | `ipfs_datasets_py.logic.backends.z3.compiler.Z3SoftwareVerificationBackend` | Live CLI adapter (`Z3SoftwareVerificationBackend@1`) |
| Typed SMT evidence | `ipfs_datasets_py.logic.backends.smt.execution_v2` (`SMTProviderEvidence@2`) | Request/response, mock rejection, pinned vs hermetic modes |
| Differential SMT | `ipfs_datasets_py.logic.backends.smt.differential` | Shared VCs through Z3/cvc5 |
| Scheduled process tier | `ipfs_datasets_py/tests/integration/logic_providers/test_scheduled_provider_tier.py` | `PinnedProcessAvailabilityReceipt@1` |
| Provider tier manifest | `ipfs_datasets_py/tests/integration/logic_providers/manifest.json` | Declares `z3` as process-backed, scheduled |

`z3` is already registered in the executable provider matrix and the scheduled
provider tier. LPC-142 **reuses** that support; it does not introduce a second
solver identity or a new install path.

**Secondary (same lane, optional when present):** `cvc5` via the same SMT
software-verification and differential surfaces. Presence of cvc5 is not
required to close LPC-142 once `z3` is exercised.

## Real-provider gate (normative)

A result may claim **real-provider smoke satisfaction** only when **all** of the
following hold:

| # | Requirement | Fail-closed if violated |
| --- | --- | --- |
| 1 | Provider id is an **already-supported** local identity (`z3`, optionally `cvc5`) | Unknown / invented provider ids |
| 2 | Evidence lane is **scheduled process** / live pinned solver, not hermetic-only | Hermetic fixture success alone |
| 3 | Record kind is `live_process` or `pinned_binary` (or equivalent live backend run) | `mock`, `metadata_only`, `hermetic_fixture`, `unavailable` |
| 4 | A real subprocess (or live backend runner bound to a resolved executable) executed | Injected fixed runners labeled as mock/hermetic |
| 5 | Command / tool / output identities are bound when the process tier is used | Missing digests or secret-bearing argv |
| 6 | Unavailable provider is **not** reported as passed | Skip-as-pass, `xfail` as green, or silent success |

### Explicit non-satisfaction rules

| Observation | Satisfies real-provider gate? |
| --- | --- |
| Hermetic fixture engine with canned `unsat`/`sat` stdout | **No** (labeled hermetic) |
| `SmtExecutionMode.MOCK` or `mock_output=…` | **No** (`MOCK_REJECTED`) |
| `SmtExecutionMode.FALLBACK` or `fallback_output=…` | **No** (`FALLBACK_REJECTED`) |
| `metadata_only` availability receipt | **No** |
| `unavailable` receipt when binary missing | **No** (typed gap, not a pass) |
| Availability flag / confidence / fluent text alone | **No** |
| Live `Z3SoftwareVerificationBackend()` with `z3` on PATH producing bound verdict | **Yes** |
| Scheduled-tier probe of `z3` with `record_kind=pinned_binary` and `executable_capability=true` | **Yes** (availability smoke) |
| Live portfolio `SolverPortfolio` invocation of allowlisted `z3` | **Yes** |

Canonical enforcement anchors already in-tree:

* `establishes_executable_capability(...)` rejects mock / metadata / hermetic /
  unavailable record kinds even when `execution_claimed=True`
  (`test_scheduled_provider_tier.py`).
* `SmtExecutionEngineV2` rejects mock and fallback before solver launch
  (`execution_v2.py`; `test_smt_execution_v2.py`).
* `non_authoritative_signal_establishes` / `mock_or_fallback_establishes_satisfiability`
  always return `False` for authority claims.
* Supervisor exact-module conformance rejects fixture-only providers
  (`test/api/test_agent_supervisor_ipfs_datasets_logic_conformance.py`).

## Labeled mock and non-real surfaces

These surfaces are **required** for hermetic CI and adversarial coverage. They
are **labeled** and **do not** close LPC-142.

| Label | Example | Lane | Real-provider gate |
| --- | --- | --- | --- |
| **MOCK** | `SmtExecutionMode.MOCK`, `mock_output={"status":"proved"}`, `mock_receipt(...)` | hermetic / adversarial | **never** |
| **HERMETIC_FIXTURE** | `hermetic_engine(z3_stdout="unsat\n…")`, `_fixed_runner("unsat\n…")` with version `z3-mock` / `z3-hermetic` | hermetic required | **never** |
| **FALLBACK** | `SmtExecutionMode.FALLBACK`, `fallback_output=…` | fail-closed | **never** |
| **METADATA_ONLY** | `metadata_only_receipt("z3")` | discovery only | **never** |
| **UNAVAILABLE** | missing `z3` on sealed PATH → typed unavailable receipt | scheduled process | **never** (gap, not pass) |
| **LIVE / PINNED** | default `Z3SoftwareVerificationBackend()`, `probe_scheduled_provider(z3)`, live portfolio | real-provider smoke | **candidate for gate** when executable present |

Naming convention used in suites (must remain visible in test code):

* Injected runners: `solver_version="z3-mock"`, `"cvc5-mock"`, `"z3-hermetic"`,
  `"mock-solver/1.0"`.
* Request source refs: `source:fixture:smt:…` for hermetic recipes.
* Receipt ids: `receipt:mock:…`, `receipt:hermetic:…`, `receipt:metadata:…`,
  `receipt:process:z3`, `receipt:unavailable:…`.

## Smoke path (existing local provider tests)

LPC-142 owns the smoke **note** and points at **existing** local-provider tests.
Do not add a new prover. Prefer one of the following already-landed paths.

### Path A — Live Z3 software-verification backend (primary)

**Suite:**
`ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py`

| Test | Provider | Label |
| --- | --- | --- |
| `test_live_z3_software_verification_backend_alone` | **real `z3`** | LIVE (gated by `shutil.which("z3")`) |
| `test_live_z3_cvc5_agree_on_reviewed_vc_fixture` | real z3+cvc5 | LIVE (both required) |
| `test_z3_software_verification_backend_interface_and_typed_theorem` | injected runner | **MOCK / hermetic** (`solver_version="Z3 version mock"`) |
| `test_differential_agreement_on_reviewed_theorem_fixture` | injected runners | **MOCK / hermetic** |

Reviewed obligation (same recipe as live and hermetic):

* `obl:vc-x-positive` — assume `x >= 1`, prove `x > 0` by negation (`unsat` core
  containing `assume_ge_one`).

**Focused live command (when `z3` is on PATH):**

```bash
python -m pytest -q \
  ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py \
  -k 'live_z3_software_verification_backend_alone'
```

**Expected LIVE outcome:**

| Field | Expected |
| --- | --- |
| `backend.is_available()` | `True` |
| `outcome.verdict` | `unsat` (`SmtSolverVerdict.UNSAT`) |
| `outcome.result.status` | `PROVED` |
| `outcome.solver_version` | non-empty string from real `z3 --version` |
| `outcome.compilation.receipt` | bound translation receipt |

**Expected when `z3` is absent:** the live test is skipped. Skip is **not** a
real-provider pass. Record disposition `unavailable` for the smoke lane.

### Path B — Real solver portfolio (Hammer allowlist)

**Suite:**
`ipfs_datasets_py/tests/integration/logic/hammers/test_solver_portfolio.py`

| Class / test | Provider | Label |
| --- | --- | --- |
| `TestZ3RealInvocation::test_asserted_tautology_is_satisfiable` | **real `z3`** | LIVE |
| `TestZ3RealInvocation::test_asserted_contradiction_is_unsatisfiable` | **real `z3`** | LIVE |
| Unit portfolio with fake process runner | simulated | **MOCK** (unit suite only) |

**Focused live command:**

```bash
python -m pytest -q \
  ipfs_datasets_py/tests/integration/logic/hammers/test_solver_portfolio.py \
  -k 'TestZ3RealInvocation'
```

**LIVE acceptance checks already asserted by the suite:**

* `record.verdict` is `SAT` / `UNSAT` from the real subprocess, never fabricated
  verified status.
* `record.solver_version` is non-empty (probed from the executable).
* `evidence.command[0] == shutil.which("z3")`.
* Raw stdout/stderr digest matches the attempt record.

### Path C — Scheduled process-backed availability tier

**Suite:**
`ipfs_datasets_py/tests/integration/logic_providers/test_scheduled_provider_tier.py`  
**Manifest:** `ipfs_datasets_py/tests/integration/logic_providers/manifest.json`

| API | Role |
| --- | --- |
| `probe_scheduled_provider(entry)` | Resolve executable on sealed PATH, run probe argv, emit receipt |
| `run_scheduled_provider_tier()` | Probe every scheduled process-backed provider |
| `mock_receipt` / `hermetic_fixture_receipt` / `metadata_only_receipt` | **Labeled non-real** controls |

Sealed validation PATH used by the harness (matches LPC admission policy):

```text
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
```

**Focused command:**

```bash
python -m pytest -q \
  ipfs_datasets_py/tests/integration/logic_providers/test_scheduled_provider_tier.py \
  -k 'scheduled_tier_probes or mock_never or hermetic_fixture_never or metadata_only_never or hermetic_and_scheduled'
```

**LIVE z3 receipt fields when binary present:**

| Field | Value |
| --- | --- |
| `provider_id` | `z3` |
| `lane` | `scheduled_process` |
| `record_kind` | `pinned_binary` |
| `available` | `true` |
| `execution_claimed` | `true` |
| `executable_capability` | `true` |
| identity digests | `command_digest`, `environment_digest`, `tool_digest`, `output_digest` |

**MOCK control (must remain false for executable capability):**

| Field | `mock_receipt("z3")` |
| --- | --- |
| `record_kind` | `mock` |
| `executable_capability` | `false` |
| gate | cannot construct receipt with `executable_capability=true` |

### Path D — Typed SMT mock rejection (negative control)

**Suite:**
`ipfs_datasets_py/tests/integration/logic_providers/test_smt_execution_v2.py`

| Test | Label | Gate contribution |
| --- | --- | --- |
| `test_mock_output_cannot_establish_authority` | **MOCK** | disposition `MOCK_REJECTED`; no sat/theorem/proof |
| `test_fallback_output_cannot_establish_authority` | **FALLBACK** | disposition `FALLBACK_REJECTED` |
| `test_differential_agree_proved_binds_core_and_replay` | **HERMETIC_FIXTURE** | hermetic success only; not real-provider |
| `test_single_solver_z3_and_cvc5_paths` | **HERMETIC_FIXTURE** | provider routing under hermetic engine |

These tests prove the negative half of LPC-142 acceptance: mocks are labeled
and cannot satisfy authority or the real-provider gate.

## Minimal operator smoke recipe

When a machine has the already-supported `z3` binary:

```bash
# 1) Resolve the local provider (must be a real executable).
command -v z3
z3 --version

# 2) Live backend smoke (primary LPC-142 exercise).
python -m pytest -q \
  ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py \
  -k 'live_z3_software_verification_backend_alone'

# 3) Negative controls (mocks labeled; must not promote).
python -m pytest -q \
  ipfs_datasets_py/tests/integration/logic_providers/test_smt_execution_v2.py \
  -k 'mock_output_cannot_establish or fallback_output_cannot_establish'

python -m pytest -q \
  ipfs_datasets_py/tests/integration/logic_providers/test_scheduled_provider_tier.py \
  -k 'mock_never_establishes or hermetic_fixture_never or metadata_only_never'
```

Board / note existence gate for this task:

```bash
test -f data/agent_supervisor/logic_platform_canonicalization/notes/real_provider_smoke.md
```

## Sealed validation environment (fail-closed)

Judge provider availability against the **authoritative validation environment**,
not an implementer-side ambient PATH:

| Parameter | Value |
| --- | --- |
| `PATH` | `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin` |
| Python | `/usr/bin/python3.12` |
| `HOME` | private `ipfs-accelerate-validation-home-*` |
| XDG | `$HOME/.cache`, `$HOME/.config`, `$HOME/.local/share`, `$HOME/.local/state` |

If `z3` is absent under that PATH:

* live tests **skip** or emit `unavailable` receipts;
* hermetic required suites (LPC-140) still must pass;
* the real-provider smoke claim is **not satisfied** and must not be reported
  as passed;
* do **not** install a new prover as part of LPC-142; use an approved
  digest-bound deployment outside this task if operators need the live lane.

## Inventory of already-supported local providers (smoke-eligible)

From `tests/integration/logic_providers/manifest.json` process-backed scheduled
entries (subset most relevant to LPC-142 smoke):

| provider_id | family | smoke priority for LPC-142 |
| --- | --- | --- |
| **z3** | smt | **primary** |
| cvc5 | smt | secondary (same SMT path) |
| tlc / apalache | state_model | optional heavy |
| vampire / eprover | atp | optional |
| lean / rocq / isabelle | kernel | optional heavy |
| proverif / tamarin | protocol | optional heavy |

LPC-142 closes when **one** primary already-supported provider (`z3`) is
exercised live **or** the environment correctly records that it is unavailable
without mock substitution. Acceptance language for the goal remains: mocks
never satisfy the real-provider gate.

## Ownership and non-goals

| Owner | Responsibility |
| --- | --- |
| This note | Real-provider smoke contract, mock labeling, gate rules, command index |
| Existing live suites (paths A–C) | Executable exercise of `z3` / scheduled probes |
| Existing mock/hermetic suites (path D + LPC-140) | Labeled non-real controls and hermetic required matrix |
| LPC-140 | Hermetic required buckets; board `--check-test-matrix` |
| LPC-141 | Direct-vs-supervisor parity (separate) |

This task does **not**:

* Add a new prover, solver binary, or provider id.
* Treat hermetic fixture success as real-provider success.
* Allow mock / metadata / availability / confidence to pass the real-provider gate.
* Skip or `xfail` hermetic required suites.
* Claim live usability when the sealed validation PATH lacks the binary.
* Absorb LPC-141 parity or LPC-150 packaging gates.

## Acceptance (LPC-142)

| Criterion | Where satisfied |
| --- | --- |
| At least one already-supported local provider is identified | § primary provider `z3` |
| Existing local provider tests exercise that provider | Paths A–C |
| Mocks are labeled | § labeled mock table; path D |
| Mocks do not satisfy the real-provider gate | § real-provider gate; enforcement anchors |
| Unavailable ≠ passed | § sealed validation environment; skip/unavailable rules |
| No new prover added | Ownership / non-goals |
| Declared output present | this file |

## File ownership

| Path | Role |
| --- | --- |
| `data/agent_supervisor/logic_platform_canonicalization/notes/real_provider_smoke.md` | This smoke contract (LPC-142 sole declared output) |
| `ipfs_datasets_py/tests/integration/logic/backends/test_z3_cvc5_software_verification.py` | Live + labeled mock Z3/cvc5 SV suite |
| `ipfs_datasets_py/tests/integration/logic/hammers/test_solver_portfolio.py` | Real allowlisted solver portfolio smoke |
| `ipfs_datasets_py/tests/integration/logic_providers/test_scheduled_provider_tier.py` | Process-backed availability receipts + mock rejection |
| `ipfs_datasets_py/tests/integration/logic_providers/test_smt_execution_v2.py` | Typed SMT mock/fallback rejection |
| `ipfs_datasets_py/tests/integration/logic_providers/manifest.json` | Scheduled provider tier declarations |
| `data/agent_supervisor/logic_platform_canonicalization/notes/test_matrix.md` | Hermetic matrix (LPC-140); references this lane |
