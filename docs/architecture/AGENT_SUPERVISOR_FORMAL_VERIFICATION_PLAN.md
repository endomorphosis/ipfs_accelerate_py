# Agent Supervisor Formal Verification Architecture

Status: capability discovery implemented  
Capability schema: `ipfs_accelerate_py/agent-supervisor/formal-verification-capabilities@1`  
Report version: `1`

## Purpose

The agent supervisor needs to route a proof obligation only to toolchains that
can plausibly service it. Runtime discovery must not confuse “installed” with
“proved,” silently activate simulated cryptography, download optional
dependencies, or make the supervisor package unimportable on a minimal host.

`agent_supervisor.formal_verification_capabilities` supplies that discovery
boundary. It produces a versioned, immutable snapshot for scheduling and
operator diagnostics. The report is routing evidence only. A later proof
attempt must produce its own obligation-bound receipt and independently
verified result.

## Safety invariants

1. A capability probe never imports a provider module, loads model weights,
   invokes a prover, compiles a circuit, downloads an artifact, or generates or
   verifies a proof.
2. `proof_attempted` and `proof_success` are always `false` in a capability
   report, every provider row, and every dependency check.
3. The simulated ZKP backend is always `degraded` and explicitly identified as
   non-cryptographic. It cannot satisfy a cryptographic or attested assurance
   requirement.
4. A real ZKP backend is available only when its executable and its circuit/key
   artifacts are both discoverable.
5. Leanstral output is a candidate. It needs a separate Lean kernel check
   before any later receipt can treat it as checked.
6. Missing optional dependencies are data, not import failures. Each missing
   package, executable, model, or circuit has an explicit reason.
7. Discovery is bounded by both a maximum check count and a wall-clock budget.
   A stalled third-party discovery hook is isolated on a daemon thread and
   cannot block supervisor startup.
8. Reports have a TTL cache. Operators can force a refresh after installing a
   toolchain or clear the cache explicitly.

## Capability matrix

The report always includes these provider families:

| Provider ID | Runtime surfaces | Readiness rule |
|---|---|---|
| `hammer` | Hammer package, built-in graph selector, E, Vampire, Z3, CVC5 | Package plus at least one ATP/SMT executable is available; package alone is degraded |
| `tdfol` | TDFOL package, spaCy, configured spaCy model | Symbolic core can operate without NLP extras, but missing NLP health is reported as degraded |
| `external_provers` | Bridge package, Z3/CVC5 Python bindings, ATP/SMT/Coq executables, SymbolicAI | Bridge plus at least one solver binding or executable is available |
| `lean` | Lean bridge and `lean`/`lake` executable | Both are required for a usable Lean route |
| `leanstral` | Leanstral integration, local weights, Transformers, PyTorch, Lean | Missing local inference or checker dependencies degrades proposal generation/checking; it never implies proof |
| `frame_logic` | F-logic package and ErgoAI executable | Structural in-memory mode is degraded; executable-backed ErgoAI is available |
| `knowledge_graphs` | Knowledge-graph package, spaCy/model, Transformers, NetworkX | Core remains usable with rule-based fallbacks; missing NLP enhancements are degraded |
| `zkp_backends` | Registry, circuit definitions, simulated/Groth16/ProveKit providers, executables, setup artifacts, py-ecc | A real backend needs executable plus artifacts; simulation alone is degraded |

The external-prover row intentionally reports Python bindings separately from
command-line executables. Installing `z3-solver` does not imply a `z3` command
is on `PATH`, and finding a command does not imply that a Python bridge can
import its bindings.

## Report contract

The top-level JSON representation has stable routing fields:

```json
{
  "schema_version": "ipfs_accelerate_py/agent-supervisor/formal-verification-capabilities@1",
  "report_version": 1,
  "generated_at": "2026-07-22T12:00:00Z",
  "duration_seconds": 0.012,
  "probe_count": 39,
  "bounded": true,
  "cache_ttl_seconds": 300.0,
  "overall_status": "degraded",
  "proof_attempted": false,
  "proof_success": false,
  "providers": {
    "lean": {
      "provider_id": "lean",
      "display_name": "Lean 4",
      "status": "unavailable",
      "reason": "Lean prover executable is not installed or configured; no Lean kernel check can run",
      "health": {
        "provider": [],
        "executable": [],
        "package": [],
        "model": [],
        "circuit": [],
        "optional_dependency": []
      },
      "proof_attempted": false,
      "proof_success": false
    }
  }
}
```

The arrays in the abbreviated example contain full
`CapabilityHealthCheck` records in a real report. All six dimensions are
always present even when a provider has no dependency in one dimension:

- `provider`: aggregate provider/backend readiness;
- `executable`: command discovery or an explicit executable override;
- `package`: importable provider modules and required Python bindings;
- `model`: local spaCy or Leanstral weight discovery;
- `circuit`: circuit definitions and configured setup/key artifacts;
- `optional_dependency`: enhancements such as spaCy, Transformers, PyTorch,
  NetworkX, SymbolicAI, and py-ecc.

Checks include status, reason, requirement flag, optional version and location,
metadata, and the invariant false proof fields. `available` means discoverable
for a later attempt; `degraded` means a safe fallback or partial surface
exists; `unavailable` means the named route cannot currently be used;
`disabled` is reserved for an explicitly disabled but installed integration.

## Import and probe boundary

The module depends only on the Python standard library. Package discovery
traverses `importlib.machinery.PathFinder` specs; unlike a dotted-name
`importlib.util.find_spec` call, it does not execute parent package
initializers or import the discovered module.
Distribution versions use `importlib.metadata`. Executable discovery uses
`shutil.which` or a validated explicit path. Artifact checks are bounded
existence checks and never parse secrets, model weights, proving keys, or
verification keys.

The default limits are:

- 2 seconds total wall-clock time;
- 96 dependency checks;
- 300 seconds cache TTL.

`FormalVerificationCapabilityProbe` accepts injected package, executable,
version, environment, and clock functions. This makes deployment matrices
deterministic in tests and lets an embedding application use its own metadata
service without weakening the report contract.

The process-wide entry point is:

```python
from ipfs_accelerate_py.agent_supervisor import (
    probe_formal_verification_capabilities,
)

report = probe_formal_verification_capabilities()
lean = report.provider("lean")
if lean.available:
    # Eligible to plan a proof attempt. No proof has happened yet.
    schedule_lean_attempt()
```

Use `force_refresh=True` after a known environment change, or call
`clear_formal_verification_capability_cache()`. A custom probe instance keeps
an independent cache and supports a deployment-specific
`FormalVerificationProbeConfig`.

## Degradation behavior

The following failures are deliberately non-fatal:

- spaCy absent: TDFOL and knowledge graphs report rule-based/degraded NLP;
- `en_core_web_sm` absent: model health names the missing model weights;
- Z3 or CVC5 bindings absent: binding package health is unavailable even if a
  similarly named executable exists;
- Lean, E, Vampire, Coq, CVC5, Z3, ErgoAI, Groth16, or ProveKit commands absent:
  each executable has its own reason;
- Leanstral weights absent: local model health is unavailable while the
  orchestration package can remain degraded for an injected managed callback;
- Groth16/ProveKit artifacts absent: real backend health remains unavailable;
- provider import hooks raise: the exception type and message become a safe
  unavailable reason;
- discovery exceeds its check/time budget: remaining checks say they were not
  inspected and carry `probe_limited: true`.

None of these cases escapes from capability discovery or prevents importing
`ipfs_accelerate_py.agent_supervisor`.

## Proof-aware scheduling boundary

Capability discovery answers only:

> Is there enough local runtime surface to schedule an attempt on this route?

It does not answer:

- whether a translation preserves the obligation;
- whether premises are complete or trusted;
- whether the solver returned valid/invalid/unknown;
- whether a candidate proof reconstructs in a trusted kernel;
- whether a ZKP binds the expected circuit, statement, witness commitment, and
  verification key;
- whether cached evidence matches the current repository tree and policy.

The scheduler must therefore apply two gates:

1. **Route gate:** use this capability report to reject unavailable routes and
   surface degraded routes to policy.
2. **Evidence gate:** after execution, accept only a versioned proof receipt
   whose obligation, source tree, translator, solver/kernel, policy, and
   resource limits are bound and independently validated.

Availability must never be copied into an evidence or assurance field.

## Operational evolution

Schema changes are additive within report version 1. Removing or changing the
meaning of a field, provider ID, health state, or safety invariant requires a
new schema/report version. Provider version commands or active canary proofs
belong in a separate, opt-in diagnostic lane because executing them has a
larger failure and resource surface than startup discovery.

Future proof contracts and receipts should consume `provider_id` plus the
specific dependency check identities used for routing. They should record a
digest of the capability snapshot as contextual evidence, never as proof
evidence. Scheduling integration should preserve a cached snapshot per worker
environment and refresh it after toolchain installation, worker replacement,
or TTL expiry.

## Validation

The focused API suite covers:

- the complete provider matrix and version fields;
- independent serialization of all health dimensions;
- explicit missing spaCy, model, binding, executable, and artifact reasons;
- safe failure of import hooks;
- partial-provider degradation;
- TTL caching, forced refresh, and cache clearing;
- maximum-check and hard wall-clock bounds;
- ZKP executable-plus-circuit readiness;
- the invariant that discovery never reports proof success.

Run it with:

```bash
PYTHONPATH=ipfs_datasets_py/ipfs_accelerate_py \
python -m pytest \
  ipfs_datasets_py/ipfs_accelerate_py/test/api/test_agent_supervisor_formal_verification_capabilities.py \
  -q
```
