# DuckDB + Quack Compatibility Profile

Status: foundation capability contract for the DuckDB/Quack control plane  
Interfaces: `QuackCompatibilityProfile@1`, `QuackCapabilityReport@1`  
Implementation: `ipfs_accelerate_py/agent_supervisor/task_sources/quack_capabilities.py`

## Purpose

Parallel supervisors share one state-owner only when every server and client
process agrees on the DuckDB build, Quack extension identity, and required
protocol surfaces. This document defines the pinned profile and the fail-closed
probe that admits or rejects a host without launching an uncontrolled service.

Capability discovery is weaker than serving. A green import of `duckdb` is not
authority, not readiness, and not a health pass.

## Pinned profile (`QuackCompatibilityProfile@1`)

| Field | Pinned value | Notes |
| --- | --- | --- |
| Profile id | `agent-supervisor-duckdb-quack-1.5` | Stable admission key |
| DuckDB minor | `1.5.x` | Major 2.x needs a separate pin and restore rehearsal |
| Extension | `quack` | Community/beta extension in the 1.5 era |
| Extension API | `quack@1` | Internal adapter boundary |
| Required functions | `quack_serve`, `quack_query` | Server start and client query entrypoints |
| Required surfaces | INSTALL/LOAD policy, `quack_serve`, `quack_query`, `ATTACH`, `whoami`, auth settings, logging, extension fingerprint | Evidence subset for DQP-004 |
| Optional exact pins | `pinned_duckdb_version`, `pinned_extension_fingerprint`, `pinned_platform` | Empty means “any attested build in the minor” |

Package installation and wheel pins land in dependency work (DQP-005). This
profile only gates probe outcomes and documents the runtime contract.

## Probe statuses

The probe returns exactly one of:

| Status | Meaning |
| --- | --- |
| `unavailable` | DuckDB cannot be imported, connected, or versioned; or the extension catalog is missing |
| `unsupported` | DuckDB version is outside the program’s support window (for example pre-1.4 or unpinned 2.x) |
| `install-required` | DuckDB is usable but the Quack extension is not installed locally |
| `load-required` | Quack is installed but not loaded, or local `LOAD` failed |
| `compatible` | Pinned minor, required functions/surfaces, and optional fingerprint/platform pins all match |
| `mismatched` | A version, platform, fingerprint, or function set disagrees with the pin |
| `experimental` | Quack surfaces work but only under experimental admission (off-minor supported build, community/unsigned origin, or non-matching optional fingerprint inside the experimental policy) |

`passes_health_check` is true only for `compatible`. Experimental hosts need an
explicit experimental admission path. Import success alone never passes.

## INSTALL / LOAD policy

Ordinary health checks call `quack_health_check()` / `probe_quack_capabilities()`
with:

- `allow_network_install=False` (default and mandatory for ordinary health)
- `allow_local_load=True` by default so an already-installed extension may be
  `LOAD`ed in-process without network access

Rules:

1. **Network install is never implicit.** `INSTALL quack` runs only when the
   caller sets `allow_network_install=True`. The report records
   `network_install_allowed` and `network_install_attempted`.
2. **Local load is not install.** `LOAD quack` uses a local install path only.
3. **Import is insufficient.** Even after `import duckdb` succeeds, the probe
   continues through catalog, load, function, surface, and fingerprint checks.
4. **No uncontrolled service.** The probe uses an in-process `:memory:`
   connection (or an injected factory). It does not bind ports or start
   `quack_serve`.

## Required surfaces and evidence

| Surface | How the probe treats it |
| --- | --- |
| INSTALL/LOAD policy | Always recorded in report details; network install remains opt-in |
| `quack_serve` / `quack_query` | Required functions via catalog or bounded existence probes |
| `ATTACH` | Observed when loaded server/client functions are present |
| `whoami` | Observed with the loaded transport surface (server identity checks belong to the state-owner task) |
| Auth settings | Queried from `duckdb_settings()` when available; implied by serve surface otherwise |
| Logging | Same settings probe; Quack’s own log is not the authoritative audit ledger |
| Extension fingerprint | `sha256:` digest over DuckDB version, extension origin/version/path, platform, and function set |

## Quack beta limitations

Quack is beta/experimental on DuckDB 1.5.x. Compatible admission still records
these limitations on every report that evaluates the extension path:

- protocol names and defaults may change before DuckDB 2.0
- server and clients must use an identical pinned build behind the internal adapter
- the default authorization callback permits every authenticated query
- there is no server push; clients poll with cursors and jitter
- one Quack server is one failure domain
- loopback bind is required unless a separately reviewed remote policy exists
- community/unsigned extension origin is not attested binary integrity

LLM-generated code and implementation-provider subprocesses never receive a
Quack token or arbitrary-SQL capability. That boundary is enforced by later
server/client tasks; this probe only establishes version and surface readiness.

## Upgrade and restart expectations

| Change | Expected outcome |
| --- | --- |
| DuckDB patch within pinned `1.5.x` without fingerprint pin | Still `compatible` when functions/surfaces hold |
| DuckDB minor change inside 1.x but off `1.5` | `experimental` if still supported and policy allows; otherwise `mismatched` / `unsupported` |
| DuckDB 2.x | `unsupported` until a new profile is pinned and rehearsed |
| Extension missing after upgrade | `install-required` (no silent network install) |
| Extension present but API symbols moved | `mismatched` |
| Platform or fingerprint pin miss | `mismatched`, or `experimental` only when the profile explicitly allows it |

Restart the state-owner after any DuckDB or Quack binary change. Clients must
re-probe and refuse mismatched server fingerprints rather than retry through an
LLM path.

## Operator usage

```python
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    default_compatibility_profile,
    quack_health_check,
    probe_quack_capabilities,
)

report = quack_health_check()  # never network-installs
assert report.beta_limitations  # always present for the beta profile
if not report.passes_health_check:
    # typed status: unavailable | unsupported | install-required |
    # load-required | mismatched | experimental
    ...

# Explicit network install is a separate operator action, not health:
# probe_quack_capabilities(allow_network_install=True)
```

## Related tasks

- DQP-005 pins installable dependencies for the optional supervisor service
- DQP-006 starts the loopback state-owner only after capability admission
- DQP-007 verifies client/server fingerprints on `ATTACH`
- DQP-038 records fingerprints and compatibility on release receipts
