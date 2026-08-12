# DuckDB / Quack Control Plane — Threat Model

Status: normative security baseline for DQP-034 / DQP-G070  
Interfaces: `QuackSecurityPolicy@1`, `QuackChaosReport@1`  
Implementation: `ipfs_accelerate_py/agent_supervisor/validation/quack_chaos.py`  
Companions: [`QUACK_COMPATIBILITY.md`](./QUACK_COMPATIBILITY.md), plan board
`docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_CONTROL_PLANE_PLAN.md`

## Purpose and security claim

Parallel supervisor daemons share **one** DuckDB control plane through a
loopback Quack state-owner. The security claim of this program is narrow and
fail-closed:

1. **Implementation-provider and LLM subprocesses never receive a Quack auth
   token or an arbitrary-SQL capability.**
2. **Unauthorized, cross-root, filesystem, extension, and host-escape
   statements are denied before any engine effect.**
3. **Optimistic same-row conflicts are bounded** by a typed, jittered retry
   policy; they never become open-ended LLM retries.
4. **Stale clients cannot write after server restart, credential rotation, or
   store-generation rotation.**
5. **Live transport tests cannot silently skip** when the pinned capability
   profile claims `compatible`.

A green unit suite, a model assertion, or a prefix-regex filter is not a
security boundary.

## Actors and trust boundaries

| Actor | Trust | Must not be trusted to |
| --- | --- | --- |
| Quack state-owner process | Exclusive owner of `control.duckdb`; holds the auth token in a mode-0600 vault behind an opaque secret handle | Publish raw tokens in status, logs, argv, exports, or provider environments |
| Typed Quack client (`QuackStateClient@1`) | Speaks only closed, parameter-bound statement templates | Interpolate identifiers or run model-supplied SQL |
| State transaction / CAS layer | Enforces generation, fence, revision, and idempotency | Auto-refresh non-retryable stale generations into success |
| Implementation provider / LLM subprocess | Untrusted code generator | Obtain Quack tokens, ATTACH the control plane, INSTALL extensions, or register Python UDFs |
| Operator / maintenance lease holder | May checkpoint, backup, restore, and rotate generation under explicit lease | Perform direct-file maintenance while ownership is live or unknown |
| Capability probe | Read-only readiness sensor | Grant authority, bind ports, or network-install extensions by default |
| Chaos harness (`QuackChaosReport@1`) | Independent evidence producer | Weaken policy, skip live gates under a compatible pin, or treat hermetic success as remote TLS proof |

Trust boundary diagram:

```text
  provider / LLM process
           |  (no token, no SQL)
           v
  supervisor control / typed client  ----templates only---->
           |                                         Quack
  secret handle only                          state-owner
           |                                         |
           v                                         v
  provider-safe environment              control.duckdb (exclusive)
```

## Protected assets

| Asset | Classification | Protection |
| --- | --- | --- |
| Quack auth token bytes | Secret | Token vault; opaque `handle:…` only on public surfaces; provider env deny-list |
| Secret handle | Public reference | May appear in status; never resolves outside the owner process |
| `control.duckdb` | Authoritative state | Exclusive owner lease, process-birth fencing, generation/fence epochs |
| Schema fingerprint / store UUID | Integrity | Ready requires live match; client attach verifies identity |
| Task / lease / event rows | Authoritative mutations | CAS revision + idempotency keys |
| Backups / restore receipts | Integrity evidence | Independent verification; restore rotates generation |
| Denial logs | Security telemetry | Digest + closed reason codes; no raw secret-bearing SQL |

## Assumptions

1. The host OS enforces file permissions on the state directory and token file
   (mode `0600` / owner-only).
2. Loopback (`127.0.0.1` / `::1` / `localhost`) is not reachable by untrusted
   remote principals on the deployment. Remote bind is unavailable by default.
3. DuckDB/Quack builds on server and clients match the pinned
   `QuackCompatibilityProfile@1` (see compatibility doc).
4. Quack beta default server authorization may permit every *authenticated*
   query; therefore the **typed client template registry** is the primary SQL
   authorization boundary for supervisor code paths.
5. One Quack server is one failure domain (not multi-primary HA).

## Threats, controls, and residual risk

### T1 — Token theft via provider or log surfaces

| | |
| --- | --- |
| **Threat** | An implementation provider subprocess, crash log, status export, or environment dump exposes the Quack token. |
| **Attacker** | Compromised model output, local operator error, or log aggregator. |
| **Control** | `TokenVault`, `sanitize_for_export`, `provider_safe_environment`, secret scan in chaos suite. Tokens never appear in argv. |
| **Evidence** | `token_isolation`, `provider_env_isolation`, `secret_scan` scenarios. |
| **Residual** | Root on the host can read the vault file; physical host compromise is out of scope. |

### T2 — Arbitrary / model-supplied SQL

| | |
| --- | --- |
| **Threat** | LLM-generated SQL mutates or exfiltrates control-plane state. |
| **Control** | `QuackStateClient.execute_sql` always raises; only named templates bind parameters; `QuackSecurityPolicy` denies free SQL, multi-statement, and comment smuggling before effect. |
| **Evidence** | `raw_sql_rejection`, `statement_authorization`. |
| **Residual** | A privileged operator with embedded DuckDB file access under a maintenance lease can run SQL outside the client — that path is explicit maintenance, not the daemon write path. |

### T3 — Filesystem, cross-root, and extension escape

| | |
| --- | --- |
| **Threat** | `ATTACH`, `COPY`, `read_csv_auto`, `INSTALL`/`LOAD`, or path traversal reaches `/etc`, sibling roots, or untrusted extensions. |
| **Control** | Parsed verb/surface policy (`forbidden_verbs`, file-path, cross-root, extension surfaces). Template registry rejects admin DDL/DCL shapes. Network install is never implicit. |
| **Evidence** | `forbidden_surface` (fail-before-effect). |
| **Residual** | OS-level mounts and compromised DuckDB builds are outside this policy. |

### T4 — Python UDF / host code escape

| | |
| --- | --- |
| **Threat** | `CREATE FUNCTION` / `python_eval` style surfaces execute model code in-process. |
| **Control** | Policy `deny_python_udfs` with explicit limitation statement; chaos scenario `python_udf_limitation`. |
| **Evidence** | Denied authorization decision with `effect_attempted=false`. |
| **Residual** | Future DuckDB features that introduce new host-escape verbs require policy revision before admission. |

### T5 — Non-loopback exposure / missing TLS

| | |
| --- | --- |
| **Threat** | Quack listens on a routable interface without review or TLS. |
| **Control** | Default bind admission is loopback-only. Remote binds require a separately reviewed `RemoteBindPolicy` (unavailable by default). Policy records that **loopback does not terminate TLS**; remote deployment must add TLS termination, OS isolation, credential rotation, and parsed statement authz. |
| **Evidence** | `loopback_bind`, `tls_boundary`. |
| **Residual** | Shared-kernel multi-tenant hosts may still share loopback; deploy with process and user isolation. |

### T6 — Same-row write races and unbounded retry

| | |
| --- | --- |
| **Threat** | Concurrent claimers thrash or double-apply side effects. |
| **Control** | Optimistic CAS on task revision; idempotency keys; bounded jittered `RetryPolicy` (`max_attempts` sealed). Four-client non-conflicting commits and hot-row conflict scenarios. |
| **Evidence** | `four_client_concurrency`, `hot_row_conflict`, `retry_jitter`, `lost_reply_idempotency`. |
| **Residual** | Pathological host scheduling can exhaust attempts; callers must surface conflict, not invent a second authority. |

### T7 — Stale client after restart / rotation

| | |
| --- | --- |
| **Threat** | A client with a pre-restart generation or pre-rotation fence continues to mutate state (split-brain writer). |
| **Control** | Store generation + fence epoch advance on restart/restore/rotation; `StaleGenerationError` / `CommandOutcome.STALE` fail closed; second owner acquisition fails closed. |
| **Evidence** | `server_restart_stale`, `stale_after_rotation`, `credential_rotation`, `split_brain_ownership`. |
| **Residual** | Clock-independent generation counters are authoritative; wall-clock alone is never a fence. |

### T8 — Silent skip of live gates

| | |
| --- | --- |
| **Threat** | CI marks Quack “compatible” while live chaos/security tests are skipped, producing false assurance. |
| **Control** | `enforce_live_gate`: if the capability profile claims `compatible`, a scenario mode/outcome of `skipped` raises `QuackChaosLiveGateError`. Hermetic execution under non-compatible profiles must be recorded as `hermetic`, not omitted. |
| **Evidence** | `live_gate_policy` + suite invariant `live_gate_enforced`. |
| **Residual** | Operators can still inject a fake “compatible” probe; release verification (DQP-039) must re-probe independently. |

### T9 — Denial / audit loss

| | |
| --- | --- |
| **Threat** | Failed attacks leave no evidence, or denial logs re-embed secrets. |
| **Control** | Denial records use SQL digests and closed reason codes only. |
| **Evidence** | `denial_logging`. |

## Authn / authz defaults (Quack beta)

| Layer | Default | Program stance |
| --- | --- | --- |
| Quack server authentication | Token required for remote clients | Token held only by state-owner vault; clients use handles/transport config that never enter provider env |
| Quack server authorization callback | Beta default may permit every authenticated query | **Not** relied on as the SQL boundary; typed client + policy are mandatory |
| Client statement surface | Closed template registry | Arbitrary SQL API raises |
| Remote bind | Denied | Requires reviewed policy + TLS statement |

A prefix regex over SQL text is **not** the security boundary. Authorization
parses a leading verb and closed surface detectors (multi-statement, comments,
file readers, cross-root paths, extension install/load, Python UDF shapes) and
denies with `effect_attempted=false`.

## Concurrency and restart model

- DuckDB concurrency is optimistic; claims are short CAS transactions.
- Lost replies replay through idempotency keys to the one committed result.
- Retry delays use exponential backoff with full jitter, capped by policy.
- Server restart and restore rotate store generation / fence epoch so stale
  sessions cannot write.
- Credential rotation destroys prior vault material; old handles do not resolve.
- Exclusive ownership prevents two live state-owners (split brain) on one
  database file.

## Chaos evidence population (`QuackChaosReport@1`)

The harness in `quack_chaos.py` executes a closed scenario set, including:

| Scenario | Acceptance link |
| --- | --- |
| `token_isolation` / `provider_env_isolation` / `secret_scan` | No provider/LLM token obtainment |
| `raw_sql_rejection` / `forbidden_surface` / `statement_authorization` | Unauthorized SQL fails before effect |
| `four_client_concurrency` / `hot_row_conflict` / `retry_jitter` | Bounded concurrency and conflicts |
| `lost_reply_idempotency` | Exactly-once command result under reply loss |
| `stale_after_rotation` / `server_restart_stale` / `credential_rotation` | Stale writers fail closed |
| `split_brain_ownership` | Second owner fails closed |
| `loopback_bind` / `tls_boundary` | Transport exposure policy |
| `python_udf_limitation` | Host-escape denial |
| `denial_logging` | Digest-only denials |
| `latency_bound` | Reviewed latency ceiling for hermetic batch |
| `live_gate_policy` | No silent skip under compatible pin |

Validation command:

```bash
python -m pytest -q test/api/test_agent_supervisor_quack_chaos.py
```

## Explicit non-goals

- Public Internet exposure of Quack.
- Multi-primary high availability or automatic failover across hosts.
- Treating vector similarity, model judgment, or export Markdown as authority.
- Inferring DuckDB 2.x safety from a 1.5.x compatible pin.
- Replacing OS identity, filesystem ACLs, or network policy.

## Residual risk summary

| Risk | Severity if control fails | Mitigated by |
| --- | --- | --- |
| Token in provider env | Critical | Provider env filter + chaos secret scan |
| Model SQL on control plane | Critical | Template-only client + policy |
| Stale writer after restart | High | Generation/fence CAS |
| Silent CI skip | High | Live gate under compatible profile |
| Remote bind without TLS | High | Loopback default + remote policy gate |
| Single-server outage | Medium (availability) | Backup/restore (DQP-033); documented non-HA |

## Change control

Changes to `QuackSecurityPolicy` defaults, admitted/forbidden verb sets, live
gate semantics, or token redaction rules require:

1. updating this threat model;
2. extending the chaos scenario population; and
3. a green `test_agent_supervisor_quack_chaos.py` run on the current tree.

Beta limitations from `DEFAULT_QUACK_BETA_LIMITATIONS` remain recorded on every
compatible capability report and must not be silently dropped from chaos
reports.
