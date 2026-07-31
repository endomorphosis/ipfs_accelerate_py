# Formal Verification Tactician Operations Runbook

**Interface:** `FormalVerificationTacticianDocumentation@1`  
**Companion product guide:** `docs/formal_verification_tactician.md`  
**Migration:** `ipfs_datasets_py/docs/logic/proof_tactician_migration.md`

This runbook is the operator contract for day-2 supervision of the goal-directed
formal verification tactician: preflight, lifecycle control, incident response,
failure/rollback states, and recovery. It is fail-closed. Missing evidence is a
blocker, not an implicit pass.

---

## 1. Scope

| In scope | Out of scope |
| --- | --- |
| Goal tactician lifecycle leases, fencing, restart | Editing historical receipts to force green gates |
| Provider/toolchain preflight and quarantine | Silent install from inventory or probes |
| Counterexample incident handling | Treating advisor drafts as proofs |
| Distinguishing implementation completeness from deployment certification | Promoting legal evidence to theorem authority |
| Rollback / demotion of property-provider pairs | Timing ratios as correctness gates |

---

## 2. Roles and surfaces

| Role | Surface | May mutate plans? |
| --- | --- | --- |
| Public consumer | `LogicVerificationAPI` / CLI / MCP | No supervisor controls |
| Supervisor worker | `GoalTacticianSupervisorLifecycle` with valid lease + fencing token | Yes, only while lease is authoritative |
| Operator | This runbook + toolchain certifiers | Policy demotion, cancel, invalidate |
| Read-only auditor | Receipts, baseline JSON, metrics | No |

Public APIs reject: `admit_goal`, `close_plan`, `mutate_supervisor`,
`force_complete`, `lease_steal`, `rewrite_event_log`, `bypass_resource_policy`,
`promote_proof_authority`, and related supervisor-only keys.

---

## 3. Preflight checklist

Run from the repository root before starting long-running tactician plans.

### 3.1 Documentation and import safety

```bash
python scripts/docs/check_agent_supervisor_docs.py
python -m pytest test/api/test_formal_verification_tactician_docs.py -q
python -c "from ipfs_datasets_py.logic.verification_api import LogicVerificationAPI; print(LogicVerificationAPI().list_providers().status)"
```

Expected: docs checker OK; docs contract tests pass; provider listing is
`declarative` without installing tools.

### 3.2 Readiness baseline (implementation ≠ deployment)

```bash
python -m pytest test/api/test_formal_verification_readiness_baseline.py -q
# Inspect machine-specific ladder:
python -c "import json; print(json.load(open('docs/architecture/formal_verification_readiness_baseline.json'))['summary'])"
```

Interpret statuses separately:

| Status ladder value | Operator meaning |
| --- | --- |
| `implemented` / `fixture_tested` | Code and offline tests exist |
| `live_tested` | Bounded live probe succeeded **on this host** |
| `installed` / `usable` | Exact binary/package identity works offline |
| `production_certified` | Hermetic cert + deployment gates — **not** automatic from code |
| `unsupported` / `unavailable` | Disclose; do not fabricate success |

### 3.3 Optional toolchain (opt-in)

```bash
# Declarative lock only (no install):
python -c "import json; print(sorted(json.load(open('config/formal_verification_toolchains.lock.json')).keys())[:20] if False else 'inspect lock file')"

# Certify live matrix when operators intentionally exercise real provers:
python tools/logic/certify_formal_verification_toolchains.py --help
python -m pytest test/integration/test_formal_verification_real_tool_matrix.py -q
```

Never run installers from CI inventory tests. Missing tools must surface as
`unavailable`, not soft success.

### 3.4 Goal-tactician public parity

```bash
python -m pytest \
  ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py \
  test/api/test_goal_tactician_cli_mcp_parity.py -q
```

---

## 4. Lifecycle operations

Module: `ipfs_accelerate_py.agent_supervisor.proof.goal_tactician_lifecycle`

### 4.1 Durable transitions

| Transition kind | Purpose |
| --- | --- |
| `end_goal` | Record formalized / selected end goal |
| `proof_graph` | Persist obligation graph projection |
| `candidate` | Admit candidate under independent validation only as candidate |
| `verification` | Bind verification receipts |
| `counterexample` | Bind open witnesses |
| `closure` | Verifier-backed counterexample closure |
| `completion` | Complete only with adequate fresh leaf + counterexample receipts |
| `control` | Cancel / timeout / backpressure |
| `tree_invalidation` | Tree identity changed; invalidate scoped work |
| `lease_acquire` / `lease_release` | Fenced worker ownership |
| `reconcile` | Restart projection from durable state + journal |

### 4.2 Plan status machine

| Status | Meaning | Allowed next actions |
| --- | --- | --- |
| `open` | Ready or resumed, not actively leased | Acquire lease, start work |
| `running` | Authoritative worker holds lease | Progress transitions, release, control |
| `blocked` | Waiting on control signal, resources, or evidence | Clear control, supply receipts, or fail |
| `completed` | Completion decision accepted | Read-only; no further mutation |
| `failed` | Terminal failure recorded | Incident review; optional new plan |
| `invalidated` | Tree / epoch / policy invalidation | Replan on new tree identity |

### 4.3 Control signals

| Signal | Effect |
| --- | --- |
| `none` | Mutations allowed under lease |
| `cancelled` | Durable cancel; further progress mutations raise control-active errors |
| `timed_out` | Durable timeout fence |
| `backpressure` | Resource/policy pause |

### 4.4 Restart procedure

1. Stop or fence stale workers (invalid fencing token → `StaleWorkerError`).
2. Load durable state file + journal
   (`goal_tactician_lifecycle.state.json`,
   `goal_tactician_lifecycle.journal.jsonl` by default).
3. Call lifecycle **reconcile** so the projection matches committed transitions.
4. Re-acquire a **new** lease for the current fencing epoch.
5. Resume only work whose cache keys still match tree / target / assumptions /
   provider / version / policy / bounds.
6. Reject stale receipts (`stale_tree`, `stale_epoch`, `stale_worker`,
   `inadequate`).

Validation:

```bash
python -m pytest \
  test/api/test_goal_tactician_supervisor_lifecycle.py \
  test/api/test_goal_tactician_supervisor_restart.py -q
```

Restart must replay **identical** authoritative state. Stale workers and stale
receipts cannot close or mutate a plan.

### 4.5 Completion gate

Completion requires:

- all selected graph leaves have adequate **fresh** receipts;
- all open counterexamples are closed with verifier-backed receipts **or**
  explicitly remain open/unknown under policy (never silent close);
- control signal is `none` (or policy-defined terminal cancel that does not
  claim success);
- tree identity matches the plan binding.

Hard blockers: false proof, false closure, secret/witness leakage, authority
escalation, unresolved cross-provider disagreement.

---

## 5. Failure and rollback matrix

Every row is a first-class non-success state. Operators must not rewrite past
receipts; demote policy or replan instead.

| Failure / state | Detection | Immediate action | Rollback / recovery |
| --- | --- | --- | --- |
| `invalid` request | Schema / forbidden control | Reject; fix client | None (no state change) |
| `unsupported` fragment | Compiler / frontend | Disclose; stop claim | Keep declaration; no fabricate |
| `unavailable` tool | Probe / runner | Report unavailable | Demote pair to shadow if enforced |
| Timeout / `timed_out` | Bounds exceeded | Durable control signal | Resume with higher explicit bounds **or** leave unknown |
| Cancel / `cancelled` | Operator or policy | Fence mutations | Replan if still desired |
| Backpressure | Resource policy | Pause leases | Scale resources or reduce concurrency |
| `blocked` plan | Missing evidence / control | Diagnose status projection | Clear control or attach receipts |
| `failed` plan | Terminal error | Incident review | New plan id; keep history |
| `invalidated` | Tree change | Stop workers | Replan on new tree identity |
| Stale worker | Fencing token mismatch | Reject mutation | Acquire new lease |
| Stale receipt | Tree/epoch mismatch | Reject completion | Re-run verification |
| Inadequate assurance | Below required level | Reject completion | Obtain higher-authority receipt |
| Counterexample open | Witness not closed | Keep open | CEGIS loop; close only on fresh matching verifier receipt |
| Unchanged witness loop | Identical failure N times | Stop synthesis cycle | Backoff; policy terminate |
| Cross-provider disagreement | Conflicting receipts | Quarantine pair | Prefer fail-closed quarantine over majority vote |
| Authority mislabel | Advisory reported as theorem | Quarantine | Demote to `declared`; open defect |
| Secret / private witness leak | Public envelope inspection | Quarantine + purge public channel | Rotate credentials; fix boundary |
| False proof / false closure | Adversarial / conformance | Hard-zero gate trip | Immediate demotion; never auto-promote |
| Legal evidence conflict | `legal_compatible=false` | Do not claim formal success from legal lane | Keep legal routing separate from proof plan |
| Cache key mismatch | Identity fields differ | Treat as miss | Recompute; never upgrade authority |
| Deployment cert missing | Completion receipt / baseline | Block production claims | Implementation may still be complete offline |

### 5.1 Rollout demotion (property/provider pairs)

Align with `ipfs_datasets_py/docs/logic/software_verification_rollout.md`:

| Trigger | Demote to |
| --- | --- |
| Tool identity change | `shadow` + invalidate warm cache keys |
| Missing optional tool | keep declaration; report `unavailable` |
| Authority mislabel / false proof | `declared` + quarantine |
| Semantic disagreement | quarantine (fail closed) |
| Operator request | explicit demotion with reason code + timestamp |

Historical receipts remain immutable. Demotion changes only live policy for new
work.

---

## 6. Incident response playbooks

### 6.1 Suspected false proof

1. Freeze promotions for the property/provider pair.
2. Collect receipt ids, tree identity, bounds, tool versions, and public
   counterexample envelopes.
3. Re-verify with `verify_receipt` and independent `check` on the same bindings.
4. If mismatch: increment hard-zero counters, demote to `declared`, open defect.
5. Do not delete historical receipts; mark them non-authoritative for policy.

### 6.2 Counterexample not replaying

1. Confirm envelope schema and public witness fields.
2. Call `replay_counterexample` with **identical** bounds.
3. If bounds changed, treat as new experiment — do not close the original.
4. Minimize with declared guarantee level; never claim global minimality without
   evidence.
5. Feed CEGIS only after independent candidate validation.

```bash
python -m pytest test/api/test_counterexample_guided_tactician.py -q
python -m pytest ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_replay.py -q
```

### 6.3 Worker fencing / dual writers

1. Inspect lease holder and fencing epoch.
2. Revoke or wait out lease (`DEFAULT_LEASE_SECONDS` is 300 unless configured).
3. Reject all mutations from stale tokens.
4. Reconcile state; resume single authoritative worker.

### 6.4 Privacy incident (leakage)

1. Quarantine the public response path immediately.
2. Confirm raw source, credentials, tokens, or private witnesses never appear in
   API/MCP/model context payloads.
3. Run adversarial privacy tests:

```bash
python -m pytest \
  test/security/test_formal_verification_tactician_adversarial.py \
  ipfs_datasets_py/tests/security/logic/test_goal_tactician_adversarial.py -q
```

4. Rotate any exposed secrets; file a hard-zero leakage gate incident.

### 6.5 Legal evidence vs formal plan confusion

If operators treat legal applicability as a software theorem:

1. Stop; reclassify evidence as legal constraint compilation only.
2. Ensure `require_legal_compatibility` results do not set theorem authority.
3. Re-run formal planning without elevating legal permissions to proof success.

---

## 7. Observability and metrics

Module: `goal_tactician_metrics` / benchmark artifact
`docs/architecture/formal_verification_tactician_benchmark.json`.

Record distributions of:

- outcome classes (`succeeded`, `partial`, `unavailable`, `unsupported`,
  `invalid`, `error`);
- authority classes;
- cache cold/warm **identity** hits (not timing correctness);
- resource bound compliance;
- open vs closed counterexamples;
- control signal counts.

Wall-clock and memory may inform capacity planning. Timing ratios must never be
the sole pass/fail gate.

```bash
python -m pytest test/benchmarks/test_formal_verification_tactician_benchmark.py -q
```

---

## 8. Provider and toolchain operations

| Action | Command / surface | Notes |
| --- | --- | --- |
| List declared providers | `LogicVerificationAPI.list_providers()` | No install |
| Probe | `probe_provider(provider_id)` | Bounded, opt-in |
| Install | Explicit installer / `install_provider` | Operator-approved only |
| Certify matrix | `tools/logic/certify_formal_verification_toolchains.py` | Hermetic lanes |
| Lock identities | `config/formal_verification_toolchains.lock.json` | Pin exact versions |

Unsupported languages/tools remain disclosed. Do not expand the promised matrix
in this runbook beyond locked/capability-census rows.

---

## 9. Executable operator smoke (public API)

```python
from ipfs_datasets_py.logic.verification_api import (
    list_goal_tactician_operations,
    formalize_goal,
    proof_status,
)

ops = list_goal_tactician_operations()
assert ops.status.value == "declarative"
assert "plan_proof" in ops.result["operations"]

# Formalize is a proposal path — not plan completion
draft = formalize_goal(
    {
        "prose": "Released leases never remain held.",
        "source_binding": {"path": "example.py", "language": "python"},
        "bounds": {"timeout_seconds": 3},
    }
)
assert draft.result.get("admitted") is not True

# Status without a plan id is non-success, not theorem success
status = proof_status({"plan_id": ""})
print(ops.status, draft.status, draft.authority, status.status)
```

---

## 10. Related validation commands

```bash
python scripts/docs/check_agent_supervisor_docs.py
python -m pytest test/api/test_formal_verification_tactician_docs.py -q
python -m pytest test/api/test_goal_tactician_supervisor_lifecycle.py \
  test/api/test_goal_tactician_supervisor_restart.py -q
python -m pytest test/api/test_counterexample_guided_tactician.py -q
python -m pytest test/api/test_formal_verification_readiness_baseline.py -q
```

---

## 11. Escalation

Escalate when hard-zero gates trip, when restart cannot reconstruct identical
authoritative state, or when deployment certification is requested without a
current-tree completion receipt. Do not close incidents by weakening authority
labels or deleting disagreeing receipts.
