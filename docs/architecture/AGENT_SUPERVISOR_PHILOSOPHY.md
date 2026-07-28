# Agent Supervisor design philosophy

`ipfs_accelerate_py.agent_supervisor` is the control plane for
**objective-driven, evidence-bounded software work**. It is broader than an
LLM wrapper: models propose plans and edits, but parsers, policy checks,
validation commands, Git isolation, leases, and typed receipts decide whether
work may advance.

This document is the product-level mental model. For operations, see the
[Operator guide](../guides/AGENT_SUPERVISOR_GUIDE.md). For package ownership,
see the [Package map](agent_supervisor/PACKAGE_MAP.md). For a short agent
checklist, see [FOR_AGENTS.md](agent_supervisor/FOR_AGENTS.md).

## What problem it solves

Teams need agents that can:

1. turn durable goals into drainable tasks,
2. implement changes in isolated lanes,
3. validate and merge under explicit policy, and
4. leave enough evidence that a later run can resume, repair, or audit the work.

Without a control plane, agent runs collapse into chat transcripts and ad-hoc
shell scripts. The supervisor makes **intent, authority, isolation, and
evidence** first-class.

## Seven pillars

### 1. Objectives are durable intent; todos are projections

An **objective heap** states goals, parents, evidence expectations, and
acceptance. A **taskboard** is a schedulable projection the implementation
daemon can drain. Regenerating or refining todos must not silently rewrite
protected intent.

### 2. Models propose; policies admit

Model output is a **proposal**. Admission requires allowlisted roots, identity
bindings, deterministic validation, and (where configured) fresh proof or test
evidence. A fluent explanation never upgrades trust.

### 3. Evidence is typed and tiered

Repository queries, tests, runtime observations, static analysis, solver
candidates, kernel-checked proofs, and cryptographic attestations are
**different classes**. The system must not promote a weaker class into a
stronger one by renaming or caching. Cache hits **re-derive** assurance; they
never invent it.

### 4. Isolation by default

Implementation work runs in worktrees/lanes with:

- task sharding,
- leases and fencing,
- protected paths (boards, sealed plans, operator inputs),
- bounded timeouts and retry budgets.

Failure should quarantine or rescue, not corrupt shared state.

### 5. One contract, three transports

Python, CLI (`ipfs-accelerate agent`), and MCP share one operation vocabulary
and the same request/result contracts. Transports differ in **how roots and
allowlists are configured**, not in what operations mean.

### 6. Domain packages encode ownership

Code is organized by **domain packages** (`control`, `proof`, `runtime`, …)
with an acyclic dependency DAG. Packages own modules and READMEs; they are not
named after taskboard prefixes. New features land in the package that owns the
concern.

### 7. Programs layer on the control plane

Self-improvement, codebase-proof, domain layout, catalog work, and similar
efforts are **programs**: boards + objectives + optional modules. They do not
fork a second supervisor. See [PROGRAMS.md](agent_supervisor/PROGRAMS.md).

## Authority ladder

Work advances only by climbing this ladder:

```text
1. Intent        objective / task identity
2. Proposal      model plan or patch
3. Validation    deterministic checks / tests / scope policy
4. Isolation     lease, worktree, protected paths
5. Evidence      receipts, cache-backed proofs, audits
6. Mutation      merge, state update, completion mark
```

Skipping a rung is a bug, not an optimization.

## Discovery vs capability vs proof

| Signal | Means | Does not mean |
| --- | --- | --- |
| Import / discovery manifest | Static operation vocabulary and schemas exist | Providers work |
| Capability report / probe | Configured backend can attempt an operation | Proof succeeded |
| Proof / validation receipt | A specific claim met a stated assurance level | Future claims are free |

Import success is never a capability signal. Capability is never proof.

## Implementation shape

```text
 Objectives / taskboards (programs)
              │
              ▼
    objectives + task_sources
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 planning   context   validation
   proof     prompt
    │         │         │
    └─────────┼─────────┘
              ▼
           control
     (Python / CLI / MCP)
              │
              ▼
    runtime + merge + rescue
    multi-lane implement daemons
```

### Codebase-proof vertical (semantic pipeline)

When the supervisor must **prove properties about code** before trusting edits:

1. **Property catalog** — reviewed properties and assurance expectations  
2. **Claim / evidence contracts** — typed claims, tiers, lifecycle  
3. **Obligation compiler** — bind obligations to cache keys and scopes  
4. **Query API** — open / satisfied / refuted / impact / proof_delta  
5. **Context capsules** — obligation-first prompts; delta retries on cache hits  
6. **Edit packets** — materialize implementable work with validation commands  
7. **Re-proof / invalidation** — lookup-before-provider; fail closed on stale trust  
8. **Policy gates** — efficiency, attestation, and promotion rules  

Module homes are primarily under `proof/`, `context/`, and `planning/` on the
domain-layout tree.

## Design non-goals

- Replacing Git, CI, or human review with model confidence  
- Treating simulated ZK or candidate solver output as production attestation  
- Encoding taskboard ticket IDs into public API names  
- Long-lived compatibility shims that freeze a flat module warehouse  

## How to go deeper

| Need | Document |
| --- | --- |
| Run the supervisor | [Operator guide](../guides/AGENT_SUPERVISOR_GUIDE.md) |
| Extend packages | [Contributor guide](agent_supervisor/FOR_CONTRIBUTORS.md) |
| Agent invariants | [Agent capsule](agent_supervisor/FOR_AGENTS.md) |
| Detailed module map | [Architecture](AGENT_SUPERVISOR_ARCHITECTURE.md) |
| Board prefix glossary | [Programs](agent_supervisor/PROGRAMS.md) |
