# Architecture Decision Records

This directory is the durable home for **Architecture Decision Records (ADRs)**
in `ipfs_accelerate_py`. ADRs capture *why* a boundary, trust tier, ownership
rule, or operational invariant exists, not a restatement of API docs or delivery
plans.

ADRs are **Current** / **Reference** architecture documentation once accepted.
They explain evidenced design. They do **not** upgrade a plan, proposal, or
roadmap item into implemented status.

## Purpose

Use an ADR when a choice:

- constrains future design or package layout;
- separates trust, authority, or failure domains;
- is likely to be “simplified away” by a later developer or agent;
- has credible alternatives with material trade-offs.

Do **not** use an ADR for:

- temporary delivery sequencing or taskboard status;
- routine bug fixes without design impact;
- restating code that is already the single source of truth without a
  decision to preserve.

Related historical notes (not ADRs) live under
[`docs/development_history/`](../../development_history/) (for example
[`ARCHIVAL_DECISIONS.md`](../../development_history/ARCHIVAL_DECISIONS.md)).
Architecture guides live under [`docs/architecture/`](../) and
[`docs/architecture/agent_supervisor/`](../agent_supervisor/).

## Collision-safe naming

| Rule | Convention |
| --- | --- |
| Filename | `NNNN-kebab-case-short-title.md` |
| Number | Zero-padded four digits (`0001`, `0002`, …) |
| Template | `0000-template.md` (never a real decision) |
| Title slug | Lowercase ASCII, hyphens only, no spaces or underscores |
| Identity | The number is the stable ADR id (`ADR-0003`); the slug may clarify topic but must not collide |

**Allocation rules:**

1. Take the next free number after the highest reserved or published ADR.
2. Never reuse a number after an ADR has been merged, even if rejected or
   superseded.
3. One ADR per file; one primary decision per ADR.
4. Parallel writers own only their numbered file. Do not edit another author’s
   ADR or this index unless your task explicitly owns the path.

## Status vocabulary

Every ADR states exactly one status. Status separates **proposals** from
**evidenced current decisions**.

| Status | Meaning | Normative? |
| --- | --- | --- |
| **Proposed** | Under review; not yet the adopted design. May describe intended change. | No — do not treat as current system law. |
| **Accepted** | Evidenced current decision; matches code, tests, or operational practice at last verification. | Yes — current design authority for its scope. |
| **Deprecated** | Still present in the tree for history, but no longer recommended for new work. | Partial — explain what replaces the practice. |
| **Superseded** | Replaced by a later ADR. Point to the successor. | No — successor is authoritative. |
| **Rejected** | Considered and not adopted. Retained so the alternative is not rediscovered without context. | No — records a negative decision. |

**Rules:**

- Only **Accepted** ADRs describe *current* normative design.
- **Proposed** ADRs must not claim “the system does X” unless X is already true
  independent of the proposal; label planned behavior as planned.
- Moving **Proposed → Accepted** requires evidence (source anchors, verification)
  and an update to `Last verified` / `Date`.
- **Superseded** ADRs set `Superseded-by`; the successor sets `Supersedes`.

## Required metadata

Copy from [`0000-template.md`](0000-template.md). Every ADR includes:

| Field | Required |
| --- | --- |
| Title | Yes |
| Status | Yes (closed vocabulary above) |
| Date | Yes (ISO `YYYY-MM-DD` of status or last material edit) |
| Last verified | Yes for Accepted; optional for Proposed |
| Deciders / authors | Recommended |
| Scope | Yes |
| Non-goals | Recommended |
| Supersedes | When replacing an earlier ADR |
| Superseded-by | When this ADR is no longer authoritative |
| Related guides | Paths to architecture guides that the decision constrains |
| Source anchors | Exact packages, modules, tests, or schemas |

## Document outline

ADRs follow the template section order:

1. **Context** — forces, constraints, and problem that made a decision necessary
2. **Decision** — the choice in normative language for the stated status
3. **Alternatives** — credible options considered (including “do nothing”)
4. **Consequences** — **positive and negative** outcomes; costs are mandatory
5. **Evidence** — source anchors and what was observed
6. **Verification** — how a reader re-checks the decision still holds
7. **Review triggers** — events that force reopening or supersession

## Writing rules

1. **Separate current from planned.** Accepted decisions cite live code and
   tests. Proposals use “we intend / plan” language and stay **Proposed** until
   evidence admits them.
2. **Record negative consequences.** Performance, operability, complexity, and
   developer friction belong in **Consequences**, not only benefits.
3. **Evaluate real alternatives.** At least two alternatives plus the chosen
   path, or an explicit statement that no credible alternative existed and why.
4. **Cite sources.** Prefer repository paths and symbols over narrative
   restatement of APIs.
5. **Do not invent guarantees.** ADRs must not create public API contracts that
   code and tests do not support.
6. **Link both ways when practical.** Guides point at constraining ADRs; ADRs
   point at the guides and packages they constrain. Index finalization for
   navigation may land in a later closeout task; individual ADR authors still
   fill Related guides and Source anchors.

## How to add an ADR

1. Copy [`0000-template.md`](0000-template.md) to
   `NNNN-short-kebab-title.md` using the next free number (after reserved
   program slots if any remain).
2. Set **Status** to **Proposed** until evidence supports **Accepted**.
3. Fill Context, Decision, Alternatives, Consequences (positive *and*
   negative), Evidence, Verification, and Review triggers.
4. Keep edits to that single file unless a task also owns the index.
5. When accepting, update Status, Date, Last verified, and evidence.

## Program reservation (documentation refresh)

IDs **0001–0006** are reserved for the documentation-refresh decision wave.
Later writers own only their numbered file; they do not edit this index solely
to claim a number.

| ID | Planned file | Topic | Producing task | Status |
| --- | --- | --- | --- | --- |
| ADR-0001 | [`0001-objectives-and-task-projections.md`](0001-objectives-and-task-projections.md) | Immutable/durable intent vs regenerable task projections | DOC-015 | Reserved |
| ADR-0002 | [`0002-model-proposals-and-evidence-admission.md`](0002-model-proposals-and-evidence-admission.md) | Models propose; evidence admits; merge ≠ acceptance | DOC-016 | Reserved |
| ADR-0003 | [`0003-capabilities-catalogs-and-routing.md`](0003-capabilities-catalogs-and-routing.md) | Capability, catalog, usage, and routing plane separation | DOC-017 | Reserved |
| ADR-0004 | [`0004-worktrees-leases-and-fencing.md`](0004-worktrees-leases-and-fencing.md) | Worktree isolation, leases, and fencing | DOC-018 | Reserved |
| ADR-0005 | [`0005-mutable-coordination-and-immutable-replication.md`](0005-mutable-coordination-and-immutable-replication.md) | Mutable coordination vs immutable replication | DOC-019 | Reserved |
| ADR-0006 | [`0006-domain-packages-and-compatibility-boundaries.md`](0006-domain-packages-and-compatibility-boundaries.md) | Domain packages and compatibility boundaries | DOC-020 | Reserved |

**Reserved** means the number and filename are allocated; the decision file may
not exist yet. Do not publish a different ADR under these numbers. After a
reserved ADR merges, treat its file status field as authoritative and refresh
this table in the index-owning closeout (or the index-owning task) when that
work runs.

## Index of published ADRs

| ADR | Title | Status | Notes |
| --- | --- | --- | --- |
| [0000](0000-template.md) | Template | n/a | Not a decision; copy only |
| 0001–0006 | See program reservation above | Reserved | Decision wave DOC-015–DOC-020 |

When new ADRs land outside the reserved set, append rows here (or in the
closeout index pass) with number, link, status, and one-line scope.

## Review and supersession

Reopen or supersede an ADR when any **Review trigger** fires, including:

- source anchors diverge from the Decision statement;
- a simpler alternative becomes viable without the recorded negative costs;
- a security, isolation, or trust invariant is weakened;
- the successor design is Accepted under a new number.

Supersession checklist:

1. Author successor ADR (new number) with `Supersedes: ADR-NNNN`.
2. Mark the old ADR **Superseded** and set `Superseded-by: ADR-MMMM`.
3. Update related guides to point at the successor.
4. Keep the old file; do not delete historical ADRs.

## Interfaces

| Interface | Version | Role |
| --- | --- | --- |
| `ArchitectureDecisionRecord` | 1 | Numbered Markdown record under this directory following the template |
| `ADRStatus` | 1 | Closed set: Proposed, Accepted, Deprecated, Superseded, Rejected |

## See also

- Template: [`0000-template.md`](0000-template.md)
- Architecture overview: [`../overview.md`](../overview.md)
- Agent supervisor philosophy (decision themes): [`../AGENT_SUPERVISOR_PHILOSOPHY.md`](../AGENT_SUPERVISOR_PHILOSOPHY.md)
- Documentation refresh plan (target tree): [`../DOCUMENTATION_REFRESH_PLAN_2026_08.md`](../DOCUMENTATION_REFRESH_PLAN_2026_08.md)
