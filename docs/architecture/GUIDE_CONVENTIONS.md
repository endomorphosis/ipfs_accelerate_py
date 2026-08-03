# Architecture guide conventions

**Status:** Current  
**Audience:** Parallel documentation authors, architecture reviewers, and
implementation agents writing or revising maintained architecture guides  
**Scope:** Required structure, vocabulary, source anchors, diagrams, trust and
failure language, verification, and volatile-claim policy for architecture
guides under `docs/architecture/`  
**Non-goals:** This file does not define the ADR template (see DOC-004 /
`docs/architecture/decisions/`), lifecycle labels for non-architecture docs
(see `docs/development/DOCUMENTATION_LIFECYCLE.md` when present), or
subsystem-specific content. It does not establish new public APIs.  
**Last verified:** baseline commit `d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15`
(program baseline 2026-08-03); conventions revalidated against the guide
contract in `DOCUMENTATION_REFRESH_PLAN_2026_08.md` and existing maintained
guides (`overview.md`, `AGENT_SUPERVISOR_PHILOSOPHY.md`,
`agent_supervisor/PACKAGE_MAP.md`, `docs/development/DOCUMENTATION_CURRENT_STATE.md`).

This document is the **ArchitectureGuideContract@1** for parallel writers.
Two independent lanes must produce consistent guides without sharing a draft
file or inventing API guarantees to paper over drift.

---

## 1. Purpose

Architecture guides explain **how the checked-out tree is structured**, **how
control and data move**, and **why that shape exists**. They are maintained
narrative over live code. They are not:

- historical phase summaries or completion reports;
- ticketboards or objective heaps (those are work identity, not product API);
- substitute API references (symbols, flags, and extras come from code and
  executable help);
- a place to invent compatibility shims that do not exist.

When code and prose disagree, **update prose to verified current behavior** or
**label a defect explicitly**. Do not invent an API, flag, or guarantee so an
old sentence remains true.

---

## 2. Required front matter

Every maintained architecture guide begins with a compact metadata block
(Markdown bold lines or a short table). All of the following fields are
**required** unless noted.

| Field | Required | Meaning |
| --- | --- | --- |
| **Status** | Yes | Closed vocabulary: `Current`, `Reference`, `Plan`, `Historical` (see §3). Architecture *guides* that claim to describe live behavior must be `Current`. |
| **Audience** | Yes | Who the page is for (e.g. developers placing code, operators, agents). |
| **Scope** | Yes | What the guide covers in one or two sentences. |
| **Non-goals** | Yes | What the guide deliberately omits, so parallel writers do not expand scope. |
| **Last verified** | Yes | Git commit SHA (full or unambiguous short form), optional date, and what was checked. Prefer the program baseline when the whole guide was revalidated; otherwise the commit used for the last full pass. |
| **Source anchors** | Yes | Repo-relative paths and key symbols that own the claims (see §5). May be a dedicated section immediately after front matter. |
| **Related ADRs / guides** | Recommended | Links to decisions and sibling guides; not a substitute for source anchors. |

Example (copy and adapt):

```markdown
# Inference runtime

**Status:** Current  
**Audience:** Developers and agents tracing request dispatch  
**Scope:** Router lifecycle, provider selection, and fallback for the inference plane  
**Non-goals:** Supervisor scheduling, MCP transport policy, package install  
**Last verified:** `d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15` (2026-08-03); paths and symbols checked against package layout  

## Source anchors

- `ipfs_accelerate_py/ipfs_accelerate.py` — …
- …
```

---

## 3. Status and normative language

### 3.1 Status vocabulary

| Status | Use when | Normative weight |
| --- | --- | --- |
| **Current** | Describes behavior of the checked-out tree | Normative for architecture narrative; still subordinate to code and tests |
| **Reference** | Stable map or glossary that may lag slightly but is intentionally maintained | Supportive; revalidate before relying on volatile claims |
| **Plan** | Intent, design proposal, or sealed plan | Non-normative for runtime behavior unless a Current guide or ADR re-states it as accepted |
| **Historical** | Point-in-time record, cutover evidence, archived design | Context only; never a current API contract |

Plans, completion summaries, and boards do **not** become Current by
proximity or filename. Prefer linking them as intent or history.

### 3.2 Current vs planned language

| Phrase family | Allowed for | Forbidden for |
| --- | --- | --- |
| **is / owns / requires / rejects** | Verified current code paths | Speculative work |
| **must / must not** (in guides) | Invariants enforced by code, policy, or tests | Wish lists |
| **planned / proposed / under DOC-…** | Explicit future work, with owner or task if known | Silent future as present tense |
| **optional / when installed / when configured** | Capability-gated features | Claiming universal availability |
| **compatibility / legacy facade** | Explicitly retained alternate surfaces | Presenting facades as preferred without saying so |

Separate **current behavior** and **planned change** into distinct subsections
when both appear on one page. Never interleave them without labels.

### 3.3 Authority ladder for claims

When sources conflict, prefer in this order:

1. Live source and tests for the claim family  
2. Installed CLI / module `--help` and discovery manifests  
3. Maintained Current architecture guide  
4. ADR accepted for that decision  
5. Plan or historical document  

Filename tokens (`complete`, `final`, `phase`, `todo`) are **not** authority.

---

## 4. Required section contract

Maintained architecture guides follow this outline **where applicable**.
Sections may be renamed for domain vocabulary, but the **content obligations**
must be met. Omit a section only when it is genuinely out of scope, and say so
under Non-goals or in a one-line note.

| # | Obligation | What to cover |
| --- | ---: | --- |
| 1 | Metadata | Status, audience, scope, non-goals, last-verified baseline (§2) |
| 2 | Context and component map | Actors, containers, packages, planes; every box maps to a live package **or** is labelled conceptual |
| 3 | Entrypoints and **Source anchors** | Canonical modules, CLIs, MCP tools, public imports; repo-relative paths (§5) |
| 4 | Flows | Control flow and data flow for the primary journeys (§6) |
| 5 | State and identity | Durable state, IDs, leases, CIDs, caches — as owned by this subsystem |
| 6 | Trust, authorization, **failure semantics** | Who may act, what fails closed, degradation and recovery (§7) |
| 7 | **Rationale** | Why this design exists (§8) |
| 8 | **Alternatives** | What was rejected and what a simpler design would break (§8) |
| 9 | **Consequences** | Positive and negative outcomes of the current shape (§8) |
| 10 | Extension and compatibility | How to extend safely; compatibility and facade boundaries |
| 11 | Operational signals | Logs, metrics, probes, receipts operators can use |
| 12 | **Verification** | Deterministic commands and review checks (§10) |
| 13 | Related ADRs / guides | Links only; no orphan claims |

Short guides (one-screen maps) may compress 4–11 into fewer headings but must
still satisfy audience, scope, status, last-verified baseline, sources, flows,
rationale, alternatives, consequences, failure semantics, and verification.

---

## 5. Source anchors

**Source anchors** are the primary evidence for architectural claims. Rules:

1. Use **repo-relative paths** (`ipfs_accelerate_py/…`, `docs/…`, `test/…`).
2. Name **symbols, modules, or packages** that own the behavior; do not paste
   large source excerpts.
3. Prefer package and module boundaries over board prefixes or ticket IDs.
4. Distinguish **current** paths from **compatibility** or **legacy** paths.
5. If a conceptual box has no live package, label it `conceptual` and point at
   the nearest owning adapter or plan.
6. When a claim depends on optional extras, cite `pyproject.toml` extras or the
   capability probe path, not import success alone.

### 5.1 Anchor table pattern

```markdown
## Source anchors

| Concern | Primary path / symbol | Notes |
| --- | --- | --- |
| Package boundary | `ipfs_accelerate_py/__init__.py` | Public re-exports |
| Inference coordinator | `ipfs_accelerate_py/ipfs_accelerate.py` | … |
| Capability report | `get_instance().get_capabilities(...)` | Runtime, not static |
```

### 5.2 Claim-family defaults

| Claim area | Prefer these anchors |
| --- | --- |
| Package metadata, extras, scripts | `pyproject.toml`, `setup.py`, installed `--help` |
| Python API | package `__init__.py`, export manifests, focused import tests |
| CLI | `cli_entry.py`, `cli.py`, parsers, help tests |
| Inference / routing | `ipfs_accelerate.py`, routers, `model_catalog/`, `endpoint_usage/` |
| MCP / MCP++ | `mcp_server/`, `mcp/` compatibility layer, conformance tests |
| IPFS / P2P | backend router, `p2p_tasks/`, workflow modules, sibling adapters |
| Agent supervisor | domain packages under `agent_supervisor/`, control contracts, runtime/daemon tests |
| Deployment / hardware | installers, Docker/deploy manifests, capability probes |

---

## 6. Flows

Document **control flow** (who decides, admits, mutates) and **data flow**
(payloads, state, evidence) for the guide’s primary journeys.

Rules:

- Name live packages and entrypoints in the same terms as the component map.
- Separate the **inference/data plane** from the **supervisor/control plane**
  unless the guide’s scope is a single plane; coupling belongs at **adapters**,
  not collapsed ownership.
- Mark async, multi-lane, and out-of-process steps explicitly.
- Include the **failure path** for the happy path (timeout, reject, fallback,
  quarantine) or point to the failure section.

Prefer short fenced text sequences or tables over prose walls:

```text
Client / CLI / MCP
        |
   admission / policy
        |
   owning service
        |
   adapter / optional capability
```

---

## 7. Trust, authorization, and failure semantics

### 7.1 Trust and discovery

Preserve repository terminology:

| Signal | Means | Does not mean |
| --- | --- | --- |
| Import / discovery | Vocabulary or module exists | Backend works |
| Capability / probe | Configured path can *attempt* work | Proof or SLA |
| Validation / tests | Deterministic checks passed for a claim set | Kernel proof |
| Proof / attestation | Stated assurance level met for a claim | Future claims free |
| Model prose | Proposal only | Admission or completion |

**Import success is never a capability signal. Capability is never proof.**

### 7.2 Failure semantics (required)

Every Current guide that describes runtime behavior must state:

1. **Fail-closed conditions** — what must not proceed without evidence or
   policy (examples: missing lease, failed validation, protected path).
2. **Degradation** — what happens when optional dependencies are absent
   (disable feature, fallback provider, error surface).
3. **Recovery** — retry, rescue, quarantine, or operator action boundaries.
4. **Non-authoritative signals** — chat logs, cache hits without re-derivation,
   and board status alone do not authorize mutation.

Do not invent silent success paths. Prefer “rejects”, “skips optional path”,
or “surfaces error X” over “handles gracefully” without a mechanism.

---

## 8. Rationale, alternatives, and consequences

These three obligations keep guides from becoming inventory lists.

### 8.1 Rationale

**Rationale** answers *why this shape exists* in the current tree: isolation,
authority separation, optional capabilities, plane boundaries, or performance
constraints. Cite source anchors or ADRs; do not rely on delivery history
alone (“because DOC-…” is not a design reason).

### 8.2 Alternatives

List **real alternatives** that were rejected or would be simpler but break an
invariant (e.g. collapsing control into inference, trusting model output as
completion, single global registry for catalog + auth + transport). For each
alternative, state the **breakage** (trust upgrade, loss of isolation, fake
capability, etc.).

### 8.3 Consequences

Record **positive and negative** consequences of the chosen design (operational
cost, dual entrypoints, facade maintenance, stricter fail-closed UX). Negative
consequences are mandatory; cheerleading is not architecture.

Guides may use headings `## Rationale`, `## Alternatives`, and
`## Consequences`, or a single `## Design rationale` with three subsections.
The words **Rationale**, alternatives, and consequences must remain findable
for review and automation.

---

## 9. Diagram vocabulary (DiagramVocabulary@1)

### 9.1 Medium

- Prefer **fenced monospace** diagrams (readable in plain Markdown and diffs).
- Use **Mermaid** only when the consuming renderer is known to support it and
  the diagram remains useful as source text if Mermaid is ignored.
- Do not embed proprietary binary diagram formats as the only explanation.

### 9.2 Naming

| Element | Rule |
| --- | --- |
| Package / module | Exact live name (`agent_supervisor.control`, `mcp_server`) |
| Conceptual boundary | Label `conceptual` or `logical`; never look like an import path |
| Compatibility surface | Label `compatibility` / `legacy` / `facade` |
| Plane | Use **inference/data plane** vs **supervisor/control plane** consistently |
| Edge | Verb that matches authority (proposes, admits, validates, merges) |

### 9.3 Layout cues

- Top-to-bottom for lifecycle; left-to-right only when layers are peer planes.
- Do not draw trust upgrades as solid arrows (model → completion). Use
  dashed or labelled “proposal” edges.
- Keep one concern per diagram; link out rather than densifying.

---

## 10. Links, examples, and volatile claims

### 10.1 Links

- Prefer **relative links** from the guide’s actual directory.
- Link to maintained Current pages and code trees; archives need explicit
  historical labelling.
- Do not deep-link into generated or vendored trees as if they were product API
  unless that is the documented surface.

### 10.2 Code and CLI examples

- Use **intentional public imports** and current console script names.
- Confirm flags with `--help` in the worktree; do not revive flags from memory
  or old summaries.
- Label prerequisites for optional extras, hardware, network, or provers.
- Network calls and live provider inference are **not** required for doc
  acceptance; mark external examples as structural or optional.

### 10.3 Volatile-claim policy

Do **not** state without date, commit, environment, and reproducible command:

- model counts, catalog sizes, or “N endpoints”;
- benchmark numbers or latency SLAs;
- test suite totals or “all tests pass” as a timeless fact;
- hardware universality (CUDA/ROCm/WebGPU/… as always-on);
- proof or prover availability as a global guarantee.

Prefer capability language: how to discover, probe, and report absence.

### 10.4 No invented API guarantees

Writers **must not**:

- invent public symbols, CLI flags, MCP tools, or extras to satisfy a sentence;
- promote board prefixes or ticket IDs into product API names;
- upgrade discovery or capability into proof by wording;
- treat plans or historical summaries as Current behavior;
- document “will always” for optional integrations.

If the tree cannot support a claim, **narrow the guide** or **record a defect
with anchors** instead of fabricating surface area.

---

## 11. Verification

Every Current architecture guide ends with a **Verification** section (or
equivalent checklist) that another engineer or agent can run without network
inference.

Minimum contents:

1. **Structural** — `git diff --check`; relative links resolve from the file’s
   directory.
2. **Anchors** — listed paths exist; named symbols resolve in those modules
   (import or `rg`/read).
3. **Executable help** — referenced CLI commands appear in `--help` or are
   labelled legacy.
4. **Focused tests** — smallest deterministic tests that protect the described
   contracts (cite paths; do not require the entire suite).
5. **Capability honesty** — optional paths described as optional; no universal
   hardware/provider claims without probes.

Example skeleton (outer fence uses four backticks so an inner command block
can nest safely):

````markdown
## Verification

```bash
git diff --check
test -f path/to/primary_module.py
rg -n 'symbol_or_operation' path/to/primary_module.py
# optional: python -m pytest path/to/focused_test.py -q
```

Review: status/audience/scope present; Source anchors current; Rationale,
alternatives, consequences, and failure semantics present; no invented APIs.
````

Supervisor-facing hubs may also require
`scripts/docs/check_agent_supervisor_docs.py` when that script’s scope includes
the page.

---

## 12. Compact template (copy-paste)

````markdown
# <Title>

**Status:** Current  
**Audience:** <who>  
**Scope:** <what this page owns>  
**Non-goals:** <what it does not own>  
**Last verified:** `<commit>` (<date>); <what was checked>

## Source anchors

| Concern | Path / symbol | Notes |
| --- | --- | --- |
| … | … | … |

## Context and components

…

## Flows

…

## State and identity

…

## Trust, authorization, and failure semantics

…

## Rationale

…

## Alternatives

…

## Consequences

…

## Extension and compatibility

…

## Operational signals

…

## Verification

```bash
git diff --check
# focused anchor and test commands
```

## Related guides and ADRs

- …
````

---

## 13. Parallel-writer rules

1. **Own only declared outputs** for the task; do not “normalize” sibling
   guides unless the task owns them.
2. **Do not edit** operator-protected program inputs (refresh plan, objective
   heap, task board) unless explicitly authorized.
3. **Shared navigation** (`docs/README.md`, `docs/INDEX.md`, hubs) is reserved
   for designated fan-in tasks; link to them, do not rewrite them early.
4. **ADRs** use the decisions tree and template; guides summarize and link,
   they do not re-host full decision records.
5. **One contract** — this file. Local style flourishes are fine; missing
   required obligations are not.

---

## 14. Review checklist

Before merging a Current architecture guide:

- [ ] Status, audience, scope, non-goals, and last-verified baseline present  
- [ ] **Source anchors** list live paths/symbols; conceptual boxes labelled  
- [ ] Current vs planned language separated  
- [ ] Flows include primary control/data paths  
- [ ] **Rationale**, **alternatives**, and **consequences** present  
- [ ] Failure semantics (fail-closed, degradation, recovery) present  
- [ ] Diagrams match live names or mark conceptual/compatibility  
- [ ] No invented API, flag, extra, or universal optional-capability claims  
- [ ] Volatile metrics dated and reproducible or omitted  
- [ ] **Verification** section runnable without live provider inference  
- [ ] Relative links and `git diff --check` clean  

---

## Related material

| Document | Role |
| --- | --- |
| [`overview.md`](overview.md) | Maintained one-screen architecture overview (to be refreshed under its owning task) |
| [`AGENT_SUPERVISOR_PHILOSOPHY.md`](AGENT_SUPERVISOR_PHILOSOPHY.md) | Supervisor mental model and authority ladder |
| [`agent_supervisor/PACKAGE_MAP.md`](agent_supervisor/PACKAGE_MAP.md) | Domain package ownership pattern |
| [`../development/DOCUMENTATION_CURRENT_STATE.md`](../development/DOCUMENTATION_CURRENT_STATE.md) | Maintained vs historical documentation policy snapshot |
| `docs/architecture/decisions/` | ADR index and template (DOC-004+) |
| `docs/development/DOCUMENTATION_LIFECYCLE.md` | Lifecycle and freshness policy when published (DOC-002) |
