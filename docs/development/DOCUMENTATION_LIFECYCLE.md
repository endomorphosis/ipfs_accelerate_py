# Documentation lifecycle, ownership, and freshness policy

This document is the normative policy for classifying documentation in this
repository, assigning ownership and source authority, deciding where new
material belongs, and deciding when existing material must be revalidated.

It does **not** reclassify individual files or rewrite shared indexes. Those
actions belong to later refresh tasks that cite this policy.

Companion pages:

- [Documentation current state](DOCUMENTATION_CURRENT_STATE.md) — maintained
  surfaces and audit checklist for the checked-out tree.
- [Testing](testing.md) — validation commands used as freshness evidence.
- [Documentation index](../INDEX.md) — audience routing and entrypoints.

## 1. Purpose and non-goals

### Purpose

Authors and agents need deterministic answers to:

1. Is this document **Current**, **Reference**, **Plan**, **Historical**,
   **Generated**, or **Vendored**?
2. What source of truth may override the document?
3. When must the document be revalidated or archived?
4. Where does new information belong?

### Non-goals

- This policy does **not** delete or mass-move historical files.
- This policy does **not** make a filename or directory path sufficient proof
  of Current status.
- This policy does **not** authorize prose to resolve contradictions that
  exist in code, package metadata, or executable help.

## 2. Closed status vocabulary

Every maintained document SHOULD declare a status using exactly one of the
values below. Status is a claim about authority relative to the checked-out
implementation; it is not a measure of writing quality or age alone.

| Status | Normative for behavior? | Meaning |
| --- | --- | --- |
| **Current** | Yes, subject to source-of-truth override | Describes landed, supported behavior for the checked-out tree. Readers and agents may treat it as an operational contract only when it agrees with live sources. |
| **Reference** | Partially | Stable orientation, philosophy, glossary, package map, or API surface description that is maintained but does not alone authorize runtime behavior. Must not invent capabilities absent from code. |
| **Plan** | No | Intended or in-progress design. Useful for sequencing work; never a substitute for landed contracts. |
| **Historical** | No | Point-in-time report, completion summary, phase write-up, session log, or superseded guide retained for context. |
| **Generated** | No (unless regenerated and revalidated) | Machine-produced inventory, export, diagram dump, or derived table. Stale the moment its generator inputs change. |
| **Vendored** | External source wins | Third-party or nested-product docs retained for convenience. This repository does not redefine their upstream contract. |

### Default classifications (fail closed)

When a document does not declare status, apply these defaults. Defaults are
conservative: ambiguity yields a non-normative class.

| Pattern or location | Default status |
| --- | --- |
| `docs/guides/`, `docs/api/`, `docs/development/` maintained pages linked from [INDEX.md](../INDEX.md) or [DOCUMENTATION_CURRENT_STATE.md](DOCUMENTATION_CURRENT_STATE.md) | **Current** only if the page is explicitly maintained there; otherwise treat as **Reference** until verified |
| `docs/architecture/*_PLAN.md`, `*_PLAN_*.md`, active execution plans | **Plan** |
| Filenames containing `summary`, `complete`, `final`, `phase`, `review`, `history`, or `implementation_complete` | **Historical** (non-normative) |
| `docs/archive/`, `docs/development_history/`, `docs/summaries/`, `docs/project/status/`, `docs/project/dashboard/`, `docs/project/summaries/` | **Historical** |
| Objective heaps (`*.objectives.md`) and task boards (`*.todo.md`) | **Plan** / execution record — non-normative for product behavior |
| ADRs under `docs/architecture/decisions/` | **Reference** for *why*; Current guides and code remain authoritative for *what* |
| Nested product trees, submodules, and third-party `README`s | **Vendored** unless a Current page republishes a verified subset |
| HTML/PDF exports under `docs/exports/` and similar generated artifacts | **Generated** |

### Plans and completion summaries are non-normative by default

The following document classes are **non-normative for runtime behavior**
unless a later, explicit, code-verified Current guide supersedes them and
states the supersession:

1. **Plans** — design plans, rollout boards, migration plans, and program
   execution plans (including this repository's documentation refresh plan).
2. **Completion summaries** — “done”, “final”, “100%”, phase-complete, and
   implementation-complete reports.
3. **Session and delivery history** — material under `docs/archive/`,
   `docs/development_history/`, and analogous project records.

A plan may describe intended behavior. A completion summary may claim work
finished at a past commit. Neither may be treated as a Current API, CLI,
MCP, or supervisor contract. When a Current page and a Plan/Historical page
disagree, **Current documentation and live code win**.

## 3. Source-of-truth authority matrix

Authority is ordered. Higher rows override lower rows when they conflict.

| Rank | Authority surface | Examples | Role |
| ---: | --- | --- | --- |
| 1 | Executable package contract | `pyproject.toml`, console scripts, installed entry points | Packaging, extras, script names |
| 2 | Live code and schemas | Modules under `ipfs_accelerate_py/`, typed schemas, policy engines | Behavior and invariants |
| 3 | Executable help and CLI parsers | `ipfs-accelerate --help`, `ipfs_accelerate --help`, argparse definitions | Flags and subcommands actually exposed |
| 4 | Automated tests | `test/`, focused contract tests | Regression proof for claimed behavior |
| 5 | **Current** documentation | Maintained guides, API overview, architecture overview | Human-oriented contract; must track 1–4 |
| 6 | **Reference** documentation | Philosophy, package maps, ADRs, glossaries | Intent and stable vocabulary |
| 7 | **Plan** documentation | `*_PLAN*.md`, objectives, todos | Sequencing only |
| 8 | **Historical** / **Generated** / **Vendored** | Archives, summaries, exports, nested READMEs | Context; never override 1–5 |

### Authority rules

1. **Code owns contradictions.** If two live sources disagree (for example
   package version metadata vs. documented version, or two entry points that
   claim the same flag set), documentation must **record the inconsistency**
   as a code-owned blocker. Prose must not invent a single story that hides
   the disagreement.
2. **Prose may not paper over source inconsistencies.** Forbidden patterns
   include:
   - choosing one of two conflicting version strings without naming both;
   - merging distinct CLI entry points (`ipfs-accelerate` vs
     `ipfs_accelerate`) into one fictional command surface;
   - describing optional extras as always available;
   - converting a Plan tense (“will”, “planned”) into Current tense without
     a landed source anchor and verification command.
3. **Capability language is mandatory** for optional hardware, providers,
   IPFS, P2P, MCP, browser runtimes, external LLMs, and theorem provers.
4. **Volatile metrics require provenance.** Fixed model counts, benchmark
   numbers, and test totals require date, commit, hardware/environment, and a
   reproducible command. Without that provenance they are Historical claims
   even if embedded in a Current page.

## 4. Required metadata (ownership, source, freshness, supersession)

New or refreshed maintained documents SHOULD carry a short header block.
Agents creating Current or Reference pages MUST include at least status,
owner, sources, and last-verified fields.

### Recommended header

```markdown
**Status:** Current | Reference | Plan | Historical | Generated | Vendored
**Owner:** team-or-role (e.g. maintainers, agent-supervisor, mcp)
**Audience:** user | developer | operator | integrator | agent | security
**Sources:** path/to/module.py; path/to/schema; `command --help`
**Last-verified:** YYYY-MM-DD @ <git-sha-or-describe>
**Freshness triggers:** paths or events that invalidate this page
**Supersedes:** optional path(s) this page replaces for readers
**Superseded-by:** optional path if this page is no longer Current
```

### Field semantics

| Field | Rule |
| --- | --- |
| **Status** | Exactly one closed vocabulary value. Filename alone is insufficient. |
| **Owner** | Role or subsystem responsible for updates when sources change. |
| **Sources** | Concrete paths or commands an auditor can open. Prefer relative repo paths. |
| **Last-verified** | Date and commit (or describe) when Sources were checked. Stale if missing on Current pages after a material source change. |
| **Freshness triggers** | Explicit invalidation events (see §6). |
| **Supersedes / Superseded-by** | Bidirectional when practical. Supersession does not delete Historical pages. |

### Ownership matrix (documentation families)

| Family | Typical owner | Default placement |
| --- | --- | --- |
| User/operator journeys | package maintainers | `docs/guides/` |
| Public Python/CLI/MCP surfaces | package maintainers | `docs/api/`, relevant guides |
| Product architecture | architecture maintainers | `docs/architecture/` (named maintained guides) |
| Agent supervisor control plane | agent-supervisor maintainers | `docs/architecture/agent_supervisor/`, operator guide |
| Decisions (ADRs) | architecture maintainers | `docs/architecture/decisions/` |
| Development policy | documentation governance | `docs/development/` |
| Program boards / objectives | program operators | `docs/architecture/*.{todo,objectives}.md` (Plan) |
| Point-in-time project records | project historians | `docs/project/`, `docs/summaries/` (Historical) |
| Archives | no active owner | `docs/archive/`, `docs/development_history/` (Historical) |

Ownership of documentation is **not** the same as code ownership of a
subsystem, but owners must update docs when their sources change.

## 5. Deterministic placement rules (for authors and agents)

When adding or relocating information, choose the **first matching** rule.
Do not invent parallel homes for the same contract.

| # | If the content is… | Place it in… | Status |
| ---: | --- | --- | --- |
| 1 | Supported install, first-run, CLI, MCP, or operator procedure | `docs/guides/` (existing guide if one owns the topic) | Current |
| 2 | Public Python exports or entrypoint inventory | `docs/api/` or the declared API overview | Current |
| 3 | Enduring “why” decision with alternatives considered | `docs/architecture/decisions/` ADR | Reference |
| 4 | Landed subsystem architecture (flows, boundaries, failures) | Named guide under `docs/architecture/` or `agent_supervisor/` | Current or Reference per guide contract |
| 5 | Intended but not-yet-landed design or rollout sequence | `docs/architecture/*_PLAN*.md` or program board | Plan |
| 6 | Point-in-time completion, phase, or session report | `docs/project/summaries/`, `docs/summaries/`, or `docs/development_history/` | Historical |
| 7 | Explicitly retired narrative retained for archaeology | `docs/archive/` | Historical |
| 8 | Machine-generated export or dump | `docs/exports/` or generator-owned path | Generated |
| 9 | Nested product or third-party documentation | Leave in place; link from Current pages only after verification | Vendored |
| 10 | Lifecycle, freshness, testing, or doc governance policy | `docs/development/` | Current |

### Placement constraints

1. **One Current home per contract.** Duplicate Current claims for the same
   flag, API, or path require consolidation or demotion of all but one page
   to Reference/Historical.
2. **Do not promote by path alone.** Moving a file into `docs/guides/` does
   not make it Current until sources and last-verified metadata are real.
3. **Do not demote by age alone.** An old page that still matches live sources
   may remain Current; a new page that only restates a plan remains Plan.
4. **Shared indexes are write-restricted.** Agents must not rewrite
   `docs/README.md` or `docs/INDEX.md` unless their task explicitly owns those
   paths. Link additions may wait for a navigation task.
5. **Protected operator inputs.** Program plan, objective heap, and task board
   files owned by operators are not rewritten by implementation agents except
   through the daemon’s authorized completion path.

### Status label snippet (for Historical / Plan pages)

When a reader could mistake a non-normative page for a Current contract, add
a banner near the top:

```markdown
> **Status: Historical** — retained for context. Not a current API or CLI
> contract. See [Documentation current state](../development/DOCUMENTATION_CURRENT_STATE.md)
> and live sources for supported behavior.
```

```markdown
> **Status: Plan** — describes intended work. Landed behavior is defined by
> code, tests, and Current guides, not by this plan.
```

## 6. Freshness triggers and revalidation rules

### Freshness triggers

A Current or Reference document is **stale** when any of the following occur
relative to its declared Sources or Last-verified commit:

| Trigger | Action |
| --- | --- |
| Source module, schema, or parser changes behavior or public names | Revalidate claims; update examples and Last-verified |
| `pyproject.toml` extras, scripts, or package metadata change | Revalidate install/CLI/packaging claims |
| Test that encodes a documented contract is removed or inverted | Revalidate or remove the claim |
| Optional dependency or capability probe semantics change | Update capability language |
| Superseding Current page is published | Set Superseded-by; demote or archive |
| Generator inputs change (for Generated docs) | Regenerate or mark Historical |
| Link targets move or disappear | Fix links or label debt; do not leave silent 404s on Current pages |

Plan and Historical pages do not require continuous freshness, but they must
not be relabeled Current without a full revalidation pass.

### Deterministic revalidation procedure (agents)

When a task requires documentation updates or claims a page is Current, run
this sequence. Skip steps only when the page’s Sources prove they are
irrelevant.

1. **Open sources.** Read the modules, schemas, and config paths listed in
   Sources (or inferred from the topic if metadata is missing).
2. **Check executable help.** Run the relevant `--help` (or inspect the
   parser). Do not invent flags from memory or from Historical examples.
3. **Check package metadata.** Confirm extras and console scripts in
   `pyproject.toml` / `setup.py` when install or entrypoint claims appear.
4. **Resolve contradictions.** If sources disagree, document both facts and
   open or cite a code fix. **Do not** smooth the conflict in prose.
5. **Update or demote.** Align the page with sources, or change Status to
   Plan/Historical/Reference with an explicit banner.
6. **Record verification.** Set Last-verified to the current date and commit.
7. **Run the smallest relevant check.** Prefer focused deterministic tests and
   `git diff --check`. See [testing.md](testing.md) and the checklist in
   [DOCUMENTATION_CURRENT_STATE.md](DOCUMENTATION_CURRENT_STATE.md).
8. **Leave unrelated files alone.** Do not reclassify the whole tree in a
   task that owns only a single policy or guide path.

### Minimum verification recipe (copy into Current pages)

```bash
git rev-parse --short HEAD
git diff --check
# plus the page-specific help and tests named under Sources
```

## 7. Archive and supersession policy

### Archive

1. Prefer **labels and navigation** over mass deletion.
2. Move or tree-place under `docs/archive/` or `docs/development_history/` when
   a page is no longer a candidate Current surface but retains audit value.
3. Archiving does not require rewriting internal historical claims; it
   requires a Status: Historical signal if the page could be mistaken for
   Current.
4. Do not delete Historical records solely because their claims are old.

### Supersession

1. The superseding page MUST be Current or Reference and MUST list Sources.
2. The superseded page SHOULD gain `Superseded-by` and Status Historical (or
   Plan if still describing unfinished work).
3. Indexes and current-state pages SHOULD point readers at the superseding
   page; updating those shared files is deferred to tasks that own them.
4. Supersession is not retroactive proof that past completion summaries were
   wrong; it only defines what readers should use now.

## 8. Handling code-owned contradictions

When documentation work discovers inconsistency among rank-1–4 sources:

| Step | Required behavior |
| --- | --- |
| Detect | Name each disagreeing source with path or command output. |
| Record | State the inconsistency explicitly in the doc or task evidence. |
| Block silent resolution | Do not pick a “winner” in prose without a code change. |
| Classify impact | If the contradiction affects a user-facing contract, the doc claim is blocked from Current certainty. |
| Escalate | Prefer a code/test fix task; documentation may only describe both sides and the discovery method. |

Examples of code-owned blockers:

- Version or metadata disagreement between packaging files and runtime.
- Two installed commands with overlapping but non-identical parsers.
- Schema field present in docs but absent from the loaded schema module.
- Test fixture and implementation disagreeing on default policy.

## 9. Interfaces for automation

The following logical interfaces are defined for agents and future checkers.
They are documentation contracts, not Python types.

### DocumentationStatus@1

```text
status: Current | Reference | Plan | Historical | Generated | Vendored
normative_for_behavior: boolean  # true only for Current (and only if sources agree)
default_if_undeclared: per §2 fail-closed table
```

### DocumentationAuthorityMap@1

```text
rank 1..4: executable and code authorities (override documentation)
rank 5:   Current documentation
rank 6:   Reference documentation
rank 7:   Plan documentation
rank 8:   Historical | Generated | Vendored
conflict_policy: record code-owned blocker; never conceal in prose
```

### FreshnessTrigger@1

```text
on_source_change: revalidate | demote | regenerate
on_packaging_change: revalidate install/CLI claims
on_test_contract_change: revalidate or remove claim
on_supersession: set Superseded-by; demote superseded page
verification: last-verified commit + reproducible commands
```

## 10. Agent checklist (single pass)

Before marking documentation work complete:

- [ ] Status is an allowed vocabulary value (or a fail-closed default applies).
- [ ] Plan and completion-summary material is not treated as Current.
- [ ] Sources are concrete and were opened during the task.
- [ ] No prose conceals a source inconsistency.
- [ ] Placement followed §5 (first matching rule).
- [ ] Freshness triggers and Last-verified updated when status is Current.
- [ ] Validation commands for the owning task were run.
- [ ] Protected operator files and undeclared paths were not modified.

## 11. Related documents

| Document | Role |
| --- | --- |
| [DOCUMENTATION_CURRENT_STATE.md](DOCUMENTATION_CURRENT_STATE.md) | Inventory of maintained surfaces and review checklist |
| [testing.md](testing.md) | Deterministic and selective test commands |
| [docs/INDEX.md](../INDEX.md) | Audience routing |
| [docs/README.md](../README.md) | Documentation orientation |

---

**Status:** Current  
**Owner:** documentation-governance  
**Audience:** maintainer, agent  
**Sources:** `docs/INDEX.md`; `docs/README.md`; `docs/development/DOCUMENTATION_CURRENT_STATE.md`; `docs/project/`; `docs/archive/`; `docs/development_history/`; `pyproject.toml`  
**Last-verified:** 2026-08-03  
**Freshness triggers:** changes to documentation index layout; new top-level doc families; packaging entrypoint renames; operator changes to governance programs  
**Supersedes:** informal ad-hoc treatment of plans and summaries as current contracts  
