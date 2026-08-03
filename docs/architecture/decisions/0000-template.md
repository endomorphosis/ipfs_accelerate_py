# ADR-NNNN: Short imperative title

<!--
Copy this file to NNNN-kebab-case-short-title.md.
Replace NNNN and the title. Delete this comment block.
Do not edit 0000-template.md to record a real decision.
-->

- **Status:** Proposed
- **Date:** YYYY-MM-DD
- **Last verified:** YYYY-MM-DD (required when Status is Accepted)
- **Deciders:** names or roles (optional but recommended)
- **Scope:** one paragraph or bullet list of what this decision covers
- **Non-goals:** what this ADR deliberately does not decide
- **Supersedes:** ADR-0000 (or “none”)
- **Superseded-by:** none (set when Status is Superseded)
- **Related guides:** paths under `docs/architecture/` (and elsewhere) constrained by this decision
- **Source anchors:** packages, modules, tests, schemas, or config that evidence the decision

## Status meanings (do not invent new values)

| Value | Use when |
| --- | --- |
| Proposed | Decision is under review; **not** yet evidenced current design |
| Accepted | Decision matches current code/tests/ops practice and is normative for Scope |
| Deprecated | Still historical; prefer another practice for new work |
| Superseded | Replaced by the ADR in Superseded-by |
| Rejected | Considered and not adopted; retained to document the negative choice |

Only **Accepted** records are current design authority. **Proposed** records
must not be treated as implemented system law.

## Context

Describe the problem, forces, constraints, and background that made a decision
necessary. Include:

- who is affected (operators, library consumers, agents, maintainers);
- which planes or packages touch the concern;
- what fails or becomes unsafe if the decision is deferred or ignored;
- any non-negotiable external constraints (security, multi-writer safety,
  optional dependency policy, compatibility promises).

Keep this section descriptive. Do not state the choice here.

## Decision

State the chosen design in clear, testable language.

- If **Status is Accepted**, write in present tense about current behavior and
  invariants (“The control plane treats model output as a proposal…”).
- If **Status is Proposed**, label intent explicitly (“We propose that…”) and
  do not claim the code already implements unlanded behavior.
- Prefer normative bullets or short paragraphs over slogans.
- Name the ownership boundary: which package or process is authoritative.

## Alternatives

List credible alternatives that were considered, including doing nothing or
keeping the previous design. For each alternative:

1. **Name** the option.
2. **Describe** how it would work.
3. **Why not chosen** (or why rejected later), with concrete failure modes.

At least two alternatives should appear unless you document that no other
credible option existed. “We did not think of alternatives” is not acceptable
for Accepted ADRs.

### Alternative A: \<name\>

- Summary:
- Expected benefits:
- Why not chosen:

### Alternative B: \<name\>

- Summary:
- Expected benefits:
- Why not chosen:

### Alternative C: Do nothing / status quo (if applicable)

- Summary:
- Why not chosen:

## Consequences

Record **both** positive and negative consequences. Negative consequences are
mandatory for Accepted ADRs; omitting them fails review.

### Positive

- Benefit and who gains it
- Invariant or failure mode that is now prevented
- Operational or audit improvement

### Negative

- Cost, complexity, or friction introduced
- Performance, latency, or resource overhead
- Developer or agent foot-guns that remain
- Compatibility or migration burden
- What becomes harder to change later

### Neutral / residual risks

- Accepted residual risks and how they are monitored or mitigated
- Explicit non-goals that remain out of scope

## Evidence

Bind the decision to repository reality. Prefer exact paths and symbols.

| Claim in Decision | Evidence (path, test, or operational check) | Notes |
| --- | --- | --- |
| … | `path/to/module.py`, `test/…` | … |

Evidence classes (do not promote a weaker class by renaming):

- source layout and package exports;
- deterministic tests and validation commands;
- runtime/ops observations with date and environment;
- schemas, contracts, or receipts.

Plans, chat transcripts, and aspirational docs are **not** sufficient to mark
an ADR **Accepted**.

## Verification

How a future reader confirms the decision still holds:

1. Commands or tests to run (from repo root when possible).
2. Code or config locations to inspect.
3. Pass/fail signals (what would prove the ADR is stale).

Example shape:

```text
# illustrative only — replace with real checks for the decision
python -m pytest path/to/relevant_tests.py -q
rg -n 'symbol_or_invariant' path/to/package
```

## Review triggers

List events that require revisiting this ADR (update, deprecate, or supersede):

- [ ] Source anchors no longer match the Decision statement
- [ ] A recorded negative consequence becomes unacceptable
- [ ] A rejected alternative becomes viable without those costs
- [ ] Security, isolation, lease/fence, or trust-tier changes touch this scope
- [ ] Related guide or package ownership is restructured
- [ ] Superseding design is Accepted under a new ADR number

When superseding: create a new ADR number; set this file to **Superseded** with
`Superseded-by`; set the successor’s `Supersedes`; do not delete this file.

## Notes (optional)

Implementation sketches, links to issues, or migration steps that are not part
of the normative Decision. Keep speculative material here so it is not confused
with Accepted design.
