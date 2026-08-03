# Documentation maintenance

**Status:** Current
**Owner:** documentation-governance
**Audience:** maintainers, pull-request authors, implementation agents
**Scope:** How to review documentation changes without suppressing drift;
which local checks exist today; which PR gates are **desired** versus
**already automated**; and explicit follow-up work for workflow, link,
export, and version-drift gaps.
**Non-goals:** This page does not add CI workflows, rewrite package metadata,
or claim automated gates that are not present in `.github/workflows/`. It does
not reclassify the full docs tree (see lifecycle and future manifest tasks).
**Sources:** `docs/development/DOCUMENTATION_LIFECYCLE.md`;
`docs/development/DOCUMENTATION_CURRENT_STATE.md`;
`docs/development/testing.md`; `scripts/docs/check_agent_supervisor_docs.py`;
`.github/workflows/README_DOCUMENTATION_MAINTENANCE.md`;
`.github/workflows/documentation-maintenance.yml`; `pytest.ini`;
`pyproject.toml`.
**Last-verified:** `efb030db743bff50afb939e89fcaa2c650d1c055` (2026-08-03);
anchors opened against live paths; automation claims limited to files that
exist on this tree.
**Freshness triggers:** new doc checkers under `scripts/docs/`; changes to
documentation workflows; lifecycle status vocabulary changes; packaging or
entrypoint renames that invalidate review checklist items.

This document is the **DocumentationReviewChecklist@1** for human and agent
reviewers. Pair it with [DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md)
for status and authority rules, and [testing.md](testing.md) for validation
commands used as freshness evidence.

## Principles

1. **Code and executable help win.** When prose and sources disagree, fix or
   demote the prose—or record a code-owned blocker. Do not invent a single
   story that hides the conflict
   ([lifecycle §3 and §8](DOCUMENTATION_LIFECYCLE.md)).
2. **Capability language is mandatory** for optional hardware, providers,
   network services, IPFS, P2P, MCP, browsers, external LLMs, and provers.
3. **Do not suppress drift.** Missing links, wrong module paths, stale
   version strings, and broken examples are defects to list—not noise to
   ignore so a checklist can be checked green.
4. **Do not claim automation you did not run** and do not describe future
   checkers as if they already gate merges.
5. **Own only your task paths.** Shared indexes (`docs/INDEX.md`,
   `docs/README.md`) and operator-protected program files are out of scope
   unless the task explicitly owns them.

## Review checklist (non-suppressing)

Complete every applicable item. If an item fails, **record the failure** in
the PR or task evidence. Do not mark the item done by narrowing scope after
the fact.

### A. Authority and status

- [ ] Status uses the closed vocabulary (`Current`, `Reference`, `Plan`,
      `Historical`, `Generated`, `Vendored`) or an explicit fail-closed
      default from the lifecycle policy.
- [ ] Plan / completion-summary / archive material is not presented as a
      live API, CLI, or install contract.
- [ ] Sources list concrete paths or commands that were opened during the
      change.
- [ ] Last-verified (or equivalent) is updated when the page is Current and
      claims were rechecked.
- [ ] Any code-owned contradiction is named with both sides; prose does not
      pick a silent winner.

### B. Paths, imports, and commands

- [ ] Module and package paths match the tree (supervisor modules use domain
      packages such as `agent_supervisor.objectives.*`, not obsolete
      package-root basenames).
- [ ] Test paths match the tree (e.g. unified CLI integration is
      `test/test_unified_cli_integration.py`, not under `test/api/`).
- [ ] CLI flags and subcommands match `--help` or the argparse definition for
      the **named** entry point (`ipfs-accelerate` and `ipfs_accelerate` are
      not interchangeable).
- [ ] Package extras and console scripts match `pyproject.toml` / `setup.py`.
- [ ] Relative markdown links resolve from the document’s actual directory.

### C. Examples and capability claims

- [ ] Code samples use current import paths and do not require undeclared
      extras without saying so.
- [ ] Optional stacks are described with discovery or failure behavior, not
      universal availability.
- [ ] Fixed model counts, benchmark numbers, and test totals include date,
      commit, environment, and a reproducible command—or are labeled
      Historical.
- [ ] Validation recipes prefer the smallest deterministic suite from
      [testing.md](testing.md).

### D. Hygiene

- [ ] `git diff --check` is clean for the change set.
- [ ] No secrets, credentials, or machine-local absolute paths in examples.
- [ ] Ticket/board IDs are kept out of primary agent-supervisor product docs
      (see local checker below); boards and objectives may retain them.

### Fail-closed recording

When a checklist item cannot be satisfied:

| Situation | Required action |
| --- | --- |
| Wrong path or flag in a page this PR owns | Fix in the same change. |
| Wrong path in a page **outside** task ownership | Record as follow-up debt; do not “fix” by editing undeclared paths. |
| Two live sources disagree | Document both; open or cite a code/test task. |
| Optional capability unavailable while reviewing | Say so; do not rewrite docs to pretend the capability is unsupported if code still has the path. |
| Desired gate not implemented in CI | List under [Automation gaps](#automation-gaps-explicit-follow-up) — do not invent a green check. |

## Local commands available today

These commands exist on the tree and may be run by contributors. **None of
them is a complete documentation CI substitute.**

```bash
# Whitespace / conflict-marker hygiene
git diff --check

# Smallest useful behavioral smoke for many doc edits
python -m pytest test/test_unified_cli_integration.py -q

# Primary agent-supervisor docs: fail on board-prefix ticket ID leakage
python scripts/docs/check_agent_supervisor_docs.py

# Packaging / entrypoint sanity when install or CLI claims change
rg -n '\[project\.scripts\]|^\[project\.optional-dependencies\]' pyproject.toml
ipfs-accelerate --help   # when the hyphenated script is installed

# Supervisor module help (domain package paths)
python -m ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon --help
python -m ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor --help
```

Use [testing.md](testing.md) when the change needs broader or
capability-gated suites. Prefer `-rs` so skips remain visible.

## What already runs in automation (honest inventory)

| Mechanism | What it actually does | What it does **not** do |
| --- | --- | --- |
| `.github/workflows/documentation-maintenance.yml` | Scheduled / manual workflow described in [`.github/workflows/README_DOCUMENTATION_MAINTENANCE.md`](../../.github/workflows/README_DOCUMENTATION_MAINTENANCE.md): codebase docstring analysis, example syntax sampling, pdoc-oriented generation, report artifacts / optional maintenance PRs | Not a required PR merge gate for every documentation change; not a full link checker; not a version-drift enforcer across packaging files |
| `scripts/docs/check_agent_supervisor_docs.py` | Fails if board-prefix ticket IDs appear in listed primary supervisor docs | Does not validate all of `docs/`; does not check links, versions, or examples |
| Default pytest `testpaths` | Runs configured API / distributed / MCP tests when CI invokes pytest that way | Does not run every `test/test_*.py` module unless named; does not validate markdown |

Do **not** document the weekly maintenance workflow as proof that every PR
already received link, export, or version review.

## Desired PR gates (not claimed as current)

The following gates are **recommended policy** for documentation and
mixed code+docs PRs. Treat them as the target contract for a separately
authorized CI/code task. **They are not all enforced today.**

| Desired gate | Intent | Suggested check (when implemented) |
| --- | --- | --- |
| Diff hygiene | Reject conflict markers and bad whitespace | `git diff --check` |
| Focused tests for touched surfaces | Behavioral smoke for CLI/API/doc-adjacent code | Named pytest paths from [testing.md](testing.md) |
| Supervisor primary-doc vocabulary | Keep product docs free of board ticket leakage | `python scripts/docs/check_agent_supervisor_docs.py` |
| Link integrity (maintained surfaces) | Fail on broken relative links in Current pages | Dedicated link checker over an allowlisted set (not yet a single repo-standard gate) |
| Export freshness labels | Generated HTML/PDF/export trees marked Generated/Historical | Manifest or path policy (see lifecycle); no full tree enforcer yet |
| Version / metadata consistency | Surface disagreement among packaging and documented versions | Explicit multi-source compare; must **report** conflicts, not pick a winner in prose |
| Capability honesty | Optional stacks not claimed universal | Review checklist §C; optional lint for “always available” language (not implemented) |

Until those gates land in CI, authors must run the applicable **local**
commands and reviewers must apply this checklist manually.

## Automation gaps (explicit follow-up)

These gaps are **known follow-up work**. Closing them requires a task that
is allowed to change workflows, scripts, or packaging—not a documentation-only
edit that pretends the automation already exists.

| Gap | Symptom today | Follow-up direction (separate authorization) |
| --- | --- | --- |
| **Workflow coverage vs PR gating** | Weekly `documentation-maintenance` analysis is not the same as blocking PR checks | Add or wire required jobs for doc PRs; keep schedule job for inventory only |
| **Link checking** | No single maintained, fail-closed relative-link gate for Current docs | Introduce an allowlisted link checker; fix or label debt rather than silencing failures |
| **Export / generated artifact drift** | Generated or export trees can go stale while still looking official | Enforce status labels and regeneration instructions; exclude Generated from Current navigation |
| **Version drift** | Packaging files, READMEs, and guides can disagree on versions | Checker that prints all disagreeing sources; docs record both sides until code fixes |
| **Path vocabulary drift** | Historical and secondary guides still cite flat supervisor modules or `test/api/test_unified_cli_integration.py` | Continue refresh tasks; see [DOCUMENTATION_DRIFT_AUDIT_2026_08.md](DOCUMENTATION_DRIFT_AUDIT_2026_08.md) for inventory |
| **Broad example execution** | Workflow may syntax-check samples; it does not prove every fenced block runs against optional extras | Keep examples honest about extras; expand executable smoke only where deterministic |

When filing follow-up work, cite this section and the lifecycle exception
policy: **no fictional automation**.

## Contributor workflow (short)

1. Edit only the paths your task owns.
2. Open the Sources for every behavioral claim you touch.
3. Run the smallest relevant checks from [testing.md](testing.md) and the
   local commands above.
4. Complete the [review checklist](#review-checklist-non-suppressing); record
   any item you cannot satisfy.
5. List automation gaps you hit instead of claiming CI covered them.
6. Leave operator-protected plan/objectives/todo inputs and undeclared paths
   alone.

## Related documents

| Document | Role |
| --- | --- |
| [DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md) | Normative status, authority, freshness, exceptions |
| [DOCUMENTATION_CURRENT_STATE.md](DOCUMENTATION_CURRENT_STATE.md) | Maintained surfaces and short audit list |
| [testing.md](testing.md) | Test selection and capability reporting |
| [DOCUMENTATION_DRIFT_AUDIT_2026_08.md](DOCUMENTATION_DRIFT_AUDIT_2026_08.md) | Point-in-time drift inventory (Historical) |
| [GUIDE_CONVENTIONS.md](../architecture/GUIDE_CONVENTIONS.md) | Architecture guide writing contract |
| [Workflow README](../../.github/workflows/README_DOCUMENTATION_MAINTENANCE.md) | Description of the weekly maintenance workflow |
