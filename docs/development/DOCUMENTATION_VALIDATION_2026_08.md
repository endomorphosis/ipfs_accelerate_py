# Documentation validation closeout (2026-08)

**Status:** Reference
**Owner:** documentation-governance
**Audience:** maintainers, operators, and agents reproducing the
documentation-refresh closeout on one exact tree
**Scope:** Offline-safe validation receipt for DOC-028 / DOC-G062. Records the
integrated-tree identity, commands, return codes, current-surface path and
navigation checks, archive debt measured separately, code-owned blockers, and
next-audit triggers.
**Non-goals:** External network availability; inventing CI gates; rewriting leaf
guides or source code; claiming full-tree link health for archives; silently
resolving packaging or CLI contradictions.
**Sources:** `docs/README.md`; `docs/INDEX.md`;
`docs/development/DOCUMENTATION_CURRENT_STATE.md`;
`docs/development/DOCUMENTATION_MANIFEST.md`;
`scripts/docs/check_agent_supervisor_docs.py`; `pyproject.toml`;
`ipfs_accelerate_py/__init__.py`; CLI and supervisor module help;
`test/test_unified_cli_integration.py`; agent-supervisor public-surface tests;
live docs and package paths listed below.
**Last verified:** 2026-08-03 at corrected validation baseline
`b334e0bf7ba6554be0c527576be56637d9357014`; §2 records the exact commands and
§3 records their observed return codes, counts, and warnings.
**Interface:** DocumentationValidationReceipt@1
**Program:** `ipfs-accelerate-documentation-refresh-v1`
**Task:** DOC-028 / goal DOC-G062

This receipt is **Reference** evidence for one closeout pass. It is not a
Current product API. Re-run the §2 matrix after material navigation or packaging
changes; publish a successor receipt rather than editing history silently.

---

## 1. Tree identity

| Field | Value |
| --- | --- |
| Validation date (UTC calendar) | 2026-08-03 |
| Program baseline commit (plan pin) | `d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15` |
| Accepted DOC-027 dependency | `014b063e3d8a79c5814f98c2ca055e762694037b` |
| DOC-028 implementation commit | `7bef572af222d90714d8e2156c734d07f68c64c2` |
| Supervisor integration commit | `01fe524abe3ec1bdeebb2bda7ed372a9e2f47023` |
| Corrected validation baseline | `b334e0bf7ba6554be0c527576be56637d9357014` |
| Baseline state | Clean worktree; parent `01fe524abe3ec1bdeebb2bda7ed372a9e2f47023`; exactly four DOC-028 owned paths differ from accepted DOC-027 |
| Owned outputs | `docs/README.md`, `docs/INDEX.md`, `docs/development/DOCUMENTATION_CURRENT_STATE.md`, this file |
| Depends on (evidence) | DOC-021–DOC-027 leaf guides, manifest, glossary, architecture hub |

Reproduce identity:

```bash
git show -s --format='%H %P %ci %s' b334e0bf7ba6554be0c527576be56637d9357014
git show -s --format='%H %P %ci %s' 01fe524abe3ec1bdeebb2bda7ed372a9e2f47023
git show -s --format='%H %ci %s' 7bef572af222d90714d8e2156c734d07f68c64c2
git diff --name-status 014b063e3d8a79c5814f98c2ca055e762694037b..b334e0bf7ba6554be0c527576be56637d9357014
git show -s --format='%H %ci %s' d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15
```

The receipt-publication commit follows the validated baseline because a commit
cannot embed its own content hash. Its only documentation change is this
identity binding; the literal gate and local-link scans are rerun after
publication.

---

## 2. Offline command matrix (acceptance + supporting checks)

Run from the repository root. **External network is out of scope.** A check
passes only when the command exits 0 (or the row explicitly records a
code-owned blocker).

### 2.1 Acceptance gate (DOC-028)

| # | Command | Intent | Expected |
| ---: | --- | --- | --- |
| A1 | `python scripts/docs/check_agent_supervisor_docs.py` | Primary supervisor docs free of board-prefix ticket IDs | exit 0 |
| A2 | `rg -q 'Documentation baseline' docs/INDEX.md` | Index carries baseline marker | exit 0 |
| A3 | `rg -q 'Last verified' docs/development/DOCUMENTATION_CURRENT_STATE.md` | Current-state page carries verification marker | exit 0 |
| A4 | `test -f docs/development/DOCUMENTATION_VALIDATION_2026_08.md` | This receipt exists | exit 0 |
| A5 | `git diff --check` | No conflict markers / bad whitespace in the diff | exit 0 |

One-liner (matches the task validation contract):

```bash
python scripts/docs/check_agent_supervisor_docs.py \
  && rg -q 'Documentation baseline' docs/INDEX.md \
  && rg -q 'Last verified' docs/development/DOCUMENTATION_CURRENT_STATE.md \
  && test -f docs/development/DOCUMENTATION_VALIDATION_2026_08.md \
  && git diff --check
```

### 2.2 Current-surface path existence

Every path below is linked from `docs/README.md` or `docs/INDEX.md` as a
**Current** or explicitly labelled **Reference** entry. Missing paths fail the
current-surface gate.

```bash
# Navigation and governance
for p in \
  docs/README.md \
  docs/INDEX.md \
  docs/development/DOCUMENTATION_CURRENT_STATE.md \
  docs/development/DOCUMENTATION_MANIFEST.md \
  docs/development/DOCUMENTATION_LIFECYCLE.md \
  docs/development/DOCUMENTATION_MAINTENANCE.md \
  docs/development/DOCUMENTATION_VALIDATION_2026_08.md \
  docs/development/testing.md \
  docs/architecture/README.md \
  docs/architecture/GLOSSARY.md \
  docs/architecture/GUIDE_CONVENTIONS.md \
  docs/architecture/overview.md \
  docs/architecture/SYSTEM_CONTEXT.md \
  docs/architecture/INFERENCE_RUNTIME.md \
  docs/architecture/MODEL_SERVICE_ROUTING.md \
  docs/architecture/MCP_RUNTIME.md \
  docs/architecture/DISTRIBUTED_RUNTIME.md \
  docs/architecture/INTEGRATION_BOUNDARIES.md \
  docs/architecture/AI_SERVICE_CATALOG.md \
  docs/architecture/decisions/README.md \
  docs/api/overview.md \
  docs/guides/getting-started/README.md \
  docs/guides/getting-started/installation.md \
  docs/guides/QUICKSTART.md \
  docs/guides/cli/README_CLI.md \
  docs/guides/MCP_SETUP_GUIDE.md \
  docs/MCP_SERVER.md \
  docs/guides/hardware/overview.md \
  docs/guides/deployment/README.md \
  docs/guides/p2p/README.md \
  docs/guides/troubleshooting/faq.md \
  docs/guides/AGENT_SUPERVISOR_GUIDE.md \
  docs/architecture/agent_supervisor/README.md \
  docs/architecture/agent_supervisor/CONTROL_PLANE.md \
  docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md \
  docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md \
  docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md \
  docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md \
  docs/architecture/agent_supervisor/FOR_AGENTS.md \
  docs/architecture/agent_supervisor/FOR_CONTRIBUTORS.md \
  docs/architecture/agent_supervisor/PACKAGE_MAP.md \
  docs/architecture/agent_supervisor/PROGRAMS.md \
  docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md \
  docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md \
  docs/architecture/decisions/0001-objectives-and-task-projections.md \
  docs/architecture/decisions/0002-model-proposals-and-evidence-admission.md \
  docs/architecture/decisions/0003-capabilities-catalogs-and-routing.md \
  docs/architecture/decisions/0004-worktrees-leases-and-fencing.md \
  docs/architecture/decisions/0005-mutable-coordination-and-immutable-replication.md \
  docs/architecture/decisions/0006-domain-packages-and-compatibility-boundaries.md
do
  test -f "$p" || { echo "MISSING: $p"; exit 1; }
done
echo "OK: current-surface paths exist"
```

### 2.3 Case-fold inventory and installation collision

```bash
# Canonical lowercase install guide must exist
test -f docs/guides/getting-started/installation.md

# Uppercase INSTALLATION.md must not return as a distinct tracked collision
# after DOC-021 (compatibility pointer may remain as INSTALLATION_GUIDE.md)
test ! -e docs/guides/getting-started/INSTALLATION.md

git ls-files 'docs/guides/getting-started/*'

python - <<'PY'
from collections import defaultdict
import subprocess

tracked = subprocess.check_output(['git', 'ls-files'], text=True).splitlines()
for label, paths in (
    ('docs', [p for p in tracked if p.startswith('docs/')]),
    ('repo', tracked),
):
    folded = defaultdict(list)
    for path in paths:
        folded[path.casefold()].append(path)
    collisions = [group for group in folded.values() if len(group) > 1]
    print(f'{label}_casefold_collisions={len(collisions)}')
    for group in collisions:
        print(' | '.join(group))
PY
```

### 2.4 Module, test, and packaging anchors

```bash
# Canonical MCP vs facade
test -d ipfs_accelerate_py/mcp_server
test -d ipfs_accelerate_py/mcp

# Supervisor control packages (sample)
test -d ipfs_accelerate_py/agent_supervisor/control
test -d ipfs_accelerate_py/agent_supervisor/entrypoints
test -f ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py

# Unified CLI integration test path (not the obsolete test/api/ location)
test -f test/test_unified_cli_integration.py
test ! -e test/api/test_unified_cli_integration.py

# Version sources (code-owned disagreement — both must be readable)
rg -n '^version\s*=' pyproject.toml setup.py
rg -n '__version__\s*=' ipfs_accelerate_py/__init__.py

# Console scripts (two distinct CLIs)
rg -n 'ipfs-accelerate|ipfs_accelerate' pyproject.toml
```

### 2.5 Navigation status labelling (spot checks)

```bash
# Index must not present Plan-only pages as the sole Start-here route
rg -n 'Start here' docs/INDEX.md
rg -n '\*\*Plan\*\*|Lifecycle|Historical' docs/INDEX.md

# Baseline markers
rg -n 'Documentation baseline' docs/INDEX.md docs/README.md
rg -n 'Last verified' docs/development/DOCUMENTATION_CURRENT_STATE.md
```

### 2.6 Status-aware local-link and debt scan

There is **no** repository-standard full-tree link CI gate
([DOCUMENTATION_MAINTENANCE.md](DOCUMENTATION_MAINTENANCE.md)). This scanner
checks every tracked Markdown file that declares `Status: Current` or
`Status: Reference`, the six navigation/governance entrypoints, and the four
owned files. It reports broader non-history and archive/history debt
separately. It checks local target existence, not URL reachability or fragment
anchors; duplicate link occurrences remain counted.

```bash
python - <<'PY'
from pathlib import Path
from urllib.parse import unquote
import re, subprocess, sys

root = Path('.').resolve()
tracked = subprocess.check_output(['git', 'ls-files', 'docs'], text=True).splitlines()
markdown = [
    root / rel for rel in tracked
    if rel.lower().endswith(('.md', '.markdown')) and (root / rel).is_file()
]
status_re = re.compile(r'^\*\*Status:\*\*\s+(?:Current|Reference)\b', re.M)
maintained = [
    path for path in markdown
    if status_re.search(path.read_text(encoding='utf-8', errors='replace'))
]
entrypoints = [
    root / 'docs/README.md',
    root / 'docs/INDEX.md',
    root / 'docs/development/DOCUMENTATION_CURRENT_STATE.md',
    root / 'docs/development/DOCUMENTATION_MANIFEST.md',
    root / 'docs/architecture/README.md',
    root / 'docs/development/DOCUMENTATION_VALIDATION_2026_08.md',
]
for path in entrypoints:
    if path not in maintained:
        maintained.append(path)
owned = [entrypoints[0], entrypoints[1], entrypoints[2], entrypoints[5]]
nonhistory = [
    path for path in markdown
    if not {'archive', 'development_history'}.intersection(path.relative_to(root).parts)
]
history = [
    path for path in markdown
    if {'archive', 'development_history'}.intersection(path.relative_to(root).parts)
]
link_re = re.compile(r'(?<!!)\[[^\]]*\]\(([^)]+)\)')

def scan(files):
    checked, missing = 0, []
    for path in files:
        text = path.read_text(encoding='utf-8', errors='replace')
        for raw in link_re.findall(text):
            target = raw.strip().strip('<>')
            if target.startswith(('#', 'http://', 'https://', 'mailto:',
                                  'tel:', 'data:')):
                continue
            href = unquote(target.split('#', 1)[0].split('?', 1)[0]).strip()
            if not href:
                continue
            dest = ((root / href.lstrip('/')) if target.startswith('/')
                    else (path.parent / href))
            checked += 1
            if not dest.exists():
                missing.append((path.relative_to(root).as_posix(), target))
    return checked, missing

failed = False
for label, files, gating in (
    ('current_reference', maintained, True),
    ('owned', owned, True),
    ('nonarchive_debt', nonhistory, False),
    ('archive_history_debt', history, False),
):
    checked, missing = scan(files)
    unique = len(set(missing))
    affected = len({source for source, _target in missing})
    print(f'{label}: files={len(files)} local_targets={checked} '
          f'missing={len(missing)} unique_missing={unique} '
          f'affected_files={affected}')
    if gating and missing:
        failed = True
        for source, target in missing:
            print(f'MISSING: {source} -> {target}')
if failed:
    sys.exit(1)
PY
```

Missing targets outside maintained/owned scope are debt, not suppressed passes;
their measured counts appear in §4.

### 2.7 Help, import, and focused public-surface checks

```bash
python -m ipfs_accelerate_py.cli --help
python -m ipfs_accelerate_py.ai_inference_cli --help
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon --help
python -m ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon --help
python -m ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor --help
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor --help

# These stale epilog examples must remain unregistered (expected rc=2).
for group in inference queue network; do
  if python -m ipfs_accelerate_py.cli "$group" --help >/dev/null 2>&1; then
    echo "UNEXPECTED REGISTERED GROUP: $group"
    exit 1
  else
    test "$?" -eq 2 || exit 1
  fi
done

IPFS_ACCEL_SKIP_CORE=1 python -c \
  "import ipfs_accelerate_py as p; import ipfs_accelerate_py.mcp_server; import ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon; print(p.__version__)"

python -m pytest test/test_unified_cli_integration.py -q -rs
python -m pytest \
  test/api/test_agent_supervisor_v2_public_api.py \
  test/api/test_agent_supervisor_semantic_layout_exports.py \
  test/api/test_agent_supervisor_entrypoint_package.py -q -rs
```

The installed console scripts were not on this checkout's `PATH`; module forms
were therefore the executable authority for this offline run. That environment
fact is not evidence that an installed distribution lacks the scripts.

---

## 3. Recorded results (closeout run)

The operator reran these checks after the supervisor integrated DOC-028 at the
exact candidate in §1. `rc=0` means the command completed successfully; an
expected nonzero probe is identified explicitly.

| Check | Observed result | Notes |
| --- | --- | --- |
| A1 supervisor ticket-ID checker | **pass**, `rc=0` | Primary surfaces free of board-prefix ticket IDs outside their historical appendix |
| A2 Documentation baseline marker | **pass**, `rc=0` | `docs/INDEX.md` contains `Documentation baseline:` dated `2026-08-03` |
| A3 Last verified marker | **pass**, `rc=0` | `DOCUMENTATION_CURRENT_STATE.md` contains `Last verified` |
| A4 validation receipt file | **pass**, `rc=0` | This file exists |
| A5 worktree and candidate-range `git diff --check` | **pass**, both `rc=0` | Clean worktree check plus `014b063e3d8a79c5814f98c2ca055e762694037b..b334e0bf7ba6554be0c527576be56637d9357014` |
| Candidate scope | **pass**, `rc=0` | Exactly the four DOC-028 owned outputs differ from the accepted DOC-027 dependency |
| Current/Reference path inventory (§2.2) | **pass**, `rc=0` | 50 files checked; 0 missing |
| Status-aware local links (§2.6) | **pass**, `rc=0` | 38 files; 931 local target occurrences; 0 missing |
| Owned-file local links (§2.6) | **pass**, `rc=0` | 4 files; 253 local target occurrences; 0 missing |
| Documentation case-fold inventory (§2.3) | **pass**, `rc=0` | 0 collisions under tracked `docs/` paths; canonical lowercase installation guide exists and uppercase collision is absent |
| Repository case-fold inventory (§2.3) | **measured debt**, `rc=0` | 6 collision groups, all under `test/`; exact pairs in §4 |
| Module/test/packaging anchors (§2.4) | **pass**, `rc=0` | Canonical MCP/facade and supervisor paths exist; unified test is at `test/test_unified_cli_integration.py`; obsolete test path is absent |
| Version agreement | **code-owned blocker** | Packaging files report `0.0.45`; runtime `__version__` reports `0.4.0` |
| Unified and direct-AI CLI help | **pass**, both `rc=0` | Unified registered groups differ from the underscore CLI by design; console scripts were not on this checkout's `PATH`, so module forms were used |
| Supervisor help (four modules) | **pass**, all `rc=0` | Implementation-supervisor module emitted the known `runpy` warning described below |
| Cold package/MCP/supervisor import | **pass**, `rc=0` | Printed runtime version `0.4.0` with `IPFS_ACCEL_SKIP_CORE=1` |
| Unified CLI integration tests | **pass**, `rc=0` | 6 passed; one `pydub` / deprecated `audioop` warning |
| Agent-supervisor public-surface tests | **pass**, `rc=0` | 10 passed; `pydub` / `audioop` and `__package__ != __spec__.parent` deprecation warnings |
| Stale unified-help examples | **code-owned blocker reproduced** | Registered choices exclude `inference`, `queue`, and `network`; all three probes return expected invalid-choice `rc=2`, while the help epilog advertises them |
| External network, hardware, provider, P2P/IPFS, and prover capability | **not tested** | Capability-gated and intentionally outside this offline run; no pass is claimed |

The implementation-supervisor help warning was:
`RuntimeWarning: ...implementation_supervisor found in sys.modules ... prior
to execution`. It did not change the successful return code. The two focused
test commands' warnings are retained above rather than hidden.

---

## 4. Link and case-fold debt (archive/history separated)

The same §2.6 scanner measured these scopes. Missing counts are link
occurrences; “unique” deduplicates identical `(source, target)` pairs. These do
**not** fail the maintained current-surface closeout, but they are not reported
as passes.

| Debt class | Observation | Handling |
| --- | --- | --- |
| Maintained/owned link targets | Current/Reference: 38 files / 931 local targets / 0 missing; owned: 4 / 253 / 0 | Gating scopes; both pass |
| Non-archive link debt | 366 Markdown files / 1,577 local targets / 37 missing occurrences / 26 unique pairs across 6 files | Preserve as measured debt; do not promote unreviewed pages |
| `archive/` + `development_history/` link debt | 127 Markdown files / 88 local targets / 56 missing occurrences / 47 unique pairs across 13 files | **Historical**; repair only in a separately owned archive task |
| Case-fold collisions under `docs/` | 0 among tracked documentation paths | Documentation closeout passes this check |
| Repository-wide case-fold collisions | 6 groups, all under `test/`: `DOCUMENTATION_UPDATE_SUMMARY`, `HF_MODEL_IMPLEMENTATION_SUMMARY`, `MIGRATION_REPORT`, `NEXT_STEPS`, `WEBGPU_BROWSER_OPTIMIZATIONS`/`WebGPU_BROWSER_OPTIMIZATIONS`, and `WEB_PLATFORM_INTEGRATION_GUIDE` (each paired with its case-fold equivalent) | Outside DOC-028 ownership; test-tree owners must reconcile before case-insensitive checkout support is claimed |
| Mixed historical titles | `COMPLETE`, `FINAL`, `SUMMARY` filenames read as guarantees | Keep **Historical**; label when linked |
| Plan pages under architecture | Many `*_PLAN*.md` and boards | Labelled **Plan** in INDEX and architecture hub |
| Empty gitlinks | Nested products may be absent offline | **Vendored**; capability-gated |
| Frozen drift audit | [DOCUMENTATION_DRIFT_AUDIT_2026_08.md](DOCUMENTATION_DRIFT_AUDIT_2026_08.md) | **Historical** evidence; not a living API guide |

The six repository-wide collision groups are:

```text
test/DOCUMENTATION_UPDATE_SUMMARY.md | test/documentation_update_summary.md
test/HF_MODEL_IMPLEMENTATION_SUMMARY.md | test/hf_model_implementation_summary.md
test/MIGRATION_REPORT.md | test/migration_report.md
test/NEXT_STEPS.md | test/next_steps.md
test/WEBGPU_BROWSER_OPTIMIZATIONS.md | test/WebGPU_BROWSER_OPTIMIZATIONS.md
test/WEB_PLATFORM_INTEGRATION_GUIDE.md | test/web_platform_integration_guide.md
```

The scanner skips external URL reachability and fragment-anchor validation.
Those omissions are explicit follow-up scope, not successful checks.

---

## 5. Code-owned blockers and follow-ups

| ID | Blocker | Owner | Doc handling |
| --- | --- | --- | --- |
| B1 | `pyproject.toml`/`setup.py` version `0.0.45` vs `__version__` `0.4.0` | packaging maintainers | Named on install/API/current-state pages |
| B2 | Unified CLI help epilog advertises unregistered `inference`, `queue`, and `network` groups | CLI maintainers | Current CLI/API docs identify the examples as stale; remove or repair them in source |
| B3 | Optional hardware/network/prover stacks | subsystem maintainers | Capability language only |
| B4 | No required PR link/anchor checker for all Current docs | documentation governance / CI | Local §2.6 path scanner; anchors/external URLs remain unchecked |
| B5 | Nested submodule emptiness offline | integration maintainers | NESTED_PACKAGES + INTEGRATION_BOUNDARIES |

The intentional `ipfs-accelerate` / `ipfs_accelerate` parser split is a public
interface boundary, not B2. Prose keeps their commands separate. It must not
pick a silent winner for B1 or present B2's stale examples as commands.

---

## 6. Navigation closeout effects (DOC-028)

| Change | Detail |
| --- | --- |
| Orientation (`docs/README.md`) | Choose-a-path table with lifecycle labels; dual CLI note; governance links |
| Index (`docs/INDEX.md`) | Current start paths; architecture and supervisor Current/Reference/Plan sections; Historical archives labelled; baseline `2026-08-03` |
| Current state | Full maintained-surface matrix, blockers, offline checklist, next-audit triggers |
| This receipt | Reproducible offline matrix and known limitations |

Readers are never routed to a **Plan** or **Historical** page as if it were
**Current** without an explicit status label.

---

## 7. Next audit triggers

Publish a new validation receipt (or refresh this one's run log) when:

1. Top-level navigation or Current guide entrypoints change.
2. Packaging scripts, extras, or version sources change.
3. Supervisor primary-doc checker paths change.
4. A repository-standard allowlisted link checker lands.
5. Code resolves B1 (version agreement) or removes B2's stale help examples.

---

## 8. Related documents

| Document | Role |
| --- | --- |
| [DOCUMENTATION_CURRENT_STATE.md](DOCUMENTATION_CURRENT_STATE.md) | Living maintained-surface matrix |
| [DOCUMENTATION_MANIFEST.md](DOCUMENTATION_MANIFEST.md) | Status inventory |
| [DOCUMENTATION_LIFECYCLE.md](DOCUMENTATION_LIFECYCLE.md) | Authority policy |
| [DOCUMENTATION_MAINTENANCE.md](DOCUMENTATION_MAINTENANCE.md) | Review checklist and automation honesty |
| [DOCUMENTATION_DRIFT_AUDIT_2026_08.md](DOCUMENTATION_DRIFT_AUDIT_2026_08.md) | Frozen Wave-0 inventory |
| [docs/INDEX.md](../INDEX.md) | Canonical navigation |
| [docs/README.md](../README.md) | Orientation |
