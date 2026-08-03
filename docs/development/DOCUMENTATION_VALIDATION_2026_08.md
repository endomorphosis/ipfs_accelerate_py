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
`ipfs_accelerate_py/__init__.py`; live docs and package paths listed below.
**Last verified:** 2026-08-03; offline matrix in §2 executed against the owned
navigation outputs and Current/Reference path set on this worktree. Record the
exact merge-target SHA with `git rev-parse HEAD` when re-running.
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
| Closeout tree | Worktree under documentation-refresh implementation; pin with `git rev-parse HEAD` on the integrated merge target before treating this receipt as merge evidence |
| Owned outputs | `docs/README.md`, `docs/INDEX.md`, `docs/development/DOCUMENTATION_CURRENT_STATE.md`, this file |
| Depends on (evidence) | DOC-021–DOC-027 leaf guides, manifest, glossary, architecture hub |

Reproduce identity:

```bash
git rev-parse HEAD
git show -s --format='%H %ci %s' HEAD
git show -s --format='%H %ci %s' d7da3d6bf8ca2f7ec870d03742b09f26e3e16d15
```

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

### 2.3 Case-fold and installation collision

```bash
# Canonical lowercase install guide must exist
test -f docs/guides/getting-started/installation.md

# Uppercase INSTALLATION.md must not return as a distinct tracked collision
# after DOC-021 (compatibility pointer may remain as INSTALLATION_GUIDE.md)
test ! -e docs/guides/getting-started/INSTALLATION.md \
  || echo "NOTE: INSTALLATION.md still present — case-fold risk on macOS/Windows"

git ls-files 'docs/guides/getting-started/*'
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

### 2.6 Relative-link sample (navigation pages only)

There is **no** repository-standard full-tree link CI gate
([DOCUMENTATION_MAINTENANCE.md](DOCUMENTATION_MAINTENANCE.md)). The closeout
checks markdown links that appear in the four owned files by resolving each
relative target from the file's directory:

```bash
python - <<'PY'
from pathlib import Path
import re, sys
root = Path('.').resolve()
files = [
    root / 'docs/README.md',
    root / 'docs/INDEX.md',
    root / 'docs/development/DOCUMENTATION_CURRENT_STATE.md',
    root / 'docs/development/DOCUMENTATION_VALIDATION_2026_08.md',
]
link_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
missing = []
checked = 0
for path in files:
    text = path.read_text(encoding='utf-8')
    for _label, target in link_re.findall(text):
        if target.startswith(('http://', 'https://', 'mailto:')):
            continue
        if target.startswith('#'):
            continue
        href = target.split('#', 1)[0]
        if not href:
            continue
        dest = (path.parent / href).resolve()
        checked += 1
        if not dest.exists():
            missing.append(f'{path.relative_to(root)} -> {target}')
print(f'checked={checked}')
if missing:
    print('MISSING LINKS:')
    for m in missing:
        print(' ', m)
    sys.exit(1)
print('OK: owned-file relative links resolve')
PY
```

Archive and development_history link debt is **out of scope** for this gate
and is measured only as bulk inventory (see §4).

---

## 3. Recorded results (closeout run)

Results for the DOC-028 closeout on this worktree. Re-run the §2 one-liner on
the merge target and append a run log if HEAD moves after this write.

| Check | Result | Notes |
| --- | --- | --- |
| A1 supervisor ticket-ID checker | **pass** | Primary surfaces (philosophy, hub, package map, FOR_AGENTS, FOR_CONTRIBUTORS, packages/*) free of board-prefix IDs; ARCH IDs only after historical appendix marker (script strips that section) |
| A2 Documentation baseline marker | **pass** | `docs/INDEX.md` contains `Documentation baseline:` dated `2026-08-03` |
| A3 Last verified marker | **pass** | `DOCUMENTATION_CURRENT_STATE.md` header contains `Last verified` |
| A4 validation receipt file | **pass** | This file |
| A5 `git diff --check` | **pass** | Owned files have no trailing whitespace or conflict markers |
| Current-surface path existence (§2.2) | **pass** | All listed Current/Reference navigation targets exist on the tree |
| Case-fold install guide (§2.3) | **pass** | Canonical `installation.md` present; `INSTALLATION.md` absent (DOC-021); `INSTALLATION_GUIDE.md` compatibility pointer remains |
| Module/test anchors (§2.4) | **pass** | `mcp_server/` + `mcp/`; supervisor packages; `test/test_unified_cli_integration.py` present; obsolete `test/api/` path absent |
| Owned-file relative links (§2.6) | **pass** | Relative targets in the four owned files resolve (files and historical directories sampled) |
| Version string agreement | **blocker (code-owned)** | `pyproject.toml`/`setup.py` = `0.0.45`; `__version__` = `0.4.0` — named on install/current-state pages |
| Dual CLI surface | **blocker (code-owned)** | `ipfs-accelerate` vs `ipfs_accelerate` distinct parsers; navigation keeps them separate |
| Full-tree archive link health | **out of scope** | Measured separately as historical debt (§4) |
| External network / optional extras | **out of scope** | Capability-gated; not a closeout failure |

### How to append a run log

```bash
{
  echo "=== DOC-028 validation run ==="
  date -u +%Y-%m-%dT%H:%M:%SZ
  git rev-parse HEAD
  python scripts/docs/check_agent_supervisor_docs.py
  rg -q 'Documentation baseline' docs/INDEX.md && echo "A2 OK"
  rg -q 'Last verified' docs/development/DOCUMENTATION_CURRENT_STATE.md && echo "A3 OK"
  test -f docs/development/DOCUMENTATION_VALIDATION_2026_08.md && echo "A4 OK"
  git diff --check && echo "A5 OK"
} 2>&1 | tee /tmp/doc-028-validation.log
```

---

## 4. Archive and history debt (measured separately)

These do **not** fail the current-surface closeout. They remain lifecycle debt.

| Debt class | Observation | Handling |
| --- | --- | --- |
| Volume | Hundreds of Markdown files under `docs/` (see manifest inventory counts) | Only manifest §§1–3 and navigation Current lists are maintained |
| Mixed historical titles | `COMPLETE`, `FINAL`, `SUMMARY` filenames read as guarantees | Keep **Historical**; label when linked |
| Plan pages under architecture | Many `*_PLAN*.md` and boards | Labelled **Plan** in INDEX and architecture hub |
| Empty gitlinks | Nested products may be absent offline | **Vendored**; capability-gated |
| Frozen drift audit | [DOCUMENTATION_DRIFT_AUDIT_2026_08.md](DOCUMENTATION_DRIFT_AUDIT_2026_08.md) | **Historical** evidence; not a living API guide |

---

## 5. Code-owned blockers and follow-ups

| ID | Blocker | Owner | Doc handling |
| --- | --- | --- | --- |
| B1 | `pyproject.toml`/`setup.py` version `0.0.45` vs `__version__` `0.4.0` | packaging maintainers | Named on install/API/current-state pages |
| B2 | `ipfs-accelerate` vs `ipfs_accelerate` distinct CLIs | CLI maintainers | Separate rows; never merged flag sets |
| B3 | Optional hardware/network/prover stacks | subsystem maintainers | Capability language only |
| B4 | No required PR link-check CI for all Current docs | documentation governance / CI | Local §2.6 sample; maintenance checklist |
| B5 | Nested submodule emptiness offline | integration maintainers | NESTED_PACKAGES + INTEGRATION_BOUNDARIES |

Prose must not pick a silent winner for B1–B2.

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
5. Code resolves B1 (version agreement) or consolidates CLI entry points.

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
