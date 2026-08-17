# Agent Supervisor Prompt Entry Points (v3)

Supported product surfaces after ASE3-009 through ASE3-012.

## Supported journey

1. **Authorize activation (ASE3-026)**  
   Dual receipts enable scoped objective refill and the detached monitor. Broad
   legacy codebase refill stays false.

2. **Open the production facade (ASE3-009)**  
   ```python
   from ipfs_accelerate_py import Supervisor

   supervisor = Supervisor.open()  # from an activated repository root
   preview = supervisor.preview("Improve validation gates")
   ```
   Cold import does not start processes, open DuckDB, or call providers.
   `Supervisor.open()` resolves one body-free
   `ProductionServiceCompositionManifest` bound to ASE3-026.

3. **CLI (ASE3-010)**  
   ```bash
   ipfs-accelerate supervisor run "Improve validation gates"
   ipfs-accelerate supervisor preview --prompt-file intent.txt --output-json
   ipfs-accelerate supervisor status --run-id RUN --output-json
   ipfs-accelerate supervisor doctor --run-id RUN
   ipfs-accelerate supervisor init --consent
   ```
   Help/parse paths are side-effect free. Expert `ipfs-accelerate agent …`
   commands remain available.

4. **MCP / MCP++ (ASE3-011)**  
   Tools (normal input is a prompt):
   - `agent_supervisor_run`
   - `agent_supervisor_preview`
   - `agent_supervisor_steer`
   - `agent_supervisor_status`
   - `agent_supervisor_follow`
   - `agent_supervisor_explain`
   - `agent_supervisor_doctor`

   Client-supplied repository paths require the server allowlist
   `IPFS_ACCELERATE_AGENT_REPOSITORY_ALLOWLIST`. Insufficient or missing
   allowlist entries fail closed.

## Safe bootstrap

1. Work in a clean integration worktree (never the operator dirty checkout).
2. Ensure ASE3-026 dual receipts and completed activation are on the tree.
3. Optionally run `ipfs-accelerate supervisor init --consent` once for local
   profile bootstrap.
4. Open the facade or CLI from that repository root.
5. Prefer `--prompt-file` / stdin over argv when prompts may contain secrets.

## Composition parity

Python, CLI, and MCP/MCP++ must report the **same** production composition CID
for preview/open discovery. Fake in-process service injection and schema-only
tool registration do not satisfy product conformance (ASE3-012).

## DuckDB connection policy

Prompt-product launch-reachable paths must not call raw `duckdb.connect`.
Production connection birth uses `connect_duckdb_with_policy`. Remaining raw
call sites must be classified as non-reachable legacy or proof-only code with
current-tree evidence (see ASE3-012 AST inventory tests).

## Typed failures

| Situation | Expected |
|-----------|----------|
| Missing activation / config | Typed configuration error, nonzero CLI exit |
| Ambiguous run selection | Typed ambiguity, nonzero CLI exit |
| Unavailable runtime binding | Typed unavailable (no simulated completion) |
| Path injection (MCP) | `path_denied` |
| Empty prompt | Invalid / nonzero |

## Related docs

- `docs/guides/AGENT_SUPERVISOR_PROMPT_RUNBOOK.md` — monitor/activation notes
- `docs/architecture/AGENT_SUPERVISOR_PROMPT_ONLY_SELF_IMPROVEMENT_V3_PLAN.md`
