# Agent capsule: working inside the agent supervisor

Short, fail-closed guidance for implementation agents. Prefer this over dumping
entire objective heaps into context.

**Human developers:** start with the
[Developer guide](DEVELOPER_GUIDE.md) and
[package README](../../../ipfs_accelerate_py/agent_supervisor/README.md).

## Hard invariants

1. **Model output is a proposal** — never mark complete without validation and
   the task’s acceptance criteria.
2. **Do not upgrade trust** — candidates, simulated ZK, and cache hits without
   re-derivation cannot grant attestation or completion authority.
3. **Protected paths are sacred** — if a path is listed as
   `--implementation-protected-path` or “operator input”, do not rewrite it
   unless the task `Outputs` explicitly own it.
4. **Taskboards are machine identity** — keep `## PREFIX-123` headers intact;
   do not renumber foreign tasks.
5. **Prefer domain imports** — import from `agent_supervisor.<domain>…`, not
   retired flat module paths.
6. **Isolation** — work in the assigned worktree/branch; do not “fix” the
   operator’s dirty main checkout as a side effect.
7. **Evidence is typed** — tests ≠ solver candidates ≠ kernel proofs ≠
   attestations.

## Where truth lives

| Artifact | Role |
| --- | --- |
| `*.objectives.md` | Durable goals and evidence expectations |
| `*.todo.md` | Drainable tasks (prefix headers are IDs) |
| Domain packages under `agent_supervisor/` | Implementation |
| Receipts / DuckDB / event logs | What actually ran |
| Sealed `*_PLAN.md` | Human design; often protected |

## Default implementation providers

When launching or diagnosing lanes (ops may override):

| Path | Default |
| --- | --- |
| Grok-first implementation | `IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=grok_quota_codex`; Grok is pinned to `grok-4.5`, and only a durable Grok quota latch authorizes direct `gpt-5.6-terra` at `medium` reasoning on a fresh retry |
| Codex | `IPFS_ACCELERATE_AGENT_CODEX_MODEL=gpt-5.6-terra` |

## Codebase-proof loop (semantic)

If the task is about proof-carrying code change:

1. Catalog / claim contracts  
2. Compile obligations + cache keys  
3. Prove via trust-aware cache (lookup before provider)  
4. Query open/satisfied/refuted/impact/proof_delta  
5. Build obligation-first context; on retry use proof_delta + cache hits  
6. Materialize edit packet + validation commands  
7. Re-prove; fail closed on missing/stale receipts  

## If you need X

| Need | Read / use |
| --- | --- |
| Mental model | [Philosophy](../AGENT_SUPERVISOR_PHILOSOPHY.md) |
| How to extend the code | [Developer guide](DEVELOPER_GUIDE.md) |
| Package placement | [Package map](PACKAGE_MAP.md) |
| Board prefix meaning | [Programs](PROGRAMS.md) |
| Operator commands | [Guide](../../guides/AGENT_SUPERVISOR_GUIDE.md) |
| Deep contracts | [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md) |
| Control operations | `control` package + discovery manifests |
| Doc hub | [README.md](README.md) |

## Done means

- Acceptance criteria in the task body are met  
- Declared validation command passes in the worktree  
- Only predicted/owned files changed (plus unavoidable locksteps)  
- No protected-path mutations  
- Board status updates only if the task/completion policy allows and does not
  trip protected-path validation  

## Do not

- Invent public APIs named after ticket prefixes  
- Treat “tests passed locally” as kernel-level proof  
- Expand scope to “while I’m here” refactors outside `Predicted files` /
  `Outputs`  
- Rewrite foreign programs’ boards to make your task look complete  
