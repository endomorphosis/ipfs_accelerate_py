# Agent capsule: working inside the agent supervisor

**Status:** Current

**Owner:** agent-supervisor maintainers

**Audience:** Implementation agents and developers reviewing agent work

**Sources:** `ipfs_accelerate_py/agent_supervisor/control/`;
`ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py`;
`ipfs_accelerate_py/agent_supervisor/grok_cli_runner.py`;
`ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py`

**Last-verified:** 2026-08-03 @ `8e940eb01`; invariants, operations, module
paths, and default provider route rechecked

**Freshness triggers:** control authority, task identity, provider routing,
prompt workflow, package-layout, or completion-policy changes

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
   retired flat module paths (for example
   `agent_supervisor.prompt.prompt_workflow`, not a flat `prompt_workflow`).
6. **Isolation** — work in the assigned worktree/branch; do not “fix” the
   operator’s dirty main checkout as a side effect.
7. **Evidence is typed** — tests ≠ solver candidates ≠ kernel proofs ≠
   attestations.
8. **Product vocabulary first** — packages and operations name the system;
   board prefixes schedule work. Do not invent public APIs named after ticket
   prefixes.

## Where truth lives

| Artifact | Role |
| --- | --- |
| `*.objectives.md` | Durable goals and evidence expectations |
| `*.todo.md` | Drainable tasks (prefix headers are IDs) |
| Domain packages under `agent_supervisor/` | Implementation |
| Receipts / DuckDB / event logs | What actually ran |
| Sealed `*_PLAN.md` | Human design; often protected |
| Architecture hub docs | CONTROL_PLANE, EXECUTION_AND_RECOVERY, PACKAGE_MAP |

## Default implementation route

When launching or diagnosing lanes:

| Stage | Current contract |
| --- | --- |
| Provider selection | Set `IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER=grok_quota_codex` |
| Primary | Authenticated Grok Build, exact model `grok-4.5` |
| Quota-only fallback | Exact `gpt-5.6-terra`, reasoning `medium` by default or explicitly sealed `high`, only after independently signed Grok quota authority on the associated fresh retry |
| Legacy compatibility | Unset / `auto` is availability-based and is not the quota-authority policy |
| All other failures | Fail closed; do not fall back |

Explicit `provider=grok` forces Grok with **no** fallback. Explicit
`provider=codex` / `provider=openai` is a direct provider selection, not the
quota-only fallback. Do not turn a generic error string, authentication error,
or missing Grok binary into fallback authority. Codex model/reasoning
environment overrides apply to direct Codex selection, not either pinned
quota fallback.

The legacy in-runner `auto` path runs Grok in a capability-restricted outer
container. Only the active worktree and Grok's ephemeral state are writable,
and peer-provider credentials, configuration, binaries, and runtime sockets
are withheld. Explicit Grok selection may use the native custom sandbox where
it is enforceable. Grok necessarily
retains its own authentication and state; do not describe this boundary as
confidentiality from Grok itself. The allowed model tools can read/search/edit
the repository but cannot run an arbitrary shell, use web or MCP meta-tools,
access memory, or spawn subagents; the supervisor runs validation separately.
Grok never receives the fallback argv. Terra becomes eligible only when the
parent observes an unchanged workspace fingerprint, the primary session's
official terminal record identifies quota exhaustion, and a separate,
tool-free `grok-4.5` probe confirms it. Treat stdout, model text, a changed
workspace, missing isolation, and ambiguous termination as denial—not quota.
This compatibility path is not the production supervisor quota-authority
policy.

The `grok_quota_codex` policy uses the authoritative handoff boundary: it
persists daemon-verified quota authority, defers and releases the Grok attempt,
then permits Terra only for the corresponding retry in a fresh fenced
worktree. The accepted hyphenated and `_fallback` / `-fallback` spellings are
compatibility aliases for this policy, not broader fallback cascades.
The runner's private receipt, stdout, model text, exit status, generic 402/429
text, and the legacy in-runner probe are only candidates; none can bypass the
daemon's independently signed, invocation/task/account/pool-bound verifier.
The supervisor must not attach `--codex-fallback-command-json` to this policy.

## Control and prompt surfaces (do not invent ops)

Closed catalog members include read, proposal, and mutation operations. Prompt
bootstrap/rescue use the shared catalog:

- proposal: `workflow_preview`, `rescue_preview`
- mutation: `workflow_materialize`, `restart`, `rescue`

Operator detail:
[AGENT_SUPERVISOR_GUIDE.md](../../guides/AGENT_SUPERVISOR_GUIDE.md).
Do not bypass `SupervisorControlService` with ad-hoc shell orchestration.

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
| Control contracts | [CONTROL_PLANE.md](CONTROL_PLANE.md) |
| Lanes / recovery | [EXECUTION_AND_RECOVERY.md](EXECUTION_AND_RECOVERY.md) |
| Prompt-first composition | [PROMPT_FIRST_RUNTIME.md](PROMPT_FIRST_RUNTIME.md) |
| Board prefix meaning | [Programs](PROGRAMS.md) |
| Operator commands | [Guide](../../guides/AGENT_SUPERVISOR_GUIDE.md) |
| Deep contracts | [Architecture](../AGENT_SUPERVISOR_ARCHITECTURE.md) |
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
- Use a board ticket or sealed plan as the primary product name for a package
  or operation
