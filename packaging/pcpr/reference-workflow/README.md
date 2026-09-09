# PCPR-060 declared reference high-level objective

These files are the Accelerate *declared* Phase-6 reference objective
for proof-carrying-platform-0.1.0. They submit the exact idea:

    Modify a typed formal-logic API while reusing unaffected proofs,
    selecting only impacted tests, rejecting stale-tree evidence, and
    producing a complete proof-carrying execution receipt.

through the repaired PCPR-004 direct-interface contracts as a
SupervisorObjectiveIntent. They are not a live Supervisor.run, CLI, or
MCP submission, not a live DuckDB or Quack materialization, not a
START, not a ContextPack (PCPR-061), not a stored root (PCPR-062), not
a freeze (PCPR-002), and not a closed PCPR release (PCPR-093/094).

- `cpython312/reference.objective.json` binds the idea digest, declared
  caller, authority checks, goals, tasks, assumptions, guarantees,
  acceptance conditions, and budgets. Exact commit and tree are bound
  by the PCPR-060 receipt `current_tree_binding` because nested
  admission rewrites HEAD. `origin/main` is not the release identity.
- `platform-catalog.json` binds Datasets and Kit objective-binding
  documents when those sibling trees are present. Sibling source is
  never required.
- Missing a live Quack-fenced state-owner session emits explicit
  operator-blocking task `pcpr-060-operator-live-objective-materialization`.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
Provider-local `~/.local` tools are not sealed-environment authority.
