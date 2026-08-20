# Autonomous Meta-Controller paired benchmark

This directory is the frozen, hermetic bootstrap corpus for
`agent-supervisor-autonomous-meta-controller-v1`.  It compares the exact
`origin/main` baseline with the candidate on the same repository tree,
objective revisions, capability fixtures, provider fixtures, fault schedule,
policy, token accounting, and human-answer fixtures.

`cases.json` contains typed decision fixtures only.  It contains no source
bodies, prompts, model transcripts, credentials, or claimed live-model
results.  `baseline_manifest.json` binds the current-main source identity and
records benchmark measurements as `not_run`; scaffolding a case is never
reported as an efficiency win.

Validate the frozen inputs with:

```bash
python3 benchmarks/agent_supervisor/autonomous_meta_controller/validate.py
```

Full paired execution and promotion qualification belongs to APMC-018 and
APMC-019.  A missing provider or a held DuckLake activation is represented as
a typed unavailable/disabled capability, not replaced by a simulation and not
used to weaken the DuckDB-through-Quack operational authority.
