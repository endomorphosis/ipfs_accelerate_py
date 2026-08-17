# LGSWF benchmark results

Generated: 2026-08-17T04:48:53Z

Control plane: DuckDB + Quack (authoritative).
DuckLake: optional non-authoritative projection; unavailable here.

## Observed modes

- Embedded one-writer DuckDB is not the live multi-writer control path.
- DuckDB + Quack control was observed at `quack:127.0.0.1:41307` (47 tasks).
- Live DuckLake was not started; reported as typed unavailable.

## Suites A-D

Sealed full-suite repetitions were not executed. Those cells are
typed unavailable/not-executed. Target numbers were not substituted.

Validate with:

```
python3 benchmarks/logic_governed_semantic_work_fabric/validate_results.py \
  --results data/agent_supervisor/logic_governed_semantic_work_fabric/benchmarks
```
