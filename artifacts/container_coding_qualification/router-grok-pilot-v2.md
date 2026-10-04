# Router coding pilot results

Status: completed

Component pilot with four seeded repairs per arm; one observation per arm.
This is not a Terminal-Bench score or a full supervisor-daemon evaluation.

| Configuration | Passed | Wall seconds | Native tokens | Solver seconds | Token change vs serial |
| --- | --- | ---: | ---: | ---: | ---: |
| serial | True | 26.48 | 54169 | 0.000 | -0.00% |
| parallel | True | 17.84 | 66331 | 0.000 | +22.45% |
| parallel_proved | True | 21.59 | 66959 | 0.063 | +23.61% |
| parallel_proved_compact | True | 20.52 | 53663 | 0.057 | -0.93% |

Provider and model: `grok_cli` / `grok-4.6`.

Token totals include reported input and output tokens for every attempted call,
including failed attempts. Cached input is a subset and is not added again.
CLI system prompts and context are included in native usage. Differences between
single runs may reflect sampling, cache effects, or provider load; they do not
establish a general token-saving or speedup claim.
