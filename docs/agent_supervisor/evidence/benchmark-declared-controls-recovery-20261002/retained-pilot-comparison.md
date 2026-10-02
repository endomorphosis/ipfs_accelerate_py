Retained pilot trials. These observations do not establish a benchmark advantage.

Token cells marked ≥ are observed cumulative lower bounds; complete totals remain unknown. Explicit native session completion is required for complete totals, including for legacy receipts. Verifier reward and supervisor task completion do not establish usage completeness.

| Arm | Trial | Reward | Native completion | Env setup s | Agent setup s | Agent s | Verifier s | Input | Cached | Output | Total | Outcome |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Native Codex harness | fix-code-vulnerability__VfZu8VN | 1.000 | N/A | 3.081 | 131.724 | 53.017 | 2.361 | 227059 | 193792 | 1571 | 228630 | official_verifier_passed |
| Supervisor: same planner, no context bundle | fix-code-vulnerability__aAjF5Sy | 0.000 | True | 3.297 | 260.922 | 122.529 | 2.635 | 190912 | 158592 | 3582 | 194494 | official_verifier_not_passed |
| Supervisor: full indexed context | fix-code-vulnerability__z4cVfSR | 1.000 | True | 3.301 | 310.569 | 236.093 | 1.980 | 21654 | 11264 | 1095 | 22749 | official_verifier_passed |

Declared controls are checked separately from reported model identity. A match is a configuration comparison, not runtime enforcement or campaign qualification.

| Trial | Reported identity matches | Declared controls match | Differences |
|---|---|---|---|
| fix-code-vulnerability__VfZu8VN | True | unknown |  |
| fix-code-vulnerability__aAjF5Sy | True | unknown |  |
| fix-code-vulnerability__z4cVfSR | True | unknown |  |

Unsuccessful attempt costs are retained separately. Values below are observed subtotals; unknown calls or counters can make the actual total larger.

| Arm | Unsuccessful trials | Known input | Known cached | Known output | Known total | Trials with unknown totals |
|---|---:|---:|---:|---:|---:|---:|
| Supervisor: same planner, no context bundle | 1 | 190912 | 158592 | 3582 | 194494 | 0 |

Cached input is already included in input. Missing values remain unknown. Provider-reported dollar costs are unavailable when shown as null in JSON; estimated Harbor costs are excluded.

No-index retains the same planner and native supervisor, with no context bundle. Setup, agent execution, and verification remain separate timings.

Provider invocation observations retain failure and timeout information independently of the eventual task outcome.

| Trial | Invocation | Phase | Status | Error | Timeout s | Native session complete | Input | Cached | Output | Total |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|
| fix-code-vulnerability__aAjF5Sy | db460b0c81be4e9d9ad72d2a18794e01 | planning | provider_returned | unknown | 90 | True | 17976 | 11264 | 1203 | 19179 |
| fix-code-vulnerability__aAjF5Sy | 6154739fecdb47e1b97e59d76eca392d | coding | provider_returned | unknown | 178 | True | 172936 | 147328 | 2379 | 175315 |
| fix-code-vulnerability__z4cVfSR | 3bba273ba54540b691a8e3a102d8c526 | planning | provider_returned | unknown | 90 | True | 21654 | 11264 | 1095 | 22749 |

Runtime archive provenance is projected from retained agent metadata. The comparison implementation has its own source SHA256 in comparison.json; it may postdate these immutable runtimes.

| Trial | Runtime archive SHA256 |
|---|---|
| fix-code-vulnerability__aAjF5Sy | 738b6c05790a65f218d39e9ae869d6fcd3cf56a39ebc0693a38ee0f42ace4231 |
| fix-code-vulnerability__z4cVfSR | 738b6c05790a65f218d39e9ae869d6fcd3cf56a39ebc0693a38ee0f42ace4231 |

Supervisor phase observations are already inside agent time. Cold initial indexing is included; these values are not added to the total or to nested helper timings. Missing phases remain unknown.

| Trial | Prepare s | Cold initial context s | Planning s | Admitted context s | Doctor s | Post-publication refresh s |
|---|---:|---:|---:|---:|---:|---:|
| fix-code-vulnerability__aAjF5Sy | 10.127 | unknown | 31.732 | unknown | unknown | unknown |
| fix-code-vulnerability__z4cVfSR | 10.095 | 60.811 | 46.671 | 11.786 | 13.286 | 64.466 |

Indexed preparation is charged to agent time. These counts describe prepared context; they do not by themselves prove worker consumption or repair execution.

| Trial | Initial indexes reused | Vector rows replayed | Full capsules | Worker capsules | Worker semantic bytes | Doctor eligibility status |
|---|---|---:|---:|---:|---:|---|
| fix-code-vulnerability__z4cVfSR | True | 426 | 531 | 1 | 27276 | abstained |

Doctor selection, worker materialization and context refresh remain separate from native completion and verifier reward. Embedding subtotals retain measured failed work when complete accounting is unavailable.

| Trial | Implementation route | Doctor status | Worker materializations observed | Refresh status | Local embedding calls | Known local call subtotal |
|---|---|---|---:|---|---:|---:|
| fix-code-vulnerability__aAjF5Sy | model_router | unknown | unknown | unknown | unknown | unknown |
| fix-code-vulnerability__z4cVfSR | doctor_contract_candidate | candidate_ready | 1 | refreshed | 16 | 16 |

Model-input audit reconstructs retained context and exact task-prompt bytes. It does not cover provider system context or establish task correctness.

| Trial | Audit status | Any native input verified | Any model input verified | All observed coding inputs verified |
|---|---|---|---|---|
| fix-code-vulnerability__aAjF5Sy | unknown | unknown | unknown | unknown |
| fix-code-vulnerability__z4cVfSR | unknown | unknown | unknown | unknown |

Reported task-prompt sizes are bytes, not token savings. Router bytes include semantic translation and any Doctor residual advisory; model bytes also include the workspace instruction.

| Trial | Invocation | Phase | Native bytes | Router bytes | Model bytes | Identifier mappings | Residual advisory bytes |
|---|---|---|---:|---:|---:|---:|---:|
| fix-code-vulnerability__aAjF5Sy | db460b0c81be4e9d9ad72d2a18794e01 | planning | 16521 | 16521 | 16521 | unknown | unknown |
| fix-code-vulnerability__aAjF5Sy | 6154739fecdb47e1b97e59d76eca392d | coding | 8131 | 8131 | 8972 | unknown | unknown |
| fix-code-vulnerability__z4cVfSR | 3bba273ba54540b691a8e3a102d8c526 | planning | 25898 | 25898 | 25898 | unknown | unknown |
