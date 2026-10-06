# Retained planning and coding transport comparison

Both fresh arms use the same frozen runtime and retain LLM planning and coding through the router.

| Attempt | Official reward | Native sessions | Input | Cached input¹ | Output | Total |
|---|---:|---:|---:|---:|---:|---:|
| First baseline: planning failure | 0.0 | 1 | 22198 | 12288 | 1539 | 23737 |
| Fresh baseline @1 | 1.0 | 2 | 286482 | 232192 | 5896 | 292378 |
| Fresh compact @2 | 0.0 | 2 | 328602 | 272128 | 7160 | 335762 |

¹ Cached input is already included in input tokens.

The first baseline consumed 23,737 complete native tokens before strict plan admission rejected an extra task title. Coding never ran. Compact-01 was prepared but never executed; it is not counted as a completed zero-cost trial.

The pair did not satisfy every comparison gate; its retained costs remain visible without a qualified saving claim.

Observed baseline minus compact total tokens: -43,384.

The compact coding CLI exited 0 after a 174.44-second session. The router then rejected its structured response with `response_envelope_binding_mismatch`. This is a strict response-envelope rejection, not a provider timeout.

The native task remained `in_progress`; the supervisor later exhausted its agent budget at 844.61 seconds. Cleanup recorded 0 remaining processes and worker cleanup return code 0. The later lifecycle timeout does not make the already completed CLI session's token counts incomplete.

One pair cannot establish a reliable or causal saving percentage. Internal model turns, solution choices, tool use, cache occupancy and output length can differ.

Prompt byte measurements cover complete initial task input, including appended advisories, and exclude provider system context and later internal history. Native cumulative token counters include the observed complete sessions.

The campaign total includes the failed first baseline and both fresh arms. Observed complete campaign tokens: 651877.

This review reads bounded metadata and implementation digests only. No model transcript, thought, task source, solution or hidden evaluator body is opened.
