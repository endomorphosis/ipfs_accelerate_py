# Qualified comparison with planning retained

The corrected-runtime pair passes every required gate: 74 for each arm, 20 cross-arm checks, and 46 independent accounting checks. Both original verifier rewards are 1.0; both native tasks completed with clean shutdown. Original native and complete model inputs reconstructed exactly, including the coding reply contract. Each arm retained one planning and one coding LLM session through `llm_router`.

| Arm | Input tokens | Cached input¹ | Output tokens | Total tokens | Coding token-count records² |
|---|---:|---:|---:|---:|---:|
| Baseline @1 | 242,507 | 176,896 | 4,485 | 246,992 | 6 |
| Compact @2 | 320,808 | 254,592 | 6,309 | 327,117 | 8 |

¹ Cached input is included in input tokens. ² Records are not API-call counts.

**The compact run used 80,125 more total tokens in this observation. No token-saving claim is supported.** The pair cost is 574,109 tokens. Including the five retained prior attempts gives 1,885,347 tokens across seven comparison attempts and thirteen native CLI sessions; earlier failures and failed audits remain recorded.

| Initial coding input | Baseline @1 bytes | Compact @2 bytes |
|---|---:|---:|
| Original native input | 62,443 | 61,172 |
| Complete model input | 59,731 | 57,160 |

The compact initial model input was 2,571 bytes smaller, while its native input was also 1,271 bytes smaller. This is not a transport-only measurement on identical native input. Both coding inputs included the same 506-byte reply instruction and requested the same 215-byte generation schema. Byte reduction did not translate into fewer total session tokens.

This is one qualified pair, not evidence of a repeatable or causal effect. Internal turns, solution choices, output lengths and cache occupancy can vary. The result supports keeping total native session tokens, task results and complete input audits as the gates for future improvements.

No prior receipt or failed audit was replaced. The review read bounded structured metadata and implementation hashes; no raw provider response, model thought, task solution, private database or hidden evaluator body was read.
