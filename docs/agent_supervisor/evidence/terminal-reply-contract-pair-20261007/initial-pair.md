# Retained planning and coding: current reply-contract pair

Both runs passed the original verifier (reward 1.0), completed the native task, and recorded clean shutdown with zero remaining processes. Planning and coding both remained LLM sessions through `llm_router`; each coding run selected the same `ordinary-completion@1` acknowledgment contract.

| Arm | Input tokens | Cached input¹ | Output tokens | Total tokens | Coding token-count records² |
|---|---:|---:|---:|---:|---:|
| Baseline @1 | 285,149 | 219,136 | 5,426 | 290,575 | 7 |
| Compact @2 | 362,131 | 294,656 | 6,655 | 368,786 | 9 |

¹ Cached input is included in input, not added again. ² Records are not API-call counts.

The compact run used **78,211 more total tokens** in this observation. Both native usage totals are complete. The two-run cost is **659,361 tokens**; including the three retained prior attempts (651,877) gives **1,311,238 tokens** across the five listed comparison attempts.

The complete initial coding input was 58,535 bytes for baseline and 58,251 for compact: 284 bytes smaller for compact. Their native inputs were 61,247 and 62,263 bytes, respectively, so this difference is not a transport-only measurement on identical native input. Both included the 506-byte reply suffix and requested the same 215-byte generation schema. Smaller initial input did not yield fewer total tokens.

The benchmark comparison remains **unqualified**, although the independent accounting review passed all 46 factual checks. Each arm fails exactly these gates:

- `complete_exact_coding_context_audit`
- `coding_reply_full_input_audit_includes_contract_before_after`
- `coding_hashes_match_reconstructed_inputs`

All cross-arm control gates pass. The frozen auditor dropped the owner-authored `provider` field while projecting its subprocess request; its new reply-contract parser therefore rejected the receipt. The original audit remains `unknown` with `InvalidReceipt`. Restoring that metadata field proves the parser defect; it does not reconstruct the complete model input. No failed native audit, receipt, preparation, frozen source, or comparison gate has been replaced or relaxed.

This pair establishes no qualified or reliable saving percentage. It supports the observed costs and outcomes above. No raw provider response, model thought, task solution, private database, or hidden evaluator content was read for this review.
