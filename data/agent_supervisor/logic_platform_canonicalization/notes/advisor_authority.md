# LPC-071 Advisor proposals vs proof authority

Advisors (models, ranking, Hammer candidates, embeddings, GraphRAG) may
propose formalizations, premises, plans, tactics, proof text, and
counterexample explanations.

They cannot mark a proposal proved. They may not:

- mark their own proposal proved
- raise evidence or receipt `semantic_authority`
- choose a trusted verification key
- skip required reconstruction
- approve production admission
- silently add assumptions
- remove a blocking proof obligation

`TacticianPlan` and `TacticianReceipt` always carry `semantic_authority=False`.
`TacticianReceipt.from_plan` binds the plan to the exact `policy.policy_id`
and rejects a mismatched policy. Goal metadata cannot claim semantic
authority.

Datasets logic owns semantic interpretation and receipt verification.
The supervisor owns scheduling and isolation only.
