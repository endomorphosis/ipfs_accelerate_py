# Residual Intelligence final merged-tree report

This is a nonauthoritative, bounded report of the final merged implementation
snapshot. It cannot promote an expert, accept a proof, or complete a task or
goal.

## Exact lineage and implementation surface

- Start: commit `84a056e41e48a81d4484be43840196578d6c87da`, tree
  `40f0771e77d394ac91d92cc1edb02f7860f6131b`.
- End: commit `5e9db00a4e9fbbb9cba00eb37e10b270fedcc0c0`, tree
  `59b11091b143293b028390b8c949b0ee7c21f6f5`.
- Scope: the end tree is the final merged implementation snapshot immediately
  before this report; the report cannot recursively qualify itself.
- The joined surface covers the residual contracts, corpus rights and semantic
  splits, benchmark, checkpoint adapter, runtime packaging, adversarial
  adapter, promotion/rollback gate, drift monitor, and the three declared
  release-report outputs. The machine report lists the exact files and symbols.

## Corpus, architecture, and experts

The only published admission is
`baguqeera3dp4lnpqc6wgksk5erxj54dt5sxhlp6opf2z5xyz6qvbg45abf2a`.
It is explicitly `training_unavailable`, despite a passing synthetic-fixture
leakage audit (one group each in training, development, held-out, and
adversarial partitions; zero cross-partition groups). Its rights root is
`baguqeerappawi4jtwvs3jfkoehdfc2xsh64lszuwjg6xzplgmckpg4ltjxda` and
its split root is
`baguqeeradbwccksw3vppbhz3xiu2l55isclxhi2xixthngpyeiy422jhmhzq`.

There is no packaged expert, admitted learned tokenizer, candidate checkpoint,
or simulated checkpoint. Consequently there are zero registered experts and
zero recorded counts for every closed expert disposition. No expert is routable.

## Metrics, costs, proofs, and validation

The frozen benchmark population is 384 cases (24 families × four partitions ×
four required case kinds). Both before and after populations preserve that 384
denominator, but zero cases were executed on this tree; every disposition and
accept outcome is therefore zero rather than an omitted denominator.

No current-tree paired benchmark, proof receipt, live GPU qualification,
promotion CAS, rollback CAS, or amortization measurement was run. Cost
denominators explicitly retain 384 frozen cases and zero local, remote,
validation, training, shadow, human-review, and rollback events. Break-even is
not applicable because no training/evaluation cost or per-use saving was
observed. The report contract itself is validated by
`test/api/residual_intelligence/test_release_report.py`; that is not a proof or
promotion receipt.

## Eligibility, rollback, and blockers

Promotion eligibility is false. Every conjunctive promotion gate is unsatisfied
or fixture-only, so no aggregate score can compensate for absent rights,
lineage, privacy, safety, quality, efficiency, autonomy, or amortization
evidence. Drift is not observed because no expert is registered; the required
action is to retain candidate-only behavior and requalify after admitted
evidence.

There is no promoted expert route to restore. The exact rollback target is
`no-promoted-expert-route`; revoke the report root and regenerate it from tree
`59b11091b143293b028390b8c949b0ee7c21f6f5` if evidence is corrected. Never
rewrite evidence or promote from a report.

Blockers are: `training_unavailable`, no admitted production corpus, no learned
tokenizer, no candidate checkpoint, staged benchmark, no current-tree promotion
evidence, and unavailable live GPU qualification.

Unsupported claims: `learned`, `verified`, `safe`, `autonomous`,
`token-efficient`, and `production-ready`.
