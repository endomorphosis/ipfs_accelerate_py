# LGCVF release disposition

## Evidence

- Formal plan CID: baguqeerabxn5kkewz44v4chz6vbt3kcozfj4rvhh4gpdp54645blhemhvloq
- Qualification result CID: baguqeeramef5roixvldnw5f3bo4oglssuqvqqmyjgogn275tjqxyebnogbtq
- Qualification authority CID: baguqeeratz3jtsy4vtif5cd534zf3telfi6qffbdq5zw4rvamges3vhdveba
- Qualification suite node IDs: ["baguqeera3zvgtvi5l3llqyyo7wr2nzz5cx2zuovoduqrza77vyzgoxy57ksq","baguqeerawyxjp67p656suousq2s2yoqpnagcljminuhw6ooto6mz23zec7ma","baguqeeraynd5gnhnlb7gfytao7zvmjsaxqk3zt3bbpx5ma7e3mzgv6wdohya","baguqeerar533oighzztc5b5bazgduez2ksn66mzh2vu6c6a6gchquvkqlf2a"]
- Benchmark result CID: baguqeerap4uxygafg7mwww22f3ogq7gbqh2igumha7wuabo3mgeximihqauq
- Benchmark authority CID: baguqeerajuq4m4rpsm55yhgzx22ikl2pnjazyvirbb5iiy4dw6lqoe5f2ixa
- Evidence cohort: hermetic_local_execution

## Disposition

- Disposition: partial
- Task implementation: incomplete
- Test success: passed_hermetic
- Objective completion: incomplete
- Release qualification: not_qualified
- Production authorization: not_authorized
- Threshold comparison: [{"comparison":"equal","disposition":"met","observed":0,"reason":"","target":0,"threshold_id":"zero_safety_floor_violations"},{"comparison":"equal","disposition":"met","observed":0,"reason":"","target":0,"threshold_id":"zero_critical_omissions_accepted"},{"comparison":"at_least","disposition":"missed","observed":0,"reason":"","target":5000,"threshold_id":"median_context_reduction_bps"},{"comparison":"at_least","disposition":"not_evaluated","observed":null,"reason":"both fixture routes made zero model calls; a repeated task with a nonzero baseline is required to measure displacement","target":5000,"threshold_id":"warm_cache_model_call_reduction_bps"},{"comparison":"at_least","disposition":"met","observed":10000,"reason":"","target":2500,"threshold_id":"symbolically_closable_deterministic_route_share_bps"},{"comparison":"at_least","disposition":"met","observed":10000,"reason":"","target":8000,"threshold_id":"unaffected_proof_test_reuse_bps"},{"comparison":"equal","disposition":"met","observed":true,"reason":"","target":true,"threshold_id":"accepted_patch_quality_not_lower"},{"comparison":"at_least","disposition":"missed","observed":3,"reason":"","target":12,"threshold_id":"representative_task_class_coverage"}]

## Blockers

- External authority gate: blocked_external_authority
- Manual authority gate: blocked_manual

## Limitations

- Limitations: ["hermetic fixture and local test evidence only","candidate-authored tests are corroborated but never self-certifying","all judged suites run from exact copied inputs with checkout writes and network denied","the protected-input root excludes only declared generated evidence outputs","external qualification and operator authorization remain unavailable","this is a hermetic local fixture, not a representative maintenance suite","threshold misses and unavailable measurements remain visible","no live-model, remote-model, external-verifier, release, or production evidence is aggregated"]
