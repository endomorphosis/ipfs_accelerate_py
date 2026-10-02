# Repository benchmark preparation

`repository_benchmark_preparation.prepare_repository_benchmark` prepares an explicit repository arm under the shared host reservation. Inventory, training splits, inference paths and proof contracts are separate immutable selections. Every source inventory entry remains visible; unselected training and proofs are not performed.

The four model policies are `model_off`, `pinned_parent`, `optional_training` and `required_training`. Actual pinned-parent inference consumes the original shared384 checkpoint. Optional training retains the parent if the child is incomplete or fails the development retention gate. Required training refuses without inference when that gate fails. Infrastructure corruption, cancellation and source/model drift remain errors; optional training is not an unrestricted exception fallback. No child is automatically promoted.

The training holdout is used for deployment selection here. It must not also be reported as an untouched final benchmark test. Model outputs remain independently checked proposals. The real qualification child reconstructed0/3, versus the parent3/3; it was retained as diagnostic evidence and never selected.

Preparation accounts scan, semantic projection, native persistence, training, inference, demand-driven proof and final validation as phases beneath one actual host parent. Queuing counts against the declared cooperative phase deadline. Native subprocess limits stay active. The preparation receipt commits exact timing bytes and the closing parent receipt reports released resources. Full daemon stage accounting and matched cold/warm benchmark measurements are separate gates.

Fourteen distinct controls pass using native owners and actual model/checker backends. Initial implementation/test defects and genuine shared-host admission refusals remain in the [evidence manifest](evidence/repository-benchmark-preparation-20261002/manifest.json). No official benchmark score or token advantage is claimed by these controls.
