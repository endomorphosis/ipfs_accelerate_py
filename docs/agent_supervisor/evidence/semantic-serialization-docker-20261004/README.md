# Semantic serialization: ordinary Source384 Docker qualification

Archive `f8cd6fe9a3283a8df668394370b122743d77bc5ffa4ffcbd5b3ddf6f421d166c`
passes ordinary Docker qualification under the unchanged five-CPU/12-GiB
profile and 90-second Source384 work deadline. Source384 takes **69.851s**,
initial context **125.658s**, warm observation **9.320s**, and the probe
**146.743s**. Preparation is 10.781s, deployment 172.836s and the controller
385.750s. These nested times are not additive. A single successful run does not
isolate a causal runtime speedup or establish consistent admission recovery.

The deployed source commits are accelerate
`e80fe6a59cacf4bb61386f0823cbd8545274173b` and datasets
`28b4a43c04e89c889b8fce7910e54fb8f67b05ff`. All 26,932 archive members pass the
stream hash/size/mode/ownership audit. The sole runtime delta is
`semantic_index/models.py`; checkpoint, embeddings, solver, reviewed intent,
task inputs, source/proof fences, resource gates and deadlines are unchanged.
The new serializer reuses one freshly generated native identity payload within
its call while preserving the public output and custom serializer path.

The native numerical worker executes the pinned checkpoint and local GTE model.
Across 31 Python files and 944 function units, 128 units are selected: 127 yield
unsupported, unverified candidates and one is deferred at the token limit.
Another 737 units are deferred by selection budget and 79 by unsupported
normalization. No learned source contract or proof authority is granted. The
native report records zero provider calls, training steps and downloads; its
`neural_inference_replayed` flag is false. Warm source observation is reported
separately from executing numerical inference.

Exact source/public-input pins and container cleanup pass. This package records
preparation qualification, not a full task result: no official verifier ran
inside this qualification and no reward or completed token score follows from
it. A separately retained trial must establish the task outcome. The backlog
remains 18 of 32 closed; no matched-arm advantage is claimed.

Only bounded execution metadata are exported. Raw source/model/credential and
hidden-verifier bodies are excluded; retained-record hashes refer to local
artifacts rather than package members.

The immutable recorded run-admission and qualification-command metadata retain
this historical component-scope wording:

> Seven actual local checkpoint/GTE source-unit inference and replay controls, with no provider call or training.

The precise scope is seven source-unit test cases: one bounded-preparation
case, five parameterized source-map tamper cases, and one actual pinned-
checkpoint/GTE inference case. The actual case uses a fresh numerical-worker
subprocess, asserts one model load and zero provider calls, then verifies warm
and reopened-registry replay in the same host process without another numerical
worker. It is not seven independent model runs or fresh-host-process replay.
The v2 joined-component and v3 pressure-component packages clarify these labels
without changing test identities, command/exit/XML records, source pins,
recorded admission/build metadata, or runtime qualification outcomes.
