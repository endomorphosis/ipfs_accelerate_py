# Qualified archive: first full trial refused at context admission

The first full trial `fix-code-vulnerability__iRSHBCs` received **official reward
0**. The supervisor did not complete the task. Source384 completed in 83.119s;
initial context took 141.498s and symbolic planning 31.976s, producing two goals
and one task with zero provider calls. Context then spent 33.594s before an
admission timeout. Doctor and provider invocation counts are both zero.

The attached primary-gate observation records `proof_memory_stall`: host memory
PSI was 12.11 percent against the unchanged 2 percent limit. The one visible
cgroup sample reports 0.0 percent memory PSI, with zero omitted scopes. Available
memory was 10230 MiB, reserved root memory zero, additional request memory
1024 MiB and required headroom 2458 MiB, so the recorded refusal was pressure,
not a headroom shortage. CPU and I/O readings were below their thresholds.
This identifies the limiting sampled scope and refusal reason, not the workload
that caused host stalls; it does not exclude earlier contribution by the trial.
The terminal gate was in backoff. The later 13.18-percent post-unwind memory
sample and the subsequently cleared host sample are separate observations, not
causal explanations or guarantees for a retry.

Supervisor failure was recorded after 218.591s; Harbor's agent execution was
220.351s and invocation 453.715s. The outer command returned zero after 455.161s.
That successful controller exit does not mean task success. The official
verifier ran and reported zero reward; its bodies are not exported here. Agent
input/output/cache token counts and cost are null. Zero provider calls describe
a failed prefix and do not establish a completed token score or an efficiency
advantage. A separately declared retry has its own evidence and does not replace
this result.

The run used qualified archive
`f8cd6fe9a3283a8df668394370b122743d77bc5ffa4ffcbd5b3ddf6f421d166c`
with the same source generation, reviewed intent, checkpoint, five-CPU/12-GiB
profile, 90-second Source384 budget, 245-second work cutoff and cleanup reserve.
The preparation's exact config hash and frozen source/task pins are retained.
The post-trial record confirms 61 unchanged frozen source pins, three unchanged
public task inputs, and absence of the exact first-trial container. A later
read-only package audit also confirms those pins and 119 prepared host source
pins. Worker cleanup returned zero; the separate remaining-processes field was
null, so no independent empty-process-list assertion is made.

The qualified preparation path does not grant proof authority: its unsupported
learned candidates remain unverified. This package establishes a completed
failed trial and its bounded admission evidence, not completion of the coding
task, admission recovery, or a matched-arm comparison. The backlog remains
18 of 32 closed.

Only bounded execution projections, safe command/exit records, input/config
hashes and public source-pin metadata are included. Raw configuration kwargs,
intent/source/model bodies, exception traces, credentials and hidden verifier
bodies are excluded. Retained-record and qualified-generation links refer to
external local artifacts, not missing package members. Recorded originals and
the source worktrees are unchanged by packaging.
