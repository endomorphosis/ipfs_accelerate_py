# Full trial refuses context replay for memory headroom

The qualified bounded-observation archive executes one fresh official Harbor
trial, fix-code-vulnerability__QWowUpS. Official reward is 0.0. Source384
completes in 77.595s; the driver records preparation 11.387s, initial context
136.021s, symbolic planning 32.598s and context 42.394s. Planning admits two
goals and one task with zero provider calls. Context replay then refuses a
child resource lease before Doctor, coding dispatch or publication.

The attached request-local primary gate records 8593 MiB available against
6144 MiB root reservations plus 2458 MiB required headroom, a 9 MiB deficit.
CPU, I/O and memory stalls are below thresholds in that sample. The final
primary decision is its backoff, followed by the lease timeout. This identifies
the observed gate refusal, not the causal composition of charged memory. The
separate post-unwind sample shows 8693 MiB available and no active leases.

The failing path is bind_admitted_context -> load_initial_context ->
validate_source384_context -> validate_shared_parent_units -> observe_current.
The last call refuses admission before AST reconstruction. Earlier batch-read
query savings do not bypass that gate. Later inspection found completed
construction objects still live while a fresh loader runs; any subsequent
lifetime correction belongs to a different generation and is not executed here.
The semantic view is a small root/reader wrapper, not all 6261 graph blocks;
its release alone is not evidence of recovering the 9 MiB shortfall.

The unchanged limits are five CPUs, 12 GiB, a 90-second Source384 deadline,
a 245-second work cutoff and 40-second cleanup reserve. Supervisor exit is a
failure after 223.031s even though the outer Harbor/controller returns zero.
Harbor setup takes 220.158s, execution 224.717s and verifier 1.896s; controller
473.885s. These are nested scopes. Worker cleanup returns zero, native remaining
process count is unavailable because native startup was not reached, and the
exact trial container is absent. Source/task pins are unchanged.

Zero provider calls describe a failed prefix, not a completed token score or
matched-arm advantage. Reviewed-intent cost is unavailable and excluded. The
checkpoint's unsupported learned candidates remain unverified. Only official
reward, permitted public-output receipts and bounded runtime metadata are
exported; verifier, model, prompt and credential bodies are excluded.
