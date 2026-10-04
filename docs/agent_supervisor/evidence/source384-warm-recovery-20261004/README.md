# Qualification-only Source384 admission recovery

The final disjoint scope contains 46 helper/probe/report controls, 50 canonical
qualification boundary controls and one real native scheduler/checkpoint control:
97 passed with no skips and stable selected source pins. The retained first run
had 40 passes and one fixture failure because the selected receipt is now copied;
that assertion was corrected to check equal content. It is historical, not an
additional successful control. Docker calls in the caller group are mocked.

The native control performs one actual pinned checkpoint/GTE inference and
captured-header preparation with Z3, then revalidates the same immutable source,
model and receipt twice. An isolated native scheduler sees authored pressure of
11 percent for 91 real seconds, followed by healthy readings. The first admission
wait times out after about 90 real seconds; the second validation succeeds after
the bounded backoff, without neural inference or renewed aggregate time. Lease
and waiting counts are zero afterward. This is real elapsed scheduler recovery
under controlled telemetry, not proof of recovery from actual host PSI. Fast
boundary cases use authored clocks; they are not elapsed performance measurements.

The explicit qualification policy permits at most two validation-only calls of
at most 90 seconds, with one requested five-second backoff, under the smaller of
180 seconds and the original 600-second probe deadline. Defaults remain one call
and 90 seconds. The unchanged local 10-percent PSI threshold, capacity/headroom
checks and recovery pacing still apply. Cancellation, malformed/contradictory
observations, source/decoder failures and expired deadlines cannot gain retry
permission. Initial context and inference are never retried. Complete-call timing
includes admission and validation; the split remains unmeasured. Runtime work and
START/STOP budgets are unchanged. Metadata confers no proof or execution authority.

Commands, normalized environment, exact selected file pins, exits and closed XML
projections are retained. Original XML bytes are identified by digest. Tracebacks,
stdout, source bodies, private stores, credentials and model files are excluded.
These controls are not a Terminal-Bench score, throughput or token comparison.
