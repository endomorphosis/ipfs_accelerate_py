# Bounded applicability replay controls

The final scopes pass 40 distinct cases: 38 focused controls and two native
controls, with no failures, errors or skips. Their selected production and test
pins agree and remain unchanged. Checkout revisions name the base revisions;
working changes are identified by exact file digests, not represented as already
committed code. The complete Docker archive is qualified separately.

The captured-source control uses the existing tiny authored header fixture,
its real DuckDB capture and six actual Z3 obligations. A native isolated scheduler
reserves all CPU slots for 46 real seconds. Replay then succeeds after more than
45 seconds; a second blocked replay expires at the same original 55-second
aggregate deadline. All seven observed version/query processes are gone and no
leases or waiters remain. Admission limits stay at most 90 seconds and every
version/query execution limit is at most five seconds. The scheduler uses
explicit authored healthy telemetry: this measures reservation contention, not
recovery from actual host PSI or benchmark performance.

The other native control uses the existing model-free admitted runtime fixture.
It executes real constructor verification, START, owner bootstrap and STOP. A
real 2.1-second validation delay reduces the issued START allowance; the native
lifecycle receives the fixed enclosing deadline. STOP keeps its 20-second bound
and succeeds even after the caller's replay scope expires. This runtime fixture
does not itself execute a header solver or a benchmark task.

Fast controls include simulated clocks and narrow boundary doubles. They cover
the ordinary 45-second ceiling, explicit local-only limits up to 120 seconds,
smaller inherited limits, thread handoff without renewal, cancellation, ongoing
pressure/headroom refusal, and refusal before the lifecycle owner is invoked.
These controls do not establish a whole-program proof, training quality,
learned-source correctness, full task completion, token savings, or matched-arm
advantage.

The original pre-implementation API failure and two earlier overlapping passing
runs remain retained. Their counts are not added to the final 40. Public XML is
an explicit projection of identities, counts, elapsed times and bounded native
properties; original XML/command/exit digests identify local retained originals.
Raw stdout, solver/model/source bodies, private stores and credentials are not
included. Commands preserve selected environment and normalize workspace paths.
