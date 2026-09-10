# SAWM native dispatch observation and pause

The SAWM owner exposes a bounded, process-bound operational pause channel beside
its existing task-status observation channel. This capability requires the new
code in an independently admitted native owner, coordinator, supervisor and
daemon. It cannot be installed into an older running owner by changing a status
file or borrowing that owner's credentials.

From the admitted checkout, use the existing SAWM operator and configuration:

```sh
python scripts/ops/agent_supervisor/semantic_addressed_world_model.py --config config/agent_supervisor_semantic_addressed_world_model_scheduler.json drain-request
python scripts/ops/agent_supervisor/semantic_addressed_world_model.py --config config/agent_supervisor_semantic_addressed_world_model_scheduler.json drain-status
python scripts/ops/agent_supervisor/semantic_addressed_world_model.py --config config/agent_supervisor_semantic_addressed_world_model_scheduler.json drain-release --request-id REQUEST_ID
```

The response to `drain-request` only acknowledges that the owner retained a
request. `dispatch_pause_observed` becomes true only after the current master
and every configured lane acknowledge the new epoch, with each daemon at its
preclaim boundary. Supervisor replacement, daemon replacement, stale heartbeat,
foreign process identity, and a newer epoch invalidate earlier acknowledgments.
Release requires the exact current request ID. Losing a response does not erase
the pause; `drain-status` recovers its ID. Listener recreation in the same owner
retains the request, epoch and replay state.

The coordinator waits before new lane launches and suspends recycle/reassignment
actions during the pause. A daemon checks the channel before a new canonical
claim. Missing or unverifiable observation defers new work. Existing native
reconciliation and resumption of already admitted attempts continue; those paths
may execute a retained provider obligation. Their phase is reported separately.
The existing explicit shutdown, finite run-window and native process-lifetime
policies retain their behavior. The pause itself sends no signals.

Native actors use their own inherited credential and verified launch admission.
The owner validates the actual kernel peer UID/birth, current native master,
owner generation and source/configuration scope, both native FLOCKs, and the
canonical DuckDB POSIX WRITE lock. A supervisor registers only its actual
managed child using the existing birth-bound child identity. An ancillary child
cannot acknowledge a lane. The public operator holds no task credential; its
same-UID pause requests must match the exact current owner/source/configuration
and master birth. Requests and replies are bounded and contain no credentials.

Neither a pause acknowledgment nor all-lane pause observation proves that a
callback, claim, pool lease or provider effect has closed. Replies explicitly
report callback custody as unknown and grant no task, completion or source
transition authority. An admitted successor still needs the existing independent
native closure and source-admission evidence. This capability does not relax
the M70 ready-owner stop refusal or manufacture a terminal receipt.

The tests qualify disposable real DuckDB writer custody, kernel-bound native
process trees and actual shared-child registration, native coordinator pause
behavior, canonical new-claim deferral and retained-attempt resumption. They do
not establish a live migration or qualify a completed taskboard. The current
source candidate remains separate from live SAWM generation 48/source `25f130c`.
