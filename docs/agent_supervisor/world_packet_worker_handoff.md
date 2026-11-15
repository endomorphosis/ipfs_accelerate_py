# Read-only worker handoff and authenticated metadata observation

This stage creates a sealed owner-side read-only handoff from the original live
native task and allocated-source assignment. Only the allocated directory mode
changes from 0700 to 0750 through its held descriptor. Directory UID, GID and
inode, selected source files, ancestor directories, Git controls, Portal files
and native authorities remain under their existing custody. The old assignment
and allocation handles remain stale; serialized observations cannot reissue them.

The worker receives an observation-only abstract Unix endpoint. It can request
current assignment metadata through one closed operation. The broker exports no
source body, SQL interface, owner credentials, native grant, compiled world
packet, execution capability or provider permission. The directory transition
grants real group-read/search of the allocated root; selected mode0644 files are
readable if that directory is reached. The broker does not deny or
completely mediate that filesystem access. The broker requires actual
SO_PEERCRED UID1001/GID1000, process birth and ancestry beneath the exact original
literal `Popen` command. One spawn and one worker are admitted, with bounded
expiry, requests and bytes. Native task, source, control and lifecycle changes
invalidate later observations. This is not world-packet dispatch readiness.

Root runs the finite inner experiment only in a fresh owned offline container
using the independently selected local image
`sha256:771900c18b663b33a5bf96e066346b2d5c842506523efb167aebe8d3a99b5c07`.
The outer driver independently checks image/container identities and removes
only its owned container. Its caller must already hold the host resource lease
under the existing pinned policy. Container-local fixture accounting is scoped to native
fixture operations; it does not create another host resource authority. No image
pull, package installation, model, provider, target-code execution or training
is part of qualification.

Final dependency qualification uses isolated locked checkouts at datasets
`8fd82826e8ee935270457acfeee185054eb44aed` and kit
`40c98c71b95be750509ffdb21e7e5494d91b4e79`. The earlier shared checkouts had
advanced; their histories, worktrees and locks were preserved. The final native
tests and Docker experiment use the new fixed copies. The current datasets
history includes the completed F16 actor-context ranking experiment. Its report
records actor accuracy improving from 156 to 158 of 168, but complete IR stays
154 of 168, below the required 155; scientific acceptance and default promotion
remain false. These are reported historical results, not a model run or a
checkpoint acceptance in this qualification.

The inner scenario uses the retained native allocation fixture. DuckDB, typed
commands, the process-serialized claim, Git allocation and lifecycle are real.
`FakeQuackTransport` remains an authored native listener seam, so actual Quack
TCP qualification stays false. Trusted fixture setup provisions only the
selected authored program files before original assignment capture.

Root installs the fixed root-owned helper
`/usr/local/libexec/ipfs-world-handoff-observe-worker` with mode0555 and grants
only its no-argument sudo transition in the disposable container. Its absolute
Python interpreter runs with `-I`, enables no-new-privileges, and imports selected
modules from the pinned read-only `/opt/ipfs-supervisor/source`,
`/opt/ipfs-supervisor/datasets` and `/opt/ipfs-supervisor/kit` snapshots plus image
standard libraries and installed dependencies. This is not a complete import census.
The experiment exports `WORKER_ENTRY_SOURCE` and its exact literal command for
root setup. Helper contents and protected ancestors are checked before launch.
The worker wrapper imports the observer client directly; it never imports the
experiment or owner fixture modules, and does not depend on the `benchmarks`
namespace. The client implementation uses standard libraries.
Same-UID or unavailable launcher profiles remain refused and unqualified.
The literal sudo launcher command and actual shebang interpreter-plus-script
worker command are independently pinned and checked against their respective
kernel processes.
The trusted `Popen` receives the allocated workspace as its cwd. Cross-UID
kernel cwd is not independently verified; `worker_workspace_verified` remains
false, and the receipt does not prove source-mapped containment.

After the real `Popen` is registered, its descriptor and independent descriptor
hash cross the anonymous stdin pipe. The fixed client waits for at most two
literal `observe` commands. The experiment retains one authenticated metadata
response, changes the actual native task body, and requires a second observation
to refuse while the child is still alive. It restores the fixture row for native
teardown, reaps the owned monitor, closes the observer and handoff, and performs
bounded held-directory cleanup. No worker receives the private database.

```text
python -m benchmarks.agent_supervisor.container_coding.terminal_world_worker_handoff_experiment \
  --scratch-root /absolute/owned/private-scratch \
  --output /absolute/new/private-report.json
```

The bounded report exports source pins, response hashes, kernel peer identity,
permission modes, namespace observations and finite checks. It exports no raw
source, prompt, SQL row, private path, owner token or grant. `validate_report`
requires an independently held FULL canonical report hash; it checks custody
and the closed observation contract, without rerunning native operations.
Failed profiles and uncertain cleanup retain private scratch and remain
unqualified. The prior cleanup owner is reused unchanged: root names are never
removed, and only censused contents under held directory descriptors are deleted.

The outer runner is
`benchmarks.agent_supervisor.container_coding/run_terminal_world_worker_handoff_docker.py`.
It requires an existing native host lease and checks the cached image, engine
resource limits, immutable container ID, unique run label and source mounts.
Private results and scratch belong to UID1000 inside the container. The runner
exports them into the private host artifact directory after the container stops,
then removes only the verified owned ID. A failed or uncertain export preserves
the owned container for recovery. This avoids depending on bind-mount ownership
across container user namespaces. Setup uses the actual supervisor account's
environment; the selected runtime snapshots are mounted read only.

Local final qualification passes 156 distinct tests: 86 native handoff cases,
36 observer protocol cases and 34 report-contract cases. The real Docker
scenario passes 12 checks, including one authenticated UID1001 observation,
native task-body mutation refusal while the child remains alive, exact worker
reaping, observer shutdown and native-context closure. The FULL canonical report
hash is `35584ba345825b0b1124e5371dcd2707ad2aa8d29458ba36c3de63f003cba8d4`;
an independently held copy of that pin validates the exported report.
Private evidence is retained at
`/home/barberb/lift_coding/artifacts/world-packet-worker-handoff-20261014-01`.
Earlier failed tests and container runs are retained separately. Every admitted
run releases its host lease and preserves the existing resource policy.

Authenticated worker metadata does not establish compiler correctness, accepted
world state, source semantics, provider ingestion, task completion, token savings
or general hostile-worker containment. Selected file witnesses do not cover the
entire repository. Those authority and whole-repository flags remain false.
