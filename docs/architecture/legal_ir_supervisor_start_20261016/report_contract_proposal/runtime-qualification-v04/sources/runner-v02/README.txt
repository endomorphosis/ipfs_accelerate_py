Explicit ordinary callable runner V02 — source-only proposal

This additive bundle changes only its copied proposed Grok runner. Actual owned
controller A and every earlier frozen generation remain unchanged. The baseline
is the proposed runner in ordinary-success-hook-v01, whose exact bytes are bound
by BASELINE.json. No test, provider, model, native, Docker or Git job was executed
by the author. AST parsing and source/hash/patch inspection are authoring checks,
not runtime qualification.

Public callable surface

  build_runner_parser() -> argparse.ArgumentParser
  parse_runner_arguments(explicit_argv) -> argparse.Namespace
  OrdinaryCallableInvocation(
      prompt: str,
      cwd: Path,
      environment: Mapping[str, str],
      temporary_root: Path,
      stdout_path: Path,
      stderr_path: Path,
      stdout_fd: int,
      stderr_fd: int,
      output_limit_bytes: int = 16 * 1024 * 1024,
  )
  validate_ordinary_callable_invocation(args, receipt_fd, invocation) -> None
  _run(args, receipt_fd, *, callable_invocation=None) -> int

The H successor owns the public effectful host wrapper. It receives an actual
immutable invocation, genuine same-process sibling native owners, and their
preacquired genuine C custody consumers. H calls the public read-only G validator
under its active concrete invocation/custody context, before its optional
known-unstarted before_run control. G._run repeats the same validation before
imports of route/provider execution helpers. G does not create native admission,
authenticate a successful outcome, alter canonical attempts, or accept a task.

The invocation snapshots the supplied environment into MappingProxyType; neither
caller dictionary changes nor dataclass field assignment can change that snapshot.
Prompt and environment are excluded from repr. Borrowed FD inputs cannot be
pickled and are never loaded from CLI, JSON, an environment variable or a receipt.
The constructor checks finite explicit argument bounds only; it does not confer
native or successful-outcome authority. Prompt is nonempty UTF-8, at most 2MiB;
environment is at most 1024 named strings and 256KiB; output is at most16MiB total.
HOME and all PATH entries are explicit absolute paths, relevant environment path
variables cannot be relative, and TMPDIR/TMP/TEMP equal the explicit temporary
root. The actual callable validator checks existing canonical unaliased paths,
private same-UID0700 temp root disjoint from workspace, exact FD/path inode
correspondence, and distinct initially empty owned regular0600 writable sinks.
The trusted caller owns those FDs, close, native admission, filesystem roots,
storage promises and final fsync/durability; G borrows and never closes them.

The parser construction changes no environment, descriptors, provider state,
route, process or working directory. Public parsing requires an explicit bounded
argv sequence and never reads sys.argv or ambient receipt environment. It writes
no help/error output to global streams; help and invalid arguments raise the
specific OrdinaryCallableRefusal instead. Module import retains the original
runner's accepted-package/source import setup; this bundle does not claim that
importing the full existing product package is free of initialization.

Supported callable path and CLI preservation

The callable path supports one ordinary Grok4.7 agent task, with explicit absolute
Grok binary and workspace. It rejects preflight, Codex fallback, route/recovery
JSON, invocation-ID/binding flags, failure nonce, supervised/declared receipt FDs
and receipt environment before route or provider effects. Both explicit and
ambient receipt controls are refused. The G validator also requires the exact
current H invocation object, not merely some active context. V01 primary-only
limitations remain: user-required Codex GPT6.1Sol high backup is pending, so this
source proposal does not complete production routing.

Callable execution uses the supplied prompt, cwd, immutable environment and
bounded output paths. It does not swap sys.stdin, sys.stdout/sys.stderr, os.environ
or tempfile.tempdir, and does not call os.chdir. Optional explicit temporary-root
parameters are threaded through ordinary prompt, home, Docker probe-config and
legacy watchdog lease allocations. The watchdog receives that same actual parent
in its own child TMPDIR environment and executes the exact pinned runner entry;
its startup receives no ambient CLI descriptor shortcut. Relevant shared helper
functions continue to receive explicit base_env/child_env. Unsupported protected,
preflight, streaming and fallback paths are refused before they can enter their
legacy global-state code. Container bootstrap environment clearing is code run
inside the isolated container, not a mutation of the trusted host environment.

The existing CLI main constructs the same original parser, keeps original
argument defaults, ambient receipt lifecycle and private _run invocation without
callable inputs. Ordinary helper default call shapes, primary argv, CLI streaming,
preflight, protected routing and fallback semantics are retained. In particular,
the original protected-route local named invocation is untouched; the new keyword
is callable_invocation. This source claim still requires the authored controls
and later genuine runtime qualification to be executed by the trusted root.

Bounded output and honest refusal

The callable primary transport uses the actual existing Popen/native/start-gate
path, explicit cwd/env and two PIPE readers. A main-thread select/read loop
checks actual H ordinary_provider_guard before every <=0.5s wait, drains stdout
and stderr concurrently, and writes only within the combined output budget.
Stderr quota-envelope retention remains128KiB, as before; total observed bytes
are counted separately. If observed output exceeds the finite capture limit,
stdout capture stops while both pipes continue draining under the finite native
deadlines. After actual wait/reap/returned observation, the callable path raises
callable_received_output_limit_exceeded. Output writer identity/offset drift,
sink overflow and write/read/guard exceptions also propagate and cannot supply
a success callback. Exact pipe readers close in finally on startup, guard and
output failures. This closes readers only; it never proves Docker/kernel/native
cleanup or releases custody. Sink bytes and diagnostics are raw private provider
output, not public evidence. No stdout/stderr is inherited by the callable model.

Creation and failure custody seams (require the H/C successor)

  ordinary_callable_context_required(invocation)
  ordinary_provider_guard()
  ordinary_private_allocations_created(original_TemporaryDirectory, prompt_Path)
  ordinary_container_creation_started(actual_lease, exact_create_argv_tuple)
  ordinary_container_created(actual_lease, verified_actual_CID)
  ordinary_allocation_close_permitted(actual_lease) -> bool

G passes the original TemporaryDirectory object to H before the lease factory,
so its finalizer cannot erase private custody paths after an unknown unwind.
Creation intent is observed immediately before the actual create subprocess;
known-created callback follows actual single-CID stdout and exact startability
checks, with H additionally checking cidfile correspondence. A rejected or
unknown creation/start state prevents G finally from closing the legacy lease
or deleting home/prompt. H/C retain original allocation objects, native consumers
and pending context. Missing/zero init PID or missing kernel scope from the real
immediate-start race must stay an actual failure, not a prewait/retry success.
G never retries startup, blindly closes unresolved leases, prunes Docker or
reconstructs native objects. H without an active handoff returns no-op/True at
these seams to retain default CLI behavior. The complete H/C source bundle must
be assembled and source-pinned with this runner before any callable use.

Authored qualification controls

40 unique unittest methods in test_callable_contract.py cover immutable finite
inputs, exact primary-only/refusal flags, effect-free parser output/errors and
bounds, original parser definition correspondence, protected-route shadowing,
explicit dual-pipe cwd, main-thread guard/read order, actual creation hook order,
private allocation close gating, and shared public/internal input validator.
They extract only selected definitions through stdlib AST; they import no product
module, create no native owner/process, and do not treat synthetic fd numbers as
grants. No methods have been executed by the author. These controls cannot prove
real owner admission, FD filesystem observations, Docker startup, known-stopped
cleanup, real provider callback, candidate acceptance or production readiness.

Next safe qualification is a root-owned finite genuine two-owner known-unstarted
call through actual H/C/G: output files and all writable roots precharged; real
consumers bound; G's actual read-only invocation validator passes; deliberate
source-pinned before_run exception occurs before G._run; no route/provider/Docker
allocation, native child attachment, callback or original attempt occurs; exact
consumer rows drain and explicit actual owners finalize. A separate future
non-model Docker qualification then tests actual create/start/pipes and unresolved
cleanup. Real provider/candidate/native acceptance and backup require independent
later qualifications. All64 canonical tasks and original two consumed attempts
remain unchanged by this bundle.
