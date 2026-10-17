This additive disabled native host successor closes repeated productive entry
through the already bound original daemon. The original host, daemon, state,
canonical board, cap2, counters and prior frozen proposals remain unchanged.
GitHub qualification remains skipped. No job has been run by this author.

The existing execute_daemon_once freshness check is insufficient by itself.
Pending custody deliberately keeps the actual ContextVar and daemon attribute
bound. A caller can call public daemon.run_once again without reentering that
outer host method. V03 then queried the original borrowed database and renewed
the phase before discovering no explicit sticky pass denial. Its repeated
require_attempt_profile checks also called guard_pass_entry, so simply making
that guard one-shot would incorrectly reject valid checks within the same pass.

V04 separates these boundaries. execute_daemon_once latches its execution attempt
before its first fallible phase check. guard_pass_entry denies pending custody,
disposition or a previous public pass attempt, then latches before any profile,
source, borrowed database or phase checks. Its validated flag is set only after
all original profile checks return. A failed first check remains spent; no reset
is added. The original actual source pin remains4984370456bd304dd9d89550c05e2d364fd9f0ca4849db560cb10caddfe3f407.

The public daemon resolves its host before calling guard_pass_entry. Therefore
the accessor now performs only the exact actual class/binding/PID/thread checks.
Those predicates are factored unchanged into _binding_identity. The original
borrowed connection validator remains intact inside _identity. Guarded methods
call it after sticky denial; this avoids an earlier SQL witness on reentry.

_require_validated_in_pass checks the actual host's execution/entry/validated
latches and pending/disposition denial before pure binding identity. Attempt
profile checks use that helper and the original repeatable profile validator.
dispatch_provider, native owner acquisition, storage charging and the actual
success freezer callback check it before their original productive effects.
An ordinary original success callback remains allowed during its validated pass;
pending/disposition state cannot be used to dispatch, allocate or freeze again.
These necessary in-memory checks create no capability from booleans or JSON.

Cleanup is separate. observe_pending, original finalizer registration/drain,
handle_attempt_exception, settle_retained_finalizers and borrowed-scope closeout
keep their original bodies and remain callable on the retained actual host.
No pending/disposition flag is cleared, and no cleanup result becomes acceptance.

Thirty authored cold/source cases exercise repeated-refusal boundaries, exact
original failure identity, no repeated phase renewal, pre-effect dispatch refusal,
monotonic latches and exact original producer/custody body preservation. Every
executable probe is negative-only and must refuse or raise its original injected
failure. No positive host, native lease, daemon, DB connection or provider outcome
is faked. The finite gated original V05 control request reserves32MiB and imports
pytest/source helper only after the original actual worker-group attachment join.

Source controls do not prove genuine successful in-pass profile validation or
public retained-host reentry with actual3CPU4096MiB/16-slot owners and borrowed DB.
Those actual host positives remain separately unqualified. This change does not
implement durable parked disposition, candidate validation, model training,
legal fidelity, broader logic families or Lean4 build correctness. Runtime stays
disabled until the separately enumerated original-object fixtures qualify.
