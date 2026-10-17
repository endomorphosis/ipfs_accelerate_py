Report-only native host V03: retained original finalizers

Source proposal only. No original checkout runtime, board, state, cap, provider or
Docker changes. This successor pins daemon-integration-v03 and the separately
frozen original_finalizer_continuations helper. H4, G3, C, borrowed-state V01 and
freezer-board V04 remain the prior reviewed sources.

The direct module wrapper retains and registers the actual nested original
finally closure and caller frame before source or borrowed-connection queries.
Registration rejects synthetic callbacks, changed sources or mismatched cells.
Physical stopped observations precede the at-most-once inner-to-outer drain.
Registration, drain and borrow-exit failures retain the original host and borrow.

Drained finalizers do not imply that a consumed attempt or candidate is settled.
A monotonic disposition_pending flag keeps the host and caller connection bound
when an ordinal, active execution, lifecycle or frozen candidate was observed.
settle_retained_finalizers observes and drains only; it never closes that borrow,
clears a disposition, parks a task, accepts a candidate or refunds an attempt.
Actual durable parking or validation handoff remains a separate unimplemented
requirement. Current cap2 is unchanged; proposed future profile still requires3.

Actual full positive host, provider-success and cleanup-failure integration remain
unqualified. Source controls and the genuine missing-host constructor case have
their own results; do not infer those results from this authored proposal.
