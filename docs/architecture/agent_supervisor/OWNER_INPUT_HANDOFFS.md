# Missing owner inputs

`DatabaseTaskSource` retains a wait for an external owner without stopping
unrelated tasks or repeatedly creating the same request. It uses the existing
DuckDB intent repository, `task_blocks`, evidence nodes, domain events, and the
typed state owner. The closed `coordinate_owner_input_v1` command requires
its own command grant; a `record_evidence`-only grant cannot create blocks or
record these reserved coordination kinds.

1. Call `request_owner_input` with the current task revision, responsible
   `owner_id`, stable `input_key`, ordered `missing_fields`, and SHA256 digest
   of the snapshot actually observed (including absence). The owner checks
   revision CAS and binds the request to the task's authority specification.
   Identical active waits return `changed=False` without another block,
   revision, or event. Continue other work via the existing readiness selector.
2. When the existing source/custody owner obtains an observation, record its
   evidence through normal `record_evidence`. Its digest must equal the input
   digest; its body must bind `owner_id`, `input_key`, and current
   `task_authority_spec_cid`. This is candidate evidence, not a substitute for
   legal review, signatures, or resource admission.
3. Call `observe_owner_input` with that evidence ID and the original request
   ID. Missing, stale, future-dated, unrelated, or differently bound evidence
   is refused. A changed digest records an addressed
   `reconsideration_requested=True` receipt. The unchanged current digest is
   a no-op, including after restart. A change back to an earlier digest is a
   new transition, retained in domain-event history.
4. The responsible owner reads `owner_input_handoffs(owner_id=...)` in its
   existing loop, draining pages with `after_event_cursor` and retaining the
   last returned `event_cursor`. Pages are ordered by each request's latest
   event so older active waits cannot hide later changes. Alternatively use
   `IntentRepository.list_events(after_global_sequence=...)`. These are
   **domain_events** cursors: the legacy `task_events` watcher observes a
   different stream. The bounded view is advisory and grants no dispatch
   authority. A stale task specification requires a fresh binding.

The owner validates changed inputs using its existing contract before
explicitly resolving a block. Notification alone never makes a task ready or
complete. Clearing one named block preserves every other active block; the
selector still checks dependencies and other gates. Replaying a resolved
request cannot reopen it.

The two coordination evidence kinds are excluded from completion eligibility,
including untyped acceptance criteria. A receipt that a handoff occurred is
not evidence that the legal task was completed.

This implements durable coordination, not legal judgment or a replacement
source reader. Legal review, source custody, funding, and model-role mappings
remain separate inputs. Their absence is not permission to infer them. This
does not start training, deployments, or hosted CI. Live coordinator loops
must adopt the API through their normal owner configuration; installing the
library does not mutate an already-running task board.
