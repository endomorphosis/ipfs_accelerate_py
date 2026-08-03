# ADR-0002: Models propose; evidence admits

- **Status:** Accepted
- **Date:** 2026-08-03
- **Last verified:** 2026-08-03
- **Deciders:** agent-supervisor control-plane maintainers; documentation-refresh program
- **Scope:** Trust boundary between model (or other untrusted provider) output and
  anything that may advance admission, merge eligibility, or authoritative task
  or goal completion in `ipfs_accelerate_py.agent_supervisor`
- **Non-goals:** Which LLM vendor or endpoint is configured; proof-kernel choice;
  worktree/lease mechanics (see planned ADR-0004); capability/catalog/routing
  separation (ADR-0003); objective-vs-todo durability (ADR-0001)
- **Supersedes:** none
- **Superseded-by:** none
- **Related guides:**
  - [`docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md`](../AGENT_SUPERVISOR_PHILOSOPHY.md)
  - [`docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md`](../agent_supervisor/PLANNING_AND_ASSURANCE.md)
  - [`docs/architecture/agent_supervisor_code_claim_evidence_contract.md`](../agent_supervisor_code_claim_evidence_contract.md)
  - [`docs/architecture/agent_supervisor/packages/proof.md`](../agent_supervisor/packages/proof.md)
  - [`docs/architecture/agent_supervisor/packages/merge.md`](../agent_supervisor/packages/merge.md)
  - [`docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md`](../AGENT_SUPERVISOR_ARCHITECTURE.md)
- **Source anchors:**
  - `ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py`
  - `ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py`
  - `ipfs_accelerate_py/agent_supervisor/objectives/goal_completion.py`
  - `ipfs_accelerate_py/agent_supervisor/planning/task_proposal_router.py`
  - `ipfs_accelerate_py/agent_supervisor/proof/code_claim_contracts.py`
  - `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_cache.py`
  - `ipfs_accelerate_py/agent_supervisor/merge/merge_train.py`
  - Focused tests under `test/api/` for proposal validation, claim/evidence
    contracts, and formal planning

## Status meanings (do not invent new values)

| Value | Use when |
| --- | --- |
| Proposed | Decision is under review; **not** yet evidenced current design |
| Accepted | Decision matches current code/tests/ops practice and is normative for Scope |
| Deprecated | Still historical; prefer another practice for new work |
| Superseded | Replaced by the ADR in Superseded-by |
| Rejected | Considered and not adopted; retained to document the negative choice |

Only **Accepted** records are current design authority. **Proposed** records
must not be treated as implemented system law.

## Context

The agent supervisor is a control plane for objective-driven software work.
Large language models and optional analysis providers produce fluent plans,
patches, diagnoses, and natural-language claims of success. Git and merge
infrastructure can land those patches on an integrate branch. Neither fluency
nor a green merge is, by itself, a trustworthy statement that acceptance
criteria hold on a bound tree.

Several forces make the trust split mandatory:

1. **Hallucination and overclaim.** Models routinely assert “done,” invent
   validation results, or narrate proof that never ran. Treating that prose as
   authority would complete tasks without evidence.
2. **Multi-writer and multi-lane pressure.** Implementation daemons, merge
   trains, and rescue loops must share a single rule for when board status may
   flip to accepted. A landed commit proves code entered history; it does not
   re-evaluate post-merge freshness, semantic gates, or proof obligations.
3. **Typed evidence ladders already exist.** `EvidenceTier` and
   `AssuranceLevel` refuse silent promotion (query fact is not kernel proof;
   cache hits re-derive assurance). Model output sits *below* those ladders as
   proposal material, not as a tier that can self-promote.
4. **Operator and audit expectations.** Fail-closed admission, allowlisted
   roots, identity bindings, and recomputed gates are how later runs resume,
   reopen stale acceptance, or deny completion without re-trusting chat.

If this decision is deferred, the system reverts to chat-transcript authority:
fluent output or merge ancestry becomes an accidental completion oracle, and
authoritative gates become decorative.

## Decision

The control plane treats **all model and optional-provider generative output as
proposals**. Deterministic policy evaluation over typed, bound evidence
authorizes advancement. Generative output starts at proposal tier; an
independently admitted provider review may satisfy only its own configured
gate. **Neither fluent model output nor a landed commit independently
authorizes task or goal completion.**

Normative rules:

1. **Proposal tier.** Plans, patches, repair suggestions, failure reviews,
   migration results, and provider status text are **proposals** or
   **nominations**. They may inform context and candidate branches after schema
   and policy checks. They never set `completion_authoritative` or
   `proof_authoritative` by their own wording.
2. **Admission is deterministic and fail-closed.** Proposal validation
   (`validation/proposal_validation.py` and related gates) checks allowlisted
   paths, identity bindings, scope, schema, and configured validation commands.
   Accepted proposal receipts still project `completion_authoritative: false`
   and `proof_authoritative: false`. Admission to implement is not admission to
   complete.
3. **Evidence is typed and tiered.** Claims bind to closed evidence tiers and
   re-derived assurance levels. Weaker classes (query/GraphRAG fact,
   observation, solver candidate) must not be renamed or cached into stronger
   ones (kernel proof, attestation). Proof-cache hits **re-derive** assurance;
   they do not invent it. A typed, independently reviewed `provider_review`
   result can satisfy that gate when policy requires it; it does not become
   deterministic proof or satisfy unrelated gates.
4. **Merge is implementation landing, not acceptance.** A successful merge
   (implementation commit, merge commit, queue/train state, Git ancestry) is
   evidence that code landed. The authoritative completion module records
   `implemented_merged_but_pending` until every bound gate in
   `AUTHORITATIVE_COMPLETION_GATE_KINDS` (`merge`, `freshness`, `semantic`,
   `proof`, `provider_review`, `deterministic_only`) recomputes cleanly from
   commit- and tree-bound evidence. Callers cannot self-assert acceptance via
   a `completion_authoritative` flag on `build_implementation_receipt`.
5. **Promotion is separate and recomputed.** `evaluate_authoritative_completion_gate`
   and `promote_authoritative_completion` recompute gates from bound evidence;
   cached gate lists are not trusted. Goal completion remains two-phase
   (`provisionally_complete` vs `verified_complete`) with fresh, tree-bound
   evidence in `objectives/goal_completion.py`.
6. **Ownership of authority.**
   - Proposal selection / plan candidates: `planning/`, `validation/`
   - Typed proof and claim lifecycle: `proof/`
   - Landing patches: `merge/`
   - Authoritative task completion: `todo_daemon/authoritative_completion.py`
   - Goal completion: `objectives/goal_completion.py`

Typical gated lifecycle (policy selects the required evidence classes; not
every class is mandatory for every task):

```text
1. Intent                 objective / task identity
2. Proposal               model plan or patch
3. Pre-merge admission    validation, scope policy, lease/worktree isolation
4. Landing                merge commit and merge receipt
5. Post-merge admission   fresh bound evidence and recomputed configured gates
6. Completion mutation    separately authorized task/goal state promotion
```

## Alternatives

### Alternative A: Direct model trust (fluent output as authority)

- **Summary:** Treat a model’s assertion that work is complete—or a
  high-confidence natural-language summary of validation—as sufficient to mark
  tasks done and advance objectives.
- **Expected benefits:** Lowest latency and least plumbing; no post-hoc gates;
  agents “feel” end-to-end autonomous.
- **Why not chosen:** Fluent output is not bound to a repository tree, does not
  recompute deterministic commands, and can invent proof or test results. Direct
  trust collapses the authority ladder at rung 2, makes multi-lane completion
  non-auditable, and cannot fail closed when the model is wrong. The control
  plane exists precisely because chat transcripts are unsafe completion oracles.
  **Fluent model output cannot independently authorize completion.**

### Alternative B: Merge-as-completion (landed commit as acceptance)

- **Summary:** When an implementation branch merges cleanly to the integrate
  target (or an implementation commit is reachable from main), automatically
  mark the taskboard item complete and treat acceptance criteria as satisfied.
- **Expected benefits:** Simple operational story (“green merge = done”);
  reuses Git as the single ledger; fewer completion schemas.
- **Why not chosen:** Merge proves *landing*, not *acceptance*. Post-merge
  validation can be stale; required semantic or proof gates may still be open;
  protected-path or identity bindings may not match the merged tree; ancestry
  alone does not re-run acceptance criteria. The implementation receipt
  deliberately separates `merged` / merge commits from
  `completion_authoritative` and keeps acceptance in
  `implemented_merged_but_pending` until promotion. **A landed commit cannot
  independently authorize completion.**

### Alternative C: Self-asserted completion flags on receipts

- **Summary:** Allow producers (model adapters, implementation daemons, or
  merge train) to set `completion_authoritative=true` on receipts when they
  believe gates passed, without independent recomputation.
- **Expected benefits:** Faster path; fewer re-evaluations under load.
- **Why not chosen:** Self-assertion is a trust smuggling channel. Code
  explicitly discards caller-supplied completion authority when building
  receipts (`build_implementation_receipt` deletes the flag) and recomputes
  every gate from bound evidence. Cached or self-asserted lists are rejected so
  a buggy or compromised producer cannot mint acceptance.

### Alternative D: Do nothing / informal trust in chat + CI green

- **Summary:** Rely on human reading of agent transcripts and ad-hoc CI without
  typed proposal vs completion separation in the supervisor.
- **Expected benefits:** No control-plane complexity.
- **Why not chosen:** Does not support multi-lane isolation, durable reopen of
  stale acceptance, fail-closed optional providers, or machine-auditable
  promotion. The repository already implements the ladder; documenting informal
  trust would contradict live gates.

## Consequences

### Positive

- Agents and operators share one rule: **propose freely, admit with evidence**.
- Hallucinated “done” claims cannot flip board or goal state without bound
  receipts.
- Merge trains can land work while acceptance stays pending, reopened, or denied
  without rewriting Git history.
- Evidence tiers and assurance levels remain comparable across programs
  (codebase-proof, self-improvement, documentation refresh).
- Auditors can re-run `evaluate_authoritative_completion_gate` from stored
  receipts without trusting narrative logs.

### Negative

- Completing a task requires more machinery than a fluent summary or a green
  merge: validation commands, gate evidence, freshness, and sometimes proof.
- Latency and operational complexity increase (post-merge revalidation, reopen
  on stale validation, pending acceptance states).
- Agents and developers may initially confuse “proposal accepted” or “merged”
  with “authoritatively completed.”
- Some optional providers look “successful” in UI text but remain non-authoritative,
  which can surprise users expecting self-certification.
- Extending gates requires coordinated schema and receipt updates rather than
  a one-line status flip.

### Neutral / residual risks

- A correct merge with incomplete gate evidence correctly stays
  `implemented_merged_but_pending`; operators must not treat that state as
  failure of Git.
- Deterministic-only policies reject observed model invocations even when
  patches look good—intentional friction.
- External completion authorities (`core/external_completion.py`) are a distinct
  class; they still must not be confused with freeform model prose.
- Residual risk: misconfigured validation commands that always pass. Mitigated
  by task-declared commands, scope policy, and separate proof tiers when
  required—not by trusting the model to invent those checks.

## Evidence

| Claim in Decision | Evidence (path, test, or operational check) | Notes |
| --- | --- | --- |
| Model/provider output is proposal-tier | `validation/proposal_validation.py` (`completion_authoritative` / `proof_authoritative` always false on proposal receipts); `planning/task_proposal_router.py`; philosophy pillar 2 | Admission ≠ completion |
| Callers cannot self-assert completion | `todo_daemon/authoritative_completion.py` `build_implementation_receipt` (`del completion_authoritative`) | Flag discarded |
| Gates recompute from bound evidence | `evaluate_authoritative_completion_gate`, `promote_authoritative_completion` | Cached lists not trusted |
| Merge ≠ authoritative acceptance | Module docstring and `ACCEPTANCE_STATE_MERGED_PENDING`; gate kinds include `merge` as one of several required gates | Landing is necessary evidence, not sufficient alone |
| Goal completion is two-phase | `objectives/goal_completion.py` | Provisional vs verified |
| Evidence tiers refuse silent promotion | `proof/code_claim_contracts.py` `EvidenceTier`; claim/evidence contract doc | Closed ladder |
| Cache re-derives, does not invent | `proof/formal_verification_cache.py`; PLANNING_AND_ASSURANCE § cache miss | Hit ≠ trust root |
| Merge train can revalidate post-merge | `merge/merge_train.py` optional `post_merge_validation` callback | When configured, admits post-merge evidence; otherwise the train only lands work and acceptance remains separately pending or denied |

## Verification

How a future reader confirms the decision still holds:

1. Inspect proposal receipts for hard-coded non-authority flags:

   ```text
   rg -n 'completion_authoritative' ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py
   ```

   Pass: properties and serialized projections keep `completion_authoritative`
   false for proposal-tier results.

2. Confirm implementation receipts cannot self-promote and that merge is not
   sole authority:

   ```text
   rg -n 'del completion_authoritative|ACCEPTANCE_STATE_MERGED_PENDING|AUTHORITATIVE_COMPLETION_GATE_KINDS' \
     ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py
   ```

   Pass: `build_implementation_receipt` discards self-asserted authority; pending
   acceptance state and multi-kind gates remain.

3. Run focused contract tests (from repo root when the suite is available):

   ```text
   python -m pytest test/api/test_agent_supervisor_code_claim_evidence_contract.py -q
   ```

4. Staleness signals (ADR must be revisited if any hold):

   - Proposal validation begins returning `completion_authoritative: true`.
   - Merge ancestry alone flips taskboard acceptance without
     `evaluate_authoritative_completion_gate`.
   - Evidence tiers allow upgrade by rename or cache hit without re-derivation.

## Review triggers

- [ ] Source anchors no longer match the Decision statement
- [ ] A recorded negative consequence becomes unacceptable
- [ ] A rejected alternative (direct model trust or merge-as-completion) becomes
      viable without those costs
- [ ] Security, isolation, lease/fence, or trust-tier changes touch this scope
- [ ] Related guide or package ownership is restructured
- [ ] Superseding design is Accepted under a new ADR number
- [ ] Authoritative gate kind set or completion schemas change
- [ ] External completion authority is merged into freeform model channels

When superseding: create a new ADR number; set this file to **Superseded** with
`Superseded-by`; set the successor’s `Supersedes`; do not delete this file.

## Notes (optional)

- Program reservation: ADR-0002 / DOC-016 in
  [`docs/architecture/decisions/README.md`](README.md) (“Models propose;
  evidence admits; merge ≠ acceptance”).
- Sibling decisions: durable objectives vs task projections (ADR-0001);
  worktrees/leases (ADR-0004). This ADR only fixes the proposal → evidence →
  completion trust boundary.
- Related philosophy ladder and pillars are narrative companions; this ADR is
  the normative decision record for that boundary.
