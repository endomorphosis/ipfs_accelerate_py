# Bounded scalar counterexample repair

The optional scalar path turns a live Intent/code counterexample into inert
repair candidates, checks every candidate, and can hand a unique accepted edit
to the existing native worker. It requires an explicit learned Intent action
contract, Security checkpoint, source file, action, parameter mapping, finite
input ranges, and Lake executable.

The advisory API is
[`prepare_scalar_repair_advice`](../../ipfs_accelerate_py/agent_supervisor/runtime/scalar_repair_advisor.py):

```python
from ipfs_accelerate_py.agent_supervisor.runtime.scalar_repair_advisor import (
    prepare_scalar_repair_advice,
)

advice = prepare_scalar_repair_advice(
    instruction=original_instruction,
    intent_config=intent_384_config,
    security_config=security_384_config,
    source_rows=[{
        "id": "source.py",
        "source_text": original_source,
        "source_sha256": original_source_sha256,
    }],
    effect_config=explicit_scalar_effect_config,
    maximum_candidates=2,
)
```

Use the [Intent384 configuration](intent_384_action_advice.md) and
[effect configuration v2](intent_code_effects.md). The latter must select one
source and action, a bijection from `left`/`right` to its two integer parameters,
finite inclusive ranges, and a live Lake checker. Optional `intent_advice` must
numerically replay against the selected Intent configuration. There is no input
for a saved starting refutation.

The consumer freshly runs Security inference on the original source, rebuilds
its Intent association, and obtains a live kernel-checked counterexample.
The datasets-owned `security.source_scalar_repair` module verifies that live
handle and proposes the two alternatives among `+`, `-`, and `*`. Each proposal
preserves operand order and all other source bytes. The existing ProgramWorld
`replace_exact_bytes` operator must independently produce exactly those bytes
in memory. Each candidate then gets fresh Security inference, source
qualification, a rebuilt association, and its own live Lake check. Predictions
are retained unchanged, including wrong predictions and unsupported outcomes.

The report retains all candidate outcomes and distinguishes satisfaction,
refutation, and no enabled inputs. `input_pins_rechecked` records the final
instruction, configuration, checkpoint-file, producer, and source-snapshot
checks. This API writes no source files and grants no execution, admission,
mutation, or completion authority. Serialized receipts remain historical
evidence. Concurrent repository checks belong to the invoking owner boundary.

[`prepare_scalar_candidate_handoff`](../../ipfs_accelerate_py/agent_supervisor/runtime/scalar_candidate_handoff.py)
invokes that consumer freshly inside an independently admitted native task.
It binds the instruction to the signed objective or immutable signed input,
rechecks the task revision and exact source preimage, and requires the complete
two-candidate population to be checked. Exactly one candidate must satisfy the
declared effects on at least one enabled input. Incomplete or ambiguous results
remain residual; no edit is selected from a saved report.

Before preparing repository context, create `.runtime` as an owner-controlled,
worker-readable directory with mode `0755`. Existing group/world-writable or
unreadable handoff directories are rejected. Full advice stays in external
owner state; the worker receives a content-addressed inert handoff under
`.runtime/scalar-handoffs`. The
[`scalar_candidate_runner`](../../ipfs_accelerate_py/agent_supervisor/runtime/scalar_candidate_runner.py)
checks that handoff and its source preimage before applying the edit in its
allocated worktree. Native validation, publication, and completion gates remain
responsible for accepting the task result.

Run the disposable local qualification with exact local checkpoint configurations:

```bash
python -m benchmarks.agent_supervisor.container_coding.native_scalar_supervision \
  --output /tmp/fresh-scalar-qualification \
  --intent-config /models/intent384-config.json \
  --security-config /models/security384-v2-config.json \
  --lake /tools/lean/bin/lake
```

The [driver](../../benchmarks/agent_supervisor/container_coding/native_scalar_supervision.py)
creates an independently authored goal, task, failing scalar source, and public
acceptance check. It prepares context and the candidate handoff, starts the
native supervisor, and records validation, published-source identity, context
refresh, and shutdown evidence. The output directory must be new. Public
acceptance executes the fixture; symbolic source analysis does not execute it.

This remains a controlled scalar profile with explicitly bounded inputs. It
does not establish arbitrary Python equivalence or the meaning of unrestricted
instructions. The learned model does not author or admit this qualification's
goal/task graph. This path does not use the general Doctor composition, and its
results are not Terminal-Bench scores or token-savings measurements. The legacy
proof producer still requires its Git checkout; gitless proof deployment is
not qualified by this path.

The [2026-10-02 qualification](evidence/scalar-repair-supervision-20261002/README.md)
records actual checkpoint inference, a completed native task, exact published
candidate bytes, context refresh, test results, and retained failed attempts.
