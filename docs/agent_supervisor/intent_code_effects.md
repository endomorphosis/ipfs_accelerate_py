# Explicit Intent/code effect advice

The supervisor can optionally check a declared association between a decoded
Intent action and the finite return observations of a SecurityIR code candidate.
The shared implementation belongs to `ipfs_datasets_py`; the supervisor retains
the original instructions, code, predictions, and separate checkpoint identities.

`prepare_intent_code_effect_advice` accepts the exact `instruction`, an existing
`intent_advice`, existing `security_advice`, and original `source_rows`. Its
configuration is closed:

```python
config = {
    "schema": "supervisor-intent-code-effect-config/v1",
    "contracts": [{
        "id": "declared-return-effect",
        "source_id": "example.py",
        "input_domains": {
            "left": {"lower": -1, "upper": 1},
            "right": {"lower": -1, "upper": 1},
        },
        "association": explicit_datasets_association,
    }],
    "lake": None,  # Or {"executable": "/tools/lake", "timeout_seconds": 60}.
}
```

The datasets association binds both original source hashes, both unchanged
candidate hashes, the exact finite source-state model, and a selected existing
Intent action. It explicitly maps typed expression symbols to observation-state
variables and existing precondition/effect statements to Boolean expressions.
No file, action, variable, predicate, or effect is guessed from the prompt.
Only the selected contracts are checked; unselected files acquire no claim.

The historical atom-only single-action Intent decoder does not generate effect
statements. Such a candidate remains unsupported for this check. The supervisor
does not add an effect to make it pass, even when its original Intent advice and
Security source prediction are otherwise usable. Intent advice validation
replays its existing numerical owner. Security advice supplies inference identity;
this extra consumer does not independently repeat Security embedding/inference.

The separate [384D action advice adapter](intent_384_action_advice.md) also feeds
this consumer when an explicitly selected action-contract checkpoint produces
a source-supported native document. It preserves the raw learned candidate
separately and replays the shared numerical owner before forwarding that
document. It does not change the historical atom-only model or its scope.

For source-audited scalar action predictions, configuration v2 avoids manually
authoring the typed effect formulas:

```python
config = {
    "schema": "supervisor-intent-code-effect-config/v2",
    "contracts": [{
        "id": "declared-return-effect",
        "source_id": "example.py",
        "action_id": "action",
        "input_parameter_mapping": {"left": "capacity", "right": "threshold"},
        "input_domains": {
            "capacity": {"lower": -1, "upper": 1},
            "threshold": {"lower": -1, "upper": 1},
        },
    }],
    "lake": {"executable": "/tools/lake", "timeout_seconds": 60},
}
```

The datasets-owned association builder translates the decoded precondition and
return equation, then independently rebuilds the association to check it. Source
selection, action selection, the bijective parameter mapping, and finite ranges
remain explicit caller declarations. The formulas are never selected to match
the observed code result. Missing effects, unsupported code, stale predictions,
or an invalid mapping fail open. Results retain the configuration digest and
exact selections. Neither v1 nor v2 performs a repair or changes a native task.

With Lake selected, the consumer verifies the live issued execution handle
before serializing evidence. `effect_status` distinguishes `satisfied`, `refuted`,
and `no_enabled_cases`. A checked counterexample or a checked all-disabled
domain can have `all_selected_contracts_checked=True` while
`selected_bounded_effects_satisfied=False`. Neither is a satisfied contract.
Without Lake, any interpretation result remains unchecked by the kernel.

For task context preparation, pass these optional arguments to
`prepare_supervised_task_context`:

- `intent_code_effect_instruction`: the original instruction, supplied explicitly;
- `intent_code_effect_intent_advice`: its existing decoded Intent sidecar;
- `intent_code_effect_config`: the configuration above.

Also select the existing `security_source_program_config`. The hook consumes that
freshly prepared Security advice, recaptures only its permitted source files,
and rechecks the native task revision and semantic source evidence afterward.
The task title is never substituted for the original instruction. With no
effect configuration, the optional consumer is not imported or called.

Invalid mappings, unavailable models/tools, unsupported candidates, and oversized
optional output leave planning and the original advisors available. All results
remain advisory: the bounded interpretation does not establish the instruction's
meaning, Python equivalence, a security policy, or execution/completion authority.
Saved receipts retain their evidence role and cannot substitute for live handles.
