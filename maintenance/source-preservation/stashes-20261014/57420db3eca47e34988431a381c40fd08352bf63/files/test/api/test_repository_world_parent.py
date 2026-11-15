"""Pinned native implementation parents gain fresh source-session ownership.

Parents and tasks are authored; source indexing and child process compilation
are real. These tests do not authenticate dispatch or measure provider tokens.
"""
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import ContextCompiler, render_context_capsule
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextCapsule, ContextReference
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_parent import (
    EFFECTS, PROVENANCE_KEY, RepositoryWorldParentError, compile_implementation_world_packet,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata
from test.api.test_repository_context_compiler import compiler, native_session
from test.api.test_repository_source_index import scheduler


TASK = "Adjust add(value) while preserving the other definitions and notes."
REPOSITORY = "parent:canonical-repository"
TREE = "parent:canonical-tree"


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def worker_compiler(tokens=20000, *, optional_items=None):
    """Allow the bounded source wrapper without changing token policy/counter."""
    original = compiler(tokens, optional_items=optional_items)
    return ContextCompiler(replace(original.budget, max_items=128),
                           estimator=original.estimator, value_policy=original.value_policy)


@pytest.fixture
def parent():
    references = tuple(ContextReference(reference_id=name, kind="diagnostic-context",
        repository_id=REPOSITORY, tree_id=TREE, path="calc.py", summary=summary,
        referenced_content_id="sha256:" + sha(summary), metadata={"required": required,
            "coverage_ids": (name,), "retained_nested_tree": "canonical-body-tree"})
        for name, summary, required in (
            ("requirement", "Preserve signatures and validation; do not change notes.", True),
            ("selected-optional", "Keep this already selected diagnostic exactly.", False)))
    with public_replay_without_metadata():
        return compiler().compile(repository_id=REPOSITORY, tree_id=TREE,
            objective_id="TASK-1", objective_revision="task:canonical-1",
            policy_id="policy:implementation-daemon", policy_revision="policy:retained-1",
            caller="agent-supervisor:implementation-daemon", stage="implementation",
            goal={"task_id": "TASK-1", "attempt": 1, "title": "Retain complete core"},
            authority={"completion_authoritative": False,
                "generic_prompt_policy": ["Preserve operator controls."],
                "implementation_timeout_policy": "retained-only",
                "implementation_context_budget": {"source": "supervisor_default",
                    "max_provider_input_bytes": None, "task_max_input_tokens": None,
                    "provider_context_window": 50000, "supervisor_max_input_tokens": 20000,
                    "reserved_output_tokens": 1000, "reserved_tool_tokens": 500},
                "edit_policy": {"allowed_paths": ["calc.py"], "protected_paths": ["notes.txt"]}},
            scope={"allowed_edit_paths": ["calc.py"], "protected_edit_paths": ["notes.txt"],
                "expected_outputs": ["calc.py"], "depends_on": ["TASK-0"]},
            acceptance={"criteria": "Preserve signatures; retain all other files.",
                "validation_commands": ["python -B -m pytest test_calc.py"],
                "all_expected_outputs_required": True}, evidence=references).capsule


def arguments(adapter, parent):
    return dict(adapter=adapter, parent=parent, rendered_prompt=render_context_capsule(parent),
        expected_parent_cid=parent.capsule_id, task_text=TASK, expected_task_text_sha256=sha(TASK),
        query="add", unresolved_question="Which current definition implements the requested change?")


def test_preserves_full_parent_and_every_selected_reference(native_session, parent):
    source, session = native_session
    packet = compile_implementation_world_packet(**arguments(session.context_adapter(worker_compiler()), parent))
    capsule = packet._capsule
    for name in ("goal", "scope", "acceptance"):
        assert plain(getattr(capsule, name)) == plain(getattr(parent, name))
    task = plain(capsule.authority["task"])
    provenance = task.pop(PROVENANCE_KEY)
    binding = task.pop("world_task_packet")
    assert task == plain(parent.authority)
    assert binding["allowed_effects"] == list(EFFECTS)
    assert binding["write_paths"] == ["calc.py"]
    assert provenance["parent_capsule_id"] == parent.capsule_id
    assert provenance["parent_budget_cid"] == parent.budget.content_id
    assert provenance["parent_prompt_sha256"] == sha(render_context_capsule(parent))
    assert provenance["parent_caller"] == parent.caller and provenance["parent_stage"] == parent.stage
    assert provenance["input_token_limit"] == 20000
    for key in ("owner_authority_verified", "allocation_verified", "intent_freshness_verified",
                "parent_tokenizer_reconstructed", "final_provider_input_measured",
                "parent_evidence_currentness_verified"):
        assert provenance[key] is False
    retained = {row.reference_id: row for row in capsule.evidence}
    assert capsule.repository_id != parent.repository_id and capsule.tree_id != parent.tree_id
    for row in parent.evidence:
        assert retained[row.reference_id] == replace(row, repository_id=capsule.repository_id,
                                                    tree_id=capsule.tree_id)
    assert retained["selected-optional"].required is False
    assert packet.validate_current(source, TASK)["execution_authority"] is False
    assert "Preserve signatures; retain all other files." in packet.prompt
    assert "python -B -m pytest test_calc.py" in packet.prompt


def test_pins_suffix_and_retries_reject_before_compilation(native_session, parent):
    _, session = native_session
    adapter = session.context_adapter(worker_compiler())
    base = arguments(adapter, parent)
    changes = [dict(expected_parent_cid="sha256:" + "0" * 64),
        dict(expected_task_text_sha256="0" * 64), dict(task_text=TASK + " changed"),
        dict(rendered_prompt=base["rendered_prompt"] + "\nPrior failure instruction"),
        dict(query=""), dict(unresolved_question="")]
    for change in changes:
        with pytest.raises(ValueError):
            compile_implementation_world_packet(**dict(base, **change))
        assert adapter._parents == {}
    for changed in (replace(parent, goal=dict(plain(parent.goal), attempt=2)),
                    replace(parent, goal=dict(plain(parent.goal), attempt=True)),
                    replace(parent, goal=dict(plain(parent.goal), task_id="OTHER")),
                    replace(parent, parent_capsule_id="prior-context"),
                    replace(parent, stage="planning")):
        with pytest.raises(ValueError):
            compile_implementation_world_packet(**arguments(adapter, changed))
        assert adapter._parents == {}


def test_malformed_scope_acceptance_and_reserved_authority_reject_before_compilation(native_session, parent):
    _, session = native_session
    adapter = session.context_adapter(worker_compiler())
    cases = []
    for paths in ([], ["../calc.py"], ["/calc.py"], ["calc.py/"], ["*.py"],
                  [".git/config"], ["calc.py", "calc.py"]):
        scope = dict(plain(parent.scope), allowed_edit_paths=paths)
        authority = plain(parent.authority)
        authority["edit_policy"]["allowed_paths"] = paths
        cases.append(replace(parent, scope=scope, authority=authority))
    cases.append(replace(parent, scope=dict(plain(parent.scope), allowed_edit_paths=["other.py"])))
    for protected in (["calc.py"], ["calc.py/child"]):
        scope = dict(plain(parent.scope), protected_edit_paths=protected)
        authority = plain(parent.authority)
        authority["edit_policy"]["protected_paths"] = protected
        cases.append(replace(parent, scope=scope, authority=authority))
    for change in ({"criteria": "none listed"}, {"criteria": ["invented"]},
                   {"validation_commands": []}, {"validation_commands": "python check.py"},
                   {"validation_commands": ["python check.py", "python check.py"]},
                   {"all_expected_outputs_required": False}):
        cases.append(replace(parent, acceptance=dict(plain(parent.acceptance), **change)))
    for name in ("world_task_packet", PROVENANCE_KEY):
        cases.append(replace(parent, authority=dict(plain(parent.authority), **{name: {"claimed": True}})))
    for changed in cases:
        with pytest.raises(ValueError):
            compile_implementation_world_packet(**arguments(adapter, changed))
        assert adapter._parents == {}


def test_reference_identity_families_and_deferred_context_are_not_upgraded(native_session, parent):
    _, session = native_session
    adapter = session.context_adapter(worker_compiler())
    first, optional = parent.evidence
    for change in (dict(repository_id=""), dict(tree_id=""),
                   dict(repository_id="", tree_id=""), dict(kind="source-formalization-context"),
                   dict(kind="repository-intent-world"), dict(reference_id="repository-symbol:forged")):
        changed = replace(parent, evidence=(replace(first, **change), optional))
        with pytest.raises(ValueError):
            compile_implementation_world_packet(**arguments(adapter, changed))
        assert adapter._parents == {}
    deferred = replace(optional, tier="expansion")
    changed = replace(parent, evidence=(first,), expansion_references=(deferred,),
        truncated=True, omissions=(optional.reference_id + ":token_budget",))
    with pytest.raises(ValueError):
        compile_implementation_world_packet(**arguments(adapter, changed))
    # Even an in-memory mutation must pass contract decoding again before rebind.
    changed = ContextCapsule.from_dict(parent.to_record())
    object.__setattr__(changed.evidence[0], "repository_id", "foreign")
    with pytest.raises(ValueError):
        compile_implementation_world_packet(**arguments(adapter, changed))
    assert adapter._parents == {}


def test_optional_budget_drop_cannot_silently_shrink_dispatched_parent(native_session, parent):
    _, session = native_session
    optional = next(row for row in parent.evidence if not row.required)
    second = replace(optional, reference_id="selected-optional-second")
    with public_replay_without_metadata():
        parent = compiler().compile(**{key: getattr(parent, key) for key in (
            "repository_id", "tree_id", "objective_id", "objective_revision",
            "policy_id", "policy_revision", "caller", "stage", "goal", "authority", "scope", "acceptance")},
            evidence=(*parent.evidence, second)).capsule
    assert sum(not row.required for row in parent.evidence) == 2
    adapter = session.context_adapter(worker_compiler(optional_items=1))
    with pytest.raises(RepositoryWorldParentError, match="parent_selected_evidence_changed"):
        compile_implementation_world_packet(**arguments(adapter, parent))


def test_parent_budget_caps_and_provider_window_reserves_are_enforced(native_session, parent):
    _, session = native_session
    for change, expected in (({"task_max_input_tokens": 1}, "parent_compiled_budget_exceeded"),
                             ({"provider_context_window": 1500}, "parent_budget_exhausted"),
                             ({"max_provider_input_bytes": 1024}, "parent_compiled_budget_exceeded"),
                             ({"task_max_input_tokens": True}, "invalid_parent_budget")):
        authority = plain(parent.authority)
        authority["implementation_context_budget"].update(change)
        changed = replace(parent, authority=authority)
        with pytest.raises(RepositoryWorldParentError, match=expected):
            compile_implementation_world_packet(**arguments(session.context_adapter(worker_compiler()), changed))
    # Structural wrapper capacity may differ; it cannot widen the parent's
    # already negotiated input-token ceiling.
    changed = replace(parent, budget=replace(parent.budget, max_input_tokens=parent.input_tokens))
    with pytest.raises(RepositoryWorldParentError, match="parent_compiled_budget_exceeded"):
        compile_implementation_world_packet(**arguments(session.context_adapter(worker_compiler()), changed))


def test_source_changes_and_closed_sessions_refuse_fresh_or_reused_packets(native_session, parent):
    source, session = native_session
    adapter = session.context_adapter(worker_compiler())
    packet = compile_implementation_world_packet(**arguments(adapter, parent))
    (source / "calc.py").write_text("def add(value):\n    return value + 7\n")
    with pytest.raises(ValueError):
        packet.validate_current(source, TASK)
    with pytest.raises(ValueError):
        compile_implementation_world_packet(**arguments(adapter, parent))
    session.close()
    with pytest.raises(ValueError):
        compile_implementation_world_packet(**arguments(adapter, parent))
    with pytest.raises(ValueError):
        packet.validate_current(source, TASK)


CHILD = r'''
import json
from pathlib import Path
import sys
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextCapsule
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_parent import compile_implementation_world_packet
from test.api.test_repository_world_parent import worker_compiler
payload = json.loads(Path(sys.argv[1]).read_text())
parent = ContextCapsule.from_dict(payload["parent"])
source = Path(sys.argv[4])
with RepositorySourceSession(source, Path(sys.argv[5])) as session:
    packet = compile_implementation_world_packet(adapter=session.context_adapter(worker_compiler()),
        parent=parent, rendered_prompt=payload["rendered_prompt"], expected_parent_cid=sys.argv[2],
        task_text=payload["task_text"], expected_task_text_sha256=sys.argv[3],
        query="add", unresolved_question="Which current definition implements the requested change?")
    result = dict(session_id=session.session_id, receipt=packet.validate_current(source, payload["task_text"]),
                  authenticated_dispatch=False, parent_tokenizer_reconstructed=False)
try:
    packet.validate_current(source, payload["task_text"])
except ValueError:
    result["closed_owner_refused"] = True
else:
    raise AssertionError("closed source owner remained usable")
print(json.dumps(result, sort_keys=True))
'''


def test_new_process_issues_fresh_packet_from_independently_pinned_parent(native_session, parent, tmp_path, record_property):
    source, session = native_session
    payload = tmp_path / "parent-data.json"
    payload.write_text(json.dumps(dict(parent=plain(parent.to_record()),
        rendered_prompt=render_context_capsule(parent), task_text=TASK), sort_keys=True))
    payload.chmod(0o600)
    completed = subprocess.run([sys.executable, "-s", "-B", "-c", CHILD, str(payload),
        parent.capsule_id, sha(TASK), str(source), str(tmp_path / "child-source-state")],
        capture_output=True, text=True, timeout=60, check=True)
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["session_id"] != session.session_id
    assert result["receipt"]["source_binding"]["generation"] == 1
    assert result["closed_owner_refused"] is True
    assert result["authenticated_dispatch"] is result["parent_tokenizer_reconstructed"] is False
    assert TASK not in json.dumps(result) and str(source) not in json.dumps(result)
    record_property("world_parent_child_cycle", json.dumps(dict(real_child_process=True,
        native_fresh_source_session=True, externally_supplied_parent_pin=True,
        closed_source_owner_refused=True, provider_calls=0, authenticated_dispatch=False,
        parent_tokenizer_reconstructed=False, full_session_token_savings=None), sort_keys=True))
