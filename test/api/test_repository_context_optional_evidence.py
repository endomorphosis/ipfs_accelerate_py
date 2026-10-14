"""Consumer freshness boundaries and ordinary advisory roadmap evidence."""
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import EvidenceExpansionRequest
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextReference, ContextTier
from ipfs_accelerate_py.agent_supervisor.runtime.repository_context_compiler import RepositoryContextAdapter
from test.api.test_repository_context_compiler import (
    QUESTION, compile_context, compiler, references, repository_references,
    native_session, required, scheduler,  # noqa: F401
)


def roadmap(session, **changes):
    status = session.status_summary()
    values = dict(reference_id="roadmap:reviewed-next-step", kind="roadmap",
        repository_id=status["datasets_head"]["repository_id"],
        tree_id="sha256:" + status["source_tree_sha256"], path="calc.py",
        summary="  Preserve the reviewed next-step explanation and its whitespace.\n",
        metadata={"priority": 1})
    values.update(changes)
    return ContextReference(**values)


@pytest.mark.parametrize("forgery", ["changed_parent", "other_adapter"])
def test_public_consumer_fence_requires_exact_parent_issued_by_this_adapter(native_session, required, forgery):
    _source, session = native_session
    adapter = RepositoryContextAdapter(session, compiler())
    parent = compile_context(adapter, required).capsule
    assert adapter.validate_current(parent, generation=1) is None
    invalid = replace(parent, goal={"intent": "A different question with broadened scope."})
    target = adapter
    if forgery == "other_adapter":
        target = RepositoryContextAdapter(session, compiler())
        invalid = parent
    with pytest.raises(ValueError, match="parent"):
        target.validate_current(invalid, generation=1)
    assert adapter.validate_current(parent, generation=1) is None


@pytest.mark.parametrize("drift", ["wrong_generation", "refreshed_generation", "source"])
def test_public_consumer_fence_refuses_generation_or_source_drift(native_session, required, drift):
    source, session = native_session
    adapter = RepositoryContextAdapter(session, compiler())
    parent = compile_context(adapter, required).capsule
    generation = 1
    if drift == "wrong_generation":
        generation = 2
    elif drift == "refreshed_generation":
        session.refresh(generation=1)
    else:
        (source / "notes.txt").write_text("Changed outside the selected arithmetic symbols.\n")
    with pytest.raises(ValueError, match="stale|refresh"):
        adapter.validate_current(parent, generation=generation)


def test_optional_roadmap_survives_native_expansion_and_retry_without_rewriting(native_session, required):
    _source, session = native_session
    adapter = RepositoryContextAdapter(session, compiler())
    original = roadmap(session)
    parent = compile_context(adapter, required, optional_evidence=(original,)).capsule
    assert original in parent.evidence
    handle = repository_references(parent)[0]
    expanded = adapter.expand_for_question(parent,
        EvidenceExpansionRequest(QUESTION, (handle.reference_id,)), generation=1).reconstructed_capsule
    diagnostic = ContextReference(reference_id="diagnostic:consumer-retry", kind="diagnostic",
        summary="The authored proposal still needs a caller-impact check.")
    retried = adapter.retry(expanded, generation=1, evidence=(diagnostic,)).reconstructed_capsule
    for capsule in (parent, expanded, retried):
        assert references(capsule)[original.reference_id].canonical_bytes() == original.canonical_bytes()
        assert original in capsule.evidence
        for invariant in required:
            assert references(capsule)[invariant.reference_id].canonical_bytes() == invariant.canonical_bytes()
        assert adapter.validate_current(capsule, generation=1) is None


@pytest.mark.parametrize("invalid", ["required_collision", "native_collision", "required_tier", "foreign_tree"])
def test_optional_evidence_cannot_replace_required_or_native_bindings(native_session, required, invalid):
    _source, session = native_session
    adapter = RepositoryContextAdapter(session, compiler())
    changes = {
        "required_collision": {"reference_id": required[0].reference_id},
        "native_collision": {"reference_id": "repository-symbol:reserved"},
        "required_tier": {"tier": ContextTier.INVARIANT},
        "foreign_tree": {"tree_id": "sha256:" + "0" * 64},
    }[invalid]
    with pytest.raises(ValueError):
        compile_context(adapter, required, optional_evidence=(roadmap(session, **changes),))
    assert not adapter._parents
    assert session.status_summary()["source_current"] is True


def test_optional_overpopulation_is_rejected_whole_before_native_retrieval(native_session, required, monkeypatch):
    _source, session = native_session
    optional = tuple(roadmap(session, reference_id=f"roadmap:{index}") for index in range(33))
    adapter = RepositoryContextAdapter(session, compiler())
    monkeypatch.setattr(session, "context_query", lambda **_kw: pytest.fail("overbound input must precede retrieval"))
    with pytest.raises(ValueError, match="references"):
        compile_context(adapter, required, optional_evidence=optional)
    assert not adapter._parents
