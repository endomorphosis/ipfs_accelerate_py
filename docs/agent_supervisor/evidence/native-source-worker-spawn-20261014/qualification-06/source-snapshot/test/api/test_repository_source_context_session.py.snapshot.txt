"""Session joins native context to canonical source before delivering it."""
import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import RepositorySourceIndexError
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import (
    RepositorySourceSession, RepositorySourceSessionError,
)
from test.api.test_repository_source_session import RecordingIndex, source


class ContextIndex(RecordingIndex):
    action = None

    def context_query(self, query, *, limit, max_bytes):
        result = dict(schema="authored-context", source_tree_sha256=self.current["source_tree_sha256"],
            semantic_state_cid="fixture-state", authority="structural_only", semantic_equivalence_claimed=False,
            proof_authority=False, execution_authority=False, completion_authority=False)
        if self.action:
            self.action(result)
        return result

    def capsule_expand(self, *, semantic_root_cid, stable_symbol_ids, max_bytes):
        return self.context_query("authored", limit=1, max_bytes=max_bytes)


@pytest.mark.parametrize("field", ["source_tree_sha256", "semantic_state_cid", "proof_authority", "authority"])
def test_context_from_foreign_source_or_authority_is_refused(source, tmp_path, field):
    with RepositorySourceSession(source, tmp_path / "state", index_factory=ContextIndex) as session:
        session._index.action = lambda value: value.update({field: "foreign"})
        with pytest.raises(RepositorySourceSessionError, match="invalid_context_binding"):
            session.context_query(generation=1, query="add")
        with pytest.raises(RepositorySourceSessionError, match="refresh_required"):
            session.read(generation=1, path="calc.py")


@pytest.mark.parametrize("operation", ["context_query", "capsule_expand"])
def test_canonical_edit_during_native_context_work_refuses_delivery(source, tmp_path, operation):
    with RepositorySourceSession(source, tmp_path / "state", index_factory=ContextIndex) as session:
        def changed(_value):
            (source / "calc.py").write_text("def outside_change(): return 1\n")
        session._index.action = changed
        arguments = dict(query="add") if operation == "context_query" else dict(
            semantic_root_cid="authored-root", stable_symbol_ids=["authored-id"])
        with pytest.raises(RepositorySourceSessionError, match="stale_source"):
            getattr(session, operation)(generation=1, **arguments)


@pytest.mark.parametrize("preserves_index", [True, False])
def test_only_native_optional_refusal_preserves_session_currentness(source, tmp_path, preserves_index):
    with RepositorySourceSession(source, tmp_path / "state", index_factory=ContextIndex) as session:
        def failed(_value):
            raise RepositorySourceIndexError("authored_refusal", preserves_index=preserves_index)
        session._index.action = failed
        with pytest.raises(RepositorySourceIndexError, match="authored_refusal"):
            session.context_query(generation=1, query="add")
        if preserves_index:
            assert session.read(generation=1, path="calc.py")["generation"] == 1
        else:
            with pytest.raises(RepositorySourceSessionError, match="refresh_required"):
                session.read(generation=1, path="calc.py")


def test_owner_cannot_exceed_final_context_budget(source, tmp_path):
    with RepositorySourceSession(source, tmp_path / "state", index_factory=ContextIndex) as session:
        session._index.action = lambda value: value.update(oversized="x" * 1024)
        with pytest.raises(RepositorySourceSessionError, match="context_response_limit"):
            session.context_query(generation=1, query="add", max_bytes=1024)
