"""Complete native semantic index joins for the opt-in finite benchmark."""
from copy import deepcopy
import threading

import duckdb
import pytest

from benchmarks.agent_supervisor.container_coding.native_repository_finite_supervision import _index, _git
from benchmarks.agent_supervisor.container_coding.terminal_codebase_semantic_index import (
    prepare_semantic_index, verify_semantic_index,
)
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract


@pytest.fixture
def prepared(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    for name, body in {
        ".gitignore": ".runtime/\n",
        "calc.py": "def increment(n: int) -> int:\n    return n + 1\n",
        "unsupported.py": "def opaque(n):\n    return eval(str(n))\n",
        "README.md": "Explicit finite fixture.\n",
    }.items():
        (repository / name).write_text(body)
    for args in (("init", "-q"), ("config", "user.name", "Qualification"),
            ("config", "user.email", "qualification@example.invalid"),
            ("add", "."), ("commit", "-qm", "Finite source fixture")):
        _git(repository, *args)
    connection = duckdb.connect(str(tmp_path / "source.duckdb"), config={"threads": 1, "memory_limit": "64MB"})
    index = _index(connection, tmp_path / "cas")
    head, descriptor = prepare_semantic_index(index=index, repository=repository,
        repository_id="repository:semantic-integration", operation_id="initial", expected_head=None,
        contract=IntegerOffsetContract("calc.py", "increment", "n", 2))
    yield index, repository, head, descriptor
    connection.close()


def test_complete_inventory_and_unproved_declaration(prepared):
    index, repository, head, descriptor = prepared
    manifest = verify_semantic_index(index=index, repository=repository,
        expected_head=head, descriptor=descriptor)
    assert {row["path"] for row in manifest["units"]} == {"calc.py", "unsupported.py", "README.md", ".gitignore"}
    selected = next(row for row in manifest["units"] if row["path"] == "calc.py")
    assert selected["declared_contract"]["offset"] == 2
    assert b"return n + 1" in index.artifacts.get_bytes(selected["source_cid"])
    assert selected["checked_evidence_references"] == []
    assert descriptor["coverage"]["source_bound_models"] == 1
    assert descriptor["coverage"]["checked_properties"] == 0
    assert descriptor["proof_authority"] is False


def test_descriptor_mutation_and_stale_source_refused(prepared):
    index, repository, head, descriptor = prepared
    for change in ("authority", "coverage", "contract", "policy", "head", "extra"):
        mutated = deepcopy(descriptor)
        if change == "authority": mutated["proof_authority"] = True
        if change == "coverage": mutated["coverage"]["checked_properties"] = 1
        if change == "contract": mutated["contract"]["offset"] = 1
        if change == "policy": mutated["policy_receipt_cid"] = descriptor["manifest_cid"]
        if change == "head": mutated["head"]["generation"] += 1
        if change == "extra": mutated["complete"] = True
        with pytest.raises(ValueError):
            verify_semantic_index(index=index, repository=repository,
                expected_head=head, descriptor=mutated)
    (repository / "calc.py").write_text("def increment(n: int) -> int:\n    return n + 2\n")
    with pytest.raises(ValueError):
        verify_semantic_index(index=index, repository=repository,
            expected_head=head, descriptor=descriptor)


def test_ambient_scope_change_refused_at_preparation(prepared):
    index, repository, head, _ = prepared
    (repository / ".git/info/exclude").write_text("hidden.py\n")
    with pytest.raises(ValueError, match="external ignore"):
        prepare_semantic_index(index=index, repository=repository,
            repository_id=head.repository_id, operation_id="different-scope", expected_head=head,
            contract=IntegerOffsetContract("calc.py", "increment", "n", 2))


def test_live_scope_replay_rejects_changed_inactive_configuration_without_writes(prepared, monkeypatch):
    index, repository, head, descriptor = prepared
    # Even an inactive comment changes the exact scope artifact. Checking it
    # cannot write that newly observed body into the immutable store.
    def forbidden(*args, **kwargs):
        raise AssertionError("live read attempted an artifact publication")
    monkeypatch.setattr(index.artifacts, "put_bytes", forbidden)
    monkeypatch.setattr(index.artifacts, "put", forbidden)
    verify_semantic_index(index=index, repository=repository,
        expected_head=head, descriptor=descriptor)
    (repository / ".git/info/exclude").write_text("# changed after preview\n")
    with pytest.raises(ValueError, match="scope differs"):
        verify_semantic_index(index=index, repository=repository,
            expected_head=head, descriptor=descriptor)


def test_live_scope_external_cancellation_after_acquisition(prepared, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import codebase_scan_policy_live as live
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
    index, repository, head, descriptor = prepared
    cancelled = threading.Event()
    original = live.load_policy_receipt
    def cancel_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled.set()
        return result
    monkeypatch.setattr(live, "load_policy_receipt", cancel_after_read)
    with pytest.raises(LeaseCancelledError):
        live.verify_policy_current(index, repository, expected_head=head,
            receipt_cid=descriptor["policy_receipt_cid"], cancel_event=cancelled)
