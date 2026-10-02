"""Prevent qualified local criteria from masking unfinished dependencies."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding.repository_backlog_audit import audit_ledger, dependencies


@pytest.fixture
def ledger(tmp_path):
    root = Path(__file__).resolve().parents[2]
    value = json.loads((root / "docs/architecture/repository_proof_index_backlog_status.json").read_text())
    # Synthetic tiny files isolate bookkeeping controls; they are not execution
    # evidence and do not qualify any real implementation criterion.
    for name, evidence in value["evidence"].items():
        p = tmp_path / name
        p.write_text(name)
        evidence.update(repository="fixture", path=name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return value, {"fixture": tmp_path}


def test_dependency_ranges_and_abbreviated_lists_are_complete():
    assert dependencies("RPI-013 through RPI-015; RPI-029/030/031") == [
        "RPI-013", "RPI-014", "RPI-015", "RPI-029", "RPI-030", "RPI-031"]
    with pytest.raises(ValueError):
        dependencies("RPI-015 through RPI-013")


def test_current_ledger_has_no_unfinished_prerequisite_hidden_as_closed(ledger):
    value, roots = ledger
    assert audit_ledger(value, repositories=roots)["valid"]


def test_local_pass_cannot_close_lifecycle_while_resource_prerequisite_is_open(ledger):
    value, roots = ledger
    next(r for r in value["criteria"] if r["id"] == "RPI-022")["production_acceptance"] = "open"
    row = next(r for r in value["criteria"] if r["id"] == "RPI-010")
    row.update(production_acceptance="closed", blocking_dependencies=[], remaining_work=[])
    report = audit_ledger(value, repositories=roots)
    assert not report["valid"]
    assert {e["kind"] for e in report["errors"]} >= {"open_prerequisite", "incorrect_summary"}


@pytest.mark.parametrize("mutation", ["changed", "missing", "unregistered", "duplicate"])
def test_missing_changed_or_omitted_evidence_never_passes_audit(ledger, mutation):
    value, roots = ledger
    name = next(iter(value["evidence"]))
    if mutation == "changed":
        (roots["fixture"] / name).write_text("changed evidence")
    elif mutation == "missing":
        (roots["fixture"] / name).unlink()
    elif mutation == "unregistered":
        value["criteria"][0]["evidence"].append("unregistered")
    else:
        value["criteria"][-1] = deepcopy(value["criteria"][0])
        with pytest.raises(ValueError):
            audit_ledger(value, repositories=roots)
        return
    assert not audit_ledger(value, repositories=roots)["valid"]
