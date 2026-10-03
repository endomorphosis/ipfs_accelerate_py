"""Closed profile parsing and explicit independent selection contracts."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from benchmarks.agent_supervisor.container_coding import repository_preparation_profile as profile
from benchmarks.agent_supervisor.container_coding.repository_benchmark_preparation import (
    PreparationBudget, RepositoryPreparationSelection,
)
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy import CodebaseScanPolicy


def model_off():
    contract = IntegerOffsetContract("calc.py", "increment", "n", 2)
    return profile.RepositoryPreparationProfile(
        RepositoryPreparationSelection(CodebaseScanPolicy(), index_contracts=(contract,)),
        PreparationBudget(memory_mb=1024, structural_memory_mb=1024))


def test_model_off_profile_roundtrip_declares_index_without_eager_proof(tmp_path):
    original = model_off()
    assert original.selection.proof_contracts == ()
    assert len(original.selection.index_contracts) == 1
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(original.to_dict()))
    assert profile.load_profile(path) == original
    assert profile.RepositoryPreparationProfile.from_dict(original.to_dict()) == original


@pytest.mark.parametrize("damage", ["extra", "schema", "selection_extra", "selection_missing",
    "budget_extra", "implicit_index", "bool_memory", "unbounded_time", "fixture", "model_missing",
    "model_unselected", "nonlist_paths", "nonlist_domains", "training_extra", "wrong_structural_memory"])
def test_closed_profile_refuses_implicit_or_malformed_work(damage):
    value = deepcopy(model_off().to_dict())
    if damage == "extra": value["extra"] = True
    elif damage == "schema": value["schema"] += "-wrong"
    elif damage == "selection_extra": value["selection"]["implicit_scan"] = True
    elif damage == "selection_missing": value["selection"].pop("proof_inputs")
    elif damage == "budget_extra": value["budget"]["infinite"] = True
    elif damage == "implicit_index": value["selection"]["index_contracts"] = None
    elif damage == "bool_memory": value["budget"]["memory_mb"] = True
    elif damage == "unbounded_time": value["budget"]["phase_seconds"] = float("inf")
    elif damage == "fixture": value["fixture"] = "inject-benchmark-labels"
    elif damage == "model_missing":
        value["selection"].update(model_policy="pinned_parent", inference_paths=["calc.py"])
    elif damage == "model_unselected":
        value["checkpoint"] = dict(checkpoint_path="/fixture/parent.json", checkpoint_sha256="a"*64,
            embedding_snapshot="/fixture/embedding")
    elif damage == "nonlist_paths": value["selection"]["inference_paths"] = "calc.py"
    elif damage == "nonlist_domains": value["selection"]["proof_inputs"] = "[]"
    elif damage == "training_extra": value["selection"]["training_selections"] = [
        dict(path="calc.py", role="train", group_id="one", label="correct")]
    else: value["budget"]["structural_memory_mb"] = None
    with pytest.raises(ValueError):
        profile.RepositoryPreparationProfile.from_dict(value)


def test_profile_loader_rejects_duplicate_keys_symlinks_and_oversized_input(tmp_path):
    path = tmp_path / "profile.json"
    value = json.dumps(model_off().to_dict())
    path.write_text(value[:-1] + ',"fixture":"finite-offset-base@1"}')
    with pytest.raises(ValueError, match="duplicate"):
        profile.load_profile(path)
    path.write_text(value)
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="exact profile"):
        profile.load_profile(alias)
    path.write_text(" " * 65537)
    with pytest.raises(ValueError, match="bounded"):
        profile.load_profile(path)


def test_numerical_and_protected_phase_memory_are_separately_declared():
    budget = PreparationBudget(memory_mb=4096, structural_memory_mb=1024)
    assert [budget.memory_for(name) for name in ("training", "inference", "scan", "validation", "proof")] == [4096,4096,1024,1024,1024]
    assert PreparationBudget().memory_for("validation") == 4096  # Existing caller semantics.
    for memory in (True, 512, 8192):
        with pytest.raises(ValueError): replace(budget, structural_memory_mb=memory)


def test_authored_cohort_is_separate_and_keeps_disjoint_role_populations():
    files, selections = profile.scalar_cohort()
    assert len(files) == len(selections) == 9
    assert all(path.startswith("cohort/") for path in files)
    assert "calc.py" not in files and "public_check.py" not in files
    assert {role:sum(row[1] == role for row in selections) for role in ("train","validation","holdout")} == {
        "train":3,"validation":3,"holdout":3}
    assert len(set(files.values())) == 9
    assert all(path == group for path, _, group in selections)
