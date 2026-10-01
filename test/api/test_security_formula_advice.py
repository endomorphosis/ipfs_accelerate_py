"""Real frozen decoder -> native plan -> DuckLake -> stale-source refresh."""
import hashlib
from pathlib import Path

import pytest

from tests.unit.logic.formalization.autoencoder.test_security_formula_decoder import formula_checkpoint
from test.api.test_codebase_autoencoder_transfer import teacher, fork, joint_inputs
from benchmarks.agent_supervisor.container_coding.test_terminal_frozen_security import checkpoint
from ipfs_accelerate_py.agent_supervisor.runtime import security_autoencoder_advisor as advisor
from ipfs_accelerate_py.agent_supervisor.runtime.security_formula_model import (
    register_security_formula_decoder, decode_registered_security_formula,
)


def test_real_formula_advice_is_indexed_planned_and_refreshed(tmp_path, checkpoint, formula_checkpoint):
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "math.py"
    source.write_text("def compute(value):\n    return (value + 8) * (value - 3)\n")
    ledger = {"math.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    before = advisor.prepare_security_advice(repository=repository, paths=["math.py"], source_hashes=ledger,
        checkpoint=checkpoint, formula_decoder=formula_checkpoint, output=tmp_path / "before")
    summary = advisor.validate_security_advice(repository=repository, expected_receipt=before)
    assert summary["formal_formula_heads_present"]
    assert summary["formalization"]["learned_formula_count"] == 1
    assert before["hydration"]["ducklake_verified"] and before["hydration"]["catalog_count"] == 6
    assert Path(before["output"], "formalization-plan.json").is_file()
    assert before["formula_registration"]["inference_probe"]["executed"]
    assert before["training_steps"] == before["provider_calls"] == before["download_calls"] == 0
    assert advisor.refresh_security_advice(repository=repository, previous=before, output=tmp_path / "unused") == before
    source.write_text("def compute(value):\n    return (value + 8) * (value - 4)\n")
    with pytest.raises(ValueError):
        advisor.validate_security_advice(repository=repository, expected_receipt=before)
    after = advisor.refresh_security_advice(repository=repository, previous=before, output=tmp_path / "after")
    advisor.validate_security_advice(repository=repository, expected_receipt=after)
    assert after["formula_decoder"] == before["formula_decoder"] == formula_checkpoint
    assert after["formalization"]["report_cid"] != before["formalization"]["report_cid"]
    assert after["hydration"]["world_record_cid"] != before["hydration"]["world_record_cid"]
    assert not after["proof_authority"] and not after["completion_authority"]


def test_registered_decoder_executes_canonical_inference(tmp_path, formula_checkpoint):
    from ipfs_accelerate_py.model_manager import ModelManager
    manager = ModelManager(storage_path=str(tmp_path / "models.json"), use_database=False,
                           enable_ipfs=False, project_legacy_models=False)
    registered = register_security_formula_decoder(manager=manager, checkpoint=formula_checkpoint)
    result = decode_registered_security_formula(manager=manager, model_id=registered["model_id"],
        source_bytes=b"def operation(value):\n    return (value - 7) * (value + 2)\n", source_path="code.py")
    assert result["status"] == "accepted" and result["learned_formula_count"] == 1
    assert result["checkpoint"] == formula_checkpoint
    assert not result["proof_authority"]
