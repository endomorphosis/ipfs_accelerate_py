"""Actual native repair with portable, frozen security advice and source refresh."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest


def _qualified_checkpoint():
    package = os.environ.get("IPFS_SECURITY_CHECKPOINT_PACKAGE")
    digest = os.environ.get("IPFS_SECURITY_CHECKPOINT_MANIFEST_SHA256")
    if not package or not digest:
        pytest.skip("explicit independently qualified portable checkpoint required")
    pytest.importorskip("torch", reason="actual frozen checkpoint inference requires PyTorch")
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
    if not probe_quack_capabilities().passes_health_check or not shutil.which("z3"):
        pytest.skip("actual Quack and Z3 required")
    if not os.environ.get("DOCTOR_COMPOSITION_LEAN") and not shutil.which("elan"):
        pytest.skip("actual Lean required")
    from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_checkpoint import load_security_checkpoint
    return load_security_checkpoint(Path(package), expected_manifest_sha256=digest)["descriptor"]


def _assert_completed_frozen_repair(result, checkpoint):
    assert result["qualified"], result.get("error")
    assert result["initial_public_check_exit_code"] != 0
    assert result["final_public_check_exit_code"] == 0
    assert result["task"]["status"] == "completed"
    assert result["start"]["status"] == result["stop"]["status"] == "succeeded"
    assert result["remaining_processes"] == result["provider_calls"] == 0
    assert result["doctor_proof"]["status"] == "proved_local_contract"
    assert result["frozen_security_prior_observation_rejected"]
    assert result["frozen_security_checkpoint_unchanged"]
    assert result["frozen_security_training_steps"] == result["frozen_security_download_calls"] == 0
    initial, refreshed = result["frozen_security_initial"], result["frozen_security_refreshed"]
    assert initial["checkpoint"] == refreshed["checkpoint"] == checkpoint
    assert initial["source_hashes"] != refreshed["source_hashes"]
    assert initial["hydration"]["world_record_cid"] != refreshed["hydration"]["world_record_cid"]
    return initial, refreshed


def test_native_repair_preserves_frozen_checkpoint_and_rejects_stale_scores(tmp_path):
    checkpoint = _qualified_checkpoint()
    from benchmarks.agent_supervisor.container_coding.native_header_supervision import qualify

    result = qualify(tmp_path / "native-frozen-security", security_checkpoint=checkpoint)
    initial, refreshed = _assert_completed_frozen_repair(result, checkpoint)
    assert initial["hydration"]["catalog_count"] == refreshed["hydration"]["catalog_count"] == 3
    assert not refreshed["summary"]["formal_formula_heads_present"]
    # The authored task's signed source/context includes advice, which stays
    # read-only while its source-bound observations become historical.
    assert (tmp_path / "native-frozen-security/repository/frozen-security-advice.json").is_file()


def test_native_repair_refreshes_frozen_formula_models_and_indexed_plans(tmp_path):
    """Real decoder, native owner lifecycle, repair proof and DuckLake refresh.

    Opt in with the classifier package/hash above and an independently qualified
    IPFS_SECURITY_FORMULA_DECODER_DESCRIPTOR JSON file. No training, downloads,
    simulated solver results or provider calls are permitted in this test.
    """
    descriptor_path = os.environ.get("IPFS_SECURITY_FORMULA_DECODER_DESCRIPTOR")
    if not descriptor_path:
        pytest.skip("explicit independently qualified formula decoder descriptor required")
    checkpoint = _qualified_checkpoint()
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_decoder import (
        load_security_formula_decoder,
    )
    from benchmarks.agent_supervisor.container_coding.native_header_supervision import qualify

    descriptor_path = Path(descriptor_path)
    descriptor_bytes = descriptor_path.read_bytes()
    decoder = json.loads(descriptor_bytes)
    load_security_formula_decoder(decoder)
    decoder_root = Path(decoder["output"])
    decoder_files = ("manifest.json", "weights.json", "config.json", "training.json")
    decoder_hashes = {
        name: hashlib.sha256((decoder_root / name).read_bytes()).hexdigest()
        for name in decoder_files
    }

    output = tmp_path / "native-frozen-formula"
    result = qualify(output, security_checkpoint=checkpoint, formula_decoder=decoder)
    initial, refreshed = _assert_completed_frozen_repair(result, checkpoint)
    assert result["frozen_formula_decoder_unchanged"]
    assert result["frozen_formula_plan_refreshed"]
    assert result["frozen_formula_ducklake_verified"]
    assert initial["formula_decoder"] == refreshed["formula_decoder"] == decoder
    assert initial["formalization"]["report_cid"] != refreshed["formalization"]["report_cid"]
    assert descriptor_path.read_bytes() == descriptor_bytes
    assert {
        name: hashlib.sha256((decoder_root / name).read_bytes()).hexdigest()
        for name in decoder_files
    } == decoder_hashes

    plans = []
    for advice in (initial, refreshed):
        hydration = advice["hydration"]
        assert hydration["ducklake_verified"] is True
        assert hydration["catalog_count"] == hydration["ducklake_catalogs"] == 6
        assert hydration["linked_count"] == hydration["ducklake_links"] == 6
        assert advice["training_steps"] == advice["provider_calls"] == advice["download_calls"] == 0
        assert advice["summary"]["formal_formula_heads_present"] is True
        formal = advice["formalization"]
        assert formal["summary"]["learned_accepted_functions"] == 2
        assert formal["summary"]["learned_formula_count"] == 6
        assert formal["summary"]["all_functions_retained"] is True
        assert formal["summary"]["whole_program_semantics_verified"] is False
        registration = advice["formula_registration"]
        assert registration["checkpoint"] == decoder
        assert registration["operation"] == "security.advise"
        assert registration["text_generation"] is False
        probe = registration["inference_probe"]
        assert probe["executed"] is True
        assert probe["status"] == "accepted"
        assert probe["learned_formula_count"] > 0
        assert probe["candidate_validation"]["source_AST_equivalent"] is True
        assert probe["candidate_validation"]["native_lowering_complete"] is True
        assert probe["proof_authority"] is False
        plan_path = Path(advice["output"]) / "formalization-plan.json"
        plan = json.loads(plan_path.read_bytes())
        assert plan["compilation"]["status"] == "compiled"
        assert plan["input_root_cid"] == formal["report_cid"]
        assert plan["model_task_count"] == 3
        assert plan["frontier_task_count"] == 1
        assert plan["execution_started"] is False
        assert plan["proof_authority"] is plan["completion_authority"] is False
        assert plan["provider_calls"] == 0
        plans.append(plan)
    assert plans[0]["source"]["repository_tree_id"] != plans[1]["source"]["repository_tree_id"]
    assert plans[0]["compilation"] != plans[1]["compilation"]
    assert (output / "repository/frozen-security-advice.json").is_file()
