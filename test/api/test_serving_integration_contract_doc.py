from pathlib import Path


def test_serving_integration_contract_doc_exists_and_has_required_sections():
    workspace_root = Path(__file__).resolve().parents[4]
    contract_doc = (
        workspace_root
        / "implementation_plan"
        / "docs"
        / "33-ipfs-accelerate-serving-integration-contract-2026-06-30.md"
    )

    assert contract_doc.exists(), "serving integration contract document must exist"

    text = contract_doc.read_text(encoding="utf-8")

    required_tokens = [
        "Canonical Runtime Entry Points",
        "Required Cross-Library Call Matrix",
        "IPFSKitStorage.store",
        "IPFSKitStorage.retrieve",
        "DatasetsManager.log_event",
        "DatasetsManager.track_provenance",
        "InferenceBackendManager.finalize_inference_result",
        "InferenceBackendManager.execute_task",
        "docker_executor.py",
        "container_backends/kubernetes/kubernetes.py",
        "p2p_tasks/orchestrator.py",
        "model_manager.py",
    ]

    for token in required_tokens:
        assert token in text, f"missing required contract token: {token}"


def test_serving_integration_contract_doc_declares_non_canonical_exclusions():
    workspace_root = Path(__file__).resolve().parents[4]
    contract_doc = (
        workspace_root
        / "implementation_plan"
        / "docs"
        / "33-ipfs-accelerate-serving-integration-contract-2026-06-30.md"
    )

    text = contract_doc.read_text(encoding="utf-8")

    required_exclusions = [
        "ipfs_accelerate_py_legacy.py",
        "backup/",
        "archive/",
        "reorganization_backup",
    ]

    for token in required_exclusions:
        assert token in text, f"missing non-canonical exclusion token: {token}"
