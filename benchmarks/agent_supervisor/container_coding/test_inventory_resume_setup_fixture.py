"""Pure guarded-transport controls with inert databases/checkpoints.

The fabricated closed records are protocol fixtures, not a native qualification,
actual training, owner replay, Git job, or worker execution. Six real frozen
producer source files are byte-checked without importing those packages.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import inventory_resume_setup_fixture as seed


def put(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write(path, value):
    return put(path, seed._wire(value))


def cid(value):
    import base64
    return "b" + base64.b32encode(b"\x01\xa9\x02\x12\x20" + hashlib.sha256(seed._wire(value)).digest()).decode().lower().rstrip("=")


@pytest.fixture
def namespace(tmp_path):
    root = tmp_path / "closed"
    native = root / "native"
    repo = native / "repository"
    source = native / "private/source-artifacts"
    captured, entries = [], []
    commit = "1" * 40
    for i in range(300):
        path = f"unit{i:03d}.py"
        raw = f"inert source member {i}\n".encode()
        put(repo / path, raw)
        raw_cid = seed._raw_cid(raw)
        relative = "source/" + raw_cid[:4] + "/" + raw_cid
        captured.append({"path": relative, **put(source / relative, raw)})
        oid = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        entries.append({"path": path, "raw_path_hex": path.encode().hex(), "size_bytes": len(raw),
            "source_cid": raw_cid, "git_blob_oid": oid, "head_blob_oid": oid,
            "acquisition": "git-object", "disposition": "clean"})
    snapshot_cid = cid({"inert": "snapshot"})
    revision = "rev:qualification:inventory-resume:snapshot:" + snapshot_cid
    manifest = {"schema": "codebase-ir-structural-manifest@1", "ast_revision_id": revision,
        "snapshot": {"schema": "ipfs-datasets.software-contracts.semantic-repository-snapshot@4", "snapshot_cid": snapshot_cid,
            "repository_id": "qualification:inventory-resume", "mode": "git-clean", "git_commit": commit,
            "git_tree": "2" * 40, "entries": entries}}
    manifest_cid = cid(manifest)
    relative = "structured/" + manifest_cid[:4] + "/" + manifest_cid
    captured.append({"path": relative, **write(source / relative, manifest)})
    write(source / "source/scan-surplus", {"inert": "old scan page excluded"})
    head = {"schema": "codebase-head@1", "repository_id": "qualification:inventory-resume", "generation": 1,
        "manifest_cid": manifest_cid, "snapshot_cid": snapshot_cid, "receipt_cid": cid({"inert": "receipt"}),
        "ast_revision_id": revision}
    put(repo / ".git/HEAD", b"ref: refs/heads/master\n")
    put(repo / ".git/refs/heads/master", (commit + "\n").encode())
    put(repo / ".git/objects/inert", b"inert Git object bytes, no Git qualification claimed")
    put(native / "private/source.duckdb", b"inert source database: never opened")
    put(native / "private/model.duckdb", b"inert model database: never opened")
    put(native / "private/model.duckdb.owner.lock", b"")
    records, models = {}, []
    for role, epochs in (("root", 1), ("child", 2)):
        version = "sha256:" + hashlib.sha256(role.encode()).hexdigest()
        parent = None if role == "root" else records["root"]["version_id"]
        contract_sha256 = seed._digest({"inert": "contract"})
        state = {"completed_epochs": epochs, "latent_width": 8, "adam": [{"step": epochs}], "contract_sha256": contract_sha256,
                 **{key: False for key in ("admitted", "qualified", "formalized", "promotion_performed")}}
        report = {"attempted_epochs": 1, "selected_total_epochs": epochs, "contract_sha256": contract_sha256,
            "backend": "native-projection-feature-autoencoder/v1",
            "codebase_provenance": {"schema": "codebase-source-feature-lineage@1", "head": head,
                "parent_version_id": parent, "implementation": {"files": seed._PRODUCERS,
                    "sha256": seed._digest(seed._PRODUCERS), "scope": "listed_local_files_only_not_execution_attestation"}},
            **{key: False for key in ("admitted", "qualified", "formalized", "promotion_performed")}}
        saved = {"contract": {"inert": "contract"}, "feature_space": {"columns": [["inert", "feature"]]},
                 "state": state, "report": report}
        raw = seed._wire(saved)
        digest = hashlib.sha256(raw).hexdigest()
        artifact = put(native / "private/model-artifacts" / digest[:2] / digest, raw)
        models.append({"path": digest[:2] + "/" + digest, **artifact})
        record = {"authority": {key: False for key in seed._TRAIN_AUTHORITY}, "checkpoint_raw_cid": seed._raw_cid(raw),
            "contract_sha256": seed._digest(saved["contract"]), "feature_space_sha256": seed._digest(saved["feature_space"]),
            "head": head, "model_head_selected": False, "parent_version_id": parent, "registry_artifact": artifact,
            "report_json": seed._wire(report).decode(), "schema": "codebase-source-feature-training@1",
            "source_model_generation": "inert-protocol", "state_sha256": seed._digest(state),
            "training_performed_during_load": False, "variant_id": "inert-variant", "version_id": version}
        write(native / (role + ".json"), record)
        records[role] = record
    owners = {"source_head": head, "model_artifacts": models, "registry": {"heads": [], "versions": [
        [record["version_id"], record["variant_id"], record["parent_version_id"], seed._wire(record["registry_artifact"]).decode(), "{}"]
        for record in records.values()]}}
    write(native / "owners-before-scans.json", owners)
    write(native / "captured-artifacts-before-scans.json", captured)
    generation = {"schema": "codebase-inventory-resume-selected-producers@1", "execution_attestation": False, "files": []}
    for name, digest in seed._PRODUCERS.items():
        raw = seed._current_producer_path(name).read_bytes()
        relative = "producers/" + name + ".py"
        pin = put(native / relative, raw)
        put(root / "datasets" / Path(*name.split(".")).with_suffix(".py"), raw)
        generation["files"].append({"name": name, "copy": relative, "path": "/inert/generation/source.py", **pin})
        assert pin["sha256"] == digest
    write(native / "generation-inputs.json", generation)
    container = {"schema": "inventory-resume-worker-offline-container-execution@1",
        "container_id": "3" * 64, "image_id": "sha256:" + "4" * 64,
        **{key: True for key in ("container_removed", "host_reservation_released", "container_results_copied",
            "retained_source_verified_after_execution", "runtime_used_retained_source_copies", "native_module_launched")},
        "host_resources_after_cleanup": {"active_lease_count": 0, "waiting_request_count": 0,
            "allocated_child_process_slots": 0, "allocated": {"cpu_slots": 0, "memory_mb": 0}}}
    write(root / "container-execution-final.json", container)
    result = {"schema": "codebase-inventory-resume-native-qualification@1", "qualified": False,
        "unknown_fitting_epochs": False, "known_actual_setup_epochs": 2, "post_setup_fit_attempt_count": 0,
        "head": head, "selected_version_id": records["child"]["version_id"],
        **{key: False for key in ("proof_authority", "source_execution_attested", "scan_execution_attested", "production_default_activated")},
        "setup_training_attempts": [{"name": role, "requested_epochs": 1, "actual_completed_epochs": 1,
            "unknown_actual_epochs_on_failure": False, "version_id": record["version_id"]} for role, record in records.items()],
        "phases": [{"name": name, "status": "completed"} for name in ("fit_private_one_epoch_root", "fit_private_same_head_one_epoch_child")],
        "final_resources": {"active_lease_count": 0, "waiting_request_count": 0}}
    write(native / "result.json", result)
    return root


def modify(path, operation):
    value = json.loads(path.read_bytes())
    operation(value)
    write(path, value)


def test_inert_closed_setup_stages_baseline_only_and_materializes_exact_bytes(namespace, tmp_path):
    receipt = seed.stage_closed_setup(namespace, tmp_path / "seed")
    names = {row["path"] for row in receipt["copied_members"]}
    assert "private/source-artifacts/source/scan-surplus" not in names
    assert all(row["path"] in names for row in [{"path": "repository/.git/HEAD"}, {"path": "root.json"}, {"path": "child.json"}])
    assert receipt["qualified"] is False and receipt["inherited_actual_setup_epochs"] == 2 and receipt["new_fitting_epochs"] == 0
    # The wrapper can harden the seed; original output modes are restored.
    for file in (tmp_path / "seed").rglob("*"):
        if file.is_file():
            file.chmod(0o400)
    output = tmp_path / "native"
    output.mkdir()
    materialized = seed.materialize_staged_setup(tmp_path / "seed", output, receipt)
    assert materialized["seed_receipt_sha256"] == seed._digest(receipt)
    assert output.joinpath("private").stat().st_mode & 0o077 == 0
    assert output.joinpath("private/model.duckdb").stat().st_mode & 0o600 == 0o600
    for row in receipt["copied_members"]:
        assert hashlib.sha256((output / row["path"]).read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.parametrize("mutation", ["container_live", "lease_live", "unknown", "qualified", "fits", "epoch_alias", "worker", "published_phase"])
def test_staging_refuses_unclosed_unknown_or_published_setup(namespace, tmp_path, mutation):
    if mutation in {"container_live", "lease_live"}:
        key = "container_removed" if mutation == "container_live" else "host_reservation_released"
        modify(namespace / "container-execution-final.json", lambda value: value.update({key: False}))
    else:
        def change(value):
            if mutation == "unknown": value["unknown_fitting_epochs"] = True
            elif mutation == "qualified": value["qualified"] = True
            elif mutation == "fits": value["post_setup_fit_attempt_count"] = 1
            elif mutation == "epoch_alias": value["setup_training_attempts"][0]["actual_completed_epochs"] = True
            elif mutation == "worker": value["worker_launched"] = False
            else: value["phases"].append({"name": "published_source_after_worker", "status": "failed"})
        modify(namespace / "native/result.json", change)
    with pytest.raises(seed.ClosedSetupError):
        seed.stage_closed_setup(namespace, tmp_path / "seed")
    assert not (tmp_path / "seed").exists()


@pytest.mark.parametrize("mutation", ["source", "git_head", "artifact", "model", "producer", "head", "parent", "extra_private", "wal"])
def test_staging_refuses_source_model_and_owner_byte_drift(namespace, tmp_path, mutation):
    native = namespace / "native"
    if mutation == "source": put(native / "repository/unit000.py", b"changed")
    elif mutation == "git_head": put(native / "repository/.git/refs/heads/master", b"9" * 40 + b"\n")
    elif mutation == "artifact":
        row = json.loads((native / "captured-artifacts-before-scans.json").read_bytes())[0]
        put(native / "private/source-artifacts" / row["path"], b"changed")
    elif mutation == "model":
        row = json.loads((native / "owners-before-scans.json").read_bytes())["model_artifacts"][0]
        put(native / "private/model-artifacts" / row["path"], b"changed")
    elif mutation == "producer":
        name = next(iter(seed._PRODUCERS))
        put(native / "producers" / (name + ".py"), b"changed")
    elif mutation == "head": modify(native / "child.json", lambda value: value["head"].update(generation=2))
    elif mutation == "parent": modify(native / "child.json", lambda value: value.update(parent_version_id=None))
    elif mutation == "extra_private": put(native / "private/unexpected-owner.db", b"extra")
    else: put(native / "private/model.duckdb.wal", b"unclosed transaction")
    with pytest.raises(seed.ClosedSetupError):
        seed.stage_closed_setup(namespace, tmp_path / "seed")


@pytest.mark.parametrize("mutation", ["symlink", "hardlink", "ancestor", "existing", "descendant"])
def test_staging_refuses_aliases_and_nonfresh_destinations(namespace, tmp_path, mutation):
    destination = tmp_path / "seed"
    if mutation in {"symlink", "hardlink"}:
        target = namespace / "native/repository/alias.py"
        if mutation == "symlink": target.symlink_to(namespace / "native/repository/unit000.py")
        else: os.link(namespace / "native/repository/unit000.py", target)
    elif mutation == "ancestor":
        alias = tmp_path / "alias"
        alias.symlink_to(namespace, target_is_directory=True)
        namespace = alias
    elif mutation == "existing": destination.mkdir()
    else: destination = namespace / "seed"
    with pytest.raises((seed.ClosedSetupError, OSError)):
        seed.stage_closed_setup(namespace, destination)


@pytest.mark.parametrize("limit", ["bytes", "files"])
def test_body_free_aggregate_preflight_refuses_before_copy(namespace, tmp_path, monkeypatch, limit):
    monkeypatch.setattr(seed, "MAX_BYTES" if limit == "bytes" else "MAX_FILES", 128 if limit == "bytes" else 3)
    monkeypatch.setattr(seed, "_copy_members", lambda *args: pytest.fail("copy after failed preflight"))
    with pytest.raises(seed.ClosedSetupError):
        seed.stage_closed_setup(namespace, tmp_path / "seed")


def test_source_mutation_during_copy_refuses_returned_receipt(namespace, tmp_path, monkeypatch):
    real_copy = seed._copy_members
    def changing(source, destination, rows, identities):
        result = real_copy(source, destination, rows, identities)
        put(namespace / "native/repository/unit000.py", b"late source mutation")
        return result
    monkeypatch.setattr(seed, "_copy_members", changing)
    with pytest.raises(seed.ClosedSetupError, match="changed"):
        seed.stage_closed_setup(namespace, tmp_path / "seed")


@pytest.mark.parametrize("mutation", ["bytes", "extra", "receipt_alias", "missing", "newfit", "state"])
def test_materializer_refuses_changed_seed_and_rehashed_receipt(namespace, tmp_path, mutation):
    receipt = seed.stage_closed_setup(namespace, tmp_path / "seed")
    receipt = deepcopy(receipt)
    if mutation == "bytes": put(tmp_path / "seed/private/source.duckdb", b"changed")
    elif mutation == "extra": put(tmp_path / "seed/repository/extra.py", b"extra")
    elif mutation == "receipt_alias": receipt["authority"]["training_executed"] = 0
    elif mutation == "missing":
        receipt["copied_members"] = [row for row in receipt["copied_members"] if row["path"] != "repository/unit000.py"]
        receipt["copied_files"] -= 1
        receipt["copied_bytes"] = sum(row["bytes"] for row in receipt["copied_members"])
    elif mutation == "newfit": receipt["new_fitting_epochs"] = 1
    else: receipt["checkpoint_states"]["child"]["state_sha256"] = "0" * 64
    with pytest.raises(seed.ClosedSetupError):
        seed.materialize_staged_setup(tmp_path / "seed", tmp_path / "native", receipt)


def test_materializer_refuses_nonempty_output_without_overwrite(namespace, tmp_path):
    receipt = seed.stage_closed_setup(namespace, tmp_path / "seed")
    put(tmp_path / "native/existing", b"preserve")
    with pytest.raises(seed.ClosedSetupError, match="empty"):
        seed.materialize_staged_setup(tmp_path / "seed", tmp_path / "native", receipt)
    assert (tmp_path / "native/existing").read_bytes() == b"preserve"


def test_duplicate_json_and_nonfinite_controls_refuse(namespace, tmp_path):
    put(namespace / "native/result.json", b'{"qualified":false,"qualified":false}')
    with pytest.raises(seed.ClosedSetupError, match="duplicate"):
        seed.stage_closed_setup(namespace, tmp_path / "seed")
    put(namespace / "native/result.json", b'{"unknown":NaN}')
    with pytest.raises(seed.ClosedSetupError, match="finite"):
        seed.stage_closed_setup(namespace, tmp_path / "other-seed")
