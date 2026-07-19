from __future__ import annotations

import importlib.util
import hashlib
import json
import signal
import subprocess
import sys
import types
from pathlib import Path

import ipfs_accelerate_py.llm_router as llm_router
from ipfs_accelerate_py.utils import llama_cpp as llama_cpp_utils


class _FakeHTTPResponse:
    status = 200

    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _load_llama_cpp_kit_class():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "ipfs_accelerate_py"
        / "worker"
        / "skillset"
        / "llama_cpp_kit.py"
    )
    spec = importlib.util.spec_from_file_location("_test_llama_cpp_kit", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.llama_cpp_kit


def test_llama_cpp_server_command_uses_modern_llama_serve():
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4:NVFP4",
        hf_file="Leanstral-1.5-119B-A6B-NVFP4.gguf",
        host="127.0.0.1",
        port=8080,
        context_size=2048,
    )

    assert llama_cpp_utils.build_llama_cpp_server_command(
        config,
        executable="/usr/bin/llama",
        command_kind="llama",
    ) == (
        "/usr/bin/llama",
        "serve",
        "-hf",
        "Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4:NVFP4",
        "--hf-file",
        "Leanstral-1.5-119B-A6B-NVFP4.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "-c",
        "2048",
    )


def test_llama_cpp_server_command_supports_legacy_llama_server_binary():
    config = llama_cpp_utils.LlamaCppServerConfig(model_ref="repo:Q4", port=9001)

    assert llama_cpp_utils.build_llama_cpp_server_command(
        config,
        executable="/opt/llama-server",
        command_kind="llama-server",
    )[:3] == ("/opt/llama-server", "-hf", "repo:Q4")


def test_llama_cpp_server_command_prefers_local_model_path():
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="repo:Q4",
        hf_file="model.gguf",
        model_path="/models/model.gguf",
        port=9001,
    )

    command = llama_cpp_utils.build_llama_cpp_server_command(
        config,
        executable="/opt/llama-server",
        command_kind="llama-server",
    )

    assert command[:3] == ("/opt/llama-server", "-m", "/models/model.gguf")
    assert "-hf" not in command
    assert "--hf-file" not in command


def test_llama_cpp_config_defaults_exact_leanstral_hf_file(monkeypatch):
    monkeypatch.delenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_REF", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_LLAMA_CPP_HF_FILE", raising=False)

    config = llama_cpp_utils.config_from_env()

    assert config.model_ref == llama_cpp_utils.DEFAULT_LEANSTRAL_MODEL_REF
    assert config.hf_file == llama_cpp_utils.DEFAULT_LEANSTRAL_FILENAME


def test_llama_cpp_config_does_not_reuse_leanstral_file_for_other_models(monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_REF", "other/repo:Q4_K_M")
    monkeypatch.delenv("IPFS_ACCELERATE_LLAMA_CPP_HF_FILE", raising=False)

    config = llama_cpp_utils.config_from_env()

    assert config.model_ref == "other/repo:Q4_K_M"
    assert config.hf_file == ""


def test_llama_cpp_config_preserves_zero_gpu_layers(monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_GPU_LAYERS", "0")

    config = llama_cpp_utils.config_from_env()
    command = llama_cpp_utils.build_llama_cpp_server_command(
        config,
        executable="/usr/bin/llama",
        command_kind="llama",
    )

    assert config.gpu_layers == 0
    assert "-ngl" in command
    assert command[command.index("-ngl") + 1] == "0"


def test_llama_cpp_config_treats_auto_gpu_layers_as_safe_sizing(monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_GPU_LAYERS", "safe")

    config = llama_cpp_utils.config_from_env()

    assert config.gpu_layers is None
    assert config.auto_sizing is True


def test_llama_cpp_config_reads_model_path_and_cid(monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_PATH", "/models/model.gguf")
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_CID", "bafytest")

    config = llama_cpp_utils.config_from_env()

    assert config.model_path == "/models/model.gguf"
    assert config.model_cid == "bafytest"


def test_llama_cpp_device_parser_extracts_cuda_memory():
    devices = llama_cpp_utils._parse_llama_cpp_device_list(
        "CUDA0: NVIDIA GB10 (124610 MiB, 120632 MiB free)\n"
    )

    assert len(devices) == 1
    assert devices[0].identifier == "CUDA0"
    assert devices[0].name == "NVIDIA GB10"
    assert devices[0].total_bytes == 124610 * 1024 * 1024
    assert devices[0].free_bytes == 120632 * 1024 * 1024


def test_llama_cpp_raw_sha256_cid_v1_matches_ipfs_raw_identity():
    sha256_hex = hashlib.sha256(b"").hexdigest()
    multihash_hex, cid = llama_cpp_utils._raw_sha256_cid_v1_from_digest(sha256_hex)

    assert multihash_hex == "1220" + sha256_hex
    assert cid == "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"


def test_llama_cpp_auto_sizing_bounds_large_nvfp4_offload(monkeypatch):
    gib = 1024**3
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4:NVFP4",
        hf_file="Leanstral-1.5-119B-A6B-NVFP4.gguf",
        context_size=128,
        auto_sizing=True,
    )
    cache = llama_cpp_utils.LlamaCppModelCacheStatus(
        repo_id="Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4",
        filename="Leanstral-1.5-119B-A6B-NVFP4.gguf",
        cache_root="/tmp/hf",
        repo_cache_dir="/tmp/hf/models--Frosty40--Leanstral",
        complete=True,
        local_path="/tmp/Leanstral.gguf",
        local_size_bytes=67 * gib,
    )
    sizing = llama_cpp_utils.LlamaCppGgufSizingInfo(
        path="/tmp/Leanstral.gguf",
        file_size_bytes=67 * gib,
        alignment=32,
        tensor_count=256,
        metadata={"llama.block_count": 64},
        layer_count=64,
        repeating_layer_bytes=tuple([gib] * 64),
        non_repeating_bytes=3 * gib,
        max_layer_bytes=gib,
        avg_layer_bytes=gib,
    )
    monkeypatch.setattr(llama_cpp_utils, "read_llama_cpp_gguf_sizing_info", lambda _path: sizing)
    monkeypatch.setattr(
        llama_cpp_utils,
        "llama_cpp_list_devices",
        lambda **_kwargs: (
            llama_cpp_utils.LlamaCppDeviceInfo(
                identifier="CUDA0",
                name="NVIDIA GB10",
                free_bytes=120 * gib,
                total_bytes=124 * gib,
            ),
        ),
    )

    sized_config, plan = llama_cpp_utils.auto_size_llama_cpp_server_config(
        config,
        cache_status=cache,
    )
    command = llama_cpp_utils.build_llama_cpp_server_command(
        sized_config,
        executable="/usr/bin/llama",
        command_kind="llama",
    )

    assert plan.reason == "calculated"
    assert plan.gpu_layers == 2
    assert plan.batch_size == 64
    assert plan.ubatch_size == 64
    assert sized_config.gpu_layers == plan.gpu_layers
    assert command[command.index("-ngl") + 1] == str(plan.gpu_layers)
    assert command[command.index("--device") + 1] == "CUDA0"
    assert command[command.index("--fit") + 1] == "on"
    assert command[command.index("--fit-target") + 1] == "8192"
    assert "--no-warmup" in command
    assert command[command.index("-b") + 1] == "64"
    assert command[command.index("-ub") + 1] == "64"


def test_llama_cpp_auto_sizing_allows_cuda13_large_nvfp4_offload(monkeypatch):
    gib = 1024**3
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4:NVFP4",
        hf_file="Leanstral-1.5-119B-A6B-NVFP4.gguf",
        context_size=128,
        auto_sizing=True,
    )
    cache = llama_cpp_utils.LlamaCppModelCacheStatus(
        repo_id="Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4",
        filename="Leanstral-1.5-119B-A6B-NVFP4.gguf",
        cache_root="/tmp/hf",
        repo_cache_dir="/tmp/hf/models--Frosty40--Leanstral",
        complete=True,
        local_path="/tmp/Leanstral.gguf",
        local_size_bytes=67 * gib,
    )
    sizing = llama_cpp_utils.LlamaCppGgufSizingInfo(
        path="/tmp/Leanstral.gguf",
        file_size_bytes=67 * gib,
        alignment=32,
        tensor_count=256,
        metadata={"llama.block_count": 64},
        layer_count=64,
        repeating_layer_bytes=tuple([gib] * 64),
        non_repeating_bytes=3 * gib,
        max_layer_bytes=gib,
        avg_layer_bytes=gib,
    )
    monkeypatch.setattr(llama_cpp_utils, "read_llama_cpp_gguf_sizing_info", lambda _path: sizing)
    monkeypatch.setattr(llama_cpp_utils, "_llama_cpp_cublas_major", lambda _executable: 13)
    monkeypatch.setattr(
        llama_cpp_utils,
        "llama_cpp_list_devices",
        lambda **_kwargs: (
            llama_cpp_utils.LlamaCppDeviceInfo(
                identifier="CUDA0",
                name="NVIDIA GB10",
                free_bytes=120 * gib,
                total_bytes=124 * gib,
            ),
        ),
    )

    sized_config, plan = llama_cpp_utils.auto_size_llama_cpp_server_config(
        config,
        executable="/opt/llama-server",
        command_kind="llama-server",
        cache_status=cache,
    )

    assert plan.reason == "calculated"
    assert plan.gpu_layers == 64
    assert plan.layer_multiplier == 1.25
    assert sized_config.gpu_layers == 64


def test_llama_cpp_auto_sizing_preserves_explicit_batch_args(monkeypatch):
    gib = 1024**3
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
        context_size=256,
        extra_args=("-b", "32", "-ub", "16"),
        auto_sizing=True,
    )
    cache = llama_cpp_utils.LlamaCppModelCacheStatus(
        repo_id="owner/model",
        filename="model.gguf",
        cache_root="/tmp/hf",
        repo_cache_dir="/tmp/hf/models--owner--model",
        complete=True,
        local_path="/tmp/model.gguf",
        local_size_bytes=8 * gib,
    )
    sizing = llama_cpp_utils.LlamaCppGgufSizingInfo(
        path="/tmp/model.gguf",
        file_size_bytes=8 * gib,
        alignment=32,
        tensor_count=64,
        metadata={"llama.block_count": 4},
        layer_count=4,
        repeating_layer_bytes=tuple([gib] * 4),
        non_repeating_bytes=gib,
        max_layer_bytes=gib,
        avg_layer_bytes=gib,
    )
    monkeypatch.setattr(llama_cpp_utils, "read_llama_cpp_gguf_sizing_info", lambda _path: sizing)
    monkeypatch.setattr(
        llama_cpp_utils,
        "llama_cpp_list_devices",
        lambda **_kwargs: (
            llama_cpp_utils.LlamaCppDeviceInfo(
                identifier="CUDA0",
                name="Test GPU",
                free_bytes=32 * gib,
                total_bytes=40 * gib,
            ),
        ),
    )

    sized_config, plan = llama_cpp_utils.auto_size_llama_cpp_server_config(
        config,
        cache_status=cache,
    )

    assert plan.enabled is True
    assert sized_config.extra_args[:4] == ("-b", "32", "-ub", "16")
    assert sized_config.extra_args[sized_config.extra_args.index("--device") + 1] == "CUDA0"
    assert sized_config.gpu_layers == plan.gpu_layers


def test_llama_cpp_model_cache_status_detects_complete_snapshot(tmp_path):
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
    )
    model_path = tmp_path / "models--owner--model" / "snapshots" / "abc123" / "model.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"gguf")

    status = llama_cpp_utils.llama_cpp_model_cache_status(config, cache_root=tmp_path)

    assert status.complete is True
    assert status.downloading is False
    assert status.local_size_bytes == 4
    assert status.message == "complete"


def test_llama_cpp_model_cache_status_complete_ignores_stale_partial(tmp_path):
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
    )
    model_path = tmp_path / "models--owner--model" / "snapshots" / "abc123" / "model.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"gguf")
    partial = tmp_path / "models--owner--model" / "blobs" / "abc.downloadInProgress"
    partial.parent.mkdir(parents=True)
    partial.write_text("stale", encoding="utf-8")

    status = llama_cpp_utils.llama_cpp_model_cache_status(config, cache_root=tmp_path)

    assert status.complete is True
    assert status.downloading is False
    assert status.partial_paths == (str(partial),)
    assert status.message == "complete"


def test_llama_cpp_model_cache_status_detects_partial_download(tmp_path):
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
    )
    partial = tmp_path / "models--owner--model" / "blobs" / "abc.downloadInProgress"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    incomplete = tmp_path / "models--owner--model" / "blobs" / "def.incomplete"
    incomplete.write_bytes(b"more")

    status = llama_cpp_utils.llama_cpp_model_cache_status(config, cache_root=tmp_path)

    assert status.complete is False
    assert status.downloading is True
    assert status.partial_size_bytes == 11
    assert status.partial_paths == (str(partial), str(incomplete))
    assert status.message == "download_in_progress"


def test_llama_cpp_model_cache_status_detects_explicit_model_path(tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
        model_path=str(model_path),
    )

    status = llama_cpp_utils.llama_cpp_model_cache_status(config)

    assert status.complete is True
    assert status.cache_backend == "model_path"
    assert status.local_path == str(model_path)


def test_prefetch_llama_cpp_model_registers_local_path_in_content_cache(monkeypatch, tmp_path):
    source = tmp_path / "source.gguf"
    source.write_bytes(b"tiny-gguf")
    cache_dir = tmp_path / "model-cache"
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE_LINK_MODES", "hardlink,copy")
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
        model_path=str(source),
    )

    status = llama_cpp_utils.prefetch_llama_cpp_model(config, model_cache=True)

    assert status.complete is True
    assert status.cache_backend == "content_addressed_disk"
    expected_sha256 = hashlib.sha256(b"tiny-gguf").hexdigest()
    expected_multihash, expected_cid = llama_cpp_utils._raw_sha256_cid_v1_from_digest(expected_sha256)
    assert status.content_sha256 == expected_sha256
    assert status.content_multihash_sha256 == expected_multihash
    assert status.content_cid_v1 == expected_cid
    assert status.content_cid_v1_path
    assert Path(status.content_cid_v1_path).exists()
    assert Path(f"{status.content_cid_v1_path}.json").exists()
    assert Path(status.local_path).exists()
    assert Path(status.local_path).read_bytes() == b"tiny-gguf"
    manifest = cache_dir / "refs" / "owner_model" / "model.gguf.json"
    assert manifest.exists()


def test_prefetch_llama_cpp_model_schedules_async_hash(monkeypatch, tmp_path):
    source = tmp_path / "source.gguf"
    source.write_bytes(b"tiny-gguf")
    cache_dir = tmp_path / "model-cache"
    captured = {}

    class FakeProcess:
        pid = 4242

    def fake_popen(command, **kwargs):
        captured["command"] = list(command)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProcess()

    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_ASYNC_MODEL_HASH_MIN_BYTES", "1")
    monkeypatch.setattr(llama_cpp_utils.subprocess, "Popen", fake_popen)
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
        model_path=str(source),
    )

    status = llama_cpp_utils.prefetch_llama_cpp_model(
        config,
        model_cache=True,
        async_hash=True,
    )

    assert status.complete is True
    assert status.cache_backend == "model_path_hash_pending"
    assert status.content_hash_pending is True
    assert status.content_sha256 == ""
    assert status.content_cid_v1 == ""
    assert status.local_path == str(source.resolve())
    assert status.content_hash_job_pid == 4242
    assert status.content_hash_job_path
    assert "--finalize-model-artifact" in captured["command"]
    assert "--finalize-source-path" in captured["command"]
    assert captured["env"]["IPFS_ACCELERATE_LLAMA_CPP_ASYNC_MODEL_HASH"] == "0"
    assert llama_cpp_utils._serve_model_cache_via_local_path(status) is True
    manifest = json.loads((cache_dir / "refs" / "owner_model" / "model.gguf.json").read_text())
    assert manifest["content_hash_pending"] is True
    assert manifest["content_hash_job_pid"] == 4242


def test_llama_cpp_async_hash_finalizer_updates_manager_without_dropping_command(
    tmp_path,
    monkeypatch,
):
    from ipfs_accelerate_py.model_manager import ModelManager

    cache_dir = tmp_path / "model-cache"
    manager_path = tmp_path / "models.json"
    source = tmp_path / "source.gguf"
    source.write_bytes(b"tiny-gguf")
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_MANAGER_PATH", str(manager_path))
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
        model_path=str(source),
        port=8128,
        context_size=512,
        gpu_layers=8,
    )
    pending_status = llama_cpp_utils.LlamaCppModelCacheStatus(
        repo_id="owner/model",
        filename="model.gguf",
        cache_root=str(cache_dir),
        repo_cache_dir=str(source.parent),
        complete=True,
        local_path=str(source),
        local_size_bytes=source.stat().st_size,
        message="complete_hash_pending",
        cache_backend="model_path_hash_pending",
        content_hash_pending=True,
    )
    command = llama_cpp_utils.build_llama_cpp_server_command(
        config,
        executable="/usr/bin/llama-server",
        command_kind="llama-server",
    )
    assert llama_cpp_utils._register_llama_cpp_model_with_manager(
        config,
        pending_status,
        command=command,
        command_kind="llama-server",
        enabled=True,
    )

    final_status = llama_cpp_utils._finalize_llama_cpp_model_artifact_hash(
        config,
        source,
        source_backend="model_path",
        track_model=True,
    )

    expected_sha256 = hashlib.sha256(b"tiny-gguf").hexdigest()
    _, expected_cid = llama_cpp_utils._raw_sha256_cid_v1_from_digest(expected_sha256)
    assert final_status.content_sha256 == expected_sha256
    assert final_status.content_cid_v1 == expected_cid
    manager = ModelManager(storage_path=str(manager_path), use_database=False, enable_ipfs=False)
    metadata = manager.get_model("owner/model:model.gguf")
    assert metadata is not None
    assert metadata.model_cid == expected_cid
    assert metadata.repository_structure["files"]["model.gguf"]["sha256"] == expected_sha256
    assert metadata.serving_config["resolved_command"] == list(command)
    assert metadata.serving_config["cache"]["content_hash_pending"] is False
    assert manager.resolve_launch_command("owner/model:model.gguf") == list(command)
    manager.close()


def test_prefetch_llama_cpp_model_can_recover_local_cid_alias(monkeypatch, tmp_path):
    source = tmp_path / "source.gguf"
    source.write_bytes(b"tiny-gguf")
    cache_dir = tmp_path / "model-cache"
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE_DIR", str(cache_dir))
    first_config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
        model_path=str(source),
    )
    first_status = llama_cpp_utils.prefetch_llama_cpp_model(first_config, model_cache=True)

    class UnusedStorage:
        def is_available(self):
            return True

        def retrieve(self, _cid):
            raise AssertionError("local CID alias should avoid remote retrieval")

    monkeypatch.setattr(llama_cpp_utils, "_ipfs_kit_model_storage", lambda: UnusedStorage())
    cid_config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="other/model:Q4",
        hf_file="model.gguf",
        model_cid=first_status.content_cid_v1,
    )

    status = llama_cpp_utils.prefetch_llama_cpp_model(
        cid_config,
        model_cache=True,
        local_files_only=True,
    )

    assert status.complete is True
    assert status.local_path == first_status.content_cid_v1_path
    assert status.content_sha256 == first_status.content_sha256
    assert status.content_cid_v1 == first_status.content_cid_v1
    assert status.ipfs_remote_attempted is False


def test_prefetch_llama_cpp_model_materializes_ipfs_cid(monkeypatch, tmp_path):
    cache_dir = tmp_path / "model-cache"
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE_DIR", str(cache_dir))

    class FakeStorage:
        cache_dir = tmp_path / "ipfs-cache"

        def is_available(self):
            return True

        def retrieve(self, cid):
            assert cid == "bafyremote"
            return b"remote-gguf"

    monkeypatch.setattr(llama_cpp_utils, "_ipfs_kit_model_storage", lambda: FakeStorage())
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="remote.gguf",
        model_cid="bafyremote",
    )

    status = llama_cpp_utils.prefetch_llama_cpp_model(
        config,
        model_cache=True,
        local_files_only=False,
    )

    assert status.complete is True
    assert status.cache_backend == "ipfs_kit"
    assert status.content_cid == "bafyremote"
    assert status.ipfs_remote_attempted is True
    assert status.ipfs_remote_loaded is True
    assert Path(status.local_path).read_bytes() == b"remote-gguf"


def test_llama_cpp_model_manager_registration_records_cache_and_command(tmp_path, monkeypatch):
    from ipfs_accelerate_py.model_manager import ModelManager

    manager_path = tmp_path / "models.json"
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"tiny-gguf")
    sha256_hex = hashlib.sha256(b"tiny-gguf").hexdigest()
    multihash_hex, cid_v1 = llama_cpp_utils._raw_sha256_cid_v1_from_digest(sha256_hex)
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_MANAGER_PATH", str(manager_path))

    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
        model_path=str(model_path),
        port=8127,
        context_size=256,
        gpu_layers=4,
        extra_args=("-b", "32", "-ub", "16"),
    )
    status = llama_cpp_utils.LlamaCppModelCacheStatus(
        repo_id="owner/model",
        filename="model.gguf",
        cache_root=str(tmp_path),
        repo_cache_dir=str(tmp_path),
        complete=True,
        local_path=str(model_path),
        local_size_bytes=model_path.stat().st_size,
        cache_backend="content_addressed_disk",
        content_sha256=sha256_hex,
        content_multihash_sha256=multihash_hex,
        content_cid_v1=cid_v1,
        content_addressed_path=str(model_path),
    )
    command = llama_cpp_utils.build_llama_cpp_server_command(
        config,
        executable="/usr/bin/llama-server",
        command_kind="llama-server",
    )

    assert llama_cpp_utils._register_llama_cpp_model_with_manager(
        config,
        status,
        auto_sizing=llama_cpp_utils.LlamaCppAutoSizingPlan(
            True,
            reason="calculated",
            gpu_layers=4,
            batch_size=32,
            ubatch_size=16,
            context_size=256,
        ),
        command=command,
        command_kind="llama-server",
        enabled=True,
    )

    manager = ModelManager(storage_path=str(manager_path), use_database=False, enable_ipfs=False)
    metadata = manager.get_model("owner/model:model.gguf")
    assert metadata is not None
    assert metadata.model_cid == cid_v1
    assert metadata.repository_structure["files"]["model.gguf"]["sha256"] == sha256_hex
    assert metadata.repository_structure["files"]["model.gguf"]["cid_v1_raw_sha256"] == cid_v1
    assert "cid_v1_path" in metadata.repository_structure["files"]["model.gguf"]
    assert metadata.serving_config["resolved_command"] == list(command)
    assert manager.resolve_launch_command("owner/model:model.gguf") == list(command)
    manager.close()


def test_ensure_llama_cpp_uses_existing_binary_without_installer(monkeypatch, tmp_path):
    llama_bin = tmp_path / "llama"
    llama_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    llama_bin.chmod(0o755)
    install_calls = []

    monkeypatch.setattr(llama_cpp_utils, "_default_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(llama_cpp_utils.shutil, "which", lambda name: str(llama_bin) if name == "llama" else None)
    monkeypatch.setattr(
        llama_cpp_utils,
        "_run_command",
        lambda *args, **kwargs: install_calls.append((args, kwargs)) or (0, "", ""),
    )
    monkeypatch.setattr(llama_cpp_utils, "_cuda_toolkit_available", lambda: False)

    result = llama_cpp_utils.ensure_llama_cpp(auto_install=True)

    assert result.available is True
    assert result.executable == str(llama_bin)
    assert result.command_kind == "llama"
    assert install_calls == []


def test_ensure_llama_cpp_auto_installs_managed_source_when_missing(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(llama_cpp_utils, "_default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(llama_cpp_utils, "find_llama_cpp_executable", lambda _explicit_binary="": ("", ""))
    monkeypatch.setattr(llama_cpp_utils, "_cuda_toolkit_available", lambda: False)
    monkeypatch.setattr(llama_cpp_utils, "_nproc", lambda: "2")
    monkeypatch.setattr(
        llama_cpp_utils.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"git", "cmake"} else None,
    )

    def fake_run(command, **_kwargs):
        calls.append(tuple(command))
        if tuple(command[:2]) == ("git", "clone"):
            source_dir = Path(command[-1])
            (source_dir / ".git").mkdir(parents=True)
        if tuple(command[:2]) == ("cmake", "--build"):
            server = tmp_path / "build" / "bin" / "llama-server"
            server.parent.mkdir(parents=True)
            server.write_text("#!/bin/sh\n", encoding="utf-8")
            server.chmod(0o755)
        return 0, "ok", ""

    monkeypatch.setattr(llama_cpp_utils, "_run_command", fake_run)

    result = llama_cpp_utils.ensure_llama_cpp(auto_install=True)

    assert result.available is True
    assert result.method == "source_build"
    assert result.command_kind == "llama-server"
    assert result.executable == str(tmp_path / "build" / "bin" / "llama-server")
    assert any(call[:2] == ("git", "clone") for call in calls)
    assert any(call[:2] == ("cmake", "--build") for call in calls)


def test_ensure_llama_cpp_prefers_managed_cuda_for_legacy_binary(monkeypatch, tmp_path):
    legacy = tmp_path / "llama"
    legacy.write_text("#!/bin/sh\n", encoding="utf-8")
    legacy.chmod(0o755)
    managed = tmp_path / "build" / "bin" / "llama-server"
    install_calls = []

    monkeypatch.setattr(
        llama_cpp_utils,
        "find_llama_cpp_executable",
        lambda _explicit_binary="": (str(legacy), "llama"),
    )
    monkeypatch.setattr(llama_cpp_utils, "_cuda_toolkit_available", lambda: True)
    monkeypatch.setattr(llama_cpp_utils, "_llama_cpp_cublas_major", lambda _executable: 12)

    def fake_install(**kwargs):
        install_calls.append(kwargs)
        return llama_cpp_utils.LlamaCppInstallResult(
            available=True,
            executable=str(managed),
            command_kind="llama-server",
            installed=True,
            method="source_build",
        )

    monkeypatch.setattr(llama_cpp_utils, "_install_llama_cpp_from_source", fake_install)

    result = llama_cpp_utils.ensure_llama_cpp(auto_install=True)

    assert result.available is True
    assert result.executable == str(managed)
    assert install_calls


def test_evict_llama_cpp_warm_servers_uses_lru_pidfile(monkeypatch, tmp_path):
    old_pid = 111
    new_pid = 222
    dead = set()
    cache_dir = tmp_path / "llama-cache"
    cache_dir.mkdir()
    (cache_dir / "server-8001.json").write_text(
        json.dumps(
            {
                "pid": old_pid,
                "base_url": "http://127.0.0.1:8001/v1",
                "started_at": 1.0,
                "last_accessed_at": 10.0,
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "server-8002.json").write_text(
        json.dumps(
            {
                "pid": new_pid,
                "base_url": "http://127.0.0.1:8002/v1",
                "started_at": 2.0,
                "last_accessed_at": 20.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(llama_cpp_utils, "_default_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(llama_cpp_utils, "_device_free_bytes", lambda _identifier="": 0)
    monkeypatch.setattr(llama_cpp_utils, "_process_alive", lambda pid: int(pid) not in dead)

    def fake_kill(pid, sig):
        if sig in {signal.SIGTERM, signal.SIGKILL}:
            dead.add(int(pid))

    monkeypatch.setattr(llama_cpp_utils.os, "kill", fake_kill)

    result = llama_cpp_utils.evict_llama_cpp_warm_servers(max_to_evict=1)

    assert result["evicted"][0]["pid"] == old_pid
    assert not (cache_dir / "server-8001.json").exists()
    assert (cache_dir / "server-8002.json").exists()


def test_ensure_llama_cpp_server_reuses_active_pidfile(monkeypatch, tmp_path):
    config = llama_cpp_utils.LlamaCppServerConfig(port=8123)
    pidfile = tmp_path / "server-8123.json"
    pidfile.write_text(
        json.dumps(
            {
                "base_url": config.base_url,
                "command": ["/usr/bin/llama", "serve"],
                "log_path": str(tmp_path / "server.log"),
                "pid": 12345,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(llama_cpp_utils, "_pidfile_path", lambda _config: pidfile)
    monkeypatch.setattr(llama_cpp_utils, "_process_alive", lambda pid: int(pid) == 12345)
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_server_ready", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        llama_cpp_utils,
        "ensure_llama_cpp",
        lambda **_kwargs: llama_cpp_utils.LlamaCppInstallResult(
            available=True,
            executable="/usr/bin/llama",
            command_kind="llama",
        ),
    )

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("active pidfile should prevent duplicate server launch")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)

    result = llama_cpp_utils.ensure_llama_cpp_server(
        config,
        autostart=True,
        startup_timeout_seconds=0,
    )

    assert result.running is False
    assert result.started is False
    assert result.pid == 12345
    assert result.message == "server_existing_startup_timeout"


def test_ensure_llama_cpp_server_restarts_active_pidfile_on_context_mismatch(
    monkeypatch, tmp_path
):
    config = llama_cpp_utils.LlamaCppServerConfig(
        port=8123,
        context_size=16384,
        extra_args=("--parallel", "1"),
    )
    pidfile = tmp_path / "server-8123.json"
    pidfile.write_text(
        json.dumps(
            {
                "base_url": config.base_url,
                "command": [
                    "/usr/bin/llama",
                    "serve",
                    "-hf",
                    config.model_ref,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8123",
                    "-c",
                    "2048",
                ],
                "log_path": str(tmp_path / "server.log"),
                "pid": 12345,
            }
        ),
        encoding="utf-8",
    )
    launched = {}
    ready_checks = {"count": 0}

    monkeypatch.setattr(llama_cpp_utils, "_pidfile_path", lambda _config: pidfile)
    monkeypatch.setattr(llama_cpp_utils, "_log_path", lambda _config: tmp_path / "new-server.log")
    monkeypatch.setattr(llama_cpp_utils, "_process_alive", lambda pid: int(pid) == 12345)

    def fake_terminate(_config, _active):
        pidfile.unlink()
        return True

    monkeypatch.setattr(llama_cpp_utils, "_terminate_active_pidfile_process", fake_terminate)
    monkeypatch.setattr(
        llama_cpp_utils,
        "ensure_llama_cpp",
        lambda **_kwargs: llama_cpp_utils.LlamaCppInstallResult(
            available=True,
            executable="/usr/bin/llama",
            command_kind="llama",
        ),
    )
    monkeypatch.setattr(
        llama_cpp_utils,
        "llama_cpp_model_cache_status",
        lambda _config: llama_cpp_utils.LlamaCppModelCacheStatus(
            repo_id="owner/model",
            filename="model.gguf",
            cache_root="/tmp/hf",
            repo_cache_dir="/tmp/hf/models--owner--model",
            complete=True,
        ),
    )

    class FakePopen:
        pid = 67890

        def __init__(self, command, **_kwargs):
            launched["command"] = list(command)

        def poll(self):
            return None

    def fake_ready(*_args, **_kwargs):
        ready_checks["count"] += 1
        return ready_checks["count"] > 1

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_server_ready", fake_ready)

    result = llama_cpp_utils.ensure_llama_cpp_server(
        config,
        autostart=True,
        startup_timeout_seconds=2,
    )

    assert result.running is True
    assert result.started is True
    assert "-c" in launched["command"]
    assert launched["command"][launched["command"].index("-c") + 1] == "16384"
    assert "--parallel" in launched["command"]
    assert launched["command"][launched["command"].index("--parallel") + 1] == "1"


def test_ensure_llama_cpp_server_autosizes_after_managed_install(monkeypatch, tmp_path):
    gib = 1024**3
    config = llama_cpp_utils.LlamaCppServerConfig(
        port=8125,
        context_size=128,
        auto_sizing=True,
    )
    managed = tmp_path / "build" / "bin" / "llama-server"
    managed.parent.mkdir(parents=True)
    managed.write_text("#!/bin/sh\n", encoding="utf-8")
    managed.chmod(0o755)
    cache = llama_cpp_utils.LlamaCppModelCacheStatus(
        repo_id="Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4",
        filename="Leanstral-1.5-119B-A6B-NVFP4.gguf",
        cache_root="/tmp/hf",
        repo_cache_dir="/tmp/hf/models--Frosty40--Leanstral",
        complete=True,
        local_path="/tmp/Leanstral.gguf",
        local_size_bytes=67 * gib,
    )
    sizing = llama_cpp_utils.LlamaCppGgufSizingInfo(
        path="/tmp/Leanstral.gguf",
        file_size_bytes=67 * gib,
        alignment=32,
        tensor_count=256,
        metadata={"llama.block_count": 36},
        layer_count=36,
        repeating_layer_bytes=tuple([gib] * 36),
        non_repeating_bytes=3 * gib,
        max_layer_bytes=gib,
        avg_layer_bytes=gib,
    )
    launched = {}
    ready_checks = {"count": 0}

    monkeypatch.setattr(llama_cpp_utils, "_pidfile_path", lambda _config: tmp_path / "server-8125.json")
    monkeypatch.setattr(llama_cpp_utils, "_log_path", lambda _config: tmp_path / "server.log")
    monkeypatch.setattr(llama_cpp_utils, "find_llama_cpp_executable", lambda _explicit_binary="": ("", ""))
    monkeypatch.setattr(
        llama_cpp_utils,
        "ensure_llama_cpp",
        lambda **_kwargs: llama_cpp_utils.LlamaCppInstallResult(
            available=True,
            executable=str(managed),
            command_kind="llama-server",
            installed=True,
            method="source_build",
        ),
    )
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_model_cache_status", lambda _config: cache)
    monkeypatch.setattr(llama_cpp_utils, "read_llama_cpp_gguf_sizing_info", lambda _path: sizing)
    monkeypatch.setattr(llama_cpp_utils, "_llama_cpp_cublas_major", lambda executable: 13 if executable else 0)
    monkeypatch.setattr(
        llama_cpp_utils,
        "llama_cpp_list_devices",
        lambda **_kwargs: (
            llama_cpp_utils.LlamaCppDeviceInfo(
                identifier="CUDA0",
                name="NVIDIA GB10",
                free_bytes=120 * gib,
                total_bytes=124 * gib,
            ),
        ),
    )
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_EVICT_TO_FIT", "0")

    class FakePopen:
        pid = 67891

        def __init__(self, command, **_kwargs):
            launched["command"] = list(command)

        def poll(self):
            return None

    def fake_ready(*_args, **_kwargs):
        ready_checks["count"] += 1
        return ready_checks["count"] > 1

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_server_ready", fake_ready)

    result = llama_cpp_utils.ensure_llama_cpp_server(
        config,
        autostart=True,
        auto_install=True,
        startup_timeout_seconds=2,
    )

    assert result.running is True
    assert result.auto_sizing.gpu_layers == 36
    assert launched["command"][launched["command"].index("-ngl") + 1] == "36"


def test_ensure_llama_cpp_server_prefetch_failure_does_not_start(monkeypatch):
    config = llama_cpp_utils.LlamaCppServerConfig(port=8124)
    status = llama_cpp_utils.LlamaCppModelCacheStatus(
        repo_id="owner/model",
        filename="model.gguf",
        cache_root="/tmp/hf",
        repo_cache_dir="/tmp/hf/models--owner--model",
        message="download_in_progress",
        downloading=True,
        partial_size_bytes=10,
    )
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_server_ready", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        llama_cpp_utils,
        "ensure_llama_cpp",
        lambda **_kwargs: llama_cpp_utils.LlamaCppInstallResult(
            available=True,
            executable="/usr/bin/llama",
            command_kind="llama",
        ),
    )
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_model_cache_status", lambda _config: status)
    monkeypatch.setattr(llama_cpp_utils, "prefetch_llama_cpp_model", lambda _config, **_kwargs: status)

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("prefetch failure should prevent server launch")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)

    result = llama_cpp_utils.ensure_llama_cpp_server(
        config,
        autostart=True,
        prefetch_model=True,
    )

    assert result.running is False
    assert result.started is False
    assert result.model_cache.message == "download_in_progress"
    assert result.message == "model_prefetch_download_in_progress"


def test_ensure_llama_cpp_server_prefetch_uses_content_cache_model_path(monkeypatch, tmp_path):
    model_path = tmp_path / "cache" / "model.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"gguf")
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
        port=8126,
    )
    status = llama_cpp_utils.LlamaCppModelCacheStatus(
        repo_id="owner/model",
        filename="model.gguf",
        cache_root=str(tmp_path / "cache"),
        repo_cache_dir=str(model_path.parent),
        complete=True,
        local_path=str(model_path),
        local_size_bytes=4,
        message="complete",
        cache_backend="content_addressed_disk",
    )
    launched = {}
    ready_checks = {"count": 0}

    monkeypatch.setattr(llama_cpp_utils, "_pidfile_path", lambda _config: tmp_path / "server-8126.json")
    monkeypatch.setattr(llama_cpp_utils, "_log_path", lambda _config: tmp_path / "server.log")
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_server_ready", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        llama_cpp_utils,
        "ensure_llama_cpp",
        lambda **_kwargs: llama_cpp_utils.LlamaCppInstallResult(
            available=True,
            executable="/usr/bin/llama",
            command_kind="llama",
        ),
    )
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_model_cache_status", lambda _config: status)
    monkeypatch.setattr(llama_cpp_utils, "prefetch_llama_cpp_model", lambda _config, **_kwargs: status)
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_EVICT_TO_FIT", "0")

    class FakePopen:
        pid = 67892

        def __init__(self, command, **_kwargs):
            launched["command"] = list(command)

        def poll(self):
            return None

    def fake_ready(*_args, **_kwargs):
        ready_checks["count"] += 1
        return ready_checks["count"] > 1

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_server_ready", fake_ready)

    result = llama_cpp_utils.ensure_llama_cpp_server(
        config,
        autostart=True,
        prefetch_model=True,
        startup_timeout_seconds=2,
    )

    assert result.running is True
    assert launched["command"][:3] == ["/usr/bin/llama", "serve", "-m"]
    assert launched["command"][3] == str(model_path)


def test_ensure_llama_cpp_server_prefetch_uses_hash_pending_model_path(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "hf" / "model.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"gguf")
    config = llama_cpp_utils.LlamaCppServerConfig(
        model_ref="owner/model:Q4",
        hf_file="model.gguf",
        port=8129,
    )
    status = llama_cpp_utils.LlamaCppModelCacheStatus(
        repo_id="owner/model",
        filename="model.gguf",
        cache_root=str(tmp_path / "hf"),
        repo_cache_dir=str(model_path.parent),
        complete=True,
        local_path=str(model_path),
        local_size_bytes=4,
        message="complete_hash_pending",
        cache_backend="hf_hash_pending",
        content_hash_pending=True,
    )
    launched = {}
    ready_checks = {"count": 0}

    monkeypatch.setattr(llama_cpp_utils, "_pidfile_path", lambda _config: tmp_path / "server-8129.json")
    monkeypatch.setattr(llama_cpp_utils, "_log_path", lambda _config: tmp_path / "server.log")
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_server_ready", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        llama_cpp_utils,
        "ensure_llama_cpp",
        lambda **_kwargs: llama_cpp_utils.LlamaCppInstallResult(
            available=True,
            executable="/usr/bin/llama",
            command_kind="llama",
        ),
    )
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_model_cache_status", lambda _config: status)
    monkeypatch.setattr(llama_cpp_utils, "prefetch_llama_cpp_model", lambda _config, **_kwargs: status)
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_EVICT_TO_FIT", "0")

    class FakePopen:
        pid = 67893

        def __init__(self, command, **_kwargs):
            launched["command"] = list(command)

        def poll(self):
            return None

    def fake_ready(*_args, **_kwargs):
        ready_checks["count"] += 1
        return ready_checks["count"] > 1

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_server_ready", fake_ready)

    result = llama_cpp_utils.ensure_llama_cpp_server(
        config,
        autostart=True,
        prefetch_model=True,
        startup_timeout_seconds=2,
    )

    assert result.running is True
    assert launched["command"][:3] == ["/usr/bin/llama", "serve", "-m"]
    assert launched["command"][3] == str(model_path)
    assert result.model_cache.content_hash_pending is True


def test_llama_cpp_router_provider_calls_openai_compatible_endpoint(monkeypatch):
    captured = {}
    llm_router.clear_llm_router_caches()
    monkeypatch.setattr(llama_cpp_utils, "llama_cpp_server_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL", "leanstral-local")

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return _FakeHTTPResponse({"choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr(llm_router.urllib.request, "urlopen", fake_urlopen)
    provider = llm_router._builtin_provider_by_name("llama_cpp")

    assert provider is not None
    assert provider.generate("prove this", max_tokens=4, temperature=0.0) == "OK"
    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert captured["payload"]["model"] == "leanstral-local"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "prove this"}]
    assert "Authorization" not in captured["headers"]


def test_llama_cpp_native_router_provider_uses_python_binding(monkeypatch):
    captured = {}

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, repo_id, filename, **kwargs):
            captured["repo_id"] = repo_id
            captured["filename"] = filename
            captured["load_kwargs"] = dict(kwargs)
            return cls()

        def create_chat_completion(self, **kwargs):
            captured["chat_kwargs"] = dict(kwargs)
            return {"choices": [{"message": {"content": "native-ok"}}]}

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_REF", "owner/repo:NVFP4")
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_HF_FILE", "model.gguf")
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_CONTEXT_SIZE", "123")
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_THREADS", "4")
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_GPU_LAYERS", "99")
    llm_router.clear_llm_router_caches()

    provider = llm_router._builtin_provider_by_name("llama_cpp_native")

    assert provider is not None
    assert provider.generate("prove this", max_tokens=4, temperature=0.0) == "native-ok"
    assert captured["repo_id"] == "owner/repo"
    assert captured["filename"] == "model.gguf"
    assert captured["load_kwargs"]["n_ctx"] == 123
    assert captured["load_kwargs"]["n_threads"] == 4
    assert captured["load_kwargs"]["n_gpu_layers"] == 99
    assert captured["chat_kwargs"]["messages"] == [{"role": "user", "content": "prove this"}]
    assert captured["chat_kwargs"]["max_tokens"] == 4


def test_llama_cpp_native_router_provider_reuses_content_cache(monkeypatch, tmp_path):
    captured = {}
    model_path = tmp_path / "cas-model.gguf"
    model_path.write_bytes(b"gguf")

    class FakeLlama:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured["load_kwargs"] = dict(kwargs)

        def create_chat_completion(self, **kwargs):
            captured["chat_kwargs"] = dict(kwargs)
            return {"choices": [{"message": {"content": "cached-native-ok"}}]}

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE", "1")
    monkeypatch.delenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_PATH", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_PATH", raising=False)
    monkeypatch.setattr(
        llama_cpp_utils,
        "llama_cpp_model_cache_status",
        lambda _config: llama_cpp_utils.LlamaCppModelCacheStatus(
            repo_id="owner/model",
            filename="model.gguf",
            cache_root=str(tmp_path),
            repo_cache_dir=str(tmp_path),
            complete=True,
            local_path=str(model_path),
            local_size_bytes=4,
            cache_backend="content_addressed_disk",
        ),
    )
    llm_router.clear_llm_router_caches()

    provider = llm_router._builtin_provider_by_name("llama_cpp_native")

    assert provider is not None
    assert provider.generate("prove this", max_tokens=4, temperature=0.0) == "cached-native-ok"
    assert captured["model_path"] == str(model_path)


def test_llama_cpp_native_router_provider_reuses_hash_pending_cache(monkeypatch, tmp_path):
    captured = {}
    model_path = tmp_path / "pending-model.gguf"
    model_path.write_bytes(b"gguf")

    class FakeLlama:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured["load_kwargs"] = dict(kwargs)

        def create_chat_completion(self, **kwargs):
            captured["chat_kwargs"] = dict(kwargs)
            return {"choices": [{"message": {"content": "pending-native-ok"}}]}

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    monkeypatch.setenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_CACHE", "1")
    monkeypatch.delenv("IPFS_ACCELERATE_LLAMA_CPP_NATIVE_MODEL_PATH", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_LLAMA_CPP_MODEL_PATH", raising=False)
    monkeypatch.setattr(
        llama_cpp_utils,
        "llama_cpp_model_cache_status",
        lambda _config: llama_cpp_utils.LlamaCppModelCacheStatus(
            repo_id="owner/model",
            filename="model.gguf",
            cache_root=str(tmp_path),
            repo_cache_dir=str(tmp_path),
            complete=True,
            local_path=str(model_path),
            local_size_bytes=4,
            cache_backend="hf_hash_pending",
            content_hash_pending=True,
        ),
    )
    llm_router.clear_llm_router_caches()

    provider = llm_router._builtin_provider_by_name("llama_cpp_native")

    assert provider is not None
    assert provider.generate("prove this", max_tokens=4, temperature=0.0) == "pending-native-ok"
    assert captured["model_path"] == str(model_path)


def test_llama_cpp_kit_can_route_to_native_provider(monkeypatch):
    captured = {}

    def fake_generate_text(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = dict(kwargs)
        return "native-kit-ok"

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)
    llama_cpp_kit = _load_llama_cpp_kit_class()
    kit = llama_cpp_kit(resources={"provider": "llama_cpp_native", "model": "Leanstral"})

    chunks = list(kit.llm_complete("prove this", max_tokens=4, stream=False))

    assert chunks == [{"text": "native-kit-ok", "done": True}]
    assert captured["prompt"] == "prove this"
    assert captured["kwargs"]["provider"] == "llama_cpp_native"
    assert captured["kwargs"]["model_name"] == "Leanstral"
    assert captured["kwargs"]["max_tokens"] == 4


def test_generate_text_batch_preserves_order_and_kwargs():
    class EchoProvider:
        def generate(self, prompt, *, model_name=None, **kwargs):
            return f"{prompt}:{model_name}:{kwargs.get('max_tokens')}"

    out = llm_router.generate_text_batch(
        ["first", "second", "third"],
        model_name="leanstral-local",
        provider="llama_cpp",
        provider_instance=EchoProvider(),
        max_workers=3,
        max_tokens=8,
        temperature=0.0,
    )

    assert out == [
        "first:leanstral-local:8",
        "second:leanstral-local:8",
        "third:leanstral-local:8",
    ]


def test_generate_text_mesh_forwards_llama_sampling_kwargs(monkeypatch):
    captured = {}

    def fake_submit_task(**kwargs):
        captured.update(kwargs)
        return "task-1"

    def fake_wait_task(task_id, *, queue_path=None, timeout_s=0):
        return {"status": "completed", "result": {"text": f"done:{task_id}"}}

    monkeypatch.setattr(llm_router, "submit_task", fake_submit_task)
    monkeypatch.setattr(llm_router, "wait_task", fake_wait_task)

    text = llm_router.generate_text_mesh(
        "draft lean proof",
        provider="llama_cpp",
        model_name="leanstral-local",
        max_tokens=16,
        temperature=0.1,
        top_p=0.9,
        stop=["```"],
        seed=7,
        response_format={"type": "json_object"},
    )

    assert text == "done:task-1"
    assert captured["task_type"] == "llm.generate"
    assert captured["provider"] == "llama_cpp"
    assert captured["model_name"] == "leanstral-local"
    assert captured["max_tokens"] == 16
    assert captured["temperature"] == 0.1
    assert captured["top_p"] == 0.9
    assert captured["stop"] == ["```"]
    assert captured["seed"] == 7
    assert captured["response_format"] == {"type": "json_object"}


def test_generate_text_mesh_batch_preserves_order(monkeypatch):
    submissions = []

    def fake_submit_task(**kwargs):
        task_id = f"task-{len(submissions)}"
        submissions.append(dict(kwargs))
        return task_id

    def fake_wait_task(task_id, *, queue_path=None, timeout_s=0):
        idx = int(str(task_id).split("-")[-1])
        prompt = submissions[idx]["prompt"]
        return {"status": "completed", "result": {"text": f"out:{prompt}"}}

    monkeypatch.setattr(llm_router, "submit_task", fake_submit_task)
    monkeypatch.setattr(llm_router, "wait_task", fake_wait_task)

    out = llm_router.generate_text_mesh_batch(
        ["a", "b", "c"],
        provider="llama_cpp",
        model_name="leanstral-local",
        max_workers=3,
        max_new_tokens=32,
        top_p=0.8,
    )

    assert out == ["out:a", "out:b", "out:c"]
    assert [entry["prompt"] for entry in submissions] == ["a", "b", "c"]
    assert all(entry["provider"] == "llama_cpp" for entry in submissions)
    assert all(entry["max_new_tokens"] == 32 for entry in submissions)
    assert all(entry["top_p"] == 0.8 for entry in submissions)


def test_p2p_worker_all_provider_policy_includes_llama_cpp_and_forwards_kwargs(monkeypatch):
    from ipfs_accelerate_py.p2p_tasks import worker as p2p_worker

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ALLOWED_LLM_PROVIDERS", "all")
    allowed = p2p_worker._allowed_llm_providers()
    assert "llama_cpp" in allowed
    assert "llama_cpp_native" in allowed

    captured = {}

    def fake_generate_text(prompt, *, model_name=None, provider=None, **kwargs):
        captured["prompt"] = prompt
        captured["model_name"] = model_name
        captured["provider"] = provider
        captured["kwargs"] = dict(kwargs)
        return "worker-ok"

    monkeypatch.setattr(llm_router, "generate_text", fake_generate_text)

    out = p2p_worker._run_llm_generate(
        {
            "model_name": "leanstral-local",
            "assigned_worker": "worker-1",
            "payload": {
                "prompt": "draft lean proof",
                "provider": "llama_cpp",
                "max_tokens": 9,
                "temperature": 0.0,
                "top_p": 0.7,
                "seed": 123,
                "response_format": {"type": "json_object"},
            },
        }
    )

    assert out["text"] == "worker-ok"
    assert out["provider"] == "llama_cpp"
    assert captured["prompt"] == "draft lean proof"
    assert captured["model_name"] == "leanstral-local"
    assert captured["provider"] == "llama_cpp"
    assert captured["kwargs"]["max_tokens"] == 9
    assert captured["kwargs"]["temperature"] == 0.0
    assert captured["kwargs"]["top_p"] == 0.7
    assert captured["kwargs"]["seed"] == 123
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}
