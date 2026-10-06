"""Native Codex schema custody and cleanup without any model invocation."""
import copy
import hashlib
import json
from pathlib import Path
import os
import stat
import subprocess
import sys
from types import MappingProxyType

import pytest


def output_schema():
    return {"type": "object", "additionalProperties": False, "required": ["status"],
            "properties": {"status": {"type": "string", "enum": ["READY"]}}}


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    workspace = tmp_path / "work space"
    workspace.mkdir()
    temporary = tmp_path / "temporary files"
    temporary.mkdir()
    executable = tmp_path / "codex"
    executable.write_text("#!" + sys.executable + "\n" + '''
import hashlib,json,os,pathlib,stat,sys
args=sys.argv[1:]
record={"argv":args,"prompt":sys.stdin.read()}
if "--output-schema" in args:
    path=pathlib.Path(args[args.index("--output-schema")+1])
    raw=path.read_bytes();info=path.lstat()
    record.update(schema=json.loads(raw),schema_bytes=len(raw),schema_sha256=hashlib.sha256(raw).hexdigest(),
                  schema_mode=stat.S_IMODE(info.st_mode),schema_regular=stat.S_ISREG(info.st_mode))
pathlib.Path("calls.json").write_text(json.dumps(record))
pathlib.Path(args[args.index("--output-last-message")+1]).write_text('{"status":"READY"}')
print(json.dumps({"type":"thread.started","thread_id":"fixture-session"}))
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":13,"output_tokens":2,"total_tokens":15}}))
if os.getenv("CODEX_FIXTURE_EXIT"):
    print("fixture failure",file=sys.stderr)
    raise SystemExit(int(os.environ["CODEX_FIXTURE_EXIT"]))
''')
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("ipfs_accelerate_py_CODEX_SANDBOX", raising=False)
    monkeypatch.delenv("CODEX_FIXTURE_EXIT", raising=False)
    monkeypatch.chdir(workspace)
    create = llm_router.tempfile.NamedTemporaryFile
    files = []
    def temporary_file(**kwargs):
        handle = create(dir=temporary, **kwargs)
        files.append(Path(handle.name))
        return handle
    monkeypatch.setattr(llm_router.tempfile, "NamedTemporaryFile", temporary_file)
    return llm_router, workspace, temporary, files


def test_native_schema_file_is_private_canonical_and_receipted(adapter):
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    router, workspace, temporary, files = adapter
    selected = output_schema()
    original = copy.deepcopy(selected)
    prompt = "literal $(unexecuted); `unchanged`"
    result = router._get_codex_cli_provider().generate(prompt, model_name="unregistered-model",
        timeout=19, reasoning_effort="high", codex_output_schema=MappingProxyType(selected))
    assert result == '{"status":"READY"}'
    call = json.loads((workspace / "calls.json").read_text())
    assert call["prompt"] == prompt
    assert call["argv"][call["argv"].index("-m") + 1] == "unregistered-model"
    assert call["schema"] == selected == original
    assert call["schema_mode"] == 0o600 and call["schema_regular"] is True
    canonical = json.dumps(selected, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    assert call["schema_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert call["schema_bytes"] == len(canonical)
    assert len(files) == 2 and all(not p.exists() for p in files)
    assert not list(temporary.iterdir())
    observation = get_last_cli_observation("codex_cli")
    assert observation["codex_output_schema_sha256"] == call["schema_sha256"]
    assert observation["codex_output_schema_bytes"] == call["schema_bytes"]
    assert observation["prompt_tokens"] == 13 and observation["completion_tokens"] == 2
    assert "codex_output_schema" not in observation


def test_ordinary_provider_omits_schema_and_retains_old_command(adapter):
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    router, workspace, _temporary, files = adapter
    assert router._get_codex_cli_provider().generate("unchanged", model_name="explicit-model") == '{"status":"READY"}'
    call = json.loads((workspace / "calls.json").read_text())
    assert call["argv"] == ["exec", "--skip-git-repo-check", "-m", "explicit-model",
                             "--output-last-message", str(files[0]), "--json", "-"]
    observation = get_last_cli_observation("codex_cli")
    assert "codex_output_schema_sha256" not in observation
    assert "codex_output_schema_bytes" not in observation
    assert len(files) == 1 and not files[0].exists()


def test_command_builder_forwards_literal_schema_path_without_other_changes():
    from ipfs_accelerate_py import llm_router
    path = Path("/schema space/literal $(unexecuted).json")
    ordinary = llm_router.build_codex_cli_command(model_name="unknown", json_mode=True)
    assert ordinary == ["codex", "exec", "-m", "unknown", "--json", "-"]
    assert llm_router.build_codex_cli_command(model_name="unknown", json_mode=True, output_schema=None) == ordinary
    assert llm_router.build_codex_cli_command(model_name="unknown", json_mode=True, output_schema=path) == [
        "codex", "exec", "-m", "unknown", "--output-schema", str(path), "--json", "-"]


@pytest.mark.parametrize("path", ["", " ", "bad\x00path", False, {"schema": True}])
def test_command_builder_rejects_bad_schema_path(path):
    from ipfs_accelerate_py import llm_router
    with pytest.raises(ValueError, match="output_schema"):
        llm_router.build_codex_cli_command(output_schema=path)


@pytest.mark.parametrize("kind", ["none", "list", "string", "cycle", "depth", "bytes", "nan", "infinity",
                                  "remote", "dynamic", "recursive", "invalid", "dialect", "non_json", "bad_key"])
def test_bad_schema_fails_before_temporary_files_or_native_process(adapter, monkeypatch, kind):
    router, workspace, temporary, files = adapter
    selected = output_schema()
    if kind == "none": selected = None
    elif kind == "list": selected = []
    elif kind == "string": selected = "schema.json"
    elif kind == "cycle": selected["cycle"] = selected
    elif kind == "depth":
        child = selected
        for _ in range(34):
            child["child"] = {}
            child = child["child"]
    elif kind == "bytes": selected["description"] = "x" * 65_536
    elif kind == "nan": selected["maximum"] = float("nan")
    elif kind == "infinity": selected["maximum"] = float("inf")
    elif kind in {"remote", "dynamic", "recursive"}:
        ref = {"remote": "$ref", "dynamic": "$dynamicRef", "recursive": "$recursiveRef"}[kind]
        selected["properties"]["status"] = {ref: "https://invalid.example/schema"}
    elif kind == "invalid": selected["properties"]["status"]["type"] = "unknown-type"
    elif kind == "dialect": selected["$schema"] = "https://invalid.example/dialect"
    elif kind == "non_json": selected["description"] = object()
    else: selected[1] = "invalid key"
    monkeypatch.setattr(router.subprocess, "run", lambda *a, **kw: pytest.fail("native process reached"))
    with pytest.raises(ValueError):
        router._get_codex_cli_provider().generate("do not dispatch", codex_output_schema=selected)
    assert not files and not list(temporary.iterdir())
    assert not (workspace / "calls.json").exists()


def test_local_definition_reference_is_forwarded_without_rewriting(adapter):
    router, workspace, _temporary, _files = adapter
    selected = output_schema()
    selected["$defs"] = {"status": selected["properties"]["status"]}
    selected["properties"]["status"] = {"$ref": "#/$defs/status"}
    router._get_codex_cli_provider().generate("local refs", codex_output_schema=selected)
    assert json.loads((workspace / "calls.json").read_text())["schema"] == selected


@pytest.mark.parametrize("failure", ["timeout", "interrupt", "missing", "build"])
def test_both_temporary_files_are_cleaned_on_exception(adapter, monkeypatch, failure):
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    router, workspace, temporary, files = adapter
    if failure == "build":
        monkeypatch.setenv("ipfs_accelerate_py_CODEX_SANDBOX", "unsupported")
        expected = ValueError
    else:
        def execute(argv, **kwargs):
            assert kwargs["input"] == "original stdin" and kwargs["timeout"] == 3
            assert Path(argv[argv.index("--output-schema") + 1]).exists()
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, 3,
                    output=b'{"type":"thread.started","thread_id":"partial-fixture"}\n', stderr=b"partial")
            if failure == "interrupt": raise KeyboardInterrupt()
            raise FileNotFoundError("missing fixture")
        monkeypatch.setattr(router.subprocess, "run", execute)
        expected = {"timeout": subprocess.TimeoutExpired, "interrupt": KeyboardInterrupt, "missing": router.LLMRouterError}[failure]
    with pytest.raises(expected):
        router._get_codex_cli_provider().generate("original stdin", timeout=3,
            reasoning_effort="high", codex_output_schema=output_schema())
    assert len(files) == 2 and all(not p.exists() for p in files)
    assert not list(temporary.iterdir()) and not (workspace / "calls.json").exists()
    if failure == "timeout":
        observation = get_last_cli_observation("codex_cli")
        assert observation["timed_out"] is True and "exit_code" not in observation
        assert observation["thread_id"] == "partial-fixture"
        assert observation["codex_output_schema_bytes"] > 0


def test_native_nonzero_exit_cannot_return_structured_partial_message(adapter, monkeypatch):
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    router, _workspace, temporary, files = adapter
    monkeypatch.setenv("CODEX_FIXTURE_EXIT", "2")
    with pytest.raises(router.LLMRouterError, match="fixture failure"):
        router._get_codex_cli_provider().generate("must fail", codex_output_schema=output_schema())
    assert len(files) == 2 and all(not p.exists() for p in files) and not list(temporary.iterdir())
    assert get_last_cli_observation("codex_cli")["exit_code"] == 2
