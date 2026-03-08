import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
IMPORT_CHECKER_PATH = REPO_ROOT / "scripts" / "dev_tools" / "comprehensive_import_checker.py"
VALIDATE_SETUP_PATH = REPO_ROOT / "scripts" / "validation" / "validate_setup.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import_checker = _load_module("comprehensive_import_checker", IMPORT_CHECKER_PATH)
validate_setup = _load_module("validate_setup", VALIDATE_SETUP_PATH)


class _FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


def test_optional_module_prefix_matches_nested_modules():
    assert import_checker._is_optional_module("sklearn")
    assert import_checker._is_optional_module("sklearn.pipeline")
    assert import_checker._is_optional_module("sentence_transformers")
    assert not import_checker._is_optional_module("json")


def test_check_imports_allows_known_optional_dependencies(tmp_path):
    sample = tmp_path / "sample_optional.py"
    sample.write_text(
        "import sklearn.pipeline\nfrom sentence_transformers import SentenceTransformer\n",
        encoding="utf-8",
    )

    ok, errors = import_checker.check_imports(sample)

    assert ok is True
    assert errors == []


def test_check_imports_still_flags_real_missing_modules(tmp_path):
    sample = tmp_path / "sample_missing.py"
    sample.write_text("import definitely_missing_pkg_12345\n", encoding="utf-8")

    ok, errors = import_checker.check_imports(sample)

    assert ok is False
    assert errors == ["Cannot import 'definitely_missing_pkg_12345'"]


def test_check_imports_allows_guarded_optional_imports(tmp_path):
    sample = tmp_path / "sample_guarded_optional.py"
    sample.write_text(
        "try:\n"
        "    import definitely_missing_pkg_12345\n"
        "except ImportError:\n"
        "    definitely_missing_pkg_12345 = None\n",
        encoding="utf-8",
    )

    ok, errors = import_checker.check_imports(sample)

    assert ok is True
    assert errors == []


def test_wait_for_http_ready_succeeds_when_probe_returns_200(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        if url.endswith("/"):
            return _FakeResponse(200)
        raise AssertionError("unexpected probe order")

    monkeypatch.setattr(validate_setup.requests, "get", fake_get)

    ready, error = validate_setup.wait_for_http_ready(
        "http://127.0.0.1:9010",
        _FakeProcess(returncode=None),
        timeout=0.1,
    )

    assert ready is True
    assert error is None
    assert calls == [("http://127.0.0.1:9010/", 2)]


def test_wait_for_http_ready_reports_early_exit():
    ready, error = validate_setup.wait_for_http_ready(
        "http://127.0.0.1:9010",
        _FakeProcess(returncode=7),
        timeout=0.1,
    )

    assert ready is False
    assert error == "server process exited early with code 7"


def test_resolve_cli_command_prefers_virtualenv_binary(tmp_path, monkeypatch):
    venv_dir = tmp_path / "venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    binary = bin_dir / "ipfs-accelerate"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setenv("VIRTUAL_ENV", str(venv_dir))
    monkeypatch.setattr(validate_setup, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_setup.shutil, "which", lambda _name: None)

    assert validate_setup._resolve_cli_command() == [str(binary)]


def test_resolve_cli_command_uses_local_venv_before_path(tmp_path, monkeypatch):
    local_bin = tmp_path / ".venv" / "bin"
    local_bin.mkdir(parents=True)
    binary = local_bin / "ipfs-accelerate"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(validate_setup, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_setup.shutil, "which", lambda _name: "/usr/bin/ipfs-accelerate")

    assert validate_setup._resolve_cli_command() == [str(binary)]


def test_resolve_cli_command_falls_back_to_module(monkeypatch, tmp_path):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(validate_setup, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_setup.shutil, "which", lambda _name: None)

    assert validate_setup._resolve_cli_command() == [validate_setup.sys.executable, "-m", "ipfs_accelerate_py.cli_entry"]
