"""Native source reads must not reuse the repair interpreter's cached modules."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.ops.agent_supervisor.spar_legacy_observation import (
    NativeObservationClient, NativeObservationError,
)


@pytest.fixture
def native_root(tmp_path):
    root = tmp_path / "accepted"
    script = root / "scripts/materialize_semantic_preserving_remodularization_program.py"
    script.parent.mkdir(parents=True)
    (root / "native_source_marker.py").write_text("VALUE = 'accepted'\n")
    script.write_text('''
import json
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
import native_source_marker

def _load_config(path):
    config = json.loads(Path(path).read_text())
    paths = {"root": "data", "merge_queue": "data/queue"}
    return SimpleNamespace(repo_root=ROOT, board_namespace="SPAR", max_lanes=3,
        runtime_paths=paths, payload={"runtime_paths": paths},
        path=lambda name: ROOT / name), config

def _runtime_paths(board):
    return {"owner": ROOT / "data/owner"}

def _assert_clean_current_tree(config):
    if config.get("reject"):
        raise RuntimeError("private diagnostic must not escape")
    return native_source_marker.VALUE, "tree"

def _source_forest(config, head):
    if config.get("mutate"):
        (ROOT / "config.json").write_text("{}")
    return {"root": str(ROOT), "head": head}

if __name__ == "__main__":
    assert sys.argv[-1] == "authoritative-status"
    print(json.dumps({"authoritative_task_observation": True,
                      "source": native_source_marker.VALUE,
                      "isolated": sys.flags.isolated}))
''')
    config = root / "config.json"
    config.write_text('{}')
    return root, config


def test_source_and_native_status_use_accepted_imports(native_root, monkeypatch):
    root, config = native_root
    monkeypatch.setitem(sys.modules, "native_source_marker", SimpleNamespace(VALUE="candidate"))
    monkeypatch.setenv("PYTHONPATH", "/nonexistent/candidate")
    client = NativeObservationClient(root)
    board, _ = client._load_config(config)
    assert board.repo_root == root
    assert board.max_lanes == 3
    assert board.path(board.runtime_paths["root"]) == root / "data"
    assert client._runtime_paths(board) == {"owner": root / "data/owner"}
    binding = client.source_binding(config)
    assert binding["head"] == "accepted"
    assert binding["forest"] == {"root": str(root), "head": "accepted"}
    assert binding["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    status = client.authoritative_status(config)
    assert status == {"authoritative_task_observation": True, "source": "accepted", "isolated": 1}
    assert sys.modules["native_source_marker"].VALUE == "candidate"


@pytest.mark.parametrize("config", [{"reject": True}, {"mutate": True}])
def test_native_rejection_or_configuration_race_cannot_be_an_observation(native_root, config):
    root, path = native_root
    path.write_text(json.dumps(config))
    with pytest.raises(NativeObservationError, match="native isolated observation failed") as exc:
        NativeObservationClient(root).source_binding(path)
    assert "private diagnostic" not in str(exc.value)


def test_full_native_snapshot_hash_has_its_own_bounded_observation_budget(monkeypatch):
    from scripts.ops.agent_supervisor import spar_legacy_observation as observation
    from scripts.ops.agent_supervisor import spar_merge_owner as role
    # A real closeout response duplicates task and receipt populations and can
    # exceed the recovery store's individual receipt budget.
    value = {"tasks": [{"evidence": "x" * (600 * 1024)}]}
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert observation.native_observation_cid(value) == "sha256:" + hashlib.sha256(raw).hexdigest()
    small = {"source": "é"}
    assert observation.native_observation_cid(small) == role._cid(small)
    monkeypatch.setattr(observation, "MAX_OUTPUT", 1024)
    with pytest.raises(NativeObservationError, match="exceeds bound"):
        observation.native_observation_cid(value)
    with pytest.raises(NativeObservationError, match="bounded JSON"):
        observation.native_observation_cid({"invalid": float("nan")})
