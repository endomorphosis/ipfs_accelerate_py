import json
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.entrypoints.logic_governed_release import build_release
def test_release_manifest():
    rec = build_release({"level": "candidate"})
    assert rec["schema"] == "lgswf/qualification-release@1"
    root = Path(__file__).resolve().parents[2]
    data = json.loads((root/"data/agent_supervisor/logic_governed_semantic_work_fabric/release/qualification-release.json").read_text())
    assert data["schema"] == rec["schema"]
