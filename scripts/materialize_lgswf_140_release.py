#!/usr/bin/env python3
from pathlib import Path
import json, subprocess
src=Path(__file__).resolve().parents[1]; dest=Path.cwd()
mod=dest/"ipfs_accelerate_py/agent_supervisor/entrypoints/logic_governed_release.py"
mod.parent.mkdir(parents=True, exist_ok=True)
mod.write_text((src/"scripts/lgswf_payloads/logic_governed_release.py").read_text(encoding="utf-8"), encoding="utf-8")
rel=dest/"data/agent_supervisor/logic_governed_semantic_work_fabric/release/qualification-release.json"
rel.parent.mkdir(parents=True, exist_ok=True)
rel.write_text('{"schema":"lgswf/qualification-release@1","level":"candidate"}\n', encoding="utf-8")
test=dest/"test/api/test_agent_supervisor_logic_governed_release.py"
test.write_text((src/"scripts/lgswf_payloads/test_agent_supervisor_logic_governed_release.py").read_text(encoding="utf-8"), encoding="utf-8")
outs=["ipfs_accelerate_py/agent_supervisor/entrypoints/logic_governed_release.py","test/api/test_agent_supervisor_logic_governed_release.py","data/agent_supervisor/logic_governed_semantic_work_fabric/release/qualification-release.json"]
add=subprocess.run(["git","--literal-pathspecs","add","--force","--",*outs], cwd=dest, text=True, capture_output=True)
print(json.dumps({"staged": add.returncode==0}))
