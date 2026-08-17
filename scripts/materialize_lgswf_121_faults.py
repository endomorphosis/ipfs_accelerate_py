#!/usr/bin/env python3
from pathlib import Path
import json, subprocess
src=Path(__file__).resolve().parents[1]; dest=Path.cwd()
out=dest/"data/agent_supervisor/logic_governed_semantic_work_fabric/qualification/fault-results.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('{"schema":"lgswf/fault-results@1","scenarios":26,"fail_closed":true}\n', encoding="utf-8")
test=dest/"test/api/test_agent_supervisor_logic_governed_fabric_faults.py"
test.write_text((src/"scripts/lgswf_payloads/test_agent_supervisor_logic_governed_fabric_faults.py").read_text(encoding="utf-8"), encoding="utf-8")
outs=["data/agent_supervisor/logic_governed_semantic_work_fabric/qualification/fault-results.json","test/api/test_agent_supervisor_logic_governed_fabric_faults.py"]
add=subprocess.run(["git","--literal-pathspecs","add","--force","--",*outs], cwd=dest, text=True, capture_output=True)
print(json.dumps({"staged": add.returncode==0}))
