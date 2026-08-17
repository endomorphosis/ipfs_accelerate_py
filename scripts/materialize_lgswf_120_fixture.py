#!/usr/bin/env python3
from pathlib import Path
import json, subprocess
src=Path(__file__).resolve().parents[1]
dest=Path.cwd()
fix=dest/"test/fixtures/logic_governed_semantic_work_fabric"
fix.mkdir(parents=True, exist_ok=True)
(fix/"manifest.json").write_text('{"supervisors": 3, "daemons": 10, "schema": "lgswf/qualification-fixture@1"}\n', encoding="utf-8")
test=dest/"test/api/test_agent_supervisor_lgswf_fixture.py"
test.write_text((src/"scripts/lgswf_payloads/test_agent_supervisor_lgswf_fixture.py").read_text(encoding="utf-8"), encoding="utf-8")
outs=["test/fixtures/logic_governed_semantic_work_fabric/manifest.json","test/api/test_agent_supervisor_lgswf_fixture.py"]
add=subprocess.run(["git","--literal-pathspecs","add","--force","--",*outs], cwd=dest, text=True, capture_output=True)
print(json.dumps({"staged": add.returncode==0}))
