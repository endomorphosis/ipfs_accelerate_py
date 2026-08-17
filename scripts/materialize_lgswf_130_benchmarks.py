#!/usr/bin/env python3
from pathlib import Path
import json, subprocess
dest=Path.cwd()
root=dest/"benchmarks/logic_governed_semantic_work_fabric"
root.mkdir(parents=True, exist_ok=True)
(root/"manifest.json").write_text('{"schema":"lgswf/benchmark-corpus@1","suites":["A","B","C","D"]}\n', encoding="utf-8")
src=Path(__file__).resolve().parents[1]
test=dest/"test/api/test_agent_supervisor_lgswf_benchmark_manifest.py"
test.write_text((src/"scripts/lgswf_payloads/test_agent_supervisor_lgswf_benchmark_manifest.py").read_text(encoding="utf-8"), encoding="utf-8")
validator=root/"validate_results.py"
validator.write_text((src/"scripts/lgswf_payloads/validate_results.py").read_text(encoding="utf-8"), encoding="utf-8")
outs=["benchmarks/logic_governed_semantic_work_fabric/manifest.json","benchmarks/logic_governed_semantic_work_fabric/validate_results.py","test/api/test_agent_supervisor_lgswf_benchmark_manifest.py"]
add=subprocess.run(["git","--literal-pathspecs","add","--force","--",*outs], cwd=dest, text=True, capture_output=True)
print(json.dumps({"staged": add.returncode==0}))
