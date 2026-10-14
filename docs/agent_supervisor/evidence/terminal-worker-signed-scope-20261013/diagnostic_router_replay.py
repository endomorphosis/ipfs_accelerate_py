"""Read-only router reproduction from retained fixture; provider/UID seams authored."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
import pytest
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextCapsule
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import render_context_capsule
from benchmarks.agent_supervisor.container_coding.test_terminal_worker_signed_scope import invoke,source_endpoint,provider
artifact,capsule,workspace=map(Path,sys.argv[1:]);payload=json.loads(artifact.read_bytes());root=Path(payload["repository"])
prompt=render_context_capsule(ContextCapsule.from_dict(json.loads(capsule.read_bytes())))
worker=SimpleNamespace(prompt=prompt,workspace=workspace,task=SimpleNamespace(task_cid=payload["task_cid"]),case=SimpleNamespace(root=root,state=root.parent/"state"),request={"artifact":artifact,"expected_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),"task_cid":payload["task_cid"]})
with pytest.MonkeyPatch.context() as patch:
 patch.chdir(workspace);patch.setenv("CODEX_HOME",str(root.parent/"authored-empty-codex-home"))
 state,observed=provider.__wrapped__(patch)
 text,receipt=invoke(worker)
 print(json.dumps({"mode":"plain","authored_provider_calls":len(observed),"closing":receipt.get("terminal_semantic_scope_dispatch_rechecked"),"external_provider_calls":0}))
 views,options=source_endpoint(worker,patch)
 text,receipt=invoke(worker,**options)
 print(json.dumps({"mode":"source-view","authored_provider_calls":len(observed),"closing":receipt.get("terminal_semantic_scope_dispatch_rechecked"),"external_provider_calls":0}))
