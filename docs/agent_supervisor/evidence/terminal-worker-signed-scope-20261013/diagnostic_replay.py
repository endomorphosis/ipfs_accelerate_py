"""Read-only reproduction from one already failed authored fixture; no pytest or provider."""
from pathlib import Path
import json,hashlib,sys
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextCapsule
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import render_context_capsule
from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import load_public_instruction
artifact,capsule,workspace=map(Path,sys.argv[1:])
payload=json.loads(artifact.read_bytes())
prompt=render_context_capsule(ContextCapsule.from_dict(json.loads(capsule.read_bytes())))
block,receipt=load_public_instruction(artifact=artifact,expected_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),task_cid=payload["task_cid"],prompt=prompt,workspace=workspace,repository=Path(payload["repository"]))
print(json.dumps({"schema":receipt["schema"],"block_bytes":len(block.encode()),"external_provider_calls":0,"qualification":False}))
