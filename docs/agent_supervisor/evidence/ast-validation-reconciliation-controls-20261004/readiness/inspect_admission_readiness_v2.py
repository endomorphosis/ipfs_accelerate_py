from pathlib import Path
import json,time
from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
import importlib.util
spec=importlib.util.spec_from_file_location('_readiness_diagnostics','/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002/benchmarks/agent_supervisor/container_coding/terminal_resource_diagnostics.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
collect_failure_admission=module.collect_failure_admission
result={"schema":"isolated-current-admission-readiness@1","prior_failure_causal_claim":False,"provider_calls":0,"training_steps":0,"production_ledger_mutated":False}
started=time.monotonic()
try:
 with acquire_codebase_resources(timeout_seconds=0,memory_mb=512):result["status"]="admitted_then_released"
except Exception as error:
 result.update(status="refused",error_type=type(error).__name__,admission=collect_failure_admission(error))
result["seconds"]=time.monotonic()-started
print(json.dumps(result,indent=2,sort_keys=True))
