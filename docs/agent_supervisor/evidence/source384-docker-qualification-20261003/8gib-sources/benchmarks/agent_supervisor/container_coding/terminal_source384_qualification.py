"""Offline original-container resource and Source384 preparation qualification.

The retained receipts establish this bounded path only, never an official task
score or source-qualified proof. No planner/provider or benchmark verifier runs.
"""
import json
from pathlib import Path
import shlex

from .benchmark_resource_profile import SOURCE384_PROFILE, SOURCE384_ENVIRONMENT

MIB = 1024 * 1024

RESOURCE_PROBE = r'''
import json,pathlib
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import collect_proof_host_resources
base=pathlib.Path('/sys/fs/cgroup'); current=base
for line in pathlib.Path('/proc/self/cgroup').read_text().splitlines():
 if line.startswith('0::'):
  relative=pathlib.Path(line[3:].lstrip('/'))
  if '..' not in relative.parts and (base/relative).is_dir(): current=base/relative
  break
host=collect_proof_host_resources()
print(json.dumps(dict(schema='terminal-source384-cgroup-observation@1',
 cgroup_path=str(current),cpu_max=(current/'cpu.max').read_text().strip(),
 memory_max=(current/'memory.max').read_text().strip(),
 detected_cpu_slots=host.cpu_slots,detected_total_memory_mb=host.total_memory_mb,
 available_memory_mb=host.available_memory_mb)))
'''

CONTEXT_PROBE = r'''
from contextlib import redirect_stdout
import hashlib,json,pathlib,signal,sys,time
started=time.monotonic()
result=dict(schema='terminal-source384-original-container-qualification@1',qualified=False,
 provider_calls=0,official_verifier_executed=False,benchmark_result=False,
 source_qualified_proof_claimed=False,training_steps=0,download_calls=0)
def expired(*args): raise TimeoutError('bounded Source384 container qualification expired')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,270)
try:
 with redirect_stdout(sys.stderr):
  from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
  from ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context import validate_source384_context, _pins, _read, MAX_INFERENCE_BYTES
  root=pathlib.Path('/app'); state=pathlib.Path('/opt/ipfs-supervisor/state/source384-qualification')
  config_path=pathlib.Path('/opt/ipfs-supervisor/models/source384/config.json')
  result['producer']=_pins()
  before=time.monotonic()
  prepared=prep.prepare(repository=root,instruction=pathlib.Path('/opt/ipfs-supervisor/source384-public-instruction.md'),state=state)
  result['prepare_seconds']=time.monotonic()-before
  before=time.monotonic()
  context=prep.initial_context(state=state,source384_config=config_path,train_autoencoder=False)
  result['initial_context_seconds']=time.monotonic()-before
  receipt=context['source384_context']
  inference_path=pathlib.Path(receipt['output'])/'inference.json'
  raw=_read(inference_path,MAX_INFERENCE_BYTES);inference=json.loads(raw)
  if hashlib.sha256(raw).hexdigest()!=receipt['inference_sha256']:raise ValueError('native inference digest differs')
  if inference['native_worker_executed'] is not True or inference['inference_executed'] is not True:
   raise ValueError('qualification requires actual pinned parent inference')
  if inference['report']['output']['model_loads']!=1:raise ValueError('one actual GTE model load required')
  if any(receipt[k] is not False for k in ('proof_authority','execution_authority','completion_authority','formalization_authority')):
   raise ValueError('Source384 preparation cannot grant authority')
  if receipt['config_path']!=str(config_path):raise ValueError('relocated model config was not consumed')
  before=time.monotonic()
  validate_source384_context(repository=root,expected_receipt=receipt)
  result['warm_observation_seconds']=time.monotonic()-before
  result.update(qualified=True,checkpoint_sha256=receipt['checkpoint_sha256'],
   config_sha256=receipt['config_sha256'],config_path=receipt['config_path'],
   inference_sha256=receipt['inference_sha256'],source_head=receipt['source_head'],
   signed_source_hashes=receipt['source_hashes'],coverage=receipt['summary']['coverage'],
   source384_summary=receipt['summary'],source384_resource_profile=receipt['resource_profile'],
   native_worker_receipt=inference['report']['worker_receipt'],native_worker_executed=True,
   inference_executed=True,neural_inference_replayed=False,
   source384_seconds=receipt['seconds'],context_seconds=context['seconds'],
   context_nonoverlapping_seconds=context['nonoverlapping_seconds'])
except BaseException as exc:
 result.update(error_type=type(exc).__name__,error=str(exc)[:2048])
finally:
 signal.setitimer(signal.ITIMER_REAL,0)
 result['seconds']=time.monotonic()-started
print(json.dumps(result,sort_keys=True,allow_nan=False))
raise SystemExit(0 if result['qualified'] else 1)
'''


def resource_options(profile):
    if profile is None:
        return {}
    if profile != SOURCE384_PROFILE:
        raise ValueError("unknown benchmark resource profile")
    return dict(SOURCE384_ENVIRONMENT)


def validate_resource_observation(value, profile):
    if profile != SOURCE384_PROFILE or type(value) is not dict:
        raise ValueError("explicit named Source384 resource observation required")
    fields = {"schema", "cgroup_path", "cpu_max", "memory_max", "detected_cpu_slots",
              "detected_total_memory_mb", "available_memory_mb"}
    if set(value) != fields or value["schema"] != "terminal-source384-cgroup-observation@1":
        raise ValueError("closed cgroup resource observation required")
    try:
        quota, period = value["cpu_max"].split()
        exact_cpu = int(period) > 0 and int(quota) == 5 * int(period)
        exact_memory = int(value["memory_max"]) == 8192 * MIB
    except (AttributeError, TypeError, ValueError):
        raise ValueError("finite enforced CPU and memory limits required") from None
    if (not exact_cpu or not exact_memory or type(value["detected_cpu_slots"]) is not int
            or value["detected_cpu_slots"] != 5 or type(value["detected_total_memory_mb"]) is not int
            or value["detected_total_memory_mb"] != 8192
            or type(value["available_memory_mb"]) is not int or not 0 <= value["available_memory_mb"] <= 8192):
        raise ValueError("actual cgroup limits differ from the named common profile")
    return value


async def observe_resources(environment, *, output, profile):
    from .terminal_deployment import PYTHON, runtime_environment
    response = await environment.exec(command=PYTHON + " -P -c " + shlex.quote(RESOURCE_PROBE),
        cwd="/app", user="supervisor", env=runtime_environment(), timeout_sec=30)
    output = Path(output)
    (output / "resource-probe.stdout").write_text(response.stdout or "")
    (output / "resource-probe.stderr").write_text(response.stderr or "")
    if response.return_code:
        raise ValueError("actual cgroup resource probe failed")
    observed = validate_resource_observation(json.loads(response.stdout), profile)
    (output / "resources.json").write_text(json.dumps(observed, indent=2) + "\n")
    return observed


async def qualify_context(environment, *, task_dir, output, manifest, profile):
    from .terminal_deployment import ROOT, PYTHON, runtime_environment, validate_source384_binding
    from ipfs_accelerate_py.agent_supervisor.runtime.source384_config import _regular_bytes
    if profile != SOURCE384_PROFILE or validate_source384_binding(manifest) is None:
        raise ValueError("Source384 qualification requires pinned assets and the explicit common profile")
    output = Path(output)
    raw = _regular_bytes(Path(task_dir).resolve(strict=True) / "instruction.md", 32768)
    instruction = output / "public-instruction.md"
    instruction.write_bytes(raw)
    await environment.upload_file(instruction, ROOT + "/source384-public-instruction.md")
    response = await environment.exec(command=PYTHON + " -P -c " + shlex.quote(CONTEXT_PROBE),
        cwd="/app", user="supervisor", env=runtime_environment(), timeout_sec=300)
    (output / "source384-probe.stdout").write_text(response.stdout or "")
    (output / "source384-probe.stderr").write_text(response.stderr or "")
    result = json.loads(response.stdout)
    (output / "source384-context.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if response.return_code or result.get("qualified") is not True:
        raise ValueError("original-container Source384 qualification failed; see retained receipt")
    if (type(result.get("provider_calls")) is not int or result["provider_calls"] != 0
            or result.get("official_verifier_executed") is not False
            or result.get("benchmark_result") is not False
            or result.get("checkpoint_sha256") != manifest["source384"]["config"]["checkpoint_sha256"]
            or result.get("config_sha256") != manifest["source384"]["config_sha256"]):
        raise ValueError("Source384 qualification identity or scope differs")
    expected = {row["path"]: row["sha256"] for row in manifest["files"]}
    producer = result["producer"]
    owner_modules = dict(producer["source_owners"])
    owner_modules.update({"ipfs_accelerate_py.agent_supervisor.runtime." + name: producer[key]
        for key, name in (("consumer", "source384_repository_context"),
                          ("config", "source384_config"), ("byte_reader", "security_autoencoder_advisor"))})
    if any(expected.get(("datasets/" if name.startswith("ipfs_datasets_py.") else "source/") + name.replace(".", "/") + ".py") != digest
           for name, digest in owner_modules.items()):
        raise ValueError("actual Source384 producer differs from runtime archive")
    return result
