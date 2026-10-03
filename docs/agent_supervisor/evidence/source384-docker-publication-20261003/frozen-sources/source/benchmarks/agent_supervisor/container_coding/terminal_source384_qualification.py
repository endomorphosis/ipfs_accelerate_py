"""Offline original-container resource and Source384 preparation qualification.

The retained receipts establish this bounded path only, never an official task
score or source-qualified proof. No planner/provider or benchmark verifier runs.
"""
import hashlib
import json
from pathlib import Path
import shlex

from .benchmark_resource_profile import SOURCE384_PROFILE, SOURCE384_ENVIRONMENT

MIB = 1024 * 1024
MAX_RESULT_BYTES = 1024 * 1024
RESULT_PATH = "/opt/ipfs-supervisor/state/source384-qualification-result.json"

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
started=time.monotonic(); phase='runtime_load';phase_started=started
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
  phase='prepare';phase_started=before=time.monotonic()
  prepared=prep.prepare(repository=root,instruction=pathlib.Path('/opt/ipfs-supervisor/source384-public-instruction.md'),state=state)
  result['prepare_seconds']=time.monotonic()-before
  phase='initial_context';phase_started=before=time.monotonic()
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
  phase='warm_observation';phase_started=before=time.monotonic()
  validate_source384_context(repository=root,expected_receipt=receipt)
  result['warm_observation_seconds']=time.monotonic()-before
  result.update(qualified=True,checkpoint_sha256=receipt['checkpoint_sha256'],
   config_sha256=receipt['config_sha256'],config_path=receipt['config_path'],
   inference_sha256=receipt['inference_sha256'],source_head=receipt['source_head'],
   signed_source_hashes=receipt['source_hashes'],coverage=receipt['summary']['coverage'],
   source384_summary=receipt['summary'],source384_resource_profile=receipt['resource_profile'],
   native_worker_receipt=inference['report']['worker_receipt'],
   native_inference_key=inference['report']['key'],native_worker_executed=True,
   inference_executed=True,neural_inference_replayed=False,
   source384_seconds=receipt['seconds'],context_seconds=context['seconds'],
   context_nonoverlapping_seconds=context['nonoverlapping_seconds'])
except BaseException as exc:
 import traceback
 traceback.print_exc(file=sys.stderr,limit=20)
 result.update(error_type=type(exc).__name__,error=str(exc)[:2048],error_phase=phase,phase_seconds=time.monotonic()-phase_started)
 try:
  from dataclasses import asdict
  from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import collect_proof_host_resources
  result['failure_resources']=asdict(collect_proof_host_resources())
 except Exception as diagnostic_error: result['failure_resource_error']=type(diagnostic_error).__name__
finally:
 signal.setitimer(signal.ITIMER_REAL,0)
 result['seconds']=time.monotonic()-started
raw=json.dumps(result,sort_keys=True,allow_nan=False).encode()
if len(raw)>1024*1024: raise ValueError('bounded qualification result required')
with pathlib.Path('/opt/ipfs-supervisor/state/source384-qualification-result.json').open('xb') as stream:stream.write(raw)
print(raw.decode())
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
        exact_memory = int(value["memory_max"]) == SOURCE384_ENVIRONMENT["override_memory_mb"] * MIB
    except (AttributeError, TypeError, ValueError):
        raise ValueError("finite enforced CPU and memory limits required") from None
    if (not exact_cpu or not exact_memory or type(value["detected_cpu_slots"]) is not int
            or value["detected_cpu_slots"] != 5 or type(value["detected_total_memory_mb"]) is not int
            or value["detected_total_memory_mb"] != SOURCE384_ENVIRONMENT["override_memory_mb"]
            or type(value["available_memory_mb"]) is not int or not 0 <= value["available_memory_mb"] <= SOURCE384_ENVIRONMENT["override_memory_mb"]):
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
    total = observed["detected_total_memory_mb"]
    headroom = total - int(total * .8)
    estimate = dict(schema="source384-admission-estimate@1", live_available_memory_mb=observed["available_memory_mb"],
        derived_default_headroom_mb=headroom, requested_parent_memory_mb=6144,
        minimum_available_memory_mb=headroom + 6144, exact_admission_decision=False,
        actual_native_admission_required=True)
    (output / "admission-estimate.json").write_text(json.dumps(estimate, indent=2) + "\n")
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
    (output / "source384-probe-status.json").write_text(json.dumps({"return_code": response.return_code}) + "\n")
    report_path = output / "source384-result.json"
    await environment.download_file(RESULT_PATH, report_path)
    result = json.loads(_regular_bytes(report_path.resolve(strict=True), MAX_RESULT_BYTES))
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
    inference_path = output / "native-inference.json"
    await environment.download_file(ROOT + "/state/source384-qualification/source384-context/inference.json", inference_path)
    inference_raw = _regular_bytes(inference_path.resolve(strict=True), 32 * 1024 * 1024)
    if (hashlib.sha256(inference_raw).hexdigest() != result["inference_sha256"]
            or json.loads(inference_raw)["report"]["key"] != result["native_inference_key"]):
        raise ValueError("exported native inference differs from qualified immutable receipt")
    return result
