import sys,json,importlib.util
sys.path[:0]=['/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002', '/home/barberb/lift_coding/.worktrees/ir-release-datasets-20261002']
for forbidden in ('torch','duckdb','numpy','pandas','multiformats','requests'):
    if importlib.util.find_spec(forbidden) is not None:raise RuntimeError('third-party dependency unexpectedly visible: '+forbidden)
from benchmarks.agent_supervisor.container_coding.terminal_deployment import verify_extracted_torch_cpu_wheel
binding=verify_extracted_torch_cpu_wheel('/home/barberb/lift_coding/artifacts/source384-torch-wheel-20261003/torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl','6f307c2c32d764ffc6ff6893b801fad6d4752f3e67966cb8abf1843427c02604',155005253)
if any(name in sys.modules for name in ('torch','duckdb','numpy','pandas','multiformats','requests')):raise RuntimeError('third-party dependency imported')
print(json.dumps({'schema':'stdlib-only-torch-wheel-verifier@1','binding':binding,'python':sys.version,'isolated':sys.flags.isolated,'no_site':sys.flags.no_site,'third_party_packages_visible':False,'third_party_modules_imported':False},sort_keys=True))
