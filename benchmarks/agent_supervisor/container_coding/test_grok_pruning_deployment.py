"""Profile installation and real worker argument parsing with authored custody."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import container_worker_deployment as deployment
from ipfs_accelerate_py.cli_runtime import grok_pruning as pruning


class Environment:
    def __init__(self):
        self.commands = []
        self.uploads = {}

    async def exec(self, *, command, **kwargs):
        self.commands.append(command)
        if "readlink" in command:
            stdout = json.dumps({name: "authored-namespace" for name in ("pid", "mnt", "net")})
        elif command.endswith("/provider-bin/grok --version"):
            stdout = "grok 1.0.46"
        else:
            stdout = ""
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    async def upload_file(self, source, target):
        self.uploads[target] = Path(source).read_bytes()


@pytest.mark.parametrize("profile", pruning.GROK_PRUNING_PROFILES)
def test_install_preserves_auth_copy_and_only_explicit_profiles_install_config(tmp_path, monkeypatch, profile):
    from benchmarks.agent_supervisor.container_coding import terminal_grok_deployment
    env = Environment()
    monkeypatch.setattr(deployment.subprocess, "check_output", lambda *_a, **_k:
        '"authored-container" "authored-image" 42 true false')
    os = __import__("os")
    original = os.readlink
    monkeypatch.setattr(os, "readlink", lambda path, **kwargs: "authored-namespace"
        if str(path).startswith("/proc/42/ns/") else original(path, **kwargs))
    monkeypatch.setattr(terminal_grok_deployment, "require_grok_version_output", lambda _: None)
    result = asyncio.run(deployment.install_worker_boundary(env, container_name="authored",
        output=tmp_path / "installation", provider="grok_cli", grok_pruning_profile=profile))
    auth = [command for command in env.commands if "grok-auth/auth.json" in command]
    assert len(auth) == 1
    assert "install -o benchmarkworker -g supervisor -m 0600" in auth[0]
    config = deployment.ROOT + "/worker-home/.grok/config.toml"
    declaration = deployment.ROOT + "/grok-pruning.json"
    if profile == "legacy":
        assert config not in env.uploads and declaration not in env.uploads
        assert "grok_pruning_profile" not in result and "grok_pruning_declaration" not in result
    else:
        assert env.uploads[config] == pruning.render_grok_pruning_config(profile)
        assert json.loads(env.uploads[declaration]) == pruning.grok_pruning_declaration(profile)
        assert result["grok_pruning_declaration"] == pruning.grok_pruning_declaration(profile)
        assert result["grok_pruning_profile"] == profile
        assert result["implementation_command"].endswith("--grok-pruning-profile " + profile)
        for path in (config, declaration):
            assert "chown root:root " + path + " && chmod 0444 " + path in env.commands
    assert result["provider_calls"] == 0


def worker(tmp_path, profile, selected, *, purpose="coding", mutation=None):
    root = tmp_path / "deployment"
    root.mkdir()
    (root / "container-boundary.json").write_text('{"provider":"grok_cli"}')
    if profile != "legacy":
        value = pruning.grok_pruning_declaration(profile)
        if mutation == "hash": value["config_sha256"] = "0" * 64
        (root / "grok-pruning.json").write_text(json.dumps(value))
    # Custody and filesystem protection are controlled here; the actual parser
    # and forwarding code execute. The file checker has its own real-FS tests.
    prelude = '''
import os,sys,types,json,pathlib
os.getuid=lambda:1001
os.geteuid=lambda:1001
for name in ('ipfs_accelerate_py','ipfs_accelerate_py.agent_supervisor',
             'ipfs_accelerate_py.agent_supervisor.runtime','ipfs_accelerate_py.agent_supervisor.todo_daemon'):
 module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
cli=types.ModuleType('ipfs_accelerate_py.cli_runtime');cli.__path__=[CLI_DIRECTORY]
sys.modules[cli.__name__]=cli
from ipfs_accelerate_py.cli_runtime import grok_pruning
grok_pruning.observe_deployed_grok_pruning=lambda profile: {}
old_lstat=pathlib.Path.lstat
def controlled_lstat(self,*args,**kwargs):
 if self.name=='grok-pruning.json':
  return types.SimpleNamespace(st_mode=0o100444,st_uid=0,st_size=self.stat().st_size)
 return old_lstat(self,*args,**kwargs)
pathlib.Path.lstat=controlled_lstat
boundary=types.ModuleType('ipfs_accelerate_py.agent_supervisor.runtime.container_worker_boundary')
boundary.verify_container_worker_boundary=lambda **kwargs:{'controlled_boundary':True}
sys.modules[boundary.__name__]=boundary
router=types.ModuleType('ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner')
router.MAX_IMPLEMENTATION_TIMEOUT_SECONDS=600
router.main=lambda:print(json.dumps({'argv':sys.argv[1:]})) or 0
sys.modules[router.__name__]=router
custody=types.ModuleType('ipfs_accelerate_py.agent_supervisor.todo_daemon.native_cli_subreaper')
custody.run_with_child_custody=lambda callback:callback()
sys.modules[custody.__name__]=custody
'''.replace("CLI_DIRECTORY", repr(str(Path(pruning.__file__).parent)))
    script = tmp_path / "worker.py"
    script.write_text(prelude + deployment.WORKER_ENTRY.replace(
        "pathlib.Path('/opt/ipfs-supervisor')", "pathlib.Path(" + repr(str(root)) + ")"))
    return subprocess.run([sys.executable, "-I", str(script), "--provider", "grok_cli",
        "--model", "grok-4.7", "--purpose", purpose, "--grok-pruning-profile", selected],
        cwd=tmp_path, input=b"", capture_output=True, timeout=20)


@pytest.mark.parametrize("profile", pruning.GROK_PRUNING_PROFILES)
def test_worker_forwards_only_exact_selected_profile(tmp_path, profile):
    result = worker(tmp_path, profile, profile)
    assert result.returncode == 0, result.stderr.decode()
    argv = json.loads(result.stdout)["argv"]
    if profile == "legacy":
        assert "--grok-pruning-profile" not in argv
    else:
        assert argv.count("--grok-pruning-profile") == 1
        assert argv[argv.index("--grok-pruning-profile") + 1] == profile


@pytest.mark.parametrize("deployed,selected,purpose,mutation", [
    ("legacy", "grok-pruning-on@1", "coding", None),
    ("grok-pruning-on@1", "legacy", "coding", None),
    ("grok-pruning-on@1", "grok-pruning-off@1", "coding", None),
    ("grok-pruning-on@1", "legacy", "planning", None),
    ("grok-pruning-on@1", "grok-pruning-on@1", "coding", "hash"),
])
def test_worker_rejects_missing_mismatched_or_unqualified_profile(tmp_path, deployed, selected, purpose, mutation):
    result = worker(tmp_path, deployed, selected, purpose=purpose, mutation=mutation)
    assert result.returncode != 0 and result.stdout == b""
    assert b"pruning" in result.stderr.lower()
