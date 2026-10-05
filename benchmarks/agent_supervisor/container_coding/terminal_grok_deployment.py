"""Explicit native Grok asset and credential transport for isolated trials."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import tarfile

GROK_VERSION = "1.0.46"
GROK_VERSION_OUTPUT = "grok 1.0.46 (2765805b9442) [stable]"
GROK_PROFILE = "grok-4.7-cli-1.0.46@1"
GROK_PATH = "providers/grok/grok"
GROK_SHA256 = "45b0943e736f00a249b9cf02af2be9e0749d97c09a6f55cfcf3029a1a836f23e"
GROK_BYTES = 142867512


def grok_binding() -> dict:
    return {"path": GROK_PATH, "version": GROK_VERSION, "version_output": GROK_VERSION_OUTPUT,
            "sha256": GROK_SHA256, "bytes": GROK_BYTES, "platform": "linux-aarch64"}


def validate_grok_binding(manifest: dict, *, required: bool = False) -> dict | None:
    binding = manifest.get("grok_cli_assets")
    if binding is None and not required:
        return None
    if type(binding) is not dict or binding != grok_binding():
        raise ValueError("exact pinned Grok CLI asset binding required")
    inventory = manifest.get("files")
    if type(inventory) is not list or any(type(row) is not dict for row in inventory):
        raise ValueError("bounded Grok runtime inventory required")
    rows = [row for row in inventory if row.get("path") == GROK_PATH]
    if len(rows) != 1 or rows[0] != {
            "path": GROK_PATH, "bytes": GROK_BYTES, "sha256": GROK_SHA256, "mode": 0o755}:
        raise ValueError("Grok runtime inventory differs from its independent pin")
    return dict(binding)


def validate_grok_binary(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != GROK_BYTES:
        raise ValueError("pinned regular Grok native binary required")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != GROK_SHA256:
            raise ValueError("Grok native binary SHA-256 differs")
    if not os.access(path, os.X_OK):
        raise ValueError("executable Grok native binary required")
    return path


def validate_auth_file(path: Path) -> Path:
    path = Path(path)
    # Never return or retain credential contents or a credential fingerprint.
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 65536:
        raise ValueError("one bounded regular auth.json file required")
    return path


def verify_grok_archive(archive: Path, manifest: dict, *, required: bool = False) -> dict | None:
    binding = validate_grok_binding(manifest, required=required)
    if binding is None:
        return None
    with tarfile.open(archive) as bundle:
        members = [member for member in bundle.getmembers() if member.name == GROK_PATH]
        if len(members) != 1:
            raise ValueError("one pinned Grok archive member required")
        member = members[0]
        if not member.isfile() or member.size != GROK_BYTES or member.mode != 0o755:
            raise ValueError("Grok archive member type, size or mode differs")
        with bundle.extractfile(member) as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != GROK_SHA256:
                raise ValueError("Grok archive binary SHA-256 differs")
    return binding


def grok_exposure_script(root: str = "/opt/ipfs-supervisor") -> str:
    """Recheck the actual container binary before exposing it to either UID."""
    return f'''import hashlib,json,os,pathlib,platform,shutil,subprocess
root=pathlib.Path({root!r});source=root/{GROK_PATH!r}
if platform.system()!='Linux' or platform.machine()!='aarch64':raise SystemExit('pinned Grok platform differs')
if source.is_symlink() or not source.is_file() or source.stat().st_size!={GROK_BYTES}:raise SystemExit('pinned Grok file differs')
with source.open('rb') as stream:
 if hashlib.file_digest(stream,'sha256').hexdigest()!={GROK_SHA256!r}:raise SystemExit('pinned Grok SHA differs')
target=root/'provider-bin/grok';target.parent.mkdir(mode=0o755,exist_ok=True)
if target.exists() or target.is_symlink():raise SystemExit('Grok exposure already exists')
shutil.copyfile(source,target);target.chmod(0o555);os.chown(target,0,0)
result=subprocess.run([str(target),'--version'],capture_output=True,text=True,timeout=15,env={{'HOME':str(root/'home'),'PATH':'/usr/bin:/bin'}})
if result.returncode or result.stdout.strip()!={GROK_VERSION_OUTPUT!r}:raise SystemExit('native Grok version differs')
print(json.dumps({{'schema':'native-grok-runtime-bundle@1','version':{GROK_VERSION!r},'sha256':{GROK_SHA256!r},'bytes':{GROK_BYTES},'provider_calls':0}}))
'''
