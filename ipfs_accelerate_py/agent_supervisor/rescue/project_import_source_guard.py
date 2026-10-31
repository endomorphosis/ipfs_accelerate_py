"""Authenticate exact original project Python source before body execution.

Canonical extraction of the accepted original guard; no programme mutation or
CLI entrypoint is carried into this reusable utility. This guard is not a sandbox.
"""
from __future__ import annotations
import hashlib,importlib,json,os,stat,sys
from pathlib import Path
from collections.abc import Mapping
MAX_BYTES=4*1024**2

def need(value,message):
    if not value:raise ValueError(message)

def plain(value):
    if isinstance(value,Mapping):return {str(k):plain(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v)for v in value]
    if value is None or type(value)in(str,int,float,bool):return value
    if callable(getattr(value,'to_dict',None)):return plain(value.to_dict())
    raise ValueError('non-JSON native projection')

def encoded(value):return(json.dumps(plain(value),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()

def digest(raw):return hashlib.sha256(raw).hexdigest()

def read_pin(row):
    need(type(row)is dict and set(row)=={'path','bytes','sha256'},'closed descriptor required')
    p=Path(row['path']);need(p.is_absolute()and p.resolve(strict=True)==p,'unaliased absolute input required')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        a=os.fstat(fd);need(stat.S_ISREG(a.st_mode)and a.st_uid==os.geteuid()and a.st_nlink==1 and 0<=a.st_size<=MAX_BYTES,'bounded owned input required')
        data=bytearray()
        while len(data)<=MAX_BYTES:
            chunk=os.read(fd,min(65536,MAX_BYTES+1-len(data)))
            if not chunk:break
            data.extend(chunk)
        b=os.fstat(fd);c=p.lstat();stamp=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
        need(stamp(a)==stamp(b)==stamp(c),'input changed during read')
    finally:os.close(fd)
    data=bytes(data);need(len(data)==row['bytes']and digest(data)==row['sha256'],'input commitment differs');return data

def pin(path):
    p=Path(path);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':digest(b)}

def write_new_or_exact(path,data):
    p=Path(path);need(p.is_absolute()and p.parent.resolve(strict=True)==p.parent,'exact existing output parent required')
    if os.path.lexists(p):
        row=pin(p);need(read_pin(row)==data,'retained output differs; reconcile without replay');return row
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return pin(p)

class GuardedProjectImports:
    """Authenticate normal project imports before executing each module body.

    This is project Python-source import enforcement, not an all-reads sandbox.
    Stdlib and third-party origins are bounded recorded metadata, separate from
    project source qualification. Canonical metadata subprocesses require their
    own original execution/source admission; this guard does not cross exec.
    """

    def __init__(self, plan, directory):
        from importlib.machinery import PathFinder
        import sysconfig

        self.path_finder = PathFinder
        self.directory = Path(directory)
        self.roots = {
            'accelerate': Path(plan['accelerate_root']).resolve(strict=True),
            'datasets': Path(plan['datasets_root']).resolve(strict=True),
            'original_datasets': Path(plan['publication_root']).resolve(strict=True),
        }
        self.pins = {str(Path(row['path']).resolve(strict=True)): row for row in plan['source_pins']}
        need(len(self.pins) == len(plan['source_pins']) <= 1024, 'closed unique project import pin population required')
        self.allowed_runtime_roots = {Path(p).resolve() for key, p in sysconfig.get_paths().items()
                                      if key in {'stdlib', 'platstdlib', 'purelib', 'platlib'} and p}
        # Venv fallback search includes the base interpreter's standard library.
        self.allowed_runtime_roots.add(Path(sys.base_prefix).resolve() / 'lib' / ('python' + str(sys.version_info.major) + '.' + str(sys.version_info.minor)))
        import site
        runtime_search_paths = {Path(p).resolve() for p in sys.path if p and Path(p).is_absolute()}
        for location in list(site.getsitepackages()) + [site.getusersitepackages()]:
            location = Path(location).resolve()
            if location in runtime_search_paths:
                self.allowed_runtime_roots.add(location)
        self.project_origins = {}
        self.runtime_origins = {}
        self.denials = []
        self.blocked = {'torch', 'transformers', 'tensorflow', 'jax', 'flax', 'sentence_transformers',
                        'openai', 'anthropic', 'litellm', 'grok_cli', 'huggingface_hub'}
        self.registered = False

    def _refuse(self, fullname, origin, reason):
        self.denials.append({'module': fullname, 'origin': str(origin), 'reason': reason})
        raise ImportError('source-bound project import refused: ' + fullname + ': ' + reason)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.blocked:
            self._refuse(fullname, '', 'model_or_provider_module_forbidden')
        spec = self.path_finder.find_spec(fullname, path, target)
        if spec is None:
            return None
        origin = spec.origin
        if origin in (None, 'built-in', 'frozen'):
            if fullname.startswith(('ipfs_accelerate_py', 'ipfs_datasets_py')):
                self._refuse(fullname, origin, 'project_namespace_without_exact_python_source')
            return spec
        source = Path(origin)
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError):
            self._refuse(fullname, origin, 'missing_import_origin')
        project = next((name for name, root in self.roots.items() if resolved.is_relative_to(root)), None)
        if fullname == 'ipfs_accelerate_py' or fullname.startswith('ipfs_accelerate_py.'):
            if not resolved.is_relative_to(self.roots['accelerate']):
                self._refuse(fullname, resolved, 'foreign_accelerate_package')
            project = 'accelerate'
        if fullname == 'ipfs_datasets_py' or fullname.startswith('ipfs_datasets_py.'):
            if not (resolved.is_relative_to(self.roots['datasets']) or resolved.is_relative_to(self.roots['original_datasets'])):
                self._refuse(fullname, resolved, 'foreign_datasets_package')
            if fullname == 'ipfs_datasets_py.logic.autoformal' or fullname.startswith('ipfs_datasets_py.logic.autoformal.'):
                if not resolved.is_relative_to(self.roots['datasets']):
                    self._refuse(fullname, resolved, 'foreign_canonical_handoff_package')
            project = 'datasets' if resolved.is_relative_to(self.roots['datasets']) else 'original_datasets'
        if project is not None:
            if resolved.suffix != '.py':
                self._refuse(fullname, resolved, 'project_origin_is_not_python_source')
            descriptor = self.pins.get(str(resolved))
            if descriptor is None:
                self._refuse(fullname, resolved, 'project_source_not_in_exact_plan_pins')
            # Execute the exact authenticated source bytes, rather than letting
            # SourceFileLoader choose an mtime-valid cached .pyc afterward.
            matched_source = read_pin(descriptor)
            original_loader = spec.loader
            guard = self
            class PinnedProjectSourceLoader:
                def get_resource_reader(self, resource_fullname):
                    # Preserve the normal filesystem package resource protocol.
                    # Python bodies still compile only authenticated bytes below.
                    from importlib.machinery import SourceFileLoader
                    need(resource_fullname == fullname
                         and type(original_loader) is SourceFileLoader
                         and original_loader.name == fullname
                         and original_loader.path == str(resolved)
                         and spec.origin == str(resolved)
                         and spec.loader is self,
                         'original package resource loader identity changed')
                    return original_loader.get_resource_reader(resource_fullname)

                def create_module(self, module_spec):
                    create = getattr(original_loader, 'create_module', None)
                    return create(module_spec) if create is not None else None

                def exec_module(self, module):
                    row = guard.project_origins[fullname]
                    row['execution_status'] = 'execution_started'
                    row.pop('execution_profile', None)
                    try:
                        current_source = read_pin(descriptor)
                        need(current_source == matched_source, 'project source changed before execution')
                        need(getattr(module.__spec__, 'origin', None) == str(resolved), 'project loader origin changed')
                        code = compile(current_source, str(resolved), 'exec', dont_inherit=True)
                        exec(code, module.__dict__)
                    except BaseException:
                        row['execution_status'] = 'execution_failed'
                        raise
                    row['execution_status'] = 'execution_completed'
                    row['execution_profile'] = 'compiled_exact_authenticated_source_bytes'

            spec.loader = PinnedProjectSourceLoader()
            previous = self.project_origins.get(fullname)
            if previous is not None:
                if any(previous.get(key) != value for key, value in
                       {'module': fullname, 'project': project, **descriptor}.items()):
                    self._refuse(fullname, resolved, 'project_resolution_changed')
            else:
                self.project_origins[fullname] = {'module': fullname, 'project': project, **descriptor,
                                                  'execution_status': 'resolution_only'}
            if len(self.project_origins) > 1024:
                self._refuse(fullname, resolved, 'project_import_count_exceeded')
        else:
            if not any(resolved.is_relative_to(root) for root in self.allowed_runtime_roots):
                self._refuse(fullname, resolved, 'runtime_origin_outside_interpreter_roots')
            info = resolved.stat()
            if info.st_size > 128 * 1024 * 1024 or len(self.runtime_origins) >= 4096:
                self._refuse(fullname, resolved, 'runtime_import_metadata_bound_exceeded')
            self.runtime_origins[fullname] = {'module': fullname, 'path': str(resolved),
                                             'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
                                             'qualification': 'runtime_origin_metadata_only'}
        return spec

    def install(self):
        need(not self.registered, 'project import guard already installed')
        need(not any(name == 'ipfs_accelerate_py' or name.startswith('ipfs_accelerate_py.')
                     or name == 'ipfs_datasets_py' or name.startswith('ipfs_datasets_py.') for name in sys.modules),
             'project modules were imported before the source guard')
        sys.meta_path.insert(0, self)
        # The existing Accelerate compatibility finder may insert itself first.
        # Each import audit event restores this guard before resolution; alias
        # loaders still import their canonical target through this same guard.
        def before_import(event, args):
            if event == 'import':
                name = args[0]
                if name.split('.')[0] in self.blocked:
                    self._refuse(name, '', 'model_or_provider_module_forbidden')
                if self.registered and (not sys.meta_path or sys.meta_path[0] is not self):
                    need(self in sys.meta_path, 'project import guard was removed')
                    sys.meta_path.remove(self)
                    sys.meta_path.insert(0, self)
        sys.addaudithook(before_import)
        self.registered = True
        return self

    def checkpoint(self, label):
        need(self.registered and self in sys.meta_path and not self.denials,
             'project import guard removed or a project import was denied')
        for name, row in self.project_origins.items():
            read_pin({k: row[k] for k in ('path', 'bytes', 'sha256')})
            module = sys.modules.get(name)
            if module is not None:
                need(Path(module.__file__).resolve(strict=True) == Path(row['path']), 'loaded project module origin changed')
        return write_new_or_exact(self.directory / ('project-import-origins-' + label + '.json'), encoded({
            'schema': 'legal-ir-pre-execution-project-import-origins/v1', 'label': label,
            'project_imports': sorted((row for row in self.project_origins.values()
                                      if row.get('execution_status') != 'resolution_only'), key=lambda row: row['module']),
            'project_resolutions': sorted((row for row in self.project_origins.values()
                                          if row.get('execution_status') == 'resolution_only'), key=lambda row: row['module']),
            'runtime_origin_metadata': sorted(self.runtime_origins.values(), key=lambda row: row['module']),
            'denials': self.denials, 'project_origins_checked_before_module_execution': True,
            'all_reads_enforced_or_attested': False, 'guard_propagated_to_exec_subprocess': False}))
