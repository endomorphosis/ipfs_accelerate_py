"""Observe final test roots without gaining or exercising signal authority."""
from pathlib import Path
import json
import sys

qualification_path = Path(sys.argv[1])
qualification = json.loads(qualification_path.read_text())
fixture_root = Path(next(value.split('=', 1)[1] for value in qualification['command']
                         if value.startswith('--basetemp=')))
worker_markers = {}
for path in fixture_root.rglob('worker.pid'):
    try:
        value = int(path.read_text().strip())
    except (OSError, ValueError):
        continue
    if value > 1:
        worker_markers[value] = str(path.relative_to(fixture_root))
live = []
unavailable = []
for path in Path('/proc').iterdir():
    if not path.name.isdigit():
        continue
    pid = int(path.name)
    try:
        raw_argv = (path / 'cmdline').read_bytes()
        matching_root = str(fixture_root).encode() in raw_argv
        try:
            observed_cwd = Path((path / 'cwd').readlink())
            matching_cwd = observed_cwd == fixture_root or observed_cwd.is_relative_to(fixture_root)
        except OSError:
            matching_cwd = False
        if not matching_root and not matching_cwd and pid not in worker_markers:
            continue
        raw_stat = (path / 'stat').read_text()
        fields = raw_stat[raw_stat.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            continue
        live.append({'pid': pid, 'state': fields[0], 'parent': int(fields[1]),
                     'group': int(fields[2]), 'session': int(fields[3]),
                     'start_ticks': fields[19], 'matching_fixture_root': matching_root, 'matching_fixture_cwd': matching_cwd,
                     'worker_marker': worker_markers.get(pid)})
    except FileNotFoundError:
        continue
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        if pid in worker_markers:
            unavailable.append({'pid': pid, 'error': type(error).__name__})
result = {'fixture_root': str(fixture_root), 'worker_marker_count': len(worker_markers),
          'matching_live_processes': live, 'unavailable_worker_observations': unavailable,
          'signal_calls': 0,
          'scope': 'Observed native argv or cwd under this exact run root, plus its recorded worker PIDs; zombies excluded. This is not a host-wide absence proof.'}
output = qualification_path.with_name(qualification_path.name.replace('.qualification.json', '.native-process-audit.json'))
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
raise SystemExit(1 if live or unavailable else 0)
