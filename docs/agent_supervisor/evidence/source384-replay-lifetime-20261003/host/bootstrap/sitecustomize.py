"""Off-tree whole-module selection for lifetime controls, not production hooks."""
from pathlib import Path
import importlib.abc
import importlib.util
import hashlib
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
GENERATION=os.environ['SOURCE384_LIFETIME_GENERATION']
assert GENERATION in ('before','proposed')
PINS=json.loads((ROOT/'source-pins.json').read_text())
TARGETS={
 'ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context':'A',
 'ipfs_datasets_py.logic.software_contracts.codebase_source_units_384':'D'}
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname not in TARGETS:return None
  key=TARGETS[fullname]; entry=PINS[key]; source=ROOT/GENERATION/key/entry['path']
  expected=entry['before_sha256' if GENERATION=='before' else 'after_sha256']
  if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ImportError('lifetime owner pin mismatch')
  return importlib.util.spec_from_file_location(fullname,source)
sys.meta_path.insert(0,Finder())
