"""Compatibility exports for the shared, closed public task declarations.

Runtime Doctor verification and benchmark preparation use the same canonical
producer; generated profile and smoke bytes remain unchanged.
"""
from ipfs_accelerate_py.agent_supervisor.runtime.terminal_task_profile import (  # noqa: F401
    INSTRUCTION, MAX_INPUTS, MAX_OUTPUT_BYTES, MAX_TOTAL_OUTPUT_BYTES, PROFILE,
    SCHEMA, DATA_SCHEMA, DATA_MEDIA_SUFFIXES, MULTITASK_SCHEMA, SMOKE, _path, instruction_sha256, normalized_instruction,
    task_profile_bytes, task_profile_index_paths, task_profile_smoke,
    task_profile_spec, task_profile_specs, task_profile_worker_inputs, validate_task_profile,
    validate_task_profile_contract, validate_task_data,
)
