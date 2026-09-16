"""Keep this overlay's ipfs_accelerate_py first when it is on PYTHONPATH.

Board extra-gate scripts insert a sealed checkout at sys.path[0]. Path pinning
alone is not enough to run overlay extra-gate: sealed launchers call APIs the
overlay server must accept. Extra-gate preload stays off until those APIs exist.
"""
