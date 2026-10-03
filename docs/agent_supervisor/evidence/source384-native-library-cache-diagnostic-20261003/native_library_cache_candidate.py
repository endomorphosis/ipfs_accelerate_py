"""Off-tree exact seven independently pinned installed native libraries."""
import json,signal,time
from codex_cache_candidate import advise_selected
ROOT='/opt/ipfs-supervisor'
ROWS=[{'path': 'home/.duckdb/extensions/v1.5.5/linux_arm64/ducklake.duckdb_extension', 'name': 'ducklake.duckdb_extension', 'role': 'installed_duckdb_extension', 'uid': 1000, 'mode': 420, 'expected_bytes': 32143862, 'sha256': 'd0b57c8e261b89a1ae367c7224f0857cfde72ab6cf2609f188e0de9b897b1088'}, {'path': 'home/.duckdb/extensions/v1.5.5/linux_arm64/httpfs.duckdb_extension', 'name': 'httpfs.duckdb_extension', 'role': 'installed_duckdb_extension', 'uid': 1000, 'mode': 420, 'expected_bytes': 19861758, 'sha256': 'eba6e263e395a83966090f1f11ade63630b1b21422f0f2813858d179d42ea1e9'}, {'path': 'home/.duckdb/extensions/v1.5.5/linux_arm64/quack.duckdb_extension', 'name': 'quack.duckdb_extension', 'role': 'installed_duckdb_extension', 'uid': 1000, 'mode': 420, 'expected_bytes': 29708110, 'sha256': '41b2b9292bfb860c5ca8c5f818f9dd7a2c6bc24f9c750cffbc3169286fe59f08'}, {'path': 'venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so', 'name': 'libtorch_cpu.so', 'role': 'installed_torch_library', 'uid': 0, 'mode': 493, 'expected_bytes': 262620152, 'sha256': '5750e23245c46c34120d6a11f8389bac3493d000ea93ebfbe129a94e61f2e784'}, {'path': 'venv/lib/python3.12/site-packages/torch/lib/libtorch_python.so', 'name': 'libtorch_python.so', 'role': 'installed_torch_library', 'uid': 0, 'mode': 493, 'expected_bytes': 28237672, 'sha256': '2634c578f87e437467f899c655240aaca2f7756c94388dd47af908bd24d0bc8b'}, {'path': 'venv/lib/python3.12/site-packages/torch/lib/libopenblas.so.0', 'name': 'libopenblas.so.0', 'role': 'installed_torch_library', 'uid': 0, 'mode': 493, 'expected_bytes': 24020337, 'sha256': '7a6446210edb096b85c713c872f0dfd3b3e0c963a77d2629dfaf83380c441746'}, {'path': 'venv/lib/python3.12/site-packages/torch/lib/libarm_compute.so', 'name': 'libarm_compute.so', 'role': 'installed_torch_library', 'uid': 0, 'mode': 493, 'expected_bytes': 17850377, 'sha256': 'cd83d6783a1acd1b9c4a6291b2567beb1c4a4ca98f82e3ccb46cd347b048c35d'}]
SOURCE_PINS_SHA256='abbb454825437057bbf33cdbd30aa5ff76fb1d1005724d602b7bc4c8bf8743c9'

def main():
    def expired(*args):raise TimeoutError('bounded native library advice expired')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        result=advise_selected(ROOT,ROWS,expected_count=7)
        result.update(schema='pinned-native-libraries-cache-advice@1',source_pins_sha256=SOURCE_PINS_SHA256,
            source_selection='three_archive_bound_extensions_plus_four_cpu_wheel_members')
        print(json.dumps(result,sort_keys=True,allow_nan=False))
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
