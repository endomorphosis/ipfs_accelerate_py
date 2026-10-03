"""Exactly 131 public installed payloads bound to the selected archive and wheel."""
import json,signal,time
if globals().get('__package__'):
    from .terminal_setup_cache_codex import advise_selected
else:
    from terminal_setup_cache_codex import advise_selected
ROOT='/opt/ipfs-supervisor'
ROWS=[
  {
    "expected_bytes": 32143862,
    "mode": 420,
    "name": "ducklake.duckdb_extension",
    "path": "home/.duckdb/extensions/v1.5.5/linux_arm64/ducklake.duckdb_extension",
    "role": "installed_duckdb_extension",
    "sha256": "d0b57c8e261b89a1ae367c7224f0857cfde72ab6cf2609f188e0de9b897b1088",
    "uid": 1000
  },
  {
    "expected_bytes": 19861758,
    "mode": 420,
    "name": "httpfs.duckdb_extension",
    "path": "home/.duckdb/extensions/v1.5.5/linux_arm64/httpfs.duckdb_extension",
    "role": "installed_duckdb_extension",
    "sha256": "eba6e263e395a83966090f1f11ade63630b1b21422f0f2813858d179d42ea1e9",
    "uid": 1000
  },
  {
    "expected_bytes": 29708110,
    "mode": 420,
    "name": "quack.duckdb_extension",
    "path": "home/.duckdb/extensions/v1.5.5/linux_arm64/quack.duckdb_extension",
    "role": "installed_duckdb_extension",
    "sha256": "41b2b9292bfb860c5ca8c5f818f9dd7a2c6bc24f9c750cffbc3169286fe59f08",
    "uid": 1000
  },
  {
    "expected_bytes": 262620152,
    "mode": 493,
    "name": "libtorch_cpu.so",
    "path": "venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so",
    "role": "installed_torch_wheel_payload",
    "sha256": "5750e23245c46c34120d6a11f8389bac3493d000ea93ebfbe129a94e61f2e784",
    "uid": 0
  },
  {
    "expected_bytes": 28237672,
    "mode": 493,
    "name": "libtorch_python.so",
    "path": "venv/lib/python3.12/site-packages/torch/lib/libtorch_python.so",
    "role": "installed_torch_wheel_payload",
    "sha256": "2634c578f87e437467f899c655240aaca2f7756c94388dd47af908bd24d0bc8b",
    "uid": 0
  },
  {
    "expected_bytes": 24020337,
    "mode": 493,
    "name": "libopenblas.so.0",
    "path": "venv/lib/python3.12/site-packages/torch/lib/libopenblas.so.0",
    "role": "installed_torch_wheel_payload",
    "sha256": "7a6446210edb096b85c713c872f0dfd3b3e0c963a77d2629dfaf83380c441746",
    "uid": 0
  },
  {
    "expected_bytes": 17850377,
    "mode": 493,
    "name": "libarm_compute.so",
    "path": "venv/lib/python3.12/site-packages/torch/lib/libarm_compute.so",
    "role": "installed_torch_wheel_payload",
    "sha256": "cd83d6783a1acd1b9c4a6291b2567beb1c4a4ca98f82e3ccb46cd347b048c35d",
    "uid": 0
  },
  {
    "expected_bytes": 13729768,
    "mode": 493,
    "name": "test_api",
    "path": "venv/lib/python3.12/site-packages/torch/bin/test_api",
    "role": "installed_torch_wheel_payload",
    "sha256": "c1b984e86af6eb145fbfd8f3e019e4569a8d07de2e03582f68dc60b447f84d6b",
    "uid": 0
  },
  {
    "expected_bytes": 12748704,
    "mode": 493,
    "name": "test_jit",
    "path": "venv/lib/python3.12/site-packages/torch/bin/test_jit",
    "role": "installed_torch_wheel_payload",
    "sha256": "af3745862bbef73c646008b242fb1253378240a39acffc3dea7174e11b25d696",
    "uid": 0
  },
  {
    "expected_bytes": 4928560,
    "mode": 493,
    "name": "test_lazy",
    "path": "venv/lib/python3.12/site-packages/torch/bin/test_lazy",
    "role": "installed_torch_wheel_payload",
    "sha256": "e4ab2bf5469e3281a222f38d5d8efafeddee7586fbc9e40990f8f482ceea17b2",
    "uid": 0
  },
  {
    "expected_bytes": 4783752,
    "mode": 493,
    "name": "protoc",
    "path": "venv/lib/python3.12/site-packages/torch/bin/protoc",
    "role": "installed_torch_wheel_payload",
    "sha256": "148926f95652cb5eaf67be9e19403b0a8f010afecae17aebea80c1f3b5881144",
    "uid": 0
  },
  {
    "expected_bytes": 4783752,
    "mode": 493,
    "name": "protoc-3.13.0.0",
    "path": "venv/lib/python3.12/site-packages/torch/bin/protoc-3.13.0.0",
    "role": "installed_torch_wheel_payload",
    "sha256": "148926f95652cb5eaf67be9e19403b0a8f010afecae17aebea80c1f3b5881144",
    "uid": 0
  },
  {
    "expected_bytes": 4364680,
    "mode": 493,
    "name": "op_registration_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/op_registration_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "34b479502c1b704cbc55cd4446b0e28459ded4ff9722150ef2d48e06a60b4b78",
    "uid": 0
  },
  {
    "expected_bytes": 2324304,
    "mode": 493,
    "name": "c10_intrusive_ptr_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_intrusive_ptr_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "b4cff2abb814b5f2c9057ca221ab1bbd198b52e6932f21e3198c7565fd88f78b",
    "uid": 0
  },
  {
    "expected_bytes": 2195793,
    "mode": 420,
    "name": "RedispatchFunctions.h",
    "path": "venv/lib/python3.12/site-packages/torch/include/ATen/RedispatchFunctions.h",
    "role": "installed_torch_wheel_payload",
    "sha256": "512efc63cb2e6b2f9b69c45de2bef8bf26abff428d4dd7fd0d65d22dea122f96",
    "uid": 0
  },
  {
    "expected_bytes": 2103320,
    "mode": 493,
    "name": "c10_small_vector_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_small_vector_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "51df0571e77e7eccbe8038050e35f904776a285c5037564018a2245f11a08bd1",
    "uid": 0
  },
  {
    "expected_bytes": 1796696,
    "mode": 493,
    "name": "libc10.so",
    "path": "venv/lib/python3.12/site-packages/torch/lib/libc10.so",
    "role": "installed_torch_wheel_payload",
    "sha256": "322122d803708b7078b7c754e4b044e6fcf9b607688ccc9be576d0a85274a8bf",
    "uid": 0
  },
  {
    "expected_bytes": 1732642,
    "mode": 420,
    "name": "VmapGeneratedPlumbing.h",
    "path": "venv/lib/python3.12/site-packages/torch/include/ATen/VmapGeneratedPlumbing.h",
    "role": "installed_torch_wheel_payload",
    "sha256": "453f463dc99ca3807e0e08422ccb7dbfc691fe5e016bbcdb86e0de8bf2fc1370",
    "uid": 0
  },
  {
    "expected_bytes": 1712376,
    "mode": 493,
    "name": "kernel_lambda_legacy_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/kernel_lambda_legacy_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "203f34112c6c621576d855ddb73988b6252b43bdb00b587dcdf71f4843f45e9b",
    "uid": 0
  },
  {
    "expected_bytes": 1655209,
    "mode": 493,
    "name": "libgomp.so.1",
    "path": "venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1",
    "role": "installed_torch_wheel_payload",
    "sha256": "7e2abc1b84b1470fa53a0b679e066c11adbc3a8d42aebf37687f6b1254421391",
    "uid": 0
  },
  {
    "expected_bytes": 1563592,
    "mode": 493,
    "name": "kernel_function_legacy_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/kernel_function_legacy_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "077080638be754dd634fb452dbf2b2ca1c6d7fe78a9578cf9fca2060d24f602a",
    "uid": 0
  },
  {
    "expected_bytes": 1553120,
    "mode": 493,
    "name": "List_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/List_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "22a950bf44751d6085f971043f5b545b85990d52169b175e6a0ec4fd7cbd3057",
    "uid": 0
  },
  {
    "expected_bytes": 1485025,
    "mode": 493,
    "name": "libgfortran.so.5",
    "path": "venv/lib/python3.12/site-packages/torch/lib/libgfortran.so.5",
    "role": "installed_torch_wheel_payload",
    "sha256": "36f49edf8de93b5ad7e3324598b921f0229e0841eea3902a4d6fc9f949c77e94",
    "uid": 0
  },
  {
    "expected_bytes": 1396680,
    "mode": 493,
    "name": "kernel_lambda_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/kernel_lambda_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "6bdc3ce9156762385a80baae94b747f3d0ea6639a76e7aba86731539717978c9",
    "uid": 0
  },
  {
    "expected_bytes": 1376254,
    "mode": 420,
    "name": "common_methods_invocations.py",
    "path": "venv/lib/python3.12/site-packages/torch/testing/_internal/common_methods_invocations.py",
    "role": "installed_torch_wheel_payload",
    "sha256": "f195caae520c73e3744af853d48c6534589e867a5378f03e5df6ee700bac6c23",
    "uid": 0
  },
  {
    "expected_bytes": 1352632,
    "mode": 493,
    "name": "test_aoti_abi_check",
    "path": "venv/lib/python3.12/site-packages/torch/bin/test_aoti_abi_check",
    "role": "installed_torch_wheel_payload",
    "sha256": "853d6819c5e1bb889d8d0b50fe21264b50a482fe2bf25d464ca6e2a8407076c7",
    "uid": 0
  },
  {
    "expected_bytes": 1310273,
    "mode": 493,
    "name": "libarm_compute_graph.so",
    "path": "venv/lib/python3.12/site-packages/torch/lib/libarm_compute_graph.so",
    "role": "installed_torch_wheel_payload",
    "sha256": "c13e3b7d7e9fba243b3f6e7c3df9e3ad62a9703ad1763a8f8402d99c04d980b0",
    "uid": 0
  },
  {
    "expected_bytes": 1236600,
    "mode": 493,
    "name": "make_boxed_from_unboxed_functor_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/make_boxed_from_unboxed_functor_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "ee5b254fd4e5fd41b6aecd65bc29f1851af2c7fd06f508bbf41e18bd9a8f0892",
    "uid": 0
  },
  {
    "expected_bytes": 1227864,
    "mode": 493,
    "name": "ivalue_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/ivalue_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "6dc3c79ae220cc8f200d33f259f9921d728005b3fcc03e3e469a3baec314d416",
    "uid": 0
  },
  {
    "expected_bytes": 1207952,
    "mode": 493,
    "name": "kernel_function_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/kernel_function_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "d27174b7598e3b783f0444c298817bb7bf5596350b50a53515073119b1b1011f",
    "uid": 0
  },
  {
    "expected_bytes": 1172128,
    "mode": 493,
    "name": "cpu_rng_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/cpu_rng_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "fd4aafc3faed39ff16470e75afecc1e0ee206a8ccff09e8515a0952f86415ef1",
    "uid": 0
  },
  {
    "expected_bytes": 1170736,
    "mode": 493,
    "name": "tensor_iterator_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/tensor_iterator_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "25343b8e84327f775c7ee03add9a9ad81cd57fe410b781369969f5a6f514e647",
    "uid": 0
  },
  {
    "expected_bytes": 1099761,
    "mode": 420,
    "name": "_VariableFunctions.pyi",
    "path": "venv/lib/python3.12/site-packages/torch/_C/_VariableFunctions.pyi",
    "role": "installed_torch_wheel_payload",
    "sha256": "c2ed2e8b08879452ca2dd1cf7571afd213f5cd1bcee4d398beb29e6980ad578c",
    "uid": 0
  },
  {
    "expected_bytes": 1099761,
    "mode": 420,
    "name": "_VF.pyi",
    "path": "venv/lib/python3.12/site-packages/torch/_VF.pyi",
    "role": "installed_torch_wheel_payload",
    "sha256": "c2ed2e8b08879452ca2dd1cf7571afd213f5cd1bcee4d398beb29e6980ad578c",
    "uid": 0
  },
  {
    "expected_bytes": 1072392,
    "mode": 493,
    "name": "Dict_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/Dict_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "b9f19b143bbdc901fc7249061308d64f87ec50098b3997962ca8de908eeadad1",
    "uid": 0
  },
  {
    "expected_bytes": 1018440,
    "mode": 493,
    "name": "test_profiler_collection",
    "path": "venv/lib/python3.12/site-packages/torch/bin/test_profiler_collection",
    "role": "installed_torch_wheel_payload",
    "sha256": "33fa7290bc696a39055b3ccf53a749caa0ef3c3e9145d3313dca0f727ddc87b1",
    "uid": 0
  },
  {
    "expected_bytes": 944264,
    "mode": 493,
    "name": "legacy_vmap_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/legacy_vmap_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "620e52119ea7f60b07d86b4caa1b2705f8dff21b5f516e76204e5d0cc9425b4e",
    "uid": 0
  },
  {
    "expected_bytes": 922808,
    "mode": 493,
    "name": "KernelFunction_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/KernelFunction_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "18c349c0b2da30b772cc6839ccfbfa19a48fd0823a7ef4b239d23e4b5b678c30",
    "uid": 0
  },
  {
    "expected_bytes": 912992,
    "mode": 493,
    "name": "c10_optional_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_optional_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "613b748b7e9815bbd9f448f735823f045f3c0415effc484e7a57d2923c24f0c2",
    "uid": 0
  },
  {
    "expected_bytes": 912720,
    "mode": 493,
    "name": "libtorchbind_test.so",
    "path": "venv/lib/python3.12/site-packages/torch/lib/libtorchbind_test.so",
    "role": "installed_torch_wheel_payload",
    "sha256": "f9f50bdbd43596954258d75e9eb30e6c6d97e23a4b83ef2c4018ff50e4dc76fa",
    "uid": 0
  },
  {
    "expected_bytes": 903819,
    "mode": 420,
    "name": "RegistrationDeclarations.h",
    "path": "venv/lib/python3.12/site-packages/torch/include/ATen/RegistrationDeclarations.h",
    "role": "installed_torch_wheel_payload",
    "sha256": "63bf47f95b482eb912722bb3ec2f7e4c0b12e9958796debbdfd57250555f898f",
    "uid": 0
  },
  {
    "expected_bytes": 902704,
    "mode": 493,
    "name": "c10_cow_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_cow_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "579bc64fa4bfb1428f38fc33bf68b9cbe72096e085c10922485c51886a8fdd35",
    "uid": 0
  },
  {
    "expected_bytes": 890912,
    "mode": 493,
    "name": "inline_container_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/inline_container_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "f1e4944c748e1dc6c57742af04cfec16668669e39e757de4a264bc2e34c98324",
    "uid": 0
  },
  {
    "expected_bytes": 865352,
    "mode": 493,
    "name": "pow_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/pow_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "09621af0de18ad108331f50492113814980c9ccf693777249f973734fe62528f",
    "uid": 0
  },
  {
    "expected_bytes": 857912,
    "mode": 493,
    "name": "basic",
    "path": "venv/lib/python3.12/site-packages/torch/test/basic",
    "role": "installed_torch_wheel_payload",
    "sha256": "ffd412e158be8975785a04fd987d24c84d71dd8201d3b4e1183954bb2238dc77",
    "uid": 0
  },
  {
    "expected_bytes": 855536,
    "mode": 493,
    "name": "MaybeOwned_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/MaybeOwned_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "27ded6f75172c38c1bc98fd9129cd37ff1f3ecc203b998f6b16281ee97dccaf1",
    "uid": 0
  },
  {
    "expected_bytes": 807032,
    "mode": 493,
    "name": "test_cpp_rpc",
    "path": "venv/lib/python3.12/site-packages/torch/bin/test_cpp_rpc",
    "role": "installed_torch_wheel_payload",
    "sha256": "88ea46b70292f70a9286c40bd7155f972736c50b5a6a81baa05e6123d3b76b86",
    "uid": 0
  },
  {
    "expected_bytes": 806560,
    "mode": 493,
    "name": "kernel_stackbased_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/kernel_stackbased_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "c4e13e4b8aeacaa65188449c2fa2abc07cade564c59575fe2660258502d68f31",
    "uid": 0
  },
  {
    "expected_bytes": 805880,
    "mode": 493,
    "name": "IListRef_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/IListRef_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "7a7718f3ecf0337d939670409323bc6acfbeff6319b0ed37c3840c10a043d89f",
    "uid": 0
  },
  {
    "expected_bytes": 801144,
    "mode": 493,
    "name": "c10_ordered_preserving_dict_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_ordered_preserving_dict_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "6c357bf135fe768afc7094d46b6f96d0288d98df355eaff707e3b5f11d25204d",
    "uid": 0
  },
  {
    "expected_bytes": 798200,
    "mode": 493,
    "name": "type_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/type_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "b7dba5cc38feb94f143b22abb8db24a5ca009c8b1a0b7dd4bdabc11568d725e1",
    "uid": 0
  },
  {
    "expected_bytes": 797344,
    "mode": 493,
    "name": "atest",
    "path": "venv/lib/python3.12/site-packages/torch/test/atest",
    "role": "installed_torch_wheel_payload",
    "sha256": "68d8b3c3d21901649fd9558903677446455dca1b33dec9cad6560d10ed522e2e",
    "uid": 0
  },
  {
    "expected_bytes": 795840,
    "mode": 493,
    "name": "extension_backend_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/extension_backend_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "75514f932a4079ad0013973def73d9b57ddb7d42c681b9c1e3cc5867c74f95d5",
    "uid": 0
  },
  {
    "expected_bytes": 795032,
    "mode": 493,
    "name": "cpu_generator_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/cpu_generator_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "acb9fa192245137627be209b2377f2cdd1dc9f70d92a109ecd9c5c39d38110e2",
    "uid": 0
  },
  {
    "expected_bytes": 787520,
    "mode": 493,
    "name": "quantized_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/quantized_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "177cf3e2389bf297536c1ab16199e8d1cf42c5e0be41016b4aad7512b0981cb6",
    "uid": 0
  },
  {
    "expected_bytes": 782856,
    "mode": 493,
    "name": "c10_ThreadLocal_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_ThreadLocal_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "da244d80172c04c6447f36d5fe28765a9fc952d2fe0ee0ef9e13cabfd7001d3c",
    "uid": 0
  },
  {
    "expected_bytes": 781088,
    "mode": 493,
    "name": "test_parallel",
    "path": "venv/lib/python3.12/site-packages/torch/test/test_parallel",
    "role": "installed_torch_wheel_payload",
    "sha256": "4b9fcbea75e688bb55d41084a1cf09972977cb2f351d6b38a8aa8ab0bfe807ca",
    "uid": 0
  },
  {
    "expected_bytes": 780552,
    "mode": 493,
    "name": "apply_utils_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/apply_utils_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "9b8520f3a2b11bae87b631a506c2aee2d7e92a7fdd11668d60c6f369a95f22a2",
    "uid": 0
  },
  {
    "expected_bytes": 779576,
    "mode": 493,
    "name": "c10_WaitCounter_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_WaitCounter_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "b15a6219d36a88d488cbc465cb6c239789b6359db933e12e88d2d551102f8680",
    "uid": 0
  },
  {
    "expected_bytes": 778312,
    "mode": 493,
    "name": "backend_fallback_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/backend_fallback_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "6167c420b29432ec560ac065ec066a25ac7fa30848d2cacdb28d66c7580e7bcc",
    "uid": 0
  },
  {
    "expected_bytes": 777944,
    "mode": 493,
    "name": "c10_DispatchKeySet_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_DispatchKeySet_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "476dd04eb71a83ab9506ad2b48b47d2e8af65c322fbb4cdc4009db1c2abd7b96",
    "uid": 0
  },
  {
    "expected_bytes": 777888,
    "mode": 493,
    "name": "scalar_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/scalar_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "c056a9e4b67867b0b5d345a2e2ff8202f9bc003f192c0221b84c74f172428e17",
    "uid": 0
  },
  {
    "expected_bytes": 777552,
    "mode": 493,
    "name": "TCPStoreTest",
    "path": "venv/lib/python3.12/site-packages/torch/bin/TCPStoreTest",
    "role": "installed_torch_wheel_payload",
    "sha256": "45e8349796950711ec76d9eaa07ab102e6a68da0d07b560f6e80b3e9de2e6d01",
    "uid": 0
  },
  {
    "expected_bytes": 776880,
    "mode": 493,
    "name": "scalar_tensor_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/scalar_tensor_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "f4a1ec4a4da649ebfd8b960283c30448efe9ed0be0430d2ac1ffb164b817b3f4",
    "uid": 0
  },
  {
    "expected_bytes": 776856,
    "mode": 493,
    "name": "c10_SymInt_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_SymInt_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "194d0bc3a1ec0a63c0a1ec8e3ee2e1f4423026dd7b303b2629a5b62844bbcf5a",
    "uid": 0
  },
  {
    "expected_bytes": 775136,
    "mode": 493,
    "name": "c10_logging_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_logging_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "c0452173af73c120ec070302b5cc4ce372096fc43b26efcc1a34c2e689494ac4",
    "uid": 0
  },
  {
    "expected_bytes": 774488,
    "mode": 493,
    "name": "math_kernel_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/math_kernel_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "3315e3ccdf93ebecbd547a187e3b951c5d65c7b7c0989dbdbaaf2896628e6d21",
    "uid": 0
  },
  {
    "expected_bytes": 772800,
    "mode": 493,
    "name": "c10_bfloat16_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_bfloat16_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "3be2fc2d3b32a9fb64c246a9742072b32836605193b09757b5e8bf2b1ef61355",
    "uid": 0
  },
  {
    "expected_bytes": 772480,
    "mode": 493,
    "name": "memory_overlapping_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/memory_overlapping_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "467a62fa0bb5e6570e2fe534092f760c0be690c36f541d3d0941f523e5d0d894",
    "uid": 0
  },
  {
    "expected_bytes": 772368,
    "mode": 493,
    "name": "half_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/half_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "038aaad87dadc896983cd966352008efd9db80da17a97b91e7957ade87e103fb",
    "uid": 0
  },
  {
    "expected_bytes": 772280,
    "mode": 493,
    "name": "c10_Enumerate_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_Enumerate_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "11e664870049b574b4f188c6e98c759e12f17b40c3166b43006278fe0c81aab8",
    "uid": 0
  },
  {
    "expected_bytes": 772216,
    "mode": 493,
    "name": "native_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/native_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "75a3c5957a7cb8e1cf57d7818a2191d390193dc4f79301c9f8f1533eb0325ac8",
    "uid": 0
  },
  {
    "expected_bytes": 771816,
    "mode": 493,
    "name": "CppSignature_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/CppSignature_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "95d31568092d5e15d6ad953f59f16b6911814987891dea8dfd08e897f1a10f16",
    "uid": 0
  },
  {
    "expected_bytes": 770984,
    "mode": 493,
    "name": "type_ptr_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/type_ptr_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "6e2ba953fbb21f10ca24f8a621dd53ed62ce5832852eb535b871e981f402fc20",
    "uid": 0
  },
  {
    "expected_bytes": 770912,
    "mode": 493,
    "name": "stride_properties_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/stride_properties_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "011844096e4342bd334d32814d4f31dc151b7c348519ea07a9f6c073ee7932be",
    "uid": 0
  },
  {
    "expected_bytes": 770824,
    "mode": 493,
    "name": "accelerator_graph_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/accelerator_graph_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "52c094b6631b5390e59e05ce0878563fcc25dc3a4acbf8ff9a4321692a845426",
    "uid": 0
  },
  {
    "expected_bytes": 770176,
    "mode": 493,
    "name": "cpu_profiling_allocator_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/cpu_profiling_allocator_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "530c200785320d3072f73b4f83c2c75456593aa2c6ec826bcece7116c9c7ac67",
    "uid": 0
  },
  {
    "expected_bytes": 769408,
    "mode": 493,
    "name": "memory_format_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/memory_format_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "8e88794276de37497bab24592e42d5f935a71cf33bab5a5b0c0d533a488b17a0",
    "uid": 0
  },
  {
    "expected_bytes": 769376,
    "mode": 493,
    "name": "dlconvertor_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/dlconvertor_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "285b43cef5bb55dd572b35a59a3c661f8c17983def5af1379067d55fcac8344c",
    "uid": 0
  },
  {
    "expected_bytes": 769120,
    "mode": 493,
    "name": "c10_string_util_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_string_util_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "bcf80c86bb77e5eee9c9fc2fb90ea219f74d88c8cb0961242a7ec83b76311f5e",
    "uid": 0
  },
  {
    "expected_bytes": 767744,
    "mode": 493,
    "name": "xla_tensor_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/xla_tensor_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "17a1a0e3587b909378d14fd5d8844cfc95a00c074e204aa2af41efab8ba7f1ea",
    "uid": 0
  },
  {
    "expected_bytes": 767392,
    "mode": 493,
    "name": "broadcast_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/broadcast_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "98e724e9a8ce9f4b3f930906ff04c093b2683d11e74fa180385f0252afae7d0f",
    "uid": 0
  },
  {
    "expected_bytes": 767248,
    "mode": 493,
    "name": "undefined_tensor_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/undefined_tensor_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "fab173ece4ee77e3b9ab6f5c4dd16398585d2545a58d36faab3f0be9cff2facb",
    "uid": 0
  },
  {
    "expected_bytes": 767176,
    "mode": 493,
    "name": "c10_complex_math_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_complex_math_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "7ac1c73f8078212031504e925e24914e2237ad060be273cb004753b678f1f052",
    "uid": 0
  },
  {
    "expected_bytes": 766664,
    "mode": 493,
    "name": "weakref_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/weakref_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "c3de570409e86ff28642e58370c78cca0541b0bcaa1a9ec6321729537f379b9d",
    "uid": 0
  },
  {
    "expected_bytes": 766416,
    "mode": 493,
    "name": "packedtensoraccessor_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/packedtensoraccessor_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "190831ddfbdac2eb264d51ea6f8ee8d10e258a184130cfaf38e9f0e9f612c977",
    "uid": 0
  },
  {
    "expected_bytes": 765448,
    "mode": 493,
    "name": "test_comms_id",
    "path": "venv/lib/python3.12/site-packages/torch/bin/test_comms_id",
    "role": "installed_torch_wheel_payload",
    "sha256": "5889e29cf7e166fa0ed76a66a3f8aaea0fc32a6547866ccc1596e9d58261286e",
    "uid": 0
  },
  {
    "expected_bytes": 764688,
    "mode": 493,
    "name": "wrapdim_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/wrapdim_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "8ccbd5f5eba7c56e240cbd96401f1d79eb0dd4fb940af9bc1770ed054c472571",
    "uid": 0
  },
  {
    "expected_bytes": 764608,
    "mode": 493,
    "name": "c10_InlineStreamGuard_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_InlineStreamGuard_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "10d3668ef8265649eaa3128cebcb53ecf205bae3bd65dfffcc3c2cfc599b5ca0",
    "uid": 0
  },
  {
    "expected_bytes": 764448,
    "mode": 493,
    "name": "thread_init_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/thread_init_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "a28961f5be4fc05134c4f4f7642379352eec6a44a3554cfdeab30efa0590578d",
    "uid": 0
  },
  {
    "expected_bytes": 764344,
    "mode": 493,
    "name": "StorageUtils_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/StorageUtils_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "be3e9c403507d84ebe6441acd6f8ad690282e500eb2c703c0462eea0e835b80a",
    "uid": 0
  },
  {
    "expected_bytes": 764264,
    "mode": 493,
    "name": "reportMemoryUsage_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/reportMemoryUsage_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "343f077563228dbbb11a9396e550c92ec732a5568aa436c6e9c97c5e4131568e",
    "uid": 0
  },
  {
    "expected_bytes": 764080,
    "mode": 493,
    "name": "operator_name_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/operator_name_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "2f7b9b605e84c7007d1201967febc6f5a7a14dd1ce3dd6a8b3eede098d41dd76",
    "uid": 0
  },
  {
    "expected_bytes": 759016,
    "mode": 493,
    "name": "c10_typeid_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_typeid_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "3eb38cd4d799f4330975e3466eca790435ce0cf3cb62eae17c67a5c6afa91f35",
    "uid": 0
  },
  {
    "expected_bytes": 755776,
    "mode": 493,
    "name": "c10_complex_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_complex_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "f37be8bf3ed74bcc3c7b603dec3cd0bb153b84b19127df1c021ea65d43b0a90a",
    "uid": 0
  },
  {
    "expected_bytes": 752504,
    "mode": 493,
    "name": "c10_Scalar_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_Scalar_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "91fbfc7d6cad4eb07064d0c9225ef787e6851d5532f2228769b4a589fde954da",
    "uid": 0
  },
  {
    "expected_bytes": 751120,
    "mode": 493,
    "name": "c10_AllocatorConfig_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_AllocatorConfig_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "615c1b14cce0807d6dc12f146348221a61fdd60744f9da8ed7f125839261ed47",
    "uid": 0
  },
  {
    "expected_bytes": 716440,
    "mode": 493,
    "name": "c10_LeftRight_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_LeftRight_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "ee06e0ab0150794b4a90039ae883c6cf3171aaec29c543f45003cb8beb76ec1d",
    "uid": 0
  },
  {
    "expected_bytes": 710040,
    "mode": 493,
    "name": "c10_SizesAndStrides_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_SizesAndStrides_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "29a01c42af1ed0ac14cf3738d214b17701e24ef83df2a8e3f7edf33dc6e27c04",
    "uid": 0
  },
  {
    "expected_bytes": 702880,
    "mode": 493,
    "name": "c10_Bitset_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_Bitset_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "3a74e3bade8766b3393348b242783cf04f2d8e714550b94d8459fbef7a58275d",
    "uid": 0
  },
  {
    "expected_bytes": 702592,
    "mode": 493,
    "name": "test_dist_autograd",
    "path": "venv/lib/python3.12/site-packages/torch/bin/test_dist_autograd",
    "role": "installed_torch_wheel_payload",
    "sha256": "e24b7ee393c5c9677519e2954d08b1c03b2df334b135800482905e843bc6b013",
    "uid": 0
  },
  {
    "expected_bytes": 702128,
    "mode": 493,
    "name": "operators_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/operators_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "9ba379133acecc2160766ad079ce7e37656e0bc0fe721771fbcc459c5fd82d9c",
    "uid": 0
  },
  {
    "expected_bytes": 699144,
    "mode": 493,
    "name": "c10_ThreadLocalDebugInfo_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_ThreadLocalDebugInfo_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "c3a2932bf7b6d9059cb760065c2958636cbb2dad45ae2e47bf066236588c01dc",
    "uid": 0
  },
  {
    "expected_bytes": 698920,
    "mode": 493,
    "name": "cpu_allocator_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/cpu_allocator_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "4fa71fd19db150c8f63a5710c1d14bedd949cfc41b8d111768856b2b65f1048b",
    "uid": 0
  },
  {
    "expected_bytes": 698264,
    "mode": 493,
    "name": "c10_DeviceGuard_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_DeviceGuard_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "6bdf2efae77515ae72ba65d2a17929f9f67215efdcbc0b1ed3386fef3ae9b121",
    "uid": 0
  },
  {
    "expected_bytes": 698064,
    "mode": 493,
    "name": "c10_InlineDeviceGuard_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_InlineDeviceGuard_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "4dce19391518521c0285d67f56181017af1d93a6d986e9345824590a93071f3b",
    "uid": 0
  },
  {
    "expected_bytes": 697680,
    "mode": 493,
    "name": "lazy_tensor_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/lazy_tensor_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "5fbc6e58fe18b94330c936d371d905e745eac3e0492444a4f84d011807d83471",
    "uid": 0
  },
  {
    "expected_bytes": 696424,
    "mode": 493,
    "name": "reduce_ops_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/reduce_ops_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "33be624259286bfea9780b1d0d6c43aa957a0e7c76b17a7898beb1f25e27028c",
    "uid": 0
  },
  {
    "expected_bytes": 695424,
    "mode": 493,
    "name": "test_privateuse1_profiler",
    "path": "venv/lib/python3.12/site-packages/torch/bin/test_privateuse1_profiler",
    "role": "installed_torch_wheel_payload",
    "sha256": "a72e36341cd235ff3a25428eedc8faa366b5bc3c6a65811dd677beec8ff27d06",
    "uid": 0
  },
  {
    "expected_bytes": 695016,
    "mode": 493,
    "name": "verify_api_visibility",
    "path": "venv/lib/python3.12/site-packages/torch/test/verify_api_visibility",
    "role": "installed_torch_wheel_payload",
    "sha256": "987595daad3a2bf005deffe0d098e2cd5a1e97aafae054b54d0b0883c263abad",
    "uid": 0
  },
  {
    "expected_bytes": 693920,
    "mode": 493,
    "name": "c10_exception_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_exception_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "aa8d7b752add43a2e089d79118462b913f0289f53c3bb0180285fd4ae637ff8e",
    "uid": 0
  },
  {
    "expected_bytes": 693352,
    "mode": 493,
    "name": "c10_TypeIndex_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_TypeIndex_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "8a526304ead9876634f106b024208a371cf7f33ff94ede6c58e2b965fd6a3f6a",
    "uid": 0
  },
  {
    "expected_bytes": 692976,
    "mode": 493,
    "name": "c10_lazy_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_lazy_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "3adcfa01ab6841bc33d9cb3ebae7bba6a5b6d90bbf518ef428d49bd51b40077d",
    "uid": 0
  },
  {
    "expected_bytes": 692944,
    "mode": 493,
    "name": "mobile_memory_cleanup",
    "path": "venv/lib/python3.12/site-packages/torch/test/mobile_memory_cleanup",
    "role": "installed_torch_wheel_payload",
    "sha256": "b21d0189e1322b7529007c6bc6dd9e8ba994dd0ad387d88a4347acb25b5b9f56",
    "uid": 0
  },
  {
    "expected_bytes": 692936,
    "mode": 493,
    "name": "op_allowlist_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/op_allowlist_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "8538893ca72159a6a52f4f15a2ce2d8d3aba9efcd3297ab8c2bfa85044568930",
    "uid": 0
  },
  {
    "expected_bytes": 692600,
    "mode": 493,
    "name": "c10_registry_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_registry_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "63037a68ab812fd25decbad4bbc3a1eadde89fd0e1ae92779dd563d73e1499fa",
    "uid": 0
  },
  {
    "expected_bytes": 692464,
    "mode": 493,
    "name": "HashStoreTest",
    "path": "venv/lib/python3.12/site-packages/torch/bin/HashStoreTest",
    "role": "installed_torch_wheel_payload",
    "sha256": "3eb40007dc43ba85739d6c2960be5a676834bff6e709d5861b0e2146db8cc3db",
    "uid": 0
  },
  {
    "expected_bytes": 691992,
    "mode": 493,
    "name": "c10_NetworkFlow_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_NetworkFlow_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "b3acffeba12f500107b674898efb7d1e691113d66223764ee1047ff95b88b973",
    "uid": 0
  },
  {
    "expected_bytes": 690504,
    "mode": 493,
    "name": "FileStoreTest",
    "path": "venv/lib/python3.12/site-packages/torch/bin/FileStoreTest",
    "role": "installed_torch_wheel_payload",
    "sha256": "f6b69e92b48cf48991d8cadda14d15803f56bb0fecb0de17125be6d6e72e161f",
    "uid": 0
  },
  {
    "expected_bytes": 689840,
    "mode": 493,
    "name": "c10_IntrusiveList_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_IntrusiveList_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "0df0e230438e3111580c2aee76815564cc6438819260dd2553d85c74ed71243f",
    "uid": 0
  },
  {
    "expected_bytes": 688752,
    "mode": 493,
    "name": "c10_ssize_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_ssize_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "2e88e2ea5296ee4812460a6b6a00b2b51f5d516575274dd1fa9f70008052b206",
    "uid": 0
  },
  {
    "expected_bytes": 686840,
    "mode": 493,
    "name": "c10_irange_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_irange_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "6799d731c6b917bc14b74f9f425173c4c5e2bf0aeb801132a9bf2a1c245c59d4",
    "uid": 0
  },
  {
    "expected_bytes": 684792,
    "mode": 493,
    "name": "c10_accumulate_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_accumulate_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "92b60ec389a8f92b5692a07768e1d8f2b05dc22540033831be27729ae0fa901d",
    "uid": 0
  },
  {
    "expected_bytes": 684144,
    "mode": 493,
    "name": "c10_Device_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_Device_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "f9668fe15ec23e705fcd16c713c467e37bea07cd7af7429171f11603840e9818",
    "uid": 0
  },
  {
    "expected_bytes": 683520,
    "mode": 493,
    "name": "c10_flags_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_flags_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "86bafabde05777f412828aab4512c9da1e1b31bc91c20d276c042aa1c832b763",
    "uid": 0
  },
  {
    "expected_bytes": 683232,
    "mode": 493,
    "name": "c10_Half_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_Half_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "462b31e54e667b531ee79b91bd926eb52021a8856419a8e15dd1cade6ac1245c",
    "uid": 0
  },
  {
    "expected_bytes": 683080,
    "mode": 493,
    "name": "c10_Synchronized_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_Synchronized_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "d033ddb051eb57c627333bdad52e5e74cb786e0a63a1e62072a61e232b8820ae",
    "uid": 0
  },
  {
    "expected_bytes": 682008,
    "mode": 493,
    "name": "c10_CompileTimeFunctionPointer_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_CompileTimeFunctionPointer_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "6a664a464e4b309edb7d4ae8bc4aab8828428affbc3fec270726210b0555f829",
    "uid": 0
  },
  {
    "expected_bytes": 681608,
    "mode": 493,
    "name": "c10_bit_cast_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_bit_cast_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "30d2a84ee7eca0b99fc19a69f3937f65552b2006fe46e982263ddfe0a4a6b396",
    "uid": 0
  },
  {
    "expected_bytes": 681200,
    "mode": 493,
    "name": "c10_tempfile_test",
    "path": "venv/lib/python3.12/site-packages/torch/test/c10_tempfile_test",
    "role": "installed_torch_wheel_payload",
    "sha256": "3d9421d088e86cdea19ccb4b0b66f6cc34f8fbb499a56f3b46bf04dc5439c061",
    "uid": 0
  },
  {
    "expected_bytes": 681120,
    "mode": 493,
    "name": "test_shim",
    "path": "venv/lib/python3.12/site-packages/torch/bin/test_shim",
    "role": "installed_torch_wheel_payload",
    "sha256": "d177a48c49536e8b8645c45702c7cd311abfbaa629df112a5e247624d6d4fff8",
    "uid": 0
  }
]
WHEEL_SHA256='6f307c2c32d764ffc6ff6893b801fad6d4752f3e67966cb8abf1843427c02604'
TORCH_REQUIREMENT='torch==2.13.0+cpu'
SOURCE_PINS_SHA256='2d0625a867bc45a9a695ae2f750a57d75179c6aa28ef31250bd478f86ec7846d'
SOURCE_SELECTION='three_archive_bound_extensions_plus_top128_cpu_wheel_members'
INSTALLER_MODE_POLICY='pip_normal_file_0666_umask022_with_executable0755'

def main():
    import platform
    if platform.machine() != 'aarch64':raise ValueError('selected native library policy requires aarch64')
    def expired(*args):raise TimeoutError('bounded native library advice expired')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        result=advise_selected(ROOT,ROWS,expected_count=131)
        result.update(schema='pinned-native-libraries-cache-advice@1',wheel_sha256=WHEEL_SHA256,torch_requirement=TORCH_REQUIREMENT,
            source_selection=SOURCE_SELECTION, source_pins_sha256=SOURCE_PINS_SHA256, installer_mode_policy=INSTALLER_MODE_POLICY)
        print(json.dumps(result,sort_keys=True,allow_nan=False))
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
