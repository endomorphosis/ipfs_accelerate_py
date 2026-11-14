# Incomplete dependency-release source preservation

This branch retains 33 exact source files from the working checkout. It is
excluded from `main`; it grants no task, repair, model, proof, or legal admission.

The current canonical `IntentRepository` has no
`release_dependency_satisfied_blocks` method. The integration test
`test/api/test_board_stall_watchdog.py::test_watchdog_creates_subgoals_for_an_idle_blocked_goal_without_setting_ready`
failed on 2026-11-14: it expected `progressed` and received `release_failed`.
Calling the operation directly on the same synthetic catalogue reproduced
`AttributeError: 'IntentRepository' object has no attribute 'release_dependency_satisfied_blocks'`.
The broader six-file run recorded 538 passing tests and two failures. The other
failure was an outdated authorization fixture corrected in main; this missing
operation remains incomplete. The failed output is preserved locally, without
publishing runtime databases, private keys, caches, or logs.

The next task is to implement the transactional operation in the existing
repository owner, with dependency, evidence, revision, hold, and admission
checks. Run these existing tests before merging this branch. Do not replace the
missing operation with a stub or treat fixture results as scientific evidence.

The preserved source payloads are:

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| `ipfs_accelerate_py/agent_supervisor/task_sources/board_stall_watchdog.py` | 18755 | `8a769e28b593a1c8889b6d24026eaea64cacb332a57c61c4f711c6eab056f02e` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/byte_target_eligibility_probe.py` | 6565 | `436d2f37d8a1225d5736dccd377a6b3802b3856ff8d3ef5030fb6c31aa6e8773` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/codec_intact_probe.py` | 7670 | `65f8603f94d60b5dc0d132d331f5fe56d905303ad295e0916fc19ac66283774f` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/consequence_audit_probe.py` | 23334 | `a09c2dcf4c8e30e1ab77b65370936bbedd12aa040a826605ef04ac17df8b40bd` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/context_budget_audit_probe.py` | 14184 | `e110e8d561cfc2d59f4a58c15cc5eb654bf47e0e76055a7688832c91bb882979` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/definition_correspondence_probe.py` | 17853 | `a146b555bb5fbf83bb4028553aa629f41c1a0999778298b9b9d07ede1497eb8f` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/dependency_block_resolution.py` | 244741 | `4973c5fd5e26a46445934c161aa23a18b1272328e06972fbc044e0bbf727a138` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/dependency_closure_probe.py` | 10696 | `b4001a4a11255ff57ea676b98914c3ec7e2c9bca86c78ff4d17399d4de747ae4` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/family_lowering_probe.py` | 9272 | `8bc8959d0e830223d3fc9978c4566ac84bb322925a6b1b10702053b0a2f84bbb` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/family_route_probe.py` | 5518 | `1ea7e171eefc9dab62ac569d0974c9bb7cc04fb456d9525abe67bed8f1bdd582` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/graph_roundtrip_probe.py` | 8358 | `0f4c51506f650875bb98ff1a8f6b85e1f9d4d97d34e388ce3e2ddc955d16f9ba` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/incremental_lake_agreement_probe.py` | 22805 | `716dfffe0e7eaec6c79e657669cad7cf59d50c8aa61700f9d532c0440b440b7c` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/invalidation_keys_probe.py` | 11964 | `50d310921069ca888ab2fc1a47586b9b566771b1cef38bede681b5f89c58e600` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/label_mask_audit_probe.py` | 16530 | `53a6a7f7b66c07ff6bb90d9da75a3d349b3fa10d7dcc839773357c93dac267e4` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/lookup_partition_probe.py` | 13361 | `f07858f5254a6dce89aafea662c878f798955cfb0236925ae4b5cc573ca14795` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/meaning_material_probe.py` | 28246 | `512ba3c59cdc41edd53e39d99b635c2334f8066f14552236c904450146d3ebba` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/modality_causal_comparison_probe.py` | 4830 | `6816e483e99f56dcdfb7303d896c8c8deef6aeb4a73594a3c862c69dcbf1a041` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/occurrence_visibility_audit_probe.py` | 22520 | `d70af73f3184bcf7654269603e20538d633a498457c045a286846f4c6e8790ae` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/population_coverage_audit_probe.py` | 7120 | `1654b11324edb03239a660952847e7090b884e8b9975509dd4e647bf69349679` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/release_coordinator_probe.py` | 6975 | `d35066f4b755664626f7552001a37b1cdbb426655f80fd36c08e32677e3582a3` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/scoped_graph_probe.py` | 6369 | `dd2499699e63b77b7643238a00c09f9b0d89adb09be8dc4e9136c8bda1fe0b13` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/shared_lake_build_probe.py` | 8790 | `8a4b51495c3f99c3a677d8c8121543f610d4018a10073a15297726ded5e4a353` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/shared_module_probe.py` | 11895 | `9eaf9b8954e9e0c70bb455e515a5f16d749ba5260b9f27d6cd602ebeb30ea173` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/snapshot_custody_probe.py` | 7696 | `3a46132c745d341cf496ae29d2cde24deb23d5696bb9e28d4b8280a5eb9cd352` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/storage_concurrency_audit_probe.py` | 7504 | `4db6c6d87e3b3c86d5d34dc0a37337ca747fca0a511554d34528a3281961bc22` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/teacher_identity_audit_probe.py` | 6931 | `70debcb03b572cc3d9e46b3a09f86ae51e2a47053cac698e27b15d0555a55bb9` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/typed_decoder_probe.py` | 4577 | `ed01c3833bde554c08bc2533ab1cea0f554e81698c4b830e97cc8cbcff09b0c6` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/unreviewed_representation_probe.py` | 2467 | `e4c97447a9161e3e48b06fa2a492515558a46a30de2fdf3c460fe09b5d34e1c8` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/v2_owner_envelope_probe.py` | 14584 | `3ea41af01e04da5fd17df4fe1c7f242cea6011a4ebbe497cfdc036b873487f6c` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/width_comparison_cell_probe.py` | 2425 | `9e4781f96b51832c2ff096cc3d7164815a8fcf38d1efac5b7ee4fcfe3181ae80` |
| `ipfs_accelerate_py/agent_supervisor/task_sources/width_control_specification_probe.py` | 6810 | `9668d08f740e172f323313a164b6261f1f23bfb54dd20a6ee0f686e93a48d8b2` |
| `test/api/test_agent_supervisor_dependency_block_release.py` | 384749 | `daa77401013b6ecafd689dd8c29bb33c46d4657b3317cac2ccaa6d33d7a8225d` |
| `test/api/test_board_stall_watchdog.py` | 10869 | `6828b882c2444c9f6639f02c03cfc6130b7c84d4c81817a5dda1e3b689bcb392` |
