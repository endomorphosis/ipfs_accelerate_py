# Finding repair taskboard (ipfs-kit-vfs-symbolic-assurance-v1)

- revision: 1
- policy_id: baguqeeraekdswwgbpwoz6rl6rii2cpjak7dyhjrlj7wktse5ret2kaxr4d7q
- tree_id: baguqeerabgy5lvrxna6yljy3ljtq2ntlnlykhzhtpfijbu4jfm72pz4ocq7q
- goal_id: VFS-G101
- goal_lineage: VFS-G000, VFS-G100, VFS-G101
- evidence: vfs/finding-taskboard@1
- executable_tasks: 1
- reviews: 1
- authorizes_repair: false
- is_completion_evidence: false

## Executable repair tasks

### VFS-R-0001 Repair vfs-pilot-seeded-contract-break: Pilot fixture marks an explicit contract break for src/broken.ts

- goal_id: VFS-G101
- parent_goal_id: VFS-G100
- goal_lineage: VFS-G000, VFS-G100, VFS-G101
- root_cause_family: vfs-pilot-seeded-contract-break
- merge_fate: pilot:swissknife:src/broken.ts
- conflict_domain: baguqeerajf34vwimd7jwe2clw7p4nlepefode2pn6z3fcaqx3fxwoeuiiepa
- resource_class: cpu-small
- token_class: small
- risk_millionths: 700000
- context_ceiling_bytes: 12288
- context_ceiling_tokens: 3072
- finding_cids: baguqeeracnblhyus4ctj327ergcid2jcpvv73irnip6zwiu5uwkctqgvqhuq
- provenance_cids: baguqeeracnblhyus4ctj327ergcid2jcpvv73irnip6zwiu5uwkctqgvqhuq, baguqeeracwfaxi7wmsspojxik4yhvcfl2rukp3ups5v7npl4xzthvbemmbdq, baguqeeramenkocyhe75p2reon7nymohjomrnju2r2t3hal3pnejqkxn5atpa, baguqeeraogb2yplmtxg5v5wbieyio5hrgqcrzjm7472yvs5ytcdsyl53uf5a, baguqeerarcvyatrmvdx45vtgnojjeaukoxehksnulydq3vtmry6dex63jx5a
- outputs: src/broken.ts
- symbols: src/broken.ts
- effects: align_observed:baguqeeraogb2yplmtxg5v5wbieyio5hrgqcrzjm7472yvs5ytcdsyl53uf5a, preserve_interfaces:pilot://swissknife/src/broken.ts, restore:vfs-pilot-seeded-contract-break, satisfy_expected:baguqeeracwfaxi7wmsspojxik4yhvcfl2rukp3ups5v7npl4xzthvbemmbdq
- dependencies: none
- validation_plan: python -m pytest test/api/test_agent_supervisor_contract_findings.py -q
- proof_plan: confirm_bound_counterexamples; recompute_contract_finding_from_evidence; verify_finding_cid_binds_expected_and_observed
- semantic_key: baguqeerahjkow7zeek5ahjtpex4cvhve4av43jsqatu2gifmooz3li7zxbxq
- executable: true

## Non-executable review records

### VFS-R-REV-0001

- finding_cids: baguqeeraq6hj37d7lo27lzeb6wtsqiof74eccyirj6bkzvkhmsf7rf4hdsxa
- reasons: not_admitted
- summary: Pilot fixture is explicitly inconclusive for src/maybe.ts
- root_cause_family: vfs-pilot-inconclusive
- executable: false
