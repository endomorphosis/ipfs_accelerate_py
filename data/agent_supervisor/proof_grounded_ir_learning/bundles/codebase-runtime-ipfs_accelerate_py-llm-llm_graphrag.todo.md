# Codebase Bundle: codebase/runtime/ipfs_accelerate_py-llm-llm_graphrag

Source todo: docs/architecture/proof_grounded_ir_learning.todo.md
Purpose: group generated codebase findings by source file and AST locality.
Conflict policy: serialize edits to one file; allow independent file bundles to run concurrently.

## PGIR-112 Resolve code annotation in ipfs_accelerate_py/llm/llm_graphrag.py:736

- Status: todo
- Completion: manual
- Priority: P3
- Track: runtime
- Depends on: 
- Outputs: /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery, ipfs_accelerate_py/llm/llm_graphrag.py
- Validation: python3 -m py_compile ipfs_accelerate_py/llm/llm_graphrag.py
- Board namespace: proof_grounded_ir_learning.todo.md
- Graph parents: PGIR-G000
- Graph depth: 1
- Goal id: PGIR-G040
- Goal lineage: PGIR-G040, PGIR-G000
- Goal registration: existing
- Canonical task key: task/v1/09937701b23cd98811e3a247a7748657fdfc4cedfcbabf90fcb082566e6bae78
- Canonical task CID: baguqeerabgjxoanshtmyqepdujd2o5egk767ythn7s5l7eh4wcbfm3tlvz4a
- Semantic identity: 09937701b23cd98811e3a247a7748657fdfc4cedfcbabf90fcb082566e6bae78
- Acceptance subset: Resolve annotated_followup at ipfs_accelerate_py/llm/llm_graphrag.py:736
- Preconditions: ipfs_accelerate_py/llm/llm_graphrag.py exists and the scan evidence remains applicable
- Effects: resolve annotated_followup in ipfs_accelerate_py/llm/llm_graphrag.py and pass focused validation
- Evidence subset: ipfs_accelerate_py/llm/llm_graphrag.py:736, /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-16-pgir-112-codebase-scan-45be53b0ad05.md
- Resource class: cpu-small
- Token class: small
- Resources: python, focused validation runner
- Merge fate: ipfs_accelerate_py/llm/llm_graphrag.py
- Rejection reasons: none
- Missing evidence: Resolve code annotation in ipfs_accelerate_py/llm/llm_graphrag.py:736
- Candidate kind: codebase_scan
- Todo vector key: 45be53b0ad0590ec
- Bundle: codebase/runtime/ipfs_accelerate_py-llm-llm_graphrag
- Bundle shard: data/agent_supervisor/proof_grounded_ir_learning/bundles/codebase-runtime-ipfs_accelerate_py-llm-llm_graphrag.todo.md
- Bundle strategy: codebase_file_ast
- Parallel lane: codebase/runtime/ipfs_accelerate_py-llm-llm_graphrag
- Conflict policy: serialize findings for the same file; allow independent file bundles to run concurrently
- Predicted files: ipfs_accelerate_py/llm/llm_graphrag.py
- AST symbols: __init__, _create_domain_detector, _create_template_selector, _enhance_result_for_domain, _format_connections_for_llm, _format_documents_for_domain, _get_cross_document_reasoning_schema, _get_evidence_chain_schema, _initialize_domain_rules, _is_domain_applicable, analyze evidence chain, analyze transitive relationships, analyze_evidence_chain, analyze_transitive_relationships, create domain detector, create template selector, datetime, datetime datetime, datetime.datetime, detect domain, detect_domain, detector, domainspecificprocessor, domainspecificprocessor create domain detector, domainspecificprocessor create template selector, domainspecificprocessor detect domain, domainspecificprocessor detector, domainspecificprocessor enhance context with domain, domainspecificprocessor get domain info, domainspecificprocessor init, domainspecificprocessor initialize domain rules, domainspecificprocessor is domain applicable, domainspecificprocessor selector, domainspecificprocessor.__init__, domainspecificprocessor._create_domain_detector, domainspecificprocessor._create_template_selector, domainspecificprocessor._initialize_domain_rules, domainspecificprocessor._is_domain_applicable, domainspecificprocessor.detect_domain, domainspecificprocessor.detector, domainspecificprocessor.enhance_context_with_domain, domainspecificprocessor.get_domain_info, domainspecificprocessor.selector, enhance context with domain, enhance cross document reasoning, enhance document connections, enhance result for domain, enhance_context_with_domain, enhance_cross_document_reasoning, enhance_document_connections, expand by graph, expand_by_graph, format connections for llm, format documents for domain, generate deep inference, generate_deep_inference, get cross document reasoning schema, get domain info, get error summary, get evidence chain schema, get latency percentiles, get model metrics, get recent interactions, get task metrics, get_domain_info, get_error_summary, get_latency_percentiles, get_model_metrics, get_recent_interactions, get_task_metrics, graphragllmprocessor, graphragllmprocessor analyze evidence chain, graphragllmprocessor analyze transitive relationships, graphragllmprocessor enhance result for domain, graphragllmprocessor expand by graph, graphragllmprocessor format documents for domain, graphragllmprocessor generate deep inference, graphragllmprocessor get cross document reasoning schema, graphragllmprocessor get evidence chain schema, graphragllmprocessor identify knowledge gaps
- AST symbol scope: file
- Merge key: codebase/runtime/ipfs_accelerate_py-llm-llm_graphrag
- Merge family: ipfs_accelerate_py/llm/llm_graphrag.py
- Merge role: codebase_scan
- Work item count: 1
- Work scope: codebase_file_ast
- Acceptance: Goal-scoped refill admitted this finding from ipfs_accelerate_py/llm/llm_graphrag.py:736 for PGIR-G040. Use evidence in /home/barberb/lift_coding/.pgir_campaign/runtime/parallel/discovery/2026-08-16-pgir-112-codebase-scan-45be53b0ad05.md, make only the smallest change required by that goal lineage, add or update focused validation when appropriate, and do not expand into adjacent cleanup.
