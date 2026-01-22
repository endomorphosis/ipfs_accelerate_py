# Manual Model Tests Analysis
Generated: 2025-03-22 20:10:49

## ⚠️ TEMPLATE CONFORMANCE ISSUES: 6 models have missing components

## Model Details

| Model | Architecture | Syntax | Hardware Detection | Mock Objects | Test Class | Pipeline Test | Result Collection |
|-------|--------------|--------|-------------------|-------------|------------|--------------|-------------------|
| layoutlmv2 | vision-encoder-text-decoder | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| layoutlmv3 | vision-encoder-text-decoder | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| clvp | speech | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| bigbird | encoder-decoder | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| seamless_m4t_v2 | speech | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| xlm_prophetnet | encoder-decoder | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

## Missing Components

### layoutlmv2
Missing: pipeline_test, result_collection

### layoutlmv3
Missing: pipeline_test, result_collection

### clvp
Missing: pipeline_test, result_collection

### bigbird
Missing: pipeline_test, result_collection

### seamless_m4t_v2
Missing: pipeline_test, result_collection

### xlm_prophetnet
Missing: pipeline_test, result_collection

## Recommendations

1. Regenerate all tests using the template system to ensure template conformance