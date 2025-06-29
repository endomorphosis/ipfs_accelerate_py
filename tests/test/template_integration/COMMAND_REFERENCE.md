# Template Integration Command Reference

## Quick Reference

| Task | Command |
|------|---------|
| List all configured models | `python model_template_fixes.py --list-models` |
| Generate all model tests | `python model_template_fixes.py --generate-all --verify` |
| Generate specific model test | `python model_template_fixes.py --generate-model MODEL --verify` |
| Generate problematic models | `python model_template_fixes.py --generate-specific --verify` |
| Fix indentation issues | `python fix_template_issues.py` |
| Run full integration workflow | `python template_integration_workflow.py` |

## Full Command Reference

### Model Template Fixes

**List all configured models:**
```bash
python model_template_fixes.py --list-models
```

**Generate a test file for a specific model:**
```bash
python model_template_fixes.py --generate-model layoutlmv2 --verify
```

**Generate test files for all models:**
```bash
python model_template_fixes.py --generate-all --verify
```

**Generate test files for specific problematic models:**
```bash
python model_template_fixes.py --generate-specific --verify
```

**Verify a specific model test file:**
```bash
python model_template_fixes.py --verify-model layoutlmv2
```

**Apply changes to architecture types:**
```bash
python model_template_fixes.py --generate-all --verify --apply
```

### Template Integration Workflow

**Run the complete integration workflow:**
```bash
python template_integration_workflow.py
```

**Skip analysis step:**
```bash
python template_integration_workflow.py --skip-analysis
```

**Skip generation step:**
```bash
python template_integration_workflow.py --skip-generation
```

**Skip verification step:**
```bash
python template_integration_workflow.py --skip-verification
```

**Apply changes to architecture types:**
```bash
python template_integration_workflow.py --apply
```

### Fix Template Issues

**Fix indentation issues for problematic models:**
```bash
python fix_template_issues.py
```

## Common Troubleshooting Commands

**View analysis report:**
```bash
cat manual_models_analysis.md
```

**View integration summary:**
```bash
cat template_integration_summary.md
```

**Check syntax of a generated file:**
```bash
python -m py_compile /path/to/fixed_tests/test_hf_layoutlmv2.py
```

**Diff original and generated files:**
```bash
diff /path/to/final_models/test_layoutlmv2.py /path/to/fixed_tests/test_hf_layoutlmv2.py
```

**Check for specific model support:**
```bash
grep -r "layoutlmv2" --include="*.py" .
```