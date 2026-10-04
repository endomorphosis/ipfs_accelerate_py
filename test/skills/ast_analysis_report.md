# Test Codebase AST Analysis Report

Generated on: 2025-03-22 22:36:19

## Summary

- **Total Files Analyzed:** 691
- **Total Classes:** 308
- **Total Methods:** 2280
- **Average Methods per Class:** 7.40
- **Total Cyclomatic Complexity:** 20744
- **Average Complexity per File:** 30.02

## from_pretrained Implementation Types

| Implementation Type | Count | Percentage |
|---------------------|-------|------------|
| Explicit Method | 142 | 20.5% |
| Alternative Method | 27 | 3.9% |
| Direct Usage | 1 | 0.1% |
| Pipeline Usage | 31 | 4.5% |
| No Usage | 8 | 1.2% |

## Potential Code Duplication

Found 24 groups of potentially similar files:

### Group 1: 37 files

Structural fingerprint: `classes:3,class_pattern:MockSentencePieceProcessor-MockTokenizer-TestModelTypeModels,methods:16,signatures:args:1,return:6-args:2,return:3-args:0:2`

Files:
- `./fixed_tests/test_hf_resnet.py`
- `./fixed_tests/test_hf_flava.py`
- `./fixed_tests/test_hf_rembert.py`
- `./fixed_tests/test_hf_mask2former.py`
- `./fixed_tests/test_hf_ernie.py`
- `./fixed_tests/test_hf_imagebind.py`
- `./fixed_tests/test_hf_mpt.py`
- `./fixed_tests/test_hf_detr.py`
- `./fixed_tests/test_hf_gemma.py`
- `./fixed_tests/test_hf_tapas.py`
- `./fixed_tests/test_hf_unispeech.py`
- `./fixed_tests/test_hf_deberta.py`
- `./fixed_tests/test_hf_camembert.py`
- `./fixed_tests/test_hf_idefics.py`
- `./fixed_tests/test_hf_albert.py`
- `./fixed_tests/test_hf_yolos.py`
- `./fixed_tests/test_hf_speech_to_text.py`
- `./fixed_tests/test_hf_squeezebert.py`
- `./fixed_tests/test_hf_roberta.py`
- `./fixed_tests/test_hf_reformer.py`
- `./fixed_tests/test_hf_electra.py`
- `./fixed_tests/test_hf_sew.py`
- `./fixed_tests/test_hf_musicgen.py`
- `./fixed_tests/test_hf_xlnet.py`
- `./fixed_tests/test_hf_funnel.py`
- `./fixed_tests/test_hf_prophetnet.py`
- `./fixed_tests/test_hf_clap.py`
- `./fixed_tests/test_hf_encodec.py`
- `./fixed_tests/test_hf_distilbert.py`
- `./fixed_tests/test_hf_sam.py`
- `./fixed_tests/test_hf_longformer.py`
- `./fixed_tests/test_hf_opt.py`
- `./fixed_tests/test_hf_hubert.py`
- `./test_hf_speech-to-text.py`
- `./test_hf_fuyu.py`
- `./test_hf_speech_to_text.py`
- `./test_hf_bigbird.py`

### Group 2: 16 files

Structural fingerprint: `classes:1,class_pattern:TestVitModels,methods:5,signatures:args:1:1-args:1,try-except,return:1-args:1,try-except,return,from_pretrained:1`

Files:
- `./fixed_tests/test_hf_bit.py`
- `./fixed_tests/test_hf_donut.py`
- `./fixed_tests/test_hf_regnet.py`
- `./fixed_tests/test_hf_levit.py`
- `./fixed_tests/test_hf_swin.py`
- `./fixed_tests/test_hf_vit.py`
- `./fixed_tests/test_hf_mlp_mixer.py`
- `./fixed_tests/test_hf_deit.py`
- `./fixed_tests/test_hf_efficientnet.py`
- `./fixed_tests/test_hf_dinov2.py`
- `./fixed_tests/test_hf_dpt.py`
- `./fixed_tests/test_hf_beit.py`
- `./fixed_tests/test_hf_mobilevit.py`
- `./fixed_tests/test_hf_mlp-mixer.py`
- `./fixed_tests/test_hf_convnext.py`
- `./fixed_tests/test_hf_segformer.py`

### Group 3: 12 files

Structural fingerprint: `classes:1,class_pattern:TestWhisperModels,methods:5,signatures:args:1:1-args:1,try-except,return:1-args:1,try-except,return,from_pretrained:1`

Files:
- `./fixed_tests/test_hf_speech-to-text-2.py`
- `./fixed_tests/test_hf_usm.py`
- `./fixed_tests/test_hf_seamless_m4t_v2.py`
- `./fixed_tests/test_hf_clvp.py`
- `./fixed_tests/test_hf_wavlm.py`
- `./fixed_tests/test_hf_bark.py`
- `./fixed_tests/test_hf_data2vec.py`
- `./fixed_tests/test_hf_wav2vec2.py`
- `./fixed_tests/test_hf_speech-to-text.py`
- `./fixed_tests/test_hf_whisper.py`
- `./fixed_tests/test_hf_speech_to_text_2.py`
- `./test_hf_bark.py`

### Group 4: 10 files

Structural fingerprint: `classes:1,class_pattern:TestVisionTextModels,methods:6,signatures:args:1:1-args:0,return:1-args:1,try-except,return:1`

Files:
- `./fixed_tests/test_hf_florence.py`
- `./fixed_tests/test_hf_vinvl.py`
- `./fixed_tests/test_hf_vilt.py`
- `./fixed_tests/test_hf_layoutlmv3.py`
- `./fixed_tests/test_hf_layoutlmv2.py`
- `./fixed_tests/test_hf_clip.py`
- `./fixed_tests/test_hf_clipseg.py`
- `./fixed_tests/test_hf_blip.py`
- `./fixed_tests/test_hf_align.py`
- `./test_hf_clipseg.py`

### Group 5: 9 files

Structural fingerprint: `classes:1,class_pattern:TestT5Models,methods:5,signatures:args:1,try-except,return,from_pretrained:2-args:1:1-args:0,try-except,return,from_pretrained:1`

Files:
- `./fixed_tests/test_hf_marian.py`
- `./fixed_tests/test_hf_xlm_prophetnet.py`
- `./fixed_tests/test_hf_mt5.py`
- `./fixed_tests/test_hf_bart.py`
- `./fixed_tests/test_hf_bigbird.py`
- `./fixed_tests/test_hf_pegasus.py`
- `./fixed_tests/test_hf_blenderbot.py`
- `./fixed_tests/test_hf_t5.py`
- `./fixed_tests/test_hf_mbart.py`

### Group 6: 7 files

Structural fingerprint: `classes:0,methods:0`

Files:
- `./test_inference_validation.py`
- `./test_toolkit.py`
- `./test_integration.py`
- `./test_indentation_tools.py`
- `./test_model_lookup.py`
- `./test_generator_fixed.py`
- `./test_model_lookup_advanced.py`

### Group 7: 4 files

Structural fingerprint: `classes:1,class_pattern:TestXLMRoBERTaModels,methods:3,signatures:args:2:1-args:0,try-except,return:1-args:1,return:1`

Files:
- `./fixed_tests/test_hf_xlm-roberta.py`
- `./fixed_tests/test_hf_xlm_roberta.py`
- `./test_hf_xlm-roberta.py`
- `./test_hf_xlm_roberta.py`

### Group 8: 4 files

Structural fingerprint: `classes:1,class_pattern:TestFlanT5Models,methods:3,signatures:args:2:1-args:0,try-except,return:1-args:1,return:1`

Files:
- `./fixed_tests/test_hf_flan-t5.py`
- `./fixed_tests/test_hf_flan_t5.py`
- `./test_hf_flan-t5.py`
- `./test_hf_flan_t5.py`

### Group 9: 4 files

Structural fingerprint: `classes:1,class_pattern:TestTransfoXlModels,methods:5,signatures:args:1,try-except,return,from_pretrained:2-args:1:1-args:0,try-except,return,from_pretrained:1`

Files:
- `./fixed_tests/test_hf_transfo-xl.py`
- `./fixed_tests/test_hf_transfo_xl.py`
- `./test_hf_transfo_xl.py`
- `./test_hf_transfo-xl.py`

### Group 10: 4 files

Structural fingerprint: `classes:1,class_pattern:TestGPTJModels,methods:3,signatures:args:2:1-args:0,try-except,return:1-args:1,return:1`

Files:
- `./fixed_tests/test_hf_gpt_j.py`
- `./fixed_tests/test_hf_gpt-j.py`
- `./test_hf_gpt-j.py`
- `./test_hf_gpt_j.py`

...and 14 more groups

## Common Patterns

### Most Common Method Names

| Method Name | Occurrences |
|-------------|-------------|
| `__init__` | 285 |
| `main` | 186 |
| `run_tests` | 181 |
| `test_pipeline` | 178 |
| `save_results` | 149 |
| `test_from_pretrained` | 142 |
| `decode` | 105 |
| `check_hardware` | 88 |
| `get_available_models` | 88 |
| `test_all_models` | 88 |

### Most Common Class Names

| Class Name | Occurrences |
|------------|-------------|
| `MockTokenizer` | 53 |
| `MockSentencePieceProcessor` | 52 |
| `TestModelTypeModels` | 37 |
| `TestVitModels` | 18 |
| `TestWhisperModels` | 12 |
| `TestVisionTextModels` | 12 |
| `TestT5Models` | 11 |
| `TestXLMRoBERTaModels` | 4 |
| `TestFlanT5Models` | 4 |
| `TestTransfoXlModels` | 4 |

### Most Common Method Patterns

| Pattern | Occurrences |
|---------|-------------|
| `args:1,return` | 459 |
| `args:0,return` | 399 |
| `args:2,return` | 257 |
| `args:0,try-except,return,from_pretrained` | 204 |
| `args:0,try-except,return` | 175 |
| `args:0` | 137 |
| `args:1,try-except,return,from_pretrained` | 130 |
| `args:2` | 103 |
| `args:1` | 97 |
| `args:3,return` | 97 |

### Most Complex Files

| File Path | Complexity |
|-----------|------------|
| `./test_generator_fixed.py` | 474 |
| `./fixed_tests/test_hf_xclip.py` | 269 |
| `./test_hf_xclip.py` | 269 |
| `./fixed_tests/test_hf_layoutlmv3.py` | 206 |
| `./fixed_tests/test_hf_layoutlmv2.py` | 206 |
| `./fixed_tests/test_hf_florence.py` | 204 |
| `./fixed_tests/test_hf_vinvl.py` | 204 |
| `./fixed_tests/test_hf_vilt.py` | 204 |
| `./fixed_tests/test_hf_clip.py` | 204 |
| `./fixed_tests/test_hf_clipseg.py` | 204 |

## Refactoring Recommendations

1. **Standardize Method Implementations**: The pattern `args:1,return` appears in 459 methods (20.1% of all methods). Consider standardizing this pattern across the codebase.

2. **Refactor Similar Files**: The largest group contains 37 similar files. Consider creating a common base class or utility functions.

3. **Standardize from_pretrained Testing**: Currently using 5 different approaches for testing from_pretrained(). Consider standardizing to the explicit method pattern.

4. **Simplify Complex Files**: The most complex file `./test_generator_fixed.py` has a complexity score of 474, which is 15.8x the average. Consider breaking it into smaller components.

5. **Template-Based Generation**: Given the structural similarities, use template-based generation for all test files to ensure consistency.

## Conclusion

The test codebase shows clear patterns that can be leveraged for standardization. While there is some duplication, it appears to be structured and intentional based on the test generator approach. The main opportunity is to standardize the from_pretrained() testing methodology across all model types.

## Next Steps

1. Create standardized templates for each model architecture type
2. Implement a unified approach to from_pretrained() testing
3. Reduce complexity in the most complex files
4. Update the test generator to ensure consistent patterns
5. Add comprehensive validation for all generated files

---

This report was generated using `generate_ast_report.py`.
To update this report, run:
```bash
python generate_ast_report.py
```
