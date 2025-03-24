# Test Codebase Refactoring Recommendations

**Generated on:** 2025-03-22 23:39:37

## Executive Summary

This analysis examines the structure and patterns of test files in the IPFS Accelerate Python project to inform a comprehensive refactoring initiative.

- **Total test files analyzed:** 2569
- **Total test classes:** 6027
- **Total test methods:** 27035
- **Potential duplicate tests:** 4254

## Test Codebase Structure

### Test Files by Directory

| Directory | Count |
|------------|-------|
| ./skills/fixed_tests | 166 |
| root | 153 |
| ./test/models/text/bert | 79 |
| ./skills/temp_generated | 55 |
| ./duckdb_api/distributed_testing/tests | 53 |
| ./skills | 43 |
| ./transformers/tests/utils | 35 |
| ./doc-builder/transformers-docs/transformers/tests/utils | 35 |
| ./transformers/tests/pipelines | 30 |
| ./doc-builder/transformers-docs/transformers/tests/pipelines | 30 |
| ./test/models/text | 24 |
| ./distributed_testing/tests | 24 |
| ./fixed_tests | 20 |
| ./skills/fixed_tests/standardized | 18 |
| ./generated_tests | 18 |
| ./duckdb_api/simulation_validation | 16 |
| ./transformers/tests | 13 |
| ./doc-builder/transformers-docs/transformers/tests | 13 |
| ./duckdb_api/distributed_testing | 13 |
| ./transformers/tests/agents | 12 |
| ./distributed_testing | 12 |
| ./doc-builder/transformers-docs/transformers/tests/agents | 12 |
| ./predictive_performance | 10 |
| ./transformers/tests/generation | 9 |
| ./transformers/tests/trainer | 9 |
| ./doc-builder/transformers-docs/transformers/tests/trainer | 9 |
| ./doc-builder/transformers-docs/transformers/tests/generation | 9 |
| ./transformers/tests/models/auto | 8 |
| ./refactored_test_suite/models/text | 8 |
| ./huggingface_doc_builder/tests | 8 |
| ./doc-builder/tests | 8 |
| ./doc-builder/transformers-docs/transformers/tests/models/auto | 8 |
| ./skills/test_output | 7 |
| ./generated_tests_enhanced | 6 |
| ./test/api/other | 6 |
| ./test/hardware/webgpu | 6 |
| ./transformers/tests/models/clip | 6 |
| ./transformers/tests/models/blip | 6 |
| ./transformers/tests/models/wav2vec2 | 6 |
| ./transformers/tests/models/whisper | 6 |
| ./skills/output_tests | 6 |
| ./generators/runners/end_to_end | 6 |
| ./priority_model_tests_fixed | 6 |
| ./generated_tests_minimal | 6 |
| ./doc-builder/transformers-docs/transformers/tests/models/blip | 6 |
| ./doc-builder/transformers-docs/transformers/tests/models/clip | 6 |
| ./doc-builder/transformers-docs/transformers/tests/models/wav2vec2 | 6 |
| ./doc-builder/transformers-docs/transformers/tests/models/whisper | 6 |
| ./duckdb_api/simulation_validation/test | 6 |
| ./final_models | 6 |
| ./priority_model_tests | 6 |
| ./generated_tests_final | 6 |
| ./fixed_generated_tests | 6 |
| ./transformers/tests/models/gpt2 | 5 |
| ./transformers/tests/models/layoutlmv3 | 5 |
| ./transformers/tests/models/speech_to_text | 5 |
| ./transformers/tests/models/bert | 5 |
| ./transformers/tests/repo_utils | 5 |
| ./skills/samples | 5 |
| ./skills/fixed_files_manual | 5 |
| ./doc-builder/transformers-docs/transformers/tests/models/speech_to_text | 5 |
| ./doc-builder/transformers-docs/transformers/tests/models/gpt2 | 5 |
| ./doc-builder/transformers-docs/transformers/tests/models/bert | 5 |
| ./doc-builder/transformers-docs/transformers/tests/models/layoutlmv3 | 5 |
| ./doc-builder/transformers-docs/transformers/tests/repo_utils | 5 |
| ./duckdb_api/visualization/advanced_visualization | 5 |
| ./medium_priority_tests | 5 |
| ./test/integration | 4 |
| ./transformers/tests/models/clvp | 4 |
| ./transformers/tests/models/bart | 4 |
| ./transformers/tests/models/marian | 4 |
| ./transformers/tests/models/xlm_roberta | 4 |
| ./transformers/tests/models/speecht5 | 4 |
| ./transformers/tests/models/idefics | 4 |
| ./transformers/tests/models/t5 | 4 |
| ./transformers/tests/models/vit | 4 |
| ./transformers/tests/models/markuplm | 4 |
| ./transformers/tests/models/distilbert | 4 |
| ./transformers/tests/models/pop2piano | 4 |
| ./transformers/tests/models/data2vec | 4 |
| ./transformers/tests/models/esm | 4 |
| ./transformers/tests/models/seamless_m4t | 4 |
| ./transformers/tests/models/xglm | 4 |
| ./transformers/tests/models/pegasus | 4 |
| ./transformers/tests/models/llava | 4 |
| ./transformers/tests/models/rag | 4 |
| ./transformers/tests/models/electra | 4 |
| ./transformers/tests/models/albert | 4 |
| ./transformers/tests/models/dpt | 4 |
| ./transformers/tests/models/vision_text_dual_encoder | 4 |
| ./transformers/tests/models/layoutlmv2 | 4 |
| ./transformers/tests/models/blenderbot | 4 |
| ./transformers/tests/models/roformer | 4 |
| ./transformers/tests/models/blenderbot_small | 4 |
| ./transformers/tests/models/mbart | 4 |
| ./transformers/tests/models/roberta | 4 |
| ./skills/backups/fixed_tests.bak.20250319_223051 | 4 |
| ./skills/ultra_simple_tests | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/xglm | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/blenderbot | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/vit | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/layoutlmv2 | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/xlm_roberta | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/electra | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/esm | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/llava | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/marian | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/roberta | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/clvp | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/roformer | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/bart | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/seamless_m4t | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/distilbert | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/speecht5 | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/mbart | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/vision_text_dual_encoder | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/markuplm | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/rag | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/pegasus | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/dpt | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/albert | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/blenderbot_small | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/pop2piano | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/idefics | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/data2vec | 4 |
| ./doc-builder/transformers-docs/transformers/tests/models/t5 | 4 |
| ./refactored_generator_suite/tests | 4 |
| ./duckdb_api/distributed_testing/dashboard/static/sounds | 4 |
| ./transformers/tests/models/pix2struct | 3 |
| ./transformers/tests/models/gemma3 | 3 |
| ./transformers/tests/models/fuyu | 3 |
| ./transformers/tests/models/udop | 3 |
| ./transformers/tests/models/vision_encoder_decoder | 3 |
| ./transformers/tests/models/dpr | 3 |
| ./transformers/tests/models/resnet | 3 |
| ./transformers/tests/models/xlm | 3 |
| ./transformers/tests/models/qwen2_vl | 3 |
| ./transformers/tests/models/maskformer | 3 |
| ./transformers/tests/models/layoutlm | 3 |
| ./transformers/tests/models/clap | 3 |
| ./transformers/tests/models/lxmert | 3 |
| ./transformers/tests/models/flaubert | 3 |
| ./transformers/tests/models/idefics2 | 3 |
| ./transformers/tests/models/longformer | 3 |
| ./transformers/tests/models/siglip | 3 |
| ./transformers/tests/models/regnet | 3 |
| ./transformers/tests/models/mgp_str | 3 |
| ./transformers/tests/models/beit | 3 |
| ./transformers/tests/models/led | 3 |
| ./transformers/tests/models/roberta_prelayernorm | 3 |
| ./transformers/tests/models/deberta_v2 | 3 |
| ./transformers/tests/models/deberta | 3 |
| ./transformers/tests/models/smolvlm | 3 |
| ./transformers/tests/models/mpnet | 3 |
| ./transformers/tests/models/mobilebert | 3 |
| ./transformers/tests/models/mllama | 3 |
| ./transformers/tests/models/llava_next | 3 |
| ./transformers/tests/models/musicgen_melody | 3 |
| ./transformers/tests/models/gemma | 3 |
| ./transformers/tests/models/encoder_decoder | 3 |
| ./transformers/tests/models/deit | 3 |
| ./transformers/tests/models/got_ocr2 | 3 |
| ./transformers/tests/models/bloom | 3 |
| ./transformers/tests/models/tapas | 3 |
| ./transformers/tests/models/llava_next_video | 3 |
| ./transformers/tests/models/bridgetower | 3 |
| ./transformers/tests/models/xlnet | 3 |
| ./transformers/tests/models/ctrl | 3 |
| ./transformers/tests/models/openai | 3 |
| ./transformers/tests/models/convnext | 3 |
| ./transformers/tests/models/donut | 3 |
| ./transformers/tests/models/idefics3 | 3 |
| ./transformers/tests/models/flava | 3 |
| ./transformers/tests/models/owlvit | 3 |
| ./transformers/tests/models/rembert | 3 |
| ./transformers/tests/models/chinese_clip | 3 |
| ./transformers/tests/models/opt | 3 |
| ./transformers/tests/models/llama | 3 |
| ./transformers/tests/models/chameleon | 3 |
| ./transformers/tests/models/funnel | 3 |
| ./transformers/tests/models/sam | 3 |
| ./transformers/tests/models/llava_onevision | 3 |
| ./transformers/tests/models/aria | 3 |
| ./transformers/tests/models/mobilevit | 3 |
| ./transformers/tests/models/segformer | 3 |
| ./transformers/tests/models/big_bird | 3 |
| ./transformers/tests/models/mt5 | 3 |
| ./transformers/tests/models/gptj | 3 |
| ./transformers/tests/models/rt_detr | 3 |
| ./transformers/tests/models/instructblipvideo | 3 |
| ./transformers/tests/models/owlv2 | 3 |
| ./transformers/tests/models/mistral | 3 |
| ./transformers/tests/models/grounding_dino | 3 |
| ./transformers/tests/models/camembert | 3 |
| ./transformers/tests/models/pixtral | 3 |
| ./transformers/tests/models/oneformer | 3 |
| ./transformers/tests/sagemaker | 3 |
| ./skills/minimal_tests | 3 |
| ./refactored_test_suite/api | 3 |
| ./refactored_test_suite/tests/models/text | 3 |
| ./refactored_test_suite/models/multimodal | 3 |
| ./refactored_test_suite/models/audio | 3 |
| ./key_models_hardware_fixes | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/idefics3 | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/mpnet | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/deit | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/rembert | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/grounding_dino | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/roberta_prelayernorm | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/llava_onevision | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/vision_encoder_decoder | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/convnext | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/instructblipvideo | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/gemma | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/tapas | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/chameleon | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/bloom | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/mllama | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/chinese_clip | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/siglip | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/mobilevit | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/regnet | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/lxmert | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/beit | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/rt_detr | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/big_bird | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/dpr | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/segformer | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/xlm | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/mistral | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/pixtral | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/clap | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/gemma3 | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/resnet | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/camembert | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/fuyu | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/udop | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/oneformer | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/led | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/qwen2_vl | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/llava_next_video | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/layoutlm | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/mobilebert | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/llava_next | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/musicgen_melody | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/bridgetower | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/gptj | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/deberta_v2 | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/mgp_str | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/mt5 | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/owlv2 | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/flaubert | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/aria | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/encoder_decoder | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/idefics2 | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/xlnet | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/maskformer | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/funnel | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/pix2struct | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/flava | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/donut | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/got_ocr2 | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/openai | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/deberta | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/smolvlm | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/longformer | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/opt | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/sam | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/ctrl | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/llama | 3 |
| ./doc-builder/transformers-docs/transformers/tests/models/owlvit | 3 |
| ./doc-builder/transformers-docs/transformers/tests/sagemaker | 3 |
| ./duckdb_api/benchmark_validation/tests | 3 |
| ./test/api/llm_providers | 2 |
| ./test/models/vision | 2 |
| ./transformers/examples/pytorch | 2 |
| ./transformers/tests/tokenization | 2 |
| ./transformers/tests/models/vilt | 2 |
| ./transformers/tests/models/zoedepth | 2 |
| ./transformers/tests/models/videomae | 2 |
| ./transformers/tests/models/moshi | 2 |
| ./transformers/tests/models/vits | 2 |
| ./transformers/tests/models/trocr | 2 |
| ./transformers/tests/models/instructblip | 2 |
| ./transformers/tests/models/yolos | 2 |
| ./transformers/tests/models/glpn | 2 |
| ./transformers/tests/models/reformer | 2 |
| ./transformers/tests/models/audio_spectrogram_transformer | 2 |
| ./transformers/tests/models/speech_encoder_decoder | 2 |
| ./transformers/tests/models/siglip2 | 2 |
| ./transformers/tests/models/paligemma | 2 |
| ./transformers/tests/models/qwen2 | 2 |
| ./transformers/tests/models/univnet | 2 |
| ./transformers/tests/models/luke | 2 |
| ./transformers/tests/models/wav2vec2_bert | 2 |
| ./transformers/tests/models/splinter | 2 |
| ./transformers/tests/models/conditional_detr | 2 |
| ./transformers/tests/models/colpali | 2 |
| ./transformers/tests/models/layoutxlm | 2 |
| ./transformers/tests/models/plbart | 2 |
| ./transformers/tests/models/roc_bert | 2 |
| ./transformers/tests/models/align | 2 |
| ./transformers/tests/models/qwen2_audio | 2 |
| ./transformers/tests/models/nougat | 2 |
| ./transformers/tests/models/vitpose | 2 |
| ./transformers/tests/models/superpoint | 2 |
| ./transformers/tests/models/cvt | 2 |
| ./transformers/tests/models/bert_generation | 2 |
| ./transformers/tests/models/textnet | 2 |
| ./transformers/tests/models/altclip | 2 |
| ./transformers/tests/models/m2m_100 | 2 |
| ./transformers/tests/models/qwen2_5_vl | 2 |
| ./transformers/tests/models/levit | 2 |
| ./transformers/tests/models/tvp | 2 |
| ./transformers/tests/models/detr | 2 |
| ./transformers/tests/models/hubert | 2 |
| ./transformers/tests/models/longt5 | 2 |
| ./transformers/tests/models/fastspeech2_conformer | 2 |
| ./transformers/tests/models/aya_vision | 2 |
| ./transformers/tests/models/efficientnet | 2 |
| ./transformers/tests/models/vivit | 2 |
| ./transformers/tests/models/gpt_neox_japanese | 2 |
| ./transformers/tests/models/vitmatte | 2 |
| ./transformers/tests/models/canine | 2 |
| ./transformers/tests/models/vit_mae | 2 |
| ./transformers/tests/models/superglue | 2 |
| ./transformers/tests/models/vipllava | 2 |
| ./transformers/tests/models/perceiver | 2 |
| ./transformers/tests/models/mobilenet_v2 | 2 |
| ./transformers/tests/models/depth_pro | 2 |
| ./transformers/tests/models/dac | 2 |
| ./transformers/tests/models/seggpt | 2 |
| ./transformers/tests/models/swin2sr | 2 |
| ./transformers/tests/models/emu3 | 2 |
| ./transformers/tests/models/convbert | 2 |
| ./transformers/tests/models/cohere | 2 |
| ./transformers/tests/models/fnet | 2 |
| ./transformers/tests/models/poolformer | 2 |
| ./transformers/tests/models/encodec | 2 |
| ./transformers/tests/models/codegen | 2 |
| ./transformers/tests/models/clipseg | 2 |
| ./transformers/tests/models/convnextv2 | 2 |
| ./transformers/tests/models/swin | 2 |
| ./transformers/tests/models/imagegpt | 2 |
| ./transformers/tests/models/git | 2 |
| ./transformers/tests/models/bark | 2 |
| ./transformers/tests/models/gpt_neo | 2 |
| ./transformers/tests/models/kosmos2 | 2 |
| ./transformers/tests/models/biogpt | 2 |
| ./transformers/tests/models/timm_wrapper | 2 |
| ./transformers/tests/models/omdet_turbo | 2 |
| ./transformers/tests/models/mvp | 2 |
| ./transformers/tests/models/prophetnet | 2 |
| ./transformers/tests/models/fsmt | 2 |
| ./transformers/tests/models/pvt | 2 |
| ./transformers/tests/models/blip_2 | 2 |
| ./transformers/tests/models/dinov2 | 2 |
| ./transformers/tests/models/mobilenet_v1 | 2 |
| ./transformers/tests/models/cpmant | 2 |
| ./transformers/tests/models/mask2former | 2 |
| ./transformers/tests/models/musicgen | 2 |
| ./transformers/tests/models/squeezebert | 2 |
| ./transformers/tests/models/swiftformer | 2 |
| ./transformers/tests/models/video_llava | 2 |
| ./transformers/tests/models/deformable_detr | 2 |
| ./transformers/tests/models/groupvit | 2 |
| ./transformers/tests/optimization | 2 |
| ./transformers/tests/quantization/compressed_tensors | 2 |
| ./transformers/tests/quantization/bnb | 2 |
| ./transformers/tests/deepspeed | 2 |
| ./refactored_test_suite/hardware/webgpu | 2 |
| ./refactored_test_suite/tests/unit | 2 |
| ./refactored_test_suite/models/vision | 2 |
| ./refactored_test_suite/models/other | 2 |
| ./distributed_testing/ci | 2 |
| ./doc-builder/transformers-docs/transformers/tests/tokenization | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/vitmatte | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/swin2sr | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/longt5 | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/prophetnet | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/cvt | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/vipllava | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/hubert | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/align | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/gpt_neo | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/superglue | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/levit | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/mobilenet_v1 | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/tvp | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/clipseg | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/codegen | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/encodec | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/nougat | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/audio_spectrogram_transformer | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/perceiver | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/depth_pro | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/plbart | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/imagegpt | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/deformable_detr | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/detr | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/vit_mae | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/conditional_detr | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/convbert | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/timm_wrapper | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/speech_encoder_decoder | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/yolos | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/trocr | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/gpt_neox_japanese | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/moshi | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/qwen2_audio | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/mobilenet_v2 | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/luke | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/convnextv2 | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/qwen2 | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/musicgen | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/groupvit | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/kosmos2 | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/glpn | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/poolformer | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/vilt | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/vits | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/univnet | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/fastspeech2_conformer | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/dinov2 | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/efficientnet | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/cpmant | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/siglip2 | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/qwen2_5_vl | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/superpoint | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/videomae | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/zoedepth | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/colpali | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/mvp | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/bert_generation | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/pvt | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/aya_vision | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/paligemma | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/fsmt | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/biogpt | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/blip_2 | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/reformer | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/splinter | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/seggpt | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/fnet | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/vivit | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/bark | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/layoutxlm | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/swin | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/roc_bert | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/cohere | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/emu3 | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/omdet_turbo | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/textnet | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/squeezebert | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/swiftformer | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/vitpose | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/altclip | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/dac | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/video_llava | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/wav2vec2_bert | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/git | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/instructblip | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/m2m_100 | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/mask2former | 2 |
| ./doc-builder/transformers-docs/transformers/tests/models/canine | 2 |
| ./doc-builder/transformers-docs/transformers/tests/optimization | 2 |
| ./doc-builder/transformers-docs/transformers/tests/deepspeed | 2 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/bnb | 2 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/compressed_tensors | 2 |
| ./doc-builder/transformers-docs/transformers/examples/pytorch | 2 |
| ./duckdb_api/distributed_testing/dashboard/tests | 2 |
| ./api/llm_providers | 2 |
| ./test/api/huggingface | 1 |
| ./test/api/local_servers | 1 |
| ./test/integration/database | 1 |
| ./test/integration/distributed | 1 |
| ./test/hardware/webgpu/compute_shaders | 1 |
| ./test/hardware/cpu | 1 |
| ./test/models/text/gpt | 1 |
| ./test/models/audio | 1 |
| ./transformers/examples/flax | 1 |
| ./transformers/examples/tensorflow | 1 |
| ./transformers/tests/fsdp | 1 |
| ./transformers/tests/models/bros | 1 |
| ./transformers/tests/models/dab_detr | 1 |
| ./transformers/tests/models/phobert | 1 |
| ./transformers/tests/models/bert_japanese | 1 |
| ./transformers/tests/models/time_series_transformer | 1 |
| ./transformers/tests/models/sew | 1 |
| ./transformers/tests/models/switch_transformers | 1 |
| ./transformers/tests/models/herbert | 1 |
| ./transformers/tests/models/byt5 | 1 |
| ./transformers/tests/models/jetmoe | 1 |
| ./transformers/tests/models/xlm_roberta_xl | 1 |
| ./transformers/tests/models/informer | 1 |
| ./transformers/tests/models/glm | 1 |
| ./transformers/tests/models/olmoe | 1 |
| ./transformers/tests/models/depth_anything | 1 |
| ./transformers/tests/models/cpm | 1 |
| ./transformers/tests/models/wav2vec2_with_lm | 1 |
| ./transformers/tests/models/barthez | 1 |
| ./transformers/tests/models/ijepa | 1 |
| ./transformers/tests/models/nllb_moe | 1 |
| ./transformers/tests/models/wav2vec2_conformer | 1 |
| ./transformers/tests/models/falcon | 1 |
| ./transformers/tests/models/modernbert | 1 |
| ./transformers/tests/models/pvt_v2 | 1 |
| ./transformers/tests/models/olmo2 | 1 |
| ./transformers/tests/models/dinat | 1 |
| ./transformers/tests/models/umt5 | 1 |
| ./transformers/tests/models/diffllama | 1 |
| ./transformers/tests/models/ernie | 1 |
| ./transformers/tests/models/mixtral | 1 |
| ./transformers/tests/models/patchtsmixer | 1 |
| ./transformers/tests/models/visual_bert | 1 |
| ./transformers/tests/models/upernet | 1 |
| ./transformers/tests/models/mobilevitv2 | 1 |
| ./transformers/tests/models/qwen2_moe | 1 |
| ./transformers/tests/models/mbart50 | 1 |
| ./transformers/tests/models/focalnet | 1 |
| ./transformers/tests/models/bertweet | 1 |
| ./transformers/tests/models/megatron_bert | 1 |
| ./transformers/tests/models/stablelm | 1 |
| ./transformers/tests/models/granite | 1 |
| ./transformers/tests/models/pegasus_x | 1 |
| ./transformers/tests/models/mamba2 | 1 |
| ./transformers/tests/models/granitemoe | 1 |
| ./transformers/tests/models/patchtst | 1 |
| ./transformers/tests/models/bit | 1 |
| ./transformers/tests/models/zamba | 1 |
| ./transformers/tests/models/unispeech | 1 |
| ./transformers/tests/models/cohere2 | 1 |
| ./transformers/tests/models/persimmon | 1 |
| ./transformers/tests/models/bamba | 1 |
| ./transformers/tests/models/rwkv | 1 |
| ./transformers/tests/models/falcon_mamba | 1 |
| ./transformers/tests/models/zamba2 | 1 |
| ./transformers/tests/models/gpt_bigcode | 1 |
| ./transformers/tests/models/dbrx | 1 |
| ./transformers/tests/models/dinov2_with_registers | 1 |
| ./transformers/tests/models/bartpho | 1 |
| ./transformers/tests/models/phimoe | 1 |
| ./transformers/tests/models/nllb | 1 |
| ./transformers/tests/models/mimi | 1 |
| ./transformers/tests/models/megatron_gpt2 | 1 |
| ./transformers/tests/models/lilt | 1 |
| ./transformers/tests/models/wavlm | 1 |
| ./transformers/tests/models/wav2vec2_phoneme | 1 |
| ./transformers/tests/models/vit_msn | 1 |
| ./transformers/tests/models/starcoder2 | 1 |
| ./transformers/tests/models/vitpose_backbone | 1 |
| ./transformers/tests/models/gemma2 | 1 |
| ./transformers/tests/models/myt5 | 1 |
| ./transformers/tests/models/unispeech_sat | 1 |
| ./transformers/tests/models/phi3 | 1 |
| ./transformers/tests/models/table_transformer | 1 |
| ./transformers/tests/models/decision_transformer | 1 |
| ./transformers/tests/models/phi | 1 |
| ./transformers/tests/models/rt_detr_v2 | 1 |
| ./transformers/tests/models/bigbird_pegasus | 1 |
| ./transformers/tests/models/mpt | 1 |
| ./transformers/tests/models/swinv2 | 1 |
| ./transformers/tests/models/hiera | 1 |
| ./transformers/tests/models/seamless_m4t_v2 | 1 |
| ./transformers/tests/models/mluke | 1 |
| ./transformers/tests/models/jamba | 1 |
| ./transformers/tests/models/nemotron | 1 |
| ./transformers/tests/models/mamba | 1 |
| ./transformers/tests/models/mra | 1 |
| ./transformers/tests/models/gpt_neox | 1 |
| ./transformers/tests/models/timesformer | 1 |
| ./transformers/tests/models/gpt_sw3 | 1 |
| ./transformers/tests/models/xmod | 1 |
| ./transformers/tests/models/moonshine | 1 |
| ./transformers/tests/models/autoformer | 1 |
| ./transformers/tests/models/recurrent_gemma | 1 |
| ./transformers/tests/models/code_llama | 1 |
| ./transformers/tests/models/olmo | 1 |
| ./transformers/tests/models/timm_backbone | 1 |
| ./transformers/tests/models/x_clip | 1 |
| ./transformers/tests/models/ibert | 1 |
| ./transformers/tests/models/sew_d | 1 |
| ./transformers/tests/models/granitemoeshared | 1 |
| ./transformers/tests/models/nystromformer | 1 |
| ./transformers/tests/models/paligemma2 | 1 |
| ./transformers/tests/models/vitdet | 1 |
| ./transformers/tests/models/yoso | 1 |
| ./transformers/tests/models/dit | 1 |
| ./transformers/tests/models/helium | 1 |
| ./transformers/tests/peft_integration | 1 |
| ./transformers/tests/tensor_parallel | 1 |
| ./transformers/tests/extended | 1 |
| ./transformers/tests/quantization/aqlm_integration | 1 |
| ./transformers/tests/quantization/quanto_integration | 1 |
| ./transformers/tests/quantization/torchao_integration | 1 |
| ./transformers/tests/quantization/higgs | 1 |
| ./transformers/tests/quantization/vptq_integration | 1 |
| ./transformers/tests/quantization/hqq | 1 |
| ./transformers/tests/quantization/autoawq | 1 |
| ./transformers/tests/quantization/eetq_integration | 1 |
| ./transformers/tests/quantization/ggml | 1 |
| ./transformers/tests/quantization/fbgemm_fp8 | 1 |
| ./transformers/tests/quantization/gptq | 1 |
| ./transformers/tests/quantization/finegrained_fp8 | 1 |
| ./transformers/tests/quantization/spqr_integration | 1 |
| ./transformers/tests/quantization/bitnet_integration | 1 |
| ./transformers/tests/bettertransformer | 1 |
| ./transformers/tests/repo_utils/modular | 1 |
| ./skills/fixed_files_new | 1 |
| ./skills/examples | 1 |
| ./refactored_test_suite | 1 |
| ./refactored_test_suite/browser | 1 |
| ./distributed_testing/integration_tests | 1 |
| ./critical_priority_tests | 1 |
| ./generators/collected_results/bert-base-uncased/cuda/20250316_154613 | 1 |
| ./generators/collected_results/bert-base-uncased/cuda/20250316_152830 | 1 |
| ./generators/collected_results/bert-base-uncased/cuda/20250316_152713 | 1 |
| ./generators/runners/end_to_end/test_output/tester_integration_0e9caf50-75f1-402b-841c-a8ea2f7135da | 1 |
| ./generators/runners/end_to_end/test_output/tester_integration_448ff63d-fde2-4a64-91ab-a2a5a061dec9 | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/openai_clip-vit-base-patch32_webnn | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/vit-base-patch16-224_rocm | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/bert-base-uncased_rocm | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/gpt2_cpu | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/vit-base-patch16-224_webnn | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/whisper-tiny_webnn | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/vit-base-patch16-224_cuda | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/whisper-tiny_openvino | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/whisper-tiny_webgpu | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/bert-base-uncased_cuda | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/gpt2_qnn | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/bert-base-uncased_webgpu | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/whisper-tiny_mps | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/openai_clip-vit-base-patch32_openvino | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/whisper-tiny_cpu | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/gpt2_webnn | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/gpt2_openvino | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/whisper-tiny_qnn | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/gpt2_webgpu | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/gpt2_mps | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/whisper-tiny_cuda | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/openai_clip-vit-base-patch32_cpu | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/openai_clip-vit-base-patch32_cuda | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/vit-base-patch16-224_mps | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/whisper-tiny_rocm | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/bert-base-uncased_mps | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/bert-base-uncased_openvino | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/openai_clip-vit-base-patch32_qnn | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/bert-base-uncased_webnn | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/openai_clip-vit-base-patch32_rocm | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/vit-base-patch16-224_cpu | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/bert-base-uncased_cpu | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/openai_clip-vit-base-patch32_webgpu | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/gpt2_cuda | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/vit-base-patch16-224_webgpu | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/openai_clip-vit-base-patch32_mps | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/bert-base-uncased_qnn | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/vit-base-patch16-224_openvino | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/gpt2_rocm | 1 |
| ./generators/runners/end_to_end/test_output/enhanced_docs_test/vit-base-patch16-224_qnn | 1 |
| ./generators/runners/end_to_end/test_output/vision_webgpu_test/vit-base-patch16-224_webgpu | 1 |
| ./generators/runners/end_to_end/test_output/tester_integration_1dd4d5d5-b105-4cae-be71-12d20634e4dc | 1 |
| ./generators/runners/end_to_end/test_output/tester_integration_0e0f5805-454a-4840-a052-355e44b4619d | 1 |
| ./generators/runners/end_to_end/test_output/bert-base-uncased_cuda | 1 |
| ./generators/runners/end_to_end/test_output/tester_integration_6bd7ab89-613b-49b9-9c56-c399a619e56d | 1 |
| ./generators/runners/end_to_end/test_output/tester_integration_69d7bdde-626d-4a94-85d9-956ae84eec68 | 1 |
| ./generators/runners/end_to_end/test_output/tester_integration_3d02f537-6f05-4108-b30f-ba90f5ec943c | 1 |
| ./fixes | 1 |
| ./enhanced_templates | 1 |
| ./models/text/bert | 1 |
| ./apis | 1 |
| ./archive | 1 |
| ./integration/distributed | 1 |
| ./integration/browser | 1 |
| ./integration/database | 1 |
| ./fixed_web_tests | 1 |
| ./doc-builder/transformers-docs/transformers/tests/extended | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/code_llama | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/autoformer | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/mamba2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/recurrent_gemma | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/vit_msn | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/unispeech | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/persimmon | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/bigbird_pegasus | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/seamless_m4t_v2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/wav2vec2_with_lm | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/rt_detr_v2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/wav2vec2_conformer | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/cohere2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/paligemma2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/nystromformer | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/zamba | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/bartpho | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/bit | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/timm_backbone | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/falcon | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/nemotron | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/megatron_gpt2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/pvt_v2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/sew | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/gpt_neox | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/umt5 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/moonshine | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/visual_bert | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/switch_transformers | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/glm | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/ijepa | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/cpm | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/mobilevitv2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/olmoe | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/jetmoe | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/dinov2_with_registers | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/unispeech_sat | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/gpt_sw3 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/dinat | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/table_transformer | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/time_series_transformer | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/ernie | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/bros | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/olmo2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/falcon_mamba | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/byt5 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/starcoder2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/x_clip | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/hiera | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/phi | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/nllb_moe | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/mpt | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/yoso | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/mra | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/dab_detr | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/helium | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/olmo | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/jamba | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/informer | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/gpt_bigcode | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/mluke | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/vitdet | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/mamba | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/wav2vec2_phoneme | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/swinv2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/sew_d | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/ibert | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/phobert | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/vitpose_backbone | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/xmod | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/granitemoe | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/herbert | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/patchtsmixer | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/diffllama | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/barthez | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/dit | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/megatron_bert | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/granite | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/myt5 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/phi3 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/timesformer | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/focalnet | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/bamba | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/bertweet | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/wavlm | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/gemma2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/bert_japanese | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/qwen2_moe | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/mimi | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/zamba2 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/patchtst | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/nllb | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/pegasus_x | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/depth_anything | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/mixtral | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/lilt | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/upernet | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/granitemoeshared | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/rwkv | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/modernbert | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/decision_transformer | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/xlm_roberta_xl | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/dbrx | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/mbart50 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/phimoe | 1 |
| ./doc-builder/transformers-docs/transformers/tests/models/stablelm | 1 |
| ./doc-builder/transformers-docs/transformers/tests/peft_integration | 1 |
| ./doc-builder/transformers-docs/transformers/tests/bettertransformer | 1 |
| ./doc-builder/transformers-docs/transformers/tests/tensor_parallel | 1 |
| ./doc-builder/transformers-docs/transformers/tests/repo_utils/modular | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/autoawq | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/finegrained_fp8 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/bitnet_integration | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/spqr_integration | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/hqq | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/eetq_integration | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/fbgemm_fp8 | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/ggml | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/vptq_integration | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/torchao_integration | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/gptq | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/higgs | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/aqlm_integration | 1 |
| ./doc-builder/transformers-docs/transformers/tests/quantization/quanto_integration | 1 |
| ./doc-builder/transformers-docs/transformers/tests/fsdp | 1 |
| ./doc-builder/transformers-docs/transformers/examples/flax | 1 |
| ./doc-builder/transformers-docs/transformers/examples/tensorflow | 1 |
| ./duckdb_api/unified_test_results_db | 1 |
| ./duckdb_api/distributed_testing/result_aggregator/tests | 1 |
| ./fixed_web_platform/test | 1 |
| ./api | 1 |

### Test File Size Distribution

| Size Range | Count |
|------------|-------|
| < 1KB | 57 |
| 1-5KB | 296 |
| 5-10KB | 629 |
| 10-20KB | 681 |
| 20-50KB | 817 |
| > 50KB | 89 |

### Test Categories

| Category | Count |
|------------|-------|
| model | 968 |
| integration | 15 |
| web | 10 |
| hardware | 9 |
| browser | 4 |

### Class Structure

#### Most Common Test Class Names

| Class Name | Count |
|------------|-------|
| MockTokenizer | 110 |
| MockSentencePieceProcessor | 77 |
| TestModelTypeModels | 44 |
| MockImage | 30 |
| MockRequests | 30 |
| TestVitModels | 27 |
| Image | 24 |
| TestGpt2Models | 22 |
| TestT5Models | 21 |
| TestWhisperModels | 19 |

#### Most Common Base Classes

| Base Class | Count |
|------------|-------|
| unittest.TestCase | 3335 |
| ModelTesterMixin | 742 |
| PipelineTesterMixin | 655 |
| GenerationTesterMixin | 273 |
| TokenizerTesterMixin | 195 |
| TFModelTesterMixin | 163 |
| ImageProcessingTestMixin | 142 |
| ProcessorTesterMixin | 99 |
| FlaxModelTesterMixin | 79 |
| TestCasePlus | 64 |

#### Class Size Distribution

| Methods per Class | Count |
|-----------------|-------|
| 0 methods | 121 |
| 1 methods | 519 |
| 2 methods | 519 |
| 3 methods | 691 |
| 4 methods | 617 |
| 5 methods | 673 |
| 6 methods | 419 |
| 7 methods | 419 |
| 8 methods | 279 |
| 9 methods | 231 |
| 10 methods | 189 |
| 11 methods | 245 |
| 12 methods | 154 |
| 13 methods | 179 |
| 14 methods | 159 |
| 15 methods | 96 |
| 16 methods | 85 |
| 17 methods | 105 |
| 18 methods | 74 |
| 19 methods | 24 |
| 20 methods | 44 |
| 21 methods | 23 |
| 22 methods | 32 |
| 23 methods | 16 |
| 24 methods | 12 |
| 25 methods | 12 |
| 26 methods | 9 |
| 27 methods | 4 |
| 28 methods | 6 |
| 29 methods | 6 |
| 30 methods | 8 |
| 31 methods | 7 |
| 32 methods | 6 |
| 33 methods | 1 |
| 34 methods | 2 |
| 36 methods | 3 |
| 37 methods | 2 |
| 43 methods | 2 |
| 48 methods | 2 |
| 53 methods | 4 |
| 55 methods | 2 |
| 57 methods | 2 |
| 59 methods | 2 |
| 61 methods | 2 |
| 63 methods | 4 |
| 64 methods | 4 |
| 67 methods | 2 |
| 69 methods | 2 |
| 74 methods | 2 |
| 97 methods | 2 |
| 111 methods | 2 |
| 112 methods | 2 |

### Test Method Analysis

#### Most Common Test Method Names

| Method Name | Count |
|------------|-------|
| test_config | 853 |
| test_model_from_pretrained | 631 |
| test_model | 582 |
| test_inputs_embeds | 518 |
| test_pipeline | 462 |
| test_model_get_set_embeddings | 332 |
| test_initialization | 306 |
| test_hidden_states_output | 289 |
| test_forward_signature | 275 |
| test_retain_grad_hidden_states_attentions | 251 |
| test_training_gradient_checkpointing | 251 |
| test_training_gradient_checkpointing_use_reentrant | 227 |
| test_training_gradient_checkpointing_use_reentrant_false | 227 |
| test_from_pretrained | 213 |
| test_attention_outputs | 211 |

#### Test Method Size Distribution

| Size Range | Count |
|------------|-------|
| 1-5 lines | 10758 |
| 6-10 lines | 2787 |
| 11-20 lines | 5124 |
| 21-50 lines | 5662 |
| > 50 lines | 2351 |

#### Common Test Fixtures/Helpers

| Method Name | Count |
|------------|-------|
| setUp | 2039 |
| prepare_config_and_inputs | 1071 |
| prepare_config_and_inputs_for_common | 980 |
| tearDown | 401 |
| setUpClass | 183 |
| _prepare_for_class | 161 |
| prepare_image_inputs | 142 |
| prepare_image_processor_dict | 136 |
| prepare_config_and_inputs_for_decoder | 99 |
| test_setup | 49 |

#### Common Import Dependencies

| Module | Count |
|--------|-------|
| transformers | 18872 |
| unittest | 3257 |
| typing | 2441 |
| test_modeling_common | 1790 |
| torch | 1643 |
| generators | 1323 |
| os | 1244 |
| sys | 896 |
| json | 871 |
| numpy | 832 |
| tempfile | 704 |
| logging | 697 |
| test_configuration_common | 694 |
| pathlib | 645 |
| time | 635 |

## Major Test Clusters

### Modeling Tests

- test_modeling_xlm_roberta.py
- test_modeling_distilbert.py
- test_modeling_wav2vec2_bert.py
- test_modeling_tf_xlm_roberta.py
- test_modeling_tf_mobilebert.py
- test_modeling_tf_rembert.py
- test_modeling_flax_distilbert.py
- test_modeling_deberta.py
- test_modeling_roc_bert.py
- test_modeling_rembert.py
- test_modeling_hubert.py
- test_modeling_convbert.py
- test_modeling_deberta_v2.py
- test_modeling_ibert.py
- test_modeling_tf_hubert.py
- test_modeling_tf_deberta.py
- test_modeling_flax_albert.py
- test_modeling_squeezebert.py
- test_modeling_tf_camembert.py
- test_modeling_xlm_roberta_xl.py
- test_modeling_flax_bert.py
- test_modeling_tf_convbert.py
- test_modeling_flax_roberta.py
- test_modeling_modernbert.py
- test_modeling_tf_deberta_v2.py
- test_modeling_visual_bert.py
- test_modeling_bert.py
- test_modeling_camembert.py
- test_modeling_bert_generation.py
- test_modeling_tf_roberta.py
- test_modeling_flaubert.py
- test_modeling_albert.py
- test_modeling_tf_distilbert.py
- test_modeling_tf_albert.py
- test_modeling_tf_bert.py
- test_modeling_roberta.py
- test_modeling_megatron_bert.py
- test_modeling_tf_roberta_prelayernorm.py
- test_modeling_flax_xlm_roberta.py
- test_modeling_flax_roberta_prelayernorm.py
- test_modeling_roberta_prelayernorm.py
- test_modeling_tf_flaubert.py
- test_modeling_mobilebert.py
- test_modeling_tf_common.py
- test_modeling_flax_common.py
- test_modeling_common.py
- test_modeling_tf_utils.py
- test_modeling_utils.py
- test_modeling_tf_core.py
- test_modeling_flax_utils.py
- test_modeling_rope_utils.py
- test_modeling_pix2struct.py
- test_modeling_bros.py
- test_modeling_dab_detr.py
- test_modeling_gemma3.py
- test_modeling_clvp.py
- test_modeling_vilt.py
- test_modeling_time_series_transformer.py
- test_modeling_sew.py
- test_modeling_zoedepth.py
- test_modeling_videomae.py
- test_modeling_moshi.py
- test_modeling_switch_transformers.py
- test_modeling_vits.py
- test_modeling_trocr.py
- test_modeling_instructblip.py
- test_modeling_yolos.py
- test_modeling_fuyu.py
- test_modeling_udop.py
- test_modeling_jetmoe.py
- test_modeling_xlm_roberta_xl.py
- test_modeling_flax_bart.py
- test_modeling_bart.py
- test_modeling_tf_bart.py
- test_modeling_informer.py
- test_modeling_tf_gpt2.py
- test_modeling_gpt2.py
- test_modeling_flax_gpt2.py
- test_modeling_flax_clip.py
- test_modeling_clip.py
- test_modeling_tf_clip.py
- test_modeling_flax_vision_encoder_decoder.py
- test_modeling_vision_encoder_decoder.py
- test_modeling_tf_vision_encoder_decoder.py
- test_modeling_dpr.py
- test_modeling_tf_dpr.py
- test_modeling_resnet.py
- test_modeling_flax_resnet.py
- test_modeling_tf_resnet.py
- test_modeling_tf_blip_text.py
- test_modeling_tf_blip.py
- test_modeling_blip.py
- test_modeling_blip_text.py
- test_modeling_tf_xlm.py
- test_modeling_xlm.py
- test_modeling_qwen2_vl.py
- test_modeling_glm.py
- test_modeling_olmoe.py
- test_modeling_depth_anything.py
- test_modeling_glpn.py
- test_modeling_maskformer.py
- test_modeling_maskformer_swin.py
- test_modeling_tf_layoutlm.py
- test_modeling_layoutlm.py
- test_modeling_ijepa.py
- test_modeling_nllb_moe.py
- test_modeling_clap.py
- test_modeling_reformer.py
- test_modeling_flax_marian.py
- test_modeling_marian.py
- test_modeling_tf_marian.py
- test_modeling_tf_lxmert.py
- test_modeling_lxmert.py
- test_modeling_audio_spectrogram_transformer.py
- test_modeling_flax_speech_encoder_decoder.py
- test_modeling_speech_encoder_decoder.py
- test_modeling_flaubert.py
- test_modeling_tf_flaubert.py
- test_modeling_siglip2.py
- test_modeling_wav2vec2_conformer.py
- test_modeling_paligemma.py
- test_modeling_idefics2.py
- test_modeling_qwen2.py
- test_modeling_tf_longformer.py
- test_modeling_longformer.py
- test_modeling_falcon.py
- test_modeling_univnet.py
- test_modeling_modernbert.py
- test_modeling_siglip.py
- test_modeling_regnet.py
- test_modeling_flax_regnet.py
- test_modeling_tf_regnet.py
- test_modeling_luke.py
- test_modeling_wav2vec2_bert.py
- test_modeling_splinter.py
- test_modeling_pvt_v2.py
- test_modeling_olmo2.py
- test_modeling_conditional_detr.py
- test_modeling_xlm_roberta.py
- test_modeling_flax_xlm_roberta.py
- test_modeling_tf_xlm_roberta.py
- test_modeling_mgp_str.py
- test_modeling_speecht5.py
- test_modeling_dinat.py
- test_modeling_umt5.py
- test_modeling_colpali.py
- test_modeling_beit.py
- test_modeling_flax_beit.py
- test_modeling_tf_led.py
- test_modeling_led.py
- test_modeling_diffllama.py
- test_modeling_roberta_prelayernorm.py
- test_modeling_flax_roberta_prelayernorm.py
- test_modeling_tf_roberta_prelayernorm.py
- test_modeling_ernie.py
- test_modeling_flax_wav2vec2.py
- test_modeling_wav2vec2.py
- test_modeling_tf_wav2vec2.py
- test_modeling_mixtral.py
- test_modeling_patchtsmixer.py
- test_modeling_visual_bert.py
- test_modeling_tf_deberta_v2.py
- test_modeling_deberta_v2.py
- test_modeling_plbart.py
- test_modeling_tf_deberta.py
- test_modeling_deberta.py
- test_modeling_idefics.py
- test_modeling_tf_idefics.py
- test_modeling_smolvlm.py
- test_modeling_roc_bert.py
- test_modeling_align.py
- test_modeling_qwen2_audio.py
- test_modeling_upernet.py
- test_modeling_vitpose.py
- test_modeling_superpoint.py
- test_modeling_tf_cvt.py
- test_modeling_cvt.py
- test_modeling_t5.py
- test_modeling_flax_t5.py
- test_modeling_tf_t5.py
- test_modeling_bert_generation.py
- test_modeling_textnet.py
- test_modeling_mpnet.py
- test_modeling_tf_mpnet.py
- test_modeling_altclip.py
- test_modeling_mobilevitv2.py
- test_modeling_tf_mobilebert.py
- test_modeling_mobilebert.py
- test_modeling_vit.py
- test_modeling_flax_vit.py
- test_modeling_tf_vit.py
- test_modeling_qwen2_moe.py
- test_modeling_markuplm.py
- test_modeling_m2m_100.py
- test_modeling_qwen2_5_vl.py
- test_modeling_mllama.py
- test_modeling_llava_next.py
- test_modeling_levit.py
- test_modeling_focalnet.py
- test_modeling_tvp.py
- test_modeling_musicgen_melody.py
- test_modeling_detr.py
- test_modeling_flax_distilbert.py
- test_modeling_tf_distilbert.py
- test_modeling_distilbert.py
- test_modeling_flax_gemma.py
- test_modeling_gemma.py
- test_modeling_pop2piano.py
- test_modeling_data2vec_vision.py
- test_modeling_data2vec_audio.py
- test_modeling_tf_data2vec_vision.py
- test_modeling_data2vec_text.py
- test_modeling_whisper.py
- test_modeling_tf_whisper.py
- test_modeling_flax_whisper.py
- test_modeling_hubert.py
- test_modeling_tf_hubert.py
- test_modeling_esm.py
- test_modeling_esmfold.py
- test_modeling_tf_esm.py
- test_modeling_megatron_bert.py
- test_modeling_encoder_decoder.py
- test_modeling_flax_encoder_decoder.py
- test_modeling_tf_encoder_decoder.py
- test_modeling_stablelm.py
- test_modeling_granite.py
- test_modeling_flax_longt5.py
- test_modeling_longt5.py
- test_modeling_pegasus_x.py
- test_modeling_mamba2.py
- test_modeling_fastspeech2_conformer.py
- test_modeling_tf_deit.py
- test_modeling_deit.py
- test_modeling_seamless_m4t.py
- test_modeling_got_ocr2.py
- test_modeling_tf_xglm.py
- test_modeling_xglm.py
- test_modeling_flax_xglm.py
- test_modeling_granitemoe.py
- test_modeling_bloom.py
- test_modeling_flax_bloom.py
- test_modeling_layoutlmv3.py
- test_modeling_tf_layoutlmv3.py
- test_modeling_patchtst.py
- test_modeling_bit.py
- test_modeling_aya_vision.py
- test_modeling_flax_pegasus.py
- test_modeling_pegasus.py
- test_modeling_tf_pegasus.py
- test_modeling_tapas.py
- test_modeling_tf_tapas.py
- test_modeling_efficientnet.py
- test_modeling_llava.py
- test_modeling_zamba.py
- test_modeling_unispeech.py
- test_modeling_cohere2.py
- test_modeling_llava_next_video.py
- test_modeling_vivit.py
- test_modeling_persimmon.py
- test_modeling_bamba.py
- test_modeling_gpt_neox_japanese.py
- test_modeling_tf_rag.py
- test_modeling_rag.py
- test_modeling_rwkv.py
- test_modeling_bridgetower.py
- test_modeling_falcon_mamba.py
- test_modeling_zamba2.py
- test_modeling_flax_electra.py
- test_modeling_tf_electra.py
- test_modeling_electra.py
- test_modeling_tf_xlnet.py
- test_modeling_xlnet.py
- test_modeling_tf_ctrl.py
- test_modeling_ctrl.py
- test_modeling_gpt_bigcode.py
- test_modeling_dbrx.py
- test_modeling_albert.py
- test_modeling_flax_albert.py
- test_modeling_tf_albert.py
- test_modeling_tf_openai.py
- test_modeling_openai.py
- test_modeling_convnext.py
- test_modeling_tf_convnext.py
- test_modeling_dinov2_with_registers.py
- test_modeling_donut_swin.py
- test_modeling_idefics3.py
- test_modeling_phimoe.py
- test_modeling_flava.py
- test_modeling_vitmatte.py
- test_modeling_canine.py
- test_modeling_mimi.py
- test_modeling_owlvit.py
- test_modeling_tf_vit_mae.py
- test_modeling_vit_mae.py
- test_modeling_megatron_gpt2.py
- test_modeling_lilt.py
- test_modeling_superglue.py
- test_modeling_wavlm.py
- test_modeling_vipllava.py
- test_modeling_dpt_hybrid.py
- test_modeling_dpt.py
- test_modeling_dpt_auto_backbone.py
- test_modeling_tf_rembert.py
- test_modeling_rembert.py
- test_modeling_vit_msn.py
- test_modeling_perceiver.py
- test_modeling_mobilenet_v2.py
- test_modeling_starcoder2.py
- test_modeling_depth_pro.py
- test_modeling_auto.py
- test_modeling_tf_auto.py
- test_modeling_flax_auto.py
- test_modeling_dac.py
- test_modeling_seggpt.py
- test_modeling_swin2sr.py
- test_modeling_vitpose_backbone.py
- test_modeling_emu3.py
- test_modeling_tf_convbert.py
- test_modeling_convbert.py
- test_modeling_chinese_clip.py
- test_modeling_cohere.py
- test_modeling_fnet.py
- test_modeling_gemma2.py
- test_modeling_flax_opt.py
- test_modeling_tf_opt.py
- test_modeling_opt.py
- test_modeling_poolformer.py
- test_modeling_encodec.py
- test_modeling_unispeech_sat.py
- test_modeling_llama.py
- test_modeling_flax_llama.py
- test_modeling_codegen.py
- test_modeling_clipseg.py
- test_modeling_chameleon.py
- test_modeling_tf_convnextv2.py
- test_modeling_convnextv2.py
- test_modeling_tf_funnel.py
- test_modeling_funnel.py
- test_modeling_tf_swin.py
- test_modeling_swin.py
- test_modeling_phi3.py
- test_modeling_imagegpt.py
- test_modeling_sam.py
- test_modeling_tf_sam.py
- test_modeling_git.py
- test_modeling_bark.py
- test_modeling_gpt_neo.py
- test_modeling_flax_gpt_neo.py
- test_modeling_table_transformer.py
- test_modeling_kosmos2.py
- test_modeling_flax_vision_text_dual_encoder.py
- test_modeling_tf_vision_text_dual_encoder.py
- test_modeling_vision_text_dual_encoder.py
- test_modeling_biogpt.py
- test_modeling_decision_transformer.py
- test_modeling_layoutlmv2.py
- test_modeling_phi.py
- test_modeling_rt_detr_v2.py
- test_modeling_llava_onevision.py
- test_modeling_bigbird_pegasus.py
- test_modeling_aria.py
- test_modeling_timm_wrapper.py
- test_modeling_mpt.py
- test_modeling_omdet_turbo.py
- test_modeling_swinv2.py
- test_modeling_mobilevit.py
- test_modeling_tf_mobilevit.py
- test_modeling_hiera.py
- test_modeling_seamless_m4t_v2.py
- test_modeling_jamba.py
- test_modeling_nemotron.py
- test_modeling_mamba.py
- test_modeling_mra.py
- test_modeling_segformer.py
- test_modeling_tf_segformer.py
- test_modeling_big_bird.py
- test_modeling_flax_big_bird.py
- test_modeling_gpt_neox.py
- test_modeling_mvp.py
- test_modeling_speech_to_text.py
- test_modeling_tf_speech_to_text.py
- test_modeling_timesformer.py
- test_modeling_xmod.py
- test_modeling_mt5.py
- test_modeling_tf_mt5.py
- test_modeling_flax_mt5.py
- test_modeling_prophetnet.py
- test_modeling_fsmt.py
- test_modeling_moonshine.py
- test_modeling_blenderbot.py
- test_modeling_tf_blenderbot.py
- test_modeling_flax_blenderbot.py
- test_modeling_pvt.py
- test_modeling_blip_2.py
- test_modeling_autoformer.py
- test_modeling_recurrent_gemma.py
- test_modeling_dinov2.py
- test_modeling_flax_dinov2.py
- test_modeling_gptj.py
- test_modeling_tf_gptj.py
- test_modeling_flax_gptj.py
- test_modeling_roformer.py
- test_modeling_flax_roformer.py
- test_modeling_tf_roformer.py
- test_modeling_olmo.py
- test_modeling_mobilenet_v1.py
- test_modeling_timm_backbone.py
- test_modeling_cpmant.py
- test_modeling_rt_detr.py
- test_modeling_rt_detr_resnet.py
- test_modeling_tf_blenderbot_small.py
- test_modeling_flax_blenderbot_small.py
- test_modeling_blenderbot_small.py
- test_modeling_mask2former.py
- test_modeling_instructblipvideo.py
- test_modeling_x_clip.py
- test_modeling_owlv2.py
- test_modeling_ibert.py
- test_modeling_sew_d.py
- test_modeling_granitemoeshared.py
- test_modeling_nystromformer.py
- test_modeling_tf_mbart.py
- test_modeling_mbart.py
- test_modeling_flax_mbart.py
- test_modeling_flax_bert.py
- test_modeling_bert.py
- test_modeling_tf_bert.py
- test_modeling_musicgen.py
- test_modeling_flax_mistral.py
- test_modeling_tf_mistral.py
- test_modeling_mistral.py
- test_modeling_grounding_dino.py
- test_modeling_paligemma2.py
- test_modeling_tf_camembert.py
- test_modeling_camembert.py
- test_modeling_vitdet.py
- test_modeling_pixtral.py
- test_modeling_oneformer.py
- test_modeling_squeezebert.py
- test_modeling_swiftformer.py
- test_modeling_tf_swiftformer.py
- test_modeling_yoso.py
- test_modeling_video_llava.py
- test_modeling_deformable_detr.py
- test_modeling_dit.py
- test_modeling_tf_groupvit.py
- test_modeling_groupvit.py
- test_modeling_tf_roberta.py
- test_modeling_roberta.py
- test_modeling_flax_roberta.py
- test_modeling_helium.py
- test_modeling_tf_common.py
- test_modeling_common.py
- test_modeling_flax_common.py
- test_modeling_vitmatte.py
- test_modeling_xglm.py
- test_modeling_tf_xglm.py
- test_modeling_flax_xglm.py
- test_modeling_swin2sr.py
- test_modeling_idefics3.py
- test_modeling_longt5.py
- test_modeling_flax_longt5.py
- test_modeling_autoformer.py
- test_modeling_flax_blenderbot.py
- test_modeling_blenderbot.py
- test_modeling_tf_blenderbot.py
- test_modeling_mamba2.py
- test_modeling_prophetnet.py
- test_modeling_tf_mpnet.py
- test_modeling_mpnet.py
- test_modeling_tf_deit.py
- test_modeling_deit.py
- test_modeling_recurrent_gemma.py
- test_modeling_cvt.py
- test_modeling_tf_cvt.py
- test_modeling_vipllava.py
- test_modeling_vit_msn.py
- test_modeling_hubert.py
- test_modeling_tf_hubert.py
- test_modeling_tf_rembert.py
- test_modeling_rembert.py
- test_modeling_align.py
- test_modeling_unispeech.py
- test_modeling_persimmon.py
- test_modeling_grounding_dino.py
- test_modeling_flax_roberta_prelayernorm.py
- test_modeling_tf_roberta_prelayernorm.py
- test_modeling_roberta_prelayernorm.py
- test_modeling_gpt_neo.py
- test_modeling_flax_gpt_neo.py
- test_modeling_superglue.py
- test_modeling_bigbird_pegasus.py
- test_modeling_llava_onevision.py
- test_modeling_tf_vit.py
- test_modeling_flax_vit.py
- test_modeling_vit.py
- test_modeling_levit.py
- test_modeling_mobilenet_v1.py
- test_modeling_flax_vision_encoder_decoder.py
- test_modeling_vision_encoder_decoder.py
- test_modeling_tf_vision_encoder_decoder.py
- test_modeling_convnext.py
- test_modeling_tf_convnext.py
- test_modeling_seamless_m4t_v2.py
- test_modeling_tvp.py
- test_modeling_clipseg.py
- test_modeling_codegen.py
- test_modeling_encodec.py
- test_modeling_layoutlmv2.py
- test_modeling_rt_detr_v2.py
- test_modeling_xlm_roberta.py
- test_modeling_tf_xlm_roberta.py
- test_modeling_flax_xlm_roberta.py
- test_modeling_speech_to_text.py
- test_modeling_tf_speech_to_text.py
- test_modeling_flax_electra.py
- test_modeling_electra.py
- test_modeling_tf_electra.py
- test_modeling_wav2vec2_conformer.py
- test_modeling_esm.py
- test_modeling_esmfold.py
- test_modeling_tf_esm.py
- test_modeling_instructblipvideo.py
- test_modeling_audio_spectrogram_transformer.py
- test_modeling_perceiver.py
- test_modeling_cohere2.py
- test_modeling_depth_pro.py
- test_modeling_plbart.py
- test_modeling_flax_gemma.py
- test_modeling_gemma.py
- test_modeling_imagegpt.py
- test_modeling_deformable_detr.py
- test_modeling_detr.py
- test_modeling_paligemma2.py
- test_modeling_tf_tapas.py
- test_modeling_tapas.py
- test_modeling_llava.py
- test_modeling_nystromformer.py
- test_modeling_tf_vit_mae.py
- test_modeling_vit_mae.py
- test_modeling_zamba.py
- test_modeling_chameleon.py
- test_modeling_conditional_detr.py
- test_modeling_tf_convbert.py
- test_modeling_convbert.py
- test_modeling_bit.py
- test_modeling_timm_wrapper.py
- test_modeling_timm_backbone.py
- test_modeling_flax_bloom.py
- test_modeling_bloom.py
- test_modeling_mllama.py
- test_modeling_speech_encoder_decoder.py
- test_modeling_flax_speech_encoder_decoder.py
- test_modeling_yolos.py
- test_modeling_chinese_clip.py
- test_modeling_falcon.py
- test_modeling_siglip.py
- test_modeling_trocr.py
- test_modeling_tf_mobilevit.py
- test_modeling_mobilevit.py
- test_modeling_regnet.py
- test_modeling_tf_regnet.py
- test_modeling_flax_regnet.py
- test_modeling_nemotron.py
- test_modeling_megatron_gpt2.py
- test_modeling_pvt_v2.py
- test_modeling_sew.py
- test_modeling_lxmert.py
- test_modeling_tf_lxmert.py
- test_modeling_tf_marian.py
- test_modeling_marian.py
- test_modeling_flax_marian.py
- test_modeling_gpt_neox_japanese.py
- test_modeling_beit.py
- test_modeling_flax_beit.py
- test_modeling_rt_detr.py
- test_modeling_rt_detr_resnet.py
- test_modeling_gpt_neox.py
- test_modeling_flax_big_bird.py
- test_modeling_big_bird.py
- test_modeling_umt5.py
- test_modeling_moshi.py
- test_modeling_moonshine.py
- test_modeling_qwen2_audio.py
- test_modeling_tf_dpr.py
- test_modeling_dpr.py
- test_modeling_visual_bert.py
- test_modeling_mobilenet_v2.py
- test_modeling_flax_roberta.py
- test_modeling_tf_roberta.py
- test_modeling_roberta.py
- test_modeling_switch_transformers.py
- test_modeling_tf_segformer.py
- test_modeling_segformer.py
- test_modeling_glm.py
- test_modeling_ijepa.py
- test_modeling_xlm.py
- test_modeling_tf_xlm.py
- test_modeling_mistral.py
- test_modeling_tf_mistral.py
- test_modeling_flax_mistral.py
- test_modeling_mobilevitv2.py
- test_modeling_luke.py
- test_modeling_olmoe.py
- test_modeling_pixtral.py
- test_modeling_tf_convnextv2.py
- test_modeling_convnextv2.py
- test_modeling_qwen2.py
- test_modeling_musicgen.py
- test_modeling_jetmoe.py
- test_modeling_blip.py
- test_modeling_tf_blip_text.py
- test_modeling_blip_text.py
- test_modeling_tf_blip.py
- test_modeling_dinov2_with_registers.py
- test_modeling_clap.py
- test_modeling_tf_groupvit.py
- test_modeling_groupvit.py
- test_modeling_unispeech_sat.py
- test_modeling_gemma3.py
- test_modeling_kosmos2.py
- test_modeling_glpn.py
- test_modeling_poolformer.py
- test_modeling_dinat.py
- test_modeling_vilt.py
- test_modeling_table_transformer.py
- test_modeling_clvp.py
- test_modeling_time_series_transformer.py
- test_modeling_ernie.py
- test_modeling_roformer.py
- test_modeling_flax_roformer.py
- test_modeling_tf_roformer.py
- test_modeling_bros.py
- test_modeling_clip.py
- test_modeling_tf_clip.py
- test_modeling_flax_clip.py
- test_modeling_olmo2.py
- test_modeling_flax_resnet.py
- test_modeling_resnet.py
- test_modeling_tf_resnet.py
- test_modeling_camembert.py
- test_modeling_tf_camembert.py
- test_modeling_falcon_mamba.py
- test_modeling_flax_gpt2.py
- test_modeling_tf_gpt2.py
- test_modeling_gpt2.py
- test_modeling_fuyu.py
- test_modeling_udop.py
- test_modeling_flax_bart.py
- test_modeling_tf_bart.py
- test_modeling_bart.py
- test_modeling_oneformer.py
- test_modeling_starcoder2.py
- test_modeling_tf_led.py
- test_modeling_led.py
- test_modeling_vits.py
- test_modeling_seamless_m4t.py
- test_modeling_qwen2_vl.py
- test_modeling_llava_next_video.py
- test_modeling_univnet.py
- test_modeling_x_clip.py
- test_modeling_distilbert.py
- test_modeling_flax_distilbert.py
- test_modeling_tf_distilbert.py
- test_modeling_hiera.py
- test_modeling_bert.py
- test_modeling_flax_bert.py
- test_modeling_tf_bert.py
- test_modeling_phi.py
- test_modeling_fastspeech2_conformer.py
- test_modeling_nllb_moe.py
- test_modeling_flax_dinov2.py
- test_modeling_dinov2.py
- test_modeling_mpt.py
- test_modeling_efficientnet.py
- test_modeling_tf_layoutlm.py
- test_modeling_layoutlm.py
- test_modeling_mobilebert.py
- test_modeling_tf_mobilebert.py
- test_modeling_llava_next.py
- test_modeling_cpmant.py
- test_modeling_siglip2.py
- test_modeling_qwen2_5_vl.py
- test_modeling_yoso.py
- test_modeling_musicgen_melody.py
- test_modeling_superpoint.py
- test_modeling_bridgetower.py
- test_modeling_videomae.py
- test_modeling_zoedepth.py
- test_modeling_mra.py
- test_modeling_dab_detr.py
- test_modeling_helium.py
- test_modeling_flax_gptj.py
- test_modeling_gptj.py
- test_modeling_tf_gptj.py
- test_modeling_colpali.py
- test_modeling_olmo.py
- test_modeling_jamba.py
- test_modeling_informer.py
- test_modeling_tf_deberta_v2.py
- test_modeling_deberta_v2.py
- test_modeling_gpt_bigcode.py
- test_modeling_mvp.py
- test_modeling_bert_generation.py
- test_modeling_pvt.py
- test_modeling_mgp_str.py
- test_modeling_vitdet.py
- test_modeling_mamba.py
- test_modeling_tf_mt5.py
- test_modeling_flax_mt5.py
- test_modeling_mt5.py
- test_modeling_aya_vision.py
- test_modeling_swinv2.py
- test_modeling_sew_d.py
- test_modeling_ibert.py
- test_modeling_owlv2.py
- test_modeling_wav2vec2.py
- test_modeling_flax_wav2vec2.py
- test_modeling_tf_wav2vec2.py
- test_modeling_paligemma.py
- test_modeling_fsmt.py
- test_modeling_biogpt.py
- test_modeling_layoutlmv3.py
- test_modeling_tf_layoutlmv3.py
- test_modeling_vitpose_backbone.py
- test_modeling_xmod.py
- test_modeling_granitemoe.py
- test_modeling_speecht5.py
- test_modeling_tf_flaubert.py
- test_modeling_flaubert.py
- test_modeling_aria.py
- test_modeling_flax_encoder_decoder.py
- test_modeling_encoder_decoder.py
- test_modeling_tf_encoder_decoder.py
- test_modeling_idefics2.py
- test_modeling_patchtsmixer.py
- test_modeling_diffllama.py
- test_modeling_blip_2.py
- test_modeling_reformer.py
- test_modeling_dit.py
- test_modeling_splinter.py
- test_modeling_megatron_bert.py
- test_modeling_mbart.py
- test_modeling_tf_mbart.py
- test_modeling_flax_mbart.py
- test_modeling_flax_vision_text_dual_encoder.py
- test_modeling_vision_text_dual_encoder.py
- test_modeling_tf_vision_text_dual_encoder.py
- test_modeling_tf_xlnet.py
- test_modeling_xlnet.py
- test_modeling_seggpt.py
- test_modeling_granite.py
- test_modeling_markuplm.py
- test_modeling_rag.py
- test_modeling_tf_rag.py
- test_modeling_fnet.py
- test_modeling_maskformer.py
- test_modeling_maskformer_swin.py
- test_modeling_whisper.py
- test_modeling_tf_whisper.py
- test_modeling_flax_whisper.py
- test_modeling_vivit.py
- test_modeling_bark.py
- test_modeling_phi3.py
- test_modeling_timesformer.py
- test_modeling_focalnet.py
- test_modeling_bamba.py
- test_modeling_swin.py
- test_modeling_tf_swin.py
- test_modeling_tf_funnel.py
- test_modeling_funnel.py
- test_modeling_flax_auto.py
- test_modeling_auto.py
- test_modeling_tf_auto.py
- test_modeling_roc_bert.py
- test_modeling_tf_pegasus.py
- test_modeling_pegasus.py
- test_modeling_flax_pegasus.py
- test_modeling_wavlm.py
- test_modeling_cohere.py
- test_modeling_emu3.py
- test_modeling_gemma2.py
- test_modeling_pix2struct.py
- test_modeling_dpt.py
- test_modeling_dpt_hybrid.py
- test_modeling_dpt_auto_backbone.py
- test_modeling_omdet_turbo.py
- test_modeling_flava.py
- test_modeling_donut_swin.py
- test_modeling_qwen2_moe.py
- test_modeling_textnet.py
- test_modeling_squeezebert.py
- test_modeling_mimi.py
- test_modeling_zamba2.py
- test_modeling_tf_swiftformer.py
- test_modeling_swiftformer.py
- test_modeling_patchtst.py
- test_modeling_pegasus_x.py
- test_modeling_vitpose.py
- test_modeling_depth_anything.py
- test_modeling_got_ocr2.py
- test_modeling_flax_albert.py
- test_modeling_albert.py
- test_modeling_tf_albert.py
- test_modeling_altclip.py
- test_modeling_tf_openai.py
- test_modeling_openai.py
- test_modeling_tf_blenderbot_small.py
- test_modeling_blenderbot_small.py
- test_modeling_flax_blenderbot_small.py
- test_modeling_tf_deberta.py
- test_modeling_deberta.py
- test_modeling_dac.py
- test_modeling_mixtral.py
- test_modeling_lilt.py
- test_modeling_pop2piano.py
- test_modeling_video_llava.py
- test_modeling_upernet.py
- test_modeling_smolvlm.py
- test_modeling_idefics.py
- test_modeling_tf_idefics.py
- test_modeling_granitemoeshared.py
- test_modeling_wav2vec2_bert.py
- test_modeling_rwkv.py
- test_modeling_modernbert.py
- test_modeling_longformer.py
- test_modeling_tf_longformer.py
- test_modeling_data2vec_text.py
- test_modeling_data2vec_vision.py
- test_modeling_data2vec_audio.py
- test_modeling_tf_data2vec_vision.py
- test_modeling_decision_transformer.py
- test_modeling_xlm_roberta_xl.py
- test_modeling_opt.py
- test_modeling_flax_opt.py
- test_modeling_tf_opt.py
- test_modeling_tf_t5.py
- test_modeling_flax_t5.py
- test_modeling_t5.py
- test_modeling_tf_sam.py
- test_modeling_sam.py
- test_modeling_git.py
- test_modeling_tf_ctrl.py
- test_modeling_ctrl.py
- test_modeling_instructblip.py
- test_modeling_dbrx.py
- test_modeling_flax_llama.py
- test_modeling_llama.py
- test_modeling_phimoe.py
- test_modeling_m2m_100.py
- test_modeling_stablelm.py
- test_modeling_mask2former.py
- test_modeling_canine.py
- test_modeling_owlvit.py
- test_modeling_utils.py
- test_modeling_tf_core.py
- test_modeling_flax_utils.py
- test_modeling_tf_utils.py
- test_modeling_rope_utils.py

### Hf Tests

- test_hf_pythia.py
- test_hf_wavlm.py
- test_hf_speech_to_text.py
- test_hf_unispeech.py
- test_hf_layoutlm.py
- test_hf_xlm.py
- test_hf_swin.py
- test_hf_swinv2.py
- test_hf_roformer.py
- test_hf_mistral.py
- test_hf_sew.py
- test_hf_umt5.py
- test_hf_plbart.py
- test_hf_resnet.py
- test_hf_qwen2.py
- test_hf_xglm.py
- test_hf_vilt.py
- test_hf_cvt.py
- test_hf_camembert.py
- test_hf_stablelm.py
- test_hf_deberta.py
- test_hf_qwen3.py
- test_hf_deberta_v2.py
- test_hf_groupvit.py
- test_hf_levit.py
- test_hf_bark.py
- test_hf_nllb.py
- test_hf_luke.py
- test_hf_distilbert.py
- test_hf_albert.py
- test_hf_t5.py
- test_hf_flava.py
- test_hf_bigbird_pegasus.py
- test_hf_mpnet.py
- test_hf_m2m_100.py
- test_hf_codellama.py
- test_hf_data2vec_audio.py
- test_hf_deit.py
- test_hf_mixtral.py
- test_hf_chinese_clip.py
- test_hf_blip.py
- test_hf_falcon.py
- test_hf_paligemma.py
- test_hf_mpt.py
- test_hf_flan_t5.py
- test_hf_speecht5.py
- test_hf_wav2vec2.py
- test_hf_gpt2.py
- test_hf_video_llava.py
- test_hf_whisper.py
- test_hf_bert.py
- test_hf_clip.py
- test_hf_perceiver.py
- test_hf_musicgen.py
- test_hf_audioldm2.py
- test_hf_funnel.py
- test_hf_gpt_neo.py
- test_hf_mobilenet_v2.py
- test_hf_open_llama.py
- test_hf_kosmos_2.py
- test_hf_mosaic_mpt.py
- test_hf_instruct_blip.py
- test_hf_bart.py
- test_hf_led.py
- test_hf_flaubert.py
- test_hf_phi.py
- test_hf_siglip.py
- test_hf_poolformer.py
- test_hf_olmo.py
- test_hf_efficientnet.py
- test_hf_blip_2.py
- test_hf_owlvit.py
- test_hf_llama.py
- test_hf_git.py
- test_hf_xlnet.py
- test_hf_convnextv2.py
- test_hf_pegasus_x.py
- test_hf_codegen.py
- test_hf_roberta.py
- test_hf_longt5.py
- test_hf_gpt_neox.py
- test_hf_fuyu.py
- test_hf_bigbird.py
- test_hf_convnext.py
- test_hf_canine.py
- test_hf_llava.py
- test_hf_llava_next.py
- test_hf_hubert.py
- test_hf_clipseg.py
- test_hf_mobilevit.py
- test_hf_vit.py
- test_hf_clap.py
- test_hf_bert_web.py
- test_hf_bert_base_uncased.py
- test_hf_argparser.py
- test_hf_transfo_xl.py
- test_hf_longt5.py
- test_hf_speech-to-text.py
- test_hf_mlp_mixer.py
- test_hf_xclip.py
- test_hf_t5_minimal.py
- test_hf_gpt-j.py
- test_hf_codellama.py
- test_hf_layoutlm.py
- test_hf_bert_minimal.py
- test_hf_stablelm.py
- test_hf_xlm-roberta.py
- test_hf_fuyu.py
- test_hf_video-llava.py
- test_hf_flan-t5.py
- test_hf_xlm_roberta.py
- test_hf_gpt_j.py
- test_hf_speech_to_text.py
- test_hf_xglm.py
- test_hf_flan_t5.py
- test_hf_gpt2_minimal.py
- test_hf_llava-next.py
- test_hf_vision-text-dual-encoder.py
- test_hf_mosaic_mpt.py
- test_hf_vision_text_dual_encoder.py
- test_hf_led.py
- test_hf_bigbird.py
- test_hf_convnextv2.py
- test_hf_pythia.py
- test_hf_mlp-mixer.py
- test_hf_clipseg.py
- test_hf_transfo-xl.py
- test_hf_vit_minimal.py
- test_hf_bark.py
- test_hf_vit.py
- test_hf_t5.py
- test_hf_gpt2.py
- test_hf_speech-to-text-2.py
- test_hf_bit.py
- test_hf_gemma3.py
- test_hf_open_llama.py
- test_hf_led.py
- test_hf_phi.py
- test_hf_resnet.py
- test_hf_florence.py
- test_hf_vinvl.py
- test_hf_donut.py
- test_hf_marian.py
- test_hf_llama_3.py
- test_hf_flava.py
- test_hf_xlm-roberta.py
- test_hf_rembert.py
- test_hf_vilt.py
- test_hf_xlm_prophetnet.py
- test_hf_poolformer.py
- test_hf_data2vec_audio.py
- test_hf_roformer.py
- test_hf_regnet.py
- test_hf_vision.py
- test_hf_usm.py
- test_hf_data2vec-text.py
- test_hf_gemma2.py
- test_hf_umt5.py
- test_hf_mixtral.py
- test_hf_ctrl.py
- test_hf_t5_standardized.py
- test_hf_mask2former.py
- test_hf_ernie.py
- test_hf_mamba.py
- test_hf_levit.py
- test_hf_m2m_100.py
- test_hf_flan-t5.py
- test_hf_stablelm.py
- test_hf_swin.py
- test_hf_imagebind.py
- test_hf_codellama.py
- test_hf_switch_transformers.py
- test_hf_mpt.py
- test_hf_gpt_j_standardized.py
- test_hf_convbert.py
- test_hf_detr.py
- test_hf_encoder_only.py
- test_hf_trocr_base.py
- test_hf_vit.py
- test_hf_mlp_mixer.py
- test_hf_llava_next.py
- test_hf_mistral.py
- test_hf_esm.py
- test_hf_deit.py
- test_hf_layoutlmv3.py
- test_hf_qwen3.py
- test_hf_gemma.py
- test_hf_mobilenet_v2.py
- test_hf_gpt-neo.py
- test_hf_qwen2.py
- test_hf_data2vec_text.py
- test_hf_layoutlmv2.py
- test_hf_transfo-xl.py
- test_hf_tapas.py
- test_hf_efficientnet.py
- test_hf_unispeech.py
- test_hf_deberta.py
- test_hf_seamless_m4t_v2.py
- test_hf_convnextv2.py
- test_hf_bert_standardized.py
- test_hf_vit_standardized.py
- test_hf_clvp.py
- test_hf_mt5.py
- test_hf_camembert.py
- test_hf_dinov2.py
- test_hf_idefics.py
- test_hf_gpt_j.py
- test_hf_dpt.py
- test_hf_wavlm.py
- test_hf_albert.py
- test_hf_git.py
- test_hf_pix2struct.py
- test_hf_data2vec-vision.py
- test_hf_yolos.py
- test_hf_speech_to_text.py
- test_hf_flamingo.py
- test_hf_bark.py
- test_hf_data2vec.py
- test_hf_squeezebert.py
- test_hf_roberta.py
- test_hf_olmo.py
- test_hf_codegen.py
- test_hf_beit.py
- test_hf_bart.py
- test_hf_bigbird.py
- test_hf_wav2vec2_bert.py
- test_hf_deberta_v2.py
- test_hf_wav2vec2.py
- test_hf_falcon.py
- test_hf_speecht5.py
- test_hf_reformer.py
- test_hf_clip.py
- test_hf_chinese_clip.py
- test_hf_gpt-neox.py
- test_hf_kosmos2.py
- test_hf_bloom.py
- test_hf_electra.py
- test_hf_sew.py
- test_hf_gpt2.py
- test_hf_gptj.py
- test_hf_mobilevit.py
- test_hf_musicgen.py
- test_hf_clipseg.py
- test_hf_ibert.py
- test_hf_llava.py
- test_hf_gpt-j.py
- test_hf_seamless_m4t.py
- test_hf_mlp-mixer.py
- test_hf_flaubert.py
- test_hf_audio.py
- test_hf_kosmos_2.py
- test_hf_llama.py
- test_hf_olmoe.py
- test_hf_xlnet.py
- test_hf_blip.py
- test_hf_pegasus.py
- test_hf_funnel.py
- test_hf_transfo_xl.py
- test_hf_speech-to-text.py
- test_hf_longt5.py
- test_hf_prophetnet.py
- test_hf_align.py
- test_hf_whisper.py
- test_hf_command_r.py
- test_hf_vision_encoder_decoder.py
- test_hf_mistral_next.py
- test_hf_decoder_only.py
- test_hf_clap.py
- test_hf_encodec.py
- test_hf_canine.py
- test_hf_vision-text-dual-encoder.py
- test_hf_distilbert.py
- test_hf_xclip.py
- test_hf_gpt_neox.py
- test_hf_paligemma.py
- test_hf_speech_to_text_2.py
- test_hf_sam.py
- test_hf_flan_t5.py
- test_hf_vision_text_dual_encoder.py
- test_hf_convnext.py
- test_hf_xlm_roberta.py
- test_hf_video_llava.py
- test_hf_encoder_decoder.py
- test_hf_longformer.py
- test_hf_multimodal.py
- test_hf_fuyu.py
- test_hf_segformer.py
- test_hf_blenderbot.py
- test_hf_trocr_large.py
- test_hf_phi3.py
- test_hf_opt.py
- test_hf_nemotron.py
- test_hf_t5.py
- test_hf_mbart.py
- test_hf_gpt_neo.py
- test_hf_data2vec-audio.py
- test_hf_hubert.py
- test_hf_data2vec_vision.py
- test_hf_bert_standardized.py
- test_hf_gpt2_standardized.py
- test_hf_albert_standardized.py
- test_hf_llama_standardized.py
- test_hf_blip_standardized.py
- test_hf_roberta_standardized.py
- test_hf_falcon_standardized.py
- test_hf_gpt_j_standardized.py
- test_hf_clip_standardized.py
- test_hf_deit_standardized.py
- test_hf_gemma_standardized.py
- test_hf_vit_standardized.py
- test_hf_bart_standardized.py
- test_hf_distilbert_standardized.py
- test_hf_t5_standardized.py
- test_hf_electra_standardized.py
- test_hf_mistral_standardized.py
- test_hf_wav2vec2_standardized.py
- test_hf_bark.py
- test_hf_speecht5.py
- test_hf_roberta.py
- test_hf_pix2struct.py
- test_hf_longt5.py
- test_hf_xlm_roberta.py
- test_hf_vit.py
- test_hf_roberta.py
- test_hf_bert.py
- test_hf_gpt2.py
- test_hf_t5.py
- test_hf_vit.py
- test_hf_gpt2.py
- test_hf_t5.py
- test_hf_bert.py
- test_hf_gpt_j.py
- test_hf_t5.py
- test_hf_vit.py
- test_hf_gpt2.py
- test_hf_bert.py
- test_hf_mosaic_mpt.py
- test_hf_bart.py
- test_hf_xlm.py
- test_hf_mistral.py
- test_hf_hubert.py
- test_hf_mobilevit.py
- test_hf_sew.py
- test_hf_levit.py
- test_hf_clip.py
- test_hf_deberta.py
- test_hf_longt5.py
- test_hf_pegasus_x.py
- test_hf_cvt.py
- test_hf_chinese_clip.py
- test_hf_mpnet.py
- test_hf_blip_2.py
- test_hf_blip.py
- test_hf_flava.py
- test_hf_perceiver.py
- test_hf_funnel.py
- test_hf_wavlm.py
- test_hf_stablelm.py
- test_hf_paligemma.py
- test_hf_falcon.py
- test_hf_whisper.py
- test_hf_llava.py
- test_hf_mpt.py
- test_hf_flan_t5.py
- test_hf_vilt.py
- test_hf_xlnet.py
- test_hf_xglm.py
- test_hf_albert.py
- test_hf_llama.py
- test_hf_data2vec_audio.py
- test_hf_convnext.py
- test_hf_swin.py
- test_hf_led.py
- test_hf_phi.py
- test_hf_gpt_neox.py
- test_hf_git.py
- test_hf_musicgen.py
- test_hf_deit.py
- test_hf_roberta.py
- test_hf_codegen.py
- test_hf_camembert.py
- test_hf_distilbert.py
- test_hf_flaubert.py
- test_hf_resnet.py
- test_hf_unispeech.py
- test_hf_efficientnet.py
- test_hf_swinv2.py
- test_hf_pythia.py
- test_hf_bigbird_pegasus.py
- test_hf_nllb.py
- test_hf_wav2vec2.py
- test_hf_xlm_roberta.py
- test_hf_whisper.py
- test_hf_t5.py
- test_hf_bert.py
- test_hf_clip.py
- test_hf_gpt2.py
- test_hf_gpt_j.py
- test_hf_encoder_decoder.py
- test_hf_decoder_only.py
- test_hf_encoder_only.py
- test_hf_gpt_neo.py
- test_hf_xlm_roberta.py
- test_hf_t5.py
- test_hf_xclip.py
- test_hf_clip.py
- test_hf_llava.py
- test_hf_detr.py
- test_hf_qwen2.py
- test_hf_t5.py
- test_hf_clap.py
- test_hf_wav2vec2.py
- test_hf_whisper.py
- test_hf_llama.py
- test_hf_llava.py
- test_hf_convnext.py
- test_hf_roberta.py
- test_hf_resnet.py
- test_hf_albert.py
- test_hf_clip.py
- test_hf_bart.py
- test_hf_blip.py
- test_hf_wav2vec2.py
- test_hf_whisper.py
- test_hf_mpt.py
- test_hf_deit.py
- test_hf_falcon.py
- test_hf_phi.py
- test_hf_hubert.py
- test_hf_deberta.py
- test_hf_distilbert.py
- test_hf_swin.py
- test_hf_mistral.py
- test_hf_llava.py
- test_hf_qwen2.py
- test_hf_vit.py
- test_hf_bert_web.py
- test_hf_argparser.py
- test_hf_bigbird.py
- test_hf_llama.py
- test_hf_whisper.py
- test_hf_bert_base_uncased.py
- test_hf_clap.py
- test_hf_vit.py
- test_hf_bert.py

### Tokenization Tests

- test_tokenization_herbert.py
- test_tokenization_distilbert.py
- test_tokenization_bert.py
- test_tokenization_albert.py
- test_tokenization_deberta.py
- test_tokenization_flaubert.py
- test_tokenization_deberta_v2.py
- test_tokenization_rembert.py
- test_tokenization_camembert.py
- test_tokenization_bert_japanese.py
- test_tokenization_squeezebert.py
- test_tokenization_bert_generation.py
- test_tokenization_roc_bert.py
- test_tokenization_phobert.py
- test_tokenization_roberta.py
- test_tokenization_bert_tf.py
- test_tokenization_mobilebert.py
- test_tokenization_bertweet.py
- test_tokenization_xlm_roberta.py
- test_tokenization_common.py
- test_tokenization_utils.py
- test_tokenization_fast.py
- test_tokenization_utils.py
- test_tokenization_phobert.py
- test_tokenization_bert_japanese.py
- test_tokenization_clvp.py
- test_tokenization_moshi.py
- test_tokenization_herbert.py
- test_tokenization_vits.py
- test_tokenization_byt5.py
- test_tokenization_udop.py
- test_tokenization_bart.py
- test_tokenization_gpt2_tf.py
- test_tokenization_gpt2.py
- test_tokenization_clip.py
- test_tokenization_dpr.py
- test_tokenization_xlm.py
- test_tokenization_cpm.py
- test_tokenization_barthez.py
- test_tokenization_layoutlm.py
- test_tokenization_reformer.py
- test_tokenization_marian.py
- test_tokenization_lxmert.py
- test_tokenization_flaubert.py
- test_tokenization_qwen2.py
- test_tokenization_longformer.py
- test_tokenization_siglip.py
- test_tokenization_luke.py
- test_tokenization_splinter.py
- test_tokenization_xlm_roberta.py
- test_tokenization_mgp_str.py
- test_tokenization_speecht5.py
- test_tokenization_led.py
- test_tokenization_wav2vec2.py
- test_tokenization_layoutxlm.py
- test_tokenization_deberta_v2.py
- test_tokenization_plbart.py
- test_tokenization_deberta.py
- test_tokenization_roc_bert.py
- test_tokenization_nougat.py
- test_tokenization_t5.py
- test_tokenization_bert_generation.py
- test_tokenization_mpnet.py
- test_tokenization_mobilebert.py
- test_tokenization_markuplm.py
- test_tokenization_m2m_100.py
- test_tokenization_mbart50.py
- test_tokenization_bertweet.py
- test_tokenization_distilbert.py
- test_tokenization_gemma.py
- test_tokenization_pop2piano.py
- test_tokenization_whisper.py
- test_tokenization_esm.py
- test_tokenization_fastspeech2_conformer.py
- test_tokenization_seamless_m4t.py
- test_tokenization_xglm.py
- test_tokenization_bloom.py
- test_tokenization_layoutlmv3.py
- test_tokenization_pegasus.py
- test_tokenization_tapas.py
- test_tokenization_gpt_neox_japanese.py
- test_tokenization_rag.py
- test_tokenization_electra.py
- test_tokenization_xlnet.py
- test_tokenization_ctrl.py
- test_tokenization_albert.py
- test_tokenization_openai.py
- test_tokenization_bartpho.py
- test_tokenization_nllb.py
- test_tokenization_canine.py
- test_tokenization_wav2vec2_phoneme.py
- test_tokenization_rembert.py
- test_tokenization_perceiver.py
- test_tokenization_auto.py
- test_tokenization_cohere.py
- test_tokenization_fnet.py
- test_tokenization_myt5.py
- test_tokenization_llama.py
- test_tokenization_codegen.py
- test_tokenization_funnel.py
- test_tokenization_biogpt.py
- test_tokenization_layoutlmv2.py
- test_tokenization_mluke.py
- test_tokenization_big_bird.py
- test_tokenization_mvp.py
- test_tokenization_speech_to_text.py
- test_tokenization_gpt_sw3.py
- test_tokenization_prophetnet.py
- test_tokenization_fsmt.py
- test_tokenization_blenderbot.py
- test_tokenization_code_llama.py
- test_tokenization_roformer.py
- test_tokenization_cpmant.py
- test_tokenization_blenderbot_small.py
- test_tokenization_mbart.py
- test_tokenization_bert_tf.py
- test_tokenization_bert.py
- test_tokenization_camembert.py
- test_tokenization_squeezebert.py
- test_tokenization_roberta.py
- test_tokenization_common.py
- test_tokenization_fast.py
- test_tokenization_utils.py
- test_tokenization_xglm.py
- test_tokenization_code_llama.py
- test_tokenization_blenderbot.py
- test_tokenization_prophetnet.py
- test_tokenization_mpnet.py
- test_tokenization_rembert.py
- test_tokenization_codegen.py
- test_tokenization_layoutlmv2.py
- test_tokenization_xlm_roberta.py
- test_tokenization_nougat.py
- test_tokenization_speech_to_text.py
- test_tokenization_electra.py
- test_tokenization_esm.py
- test_tokenization_perceiver.py
- test_tokenization_plbart.py
- test_tokenization_gemma.py
- test_tokenization_tapas.py
- test_tokenization_bartpho.py
- test_tokenization_bloom.py
- test_tokenization_siglip.py
- test_tokenization_lxmert.py
- test_tokenization_marian.py
- test_tokenization_gpt_neox_japanese.py
- test_tokenization_big_bird.py
- test_tokenization_moshi.py
- test_tokenization_dpr.py
- test_tokenization_roberta.py
- test_tokenization_xlm.py
- test_tokenization_cpm.py
- test_tokenization_luke.py
- test_tokenization_qwen2.py
- test_tokenization_gpt_sw3.py
- test_tokenization_clvp.py
- test_tokenization_roformer.py
- test_tokenization_clip.py
- test_tokenization_camembert.py
- test_tokenization_gpt2_tf.py
- test_tokenization_gpt2.py
- test_tokenization_udop.py
- test_tokenization_byt5.py
- test_tokenization_bart.py
- test_tokenization_led.py
- test_tokenization_vits.py
- test_tokenization_seamless_m4t.py
- test_tokenization_distilbert.py
- test_tokenization_bert_tf.py
- test_tokenization_bert.py
- test_tokenization_fastspeech2_conformer.py
- test_tokenization_layoutlm.py
- test_tokenization_mobilebert.py
- test_tokenization_cpmant.py
- test_tokenization_deberta_v2.py
- test_tokenization_mvp.py
- test_tokenization_mluke.py
- test_tokenization_bert_generation.py
- test_tokenization_mgp_str.py
- test_tokenization_wav2vec2_phoneme.py
- test_tokenization_wav2vec2.py
- test_tokenization_fsmt.py
- test_tokenization_phobert.py
- test_tokenization_biogpt.py
- test_tokenization_layoutlmv3.py
- test_tokenization_herbert.py
- test_tokenization_speecht5.py
- test_tokenization_flaubert.py
- test_tokenization_reformer.py
- test_tokenization_barthez.py
- test_tokenization_splinter.py
- test_tokenization_mbart.py
- test_tokenization_xlnet.py
- test_tokenization_markuplm.py
- test_tokenization_myt5.py
- test_tokenization_rag.py
- test_tokenization_fnet.py
- test_tokenization_whisper.py
- test_tokenization_layoutxlm.py
- test_tokenization_bertweet.py
- test_tokenization_funnel.py
- test_tokenization_auto.py
- test_tokenization_roc_bert.py
- test_tokenization_pegasus.py
- test_tokenization_cohere.py
- test_tokenization_bert_japanese.py
- test_tokenization_squeezebert.py
- test_tokenization_nllb.py
- test_tokenization_albert.py
- test_tokenization_openai.py
- test_tokenization_blenderbot_small.py
- test_tokenization_deberta.py
- test_tokenization_pop2piano.py
- test_tokenization_longformer.py
- test_tokenization_t5.py
- test_tokenization_ctrl.py
- test_tokenization_mbart50.py
- test_tokenization_llama.py
- test_tokenization_m2m_100.py
- test_tokenization_canine.py
- test_tokenization_utils.py

### Image Tests

- test_image_transforms.py
- test_image_processing_common.py
- test_image_utils.py
- test_image_processing_utils.py
- test_image_processing_pix2struct.py
- test_image_processing_gemma3.py
- test_image_processing_vilt.py
- test_image_processing_zoedepth.py
- test_image_processing_videomae.py
- test_image_processing_yolos.py
- test_image_processing_fuyu.py
- test_image_processing_clip.py
- test_image_processing_blip.py
- test_image_processing_qwen2_vl.py
- test_image_processing_glpn.py
- test_image_processing_maskformer.py
- test_image_processing_siglip2.py
- test_image_processing_idefics2.py
- test_image_processing_siglip.py
- test_image_processing_conditional_detr.py
- test_image_processing_beit.py
- test_image_processing_idefics.py
- test_image_processing_smolvlm.py
- test_image_processing_nougat.py
- test_image_processing_vitpose.py
- test_image_processing_superpoint.py
- test_image_processing_textnet.py
- test_image_processing_vit.py
- test_image_processing_mllama.py
- test_image_processing_llava_next.py
- test_image_processing_levit.py
- test_image_processing_tvp.py
- test_image_processing_detr.py
- test_image_processing_deit.py
- test_image_processing_got_ocr2.py
- test_image_processing_layoutlmv3.py
- test_image_processing_efficientnet.py
- test_image_processing_llava.py
- test_image_processing_llava_next_video.py
- test_image_processing_vivit.py
- test_image_processing_bridgetower.py
- test_image_processing_convnext.py
- test_image_processing_donut.py
- test_image_processing_idefics3.py
- test_image_processing_flava.py
- test_image_processing_vitmatte.py
- test_image_processing_owlvit.py
- test_image_processing_superglue.py
- test_image_processing_dpt.py
- test_image_processing_mobilenet_v2.py
- test_image_processing_depth_pro.py
- test_image_processing_auto.py
- test_image_processing_seggpt.py
- test_image_processing_swin2sr.py
- test_image_processing_chinese_clip.py
- test_image_processing_poolformer.py
- test_image_processing_chameleon.py
- test_image_processing_imagegpt.py
- test_image_processing_layoutlmv2.py
- test_image_processing_llava_onevision.py
- test_image_processing_aria.py
- test_image_processing_timm_wrapper.py
- test_image_processing_mobilevit.py
- test_image_processing_segformer.py
- test_image_processing_pvt.py
- test_image_processing_mobilenet_v1.py
- test_image_processing_rt_detr.py
- test_image_processing_mask2former.py
- test_image_processing_instrictblipvideo.py
- test_image_processing_owlv2.py
- test_image_processing_grounding_dino.py
- test_image_processing_pixtral.py
- test_image_processing_oneformer.py
- test_image_processing_video_llava.py
- test_image_processing_deformable_detr.py
- test_image_question_answering.py
- test_image_transforms.py
- test_image_processing_common.py
- test_image_question_answering.py
- test_image_processing_vitmatte.py
- test_image_processing_swin2sr.py
- test_image_processing_idefics3.py
- test_image_processing_deit.py
- test_image_processing_grounding_dino.py
- test_image_processing_superglue.py
- test_image_processing_llava_onevision.py
- test_image_processing_vit.py
- test_image_processing_levit.py
- test_image_processing_mobilenet_v1.py
- test_image_processing_convnext.py
- test_image_processing_tvp.py
- test_image_processing_layoutlmv2.py
- test_image_processing_nougat.py
- test_image_processing_instrictblipvideo.py
- test_image_processing_depth_pro.py
- test_image_processing_imagegpt.py
- test_image_processing_deformable_detr.py
- test_image_processing_detr.py
- test_image_processing_llava.py
- test_image_processing_chameleon.py
- test_image_processing_conditional_detr.py
- test_image_processing_timm_wrapper.py
- test_image_processing_mllama.py
- test_image_processing_yolos.py
- test_image_processing_chinese_clip.py
- test_image_processing_siglip.py
- test_image_processing_mobilevit.py
- test_image_processing_beit.py
- test_image_processing_rt_detr.py
- test_image_processing_mobilenet_v2.py
- test_image_processing_segformer.py
- test_image_processing_pixtral.py
- test_image_processing_blip.py
- test_image_processing_gemma3.py
- test_image_processing_glpn.py
- test_image_processing_poolformer.py
- test_image_processing_vilt.py
- test_image_processing_clip.py
- test_image_processing_fuyu.py
- test_image_processing_oneformer.py
- test_image_processing_qwen2_vl.py
- test_image_processing_llava_next_video.py
- test_image_processing_efficientnet.py
- test_image_processing_llava_next.py
- test_image_processing_siglip2.py
- test_image_processing_superpoint.py
- test_image_processing_bridgetower.py
- test_image_processing_videomae.py
- test_image_processing_zoedepth.py
- test_image_processing_pvt.py
- test_image_processing_owlv2.py
- test_image_processing_layoutlmv3.py
- test_image_processing_aria.py
- test_image_processing_idefics2.py
- test_image_processing_seggpt.py
- test_image_processing_maskformer.py
- test_image_processing_vivit.py
- test_image_processing_auto.py
- test_image_processing_pix2struct.py
- test_image_processing_dpt.py
- test_image_processing_flava.py
- test_image_processing_donut.py
- test_image_processing_textnet.py
- test_image_processing_vitpose.py
- test_image_processing_got_ocr2.py
- test_image_processing_video_llava.py
- test_image_processing_smolvlm.py
- test_image_processing_idefics.py
- test_image_processing_mask2former.py
- test_image_processing_owlvit.py
- test_image_utils.py
- test_image_processing_utils.py

### Processor Tests

- test_processor_wav2vec2_bert.py
- test_processor_pix2struct.py
- test_processor_clvp.py
- test_processor_trocr.py
- test_processor_instructblip.py
- test_processor_fuyu.py
- test_processor_udop.py
- test_processor_clip.py
- test_processor_blip.py
- test_processor_qwen2_vl.py
- test_processor_wav2vec2_with_lm.py
- test_processor_clap.py
- test_processor_paligemma.py
- test_processor_idefics2.py
- test_processor_wav2vec2_bert.py
- test_processor_mgp_str.py
- test_processor_speecht5.py
- test_processor_wav2vec2.py
- test_processor_layoutxlm.py
- test_processor_idefics.py
- test_processor_smolvlm.py
- test_processor_align.py
- test_processor_qwen2_audio.py
- test_processor_altclip.py
- test_processor_markuplm.py
- test_processor_qwen2_5_vl.py
- test_processor_mllama.py
- test_processor_llava_next.py
- test_processor_musicgen_melody.py
- test_processor_pop2piano.py
- test_processor_whisper.py
- test_processor_seamless_m4t.py
- test_processor_got_ocr2.py
- test_processor_layoutlmv3.py
- test_processor_aya_vision.py
- test_processor_llava.py
- test_processor_llava_next_video.py
- test_processor_bridgetower.py
- test_processor_donut.py
- test_processor_idefics3.py
- test_processor_flava.py
- test_processor_owlvit.py
- test_processor_vipllava.py
- test_processor_auto.py
- test_processor_emu3.py
- test_processor_chinese_clip.py
- test_processor_clipseg.py
- test_processor_chameleon.py
- test_processor_sam.py
- test_processor_git.py
- test_processor_bark.py
- test_processor_kosmos2.py
- test_processor_vision_text_dual_encoder.py
- test_processor_layoutlmv2.py
- test_processor_llava_onevision.py
- test_processor_aria.py
- test_processor_omdet_turbo.py
- test_processor_speech_to_text.py
- test_processor_blip_2.py
- test_processor_instructblipvideo.py
- test_processor_owlv2.py
- test_processor_musicgen.py
- test_processor_grounding_dino.py
- test_processor_pixtral.py
- test_processor_oneformer.py
- test_processor_idefics3.py
- test_processor_vipllava.py
- test_processor_align.py
- test_processor_grounding_dino.py
- test_processor_llava_onevision.py
- test_processor_wav2vec2_with_lm.py
- test_processor_clipseg.py
- test_processor_layoutlmv2.py
- test_processor_speech_to_text.py
- test_processor_instructblipvideo.py
- test_processor_llava.py
- test_processor_chameleon.py
- test_processor_mllama.py
- test_processor_chinese_clip.py
- test_processor_trocr.py
- test_processor_qwen2_audio.py
- test_processor_pixtral.py
- test_processor_musicgen.py
- test_processor_blip.py
- test_processor_clap.py
- test_processor_kosmos2.py
- test_processor_clvp.py
- test_processor_clip.py
- test_processor_fuyu.py
- test_processor_udop.py
- test_processor_oneformer.py
- test_processor_seamless_m4t.py
- test_processor_qwen2_vl.py
- test_processor_llava_next_video.py
- test_processor_llava_next.py
- test_processor_qwen2_5_vl.py
- test_processor_musicgen_melody.py
- test_processor_bridgetower.py
- test_processor_mgp_str.py
- test_processor_aya_vision.py
- test_processor_owlv2.py
- test_processor_wav2vec2.py
- test_processor_paligemma.py
- test_processor_layoutlmv3.py
- test_processor_speecht5.py
- test_processor_aria.py
- test_processor_idefics2.py
- test_processor_blip_2.py
- test_processor_vision_text_dual_encoder.py
- test_processor_markuplm.py
- test_processor_whisper.py
- test_processor_bark.py
- test_processor_layoutxlm.py
- test_processor_auto.py
- test_processor_emu3.py
- test_processor_pix2struct.py
- test_processor_omdet_turbo.py
- test_processor_flava.py
- test_processor_donut.py
- test_processor_got_ocr2.py
- test_processor_altclip.py
- test_processor_pop2piano.py
- test_processor_smolvlm.py
- test_processor_idefics.py
- test_processor_wav2vec2_bert.py
- test_processor_sam.py
- test_processor_git.py
- test_processor_instructblip.py
- test_processor_owlvit.py

### Pipelines Tests

- test_pipelines_zero_shot_image_classification.py
- test_pipelines_summarization.py
- test_pipelines_text_to_audio.py
- test_pipelines_audio_classification.py
- test_pipelines_automatic_speech_recognition.py
- test_pipelines_image_text_to_text.py
- test_pipelines_question_answering.py
- test_pipelines_translation.py
- test_pipelines_visual_question_answering.py
- test_pipelines_zero_shot_object_detection.py
- test_pipelines_video_classification.py
- test_pipelines_zero_shot_audio_classification.py
- test_pipelines_text_classification.py
- test_pipelines_zero_shot.py
- test_pipelines_image_classification.py
- test_pipelines_common.py
- test_pipelines_text_generation.py
- test_pipelines_image_segmentation.py
- test_pipelines_table_question_answering.py
- test_pipelines_token_classification.py
- test_pipelines_feature_extraction.py
- test_pipelines_fill_mask.py
- test_pipelines_image_to_text.py
- test_pipelines_depth_estimation.py
- test_pipelines_mask_generation.py
- test_pipelines_object_detection.py
- test_pipelines_text2text_generation.py
- test_pipelines_document_question_answering.py
- test_pipelines_image_to_image.py
- test_pipelines_image_feature_extraction.py
- test_pipelines_zero_shot.py
- test_pipelines_text_generation.py
- test_pipelines_text2text_generation.py
- test_pipelines_image_to_image.py
- test_pipelines_token_classification.py
- test_pipelines_mask_generation.py
- test_pipelines_image_to_text.py
- test_pipelines_question_answering.py
- test_pipelines_image_text_to_text.py
- test_pipelines_document_question_answering.py
- test_pipelines_image_classification.py
- test_pipelines_zero_shot_object_detection.py
- test_pipelines_zero_shot_image_classification.py
- test_pipelines_common.py
- test_pipelines_text_classification.py
- test_pipelines_fill_mask.py
- test_pipelines_video_classification.py
- test_pipelines_automatic_speech_recognition.py
- test_pipelines_feature_extraction.py
- test_pipelines_image_segmentation.py
- test_pipelines_zero_shot_audio_classification.py
- test_pipelines_text_to_audio.py
- test_pipelines_table_question_answering.py
- test_pipelines_visual_question_answering.py
- test_pipelines_audio_classification.py
- test_pipelines_translation.py
- test_pipelines_image_feature_extraction.py
- test_pipelines_summarization.py
- test_pipelines_object_detection.py
- test_pipelines_depth_estimation.py

### Feature Tests

- test_feature_extraction_common.py
- test_feature_extraction_utils.py
- test_feature_extraction_clvp.py
- test_feature_extraction_clap.py
- test_feature_extraction_audio_spectrogram_transformer.py
- test_feature_extraction_univnet.py
- test_feature_extraction_speecht5.py
- test_feature_extraction_wav2vec2.py
- test_feature_extraction_markuplm.py
- test_feature_extraction_musicgen_melody.py
- test_feature_extraction_pop2piano.py
- test_feature_extraction_whisper.py
- test_feature_extraction_seamless_m4t.py
- test_feature_extraction_auto.py
- test_feature_extraction_dac.py
- test_feature_extraction_encodec.py
- test_feature_extraction_speech_to_text.py
- test_feature_extraction_common.py
- test_feature_extraction_encodec.py
- test_feature_extraction_speech_to_text.py
- test_feature_extraction_audio_spectrogram_transformer.py
- test_feature_extraction_clap.py
- test_feature_extraction_clvp.py
- test_feature_extraction_seamless_m4t.py
- test_feature_extraction_univnet.py
- test_feature_extraction_musicgen_melody.py
- test_feature_extraction_wav2vec2.py
- test_feature_extraction_speecht5.py
- test_feature_extraction_markuplm.py
- test_feature_extraction_whisper.py
- test_feature_extraction_auto.py
- test_feature_extraction_dac.py
- test_feature_extraction_pop2piano.py
- test_feature_extraction_utils.py

### Bert-base-uncased Tests

- test_bert-base-uncased.py
- test_bert-base-uncased_rocm.py
- test_bert-base-uncased.py
- test_bert-base-uncased_cpu.py
- test_bert-base-uncased_webnn.py
- test_bert-base-uncased_openvino.py
- test_bert-base-uncased_cuda.py
- test_bert-base-uncased_webgpu.py
- test_bert-base-uncased_mps.py
- test_bert-base-uncased_qnn.py
- test_bert-base-uncased.py
- test_bert-base-uncased_cuda.py
- test_bert-base-uncased_cuda.py
- test_bert-base-uncased_cuda.py
- test_bert-base-uncased_cpu.py
- test_bert-base-uncased_cpu.py
- test_bert-base-uncased_rocm.py
- test_bert-base-uncased_cuda.py
- test_bert-base-uncased_webgpu.py
- test_bert-base-uncased_mps.py
- test_bert-base-uncased_openvino.py
- test_bert-base-uncased_webnn.py
- test_bert-base-uncased_cpu.py
- test_bert-base-uncased_qnn.py
- test_bert-base-uncased_cpu.py
- test_bert-base-uncased_cpu.py
- test_bert-base-uncased_cuda.py
- test_bert-base-uncased_cpu.py
- test_bert-base-uncased_cpu.py
- test_bert-base-uncased_cpu.py

### Bert Tests

- test_bert_simple.py
- test_bert_fixed.py
- test_bert_fixed_from_updated.py
- test_bert_template.py
- test_bert.py
- test_bert_base_uncased.py
- test_bert_simple.py
- test_bert_fixed_from_updated.py
- test_bert_fixed.py
- test_bert_template.py
- test_bert_example.py
- test_bert_simple.py
- test_bert_fixed.py
- test_bert_qualcomm.py
- test_bert_base.py
- test_bert.py
- test_bert_base_uncased.py
- test_bert.py
- test_bert.py
- test_bert.py
- test_bert.py

### Trainer Tests

- test_trainer_ext.py
- test_trainer_distributed.py
- test_trainer_distributed_loss.py
- test_trainer.py
- test_trainer_seq2seq.py
- test_trainer_tpu.py
- test_trainer_callback.py
- test_trainer_utils.py
- test_trainer_fsdp.py
- test_trainer_ext.py
- test_trainer_distributed.py
- test_trainer_utils.py
- test_trainer_distributed_loss.py
- test_trainer_seq2seq.py
- test_trainer.py
- test_trainer_fsdp.py
- test_trainer_tpu.py
- test_trainer_callback.py

### Gpt2 Tests

- test_gpt2.py
- test_gpt2_webgpu.py
- test_gpt2_cpu.py
- test_gpt2_qnn.py
- test_gpt2_webnn.py
- test_gpt2_openvino.py
- test_gpt2_webgpu.py
- test_gpt2_mps.py
- test_gpt2_cuda.py
- test_gpt2_rocm.py
- test_gpt2.py
- test_gpt2.py
- test_gpt2.py
- test_gpt2.py
- test_gpt2.py
- test_gpt2.py

### Openai Tests

- test_openai_api.py
- test_openai_api.py
- test_openai_clip-vit-base-patch32_webgpu.py
- test_openai_clip-vit-base-patch32_webnn.py
- test_openai_clip-vit-base-patch32_openvino.py
- test_openai_clip-vit-base-patch32_cpu.py
- test_openai_clip-vit-base-patch32_cuda.py
- test_openai_clip-vit-base-patch32_qnn.py
- test_openai_clip-vit-base-patch32_rocm.py
- test_openai_clip-vit-base-patch32_webgpu.py
- test_openai_clip-vit-base-patch32_mps.py
- test_openai_api_fixed.py
- test_openai_client.py
- test_openai_api.py

### Vit-base-patch16-224 Tests

- test_vit-base-patch16-224.py
- test_vit-base-patch16-224_webgpu.py
- test_vit-base-patch16-224.py
- test_vit-base-patch16-224_rocm.py
- test_vit-base-patch16-224_webnn.py
- test_vit-base-patch16-224_cuda.py
- test_vit-base-patch16-224_mps.py
- test_vit-base-patch16-224_cpu.py
- test_vit-base-patch16-224_webgpu.py
- test_vit-base-patch16-224_openvino.py
- test_vit-base-patch16-224_qnn.py
- test_vit-base-patch16-224_webgpu.py

### Dashboard Tests

- test_dashboard_enhanced_visualization.py
- test_basic_dashboard_integration.py
- test_dashboard_visualization_web_integration.py
- test_dashboard_integration.py
- test_dashboard_integration.py
- test_basic_dashboard_integration.py
- test_dashboard_integration.py
- test_dashboard_integration.py
- test_dashboard_integration.py
- test_dashboard_integration.py
- test_dashboard_regression_integration.py
- test_dashboard_visualization_web_integration.py

### Coordinator Tests

- test_coordinator_circuit_breaker_integration.py
- test_coordinator_orchestrator_integration.py
- test_coordinator_error_integration.py
- test_coordinator_integration.py
- test_coordinator_failover.py
- test_coordinator.py
- test_coordinator_redundancy.py
- test_coordinator_integration.py
- test_coordinator_load_balancer.py
- test_coordinator_circuit_breaker_integration.py
- test_coordinator_orchestrator_integration.py
- test_coordinator_error_integration.py

### Model Tests

- test_model_card.py
- test_model_output.py
- test_model_zoo.py
- test_advanced_model_selection.py
- test_model_lookup.py
- test_model_lookup_advanced.py
- test_model_api.py
- test_model_update_pipeline.py
- test_model_zoo.py
- test_model_output.py
- test_model_card.py
- test_model_selection.py

### Whisper-tiny Tests

- test_whisper-tiny.py
- test_whisper-tiny_webgpu.py
- test_whisper-tiny.py
- test_whisper-tiny_webnn.py
- test_whisper-tiny_openvino.py
- test_whisper-tiny_webgpu.py
- test_whisper-tiny_mps.py
- test_whisper-tiny_cpu.py
- test_whisper-tiny_qnn.py
- test_whisper-tiny_cuda.py
- test_whisper-tiny_rocm.py

### Multi Tests

- test_multi_model_web_integration.py
- test_multi_model_resource_pool_integration.py
- test_multi_node_model_parallel.py
- test_multi_node_data_parallel.py
- test_multi_model_resource_pool_integration.py
- test_multi_model_web_integration.py
- test_multi_model_execution.py
- test_multi_node_data_parallel.py
- test_multi_node_model_parallel.py
- test_multi_device_orchestrator_with_drm.py
- test_multi_device_orchestrator.py

### Resource Tests

- test_basic_resource_pool_fault_tolerance.py
- test_resource_pool_enhanced.py
- test_resource_pool_bridge_integration.py
- test_resource_pool_integration.py
- test_resource_pool_with_recovery.py
- test_resource_pool_bridge_recovery.py
- test_resource_pool_integration.py
- test_resource_pool_bridge_integration.py
- test_resource_optimization.py
- test_resource_performance_predictor.py

### Visualization Tests

- test_visualization_direct.py
- test_visualization_standalone.py
- test_advanced_visualization.py
- test_visualization_dashboard_integration.py
- test_visualization.py
- test_visualization_minimal.py
- test_visualization_dashboard.py
- test_visualization_db_connector.py
- test_visualization.py
- test_visualization_dashboard_integration.py

### Error Tests

- test_error_recovery_db_integration.py
- test_error_visualization_dashboard_integration.py
- test_error_recovery_db_integration.py
- test_error_recovery_performance.py
- test_error_notification_system.py
- test_error_visualization.py
- test_error_visualization_comprehensive.py
- test_error_visualization_dashboard_integration.py
- test_error_visualization_e2e.py
- test_error_visualization_realtime.py

### Configuration Tests

- test_configuration_common.py
- test_configuration_utils.py
- test_configuration_llava.py
- test_configuration_auto.py
- test_configuration_utils.py
- test_configuration_common.py
- test_configuration_llava.py
- test_configuration_auto.py
- test_configuration_utils.py
- test_configuration_utils.py

### Convert Tests

- test_convert_slow_tokenizer.py
- test_convert_to_notebook.py
- test_convert_doc_file.py
- test_convert_md_to_mdx.py
- test_convert_rst_to_mdx.py
- test_convert_md_to_mdx.py
- test_convert_to_notebook.py
- test_convert_rst_to_mdx.py
- test_convert_doc_file.py
- test_convert_slow_tokenizer.py

### Ipfs Tests

- test_ipfs_ultra_low_precision_integration.py
- test_ipfs_resource_pool_integration.py
- test_ipfs_accelerate.py
- test_ipfs_accelerate_fixed.py
- test_ipfs_ultra_low_precision_integration.py
- test_ipfs_resource_pool_integration.py
- test_ipfs_accelerate_webnn_webgpu.py
- test_ipfs_accelerate_with_cross_browser.py
- test_ipfs_accelerate_fixed.py

### Vit Tests

- test_vit_custom.py
- test_vit.py
- test_vit.py
- test_vit.py
- test_vit_fixed.py
- test_vit.py
- test_vit.py
- test_vit.py
- test_vit.py

### Enhanced Tests

- test_enhanced_resource_pool.py
- test_enhanced_hardware_taxonomy.py
- test_enhanced_hardware_capability.py
- test_enhanced_reports.py
- test_enhanced_documentation.py
- test_enhanced_reporting.py
- test_enhanced_statistical_validator.py
- test_enhanced_visualization_ui.py
- test_enhanced_hardware_taxonomy.py

### Hardware Tests

- test_hardware_taxonomy_integration.py
- test_hardware_aware_scheduler.py
- test_hardware_capability_detector.py
- test_hardware_utilization_monitor.py
- test_hardware_test_matcher.py
- test_hardware.py
- test_hardware_abstraction_layer.py
- test_hardware_fault_tolerance.py
- test_hardware_taxonomy_integration.py

### Drm Tests

- test_drm_integration.py
- test_drm_integration.py
- test_drm_visualization_integration.py
- test_drm_real_time_dashboard.py
- test_drm_external_monitoring_e2e.py
- test_drm_external_monitoring.py
- test_drm_integration.py
- test_drm_dashboard_integration.py

### Fault Tests

- test_fault_tolerant_cross_browser_model_sharding.py
- test_fault_tolerant_cross_browser_model_sharding_validation.py
- test_fault_tolerance_integration.py
- test_fault_tolerance.py
- test_fault_tolerance_system.py
- test_fault_tolerance_integration.py
- test_fault_tolerance_visualization.py
- test_fault_tolerant_model_sharding.py

### Clip Tests

- test_clip.py
- test_clip.py
- test_clip.py
- test_clip.py
- test_clip.py
- test_clip.py
- test_clip.py
- test_clip.py

### Integration Tests

- test_integration.py
- test_integration.py
- test_integration.py
- test_integration.py
- test_integration.py
- test_integration.py
- test_integration.py
- test_integration.py

### Processing Tests

- test_processing_common.py
- test_processing_utils.py
- test_processing_gemma3.py
- test_processing_colpali.py
- test_processing_common.py
- test_processing_gemma3.py
- test_processing_colpali.py
- test_processing_utils.py

### Api Tests

- test_api_backend_converter.py
- test_api_backend_converter_integration.py
- test_api_backend_converter_integration.py
- test_api_backend_converter.py
- test_api_endpoints.py
- test_api_backend.py
- test_api_endpoints.py

### T5 Tests

- test_t5.py
- test_t5.py
- test_t5.py
- test_t5.py
- test_t5.py
- test_t5.py
- test_t5.py

### Load Tests

- test_load_balancer_resource_pool_integration.py
- test_load_balancer_resource_pool_integration.py
- test_load_balancer_stress.py
- test_basic_load_balancer.py
- test_load_balancer.py
- test_load_balancer_fault_tolerance.py
- test_load_balancer_monitoring.py

### Web Tests

- test_web_resource_pool_integration.py
- test_web_resource_pool.py
- test_web_resource_pool_fault_tolerance_integration.py
- test_web_resource_pool_integration.py
- test_web_resource_pool_fault_tolerance_integration.py
- test_web_resource_pool_adapter.py

### Generator Tests

- test_generator_fixed.py
- test_generator.py
- test_generator_integration.py
- test_generator_test_suite.py
- test_generator_fixed.py
- test_generator_integration.py

### Duckdb Tests

- test_duckdb_api.py
- test_duckdb_api.py
- test_duckdb_integration.py
- test_duckdb_integration.py
- test_duckdb_integration.py
- test_duckdb_integration.py

### Whisper Tests

- test_whisper.py
- test_whisper.py
- test_whisper.py
- test_whisper.py
- test_whisper.py
- test_whisper.py

### Worker Tests

- test_worker_reconnection_integration.py
- test_worker_auto_discovery_with_ci.py
- test_worker.py
- test_worker_thermal_management.py
- test_worker_reconnection.py
- test_worker_reconnection_integration.py

### Monitoring Tests

- test_monitoring_dashboard_integration.py
- test_monitoring.py
- test_monitoring.py
- test_monitoring_dashboard_integration.py
- test_monitoring_dashboard.py
- test_monitoring_dashboard.py

### Check Tests

- test_check_copies.py
- test_check_dummies.py
- test_check_docstrings.py
- test_check_copies.py
- test_check_docstrings.py
- test_check_dummies.py

### Distributed Tests

- test_distributed_testing_integration.py
- test_distributed_coordinator.py
- test_distributed_error_handler.py
- test_distributed_coordinator.py
- test_distributed_error_handler.py

### Template Tests

- test_template_db_migration.py
- test_template_generator.py
- test_template_system.py
- test_template_enhancements.py
- test_template_generator.py

### Db Tests

- test_db_integration.py
- test_db_performance_metrics.py
- test_db_performance.py
- test_db_performance_optimization.py
- test_db_integration.py

### Dynamic Tests

- test_dynamic_module_utils.py
- test_dynamic_resource_manager.py
- test_dynamic_module_utils.py
- test_dynamic_resource_management_visualization.py
- test_dynamic_resource_manager.py

### Utils Tests

- test_utils.py
- test_utils.py
- test_utils.py
- test_utils.py
- test_utils.py

### Validation Tests

- test_validation_reporter.py
- test_validation.py
- test_validation_reporter.py
- test_validation_dashboard.py

### Simulation Tests

- test_simulation_validation_foundation.py
- test_simulation_drift_detector.py
- test_simulation_statistical_validator.py
- test_simulation_calibrator.py

### Browser Tests

- test_browser_environment_validation.py
- test_browser_performance_optimizer.py
- test_browser_failure_injector.py
- test_browser_recovery_strategies.py

### Webgpu Tests

- test_webgpu_ulp_demo.py
- test_webgpu_matmul.py
- test_webgpu_ulp_demo.py
- test_webgpu_detection.py

### Ci Tests

- test_ci_integration.py
- test_ci_clients.py
- test_ci_client_implementations.py
- test_ci_integration.py

### Circuit Tests

- test_circuit_breaker_integration.py
- test_circuit_breaker_integration.py
- test_circuit_breaker.py
- test_circuit_breaker_visualization.py

### E2e Tests

- test_e2e_visualization_db_integration.py
- test_e2e_integrated_system.py
- test_e2e_visualization_db_integration.py
- test_comprehensive_e2e.py

### Backbone Tests

- test_backbone_common.py
- test_backbone_utils.py
- test_backbone_common.py
- test_backbone_utils.py

### Activations Tests

- test_activations.py
- test_activations_tf.py
- test_activations.py
- test_activations_tf.py

### Import Tests

- test_import_structure.py
- test_import_utils.py
- test_import_structure.py
- test_import_utils.py

### Fsdp Tests

- test_fsdp.py
- test_fsdp.py
- test_fsdp.py
- test_fsdp.py

### Optimization Tests

- test_optimization_tf.py
- test_optimization.py
- test_optimization_tf.py
- test_optimization.py

### Beam Tests

- test_beam_search.py
- test_beam_constraints.py
- test_beam_constraints.py
- test_beam_search.py

### Compressed Tests

- test_compressed_tensors.py
- test_compressed_models.py
- test_compressed_tensors.py
- test_compressed_models.py

### Cross Tests

- test_cross_model_tensor_sharing.py
- test_cross_browser_model_sharding.py
- test_cross_platform_worker_support.py

### Selenium Tests

- test_selenium_import.py
- test_selenium_browser_integration.py
- test_selenium_browser_integration.py

### Peft Tests

- test_peft_integration.py
- test_peft_integration.py
- test_peft_integration.py

### Reporter Tests

- test_reporter_artifact_integration.py
- test_reporter_artifact_integration.py
- test_reporter.py

### Sound Tests

- test_sound_notification_integration.py
- test_sound_files.py
- test_sound_notification_integration.py

### Single Tests

- test_single_node_gpu.py
- test_single_model_hardware.py
- test_single_node_gpu.py

### Data Tests

- test_data_collator.py
- test_data_collator.py
- test_data_generator.py

### Ollama Tests

- test_ollama_backoff_comprehensive.py
- test_ollama_mock.py
- test_ollama_backoff.py

### Result Tests

- test_result_aggregator.py
- test_result_aggregator.py
- test_basic_result_aggregator.py

### Performance Tests

- test_performance_trend_analyzer.py
- test_performance_optimizer.py
- test_performance_trend_analyzer.py

### Improved Tests

- test_improved_renderer.py
- test_improved_converter.py

### Database Tests

- test_database_predictive_analytics.py
- test_database_predictive_analytics.py

### Fast Tests

- test_fast_api.py
- test_fast_api.py

### Mock Tests

- test_mock_detection_visualization.py
- test_mock_detection.py

### Samsung Tests

- test_samsung_npu_basic.py
- test_samsung_npu_comparison.py

### Flax Tests

- test_flax_examples.py
- test_flax_examples.py

### Accelerate Tests

- test_accelerate_examples.py
- test_accelerate_examples.py

### Pytorch Tests

- test_pytorch_examples.py
- test_pytorch_examples.py

### Tensorflow Tests

- test_tensorflow_examples.py
- test_tensorflow_examples.py

### Sequence Tests

- test_sequence_feature_extraction_common.py
- test_sequence_feature_extraction_common.py

### Training Tests

- test_training_args.py
- test_training_args.py

### Pipeline Tests

- test_pipeline_mixin.py
- test_pipeline_mixin.py

### Logging Tests

- test_logging.py
- test_logging.py

### Cli Tests

- test_cli.py
- test_cli.py

### Doc Tests

- test_doc_samples.py
- test_doc_samples.py

### Versions Tests

- test_versions_utils.py
- test_versions_utils.py

### Cache Tests

- test_cache_utils.py
- test_cache_utils.py

### Chat Tests

- test_chat_template_utils.py
- test_chat_template_utils.py

### Skip Tests

- test_skip_decorators.py
- test_skip_decorators.py

### Audio Tests

- test_audio_utils.py
- test_audio_utils.py

### Deprecation Tests

- test_deprecation.py
- test_deprecation.py

### Add Tests

- test_add_new_model_like.py
- test_add_new_model_like.py

### Hub Tests

- test_hub_utils.py
- test_hub_utils.py

### Offline Tests

- test_offline.py
- test_offline.py

### File Tests

- test_file_utils.py
- test_file_utils.py

### Generic Tests

- test_generic.py
- test_generic.py

### Retrieval Tests

- test_retrieval_rag.py
- test_retrieval_rag.py

### Final Tests

- test_final_answer.py
- test_final_answer.py

### Text Tests

- test_text_to_speech.py
- test_text_to_speech.py

### Translation Tests

- test_translation.py
- test_translation.py

### Search Tests

- test_search.py
- test_search.py

### Tools Tests

- test_tools_common.py
- test_tools_common.py

### Agents Tests

- test_agents.py
- test_agents.py

### Agent Tests

- test_agent_types.py
- test_agent_types.py

### Document Tests

- test_document_question_answering.py
- test_document_question_answering.py

### Speech Tests

- test_speech_to_text.py
- test_speech_to_text.py

### Python Tests

- test_python_interpreter.py
- test_python_interpreter.py

### Tensor Tests

- test_tensor_parallel.py
- test_tensor_parallel.py

### Candidate Tests

- test_candidate_generator.py
- test_candidate_generator.py

### Streamers Tests

- test_streamers.py
- test_streamers.py

### Logits Tests

- test_logits_process.py
- test_logits_process.py

### Stopping Tests

- test_stopping_criteria.py
- test_stopping_criteria.py

### Aqlm Tests

- test_aqlm.py
- test_aqlm.py

### Quanto Tests

- test_quanto.py
- test_quanto.py

### Torchao Tests

- test_torchao.py
- test_torchao.py

### Higgs Tests

- test_higgs.py
- test_higgs.py

### Vptq Tests

- test_vptq.py
- test_vptq.py

### Hqq Tests

- test_hqq.py
- test_hqq.py

### Awq Tests

- test_awq.py
- test_awq.py

### Eetq Tests

- test_eetq.py
- test_eetq.py

### Mixed Tests

- test_mixed_int8.py
- test_mixed_int8.py

### 4bit Tests

- test_4bit.py
- test_4bit.py

### Ggml Tests

- test_ggml.py
- test_ggml.py

### Fbgemm Tests

- test_fbgemm_fp8.py
- test_fbgemm_fp8.py

### Gptq Tests

- test_gptq.py
- test_gptq.py

### Fp8 Tests

- test_fp8.py
- test_fp8.py

### Spqr Tests

- test_spqr.py
- test_spqr.py

### Bitnet Tests

- test_bitnet.py
- test_bitnet.py

### Get Tests

- test_get_test_info.py
- test_get_test_info.py

### Tests Tests

- test_tests_fetcher.py
- test_tests_fetcher.py

### Conversion Tests

- test_conversion_order.py
- test_conversion_order.py

### Deepspeed Tests

- test_deepspeed.py
- test_deepspeed.py

### Llama Tests

- test_llama.py
- test_llama.py

### Batch Tests

- test_batch_generator.py
- test_batch_generator_minimal.py

### Artifact Tests

- test_artifact_url_retrieval.py
- test_artifact_handling.py

### Dependency Tests

- test_dependency_manager.py
- test_dependency_manager.py

### Integrated Tests

- test_integrated_analysis_system.py
- test_integrated_visualization_reports.py

### Autodoc Tests

- test_autodoc.py
- test_autodoc.py

### Style Tests

- test_style_doc.py
- test_style_doc.py

### Build Tests

- test_build_doc.py
- test_build_doc.py

### Unified Tests

- test_unified_component_tester.py
- test_unified_test_db.py

### Calibrator Tests

- test_advanced_calibrator.py
- test_advanced_calibrator.py

### Benchmark Tests

- test_benchmark_validation.py
- test_benchmark.py

### Regression Tests

- test_regression_visualization.py
- test_regression_detection.py

### Auto Tests

- test_auto_recovery.py
- test_auto_recovery_system.py

### End Tests

- test_end_to_end_fault_tolerance.py
- test_end_to_end_framework.py

## Identified Issues

1. **Inconsistent Base Classes**: Multiple inheritance patterns without standardization
2. **Duplicate Test Methods**: Same test methods implemented across multiple files
3. **Inconsistent Naming Conventions**: Mixed naming patterns for test methods
4. **Redundant Fixtures**: Similar setup/teardown methods duplicated across files
5. **Size Distribution Issues**: Some tests are too large, others too small
6. **Directory Organization**: Tests scattered across multiple directories without clear organization

## Comprehensive Refactoring Recommendations

### 1. Standardize Test Structure

Create a hierarchy of base test classes:

- : Core functionality for all tests
- : Specialized functionality for ML model testing
- : Specialized functionality for browser testing
- : Specialized functionality for hardware compatibility testing
- : Specialized functionality for API testing

### 2. Implement Consistent Naming Conventions

- Adopt a clear naming convention for test methods (e.g.,  or )
- Group tests by functionality, not implementation
- Use descriptive names that clearly indicate what is being tested

### 3. Extract Common Test Utilities

Create shared utility modules:

- : Common setup and teardown functionality
- : Standard mock objects and factories
- : Custom assertion helpers
- : Test data generation utilities

### 4. Reorganize Directory Structure

Organize tests into a more logical structure:

- : Unit tests for individual components
- : Integration tests between components
- : Hardware-specific tests
- : Browser-specific tests
- : ML model tests
- : End-to-end tests

### 5. Consolidate Duplicate Tests

- Identify and merge duplicate test implementations
- Create parameterized tests for similar functionality across different models/components
- Develop a test registry to track test coverage and prevent duplication

### 6. Implement Test Size Standards

- Limit test methods to 10-20 lines when possible
- Extract helper methods for complex setup/assertions
- Use composition instead of inheritance for test reuse

### 7. Develop Deprecation Strategy

- Identify tests that are no longer relevant or redundant
- Create a migration path for deprecating old tests
- Document reasons for deprecation

## Implementation Plan

### Phase 1: Foundation (2 weeks)

1. Create base test classes and utilities
2. Develop naming convention guidelines
3. Implement directory structure reorganization

### Phase 2: Migration (3 weeks)

1. Convert high-priority tests to new structure
2. Consolidate duplicate tests
3. Implement parameterized testing

### Phase 3: Cleanup (2 weeks)

1. Deprecate unnecessary tests
2. Refine documentation
3. Create automated enforcement of test standards

### Phase 4: Validation (1 week)

1. Verify test coverage is maintained
2. Ensure all tests pass consistently
3. Measure performance improvements
