# Hugging Face Model Test Coverage Report

*Generated on: 2025-03-01*

## Implementation Status

- **Total Model Types**: 299
- **Implemented Tests**: 104 (34.8%)
- **Remaining to Implement**: 195 (65.2%)

## Pipeline Task Coverage

| Pipeline Task | Implemented | Total | Coverage | Key Missing Models |
|---------------|-------------|-------|----------|-------------------|
| object-detection | 2 | 13 | 15.4% | vitdet, conditional_detr, deformable_detr |
| image-classification | 14 | 51 | 27.5% | resnet, flava, swinv2 |
| document-question-answering | 3 | 10 | 30.0% | donut-swin, markuplm, bros |
| text-generation | 25 | 74 | 33.8% | gpt_neox, rwkv, gpt_neox_japanese |
| image-segmentation | 4 | 11 | 36.4% | oneformer, clipseg, maskformer |
| image-to-text | 12 | 32 | 37.5% | donut-swin, align, blip-2 |
| text-classification | 14 | 36 | 38.9% | canine, megatron-bert, rembert |
| visual-question-answering | 14 | 35 | 40.0% | visual_bert, flava, altclip |
| text2text-generation | 8 | 20 | 40.0% | longt5, switch_transformers, bigbird_pegasus |
| text-to-audio | 6 | 15 | 40.0% | qwen2_audio, univnet, clvp |
| fill-mask | 13 | 32 | 40.6% | big_bird, megatron-bert, rembert |
| audio-classification | 6 | 14 | 42.9% | audio-spectrogram-transformer, unispeech-sat, dac |
| automatic-speech-recognition | 9 | 21 | 42.9% | qwen2_audio, speech_to_text, unispeech-sat |
| translation_XX_to_YY | 6 | 13 | 46.2% | nllb-moe, fsmt, lilt |
| feature-extraction | 14 | 29 | 48.3% | flava, realm, align |
| depth-estimation | 2 | 4 | 50.0% | zoedepth, glpn |
| summarization | 6 | 12 | 50.0% | longt5, bigbird_pegasus, pegasus_x |
| question-answering | 12 | 20 | 60.0% | big_bird, realm, luke |
| token-classification | 13 | 21 | 61.9% | canine, markuplm, funnel |
| time-series-prediction | 4 | 5 | 80.0% | patchtsmixer |
| table-question-answering | 1 | 1 | 100.0% | None |
| sentiment-analysis | 3 | 3 | 100.0% | None |
| conversational | 2 | 2 | 100.0% | None |
| protein-folding | 1 | 1 | 100.0% | None |
| zero-shot-classification | 4 | 4 | 100.0% | None |

## Next Steps

### Critical Models to Implement

The following models should be implemented first as they provide unique capabilities or fill gaps in pipeline coverage:

- **big_bird**: fill-mask, question-answering
- **canine**: token-classification, text-classification
- **donut-swin**: document-question-answering, image-to-text
- **gpt_neox**: text-generation
- **markuplm**: token-classification, document-question-answering
- **qwen2_audio**: automatic-speech-recognition, text-to-audio
- **resnet**: image-classification
- **rwkv**: text-generation
- **visual_bert**: visual-question-answering
- **zoedepth**: depth-estimation

### High Priority Models

These models are widely used and should be implemented after the critical models:

- **audio-spectrogram-transformer**: audio-classification
- **flava**: image-classification, feature-extraction, visual-question-answering
- **gpt_neox_japanese**: text-generation
- **longt5**: text2text-generation, summarization
- **megatron-bert**: fill-mask, text-classification
- **nllb-moe**: translation_XX_to_YY
- **oneformer**: image-segmentation
- **realm**: feature-extraction, question-answering
- **rembert**: fill-mask, text-classification
- **speech_to_text**: automatic-speech-recognition
- *...and 7 more high priority models*

## Implementation Timeline

1. **Phase 1 (Critical Models)**: Focus on models that provide unique capabilities or fill gaps in pipeline coverage
2. **Phase 2 (High Priority Models)**: Implement widely used models across various tasks
3. **Phase 3 (Medium Priority Models)**: Complete coverage for remaining models

To generate tests for the next batch of models, run:

```bash
python3 complete_test_coverage.py --batch 5 --priority critical
```

## Recently Generated Tests

*Error listing recent tests: name 'model_to_pipeline' is not defined*
