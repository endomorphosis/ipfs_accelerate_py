# Hugging Face Model Test Coverage Report

*Generated on: 2025-03-01*

## Implementation Status

- **Total Model Types**: 299
- **Implemented Tests**: 115 (38.5%)
- **Remaining to Implement**: 184 (61.5%)

## Pipeline Task Coverage

| Pipeline Task | Implemented | Total | Coverage | Key Missing Models |
|---------------|-------------|-------|----------|-------------------|
| object-detection | 2 | 13 | 15.4% | vitdet, conditional_detr, deformable_detr |
| image-classification | 15 | 51 | 29.4% | flava, swinv2, vit_mae |
| text-generation | 27 | 74 | 36.5% | gpt_neox_japanese, bert-generation, biogpt |
| text2text-generation | 8 | 20 | 40.0% | longt5, switch_transformers, bigbird_pegasus |
| image-to-text | 13 | 32 | 40.6% | align, blip-2, chameleon |
| text-classification | 15 | 36 | 41.7% | megatron-bert, rembert, convbert |
| visual-question-answering | 15 | 35 | 42.9% | flava, altclip, blip-2 |
| audio-classification | 6 | 14 | 42.9% | audio-spectrogram-transformer, unispeech-sat, dac |
| fill-mask | 14 | 32 | 43.8% | megatron-bert, rembert, convbert |
| image-segmentation | 5 | 11 | 45.5% | oneformer, clipseg, maskformer |
| translation_XX_to_YY | 6 | 13 | 46.2% | nllb-moe, fsmt, lilt |
| text-to-audio | 7 | 15 | 46.7% | univnet, clvp, fastspeech2_conformer |
| automatic-speech-recognition | 10 | 21 | 47.6% | speech_to_text, unispeech-sat, mctct |
| feature-extraction | 14 | 29 | 48.3% | flava, realm, align |
| document-question-answering | 5 | 10 | 50.0% | bros, layoutlmv2, nougat |
| summarization | 6 | 12 | 50.0% | longt5, bigbird_pegasus, pegasus_x |
| question-answering | 13 | 20 | 65.0% | realm, luke, qdqbert |
| token-classification | 15 | 21 | 71.4% | funnel, graphormer, layoutlmv2 |
| depth-estimation | 3 | 4 | 75.0% | glpn |
| time-series-prediction | 4 | 5 | 80.0% | patchtsmixer |
| zero-shot-classification | 4 | 4 | 100.0% | None |
| protein-folding | 1 | 1 | 100.0% | None |
| conversational | 2 | 2 | 100.0% | None |
| sentiment-analysis | 3 | 3 | 100.0% | None |
| table-question-answering | 1 | 1 | 100.0% | None |

## Next Steps

### Critical Models to Implement

The following models should be implemented first as they provide unique capabilities or fill gaps in pipeline coverage:


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
