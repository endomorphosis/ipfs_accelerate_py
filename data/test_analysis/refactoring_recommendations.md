# Test Refactoring Recommendations

Generated: 2025-03-27 21:15:27

## Summary

- Total test files analyzed: 2838
- Total classes: 6427
- Total test methods: 27356
- Similar class groups identified: 58611
- Inheritance clusters identified: 49

## Common Patterns

### Common Base Classes

| Base Class | Usage Count |
|------------|-------------|
| unittest.TestCase | 3385 |
| ModelTesterMixin | 746 |
| PipelineTesterMixin | 657 |
| GenerationTesterMixin | 276 |
| TokenizerTesterMixin | 195 |
| TFModelTesterMixin | 163 |
| ImageProcessingTestMixin | 143 |
| ProcessorTesterMixin | 102 |
| FlaxModelTesterMixin | 79 |
| TestCasePlus | 70 |

### Common Test Framework Imports

| Import | Usage Count |
|--------|-------------|
| unittest | 1967 |
| unittest.mock | 1344 |
| pytest | 247 |
| _pytest.doctest | 14 |
| unittest.util | 6 |
| _pytest.outcomes | 2 |
| _pytest.pathlib | 2 |

### Common Test Method Names

| Method Name | Usage Count |
|------------|-------------|
| test_config | 856 |
| test_model_from_pretrained | 632 |
| test_model | 584 |
| test_inputs_embeds | 521 |
| test_pipeline | 417 |
| test_model_get_set_embeddings | 333 |
| test_initialization | 308 |
| test_hidden_states_output | 289 |
| test_forward_signature | 275 |
| test_training_gradient_checkpointing | 253 |
| test_retain_grad_hidden_states_attentions | 251 |
| test_training_gradient_checkpointing_use_reentrant | 229 |
| test_training_gradient_checkpointing_use_reentrant_false | 229 |
| test_attention_outputs | 211 |
| test_from_pretrained | 198 |

## Most Similar Classes

These classes have high similarity and may be candidates for consolidation.

### Similarity Group 1 (Similarity: 1.00)

- Class 1: `test_hf_pythia.py:TestPythiaModels`
- Class 2: `test_hf_wavlm.py:TestWavlmModels`

Common methods:
- `run_tests`
- `test_pipeline`
- `__init__`

### Similarity Group 2 (Similarity: 1.00)

- Class 1: `test_hf_pythia.py:TestPythiaModels`
- Class 2: `test_hf_speech_to_text.py:TestSpeechToTextModels`

Common methods:
- `run_tests`
- `test_pipeline`
- `__init__`

### Similarity Group 3 (Similarity: 1.00)

- Class 1: `test_hf_pythia.py:TestPythiaModels`
- Class 2: `test_hf_unispeech.py:TestUnispeechModels`

Common methods:
- `run_tests`
- `test_pipeline`
- `__init__`

### Similarity Group 4 (Similarity: 1.00)

- Class 1: `test_hf_pythia.py:TestPythiaModels`
- Class 2: `test_hf_layoutlm.py:TestLayoutlmModels`

Common methods:
- `run_tests`
- `test_pipeline`
- `__init__`

### Similarity Group 5 (Similarity: 1.00)

- Class 1: `test_hf_pythia.py:TestPythiaModels`
- Class 2: `test_hf_xlm.py:TestXlmModels`

Common methods:
- `run_tests`
- `test_pipeline`
- `__init__`

### Similarity Group 6 (Similarity: 1.00)

- Class 1: `test_hf_pythia.py:TestPythiaModels`
- Class 2: `test_hf_swin.py:TestSwinModels`

Common methods:
- `run_tests`
- `test_pipeline`
- `__init__`

### Similarity Group 7 (Similarity: 1.00)

- Class 1: `test_hf_pythia.py:TestPythiaModels`
- Class 2: `test_hf_swinv2.py:TestSwinv2Models`

Common methods:
- `run_tests`
- `test_pipeline`
- `__init__`

### Similarity Group 8 (Similarity: 1.00)

- Class 1: `test_hf_pythia.py:TestPythiaModels`
- Class 2: `test_hf_sew.py:TestSewModels`

Common methods:
- `run_tests`
- `test_pipeline`
- `__init__`

### Similarity Group 9 (Similarity: 1.00)

- Class 1: `test_hf_pythia.py:TestPythiaModels`
- Class 2: `test_hf_umt5.py:TestUmt5Models`

Common methods:
- `run_tests`
- `test_pipeline`
- `__init__`

### Similarity Group 10 (Similarity: 1.00)

- Class 1: `test_hf_pythia.py:TestPythiaModels`
- Class 2: `test_hf_plbart.py:TestPlbartModels`

Common methods:
- `run_tests`
- `test_pipeline`
- `__init__`

## Inheritance Clusters

These groups of classes are related through inheritance and may benefit from standardization.

### Cluster 1 (Size: 486)

Classes in this cluster:
- `test_modeling_dpt_auto_backbone.py:DPTModelTest`
- `test_modeling_unispeech_sat.py:UniSpeechSatModelTest`
- `test_modeling_siglip2.py:Siglip2ModelTesterMixin`
- `test_modeling_deberta_v2.py:DebertaV2ModelTest`
- `test_modeling_t5.py:T5ModelTest`
- `test_modeling_tf_blenderbot_small.py:TFBlenderbotSmallModelTest`
- `test_modeling_blip.py:BlipModelTest`
- `test_modeling_rwkv.py:RwkvModelTest`
- `test_modeling_wav2vec2.py:Wav2Vec2ModelTest`
- `test_modeling_clipseg.py:CLIPSegVisionModelTest`
- `test_modeling_depth_pro.py:DepthProModelTest`
- `test_modeling_mixtral.py:MixtralModelTest`
- `test_modeling_tf_lxmert.py:TFLxmertModelTest`
- `test_modeling_dinov2_with_registers.py:Dinov2WithRegistersBackboneTest`
- `test_modeling_superglue.py:SuperGlueModelTest`
- `test_modeling_clap.py:ClapTextModelTest`
- `test_modeling_tf_vit_mae.py:TFViTMAEModelTest`
- `test_modeling_tf_wav2vec2.py:TFWav2Vec2ModelTest`
- `test_modeling_mbart.py:MBartStandaloneDecoderModelTest`
- `test_modeling_wav2vec2_conformer.py:Wav2Vec2ConformerModelTest`
- `test_modeling_unispeech.py:UniSpeechRobustModelTest`
- `test_modeling_vit_msn.py:ViTMSNModelTest`
- `test_modeling_tf_deberta.py:TFDebertaModelTest`
- `test_modeling_tf_distilbert.py:TFDistilBertModelTest`
- `test_modeling_bark.py:BarkFineModelTest`
- `test_modeling_idefics3.py:Idefics3ForConditionalGenerationModelTest`
- `test_modeling_gptj.py:GPTJModelTest`
- `test_modeling_dab_detr.py:DabDetrModelTest`
- `test_modeling_tf_led.py:TFLEDModelTest`
- `test_modeling_flava.py:FlavaForPreTrainingTest`
- `test_modeling_swinv2.py:Swinv2ModelTest`
- `test_modeling_cvt.py:CvtModelTest`
- `test_modeling_efficientnet.py:EfficientNetModelTest`
- `test_modeling_tf_tapas.py:TFTapasModelTest`
- `test_modeling_glpn.py:GLPNModelTest`
- `test_modeling_funnel.py:FunnelModelTest`
- `test_modeling_mobilenet_v1.py:MobileNetV1ModelTest`
- `test_modeling_gpt_neo.py:GPTNeoModelTest`
- `test_modeling_clvp.py:ClvpModelForConditionalGenerationTest`
- `test_modeling_superpoint.py:SuperPointModelTest`
- `test_modeling_x_clip.py:XCLIPTextModelTest`
- `test_modeling_convnext.py:ConvNextModelTest`
- `test_modeling_depth_anything.py:DepthAnythingModelTest`
- `test_modeling_owlvit.py:OwlViTVisionModelTest`
- `test_modeling_musicgen.py:MusicgenTest`
- `test_modeling_canine.py:CanineModelTest`
- `test_modeling_tf_vit.py:TFViTModelTest`
- `test_modeling_bark.py:BarkCoarseModelTest`
- `test_modeling_sew_d.py:SEWDModelTest`
- `test_modeling_blip_2.py:Blip2TextModelWithProjectionTest`
- `test_modeling_blip.py:BlipTextModelTest`
- `test_modeling_encodec.py:EncodecModelTest`
- `test_modeling_bit.py:BitBackboneTest`
- `test_modeling_m2m_100.py:M2M100ModelTest`
- `test_modeling_bart.py:BartStandaloneDecoderModelTest`
- `test_modeling_swiftformer.py:SwiftFormerModelTest`
- `test_modeling_siglip.py:SiglipForImageClassificationModelTest`
- `test_modeling_seamless_m4t_v2.py:SeamlessM4Tv2ModelWithSpeechInputTest`
- `test_modeling_megatron_bert.py:MegatronBertModelTest`
- `test_modeling_led.py:LEDModelTest`
- `test_modeling_perceiver.py:PerceiverModelTest`
- `test_modeling_tf_deit.py:TFDeiTModelTest`
- `test_modeling_tf_xlm.py:TFXLMModelTest`
- `test_modeling_tf_marian.py:TFMarianModelTest`
- `test_modeling_tf_blip.py:TFBlipVQAModelTest`
- `test_modeling_maskformer_swin.py:MaskFormerSwinModelTest`
- `test_modeling_mgp_str.py:MgpstrModelTest`
- `test_modeling_donut_swin.py:DonutSwinModelTest`
- `test_modeling_owlvit.py:OwlViTModelTest`
- `test_modeling_tf_wav2vec2.py:TFWav2Vec2RobustModelTest`
- `test_modeling_hubert.py:HubertRobustModelTest`
- `test_modeling_deformable_detr.py:DeformableDetrModelTest`
- `test_modeling_gemma.py:GemmaModelTest`
- `test_modeling_autoformer.py:AutoformerModelTest`
- `test_modeling_siglip.py:SiglipModelTesterMixin`
- `test_modeling_tf_bart.py:TFBartModelTest`
- `test_modeling_idefics.py:IdeficsModelTest`
- `test_modeling_xlnet.py:XLNetModelTest`
- `test_modeling_fastspeech2_conformer.py:FastSpeech2ConformerModelTest`
- `test_modeling_speech_to_text.py:Speech2TextModelTest`
- `test_modeling_tf_albert.py:TFAlbertModelTest`
- `test_modeling_swinv2.py:Swinv2BackboneTest`
- `test_modeling_tf_blip.py:TFBlipTextImageModelTest`
- `test_modeling_flava.py:FlavaTextModelTest`
- `test_modeling_vilt.py:ViltForImagesAndTextClassificationModelTest`
- `test_modeling_clip.py:CLIPForImageClassificationModelTest`
- `test_modeling_emu3.py:Emu3Text2TextModelTest`
- `test_modeling_clip.py:CLIPModelTesterMixin`
- `test_modeling_mvp.py:MvpStandaloneDecoderModelTest`
- `test_modeling_tf_opt.py:TFOPTModelTest`
- `test_modeling_tf_data2vec_vision.py:TFData2VecVisionModelTest`
- `test_modeling_detr.py:DetrModelTest`
- `test_modeling_levit.py:LevitModelTest`
- `test_modeling_whisper.py:WhisperModelTest`
- `test_modeling_tf_openai.py:TFOpenAIGPTModelTest`
- `test_modeling_common.py:ModelTesterMixin`
- `test_modeling_bart.py:BartModelTest`
- `test_modeling_poolformer.py:PoolFormerModelTest`
- `test_modeling_musicgen_melody.py:MusicgenMelodyTest`
- `test_modeling_mask2former.py:Mask2FormerModelTest`
- `test_modeling_blenderbot_small.py:BlenderbotSmallModelTest`
- `test_modeling_visual_bert.py:VisualBertModelTest`
- `test_modeling_owlv2.py:Owlv2TextModelTest`
- `test_modeling_tf_groupvit.py:TFGroupViTModelTest`
- `test_modeling_blip_2.py:Blip2VisionModelWithProjectionTest`
- `test_modeling_x_clip.py:XCLIPModelTest`
- `test_modeling_tf_blip_text.py:BlipTextModelTest`
- `test_modeling_tf_groupvit.py:TFGroupViTTextModelTest`
- `test_modeling_longt5.py:LongT5TGlobalModelTest`
- `test_modeling_zoedepth.py:ZoeDepthModelTest`
- `test_modeling_mobilebert.py:MobileBertModelTest`
- `test_modeling_longt5.py:LongT5EncoderOnlyTGlobalModelTest`
- `test_modeling_layoutlm.py:LayoutLMModelTest`
- `test_modeling_blip.py:BlipVQAModelTest`
- `test_modeling_xmod.py:XmodModelTest`
- `test_modeling_pegasus_x.py:PegasusXStandaloneDecoderModelTest`
- `test_modeling_openai.py:OpenAIGPTModelTest`
- `test_modeling_udop.py:UdopEncoderOnlyModelTest`
- `test_modeling_bark.py:BarkSemanticModelTest`
- `test_modeling_vitpose.py:VitPoseModelTest`
- `test_modeling_instructblip.py:InstructBlipVisionModelTest`
- `test_modeling_swin.py:SwinModelTest`
- `test_modeling_mamba.py:MambaModelTest`
- `test_modeling_clvp.py:ClvpEncoderTest`
- `test_modeling_bros.py:BrosModelTest`
- `test_modeling_squeezebert.py:SqueezeBertModelTest`
- `test_modeling_tf_clip.py:TFCLIPVisionModelTest`
- `test_modeling_phimoe.py:PhimoeModelTest`
- `test_modeling_clap.py:ClapModelTest`
- `test_modeling_fastspeech2_conformer.py:FastSpeech2ConformerWithHifiGanTest`
- `test_modeling_gpt_bigcode.py:GPTBigCodeModelTest`
- `test_modeling_convnextv2.py:ConvNextV2ModelTest`
- `test_backbone_common.py:BackboneTesterMixin`
- `test_modeling_groupvit.py:GroupViTVisionModelTest`
- `test_modeling_gemma2.py:Gemma2ModelTest`
- `test_modeling_tf_roberta.py:TFRobertaModelTest`
- `test_modeling_speecht5.py:SpeechT5ModelTest`
- `test_modeling_regnet.py:RegNetModelTest`
- `test_modeling_tf_xglm.py:TFXGLMModelTest`
- `test_modeling_speecht5.py:SpeechT5ForTextToSpeechTest`
- `test_modeling_tf_blip.py:TFBlipModelTest`
- `test_modeling_prophetnet.py:ProphetNetStandaloneEncoderModelTest`
- `test_modeling_resnet.py:ResNetBackboneTest`
- `test_modeling_biogpt.py:BioGptModelTest`
- `test_modeling_beit.py:BeitModelTest`
- `test_modeling_electra.py:ElectraModelTest`
- `test_modeling_musicgen_melody.py:MusicgenMelodyDecoderTest`
- `test_modeling_vivit.py:VivitModelTest`
- `test_modeling_beit.py:BeitBackboneTest`
- `test_modeling_data2vec_audio.py:Data2VecAudioModelTest`
- `test_modeling_pegasus.py:PegasusModelTest`
- `test_modeling_vitpose_backbone.py:VitPoseBackboneModelTest`
- `test_modeling_siglip2.py:Siglip2ForImageClassificationModelTest`
- `test_modeling_table_transformer.py:TableTransformerModelTest`
- `test_modeling_tf_swiftformer.py:TFSwiftFormerModelTest`
- `test_modeling_jetmoe.py:JetMoeModelTest`
- `test_modeling_xlm_roberta_xl.py:XLMRobertaXLModelTest`
- `test_modeling_owlvit.py:OwlViTTextModelTest`
- `test_modeling_aya_vision.py:AyaVisionModelTest`
- `test_modeling_moonshine.py:MoonshineModelTest`
- `test_modeling_bloom.py:BloomModelTest`
- `test_modeling_tf_rembert.py:TFRemBertModelTest`
- `test_modeling_tf_funnel.py:TFFunnelModelTest`
- `test_modeling_starcoder2.py:Starcoder2ModelTest`
- `test_modeling_vitdet.py:VitDetModelTest`
- `test_modeling_esm.py:EsmModelTest`
- `test_modeling_sam.py:SamModelTest`
- `test_modeling_speecht5.py:SpeechT5ForSpeechToTextTest`
- `test_modeling_chinese_clip.py:ChineseCLIPModelTest`
- `test_modeling_blip.py:BlipTextImageModelTest`
- `test_modeling_tf_core.py:TFCoreModelTesterMixin`
- `test_modeling_flaubert.py:FlaubertModelTest`
- `test_modeling_tf_groupvit.py:TFGroupViTVisionModelTest`
- `test_modeling_marian.py:MarianStandaloneDecoderModelTest`
- `test_modeling_blip_2.py:Blip2ForConditionalGenerationDecoderOnlyTest`
- `test_modeling_jamba.py:JambaModelTest`
- `test_modeling_paligemma2.py:PaliGemma2ForConditionalGenerationModelTest`
- `test_modeling_audio_spectrogram_transformer.py:ASTModelTest`
- `test_modeling_siglip2.py:Siglip2TextModelTest`
- `test_modeling_stablelm.py:StableLmModelTest`
- `test_modeling_paligemma.py:PaliGemmaForConditionalGenerationModelTest`
- `test_modeling_tf_pegasus.py:TFPegasusModelTest`
- `test_modeling_sew.py:SEWModelTest`
- `test_utils.py:GenerationTesterMixin`
- `test_modeling_got_ocr2.py:GotOcr2ModelTest`
- `test_modeling_align.py:AlignTextModelTest`
- `test_modeling_owlv2.py:Owlv2ModelTest`
- `test_modeling_kosmos2.py:Kosmos2ModelTest`
- `test_modeling_distilbert.py:DistilBertModelTest`
- `test_modeling_vitmatte.py:VitMatteModelTest`
- `test_modeling_siglip2.py:Siglip2VisionModelTest`
- `test_modeling_tf_dpr.py:TFDPRModelTest`
- `test_modeling_altclip.py:AltCLIPTextModelTest`
- `test_modeling_instructblipvideo.py:InstructBlipVideoForConditionalGenerationDecoderOnlyTest`
- `test_modeling_tf_mobilebert.py:TFMobileBertModelTest`
- `test_modeling_dac.py:DacModelTest`
- `test_modeling_tf_bert.py:TFBertModelTest`
- `test_modeling_prophetnet.py:ProphetNetStandaloneDecoderModelTest`
- `test_modeling_tf_resnet.py:TFResNetModelTest`
- `test_modeling_git.py:GitModelTest`
- `test_modeling_fuyu.py:FuyuModelTest`
- `test_modeling_dinat.py:DinatModelTest`
- `test_modeling_seamless_m4t.py:SeamlessM4TModelWithTextInputTest`
- `test_modeling_marian.py:MarianModelTest`
- `test_modeling_tf_speech_to_text.py:TFSpeech2TextModelTest`
- `test_modeling_nystromformer.py:NystromformerModelTest`
- `test_modeling_llava_onevision.py:LlavaOnevisionForConditionalGenerationModelTest`
- `test_modeling_switch_transformers.py:SwitchTransformersModelTest`
- `test_modeling_instructblip.py:InstructBlipForConditionalGenerationDecoderOnlyTest`
- `test_modeling_qwen2.py:Qwen2ModelTest`
- `test_modeling_align.py:AlignModelTest`
- `test_modeling_fsmt.py:FSMTModelTest`
- `test_modeling_moshi.py:MoshiTest`
- `test_modeling_grounding_dino.py:GroundingDinoModelTest`
- `test_modeling_aria.py:AriaForConditionalGenerationModelTest`
- `test_modeling_ibert.py:IBertModelTest`
- `test_modeling_nllb_moe.py:NllbMoeModelTest`
- `test_modeling_idefics.py:IdeficsForVisionText2TextTest`
- `test_modeling_granitemoe.py:GraniteMoeModelTest`
- `test_modeling_tf_longformer.py:TFLongformerModelTest`
- `test_modeling_tf_segformer.py:TFSegformerModelTest`
- `test_modeling_roformer.py:RoFormerModelTest`
- `test_modeling_vitpose_backbone.py:VitPoseBackboneTest`
- `test_modeling_clip.py:CLIPModelTest`
- `test_modeling_zamba2.py:Zamba2ModelTest`
- `test_modeling_blenderbot.py:BlenderbotModelTest`
- `test_modeling_altclip.py:AltCLIPVisionModelTest`
- `test_modeling_tf_roformer.py:TFRoFormerModelTest`
- `test_modeling_blip.py:BlipVisionModelTest`
- `test_modeling_bigbird_pegasus.py:BigBirdPegasusModelTest`
- `test_modeling_siglip.py:SiglipVisionModelTest`
- `test_modeling_modernbert.py:ModernBertModelTest`
- `test_modeling_blip.py:BlipTextRetrievalModelTest`
- `test_modeling_phi3.py:Phi3ModelTest`
- `test_modeling_textnet.py:TextNetModelTest`
- `test_modeling_speecht5.py:SpeechT5ForSpeechToSpeechTest`
- `test_modeling_pegasus_x.py:PegasusXModelTest`
- `test_modeling_lilt.py:LiltModelTest`
- `test_modeling_mpt.py:MptModelTest`
- `test_modeling_resnet.py:ResNetModelTest`
- `test_modeling_mllama.py:MllamaForCausalLMModelTest`
- `test_modeling_tf_xlnet.py:TFXLNetModelTest`
- `test_modeling_tf_gptj.py:TFGPTJModelTest`
- `test_modeling_data2vec_vision.py:Data2VecVisionModelTest`
- `test_modeling_tvp.py:TVPModelTest`
- `test_modeling_imagegpt.py:ImageGPTModelTest`
- `test_modeling_hiera.py:HieraModelTest`
- `test_modeling_longt5.py:LongT5ModelTest`
- `test_modeling_qwen2_vl.py:Qwen2VLModelTest`
- `test_modeling_bert_generation.py:BertGenerationEncoderTest`
- `test_modeling_timesformer.py:TimesformerModelTest`
- `test_modeling_vitdet.py:VitDetBackboneTest`
- `test_modeling_tf_layoutlmv3.py:TFLayoutLMv3ModelTest`
- `test_modeling_rembert.py:RemBertModelTest`
- `test_modeling_bert.py:BertModelTest`
- `test_modeling_dbrx.py:DbrxModelTest`
- `test_modeling_plbart.py:PLBartModelTest`
- `test_modeling_qwen2_5_vl.py:Qwen2_5_VLModelTest`
- `test_modeling_data2vec_text.py:Data2VecTextModelTest`
- `test_modeling_switch_transformers.py:SwitchTransformersEncoderOnlyModelTest`
- `test_modeling_opt.py:OPTModelTest`
- `test_modeling_clip.py:CLIPTextModelTest`
- `test_modeling_yolos.py:YolosModelTest`
- `test_modeling_longformer.py:LongformerModelTest`
- `test_modeling_tf_ctrl.py:TFCTRLModelTest`
- `test_modeling_xlm.py:XLMModelTest`
- `test_modeling_llama.py:LlamaModelTest`
- `test_modeling_funnel.py:FunnelBaseModelTest`
- `test_modeling_vilt.py:ViltModelTest`
- `test_modeling_recurrent_gemma.py:RecurrentGemmaModelTest`
- `test_modeling_tf_layoutlm.py:TFLayoutLMModelTest`
- `test_modeling_tf_mistral.py:TFMistralModelTest`
- `test_modeling_deit.py:DeiTModelTest`
- `test_modeling_mistral3.py:Mistral3ModelTest`
- `test_modeling_pix2struct.py:Pix2StructModelTest`
- `test_modeling_fnet.py:FNetModelTest`
- `test_modeling_tf_swin.py:TFSwinModelTest`
- `test_modeling_vits.py:VitsModelTest`
- `test_modeling_falcon_mamba.py:FalconMambaModelTest`
- `test_modeling_cohere.py:CohereModelTest`
- `test_modeling_pvt_v2.py:PvtV2BackboneTest`
- `test_modeling_tf_blip.py:TFBlipTextRetrievalModelTest`
- `test_modeling_mvp.py:MvpModelTest`
- `test_modeling_phi.py:PhiModelTest`
- `test_modeling_bigbird_pegasus.py:BigBirdPegasusStandaloneDecoderModelTest`
- `test_modeling_tf_regnet.py:TFRegNetModelTest`
- `test_modeling_tf_mpnet.py:TFMPNetModelTest`
- `test_modeling_tf_esm.py:TFEsmModelTest`
- `test_modeling_tf_blip.py:TFBlipTextModelTest`
- `test_modeling_informer.py:InformerModelTest`
- `test_modeling_olmo2.py:Olmo2ModelTest`
- `test_modeling_tf_t5.py:TFT5EncoderOnlyModelTest`
- `test_modeling_llava_next_video.py:LlavaNextVideoForConditionalGenerationModelTest`
- `test_modeling_glm.py:GlmModelTest`
- `test_modeling_mt5.py:MT5ModelTest`
- `test_modeling_dinov2_with_registers.py:Dinov2WithRegistersModelTest`
- `test_modeling_flava.py:FlavaModelTest`
- `test_modeling_unispeech_sat.py:UniSpeechSatRobustModelTest`
- `test_modeling_swin2sr.py:Swin2SRModelTest`
- `test_modeling_tf_mobilevit.py:TFMobileViTModelTest`
- `test_modeling_hiera.py:HieraBackboneTest`
- `test_modeling_maskformer.py:MaskFormerModelTest`
- `test_modeling_wav2vec2_bert.py:Wav2Vec2BertModelTest`
- `test_modeling_esmfold.py:EsmFoldModelTest`
- `test_modeling_siglip.py:SiglipTextModelTest`
- `test_modeling_tapas.py:TapasModelTest`
- `test_modeling_prompt_depth_anything.py:PromptDepthAnythingModelTest`
- `test_modeling_wav2vec2.py:Wav2Vec2RobustModelTest`
- `test_modeling_prophetnet.py:ProphetNetModelTest`
- `test_modeling_tf_clip.py:TFCLIPTextModelTest`
- `test_modeling_patchtst.py:PatchTSTModelTest`
- `test_modeling_bridgetower.py:BridgeTowerModelTest`
- `test_modeling_granite.py:GraniteModelTest`
- `test_modeling_mpnet.py:MPNetModelTest`
- `test_modeling_llava.py:LlavaForConditionalGenerationModelTest`
- `test_modeling_nemotron.py:NemotronModelTest`
- `test_modeling_helium.py:HeliumModelTest`
- `test_modeling_diffllama.py:DiffLlamaModelTest`
- `test_modeling_tf_whisper.py:TFWhisperModelTest`
- `test_modeling_align.py:AlignVisionModelTest`
- `test_modeling_olmo.py:OlmoModelTest`
- `test_modeling_oneformer.py:OneFormerModelTest`
- `test_modeling_video_llava.py:VideoLlavaForConditionalGenerationModelTest`
- `test_modeling_tf_funnel.py:TFFunnelBaseModelTest`
- `test_modeling_clip.py:CLIPVisionModelTest`
- `test_modeling_emu3.py:Emu3Vision2TextModelTest`
- `test_modeling_gpt_neox_japanese.py:GPTNeoXModelJapaneseTest`
- `test_modeling_chameleon.py:ChameleonModelTest`
- `test_modeling_flava.py:FlavaMultimodalModelTest`
- `test_modeling_videomae.py:VideoMAEModelTest`
- `test_modeling_tf_clip.py:TFCLIPModelTest`
- `test_modeling_reformer.py:ReformerTesterMixin`
- `test_modeling_smolvlm.py:SmolVLMForConditionalGenerationModelTest`
- `test_modeling_zamba.py:ZambaModelTest`
- `test_modeling_umt5.py:UMT5EncoderOnlyModelTest`
- `test_modeling_cpmant.py:CpmAntModelTest`
- `test_modeling_textnet.py:TextNetBackboneTest`
- `test_modeling_reformer.py:ReformerLSHAttnModelTest`
- `test_modeling_idefics2.py:Idefics2ForConditionalGenerationModelTest`
- `test_modeling_blenderbot.py:BlenderbotStandaloneDecoderModelTest`
- `test_modeling_pvt_v2.py:PvtV2ModelTest`
- `test_modeling_blip_2.py:Blip2ModelTest`
- `test_modeling_albert.py:AlbertModelTest`
- `test_modeling_mamba2.py:Mamba2ModelTest`
- `test_modeling_splinter.py:SplinterModelTest`
- `test_modeling_layoutlmv2.py:LayoutLMv2ModelTest`
- `test_modeling_vipllava.py:VipLlavaForConditionalGenerationModelTest`
- `test_modeling_focalnet.py:FocalNetBackboneTest`
- `test_modeling_bit.py:BitModelTest`
- `test_modeling_qwen2_moe.py:Qwen2MoeModelTest`
- `test_modeling_pegasus.py:PegasusStandaloneDecoderModelTest`
- `test_modeling_t5.py:T5EncoderOnlyModelTest`
- `test_modeling_granitemoeshared.py:GraniteMoeSharedModelTest`
- `test_modeling_omdet_turbo.py:OmDetTurboModelTest`
- `test_modeling_blenderbot_small.py:BlenderbotSmallStandaloneDecoderModelTest`
- `test_modeling_groupvit.py:GroupViTModelTest`
- `test_modeling_llava_next.py:LlavaNextForConditionalGenerationModelTest`
- `test_modeling_bamba.py:BambaModelTest`
- `test_modeling_moshi.py:MoshiDecoderTest`
- `test_modeling_dpr.py:DPRModelTest`
- `test_modeling_seggpt.py:SegGptModelTest`
- `test_modeling_tf_cvt.py:TFCvtModelTest`
- `test_modeling_smolvlm.py:SmolVLMModelTest`
- `test_modeling_owlvit.py:OwlViTForObjectDetectionTest`
- `test_modeling_tf_sam.py:TFSamModelTest`
- `test_modeling_markuplm.py:MarkupLMModelTest`
- `test_modeling_git.py:GitVisionModelTest`
- `test_modeling_dinov2.py:Dinov2ModelTest`
- `test_modeling_chinese_clip.py:ChineseCLIPVisionModelTest`
- `test_modeling_idefics2.py:Idefics2ModelTest`
- `test_modeling_pop2piano.py:Pop2PianoModelTest`
- `test_modeling_mbart.py:MBartModelTest`
- `test_modeling_tf_hubert.py:TFHubertModelTest`
- `test_modeling_blip_text.py:BlipTextModelTest`
- `test_modeling_luke.py:LukeModelTest`
- `test_modeling_mistral.py:MistralModelTest`
- `test_modeling_tf_hubert.py:TFHubertRobustModelTest`
- `test_modeling_gemma3.py:Gemma3Vision2TextModelTest`
- `test_modeling_mllama.py:MllamaForConditionalGenerationModelTest`
- `test_modeling_chameleon.py:ChameleonVision2SeqModelTest`
- `test_modeling_siglip.py:SiglipModelTest`
- `test_modeling_ernie.py:ErnieModelTest`
- `test_modeling_x_clip.py:XCLIPVisionModelTest`
- `test_modeling_pix2struct.py:Pix2StructTextModelTest`
- `test_modeling_altclip.py:AltCLIPModelTest`
- `test_modeling_phi4_multimodal.py:Phi4MultimodalModelTest`
- `test_modeling_maskformer_swin.py:MaskFormerSwinBackboneTest`
- `test_modeling_udop.py:UdopModelTest`
- `test_modeling_tf_electra.py:TFElectraModelTest`
- `test_modeling_tf_blenderbot.py:TFBlenderbotModelTest`
- `test_modeling_tf_t5.py:TFT5ModelTest`
- `test_modeling_speecht5.py:SpeechT5HifiGanTest`
- `test_modeling_idefics3.py:Idefics3ModelTest`
- `test_modeling_rt_detr_v2.py:RTDetrV2ModelTest`
- `test_modeling_tf_convnext.py:TFConvNextModelTest`
- `test_modeling_time_series_transformer.py:TimeSeriesTransformerModelTest`
- `test_modeling_tf_convbert.py:TFConvBertModelTest`
- `test_modeling_clvp.py:ClvpDecoderTest`
- `test_modeling_tf_common.py:TFModelTesterMixin`
- `test_modeling_owlv2.py:Owlv2VisionModelTest`
- `test_modeling_tf_mbart.py:TFMBartModelTest`
- `test_modeling_roberta.py:RobertaModelTest`
- `test_modeling_dinat.py:DinatBackboneTest`
- `test_modeling_persimmon.py:PersimmonModelTest`
- `test_modeling_flava.py:FlavaImageCodebookTest`
- `test_modeling_seamless_m4t.py:SeamlessM4TModelWithSpeechInputTest`
- `test_modeling_mimi.py:MimiModelTest`
- `test_modeling_flava.py:FlavaImageModelTest`
- `test_modeling_umt5.py:UMT5ModelTest`
- `test_modeling_blip_2.py:Blip2VisionModelTest`
- `test_modeling_whisper.py:WhisperEncoderModelTest`
- `test_modeling_whisper.py:WhisperStandaloneDecoderModelTest`
- `test_modeling_deberta.py:DebertaModelTest`
- `test_modeling_xglm.py:XGLMModelTest`
- `test_modeling_gpt2.py:GPT2ModelTest`
- `test_pipeline_mixin.py:PipelineTesterMixin`
- `test_modeling_mobilevitv2.py:MobileViTV2ModelTest`
- `test_modeling_mt5.py:MT5EncoderOnlyModelTest`
- `test_modeling_tf_deberta_v2.py:TFDebertaModelTest`
- `test_modeling_ijepa.py:IJepaModelTest`
- `test_modeling_gpt_neox.py:GPTNeoXModelTest`
- `test_modeling_groupvit.py:GroupViTTextModelTest`
- `test_modeling_longt5.py:LongT5EncoderOnlyModelTest`
- `test_modeling_falcon.py:FalconModelTest`
- `test_modeling_cohere2.py:Cohere2ModelTest`
- `test_modeling_qwen2_audio.py:Qwen2AudioForConditionalGenerationModelTest`
- `test_modeling_dpt_hybrid.py:DPTModelTest`
- `test_modeling_trocr.py:TrOCRStandaloneDecoderModelTest`
- `test_modeling_timm_wrapper.py:TimmWrapperModelTest`
- `test_modeling_gpt_bigcode.py:GPTBigCodeMHAModelTest`
- `test_modeling_univnet.py:UnivNetModelTest`
- `test_modeling_plbart.py:PLBartStandaloneDecoderModelTest`
- `test_modeling_big_bird.py:BigBirdModelTest`
- `test_modeling_pvt.py:PvtModelTest`
- `test_modeling_instructblipvideo.py:InstructBlipVideoVisionModelTest`
- `test_modeling_owlv2.py:Owlv2ForObjectDetectionTest`
- `test_modeling_tf_blip.py:TFBlipVisionModelTest`
- `test_modeling_roberta_prelayernorm.py:RobertaPreLayerNormModelTest`
- `test_modeling_layoutlmv3.py:LayoutLMv3ModelTest`
- `test_modeling_clipseg.py:CLIPSegTextModelTest`
- `test_modeling_vit.py:ViTModelTest`
- `test_modeling_focalnet.py:FocalNetModelTest`
- `test_modeling_tf_idefics.py:TFIdeficsForVisionText2TextTest`
- `test_modeling_mobilenet_v2.py:MobileNetV2ModelTest`
- `test_modeling_rt_detr.py:RTDetrModelTest`
- `test_modeling_pix2struct.py:Pix2StructVisionModelTest`
- `test_modeling_dinov2.py:Dinov2BackboneTest`
- `test_modeling_blip_2.py:Blip2TextRetrievalModelTest`
- `test_modeling_wavlm.py:WavLMModelTest`
- `test_modeling_segformer.py:SegformerModelTest`
- `test_modeling_timm_backbone.py:TimmBackboneModelTest`
- `test_modeling_olmoe.py:OlmoeModelTest`
- `test_modeling_convbert.py:ConvBertModelTest`
- `test_modeling_chinese_clip.py:ChineseCLIPTextModelTest`
- `test_modeling_clap.py:ClapAudioModelTest`
- `test_modeling_tf_idefics.py:TFIdeficsModelTest`
- `test_modeling_dpt.py:DPTModelTest`
- `test_modeling_rt_detr_resnet.py:RTDetrResNetBackboneTest`
- `test_modeling_pvt_v2.py:PvtV2ModelTester`
- `test_modeling_tf_gpt2.py:TFGPT2ModelTest`
- `test_modeling_mra.py:MraModelTest`
- `test_modeling_conditional_detr.py:ConditionalDetrModelTest`
- `test_modeling_tf_flaubert.py:TFFlaubertModelTest`
- `test_modeling_seamless_m4t_v2.py:SeamlessM4Tv2ModelWithTextInputTest`
- `test_modeling_patchtsmixer.py:PatchTSMixerModelTest`
- `test_modeling_siglip2.py:Siglip2ModelTest`
- `test_modeling_mobilevit.py:MobileViTModelTest`
- `test_modeling_convnext.py:ConvNextBackboneTest`
- `test_modeling_tf_roberta_prelayernorm.py:TFRobertaPreLayerNormModelTest`
- `test_modeling_upernet.py:UperNetModelTest`
- `test_modeling_yoso.py:YosoModelTest`
- `test_modeling_hubert.py:HubertModelTest`
- `test_modeling_vit_mae.py:ViTMAEModelTest`
- `test_modeling_reformer.py:ReformerLocalAttnModelTest`
- `test_modeling_decision_transformer.py:DecisionTransformerModelTest`
- `test_modeling_roc_bert.py:RoCBertModelTest`
- `test_modeling_lxmert.py:LxmertModelTest`
- `test_modeling_colpali.py:ColPaliForRetrievalModelTest`
- `test_modeling_pixtral.py:PixtralVisionModelModelTest`
- `test_modeling_gemma3.py:Gemma3ModelTest`
- `test_modeling_ctrl.py:CTRLModelTest`
- `test_modeling_clipseg.py:CLIPSegModelTest`
- `test_modeling_swin.py:SwinBackboneTest`
- `test_modeling_tf_convnextv2.py:TFConvNextV2ModelTest`
- `test_modeling_codegen.py:CodeGenModelTest`
- `test_modeling_musicgen.py:MusicgenDecoderTest`

Common base classes:
- `ModelTesterMixin`
- `PipelineTesterMixin`
- `unittest.TestCase`
- `TFModelTesterMixin`
- `GenerationTesterMixin`
- `TFCoreModelTesterMixin`
- `ViltModelTest`
- `CLIPModelTesterMixin`
- `BackboneTesterMixin`
- `ReformerTesterMixin`
- `Siglip2ModelTesterMixin`
- `SiglipModelTesterMixin`
- `IdeficsModelTest`
- `TFIdeficsModelTest`
- `LongT5ModelTest`
- `LongT5EncoderOnlyModelTest`
- `CohereModelTest`
- `GPTBigCodeModelTest`
- `FlavaModelTest`
- `GemmaModelTest`

### Cluster 2 (Size: 97)

Classes in this cluster:
- `test_tokenization_electra.py:ElectraTokenizationTest`
- `test_tokenization_mobilebert.py:MobileBERTTokenizationTest`
- `test_tokenization_gemma.py:GemmaTokenizationTest`
- `test_tokenization_led.py:TestTokenizationLED`
- `test_tokenization_wav2vec2.py:Wav2Vec2CTCTokenizerTest`
- `test_tokenization_camembert.py:CamembertTokenizationTest`
- `test_tokenization_ctrl.py:CTRLTokenizationTest`
- `test_tokenization_bert_generation.py:BertGenerationTokenizationTest`
- `test_tokenization_big_bird.py:BigBirdTokenizationTest`
- `test_tokenization_deberta_v2.py:DebertaV2TokenizationTest`
- `test_tokenization_albert.py:AlbertTokenizationTest`
- `test_tokenization_siglip.py:SiglipTokenizationTest`
- `test_tokenization_prophetnet.py:ProphetNetTokenizationTest`
- `test_tokenization_dpr.py:DPRContextEncoderTokenizationTest`
- `test_tokenization_gpt2.py:GPT2TokenizationTest`
- `test_tokenization_reformer.py:ReformerTokenizationTest`
- `test_tokenization_herbert.py:HerbertTokenizationTest`
- `test_tokenization_vits.py:VitsTokenizerTest`
- `test_tokenization_lxmert.py:LxmertTokenizationTest`
- `test_tokenization_pegasus.py:BigBirdPegasusTokenizationTest`
- `test_tokenization_biogpt.py:BioGptTokenizationTest`
- `test_tokenization_funnel.py:FunnelTokenizationTest`
- `test_tokenization_mbart50.py:MBart50TokenizationTest`
- `test_tokenization_layoutlmv3.py:LayoutLMv3TokenizationTest`
- `test_tokenization_clip.py:CLIPTokenizationTest`
- `test_tokenization_dpr.py:DPRQuestionEncoderTokenizationTest`
- `test_tokenization_myt5.py:MyT5TokenizationTest`
- `test_tokenization_udop.py:UdopTokenizationTest`
- `test_tokenization_roc_bert.py:BertTokenizationTest`
- `test_tokenization_longformer.py:LongformerTokenizationTest`
- `test_tokenization_mbart.py:MBartTokenizationTest`
- `test_tokenization_fnet.py:FNetTokenizationTest`
- `test_tokenization_fastspeech2_conformer.py:FastSpeech2ConformerTokenizerTest`
- `test_tokenization_mluke.py:MLukeTokenizerTest`
- `test_tokenization_barthez.py:BarthezTokenizationTest`
- `test_tokenization_nougat.py:NougatTokenizationTest`
- `test_tokenization_gpt_neox_japanese.py:GPTNeoXJapaneseTokenizationTest`
- `test_tokenization_moshi.py:MoshiTokenizationTest`
- `test_tokenization_pegasus.py:PegasusTokenizationTest`
- `test_tokenization_mvp.py:TestTokenizationMvp`
- `test_tokenization_squeezebert.py:SqueezeBertTokenizationTest`
- `test_tokenization_speecht5.py:SpeechT5TokenizerTest`
- `test_tokenization_nllb.py:NllbTokenizationTest`
- `test_tokenization_rembert.py:RemBertTokenizationTest`
- `test_tokenization_cpmant.py:CPMAntTokenizationTest`
- `test_tokenization_bert_tf.py:BertTokenizationTest`
- `test_tokenization_plbart.py:PLBartTokenizationTest`
- `test_tokenization_xlnet.py:XLNetTokenizationTest`
- `test_tokenization_mpnet.py:MPNetTokenizerTest`
- `test_tokenization_qwen2.py:Qwen2TokenizationTest`
- `test_tokenization_splinter.py:SplinterTokenizationTest`
- `test_tokenization_gpt_sw3.py:GPTSw3TokenizationTest`
- `test_tokenization_clvp.py:ClvpTokenizationTest`
- `test_tokenization_seamless_m4t.py:SeamlessM4TTokenizationTest`
- `test_tokenization_fast.py:PreTrainedTokenizationFastTest`
- `test_tokenization_tapas.py:TapasTokenizationTest`
- `test_tokenization_layoutlmv2.py:LayoutLMv2TokenizationTest`
- `test_tokenization_bert_japanese.py:BertJapaneseTokenizationTest`
- `test_tokenization_blenderbot_small.py:BlenderbotSmallTokenizerTest`
- `test_tokenization_xlm.py:XLMTokenizationTest`
- `test_tokenization_cohere.py:CohereTokenizationTest`
- `test_tokenization_openai.py:OpenAIGPTTokenizationTestWithSpacy`
- `test_tokenization_phobert.py:PhobertTokenizationTest`
- `test_tokenization_bart.py:TestTokenizationBart`
- `test_tokenization_code_llama.py:CodeLlamaTokenizationTest`
- `test_tokenization_luke.py:LukeTokenizerTest`
- `test_tokenization_byt5.py:ByT5TokenizationTest`
- `test_tokenization_flaubert.py:FlaubertTokenizationTest`
- `test_tokenization_marian.py:MarianTokenizationTest`
- `test_tokenization_mgp_str.py:MgpstrTokenizationTest`
- `test_tokenization_codegen.py:CodeGenTokenizationTest`
- `test_tokenization_openai.py:OpenAIGPTTokenizationTest`
- `test_tokenization_roformer.py:RoFormerTokenizationTest`
- `test_tokenization_bert.py:BertTokenizationTest`
- `test_tokenization_fsmt.py:FSMTTokenizationTest`
- `test_tokenization_common.py:TokenizerTesterMixin`
- `test_tokenization_dpr.py:DPRReaderTokenizationTest`
- `test_tokenization_roberta.py:RobertaTokenizationTest`
- `test_tokenization_markuplm.py:MarkupLMTokenizationTest`
- `test_tokenization_deberta.py:DebertaTokenizationTest`
- `test_tokenization_bartpho.py:BartphoTokenizerTest`
- `test_tokenization_layoutlm.py:LayoutLMTokenizationTest`
- `test_tokenization_xlm_roberta.py:XLMRobertaTokenizationTest`
- `test_tokenization_wav2vec2_phoneme.py:Wav2Vec2PhonemeCTCTokenizerTest`
- `test_tokenization_bloom.py:BloomTokenizationTest`
- `test_tokenization_whisper.py:WhisperTokenizerTest`
- `test_tokenization_m2m_100.py:M2M100TokenizationTest`
- `test_tokenization_speech_to_text.py:SpeechToTextTokenizerTest`
- `test_tokenization_layoutxlm.py:LayoutXLMTokenizationTest`
- `test_tokenization_xglm.py:XGLMTokenizationTest`
- `test_tokenization_bert_japanese.py:BertJapaneseCharacterTokenizationTest`
- `test_tokenization_canine.py:CanineTokenizationTest`
- `test_tokenization_perceiver.py:PerceiverTokenizationTest`
- `test_tokenization_llama.py:LlamaTokenizationTest`
- `test_tokenization_bertweet.py:BertweetTokenizationTest`
- `test_tokenization_t5.py:T5TokenizationTest`
- `test_tokenization_distilbert.py:DistilBertTokenizationTest`

Common base classes:
- `TokenizerTesterMixin`
- `unittest.TestCase`
- `BertTokenizationTest`
- `OpenAIGPTTokenizationTest`

### Cluster 3 (Size: 84)

Classes in this cluster:
- `test_hf_blip.py:ModelTest`
- `test_hf_gpt2_standardized.py:ModelTest`
- `test_wav2vec2_base_960h.py:TestSpeechModel`
- `test_utils.py:TestUtils`
- `test_webgpu_detection.py:TestWebGPUDetection`
- `test_hf_bloom_standardized.py:TestBloomModels`
- `test_ipfs_accelerate_with_cross_browser.py:TestIPFSAcceleratedBrowserSharding`
- `test_single_model_hardware.py:TestSingleModelHardware`
- `test_ipfs_accelerate_webnn_webgpu.py:TestIPFSAccelerateWebNNWebGPU`
- `test_bert_fixed.py:ModelTest`
- `test_bert_simple.py:TestBertSimple`
- `test_hf_fuyu_standardized.py:ModelTest`
- `test_vit-base-patch16-224.py:TestVit_base_patch16_224VitModel`
- `test_hf_xclip_standardized.py:ModelTest`
- `test_hf_git_standardized.py:ModelTest`
- `test_bert_fixed.py:TestBertFixed`
- `test_hf_wav2vec2.py:TestWav2Vec2Models`
- `test_bert_simple.py:ModelTest`
- `test_whisper-tiny.py:ModelTest`
- `test_ollama_backoff.py:TestOllamaBackoff`
- `test_claude.py:ModelTest`
- `test_model_api.py:TestModelAPI`
- `test_claude.py:TestClaude`
- `test_bert_base.py:TestBertBaseModel`
- `test_api_backend.py:TestAPIBackend`
- `test_hf_git_standardized.py:TestGitModels`
- `test_hf_qwen2.py:TestQwen2Models`
- `test_hf_paligemma_standardized.py:TestPaligemmaModels`
- `test_hf_gpt2_standardized.py:TestGPT2`
- `refactored_encoder_decoder_template.py:TestEncoderDecoderModel`
- `test_hf_gemma_standardized.py:ModelTest`
- `refactored_vision_template.py:TestVitModel`
- `test_hf_t5.py:TestT5Models`
- `refactored_decoder_only_template.py:TestDecoderModel`
- `refactored_encoder_only_template.py:TestEncoderModel`
- `refactored_multimodal_template.py:TestMultimodalModel`
- `test_hf_falcon_standardized.py:TestFalcon`
- `test_flava_full.py:TestFlavaFull`
- `test_hf_xclip_standardized.py:TestXClipModels`
- `test_hf_detr.py:TestDETRModels`
- `test_retrieval_rag.py:RagRetrieverTest`
- `test_bert-base-uncased.py:ModelTest`
- `test_hf_llava.py:TestLLaVAModels`
- `test_blip_image_captioning_base.py:TestBlipImageCaptioningBase`
- `test_hf_idefics_standardized.py:ModelTest`
- `test_whisper_tiny.py:TestSpeechModel`
- `test_bert_base_uncased.py:TestBertModel`
- `test_hf_gpt_neo_standardized.py:ModelTest`
- `test_whisper-tiny.py:TestWhisperTiny`
- `test_hf_wav2vec2.py:ModelTest`
- `test_vit_base_patch16_224.py:TestVitModel`
- `test_hf_paligemma_standardized.py:ModelTest`
- `test_hf_xclip.py:TestXCLIPModels`
- `test_hf_bloom_standardized.py:ModelTest`
- `test_roberta_base.py:TestBertModel`
- `test_hf_llava_next_standardized.py:ModelTest`
- `test_hf_flamingo_standardized.py:TestFlamingoModels`
- `test_hf_t5.py:ModelTest`
- `test_hf_gpt_neo_standardized.py:TestGPTNeo`
- `test_hf_whisper.py:ModelTest`
- `test_hf_flamingo_standardized.py:ModelTest`
- `test_ollama_backoff_comprehensive.py:TestOllamaBackoffComprehensive`
- `test_hf_llava_next_standardized.py:TestLlavaNextModels`
- `test_clip_vit_large_patch14.py:TestClipVitLargePatch14`
- `test_clip_vit_base_patch32.py:TestClipVitBasePatch32`
- `test_llama.py:TestLlamaModel`
- `test_tokenization_rag.py:RagTokenizerTest`
- `test_hf_vit.py:ModelTest`
- `test_bert_qualcomm.py:TestBertQualcomm`
- `test_hf_clip.py:ModelTest`
- `temp_test.py:TestSpeechModel`
- `test_ollama_mock.py:TestOllamaMock`
- `test_hf_fuyu_standardized.py:TestFuyuModels`
- `test_hf_falcon_standardized.py:ModelTest`
- `test_hf_idefics_standardized.py:TestIdeficsModels`
- `test_groq_models.py:TestGroqModels`
- `test_selenium_browser_integration.py:TestCase`
- `test_gpt2.py:TestGptModel`
- `test_blip_vqa_base.py:TestBlipVqaBase`
- `test_claude_api.py:TestClaudeAPI`
- `test_hf_gemma_standardized.py:TestGemmaModels`
- `refactored_speech_template.py:TestSpeechModel`
- `test_hf_clip.py:TestCLIPModels`
- `test_hf_llava.py:ModelTest`

Common base classes:
- `unittest.TestCase`
- `ModelTest`
- `TestCase`

### Cluster 4 (Size: 74)

Classes in this cluster:
- `test_image_processing_blip.py:BlipImageProcessingTestFourChannels`
- `test_image_processing_zoedepth.py:ZoeDepthImageProcessingTest`
- `test_image_processing_chinese_clip.py:ChineseCLIPImageProcessingTestFourChannels`
- `test_image_processing_mobilevit.py:MobileViTImageProcessingTest`
- `test_image_processing_efficientnet.py:EfficientNetImageProcessorTest`
- `test_image_processing_maskformer.py:MaskFormerImageProcessingTest`
- `test_image_processing_bridgetower.py:BridgeTowerImageProcessingTest`
- `test_image_processing_rt_detr.py:RtDetrImageProcessingTest`
- `test_image_processing_owlv2.py:Owlv2ImageProcessingTest`
- `test_image_processing_idefics.py:IdeficsImageProcessingTest`
- `test_image_processing_deit.py:DeiTImageProcessingTest`
- `test_image_processing_smolvlm.py:SmolVLMImageProcessingTest`
- `test_image_processing_oneformer.py:OneFormerImageProcessingTest`
- `test_image_processing_donut.py:DonutImageProcessingTest`
- `test_image_processing_common.py:ImageProcessingTestMixin`
- `test_image_processing_glpn.py:GLPNImageProcessingTest`
- `test_image_processing_levit.py:LevitImageProcessingTest`
- `test_image_processing_idefics3.py:Idefics3ImageProcessingTest`
- `test_image_processing_vitmatte.py:VitMatteImageProcessingTest`
- `test_image_processing_siglip.py:SiglipImageProcessingTest`
- `test_image_processing_mobilenet_v2.py:MobileNetV2ImageProcessingTest`
- `test_image_processing_gemma3.py:Gemma3ImageProcessingTest`
- `test_image_processing_conditional_detr.py:ConditionalDetrImageProcessingTest`
- `test_image_processing_vit.py:ViTImageProcessingTest`
- `test_image_processing_llava_next_video.py:LlavaNextVideoProcessingTest`
- `test_image_processing_depth_pro.py:DepthProImageProcessingTest`
- `test_image_processing_prompt_depth_anything.py:PromptDepthAnythingImageProcessingTest`
- `test_image_processing_blip.py:BlipImageProcessingTest`
- `test_image_processing_llava_onevision.py:LlavaOnevisionImageProcessingTest`
- `test_image_processing_segformer.py:SegformerImageProcessingTest`
- `test_image_processing_instrictblipvideo.py:InstructBlipVideoProcessingTest`
- `test_image_processing_pixtral.py:PixtralImageProcessingTest`
- `test_image_processing_siglip2.py:Siglip2ImageProcessingTest`
- `test_image_processing_textnet.py:TextNetImageProcessingTest`
- `test_image_processing_beit.py:BeitImageProcessingTest`
- `test_image_processing_detr.py:DetrImageProcessingTest`
- `test_image_processing_imagegpt.py:ImageGPTImageProcessingTest`
- `test_image_processing_qwen2_vl.py:Qwen2VLImageProcessingTest`
- `test_image_processing_mobilenet_v1.py:MobileNetV1ImageProcessingTest`
- `test_image_processing_videomae.py:VideoMAEImageProcessingTest`
- `test_image_processing_deformable_detr.py:DeformableDetrImageProcessingTest`
- `test_image_processing_mllama.py:MllamaImageProcessingTest`
- `test_image_processing_owlvit.py:OwlViTImageProcessingTest`
- `test_image_processing_tvp.py:TvpImageProcessingTest`
- `test_image_processing_clip.py:CLIPImageProcessingTest`
- `test_image_processing_chameleon.py:ChameleonImageProcessingTest`
- `test_image_processing_layoutlmv2.py:LayoutLMv2ImageProcessingTest`
- `test_image_processing_idefics2.py:Idefics2ImageProcessingTest`
- `test_image_processing_superglue.py:SuperGlueImageProcessingTest`
- `test_image_processing_seggpt.py:SegGptImageProcessingTest`
- `test_image_processing_mask2former.py:Mask2FormerImageProcessingTest`
- `test_image_processing_swin2sr.py:Swin2SRImageProcessingTest`
- `test_image_processing_vilt.py:ViltImageProcessingTest`
- `test_image_processing_pvt.py:PvtImageProcessingTest`
- `test_image_processing_poolformer.py:PoolFormerImageProcessingTest`
- `test_image_processing_aria.py:AriaImageProcessingTest`
- `test_image_processing_dpt.py:DPTImageProcessingTest`
- `test_image_processing_grounding_dino.py:GroundingDinoImageProcessingTest`
- `test_image_processing_pix2struct.py:Pix2StructImageProcessingTest`
- `test_image_processing_vivit.py:VivitImageProcessingTest`
- `test_image_processing_video_llava.py:VideoLlavaImageProcessingTest`
- `test_image_processing_pix2struct.py:Pix2StructImageProcessingTestFourChannels`
- `test_image_processing_superpoint.py:SuperPointImageProcessingTest`
- `test_image_processing_chinese_clip.py:ChineseCLIPImageProcessingTest`
- `test_image_processing_llava.py:LlavaImageProcessingTest`
- `test_image_processing_vitpose.py:VitPoseImageProcessingTest`
- `test_image_processing_yolos.py:YolosImageProcessingTest`
- `test_image_processing_llava_next.py:LlavaNextImageProcessingTest`
- `test_image_processing_flava.py:FlavaImageProcessingTest`
- `test_image_processing_got_ocr2.py:GotOcr2ProcessingTest`
- `test_image_processing_layoutlmv3.py:LayoutLMv3ImageProcessingTest`
- `test_image_processing_nougat.py:NougatImageProcessingTest`
- `test_image_processing_common.py:AnnotationFormatTestMixin`
- `test_image_processing_convnext.py:ConvNextImageProcessingTest`

Common base classes:
- `ImageProcessingTestMixin`
- `unittest.TestCase`
- `AnnotationFormatTestMixin`

### Cluster 5 (Size: 53)

Classes in this cluster:
- `test_processing_common.py:ProcessorTesterMixin`
- `test_processor_idefics.py:IdeficsProcessorTest`
- `test_processor_wav2vec2_bert.py:Wav2Vec2BertProcessorTest`
- `test_processor_llava_next_video.py:LlavaNextVideoProcessorTest`
- `test_processing_gemma3.py:Gemma3ProcessorTest`
- `test_processor_layoutxlm.py:LayoutXLMProcessorTest`
- `test_processor_align.py:AlignProcessorTest`
- `test_processor_idefics2.py:Idefics2ProcessorTest`
- `test_processor_layoutlmv3.py:LayoutLMv3ProcessorTest`
- `test_processor_owlvit.py:OwlViTProcessorTest`
- `test_processor_blip_2.py:Blip2ProcessorTest`
- `test_processor_fuyu.py:FuyuProcessingTest`
- `test_processor_chinese_clip.py:ChineseCLIPProcessorTest`
- `test_processor_qwen2_5_vl.py:Qwen2_5_VLProcessorTest`
- `test_processor_idefics3.py:Idefics3ProcessorTest`
- `test_processor_flava.py:FlavaProcessorTest`
- `test_processor_llava.py:LlavaProcessorTest`
- `test_processor_sam.py:SamProcessorTest`
- `test_processor_llava_onevision.py:LlavaOnevisionProcessorTest`
- `test_processing_shieldgemma2.py:ShieldGemma2ProcessorTest`
- `test_processor_owlv2.py:Owlv2ProcessorTest`
- `test_processing_colpali.py:ColPaliProcessorTest`
- `test_processor_mistral3.py:Mistral3ProcessorTest`
- `test_processor_qwen2_audio.py:Qwen2AudioProcessorTest`
- `test_processor_udop.py:UdopProcessorTest`
- `test_processor_kosmos2.py:Kosmos2ProcessorTest`
- `test_processor_trocr.py:TrOCRProcessorTest`
- `test_processor_mllama.py:MllamaProcessorTest`
- `test_processor_clipseg.py:CLIPSegProcessorTest`
- `test_processor_vision_text_dual_encoder.py:VisionTextDualEncoderProcessorTest`
- `test_processor_clip.py:CLIPProcessorTest`
- `test_processor_qwen2_vl.py:Qwen2VLProcessorTest`
- `test_processor_pixtral.py:PixtralProcessorTest`
- `test_processor_emu3.py:Emu3ProcessorTest`
- `test_processor_smolvlm.py:SmolVLMProcessorTest`
- `test_processor_got_ocr2.py:GotOcr2ProcessorTest`
- `test_processor_aya_vision.py:AyaVisionProcessorTest`
- `test_processor_git.py:GitProcessorTest`
- `test_processor_donut.py:DonutProcessorTest`
- `test_processor_grounding_dino.py:GroundingDinoProcessorTest`
- `test_processor_bridgetower.py:BridgeTowerProcessorTest`
- `test_processor_blip.py:BlipProcessorTest`
- `test_processor_wav2vec2.py:Wav2Vec2ProcessorTest`
- `test_processor_instructblipvideo.py:InstructBlipVideoProcessorTest`
- `test_processor_aria.py:AriaProcessorTest`
- `test_processor_pix2struct.py:Pix2StructProcessorTest`
- `test_processor_paligemma.py:PaliGemmaProcessorTest`
- `test_processor_omdet_turbo.py:OmDetTurboProcessorTest`
- `test_processor_altclip.py:AltClipProcessorTest`
- `test_processor_chameleon.py:ChameleonProcessorTest`
- `test_processor_instructblip.py:InstructBlipProcessorTest`
- `test_processor_layoutlmv2.py:LayoutLMv2ProcessorTest`
- `test_processor_llava_next.py:LlavaNextProcessorTest`

Common base classes:
- `ProcessorTesterMixin`
- `unittest.TestCase`

## Redundant Import Patterns

These sets of imports appear together in multiple files and may be candidates for consolidation in utility modules.

### Import Pattern 1 (Used in 211 files)

Imports:
- `sys`
- `argparse`
- `from pathlib import Path`
- `logging`
- `importlib.util`
- `os`

Files using this pattern:
- test_hf_mamba.py
- test_hf_swin2sr.py
- test_hf_univnet.py
- test_hf_autoformer.py
- test_hf_vivit.py
- ... and 206 more files

### Import Pattern 2 (Used in 147 files)

Imports:
- `from typing import Dict`
- `from typing import Any`
- `from typing import Optional`
- `tokenizers`
- `traceback`
- `from unittest.mock import patch`
- `argparse`
- `torch`
- `logging`
- `json`
- `time`
- `sys`
- `from typing import List`
- `datetime`
- `transformers`
- `from unittest.mock import MagicMock`
- `os`

Files using this pattern:
- test_hf_pythia.py
- test_hf_wavlm.py
- test_hf_speech_to_text.py
- test_hf_unispeech.py
- test_hf_layoutlm.py
- ... and 142 more files

### Import Pattern 3 (Used in 67 files)

Imports:
- `from generators.hardware.hardware_detection import HAS_WEBNN`
- `tokenizers`
- `traceback`
- `from generators.hardware.hardware_detection import HAS_MPS`
- `from unittest.mock import Mock`
- `numpy as np`
- `anyio`
- `torch`
- `from typing import Union`
- `time`
- `from generators.hardware.hardware_detection import HAS_WEBGPU`
- `from typing import List`
- `from generators.hardware.hardware_detection import HAS_CUDA`
- `transformers`
- `from unittest.mock import MagicMock`
- `from typing import Dict`
- `from openvino.runtime import Core`
- `from generators.hardware.hardware_detection import HAS_OPENVINO`
- `from typing import Any`
- `sentencepiece`
- `from typing import Optional`
- `from pathlib import Path`
- `ctypes.util`
- `from unittest.mock import patch`
- `from generators.hardware.hardware_detection import detect_all_hardware`
- `from generators.hardware.hardware_detection import HAS_ROCM`
- `argparse`
- `logging`
- `openvino`
- `json`
- `sys`
- `ctypes`
- `datetime`
- `os`

Files using this pattern:
- test_hf_speech-to-text.py
- test_hf_fuyu.py
- fixed_bert_template.py
- test_hf_speech_to_text.py
- test_hf_bigbird.py
- ... and 62 more files

### Import Pattern 4 (Used in 47 files)

Imports:
- `argparse`
- `torch`
- `logging`
- `json`
- `time`
- `sys`
- `from pathlib import Path`
- `transformers`
- `from unittest.mock import MagicMock`
- `os`

Files using this pattern:
- test_hf_t5_minimal.py
- test_hf_gpt-j.py
- test_hf_bert_minimal.py
- test_hf_xlm-roberta.py
- test_hf_flan-t5.py
- ... and 42 more files

### Import Pattern 5 (Used in 24 files)

Imports:
- `from generators.hardware.hardware_detection import HAS_WEBNN`
- `traceback`
- `from generators.hardware.hardware_detection import HAS_MPS`
- `from unittest.mock import Mock`
- `requests`
- `numpy as np`
- `from io import BytesIO`
- `torch`
- `from typing import Union`
- `time`
- `from PIL import Image`
- `from generators.hardware.hardware_detection import HAS_WEBGPU`
- `from typing import List`
- `from generators.hardware.hardware_detection import HAS_CUDA`
- `transformers`
- `from unittest.mock import MagicMock`
- `from typing import Dict`
- `from openvino.runtime import Core`
- `from generators.hardware.hardware_detection import HAS_OPENVINO`
- `from typing import Any`
- `from typing import Optional`
- `from pathlib import Path`
- `ctypes.util`
- `from unittest.mock import patch`
- `from generators.hardware.hardware_detection import detect_all_hardware`
- `from generators.hardware.hardware_detection import HAS_ROCM`
- `argparse`
- `logging`
- `openvino`
- `json`
- `sys`
- `ctypes`
- `datetime`
- `os`

Files using this pattern:
- test_hf_mlp_mixer.py
- test_hf_mlp-mixer.py
- test_hf_bit.py
- test_hf_donut.py
- test_hf_poolformer.py
- ... and 19 more files

## Refactoring Recommendations

Based on the analysis, here are the key recommendations:

1. **Standardize Base Classes**: Consolidate test classes to use a consistent set of base classes:
   - `unittest.TestCase`
   - `ModelTesterMixin`
   - `PipelineTesterMixin`

2. **Standardize Test Method Naming**: Adopt consistent naming conventions for test methods based on current patterns.

3. **Consolidate Similar Classes**: Merge highly similar classes, particularly:
   - `test_hf_pythia.py:TestPythiaModels` and `test_hf_wavlm.py:TestWavlmModels` (similarity: 1.00)
   - `test_hf_pythia.py:TestPythiaModels` and `test_hf_speech_to_text.py:TestSpeechToTextModels` (similarity: 1.00)
   - `test_hf_pythia.py:TestPythiaModels` and `test_hf_unispeech.py:TestUnispeechModels` (similarity: 1.00)

4. **Create Utility Modules**: Create common test utility modules to minimize import duplication:
   - Create a `test_utils.py` module with common imports and helper functions
   - Create a `test_fixtures.py` module with common test fixtures

5. **Refactor Inheritance Hierarchy**: Streamline the inheritance structure to reduce complexity:
   - Create a consistent set of base test classes
   - Remove redundant inheritance levels
   - Use composition over inheritance where appropriate

6. **Improve Documentation**: Add or improve docstrings in test methods and classes to clarify purpose.

## Next Steps

1. Prioritize refactoring efforts based on this analysis
2. Create a refactoring plan with specific tasks
3. Implement changes gradually, starting with base classes and utility modules
4. Validate each change with comprehensive test runs
5. Update documentation to reflect the new test organization
