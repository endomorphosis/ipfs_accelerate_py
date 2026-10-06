# Retained contextual LegalIR 384D selected checkpoint

This release mirrors the exact complete original selected state, with SHA256 `0ac5c21656187d1d8040a02bcc0b4716a8093db17f05adc9a22e99d5b6b2cc00`, already published in [the aggregate repository](https://huggingface.co/Publicus/legal-ir-autoencoder/blob/49839a5e55e2ef8ab3815f4e2650dc69ee72d5b8/releases/20261004-additional-decoder-cutoff-v1/checkpoints/decoder-selected-continuation-20261004/384-selected-followup-lr0001-1729/selected-state.json). It preserves all 32 saved model-state entries.

The retained input contract uses a paragraph vector plus ordered literal-source clause vectors. Apply the original saved TRAIN-only input transform to raw clause vectors first, then zero-pad to eight positions with a boolean mask. Paragraph and clause feature normalizations are separate saved decoder stages; do not apply those transforms twice. Exact original cache, preprocessing and producer pins are in `manifest.json`. This mirror includes checkpoint bytes and metadata; it references the existing caches without uploading them.

The encoder experiment admits 512 source tokens and the decoder output limit is 512 tokens.

The output is an ordered canonical rules array with modality, actor, action, object, conditions, exceptions and temporal fields. The observed qualifier fields are empty. Historical replay reconstructed 48/48 previously exposed authored paragraphs exactly as canonical IR, covering 180 rules. Those results describe the retained regression cohort; this publication performs no new inference or evaluation.

Native IR schema, decoder profile and decoder format identities remain unknown (`null`). Runtime, source fidelity, teacher and proof qualifications remain false. Original-prose reconstruction, independent held-out accuracy, paragraph-vector-only decoding and larger text spans require their own evaluation.

This release adds explicit versioned files. Existing repository paths, root model card, defaults and visibility are preserved. The original ModelManager record retains its immutable aggregate publication binding.
