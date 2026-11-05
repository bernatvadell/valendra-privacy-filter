# Mixed OpenPII/Nemotron warm-start run

On 2026-11-05, the user requested stopping the initial causal run and continuing with the complete prepared OpenPII/Nemotron mixture. The initial run was stopped after its complete `checkpoint-13500`; its best and latest checkpoints were retained.

- Initialization: all trained Qwen3 backbone and BIOES head weights from step 13,500, SHA-256 `1f989d08382dce6598f94a307ba6a8ec23fb1f0b103bf55408b1b1d4c4fddc64`.
- Mixed dataset manifest SHA-256: `9af56087c7655f676c1cb64d214ed2d9d1a5cac78e09aaaee59abd16cd3a0f18`; the original NVIDIA test is kept separately.
- This is a new experiment: FP32 AdamW and scheduler state start fresh, BF16 autocast is used, and early stopping cannot trigger before epoch one. The original optimizer/scheduler and validation history are not imported.
- A disposable preflight on the RTX 4090 loaded the original checkpoint exactly and performed a BF16 update on one **512-token NVIDIA training window** with 20 positive labels. Backbone and head changed; parameters, gradients and AdamW states were FP32; logits were BF16; peak allocated VRAM was 10,807,799,808 bytes. The preflight did not save or alter the source checkpoint.
- The first new-phase checkpoint was saved at step 500 of a maximum of 151,860. Its Safetensors weights, optimizer, scheduler, RNG and minimum-epoch callback state were present; the saved initialization hash matched the run identity. On the new 2,048-window validation subset, token-span precision was 0.7774, recall 0.8683, F1 0.8203 and F2 0.8484. These early figures are not directly comparable with metrics from the previous dataset and are not real-world generalization results.

The model and dataset files remain outside version control. The mixed validation includes curated examples from the earlier validation process and is not independent final evidence; evaluate the reserved test sets only after model selection.
