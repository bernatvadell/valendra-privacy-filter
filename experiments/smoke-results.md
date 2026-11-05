# GPU integration smoke results

Observed on the supplied NVIDIA GeForce RTX 4090 (24,564 MiB VRAM; compute capability 8.9), PyTorch 2.8.0+cu128, CUDA 12.8 and Transformers 4.55.4. Checkpoint revision: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.

An official-checkpoint causal smoke performed one supervised BF16-autocast forward/backward/AdamW update and saved and reloaded Safetensors. Only `score.weight` and `score.bias` were missing from the base checkpoint, as expected. Both the backbone and classification head changed. Master parameters, gradients and AdamW moments were FP32; logits were BF16. Peak allocated GPU memory was 10,802,505,216 bytes (approximately 10.06 GiB) and elapsed time was 12.01 seconds. The input was short; this is **not** the peak for 512-token training.

A subsequent run used a **512-token labeled batch** (320 positive BIOES tokens), SDPA and gradient checkpointing: the same forward/backward/update/reload checks passed; peak allocated GPU memory was 10,807,799,808 bytes (approximately 10.07 GiB) and elapsed time was 7.70 seconds. These are measured on one batch and are not a worst-case production capacity estimate.

An additional two-step causal run resumed from its step-one checkpoint. The resumed second-step loss and gradient norm matched the uninterrupted run at the precision printed by Trainer (loss 2.0503, gradient norm 384.5146).

The full checkpoint was also loaded on the server CPU. FP32 inference worked. Explicit BF16 parameters and BF16 logits also worked on this machine's AVX2 CPU, although PyTorch reported neither AVX-512 BF16 nor AMX support; this does not establish native BF16 execution or performance for every operation.

Separately, both attention modes ran the same two-step full-fine-tuning protocol (seed 42, batch 1, no warmup, 2e-5 learning rate, gradient checkpointing) on the same small sample: 30 OpenPII records and 32 synthetic records before grouping/deduplication. The retained split counts were 31 train, 10 validation, 13 test and 8 challenge documents. This initial convenience sample happened to contain predominantly English OpenPII records; subsequent preparation code adds language quotas and prioritizes Spain when a source limit is requested.

On the 13-document smoke test split, exact-span metrics were:

| Mode | True positives | False positives | False negatives | Precision / recall / F1 / F2 |
| --- | ---: | ---: | ---: | --- |
| Causal | 0 | 443 | 54 | 0 / 0 / 0 / 0 |
| Bidirectional | 0 | 43 | 54 | 0 / 0 / 0 / 0 |

This is a pipeline check after two optimization steps, not a meaningful quality comparison. The sample is synthetic, very small and not representative of production or of real-world generalization. These weights are not a usable privacy filter.

An inference benchmark on the causal two-step checkpoint used 8 short test documents, 3 repetitions, one warmup, BF16 and batch size 1: 24 documents and 963 tokens, p50 31.30 ms, p95 31.74 ms, p99 45.47 ms, 1,243.79 tokens/s and 31.00 documents/s. Sampled process RSS peaked at 1,997,250,560 bytes, PyTorch allocated GPU memory peaked at 1,202,313,728 bytes, and process CPU use was 100.74% of one core. These short inputs and this hardware are essential context; the numbers are not a 512-token or production-load benchmark.
