# Full causal BIOES training run

Snapshot taken on 2026-11-05 while training was in progress. The authoritative state is the GPU host's `experiments/runs/causal/` and `experiments/runs/full-causal.log`; this document does not assert that the run has finished.

- Backbone: `Qwen/Qwen3-0.6B-Base`, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
- Model: native Qwen3 token-classification head, 41 BIOES labels, causal attention, all parameters trainable.
- Precision: BF16 autocast with FP32 master weights and AdamW states. Hardware: one RTX 4090.
- Configuration: `configs/train-causal.yaml`: 3 epochs maximum, batch 1, gradient accumulation 8, gradient checkpointing, initial LR 2e-5, 5% warmup, early stopping after warmup, checkpoint/evaluation every 500 optimizer steps.
- Training prioritizes `es`/`ES` windows at four occurrences per epoch; other training windows occur once. Early stopping uses a deterministic 2,048-window validation subset.
- Processed dataset manifest SHA-256: `a97da9c6974051d49ecfe2a7e8722d97d4c61ae95c26f9e283fa77db27ad500d`.
- Dataset snapshot: 195,528 unique training documents (44,015 `es`/`ES`; 151,513 English), 37,654 validation, 37,468 test and 15,002 challenge documents. The source contributes 179,526 training documents; local synthetic templates contribute 16,002. Preparation removed 18,992 exact duplicates across splits.
- The original validation split has no `DATE_OF_BIRTH` examples. A later audit also found repeated synthetic identifier values across the original splits; validation scores may be optimistic. See [the Nemotron dataset assessment](../docs/nemotron-pii-assessment.md).

At 2026-11-05 00:23 UTC, `checkpoint-500` had been saved with model Safetensors, optimizer, scheduler, RNG and Trainer state. Its `trainer_state.json` recorded 500 of a maximum of 122,841 steps, training loss 1.089, and automatic token-span F1 0.7727 on the 2,048-window validation subset. This early metric is from a predominantly synthetic validation set, excludes `DATE_OF_BIRTH`, and is not evidence of real-world generalization. Training was still active at this snapshot.

The initial run stopped at step 8,000 (epoch 0.1954) after three evaluations without improving on step 6,500. Its exported best checkpoint had token-span F1 0.9093, precision 0.8853 and recall 0.9346 on that same validation subset. At the user's request, the run is being continued from checkpoint 8,000 with early stopping disabled until epoch 1.0. The optimizer, scheduler, RNG, sampled-data order, and prior best metric are restored; the changed stopping policy is recorded separately with a new protocol fingerprint.

The continuation started at 2026-11-05 06:22 UTC. Its policy fingerprint is `b49ac0781f90faf2dbdf3c31b32046eb1d3fe531b937ebba3cbce32cabc9ba1c`. At the subsequent snapshot, training had advanced past epoch 0.1966. The original step-6,500 export was preserved separately. The full training logs and checkpoints are not distributed with the source repository.

On 2026-11-05 the user requested stopping this phase and continuing with the mixed OpenPII/Nemotron data. The last complete checkpoint was step 13,500; the best prior checkpoint was step 11,000. See [the mixed warm-start record](mixed-warmstart-run.md).

The data and training results are stored outside version control. Neither the two-step smoke models nor these preliminary training snapshots are validated production models.
