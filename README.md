# Valendra Privacy Filter

A research pipeline for detecting and redacting PII with token classification, not text generation. It uses the complete [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) backbone, a trainable BIOES head, BF16 mixed-precision full fine-tuning, and exact character-span evaluation. Causal Qwen3 is the baseline; full bidirectional attention is a separate experiment.

**Status:** the pipeline, including both attention modes, has been exercised on a GPU using very small smoke datasets. No useful trained privacy model or evidence of real-world generalization is included. See [measured smoke results](experiments/smoke-results.md) before interpreting any numbers.

## Requirements and setup

- Python 3.10–3.12, [`uv`](https://docs.astral.sh/uv/) 0.13.6, and a CUDA GPU supporting BF16 for training. CUDA driver and wheel must be compatible with the device.
- CPU or CUDA for inference. Internet access and sufficient disk space are needed to fetch the public dataset and the approximately 1.2 GB base checkpoint. Dataset preparation can read several gigabytes even for a small balanced sample because the source is streamed in its published order.
- Core dependencies are pinned in `pyproject.toml`; transitive versions are pinned in `uv.lock`. The model and dataset revisions are recorded in the source and dataset manifest.

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run valendra-privacy-filter --help
```

Unit tests instantiate a tiny Qwen3 architecture and do not download model weights. To additionally use the official tokenizer/checkpoint and CUDA:

```bash
RUN_OFFICIAL_INTEGRATION=1 uv run pytest tests/test_official_integration.py -q
RUN_GPU_INTEGRATION=1 uv run pytest tests/test_official_integration.py -q
uv run valendra-privacy-filter smoke-test --output-dir experiments/runs/official-smoke --sequence-length 512
```

The last command uses a labeled synthetic batch, checks BF16 logits and FP32 parameters/gradients/Adam states, updates backbone and head, and saves and reloads Safetensors. Its output checkpoint has seen only one batch and is not a trained detector. `--attention-mode bidirectional` runs the experimental variant separately.

## Dataset preparation

Source: [Ai4Privacy OpenPII 1M](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m), revision `ecfdc547f4a0955600cfe6ab98ba2a162207fcc0`, CC BY 4.0. Only original `source_text` and `privacy_mask` offsets are used; mBERT token labels are ignored. Source offsets, values, category mappings, overlaps, and tokenizer boundaries are validated. Unknown and unsupported annotated categories are ignored in the loss (`-100`) and counted, not silently labeled `O`. In particular, a generic `DATE` is **not** assumed to be a date of birth. See [data policy](docs/data-policy.md) for category mapping, synthetic positives and negatives, and limitations.

Prepare a small, reproducible subset and independent synthetic template families:

```bash
uv run valendra-privacy-filter generate-synthetic --output-dir data/raw/synthetic --seed 42 --per-split 64
uv run valendra-privacy-filter validate-annotations --input data/raw/synthetic/train.jsonl
uv run valendra-privacy-filter prepare-dataset --synthetic-dir data/raw/synthetic \
  --output-dir data/processed-smoke --openpii-limit 30 --max-length 512 --overlap 64 \
  --taxonomy configs/taxonomy.yaml --seed 42
```

The OpenPII limit counts selected source documents before deduplication, with a 2:1 Spanish:English target when both are selected; Spanish defaults to region `ES`. `--spanish-all-regions` also includes other Spanish regions. Remove `--openpii-limit` for the full selected source, or use `--skip-openpii` for a synthetic-only local pipeline check. Results are `train`, `validation`, `test`, and `challenge` Parquet files plus `manifest.json`. The manifest records checksums, source revisions/licenses, label and language distributions, ignored categories, duplicates, and alignment issues. Training and evaluation verify file checksums.

Exact duplicate texts are removed; conflicting annotations for an exact duplicate stop preparation. OpenPII template signatures and synthetic template-family IDs are assigned to only one split. This heuristic detects literal template variants, not every semantic paraphrase or unannotated sensitive value. Challenge templates are independent of synthetic training templates. The public source is itself synthetic, so neither its test split nor the synthetic challenge establishes real-world generalization.

### Optional NVIDIA data in a separate experiment

[NVIDIA Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII), revision `b70ffaf5ff39e079776134c5bf4381f00a9fd1ed`, is an English synthetic dataset licensed **CC BY 4.0**; credit NVIDIA Corporation. Its published files can be used without running another model or a data-generation API. The separately licensed NVIDIA GLiNER model is not used. See [the source assessment](docs/nemotron-pii-assessment.md).

With an existing baseline dataset in `data/processed`, create new directories; the commands never overwrite the baseline:

```bash
uv run valendra-privacy-filter prepare-nemotron --baseline-dir data/processed \
  --output-dir data/processed-nemotron-v1 --label-policy configs/nemotron-labels.yaml
uv run valendra-privacy-filter prepare-address-challenge --baseline-dir data/processed \
  --output-dir data/processed-address-challenge-v1 --count 2000 --seed 42
uv run valendra-privacy-filter merge-datasets --baseline-dir data/processed \
  --nvidia-dir data/processed-nemotron-v1 --challenge-supplement-dir data/processed-address-challenge-v1 \
  --output-dir data/processed-openpii-nemotron-v1
uv run valendra-privacy-filter audit-dataset-isolation --data-dir data/processed-openpii-nemotron-v1 \
  --holdout-dir data/processed-nemotron-v1 --output experiments/isolation-audit.json
uv run valendra-privacy-filter audit-dataset-annotations --data-dir data/processed-openpii-nemotron-v1 \
  --output experiments/annotation-audit.json
```

NVIDIA's original test rows stay exclusively in the separate Nemotron `test.parquet`; only its original train rows are grouped into mixed train/validation. Strict, bounded parsing validates source values and offsets, preserving native labels; unsupported classes are ignored and counted. The combined dataset excludes baseline records with cross-split exact-text, template-family or high-signal identifier conflicts, plus cross-source and NVIDIA train/test overlaps. High-signal fingerprints cover email, phone, government/financial identifiers, secrets, and sufficiently specific addresses. Hashes are held in memory and never written to reports. The policy is conservative and may remove recurring fictional placeholders; it does not prove separation of all people, short identifiers, incomplete annotations or semantic paraphrases. Both the NVIDIA test and the project challenge remain synthetic.

The Python distribution includes a default source-label policy, so `prepare-nemotron` also works outside a source checkout. Pass `--label-policy` to use an explicit policy file.

For a small pipeline smoke, pass `--train-limit 200 --test-limit 200` to `prepare-nemotron` and `--allow-sampled-holdout` to `merge-datasets`. A sampled holdout is explicitly recorded and is not an evaluation of the full NVIDIA test.

## Training, resumption and experiments

```bash
# Quick pipeline verification, not model training evidence
uv run valendra-privacy-filter train --config configs/train-smoke-causal.yaml
uv run valendra-privacy-filter train --config configs/train-smoke-bidirectional.yaml

# Full run: prepare data/processed first and adjust settings for your GPU/data
uv run valendra-privacy-filter train --config configs/train-causal.yaml
uv run valendra-privacy-filter train --config configs/train-bidirectional.yaml

# Later, as a separate run after preparing the mixed dataset above
uv run valendra-privacy-filter train --config configs/train-openpii-nemotron-causal.yaml
uv run valendra-privacy-filter train --config configs/train-openpii-nemotron-bidirectional.yaml

# Resume an interrupted run from a checkpoint in that same run directory
uv run valendra-privacy-filter train --config configs/train-causal.yaml \
  --resume-from-checkpoint experiments/runs/causal/checkpoint-500
```

If a run has already stopped early, an explicitly approved continuation can defer further early stopping until a minimum epoch without restarting weights or optimizer state:

```bash
uv run valendra-privacy-filter train --config configs/train-causal.yaml \
  --resume-from-checkpoint experiments/runs/causal/checkpoint-8000 \
  --min-early-stopping-epoch 1.0
```

This records `continuation_policy.json`, keeps the original run identity, gives subsequent checkpoints a distinct protocol fingerprint, and resets early-stopping patience while preserving the prior best metric. Future resumes from a continuation checkpoint automatically reuse that policy. Reports from different protocols cannot be compared as equivalent.

When changing datasets, initialize a **new** experiment from an existing trained checkpoint rather than resuming its Trainer state:

```bash
uv run valendra-privacy-filter train --config configs/train-openpii-nemotron-warmstart.yaml \
  --initialize-from-checkpoint /path/to/previous-run/checkpoint-13500
```

The source checkpoint must contain model Safetensors, `config.json` and `trainer_state.json`, with its original `run_identity.json` in the parent directory. The loader checks the Qwen3 base revision, architecture, attention mode and BIOES label order; it rejects missing or unexpected weights and verifies the loaded classifier and backbone tensors. All weights remain trainable. The new phase begins at step zero with fresh FP32 AdamW state and a fresh scheduler, and records source weight/config hashes and a distinct protocol identity. Resume this new phase using its own `--resume-from-checkpoint`, without `--initialize-from-checkpoint`; the original checkpoint is not required for resumption. The mixed validation reuses curated examples from the earlier validation process and is not an untouched final test.

The YAML configurations are starting points, **not validated optimal hyperparameters**. All backbone and head parameters remain trainable. Trainer uses BF16 autocast, FP32 master parameters and FP32 AdamW moments, linear scheduling, warmup, accumulation, clipping, optional gradient checkpointing, periodic evaluation/checkpoints, early stopping, and Safetensors model weights. Early-stopping patience starts only after warmup completes. To prioritize Spain without dropping English examples, the full-run configs sample each `es`/`ES` training window four times per epoch; the exported dataset remains deduplicated and other windows appear once. For practical repeated validation they use a deterministic seed-selected subset of 2,048 validation windows for early stopping. Use the separate evaluation command for complete test and challenge reports. Optimizer/RNG/callback state needed to resume is stored by Trainer alongside those weights; resume only from checkpoints you trust. A run identity binds each output directory to the model revision, dataset manifest, seed, attention mode and training protocol. Resume with the same configuration and directory. Exports live under each run's `final/`.

Both configurations use the same settings except attention mode and output directory. The baseline preserves the native causal mask. The bidirectional subclass passes an explicit 4D padding mask, disables caching and uses tested eager/SDPA backends; it does not merely toggle a config flag. Tests verify future-token influence, causal isolation and padding invariance. The mode is saved in the model config.

The 596,049,920-parameter checkpoint needs roughly 9.54 GB decimal (8.88 GiB) just for FP32 parameters, gradients, and two FP32 Adam moments, excluding activations, BF16 autocast caches, allocator/workspace overhead and temporary tensors. Actual peak depends on sequence length, batching, backend and checkpointing. See [GPU measurements](experiments/smoke-results.md) rather than treating this lower bound as capacity planning.

## Evaluation, calibration and benchmark

Evaluation decodes overlapping windows globally using constrained BIOES Viterbi, then scores exact `(start, end, type)` spans. Reports include precision, recall, F1, F2, per-category and per-language breakdowns, false positives per document, negative-document rates, and a confusion matrix. Confusions pair wrong types only when boundaries match exactly; unmatched spans use `NONE`. Predictions overlapping intentionally ignored annotations are excluded. The Trainer's early-stopping metric is an exact **token-span** metric; the CLI evaluation reports exact **character spans**.

```bash
uv run valendra-privacy-filter evaluate --checkpoint experiments/runs/causal/final --data-dir data/processed \
  --split test --device cuda --precision bf16 --output experiments/runs/causal-test.json
uv run valendra-privacy-filter evaluate --checkpoint experiments/runs/bidirectional/final --data-dir data/processed \
  --split test --device cuda --precision bf16 --output experiments/runs/bidirectional-test.json
uv run valendra-privacy-filter compare-experiments --first experiments/runs/causal-test.json \
  --second experiments/runs/bidirectional-test.json --output experiments/runs/comparison.json
```

To evaluate against the separately reserved NVIDIA holdout, use `--data-dir data/processed-nemotron-v1 --split test`. Do not use this holdout for model selection or merge it into training.

Repeat with `--split challenge` to keep challenge results separate. An independently prepared annotated dataset may be used for evaluation: training and evaluation manifest fingerprints are both reported. Comparison requires identical training/evaluation fingerprints, seed, base revision, protocol and inference windows. Real, independently annotated evaluation data is still needed to assess deployment suitability, especially for secrets, identifiers, Unicode, code and negative examples.

The default span score is the geometric mean of the selected tags' token softmax probabilities after optional temperature scaling; it is **not automatically a calibrated span probability**. Fit a token temperature on validation only, then pass the artifact to `detect`, `redact`, `serve` or `evaluate` using `--calibration`:

```bash
uv run valendra-privacy-filter calibrate --checkpoint experiments/runs/causal/final --data-dir data/processed \
  --device cuda --output experiments/runs/causal/calibration.json
```

For performance, provide a UTF-8 file with one document per line; only aggregate results are printed and saved:

```bash
uv run valendra-privacy-filter benchmark --checkpoint experiments/runs/causal/final \
  --input data/raw/benchmark-lines.txt --output experiments/runs/causal-benchmark.json \
  --device cuda --precision bf16 --repeats 3
```

The report records p50/p95/p99 latency, tokens/s, documents/s, sampled process RSS, peak PyTorch allocated VRAM and CPU time as a percentage of one core, along with device, precision, window and batch settings. It does not report a system-wide peak RSS or GPU reserved memory.

### Compare three privacy models

The local comparison runs Valendra, `openai/privacy-filter` and `nvidia/gliner-PII` sequentially in isolated processes. The OpenAI model and runtime are Apache 2.0; NVIDIA's model is governed by the NVIDIA Open Model License. The first run downloads their pinned public weights. Install the two independent locked environments alongside the main environment:

```bash
uv sync --locked --extra dev
uv sync --locked --project benchmarks/openai
uv sync --locked --project benchmarks/gliner
```

Use an existing checkpoint and a checksum-verified prepared dataset. The destination must not already exist:

```bash
.venv/bin/python scripts/run_comparison.py \
  --checkpoint /path/to/checkpoint --data-dir /path/to/data \
  --split validation --device cpu --precision fp32 \
  --limit 100 --sample-seed 42 --warmup 2 \
  --output-dir experiments/runs/interim-comparison
```

Each model sees the same deterministic sample of unique documents from the entire split. Omit both `--limit` and `--sample-seed` to evaluate the complete split. Reports record dataset and document-order hashes, runtime settings, strict character-span scores, type-specific results for a common subset, and latency and memory measurements; they do not contain input text. Exact untyped and typed-common scores answer different questions, because model taxonomies and supported categories differ. FP32 is used for Valendra and GLiNER in this example; OpenAI's official runtime uses its native parameter precision. The validation data and any intermediate checkpoint are for development, not an independent final assessment. Synthetic validation data alone does not establish real-world performance.

## Python and command-line inference

```python
from valendra_privacy_filter.inference import Detector

detector = Detector.from_pretrained(
    "experiments/runs/causal/final", device="cpu", precision="fp32",
    max_length=512, overlap=64,
)
result = detector.detect("Soy Ana García y mi correo es ana@example.com")
redacted = detector.redact("Soy Ana García y mi correo es ana@example.com")
```

`detect` returns an `entities` list containing `type`, `start`, `end`, `value` and `score` for each span. Offsets are half-open Python Unicode code-point indices into the original string; `text[start:end] == value`. Long documents are tokenized once, window logits are averaged at shared token positions, and a single global Viterbi path is decoded. Overlapping reconstructed intervals are resolved by confidence with deterministic tie-breaking. Redaction replaces spans from right to left and preserves all other content.

Use stdin or a UTF-8 file to avoid putting text in command arguments:

```bash
uv run valendra-privacy-filter detect --checkpoint experiments/runs/causal/final --input document.txt
uv run valendra-privacy-filter redact --checkpoint experiments/runs/causal/final --input document.txt
```

Precision is explicit: `--precision fp32` is the CPU default; `bf16` must be requested. There is no application-level automatic precision fallback. BF16 CPU support and speed depend on the PyTorch build, hardware and individual kernels. On the tested AVX2-only server, full-model BF16 input inference ran and produced BF16 logits without native AVX-512 BF16 or AMX support; this does not prove native BF16 computation for every operation. Transformers' Qwen3 implementation deliberately performs RoPE, RMSNorm variance and attention softmax computations in FP32 where applicable. Unsupported operations raise an error.

## HTTP API and Docker

```bash
uv run valendra-privacy-filter serve --checkpoint experiments/runs/causal/final \
  --host 127.0.0.1 --port 8000 --device cpu --precision fp32 \
  --max-length 512 --overlap 64 --max-chars 100000 --max-batch 32 --max-body-bytes 1000000

curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/detect -H 'Content-Type: application/json' \
  -d '{"text":"Soy Ana García y mi correo es ana@example.com"}'
curl -s -X POST http://127.0.0.1:8000/redact -H 'Content-Type: application/json' \
  -d '{"texts":["Soy Ana García", "No hay datos personales"]}'
```

The API accepts either `text` or `texts`; batched responses use `results`. One model is loaded per process, body/document/batch limits are enforced, and `/health` provides aggregate request/document/error counts. The service does not log input texts or detected values; the CLI server disables access logging. Treat responses and any input files as sensitive.

```bash
docker build -t valendra-privacy-filter .
docker run --rm -p 8000:8000 \
  -v "$PWD/experiments/runs/causal/final:/models/valendra-privacy-filter:ro" valendra-privacy-filter
# For a configured NVIDIA container runtime, add --gpus all and pass
# --checkpoint /models/valendra-privacy-filter --host 0.0.0.0 --device cuda --precision bf16
```

The image uses a tagged NVIDIA CUDA runtime base; pin its image digest in a deployment environment if immutable base-image reproducibility is required.

## Licenses and attribution

Repository code: [Apache 2.0](LICENSE). Qwen3-0.6B-Base: Apache 2.0, © Qwen team / Alibaba; model revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Transformers, PyTorch and other dependencies retain their own licenses. OpenPII: **CC BY 4.0**, © 2026 Ai Suisse SA; credit **Ai4Privacy / Ai Suisse SA** and link to [the dataset](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m). Nemotron-PII: **CC BY 4.0**, © NVIDIA Corporation; credit NVIDIA and link to [the dataset](https://huggingface.co/datasets/nvidia/Nemotron-PII). The code/model license does not replace the datasets' attribution terms. Generated synthetic examples are project-created and contain deliberately fictional secrets.
