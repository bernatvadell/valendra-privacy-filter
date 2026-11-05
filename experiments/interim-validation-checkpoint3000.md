# Interim three-model validation: checkpoint 3,000

On 2026-11-05, the user requested a preliminary comparison while the mixed-data training continued. This is a development snapshot, not a final evaluation. The copied Valendra checkpoint was the best saved checkpoint at the time, and its local and remote Safetensors SHA-256 matched: `7fda6b2bf3b44e8acf0351b08de07d76d99f55371e04cc55b7df2aa95408b976`.

The same 100 unique documents were selected by a deterministic seed-42 hash sample from the full mixed validation split, whose manifest SHA-256 is `9af56087c7655f676c1cb64d214ed2d9d1a5cac78e09aaaee59abd16cd3a0f18`. Their ordered-document fingerprint is `c1ad37a2b9ddcc62db831248bd0eb6d847dd8f0e9a08c109c72f1696d13e4521`. The sample contains 296 annotated spans: 41 project-synthetic documents, 34 OpenPII and 25 Nemotron; 75 are English and 25 Spanish. Of the Spanish examples, 23 are project-synthetic and two are OpenPII. There are no fully negative documents without ignored annotations.

The models ran sequentially on an Apple M5 CPU, batch size one, two warmup documents, with four OpenMP/MKL threads. Valendra and GLiNER ran in FP32; the official OpenAI runtime used its native BF16 weights. The RTX 4090 training run was not interrupted. GLiNER used threshold 0.3 and the prompts in `configs/comparison-models.yaml`. See the [aggregate report](interim-validation-checkpoint3000.json) for model revisions, weights hashes, detailed settings and breakdowns.

## Exact character-span results

Untyped scoring requires exactly matching start and end offsets but ignores category. Typed-common scoring also requires the correct category among PERSON, EMAIL, PHONE, ADDRESS, URL and SECRET; categories outside this subset are not directly comparable.

| Model | Untyped precision | Untyped recall | Untyped F1 | Typed-common F1 | CPU p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Valendra, step 3,000 | 0.867 | 0.926 | 0.895 | 0.860 | 106 ms |
| OpenAI Privacy Filter | 0.783 | 0.574 | 0.663 | 0.583 | 407 ms |
| NVIDIA GLiNER-PII | 0.539 | 0.561 | 0.550 | 0.473 | 169 ms |

The respective untyped TP/FP/FN counts are 274/42/22, 170/47/126 and 166/142/130. Latency is illustrative of this CPU, sample, runtime and precision configuration; it is not a GPU comparison.

| Model | Untyped F1: English, 75 docs | Untyped F1: Spanish, 25 docs |
| --- | ---: | ---: |
| Valendra | 0.965 | 0.696 |
| OpenAI | 0.675 | 0.627 |
| GLiNER | 0.624 | 0.315 |

For Valendra, exact-boundary URL recall was 8/22 (0.364) and address recall was 46/52 (0.885). These are small counts, and the Spanish/source breakdown is confounded by sample composition.

## Interpretation

- Valendra scored highest on this small, same-distribution validation sample. It is not an independent real-world test or proof of general superiority. The checkpoint was selected using validation metrics, and this validation reuses material from earlier model selection.
- Both Valendra and GLiNER were trained using Nemotron-derived data. Results on that source should not be described as independent out-of-distribution evidence for either model.
- Exact boundaries and annotation policy matter: a model that detects part of an entity or uses different boundaries is scored as a miss and an extra span. Ignored annotations and predictions overlapping them are excluded under the recorded common policy; the models' output taxonomies and GLiNER prompts also differ.
- The sample has no clean negative documents, so it cannot estimate false-positive rates on negative documents. Test broader Spanish/English data and independently annotated real-world examples after fixing the final model and evaluation policy.
