"""Load pinned baselines in their own process without changing training dependencies."""

from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from valendra_privacy_filter.comparison.adapters import GLiNERAdapter, OpenAIAdapter, ValendraAdapter
from valendra_privacy_filter.datasets.prepare import _sha256_file


def validate_precision(model_name, precision, device):
    if model_name not in {"valendra", "openai", "gliner"}:
        raise ValueError("Unsupported comparison model")
    if model_name == "openai" and precision != "native":
        raise ValueError("OpenAI OPF requires native checkpoint precision")
    if model_name != "openai" and precision not in {"fp32", "bf16"}:
        raise ValueError("Valendra and GLiNER require fp32 or bf16 precision")
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("Comparison device must be cpu or cuda")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device is unavailable")
        if precision in {"bf16", "native"}:
            with torch.cuda.device(selected):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("Requested BF16 CUDA execution is unsupported")
    return selected


def _parameter_dtypes(model):
    result = Counter()
    for parameter in model.parameters():
        if not parameter.dtype.is_floating_point:
            raise ValueError("Quantized model parameters are not supported in this comparison")
        result[str(parameter.dtype)] += parameter.numel()
    if not result:
        raise ValueError("Model has no measurable parameters")
    return dict(sorted(result.items()))


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def load_comparison_model(model_name, settings, *, device, precision, checkpoint=None,
                          max_length=512, overlap=64):
    selected = validate_precision(model_name, precision, device)
    started = time.perf_counter()
    metadata = {"requested_precision": precision, "quantization": "none", "torch_version": torch.__version__}
    if model_name == "valendra":
        from valendra_privacy_filter.inference import Detector

        if checkpoint is None:
            raise ValueError("Valendra checkpoint is required")
        checkpoint = Path(checkpoint)
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise ValueError("Valendra Safetensors checkpoint is required")
        detector = Detector.from_pretrained(checkpoint, device=str(selected), precision=precision,
                                            max_length=max_length, overlap=overlap)
        adapter = ValendraAdapter(detector)
        metadata.update({"model_id": "Qwen/Qwen3-0.6B-Base", "weights_sha256": _sha256_file(weights),
                         "parameter_dtypes": _parameter_dtypes(detector.model),
                         "attention_mode": detector.model.config.privacy_attention_mode,
                         "max_length": max_length, "overlap": overlap,
                         "transformers_version": _package_version("transformers"), "span_scores": True})
        identity_path = checkpoint.parent / "run_identity.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            metadata["model_revision"] = identity.get("model_revision")
            metadata["training_dataset_manifest_sha256"] = identity.get("dataset_manifest_sha256")
            metadata["training_protocol_sha256"] = (getattr(detector.model.config, "privacy_continuation_protocol_sha256", None)
                                                     or identity.get("protocol_sha256"))
    elif model_name == "openai":
        from huggingface_hub import snapshot_download
        from opf import OPF

        distribution = importlib.metadata.distribution("opf")
        origin = json.loads(distribution.read_text("direct_url.json") or "{}")
        if origin.get("vcs_info", {}).get("commit_id") != settings["runtime_revision"]:
            raise ValueError("OPF runtime revision differs from the comparison configuration")
        snapshot = Path(snapshot_download(repo_id=settings["model_id"], revision=settings["model_revision"],
                                          allow_patterns=["original/*"]))
        original = snapshot / "original"
        weights = original / "model.safetensors"
        if not weights.is_file():
            raise ValueError("Pinned OPF original Safetensors checkpoint is missing")
        redactor = OPF(model=str(original), device=str(selected), output_mode="typed", decode_mode="viterbi")
        runtime = redactor.get_runtime()
        adapter = OpenAIAdapter(redactor)
        calibration = original / "viterbi_calibration.json"
        metadata.update({"model_id": settings["model_id"], "model_revision": settings["model_revision"],
                         "runtime_revision": settings["runtime_revision"], "license": settings["license"],
                         "weights_sha256": _sha256_file(weights), "parameter_dtypes": _parameter_dtypes(runtime.model),
                         "runtime_context_window": runtime.n_ctx, "decode_mode": "viterbi",
                         "calibration_sha256": _sha256_file(calibration) if calibration.is_file() else None,
                         "span_scores": False, "opf_version": _package_version("opf"),
                         "tiktoken_version": _package_version("tiktoken")})
    else:
        from gliner import GLiNER
        from huggingface_hub import snapshot_download

        if _package_version("gliner") != str(settings["runtime_version"]):
            raise ValueError("GLiNER runtime version differs from the comparison configuration")
        snapshot = Path(snapshot_download(repo_id=settings["model_id"], revision=settings["model_revision"],
                                          allow_patterns=["gliner_config.json", "model.safetensors", "pytorch_model.bin",
                                                          "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                                                          "added_tokens.json", "spm.model"]))
        weights = snapshot / "model.safetensors"
        if not weights.is_file():
            weights = snapshot / "pytorch_model.bin"
        if not weights.is_file():
            raise ValueError("Pinned GLiNER checkpoint is missing")
        model = GLiNER.from_pretrained(str(snapshot), map_location="cpu", strict=True, local_files_only=True)
        model = model.to(device=selected, dtype=torch.float32 if precision == "fp32" else torch.bfloat16).eval()
        adapter = GLiNERAdapter(model, list(settings["label_mapping"]), threshold=settings["threshold"],
                                 overlap_words=settings["overlap_words"])
        metadata.update({"model_id": settings["model_id"], "model_revision": settings["model_revision"],
                         "license": settings["license"], "weights_sha256": _sha256_file(weights),
                         "parameter_dtypes": _parameter_dtypes(model), "gliner_version": _package_version("gliner"),
                         "transformers_version": _package_version("transformers"),
                         "label_prompts": list(settings["label_mapping"]), "threshold": settings["threshold"],
                         "overlap_words": settings["overlap_words"], "max_words": adapter.max_words,
                         "max_encoder_tokens": adapter.max_tokens, "span_scores": True})
    metadata["load_seconds"] = time.perf_counter() - started
    return adapter, metadata
