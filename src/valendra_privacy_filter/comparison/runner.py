"""Score one model per process and compare aggregate reports on identical inputs."""

import hashlib
import heapq
import itertools
from pathlib import Path
import platform
import time

import numpy as np
import psutil
import pyarrow.parquet as pq
import torch

from valendra_privacy_filter.comparison.adapters import ComparisonAdapterError
from valendra_privacy_filter.comparison.metrics import COMMON_CATEGORIES, ComparisonMetrics
from valendra_privacy_filter.datasets.prepare import verify_dataset


def _implementation_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = ("comparison/adapters.py", "comparison/config.py", "comparison/loaders.py", "comparison/metrics.py",
             "comparison/runner.py", "comparison/worker.py", "datasets/prepare.py", "evaluation/metrics.py",
             "inference/detector.py", "models/decode.py")
    digest = hashlib.sha256()
    for name in paths:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _documents(dataset_dir, split, limit=None, sample_seed=None):
    path = Path(dataset_dir) / f"{split}.parquet"
    selected = None
    if sample_seed is not None:
        if type(sample_seed) is not int or type(limit) is not int or limit < 1:
            raise ValueError("Seeded document sampling requires an integer seed and a positive limit")
        ids = set()
        for batch in pq.ParquetFile(path).iter_batches(columns=["document_id"]):
            ids.update(batch.column(0).to_pylist())
        selected = set(heapq.nsmallest(limit, ids, key=lambda value: hashlib.sha256(
            f"{sample_seed}:{value}".encode("ascii")).digest()))
    seen = set()
    columns = ["document_id", "text", "language", "source", "spans", "ignored"]
    emitted = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=columns):
        for row in batch.to_pylist():
            if selected is not None and row["document_id"] not in selected:
                continue
            if row["document_id"] in seen:
                continue
            if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row["document_id"]:
                raise ValueError("Dataset document hash mismatch")
            seen.add(row["document_id"])
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _detect(adapter, text):
    try:
        return adapter.detect(text)
    except ComparisonAdapterError:
        raise
    except Exception as exc:
        raise RuntimeError(f"model_inference_failed:{type(exc).__name__}") from None


def score_dataset(adapter, dataset_dir, split, label_map, *, model_name, config_sha256,
                  device="cpu", warmup=2, limit=None, common_categories=COMMON_CATEGORIES,
                  model_metadata=None, sample_seed=None):
    if split not in {"train", "validation", "test", "challenge"}:
        raise ValueError("Unsupported split")
    if model_name not in {"valendra", "openai", "gliner"}:
        raise ValueError("Unsupported comparison model")
    if (type(warmup) is not int or warmup < 0 or (limit is not None and (type(limit) is not int or limit < 1))
            or (sample_seed is not None and (type(sample_seed) is not int or limit is None))):
        raise ValueError("Invalid warmup or document limit")
    dataset_dir = Path(dataset_dir)
    verify_dataset(dataset_dir, [split])
    dataset_sha = hashlib.sha256((dataset_dir / "manifest.json").read_bytes()).hexdigest()
    torch_device = torch.device(device)
    if torch_device.type not in {"cpu", "cuda"} or (torch_device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested comparison device is unavailable")
    is_cuda = torch_device.type == "cuda"
    for row in itertools.islice(_documents(dataset_dir, split, limit, sample_seed), warmup):
        _detect(adapter, row["text"])
    if is_cuda:
        torch.cuda.synchronize(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
    process = psutil.Process()
    cpu_before = process.cpu_times()
    rss_peak = process.memory_info().rss
    score = ComparisonMetrics(label_map, common_categories)
    document_order = hashlib.sha256()
    latencies, characters, token_count = [], 0, 0
    token_counter = getattr(adapter, "count_tokens", None)
    started = time.perf_counter()
    for row in _documents(dataset_dir, split, limit, sample_seed):
        text = row["text"]
        if callable(token_counter):
            try:
                token_count += int(token_counter(text))
            except Exception as exc:
                raise RuntimeError(f"token_count_failed:{type(exc).__name__}") from None
        if is_cuda:
            torch.cuda.synchronize(torch_device)
        before = time.perf_counter()
        predicted = _detect(adapter, text)
        if is_cuda:
            torch.cuda.synchronize(torch_device)
        latencies.append(time.perf_counter() - before)
        characters += len(text)
        document_order.update((row["document_id"] + "\n").encode("ascii"))
        score.add({"text": text, "language": row["language"], "source": row["source"],
                   "gold": row["spans"], "ignored": row["ignored"], "predicted": predicted})
        rss_peak = max(rss_peak, process.memory_info().rss)
    elapsed = time.perf_counter() - started
    metrics = score.report()
    if not metrics["documents"]:
        raise ValueError("Evaluation split is empty")
    inference_seconds = sum(latencies)
    cpu_after = process.cpu_times()
    cpu_seconds = (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
    metadata = {"model": model_name, "dataset_manifest_sha256": dataset_sha, "split": split,
                "document_order_sha256": document_order.hexdigest(), "comparison_config_sha256": config_sha256,
                "implementation_sha256": _implementation_sha256(),
                "documents": metrics["documents"], "limit": limit, "sample_seed": sample_seed,
                "common_categories": sorted(common_categories),
                "device": str(torch_device), "hardware": torch.cuda.get_device_name(torch_device) if is_cuda else platform.processor() or platform.machine(),
                "warmup_documents": min(warmup, metrics["documents"]), "batch_size": 1,
                "scoring": "exact_character_spans_v1", "model_details": model_metadata or {}}
    performance = {
        "latency_ms": {f"p{percentile}": float(np.percentile(latencies, percentile) * 1000) for percentile in (50, 95, 99)},
        "documents_per_second": metrics["documents"] / inference_seconds,
        "characters_per_second": characters / inference_seconds,
        "input_tokens_per_second": token_count / inference_seconds if callable(token_counter) else None,
        "input_tokens": token_count if callable(token_counter) else None,
        "input_token_definition": "model tokenizer on original text; excludes prompt and repeated windows" if callable(token_counter) else None,
        "inference_seconds": inference_seconds, "end_to_end_seconds": elapsed,
        "sampled_peak_rss_bytes": rss_peak,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(torch_device) if is_cuda else None,
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(torch_device) if is_cuda else None,
        "cpu_process_percent_one_core": 100 * cpu_seconds / elapsed,
    }
    return {"metadata": metadata, "metrics": metrics, "performance": performance}


def compare_baseline_reports(reports):
    reports = list(reports)
    if len(reports) != 3 or {item["metadata"].get("model") for item in reports} != {"valendra", "openai", "gliner"}:
        raise ValueError("Expected exactly one report per model")
    keys = ("dataset_manifest_sha256", "split", "document_order_sha256", "comparison_config_sha256",
             "implementation_sha256", "documents", "device", "hardware", "limit", "sample_seed",
             "common_categories", "scoring", "batch_size", "warmup_documents")
    reference = reports[0]["metadata"]
    if any(any(key not in item["metadata"] or item["metadata"][key] != reference.get(key) for key in keys)
           for item in reports):
        raise ValueError("Incompatible comparison report inputs or hardware")
    if any(item["metrics"].get("documents") != reference["documents"] for item in reports):
        raise ValueError("Incompatible comparison report denominators")
    return {"dataset_manifest_sha256": reference["dataset_manifest_sha256"], "split": reference["split"],
            "documents": reference["documents"], "models": {item["metadata"]["model"]: item for item in reports}}
