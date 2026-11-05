"""Aggregate performance measurement without logging input documents."""

import platform
import time

import numpy as np
import psutil
import torch


def benchmark(detector, texts, warmup=1, repeats=3):
    texts = list(texts)
    if not texts or repeats < 1 or warmup < 0:
        raise ValueError("Benchmark needs input, nonnegative warmup and at least one repeat")
    is_cuda = detector.device.type == "cuda"
    for _ in range(warmup):
        detector.detect(texts[0])
    if is_cuda:
        torch.cuda.synchronize(detector.device)
        torch.cuda.reset_peak_memory_stats(detector.device)
    process = psutil.Process()
    initial_cpu = process.cpu_times()
    rss_peak = process.memory_info().rss
    timings, token_count = [], 0
    started = time.perf_counter()
    for _ in range(repeats):
        for text in texts:
            count = len(detector.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
            before = time.perf_counter()
            detector.detect(text)
            if is_cuda:
                torch.cuda.synchronize(detector.device)
            timings.append((time.perf_counter() - before) * 1000)
            token_count += count
            rss_peak = max(rss_peak, process.memory_info().rss)
    elapsed = time.perf_counter() - started
    final_cpu = process.cpu_times()
    cpu_seconds = (final_cpu.user + final_cpu.system) - (initial_cpu.user + initial_cpu.system)
    return {
        "documents": len(timings), "tokens": token_count, "elapsed_seconds": elapsed,
        "latency_ms": {f"p{p}": float(np.percentile(timings, p)) for p in (50, 95, 99)},
        "tokens_per_second": token_count / elapsed, "documents_per_second": len(timings) / elapsed,
        "ram_rss_bytes": rss_peak, "cpu_process_percent_one_core": 100 * cpu_seconds / elapsed,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(detector.device) if is_cuda else None,
        "device": str(detector.device), "precision": str(next(detector.model.parameters()).dtype),
        "hardware": torch.cuda.get_device_name(detector.device) if is_cuda else platform.processor(),
        "max_length": detector.max_length, "overlap": detector.overlap, "batch_size": 1,
        "warmup": warmup, "repeats": repeats,
    }
