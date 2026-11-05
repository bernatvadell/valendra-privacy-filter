"""Run each baseline sequentially in its isolated Python environment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from valendra_privacy_filter.comparison.config import load_comparison_config
from valendra_privacy_filter.comparison.runner import compare_baseline_reports
from valendra_privacy_filter.datasets.prepare import verify_dataset


def run_suite(*, checkpoint, data_dir, split, output_dir, config, python_paths,
              device="cuda", precision="bf16", warmup=2, limit=None, sample_seed=None):
    root = Path(__file__).resolve().parents[3]
    worker = root / "scripts/benchmark_model.py"
    checkpoint, data_dir, output_dir, config = map(lambda p: Path(p).resolve(), (checkpoint, data_dir, output_dir, config))
    if not worker.is_file() or not (checkpoint / "model.safetensors").is_file():
        raise ValueError("A source checkout and Valendra Safetensors checkpoint are required")
    if output_dir.exists():
        raise ValueError("Output directory must not already exist")
    if set(python_paths) != {"valendra", "openai", "gliner"} or any(not Path(path).is_file() for path in python_paths.values()):
        raise ValueError("Three Python executables are required")
    if precision not in {"fp32", "bf16"} or split not in {"validation", "test", "challenge"}:
        raise ValueError("Invalid benchmark precision or evaluation split")
    if (limit is not None and (type(limit) is not int or limit < 1)) or (
            sample_seed is not None and (type(sample_seed) is not int or limit is None)):
        raise ValueError("Seeded document sampling requires an integer seed and a positive limit")
    policy = load_comparison_config(config)
    manifest = verify_dataset(data_dir, [split])
    if not manifest["splits"][split]["documents"]:
        raise ValueError("Evaluation split is empty")
    dataset_sha = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    output_dir.mkdir(parents=True)
    # The public baselines do not need cloud or API credentials. Preserve the
    # runtime and cache settings needed by isolated local processes only.
    allowed = ("PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "LC_CTYPE",
               "SYSTEMROOT", "WINDIR", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
               "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "SSL_CERT_FILE", "SSL_CERT_DIR",
               "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_XET",
               "XDG_CACHE_HOME", "TIKTOKEN_CACHE_DIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1"})
    reports = []
    for name in ("valendra", "openai", "gliner"):
        path = output_dir / f"{name}.json"
        command = [os.path.abspath(python_paths[name]), str(worker), "--model", name,
                   "--data-dir", str(data_dir), "--split", split, "--config", str(config), "--device", device,
                   "--precision", "native" if name == "openai" else precision, "--output", str(path),
                   "--warmup", str(warmup)]
        if name == "valendra":
            command += ["--checkpoint", str(checkpoint)]
        if limit is not None:
            command += ["--limit", str(limit)]
        if sample_seed is not None:
            command += ["--sample-seed", str(sample_seed)]
        subprocess.run(command, check=True, env=env)
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    result = compare_baseline_reports(reports)
    expected = min(manifest["splits"][split]["documents"], limit) if limit is not None else manifest["splits"][split]["documents"]
    if (result["dataset_manifest_sha256"] != dataset_sha or result["split"] != split or result["documents"] != expected
            or any(item["metadata"]["comparison_config_sha256"] != policy["sha256"] for item in reports)):
        raise ValueError("Comparison reports do not match the requested dataset or policy")
    with (output_dir / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    return result


def main(argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Compare Valendra, OpenAI and GLiNER sequentially.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test", "challenge"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/comparison-models.yaml")
    parser.add_argument("--valendra-python", default=str(root / ".venv/bin/python"))
    parser.add_argument("--openai-python", default=str(root / "benchmarks/openai/.venv/bin/python"))
    parser.add_argument("--gliner-python", default=str(root / "benchmarks/gliner/.venv/bin/python"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-seed", type=int)
    args = parser.parse_args(argv)
    try:
        result = run_suite(checkpoint=args.checkpoint, data_dir=args.data_dir, split=args.split,
                           output_dir=args.output_dir, config=args.config,
                           python_paths={"valendra": args.valendra_python, "openai": args.openai_python,
                                         "gliner": args.gliner_python},
                           device=args.device, precision=args.precision, warmup=args.warmup, limit=args.limit,
                           sample_seed=args.sample_seed)
    except Exception as exc:
        print(f"Comparison suite failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps({"split": result["split"], "documents": result["documents"], "report_written": True}))
    return 0
