"""Run one pinned comparison backend and write only aggregate results."""

import argparse
import json
from pathlib import Path
import sys

from valendra_privacy_filter.comparison.adapters import ComparisonAdapterError
from valendra_privacy_filter.comparison.config import load_comparison_config
from valendra_privacy_filter.comparison.loaders import load_comparison_model
from valendra_privacy_filter.comparison.runner import score_dataset
from valendra_privacy_filter.datasets.prepare import verify_dataset

SAFE_ADAPTER_ERRORS = frozenset({
    "invalid_prediction_offsets_or_label", "invalid_prediction_score", "openai_typed_output_required",
    "openai_tokenizer_roundtrip_mismatch", "gliner_invalid_token_offsets", "gliner_window_capacity_insufficient",
    "gliner_window_tokenization_mismatch", "gliner_unexpected_truncation", "gliner_unrequested_label",
    "gliner_window_capacity_insufficient_for_overlap",
})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score a local PII baseline on one prepared dataset split.")
    parser.add_argument("--model", choices=("valendra", "openai", "gliner"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test", "challenge"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16", "native"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or not args.output.parent.is_dir():
            raise ValueError("Report destination must be new and its parent must exist")
        if args.model != "valendra" and args.checkpoint is not None:
            raise ValueError("External baselines always use their pinned model revisions")
        config = load_comparison_config(args.config)
        verify_dataset(args.data_dir, [args.split])
        settings = config["models"][args.model]
        adapter, details = load_comparison_model(args.model, settings, device=args.device, precision=args.precision,
                                                  checkpoint=args.checkpoint, max_length=args.max_length,
                                                  overlap=args.overlap)
        details["label_mapping"] = settings["label_mapping"]
        report = score_dataset(adapter, args.data_dir, args.split, settings["label_mapping"],
                               model_name=args.model, config_sha256=config["sha256"], device=args.device,
                               warmup=args.warmup, limit=args.limit, common_categories=config["common_categories"],
                               model_metadata=details, sample_seed=args.sample_seed)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        print(json.dumps({"model": args.model, "documents": report["metrics"]["documents"], "report_written": True}))
        return 0
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ComparisonAdapterError) and str(exc) in SAFE_ADAPTER_ERRORS else type(exc).__name__
        print(f"Comparison failed: {reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
