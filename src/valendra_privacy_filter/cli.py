"""Command line entry points; never print raw source inputs in diagnostics."""

import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Optional

import typer
import yaml

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _input_text(path):
    return Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()


def _detector(checkpoint, device, precision, max_length, overlap, calibration):
    from valendra_privacy_filter.inference import Detector
    temperature = 1.0
    if calibration:
        temperature = float(json.loads(Path(calibration).read_text(encoding="utf-8"))["temperature"])
    return Detector.from_pretrained(checkpoint, device=device, precision=precision,
                                    max_length=max_length, overlap=overlap, temperature=temperature)


@app.command("generate-synthetic")
def generate_synthetic(output_dir: Path = typer.Option(...), seed: int = 42, per_split: int = 100):
    from valendra_privacy_filter.datasets.synthetic import generate_synthetic as generate
    rows = generate(seed=seed, per_split=per_split)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    typer.echo(json.dumps({split: len(items) for split, items in rows.items()}))


@app.command("validate-annotations")
def validate_annotations(input: Path = typer.Option(..., exists=True)):
    from valendra_privacy_filter.datasets.records import normalize_record
    valid = invalid = 0
    for row in _read_rows(input):
        try:
            normalize_record(row)
            valid += 1
        except (ValueError, KeyError, TypeError):
            invalid += 1
    typer.echo(json.dumps({"valid": valid, "invalid": invalid}))
    if invalid:
        raise typer.Exit(code=1)


@app.command("prepare-dataset")
def prepare_dataset(output_dir: Path = typer.Option(...), synthetic_dir: Optional[Path] = None,
                    openpii_limit: Optional[int] = None, skip_openpii: bool = False,
                    max_length: int = 512, overlap: int = 64, seed: int = 42,
                    taxonomy: Optional[Path] = None, unknown_policy: str = "ignore",
                    spanish_all_regions: bool = False):
    from transformers import AutoTokenizer
    from valendra_privacy_filter.datasets.prepare import prepare_records, stream_openpii, SPLITS
    from valendra_privacy_filter.training.smoke import MODEL_ID, MODEL_REVISION
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sources = {}
    if synthetic_dir:
        sources.update({split: _read_rows(synthetic_dir / f"{split}.jsonl") for split in SPLITS})
    if not skip_openpii:
        sources["openpii"] = stream_openpii(limit=openpii_limit, spanish_regions=() if spanish_all_regions else ("ES",))
    if not sources:
        raise typer.BadParameter("At least one data source is required")
    if openpii_limit is not None and openpii_limit < 1:
        raise typer.BadParameter("openpii-limit must be positive")
    categories = yaml.safe_load(taxonomy.read_text())["categories"] if taxonomy else None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    source_parameters = {"openpii_limit": openpii_limit, "skip_openpii": skip_openpii,
                         "spanish_regions": "all" if spanish_all_regions else ["ES"], "languages": ["es", "en"]}
    manifest = prepare_records(sources, tokenizer, output_dir, max_length, overlap, seed, categories,
                               unknown_policy, source_parameters)
    typer.echo(json.dumps({"splits": manifest["splits"], "duplicates": manifest["duplicates_removed"],
                           "quarantine": manifest["quarantine_reasons"]}, ensure_ascii=False))


@app.command("prepare-nemotron")
def prepare_nemotron(baseline_dir: Path = typer.Option(..., exists=True), output_dir: Path = typer.Option(...),
                     label_policy: Optional[Path] = typer.Option(None, exists=True),
                     train_limit: Optional[int] = None, test_limit: Optional[int] = None, seed: int = 42):
    from transformers import AutoTokenizer
    from valendra_privacy_filter.datasets.nemotron import prepare_nemotron_records, stream_nemotron
    from valendra_privacy_filter.training.smoke import MODEL_ID, MODEL_REVISION

    if any(limit is not None and limit < 1 for limit in (train_limit, test_limit)):
        raise typer.BadParameter("Source limits must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    manifest = prepare_nemotron_records(
        baseline_dir, output_dir, tokenizer, label_policy, train_rows=stream_nemotron("train", train_limit),
        test_rows=stream_nemotron("test", test_limit), seed=seed,
        train_is_complete=train_limit is None, test_is_complete=test_limit is None,
    )
    typer.echo(json.dumps({"splits": manifest["splits"], "filter_statistics": manifest["source_parameters"]["filter_statistics"],
                           "duplicates": manifest["duplicates_removed"], "quarantine": manifest["quarantine_reasons"]},
                          ensure_ascii=False))


@app.command("merge-datasets")
def merge_datasets(baseline_dir: Path = typer.Option(..., exists=True), nvidia_dir: Path = typer.Option(..., exists=True),
                   output_dir: Path = typer.Option(...), allow_sampled_holdout: bool = False,
                   challenge_supplement_dir: Optional[Path] = None):
    from valendra_privacy_filter.datasets.mixed import merge_prepared_datasets
    manifest = merge_prepared_datasets(baseline_dir, nvidia_dir, output_dir, allow_sampled_holdout,
                                       challenge_supplement_dir)
    typer.echo(json.dumps({"splits": manifest["splits"], "leakage_checks": manifest["leakage_checks"]}, ensure_ascii=False))


@app.command("prepare-address-challenge")
def prepare_address_challenge(baseline_dir: Path = typer.Option(..., exists=True), output_dir: Path = typer.Option(...),
                              count: int = 1000, seed: int = 42):
    from transformers import AutoTokenizer
    from valendra_privacy_filter.datasets.challenge import prepare_address_challenge as prepare
    from valendra_privacy_filter.training.smoke import MODEL_ID, MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    manifest = prepare(baseline_dir, output_dir, tokenizer, count=count, seed=seed)
    typer.echo(json.dumps({"challenge": manifest["splits"]["challenge"]}, ensure_ascii=False))


@app.command("audit-dataset-isolation")
def audit_dataset_isolation(data_dir: Path = typer.Option(..., exists=True), holdout_dir: Optional[Path] = None,
                            output: Optional[Path] = None):
    from valendra_privacy_filter.datasets.mixed import audit_prepared_isolation
    result = audit_prepared_isolation(data_dir, holdout_dir)
    result["dataset_manifest_sha256"] = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    if holdout_dir is not None:
        result["holdout_manifest_sha256"] = hashlib.sha256((holdout_dir / "manifest.json").read_bytes()).hexdigest()
    if output is not None:
        _write_json(output, result)
    typer.echo(json.dumps(result, indent=2))
    if not result["passed"]:
        raise typer.Exit(code=1)


@app.command("audit-dataset-annotations")
def audit_dataset_annotations(data_dir: Path = typer.Option(..., exists=True), output: Optional[Path] = None):
    from valendra_privacy_filter.datasets.audit import audit_prepared_annotations
    result = audit_prepared_annotations(data_dir)
    result["dataset_manifest_sha256"] = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
    if output is not None:
        _write_json(output, result)
    typer.echo(json.dumps(result, indent=2))
    if not result["passed"]:
        raise typer.Exit(code=1)


@app.command("smoke-test")
def smoke_test(output_dir: Path = typer.Option(...), device: str = "cuda", attention_mode: str = "causal",
               sequence_length: int = 512, gradient_checkpointing: bool = True):
    from valendra_privacy_filter.training.smoke import official_smoke
    typer.echo(json.dumps(official_smoke(output_dir, device=device, attention_mode=attention_mode,
                                         sequence_length=sequence_length, gradient_checkpointing=gradient_checkpointing), indent=2))


@app.command("train")
def train(config: Path = typer.Option(..., exists=True), resume_from_checkpoint: Optional[Path] = None,
          min_early_stopping_epoch: Optional[float] = None, initialize_from_checkpoint: Optional[Path] = None):
    from valendra_privacy_filter.training.train import train as execute
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    typer.echo(json.dumps(execute(settings, str(resume_from_checkpoint) if resume_from_checkpoint else None,
                                  min_early_stopping_epoch=min_early_stopping_epoch,
                                  initialize_from_checkpoint=str(initialize_from_checkpoint) if initialize_from_checkpoint else None), indent=2))


@app.command("evaluate")
def evaluate(checkpoint: Path = typer.Option(..., exists=True), data_dir: Path = typer.Option(..., exists=True),
             split: str = "test", output: Path = typer.Option(...), device: str = "cpu", precision: str = "fp32",
             max_length: int = 512, overlap: int = 64, calibration: Optional[Path] = None):
    from valendra_privacy_filter.evaluation.runner import evaluate_split
    detector = _detector(checkpoint, device, precision, max_length, overlap, calibration)
    result = evaluate_split(detector, data_dir, split, checkpoint)
    _write_json(output, result)
    typer.echo(json.dumps({"output": str(output), "documents": result["documents"]}))


@app.command("compare-experiments")
def compare_experiments(first: Path = typer.Option(..., exists=True), second: Path = typer.Option(..., exists=True),
                        output: Path = typer.Option(...)):
    from valendra_privacy_filter.evaluation.runner import compare_reports
    result = compare_reports(json.loads(first.read_text()), json.loads(second.read_text()))
    _write_json(output, result)
    typer.echo(json.dumps({"output": str(output), "split": result["split"]}))


@app.command("compare-baselines")
def compare_baselines(valendra: Path = typer.Option(..., exists=True), openai: Path = typer.Option(..., exists=True),
                      gliner: Path = typer.Option(..., exists=True), output: Path = typer.Option(...)):
    from valendra_privacy_filter.comparison.runner import compare_baseline_reports

    if output.exists() or not output.parent.is_dir():
        raise typer.BadParameter("Output must be new and its parent directory must exist")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (valendra, openai, gliner)]
    result = compare_baseline_reports(reports)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    typer.echo(json.dumps({"split": result["split"], "documents": result["documents"], "report_written": True}))


@app.command("calibrate")
def calibrate(checkpoint: Path = typer.Option(..., exists=True), data_dir: Path = typer.Option(..., exists=True),
              output: Path = typer.Option(...), device: str = "cpu", max_tokens: int = 100_000):
    from valendra_privacy_filter.evaluation.calibration import calibrate_checkpoint
    typer.echo(json.dumps(calibrate_checkpoint(checkpoint, data_dir, output, device, max_tokens), indent=2))


@app.command("benchmark")
def benchmark(checkpoint: Path = typer.Option(..., exists=True), input: Path = typer.Option(..., exists=True),
              output: Path = typer.Option(...), device: str = "cpu", precision: str = "fp32",
              max_length: int = 512, overlap: int = 64, repeats: int = 3):
    from valendra_privacy_filter.evaluation.benchmark import benchmark as run
    detector = _detector(checkpoint, device, precision, max_length, overlap, None)
    texts = input.read_text(encoding="utf-8").splitlines()
    result = run(detector, texts, repeats=repeats)
    _write_json(output, result)
    typer.echo(json.dumps(result, indent=2))


@app.command("detect")
def detect(checkpoint: Path = typer.Option(..., exists=True), input: Optional[Path] = None,
           device: str = "cpu", precision: str = "fp32", max_length: int = 512, overlap: int = 64,
           calibration: Optional[Path] = None):
    detector = _detector(checkpoint, device, precision, max_length, overlap, calibration)
    typer.echo(json.dumps(detector.detect(_input_text(input)), ensure_ascii=False))


@app.command("redact")
def redact(checkpoint: Path = typer.Option(..., exists=True), input: Optional[Path] = None,
           device: str = "cpu", precision: str = "fp32", max_length: int = 512, overlap: int = 64,
           calibration: Optional[Path] = None):
    detector = _detector(checkpoint, device, precision, max_length, overlap, calibration)
    typer.echo(detector.redact(_input_text(input)), nl=False)


@app.command("serve")
def serve(checkpoint: Path = typer.Option(..., exists=True), host: str = "127.0.0.1", port: int = 8000,
          device: str = "cpu", precision: str = "fp32", max_length: int = 512, overlap: int = 64,
          max_chars: int = 100_000, max_batch: int = 32, max_body_bytes: int = 1_000_000,
          calibration: Optional[Path] = None):
    import uvicorn
    from valendra_privacy_filter.api.app import create_app
    detector = _detector(checkpoint, device, precision, max_length, overlap, calibration)
    uvicorn.run(create_app(detector, max_chars, max_batch, max_body_bytes), host=host, port=port, access_log=False)


if __name__ == "__main__":
    app()
