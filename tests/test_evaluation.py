import pytest
import torch

from valendra_privacy_filter.evaluation.metrics import evaluate_documents
from valendra_privacy_filter.evaluation.calibration import fit_temperature
from valendra_privacy_filter.evaluation.benchmark import benchmark
from valendra_privacy_filter.evaluation.runner import compare_reports, evaluate_split
from valendra_privacy_filter.datasets.prepare import prepare_records
from valendra_privacy_filter.datasets.synthetic import generate_synthetic
from valendra_privacy_filter.inference import Detector
from test_inference import CharacterModel, CharacterTokenizer
from test_data import WindowTokenizer


def test_exact_span_metrics_confusions_languages_and_negatives():
    documents = [
        {"language": "es", "gold": [
            {"start": 0, "end": 3, "label": "PERSON"},
            {"start": 5, "end": 8, "label": "EMAIL"},
        ], "predicted": [
            {"start": 0, "end": 3, "type": "PERSON"},
            {"start": 5, "end": 8, "type": "PHONE"},
            {"start": 10, "end": 12, "type": "SECRET"},
        ]},
        {"language": "en", "gold": [], "predicted": [
            {"start": 2, "end": 4, "type": "PERSON"},
        ]},
    ]
    metrics = evaluate_documents(documents)
    assert metrics["overall"]["tp"] == 1
    assert metrics["overall"]["fp"] == 3
    assert metrics["overall"]["fn"] == 1
    assert metrics["overall"]["precision"] == pytest.approx(0.25)
    assert metrics["overall"]["recall"] == pytest.approx(0.5)
    assert metrics["overall"]["f1"] == pytest.approx(1 / 3)
    assert metrics["overall"]["f2"] == pytest.approx(5 / 12)
    assert metrics["by_category"]["EMAIL"]["recall"] == 0
    assert metrics["confusion"]["EMAIL"]["PHONE"] == 1
    assert metrics["false_positives_per_document"] == 1.5
    assert metrics["negative_documents"]["count"] == 1
    assert metrics["negative_documents"]["with_false_positive"] == 1
    assert metrics["by_language"]["es"]["tp"] == 1


def test_predictions_in_ignored_spans_are_not_counted():
    metrics = evaluate_documents([{"language": "es", "gold": [],
                                   "ignored": [{"start": 0, "end": 4}],
                                   "predicted": [{"start": 0, "end": 4, "type": "PERSON"}]}])
    assert metrics["overall"]["fp"] == 0
    assert metrics["negative_documents"]["count"] == 0


def test_temperature_is_fitted_without_using_ignored_labels():
    logits = torch.tensor([[8.0, 0.0], [8.0, 0.0], [0.0, 100.0]])
    labels = torch.tensor([0, 1, -100])
    temperature = fit_temperature(logits, labels)
    assert temperature > 1.0
    assert temperature == pytest.approx(fit_temperature(logits[:2], labels[:2]), rel=1e-5)
    with torch.inference_mode():
        inference_logits = logits.clone()
        inference_labels = labels.clone()
    assert fit_temperature(inference_logits, inference_labels) == pytest.approx(temperature, rel=1e-5)


def test_benchmark_reports_aggregate_performance_without_text():
    detector = Detector(CharacterModel(), CharacterTokenizer(), max_length=8, overlap=2)
    report = benchmark(detector, ["Hi Ada!", "Hello!"], warmup=0, repeats=1)
    assert report["documents"] == 2
    assert report["tokens"] == 13
    assert 0 < report["latency_ms"]["p50"] <= report["latency_ms"]["p99"]
    assert report["documents_per_second"] > 0
    assert report["ram_rss_bytes"] > 0
    assert "Ada" not in str(report)


def test_comparison_rejects_different_training_protocols():
    common = {"split": "test", "dataset_manifest_sha256": "data", "seed": 42,
              "training_dataset_manifest_sha256": "training-data", "model_revision": "model",
              "max_length": 512, "overlap": 64, "protocol_sha256": "same"}
    causal = {"metadata": {**common, "attention_mode": "causal"}}
    bidirectional = {"metadata": {**common, "attention_mode": "bidirectional"}}
    assert compare_reports(causal, bidirectional)["split"] == "test"
    bidirectional["metadata"]["protocol_sha256"] = "different"
    with pytest.raises(ValueError, match="protocol"):
        compare_reports(causal, bidirectional)


def test_evaluation_accepts_an_independent_dataset_with_separate_provenance(tmp_path):
    import json

    data = tmp_path / "data"
    prepare_records(generate_synthetic(per_split=1), WindowTokenizer(), data, max_length=100, overlap=10)
    run = tmp_path / "run"
    checkpoint = run / "final"
    checkpoint.mkdir(parents=True)
    (run / "run_identity.json").write_text(json.dumps({"dataset_manifest_sha256": "training-digest",
                                                       "seed": 42, "model_revision": "model",
                                                       "protocol_sha256": "protocol"}))
    detector = Detector(CharacterModel(), CharacterTokenizer(), max_length=128, overlap=16)
    result = evaluate_split(detector, data, "test", checkpoint)
    assert result["documents"] == 1
    assert result["metadata"]["training_dataset_manifest_sha256"] == "training-digest"
    assert result["metadata"]["dataset_manifest_sha256"] != "training-digest"
    detector.model.config.privacy_continuation_protocol_sha256 = "continuation-protocol"
    continued = evaluate_split(detector, data, "test", checkpoint)
    assert continued["metadata"]["protocol_sha256"] == "continuation-protocol"
    assert continued["metadata"]["base_protocol_sha256"] == "protocol"
