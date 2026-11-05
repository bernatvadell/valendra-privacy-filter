import pytest

from valendra_privacy_filter.comparison.metrics import ComparisonMetrics
from valendra_privacy_filter.comparison.runner import _documents, compare_baseline_reports, score_dataset
from valendra_privacy_filter.datasets.prepare import prepare_records
from test_data import WindowTokenizer


def test_untyped_scoring_keeps_broader_dates_without_mislabeling_birth_date():
    text = "Ana: 1990-01-01"
    scorer = ComparisonMetrics({"private_person": "PERSON", "private_date": None})
    scorer.add({"text": text, "language": "es", "source": "challenge", "ignored": [],
                "gold": [{"start": 0, "end": 3, "label": "PERSON"},
                         {"start": 5, "end": 15, "label": "DATE_OF_BIRTH"}],
                "predicted": [{"start": 0, "end": 3, "label": "private_person", "value": "Ana"},
                              {"start": 5, "end": 15, "label": "private_date", "value": "1990-01-01"}]})
    report = scorer.report()
    assert report["untyped"]["f1"] == 1.0
    assert report["typed_common"]["f1"] == 1.0
    assert report["untyped_recall_by_gold_category"]["DATE_OF_BIRTH"]["recall"] == 1.0
    assert report["unmapped_predictions"]["private_date"] == 1
    assert report["typed_excluded_gold"]["DATE_OF_BIRTH"] == 1


def test_ignore_masks_negatives_and_unicode_offsets_use_same_denominators():
    scorer = ComparisonMetrics({"email": "EMAIL", "other": None})
    scorer.add({"text": "💬ana@example.com!", "language": "es", "source": "test",
                "gold": [{"start": 1, "end": 16, "label": "EMAIL"}], "ignored": [],
                "predicted": [{"start": 1, "end": 16, "label": "email", "value": "ana@example.com"}]})
    scorer.add({"text": "12345", "language": "en", "source": "test", "gold": [],
                "ignored": [{"start": 0, "end": 2}],
                "predicted": [{"start": 0, "end": 2, "label": "other", "value": "12"}]})
    scorer.add({"text": "nothing", "language": "en", "source": "test", "gold": [], "ignored": [],
                "predicted": [{"start": 0, "end": 3, "label": "other", "value": "not"}]})
    report = scorer.report()
    assert report["documents"] == 3
    assert report["untyped"]["tp"] == 1 and report["untyped"]["fp"] == 1
    assert report["false_positives_per_document"] == pytest.approx(1 / 3)
    assert report["negative_documents"] == {"count": 1, "with_false_positive": 1, "false_positive_rate": 1.0}
    assert report["ignored_predictions"] == 1
    assert report["by_language"]["es"]["untyped"]["f1"] == 1.0


def test_typed_common_reports_precision_and_recall_by_category():
    scorer = ComparisonMetrics({"name": "PERSON", "address": "ADDRESS"})
    scorer.add({"text": "Ana x@y.es", "language": "es", "source": "test", "ignored": [],
                "gold": [{"start": 0, "end": 3, "label": "PERSON"},
                         {"start": 4, "end": 10, "label": "EMAIL"}],
                "predicted": [{"start": 0, "end": 3, "label": "name", "value": "Ana"},
                              {"start": 4, "end": 10, "label": "address", "value": "x@y.es"}]})
    report = scorer.report()
    assert report["typed_by_category"]["PERSON"]["f1"] == 1.0
    assert report["typed_by_category"]["EMAIL"]["recall"] == 0.0
    assert report["typed_by_category"]["ADDRESS"]["precision"] == 0.0


@pytest.mark.parametrize("prediction", [
    {"start": 0, "end": 100, "label": "email", "value": "secret"},
    {"start": 0, "end": 3, "label": "email", "value": "wrong"},
    {"start": True, "end": 3, "label": "email", "value": "Ana"},
])
def test_invalid_predictions_fail_without_exposing_input(prediction):
    scorer = ComparisonMetrics({"email": "EMAIL"})
    with pytest.raises(ValueError, match="invalid prediction") as error:
        scorer.add({"text": "Ana", "gold": [], "ignored": [], "predicted": [prediction]})
    assert "Ana" not in str(error.value)


def test_dataset_scoring_is_aggregate_only_and_has_stable_document_fingerprint(tmp_path):
    class FakeAdapter:
        def detect(self, text):
            return [{"start": 0, "end": 3, "label": "name", "value": "Ana", "score": 0.9}]

        def count_tokens(self, text):
            return len(text)

    text = "Ana llegó."
    record = {"source_text": text, "language": "es", "region": "ES", "source": "openpii",
              "privacy_mask": [{"start": 0, "end": 3, "label": "PERSON", "value": "Ana"}]}
    prepare_records({"test": [record]}, WindowTokenizer(), tmp_path, max_length=64, overlap=8)
    first = score_dataset(FakeAdapter(), tmp_path, "test", {"name": "PERSON"}, model_name="valendra",
                          config_sha256="policy", device="cpu", warmup=0)
    second = score_dataset(FakeAdapter(), tmp_path, "test", {"name": "PERSON"}, model_name="valendra",
                           config_sha256="policy", device="cpu", warmup=0)
    assert first["metrics"]["untyped"]["f1"] == 1.0
    assert first["performance"]["documents_per_second"] > 0
    assert first["metadata"]["document_order_sha256"] == second["metadata"]["document_order_sha256"]
    assert len(first["metadata"]["implementation_sha256"]) == 64
    assert "Ana" not in str(first)


def test_sample_selects_documents_from_whole_split_deterministically(tmp_path):
    records = [{"source_text": f"Ana item {number}", "language": "es", "region": "ES", "source": "openpii",
                "privacy_mask": [{"start": 0, "end": 3, "label": "PERSON", "value": "Ana"}]}
               for number in range(30)]
    prepare_records({"validation": records}, WindowTokenizer(), tmp_path, max_length=64, overlap=8)
    all_ids = [row["document_id"] for row in _documents(tmp_path, "validation")]
    first = [row["document_id"] for row in _documents(tmp_path, "validation", 8, sample_seed=42)]
    repeated = [row["document_id"] for row in _documents(tmp_path, "validation", 8, sample_seed=42)]
    different = [row["document_id"] for row in _documents(tmp_path, "validation", 8, sample_seed=43)]
    assert len(first) == len(set(first)) == 8
    assert first == repeated
    assert first != all_ids[:8]
    assert first != different


def test_cross_model_report_requires_identical_input_and_all_three_models():
    common = {"dataset_manifest_sha256": "data", "split": "test", "document_order_sha256": "docs",
              "comparison_config_sha256": "policy", "documents": 42, "device": "cpu", "hardware": "cpu",
              "limit": None, "common_categories": ["PERSON", "EMAIL"], "scoring": "exact_character_spans_v1",
              "batch_size": 1, "warmup_documents": 2, "implementation_sha256": "code", "sample_seed": 42}
    reports = [{"metadata": {**common, "model": name}, "metrics": {"documents": 42}}
               for name in ("valendra", "openai", "gliner")]
    assert set(compare_baseline_reports(reports)["models"]) == {"valendra", "openai", "gliner"}
    reports[2]["metadata"]["document_order_sha256"] = "different"
    with pytest.raises(ValueError, match="(?i)incompatible"):
        compare_baseline_reports(reports)


def test_cross_model_report_rejects_different_sampling_seeds():
    common = {"dataset_manifest_sha256": "data", "split": "validation", "document_order_sha256": "docs",
              "comparison_config_sha256": "policy", "documents": 42, "device": "cpu", "hardware": "cpu",
              "limit": 42, "common_categories": ["PERSON"], "scoring": "exact_character_spans_v1",
              "batch_size": 1, "warmup_documents": 2, "implementation_sha256": "code", "sample_seed": 42}
    reports = [{"metadata": {**common, "model": name}, "metrics": {"documents": 42}}
               for name in ("valendra", "openai", "gliner")]
    reports[2]["metadata"]["sample_seed"] = 43
    with pytest.raises(ValueError, match="(?i)incompatible"):
        compare_baseline_reports(reports)


def test_cross_model_report_rejects_different_benchmark_warmup():
    common = {"dataset_manifest_sha256": "data", "split": "test", "document_order_sha256": "docs",
              "comparison_config_sha256": "policy", "documents": 42, "device": "cpu", "hardware": "cpu",
              "limit": None, "common_categories": ["PERSON"], "scoring": "exact_character_spans_v1",
              "batch_size": 1, "warmup_documents": 2, "implementation_sha256": "code", "sample_seed": None}
    reports = [{"metadata": {**common, "model": name}, "metrics": {"documents": 42}}
               for name in ("valendra", "openai", "gliner")]
    reports[2]["metadata"]["warmup_documents"] = 3
    with pytest.raises(ValueError, match="(?i)incompatible"):
        compare_baseline_reports(reports)
