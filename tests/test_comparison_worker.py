import json
from pathlib import Path

from valendra_privacy_filter.comparison import worker
from valendra_privacy_filter.datasets.prepare import prepare_records
from test_data import WindowTokenizer


CONFIG = Path(__file__).resolve().parents[1] / "configs/comparison-models.yaml"


def test_worker_writes_only_aggregate_report_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    class FakeAdapter:
        def detect(self, text):
            return [{"start": 0, "end": 3, "value": "Ana", "label": "private_person", "score": None}]

        def count_tokens(self, text):
            return len(text)

    monkeypatch.setattr(worker, "load_comparison_model", lambda *args, **kwargs: (FakeAdapter(), {"runtime": "fake"}))
    text = "Ana arrived."
    row = {"source_text": text, "source": "openpii", "language": "en", "region": "US",
           "privacy_mask": [{"start": 0, "end": 3, "value": "Ana", "label": "PERSON"}]}
    data, report = tmp_path / "data", tmp_path / "report.json"
    prepare_records({"test": [row]}, WindowTokenizer(), data, max_length=64, overlap=8)
    args = ["--model", "openai", "--data-dir", str(data), "--split", "test", "--config", str(CONFIG),
            "--device", "cpu", "--precision", "native", "--output", str(report), "--warmup", "0"]
    assert worker.main(args) == 0
    result = json.loads(report.read_text())
    assert result["metrics"]["untyped"]["f1"] == 1.0
    assert "Ana" not in report.read_text()
    assert "Ana" not in capsys.readouterr().out
    assert worker.main(args) == 1
    assert json.loads(report.read_text()) == result


def test_worker_failure_does_not_expose_input_or_write_partial_report(tmp_path, monkeypatch, capsys):
    class Broken:
        def detect(self, text):
            raise ValueError("sensitive text: " + text)

    monkeypatch.setattr(worker, "load_comparison_model", lambda *args, **kwargs: (Broken(), {}))
    row = {"source_text": "Ana", "source": "openpii", "language": "es", "region": "ES",
           "privacy_mask": [{"start": 0, "end": 3, "value": "Ana", "label": "PERSON"}]}
    data, report = tmp_path / "data", tmp_path / "report.json"
    prepare_records({"test": [row]}, WindowTokenizer(), data, max_length=64, overlap=8)
    args = ["--model", "openai", "--data-dir", str(data), "--split", "test", "--config", str(CONFIG),
            "--device", "cpu", "--precision", "native", "--output", str(report), "--warmup", "0"]
    assert worker.main(args) == 1
    assert not report.exists()
    assert "Ana" not in capsys.readouterr().err
