import json
import hashlib
import os
import sys
from pathlib import Path

import pytest

from valendra_privacy_filter.comparison.suite import run_suite
from valendra_privacy_filter.datasets.prepare import prepare_records
from test_data import WindowTokenizer


def test_suite_runs_models_sequentially_and_compares_identical_reports(tmp_path, monkeypatch):
    data, checkpoint, output = (tmp_path / name for name in ("data", "model", "reports"))
    row = {"source_text": "Ana", "source": "openpii", "language": "es", "region": "ES",
           "privacy_mask": [{"start": 0, "end": 3, "label": "PERSON", "value": "Ana"}]}
    prepare_records({"test": [row]}, WindowTokenizer(), data, max_length=64, overlap=8)
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"synthetic")
    calls = []
    config = Path(__file__).resolve().parents[1] / "configs/comparison-models.yaml"
    common = {"dataset_manifest_sha256": hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest(),
              "split": "test", "document_order_sha256": "docs",
              "comparison_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
              "implementation_sha256": "code", "documents": 1,
              "device": "cpu", "hardware": "cpu", "limit": 5,
              "common_categories": sorted(["PERSON", "EMAIL", "PHONE", "ADDRESS", "URL", "SECRET"]),
              "scoring": "exact_character_spans_v1", "batch_size": 1, "warmup_documents": 1, "sample_seed": 42}

    def fake_run(command, check, env):
        assert command[0] == os.path.abspath(sys.executable)
        assert "AWS_BEARER_TOKEN_BEDROCK" not in env
        assert "OPENAI_API_KEY" not in env
        assert command[command.index("--sample-seed") + 1] == "42"
        name = command[command.index("--model") + 1]
        calls.append(name)
        path = Path(command[command.index("--output") + 1])
        path.write_text(json.dumps({"metadata": {**common, "model": name}, "metrics": {"documents": 1}}))

    monkeypatch.setattr("valendra_privacy_filter.comparison.suite.subprocess.run", fake_run)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fictional-test-value")
    monkeypatch.setenv("OPENAI_API_KEY", "fictional-test-value")
    result = run_suite(checkpoint=checkpoint, data_dir=data, split="test", output_dir=output, config=config,
                       python_paths={name: sys.executable for name in ("valendra", "openai", "gliner")},
                       device="cpu", precision="fp32", warmup=1, limit=5, sample_seed=42)
    assert calls == ["valendra", "openai", "gliner"]
    assert set(result["models"]) == {"valendra", "openai", "gliner"}
    assert (output / "comparison.json").is_file()
    with pytest.raises(ValueError, match="exist"):
        run_suite(checkpoint=checkpoint, data_dir=data, split="test", output_dir=output, config=config,
                  python_paths={name: sys.executable for name in ("valendra", "openai", "gliner")},
                  device="cpu", precision="fp32", warmup=1, limit=5)
