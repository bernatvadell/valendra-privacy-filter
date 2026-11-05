import json

from typer.testing import CliRunner

from valendra_privacy_filter.cli import app


def test_synthetic_generation_and_annotation_validation(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["generate-synthetic", "--output-dir", str(tmp_path), "--per-split", "8"])
    assert result.exit_code == 0, result.output
    assert all((tmp_path / f"{split}.jsonl").exists() for split in ("train", "validation", "test", "challenge"))
    result = runner.invoke(app, ["validate-annotations", "--input", str(tmp_path / "train.jsonl")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["valid"] == 8


def test_train_command_exposes_minimum_epoch_for_continuation():
    result = CliRunner().invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "min-early-stopping-epoch" in result.output
    assert "initialize-from-checkpoint" in result.output


def test_cli_exposes_separate_nvidia_preparation_and_mixed_dataset_commands():
    runner = CliRunner()
    for command, required in [("prepare-nemotron", ("baseline-dir", "output-dir", "test-limit")),
                              ("merge-datasets", ("baseline-dir", "nvidia-dir", "output-dir"))]:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
        for option in required:
            assert option in result.output


def test_cli_compares_three_aggregate_reports_without_overwriting(tmp_path):
    common = {"dataset_manifest_sha256": "data", "split": "test", "document_order_sha256": "docs",
              "comparison_config_sha256": "policy", "documents": 1, "device": "cpu", "hardware": "cpu",
              "limit": None, "common_categories": ["PERSON"], "scoring": "exact_character_spans_v1",
              "batch_size": 1, "warmup_documents": 0, "implementation_sha256": "code", "sample_seed": None}
    paths = {}
    for name in ("valendra", "openai", "gliner"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"metadata": {**common, "model": name}, "metrics": {"documents": 1}}))
        paths[name] = path
    output = tmp_path / "comparison.json"
    args = ["compare-baselines", "--valendra", str(paths["valendra"]), "--openai", str(paths["openai"]),
            "--gliner", str(paths["gliner"]), "--output", str(output)]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert set(json.loads(output.read_text())["models"]) == {"valendra", "openai", "gliner"}
    assert runner.invoke(app, args).exit_code != 0
