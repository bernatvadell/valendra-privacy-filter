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


def test_cli_exposes_separate_nvidia_preparation_and_mixed_dataset_commands():
    runner = CliRunner()
    for command, required in [("prepare-nemotron", ("baseline-dir", "output-dir", "test-limit")),
                              ("merge-datasets", ("baseline-dir", "nvidia-dir", "output-dir"))]:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, result.output
        for option in required:
            assert option in result.output
