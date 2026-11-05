import os
from pathlib import Path
import shlex
import subprocess
import sys


def test_launcher_records_environment_without_pip(tmp_path):
    root = tmp_path / "checkout"
    script = root / "scripts/run_mixed_warmstart.sh"
    script.parent.mkdir(parents=True)
    original = Path(__file__).resolve().parents[1] / "scripts/run_mixed_warmstart.sh"
    script.write_text(original.read_text())
    checkpoint = root / "source/checkpoint-1"
    checkpoint.mkdir(parents=True)
    for name in ("model.safetensors", "config.json", "trainer_state.json"):
        (checkpoint / name).touch()
    (checkpoint.parent / "run_identity.json").touch()
    python = root / "fake-python"
    python.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = '-c' ]; then\n"
        "  case \"$2\" in 'import shutil,torch;'*) exit 0;; esac\n"
        f"  exec {shlex.quote(sys.executable)} \"$@\"\n"
        "fi\n"
        "if [ \"$1\" = '-m' ] && [ \"$2\" = 'valendra_privacy_filter.cli' ]; then exit 0; fi\n"
        "exit 72\n"
    )
    python.chmod(0o755)
    result = subprocess.run(["bash", str(script), str(checkpoint)],
                            env={"PATH": os.environ["PATH"], "PRIVACY_FILTER_PYTHON": str(python)},
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "torch==" in (root / "experiments/runs/mixed-warmstart-environment.txt").read_text()
