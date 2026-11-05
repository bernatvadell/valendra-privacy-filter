"""The public package and executable use the project brand."""

import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess
import sys


def test_distribution_and_python_namespace_use_valendra_name():
    assert importlib.util.find_spec("valendra_privacy_filter") is not None
    package = importlib.metadata.distribution("valendra-privacy-filter")
    entry = next(e for e in package.entry_points if e.group == "console_scripts" and e.name == "valendra-privacy-filter")
    assert entry.value == "valendra_privacy_filter.cli:app"


def test_valendra_executable_exposes_dataset_workflows():
    executable = Path(sys.executable).parent / "valendra-privacy-filter"
    assert executable.is_file()
    result = subprocess.run([str(executable), "--help"], capture_output=True, text=True, check=True)
    assert "prepare-nemotron" in result.stdout
