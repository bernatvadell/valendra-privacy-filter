"""Run the project comparison worker from any of its isolated environments."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valendra_privacy_filter.comparison.worker import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
