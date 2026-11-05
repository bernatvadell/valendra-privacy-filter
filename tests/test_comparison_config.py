import json
from pathlib import Path

import pytest
import yaml

from valendra_privacy_filter.comparison.config import load_comparison_config
from valendra_privacy_filter.comparison.loaders import validate_precision


CONFIG = Path(__file__).resolve().parents[1] / "configs/comparison-models.yaml"


def test_pinned_configuration_keeps_ambiguous_openai_categories_unmapped():
    config = load_comparison_config(CONFIG)
    assert config["models"]["openai"]["label_mapping"]["private_date"] is None
    assert config["models"]["openai"]["label_mapping"]["account_number"] is None
    assert len(config["models"]["gliner"]["label_mapping"]) <= 25
    assert len(config["sha256"]) == 64


def test_configuration_rejects_broad_date_mapped_to_birth_date(tmp_path):
    document = yaml.safe_load(CONFIG.read_text())
    document["models"]["openai"]["label_mapping"]["private_date"] = "DATE_OF_BIRTH"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="ambiguous"):
        load_comparison_config(path)


def test_external_backends_require_explicit_supported_precision():
    validate_precision("openai", "native", "cpu")
    validate_precision("gliner", "fp32", "cpu")
    with pytest.raises(ValueError, match="native"):
        validate_precision("openai", "bf16", "cpu")
    with pytest.raises(ValueError, match="precision"):
        validate_precision("gliner", "native", "cpu")
