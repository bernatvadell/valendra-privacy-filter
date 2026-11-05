"""Load and validate a frozen, explicit cross-model benchmark policy."""

import hashlib
import math
from pathlib import Path
import re

import yaml

from valendra_privacy_filter.labels import DEFAULT_CATEGORIES


def load_comparison_config(path):
    raw = Path(path).read_bytes()
    document = yaml.safe_load(raw)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Invalid comparison configuration version")
    categories, models = document.get("common_categories"), document.get("models")
    if (not isinstance(categories, list) or not categories or len(set(categories)) != len(categories)
            or not set(categories).issubset(DEFAULT_CATEGORIES)):
        raise ValueError("Invalid common categories")
    if not isinstance(models, dict) or set(models) != {"valendra", "openai", "gliner"}:
        raise ValueError("Comparison must configure exactly the three requested models")
    for name, settings in models.items():
        if not isinstance(settings, dict):
            raise ValueError("Invalid model settings")
        mapping = settings.get("label_mapping")
        if not isinstance(mapping, dict) or not mapping or any(not isinstance(key, str) or not key for key in mapping):
            raise ValueError("Invalid native label mapping")
        if any(value is not None and value not in DEFAULT_CATEGORIES for value in mapping.values()):
            raise ValueError("Invalid mapped category")
        if name != "valendra":
            if not isinstance(settings.get("model_id"), str) or not settings["model_id"]:
                raise ValueError("External model id is required")
            revision = settings.get("model_revision")
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("External model revision must be a full commit hash")
    if any(key not in models["openai"]["label_mapping"] or models["openai"]["label_mapping"][key] is not None
           for key in ("private_date", "account_number")):
        raise ValueError("OpenAI's ambiguous date and account labels must remain unmapped")
    gliner = models["gliner"]
    if len(gliner["label_mapping"]) > 25:
        raise ValueError("GLiNER label prompts exceed the configured model limit")
    threshold = gliner.get("threshold")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid GLiNER threshold")
    overlap = gliner.get("overlap_words")
    if type(overlap) is not int or overlap < 0:
        raise ValueError("Invalid GLiNER overlap")
    document["sha256"] = hashlib.sha256(raw).hexdigest()
    return document
