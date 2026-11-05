"""Validate OpenPII original character spans and normalize its taxonomy."""

import hashlib
import re

from valendra_privacy_filter.labels import DEFAULT_CATEGORIES

MAPPING = {
    "GIVENNAME": "PERSON", "SURNAME": "PERSON", "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE", "STREET": "ADDRESS", "BUILDINGNUM": "ADDRESS",
    "ZIPCODE": "ADDRESS", "CITY": "ADDRESS", "IDCARDNUM": "GOVERNMENT_ID",
    "DRIVERLICENSENUM": "GOVERNMENT_ID", "TAXNUM": "GOVERNMENT_ID",
    "SOCIALNUM": "GOVERNMENT_ID", "PASSPORTNUM": "GOVERNMENT_ID",
    "CREDITCARDNUMBER": "FINANCIAL_ID", **{name: name for name in DEFAULT_CATEGORIES},
}


def _checked_annotations(row):
    text = row["source_text"]
    if not isinstance(text, str):
        raise ValueError("source_text must be a string")
    result = []
    for original in row.get("privacy_mask", []):
        item = dict(original)
        start, end = item.get("start"), item.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            raise ValueError("invalid span bounds")
        if text[start:end] != item.get("value"):
            raise ValueError("span value does not match source text")
        if not isinstance(item.get("label"), str) or not item["label"]:
            raise ValueError("span label missing")
        result.append(item)
    return sorted(result, key=lambda item: (item["start"], item["end"], item["label"]))


def normalize_record(row, unknown_policy="ignore"):
    if unknown_policy not in {"ignore", "error"}:
        raise ValueError("unknown_policy must be ignore or error")
    annotations = _checked_annotations(row)
    text = row["source_text"]
    spans, ignored = [], []
    for original in annotations:
        mapped = original["normalized_label"] if "normalized_label" in original else MAPPING.get(original["label"])
        if mapped is not None and mapped not in DEFAULT_CATEGORIES:
            raise ValueError("unsupported normalized label")
        item = {"start": original["start"], "end": original["end"], "label": mapped or original["label"]}
        if mapped is None:
            if unknown_policy == "error":
                raise ValueError(f"unsupported label: {original['label']}")
            ignored.append(item)
        elif (mapped == "PERSON" and spans and spans[-1]["label"] == "PERSON"
              and spans[-1]["end"] <= item["start"]
              and text[spans[-1]["end"]:item["start"]].isspace()):
            spans[-1]["end"] = item["end"]
        else:
            spans.append(item)
    return {
        "text": text, "spans": spans, "ignored": ignored, "raw_annotations": annotations,
        "language": row.get("language", "unknown"), "region": row.get("region", ""),
        "source": row.get("source", "openpii"), "uid": str(row.get("uid", "")),
    }


def template_signature(row):
    """Hash normalized original text with all original annotated values removed."""
    text = row["source_text"]
    cursor, parts = 0, []
    for item in _checked_annotations(row):
        if item["start"] < cursor:
            continue
        parts.extend((text[cursor:item["start"]], f"<{item['label']}>") )
        cursor = item["end"]
    parts.append(text[cursor:])
    normalized = re.sub(r"\s+", " ", "".join(parts)).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
