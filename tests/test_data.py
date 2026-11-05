import pytest

from valendra_privacy_filter.datasets.alignment import align_labels
from valendra_privacy_filter.datasets.records import normalize_record, template_signature
from valendra_privacy_filter.datasets.synthetic import generate_synthetic
from valendra_privacy_filter.datasets.prepare import prepare_records, stream_openpii, verify_dataset
from valendra_privacy_filter.labels import label_names


def span(text, value, label):
    start = text.index(value)
    return {"start": start, "end": start + len(value), "value": value, "label": label}


def test_validates_spans_merges_name_and_ignores_generic_date():
    text = "Ana García nació el 12/04/1990."
    row = {"source_text": text, "language": "es", "privacy_mask": [
        span(text, "Ana", "GIVENNAME"), span(text, "García", "SURNAME"),
        span(text, "12/04/1990", "DATE"),
    ]}
    record = normalize_record(row)
    assert [(s["label"], text[s["start"]:s["end"]]) for s in record["spans"]] == [("PERSON", "Ana García")]
    assert record["ignored"][0]["label"] == "DATE"
    broken = dict(row, privacy_mask=[{"start": 0, "end": 3, "value": "Eva", "label": "GIVENNAME"}])
    with pytest.raises(ValueError, match="span"):
        normalize_record(broken)


def test_unknown_type_is_never_silent_and_can_be_rejected():
    text = "Employee ID: abc"
    row = {"source_text": text, "privacy_mask": [span(text, "abc", "NEW_TYPE")]}
    assert normalize_record(row)["ignored"][0]["label"] == "NEW_TYPE"
    with pytest.raises(ValueError, match="NEW_TYPE"):
        normalize_record(row, unknown_policy="error")


def test_bioes_offsets_and_ignored_positions():
    text = "Ana García, nació 1990"
    spans = [{"start": 0, "end": 10, "label": "PERSON"}]
    ignored = [{"start": 18, "end": 22, "label": "DATE"}]
    offsets = [(0, 0), (0, 3), (3, 7), (7, 10), (10, 11), (12, 17), (18, 22), (0, 0)]
    names = label_names()
    labels = align_labels(text, offsets, spans, ignored, names)
    assert [names[x] if x != -100 else -100 for x in labels] == [
        -100, "B-PERSON", "I-PERSON", "E-PERSON", "O", "O", -100, -100
    ]


def test_unrepresentable_and_overlapping_spans_are_not_supervised():
    text = "abcd"
    offsets = [(0, 1), (1, 2), (2, 3), (3, 4)]
    spans = [{"start": 0, "end": 3, "label": "PERSON"}, {"start": 2, "end": 4, "label": "SECRET"}]
    assert align_labels(text, offsets, spans, [], label_names()) == [-100] * 4
    assert align_labels(text, [(0, 4)], [{"start": 1, "end": 3, "label": "PERSON"}], [], label_names()) == [-100]
    labels, diagnostics = align_labels(text, offsets, spans, [], label_names(), return_diagnostics=True)
    assert labels == [-100] * 4
    assert diagnostics["overlapping_entities"] == 2
    _, diagnostics = align_labels(text, [(0, 4)], [{"start": 1, "end": 3, "label": "PERSON"}], [], label_names(), return_diagnostics=True)
    assert diagnostics["unrepresentable_or_window_boundary"] == 1


def test_template_groups_and_synthetic_split_isolation():
    left = {"source_text": "Hola Ana.", "privacy_mask": [span("Hola Ana.", "Ana", "GIVENNAME")]}
    right = {"source_text": "Hola Eva.", "privacy_mask": [span("Hola Eva.", "Eva", "GIVENNAME")]}
    assert template_signature(left) == template_signature(right)
    first = generate_synthetic(seed=42, per_split=8)
    assert first == generate_synthetic(seed=42, per_split=8)
    groups = {split: {row["template_id"] for row in rows} for split, rows in first.items()}
    for split, ids in groups.items():
        assert all(not ids.intersection(other) for name, other in groups.items() if name != split)


def test_synthetic_covers_taxonomy_structured_data_and_real_overlap():
    rows = [row for split in generate_synthetic(seed=9, per_split=80).values() for row in split]
    labels = {item["label"] for row in rows for item in row["privacy_mask"]}
    assert labels == {"PERSON", "EMAIL", "PHONE", "ADDRESS", "GOVERNMENT_ID", "FINANCIAL_ID",
                      "SECRET", "IP_ADDRESS", "DATE_OF_BIRTH", "URL"}
    assert any('"api_key"' in row["source_text"] for row in rows)
    assert any(any(a["start"] < b["end"] and b["start"] < a["end"]
                   for i, a in enumerate(row["privacy_mask"]) for b in row["privacy_mask"][i + 1:]) for row in rows)


class WindowTokenizer:
    is_fast = True

    def __call__(self, text, max_length, stride, **kwargs):
        ids = [ord(c) for c in text]
        result = {"input_ids": [], "attention_mask": [], "offset_mapping": []}
        for start in range(0, len(text), max_length - stride):
            end = min(start + max_length, len(text))
            result["input_ids"].append(ids[start:end])
            result["attention_mask"].append([1] * (end - start))
            result["offset_mapping"].append([(i, i + 1) for i in range(start, end)])
            if end == len(text):
                break
        return result


def test_prepare_exports_parquet_and_no_group_leakage(tmp_path):
    import pyarrow.parquet as pq

    rows = generate_synthetic(seed=7, per_split=8)
    manifest = prepare_records(rows, WindowTokenizer(), tmp_path, max_length=40, overlap=10)
    assert set(manifest["splits"]) == {"train", "validation", "test", "challenge"}
    assert manifest["splits"]["train"]["sources"]["synthetic"] > 0
    assert manifest["splits"]["train"]["regions"]["ES"] > 0
    assert "alignment_issues_by_window" in manifest
    groups = []
    for split in rows:
        table = pq.read_table(tmp_path / f"{split}.parquet").to_pylist()
        assert table
        assert all(len(row["input_ids"]) <= 40 and len(row["labels"]) == len(row["input_ids"]) for row in table)
        groups.append({row["group_id"] for row in table})
    assert all(not left.intersection(right) for i, left in enumerate(groups) for right in groups[i + 1:])
    assert (tmp_path / "manifest.json").is_file()
    verify_dataset(tmp_path, ["test"])
    with (tmp_path / "test.parquet").open("ab") as stream:
        stream.write(b"unexpected")
    with pytest.raises(ValueError, match="checksum"):
        verify_dataset(tmp_path, ["test"])


def test_small_openpii_subset_prioritizes_spain_and_does_not_stop_at_english(monkeypatch):
    import datasets

    rows = ([{"language": "en", "region": "US"} for _ in range(5)]
            + [{"language": "es", "region": "MX"}]
            + [{"language": "es", "region": "ES"} for _ in range(5)])
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: iter(rows))
    result = list(stream_openpii(limit=4))
    assert sum(row["language"] == "es" and row["region"] == "ES" for row in result) == 3
    assert sum(row["language"] == "en" for row in result) == 1
    assert all(row["region"] != "MX" for row in result)
