import json
from pathlib import Path

import pytest

from valendra_privacy_filter.datasets.nemotron import (AnnotationError, adapt_nemotron_record,
                                                  load_label_policy, prepare_nemotron_records, stream_nemotron)
from valendra_privacy_filter.datasets.records import normalize_record
from valendra_privacy_filter.datasets.mixed import (BaselineIndex, NemotronFilter, canonical_signature,
                                                merge_prepared_datasets, audit_prepared_isolation,
                                                identity_fingerprints)
from valendra_privacy_filter.datasets.prepare import prepare_records, verify_dataset, nemotron_group_split
from test_data import WindowTokenizer


POLICY_PATH = Path(__file__).resolve().parents[1] / "configs/nemotron-labels.yaml"


def annotation(text, value, label, raw_value=None):
    start = text.index(value)
    return {"start": start, "end": start + len(value),
            "text": value if raw_value is None else raw_value, "label": label}


def test_converts_original_spans_and_explicitly_ignores_unsupported_types():
    text = "Ana García, born 1990-01-01, at Acme, uses 42 and key SYNTHETIC-KEY."
    original = [annotation(text, "Ana", "first_name"), annotation(text, "García", "last_name"),
                annotation(text, "1990-01-01", "date_of_birth"), annotation(text, "Acme", "company_name"),
                annotation(text, "42", "age", raw_value=42), annotation(text, "SYNTHETIC-KEY", "api_key")]
    row = {"uid": "fake-id", "locale": "intl", "text": text, "spans": repr(original)}
    converted = adapt_nemotron_record(row, load_label_policy(POLICY_PATH))
    assert converted["language"] == "en"
    assert converted["region"] == "INTL"
    assert [s["label"] for s in converted["privacy_mask"]] == [s["label"] for s in original]
    normalized = normalize_record(converted)
    assert [(s["label"], text[s["start"]:s["end"]]) for s in normalized["spans"]] == [
        ("PERSON", "Ana García"), ("DATE_OF_BIRTH", "1990-01-01"), ("SECRET", "SYNTHETIC-KEY")]
    assert [s["label"] for s in normalized["ignored"]] == ["company_name", "age"]


def test_generic_date_and_unknown_types_are_never_silently_o():
    text = "Meeting 2024-01-01, reference ABC-12."
    row = {"text": text, "locale": "us", "uid": "1", "spans": [
        annotation(text, "2024-01-01", "date"), annotation(text, "ABC-12", "new_vendor_type")]}
    record = normalize_record(adapt_nemotron_record(row, load_label_policy(POLICY_PATH)))
    assert record["spans"] == []
    assert [s["label"] for s in record["ignored"]] == ["date", "new_vendor_type"]


@pytest.mark.parametrize("bad_spans,reason", [
    ("__import__('os').system('false')", "invalid_spans_literal"),
    ("[", "invalid_spans_literal"),
    (repr([{"start": 0, "end": 2, "text": "XX", "label": "first_name"}]), "span_value_mismatch"),
    (repr([{"start": 0, "end": 100, "text": "A", "label": "first_name"}]), "invalid_span_bounds"),
    (repr([{"start": True, "end": 2, "text": "n", "label": "first_name"}]), "invalid_span_bounds"),
])
def test_rejects_corrupt_annotations_without_leaking_text(bad_spans, reason):
    with pytest.raises(AnnotationError, match=reason) as error:
        adapt_nemotron_record({"text": "Ana", "uid": "u", "locale": "us", "spans": bad_spans},
                              load_label_policy(POLICY_PATH))
    assert "Ana" not in str(error.value)


def test_numeric_value_must_exactly_match_original_text():
    text = "Amount 0042"
    row = {"text": text, "uid": "u", "locale": "us", "spans": [annotation(text, "0042", "age", 42)]}
    with pytest.raises(AnnotationError, match="span_value_mismatch"):
        adapt_nemotron_record(row, load_label_policy(POLICY_PATH))


def test_label_policy_rejects_implicit_or_invalid_mappings(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text("unknown: ignore\nmapping:\n  email: WRONG\n", encoding="utf-8")
    with pytest.raises(ValueError, match="category"):
        load_label_policy(path)
    path.write_text("mapping:\n  email: EMAIL\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_label_policy(path)


def test_default_nemotron_policy_is_available_in_installed_package():
    from importlib.resources import files

    bundled = files("valendra_privacy_filter").joinpath("resources/nemotron-labels.yaml")
    assert bundled.read_text(encoding="utf-8") == POLICY_PATH.read_text(encoding="utf-8")
    assert load_label_policy(None) == load_label_policy(POLICY_PATH)


def test_nvidia_preparation_uses_bundled_policy_outside_checkout(tmp_path, monkeypatch):
    import hashlib

    baseline, output = tmp_path / "baseline", tmp_path / "nvidia"
    prepare_records({}, WindowTokenizer(), baseline, max_length=64, overlap=8)
    monkeypatch.chdir(tmp_path)
    manifest = prepare_nemotron_records(baseline, output, WindowTokenizer(), None,
                                        train_rows=[], test_rows=[])
    assert manifest["source_parameters"]["label_policy_sha256"] == hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()


def test_canonical_template_matches_distinct_source_label_names():
    left = {"text": "Hi Ana.", "spans": [{"start": 3, "end": 6, "label": "PERSON"}], "ignored": []}
    right = {"text": "Hi Eva.", "spans": [{"start": 3, "end": 6, "label": "PERSON"}], "ignored": []}
    assert canonical_signature(**left) == canonical_signature(**right)


def test_baseline_and_nvidia_holdout_templates_never_enter_training(tmp_path):
    baseline = tmp_path / "baseline"
    text = "Contact Ana now."
    row = {"source_text": text, "language": "es", "region": "ES", "source": "openpii",
           "privacy_mask": [{"start": 8, "end": 11, "value": "Ana", "label": "GIVENNAME"}]}
    prepare_records({"train": [row]}, WindowTokenizer(), baseline, max_length=64, overlap=8)
    index = BaselineIndex.from_directory(baseline)
    assert len(index.document_ids) == 1
    policy = load_label_policy(POLICY_PATH)
    filtered = NemotronFilter(index, policy)

    def nvidia(text, value, label="first_name"):
        return {"text": text, "uid": value, "locale": "us", "spans": [annotation(text, value, label)]}

    test_rows = [nvidia("Contact Eva now.", "Eva"), nvidia("Born 1990-01-01!", "1990-01-01", "date_of_birth")]
    accepted_test = list(filtered.rows("test", test_rows))
    assert len(accepted_test) == 1
    assert filtered.stats["test"]["baseline_template_overlap"] == 1
    train_rows = [nvidia("Born 1980-02-03!", "1980-02-03", "date_of_birth"),
                  nvidia("Phone +1-555-0100.", "+1-555-0100", "phone_number")]
    accepted_train = list(filtered.rows("train", train_rows))
    assert len(accepted_train) == 1
    assert filtered.stats["train"]["nvidia_test_template_overlap"] == 1
    assert accepted_train[0]["source"] == "nemotron"


def test_train_filter_cannot_run_before_reserving_test(tmp_path):
    baseline = tmp_path / "baseline"
    prepare_records({}, WindowTokenizer(), baseline, max_length=64, overlap=8)
    filtered = NemotronFilter(BaselineIndex.from_directory(baseline), load_label_policy(POLICY_PATH))
    with pytest.raises(ValueError, match="test"):
        list(filtered.rows("train", []))


def test_prepare_nvidia_preserves_original_test_and_baseline_integrity(tmp_path):
    import hashlib
    import pyarrow.parquet as pq

    baseline, output = tmp_path / "baseline", tmp_path / "nvidia"
    text = "Contact Ana now."
    row = {"source_text": text, "language": "es", "region": "ES", "source": "openpii",
           "privacy_mask": [{"start": 8, "end": 11, "value": "Ana", "label": "GIVENNAME"}]}
    prepare_records({"train": [row]}, WindowTokenizer(), baseline, max_length=64, overlap=8)
    original_hash = hashlib.sha256((baseline / "train.parquet").read_bytes()).hexdigest()

    def nvidia(text, value, label):
        return {"text": text, "uid": value, "locale": "us", "spans": [annotation(text, value, label)]}

    test_rows = [nvidia("Contact Eva now.", "Eva", "first_name"),
                 nvidia("Born 1990-01-01!", "1990-01-01", "date_of_birth")]
    train_rows = [nvidia("Born 1980-02-03!", "1980-02-03", "date_of_birth"),
                  nvidia("Phone +1-555-0100.", "+1-555-0100", "phone_number"),
                  nvidia("Meeting 2024-01-01.", "2024-01-01", "date")]
    result = prepare_nemotron_records(baseline, output, WindowTokenizer(), POLICY_PATH,
                                      train_rows=train_rows, test_rows=test_rows)
    assert result["sources"][0]["id"] == "nvidia/Nemotron-PII"
    assert result["source_parameters"]["filter_statistics"]["test"]["baseline_template_overlap"] == 1
    assert result["source_parameters"]["filter_statistics"]["train"]["nvidia_test_template_overlap"] == 1
    assert result["splits"]["test"]["documents"] == 1
    assert result["postwrite_isolation_audit"]["passed"]
    assert sum(result["splits"][s]["documents"] for s in ("train", "validation")) == 2
    gold = pq.read_table(output / "test.parquet").to_pylist()
    assert gold[0]["spans"][0]["label"] == "DATE_OF_BIRTH"
    assert gold[0]["raw_annotations"][0]["label"] == "date_of_birth"
    assert hashlib.sha256((baseline / "train.parquet").read_bytes()).hexdigest() == original_hash
    verify_dataset(output)
    with pytest.raises(ValueError, match="empty"):
        prepare_nemotron_records(baseline, output, WindowTokenizer(), POLICY_PATH,
                                 train_rows=[], test_rows=[])


def test_nvidia_download_is_revision_pinned_and_respects_limit(tmp_path, monkeypatch):
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"uid": "1", "locale": "us", "text": "A", "spans": "[]"},
        {"uid": "2", "locale": "intl", "text": "B", "spans": "[]"},
    ]), path)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    rows = list(stream_nemotron("train", limit=1))
    assert len(rows) == 1 and rows[0]["uid"] == "1"
    assert calls[0]["revision"] == "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed"
    assert calls[0]["filename"] == "data/train-00000-of-00001.parquet"
    with pytest.raises(ValueError, match="positive"):
        list(stream_nemotron("test", limit=0))


def test_mixed_dataset_excludes_cross_split_templates_and_keeps_nvidia_test_separate(tmp_path):
    import hashlib
    import pyarrow.parquet as pq

    baseline, nvidia, mixed = tmp_path / "baseline", tmp_path / "nvidia", tmp_path / "mixed"

    def baseline_row(text, value, label):
        x = annotation(text, value, label)
        return {"source_text": text, "source": "openpii", "language": "es", "region": "ES",
                "privacy_mask": [{"start": x["start"], "end": x["end"], "value": value, "label": label}]}

    # Same normalized template but different original source labels in separate baseline splits.
    base_rows = {"train": [baseline_row("Hello Ana.", "Ana", "GIVENNAME")],
                 "validation": [baseline_row("Hello Eva.", "Eva", "PERSON")],
                 "test": [baseline_row("Ticket for Luis!", "Luis", "PERSON")]}
    prepare_records(base_rows, WindowTokenizer(), baseline, max_length=64, overlap=8)
    before = hashlib.sha256((baseline / "train.parquet").read_bytes()).hexdigest()

    def nvidia_row(text, value, label):
        return {"text": text, "uid": value, "locale": "us", "spans": [annotation(text, value, label)]}

    prepare_nemotron_records(baseline, nvidia, WindowTokenizer(), POLICY_PATH,
                             train_rows=[nvidia_row("Account TEST-123!", "TEST-123", "account_number")],
                             test_rows=[nvidia_row("Born 1990-01-01!", "1990-01-01", "date_of_birth")],
                             train_is_complete=True, test_is_complete=True)
    manifest = merge_prepared_datasets(baseline, nvidia, mixed)
    assert manifest["leakage_checks"]["baseline_cross_split_templates"] == 1
    assert manifest["leakage_checks"]["baseline_documents_excluded"] == 2
    assert manifest["sources"][-1]["id"] == "nvidia/Nemotron-PII"
    assert manifest["splits"]["test"]["documents"] == 1
    assert all(row["source"] != "nemotron" for row in pq.read_table(mixed / "test.parquet").to_pylist())
    assert pq.read_table(nvidia / "test.parquet").to_pylist()[0]["source"] == "nemotron"
    assert hashlib.sha256((baseline / "train.parquet").read_bytes()).hexdigest() == before
    verify_dataset(mixed)
    audit = audit_prepared_isolation(mixed, nvidia)
    assert audit["passed"]
    assert audit["holdout_overlap_documents"] == 0
    assert not audit_prepared_isolation(baseline)["passed"]


def test_mixed_dataset_refuses_partial_nvidia_holdout_without_explicit_smoke_option(tmp_path):
    baseline, nvidia = tmp_path / "baseline", tmp_path / "nvidia"
    prepare_records({}, WindowTokenizer(), baseline, max_length=64, overlap=8)
    prepare_nemotron_records(baseline, nvidia, WindowTokenizer(), POLICY_PATH, train_rows=[], test_rows=[])
    with pytest.raises(ValueError, match="sample"):
        merge_prepared_datasets(baseline, nvidia, tmp_path / "mixed")
    manifest = merge_prepared_datasets(baseline, nvidia, tmp_path / "smoke", allow_sampled_holdout=True)
    assert manifest["source_parameters"]["sampled_nvidia_holdout"] is True


def test_mixed_dataset_excludes_cross_split_synthetic_family_even_when_templates_differ(tmp_path):
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq

    baseline, nvidia, mixed = tmp_path / "baseline", tmp_path / "nvidia", tmp_path / "mixed"
    rows = {
        "train": [{"source_text": "First family example.", "privacy_mask": [], "source": "synthetic",
                   "template_id": "family-a", "language": "en", "region": "US"}],
        "validation": [{"source_text": "Another related variant.", "privacy_mask": [], "source": "synthetic",
                        "template_id": "family-b", "language": "en", "region": "US"}],
    }
    prepare_records(rows, WindowTokenizer(), baseline, max_length=64, overlap=8)
    path = baseline / "validation.parquet"
    table = pq.read_table(path)
    replacement = pa.array(["synthetic:family-a"] * len(table), type=pa.string())
    table = table.set_column(table.schema.get_field_index("group_id"), "group_id", replacement)
    pq.write_table(table, path)
    manifest_path = baseline / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["validation"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    prepare_nemotron_records(baseline, nvidia, WindowTokenizer(), POLICY_PATH,
                             train_rows=[], test_rows=[], test_is_complete=False)
    result = merge_prepared_datasets(baseline, nvidia, mixed, allow_sampled_holdout=True)
    assert result["leakage_checks"]["baseline_cross_split_groups"] == 1
    assert result["leakage_checks"]["baseline_documents_excluded"] == 2


def test_mixed_dataset_excludes_entire_families_linked_by_cross_split_template(tmp_path):
    baseline, nvidia, mixed = tmp_path / "baseline", tmp_path / "nvidia", tmp_path / "mixed"

    def synthetic(text, person, family):
        x = annotation(text, person, "PERSON")
        return {"source_text": text, "source": "synthetic", "template_id": family,
                "language": "es", "region": "ES", "privacy_mask": [
                    {"start": x["start"], "end": x["end"], "label": "PERSON", "value": person}]}

    rows = {"train": [synthetic("Hello Ana.", "Ana", "a"), synthetic("Saludos Juan.", "Juan", "a")],
            "validation": [synthetic("Hello Eva.", "Eva", "b"), synthetic("Adios Luz.", "Luz", "b")]}
    prepare_records(rows, WindowTokenizer(), baseline, max_length=64, overlap=8)
    prepare_nemotron_records(baseline, nvidia, WindowTokenizer(), POLICY_PATH,
                             train_rows=[], test_rows=[], test_is_complete=False)
    result = merge_prepared_datasets(baseline, nvidia, mixed, allow_sampled_holdout=True)
    assert result["leakage_checks"]["baseline_cross_split_templates"] == 1
    assert result["leakage_checks"]["baseline_documents_excluded"] == 4


def test_identity_fingerprints_match_formatted_ids_without_exposing_values():
    text_a, text_b = "+34 600 123 456", "34600123456"
    a = identity_fingerprints(text_a, [{"start": 0, "end": len(text_a), "label": "PHONE"}])
    b = identity_fingerprints(text_b, [{"start": 0, "end": len(text_b), "label": "PHONE"}])
    assert a == b and len(a) == 1
    assert text_a not in repr(a) and text_b not in repr(b)


def test_identity_fingerprints_include_specific_addresses_but_not_generic_cities():
    text_a, text_b = "123 Example Street, Springfield", "123   example street, springfield"
    first = identity_fingerprints(text_a, [{"start": 0, "end": len(text_a), "label": "ADDRESS"}])
    second = identity_fingerprints(text_b, [{"start": 0, "end": len(text_b), "label": "ADDRESS"}])
    assert first == second and len(first) == 1
    assert not identity_fingerprints("Madrid", [{"start": 0, "end": 6, "label": "ADDRESS"}])


def test_nvidia_rejects_identifiers_shared_with_baseline_or_reserved_test(tmp_path):
    baseline = tmp_path / "baseline"
    original = "Reach ana@example.com."
    base = {"source_text": original, "language": "es", "region": "ES", "source": "openpii",
            "privacy_mask": [{**annotation(original, "ana@example.com", "EMAIL"), "value": "ana@example.com"}]}
    prepare_records({"train": [base]}, WindowTokenizer(), baseline, max_length=80, overlap=8)
    filtered = NemotronFilter(BaselineIndex.from_directory(baseline), load_label_policy(POLICY_PATH))

    def nvidia(text, value, label):
        return {"text": text, "uid": value, "locale": "us", "spans": [annotation(text, value, label)]}

    test_rows = [nvidia("Account belongs to ana@example.com!", "ana@example.com", "email"),
                 nvidia("Secret is SYNTHETIC-ABC-12345.", "SYNTHETIC-ABC-12345", "api_key")]
    assert len(list(filtered.rows("test", test_rows))) == 1
    assert filtered.stats["test"]["baseline_identity_overlap"] == 1
    train_rows = [nvidia("Header key=SYNTHETIC-ABC-12345", "SYNTHETIC-ABC-12345", "api_key")]
    assert list(filtered.rows("train", train_rows)) == []
    assert filtered.stats["train"]["nvidia_test_identity_overlap"] == 1


def test_nvidia_training_and_validation_never_share_identity_values(tmp_path):
    baseline = tmp_path / "baseline"
    prepare_records({}, WindowTokenizer(), baseline, max_length=128, overlap=8)
    filtered = NemotronFilter(BaselineIndex.from_directory(baseline), load_label_policy(POLICY_PATH), seed=42)
    list(filtered.rows("test", []))
    policy = load_label_policy(POLICY_PATH)
    value = "nobody-unique@example.com"
    candidates = {}
    for i in range(100):
        text = f"Message variant {i}: {value}."
        raw = {"text": text, "uid": str(i), "locale": "us", "spans": [annotation(text, value, "email")]}
        normalized = normalize_record(adapt_nemotron_record(raw, policy))
        sig = canonical_signature(normalized["text"], normalized["spans"], normalized["ignored"])
        candidates.setdefault(nemotron_group_split("nemotron:" + sig, 42), raw)
        if len(candidates) == 2:
            break
    assert set(candidates) == {"train", "validation"}
    accepted = list(filtered.rows("train", [candidates["train"], candidates["validation"]]))
    assert len(accepted) == 1
    assert filtered.stats["train"]["nvidia_cross_split_identity_overlap"] == 1


def test_mixed_dataset_drops_baseline_holdout_groups_with_shared_identifiers(tmp_path):
    baseline, nvidia, mixed = tmp_path / "baseline", tmp_path / "nvidia", tmp_path / "mixed"

    def row(text):
        x = annotation(text, "ana@example.com", "EMAIL")
        return {"source_text": text, "language": "en", "region": "US", "source": "openpii",
                "privacy_mask": [{"start": x["start"], "end": x["end"], "label": "EMAIL", "value": x["text"]}]}

    prepare_records({"train": [row("Reach ana@example.com.")],
                     "validation": [row("Contact ana@example.com please.")]},
                    WindowTokenizer(), baseline, max_length=80, overlap=8)
    prepare_nemotron_records(baseline, nvidia, WindowTokenizer(), POLICY_PATH,
                             train_rows=[], test_rows=[], test_is_complete=False)
    result = merge_prepared_datasets(baseline, nvidia, mixed, allow_sampled_holdout=True)
    assert result["leakage_checks"]["baseline_cross_split_identity_values"] == 1
    assert result["splits"]["train"]["documents"] == 1
    assert result["splits"]["validation"]["documents"] == 0
    assert audit_prepared_isolation(mixed)["passed"]


def test_shared_identifier_removes_only_conflicting_document_not_its_entire_holdout_family(tmp_path):
    baseline, nvidia, mixed = tmp_path / "baseline", tmp_path / "nvidia", tmp_path / "mixed"

    def synthetic(text, email, family):
        x = annotation(text, email, "EMAIL")
        return {"source_text": text, "source": "synthetic", "template_id": family,
                "language": "en", "region": "US", "privacy_mask": [
                    {"start": x["start"], "end": x["end"], "label": "EMAIL", "value": email}]}

    rows = {
        "train": [synthetic("Email ana@example.com.", "ana@example.com", "training")],
        "validation": [synthetic("Contact ana@example.com now.", "ana@example.com", "heldout"),
                       synthetic("Contact eva@example.com now.", "eva@example.com", "heldout")],
    }
    prepare_records(rows, WindowTokenizer(), baseline, max_length=80, overlap=8)
    prepare_nemotron_records(baseline, nvidia, WindowTokenizer(), POLICY_PATH,
                             train_rows=[], test_rows=[], test_is_complete=False)
    result = merge_prepared_datasets(baseline, nvidia, mixed, allow_sampled_holdout=True)
    assert result["splits"]["train"]["documents"] == 1
    assert result["splits"]["validation"]["documents"] == 1
    assert result["leakage_checks"]["baseline_identity_documents_excluded_by_split"]["validation"] == 1
    assert audit_prepared_isolation(mixed)["passed"]
