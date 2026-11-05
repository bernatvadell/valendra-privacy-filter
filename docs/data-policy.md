# Data and annotation policy

## Source labels

| OpenPII original | Project label |
| --- | --- |
| `GIVENNAME`, `SURNAME` | `PERSON`; whitespace-adjacent components are joined |
| `EMAIL` | `EMAIL` |
| `TELEPHONENUM` | `PHONE` |
| `STREET`, `BUILDINGNUM`, `ZIPCODE`, `CITY` | `ADDRESS` components |
| `IDCARDNUM`, `DRIVERLICENSENUM`, `TAXNUM`, `SOCIALNUM`, `PASSPORTNUM` | `GOVERNMENT_ID` |
| `CREDITCARDNUMBER` | `FINANCIAL_ID` |
| `DATE`, `TITLE`, `AGE`, `GENDER`, `SEX`, or any unknown type | Ignored (`-100`) and counted; `--unknown-policy error` fails instead |

There is no automatic rule that turns a generic date into a birth date. Explicitly labeled synthetic `DATE_OF_BIRTH` examples and NVIDIA's explicitly annotated `date_of_birth` spans are supervised. The ten category names in `configs/taxonomy.yaml` produce 41 labels including `O`. Narrower configured taxonomies mask categories omitted from the list. Offsets are Unicode code-point indices, not UTF-8 byte positions.

An annotation is valid only if its bounds are in range and its declared value is exactly the corresponding text slice. Conflicting duplicate texts abort preparation. Nested/overlapping labels cannot be represented by one BIOES sequence: affected tokens are ignored, original annotations remain available, and issues are counted by window. Partially truncated or non-representable token boundaries are also ignored and counted. Other invalid records are quarantined by reason count. Unannotated source text is supervised as `O` except where an ignored annotation intersects it; source annotations can still be incomplete, and the resulting false-negative supervision cannot be detected automatically.

Train, validation and test assignment uses a seeded hash of the text template after replacing original annotated values by their labels. Synthetic template families are explicitly assigned to exactly one split, with challenge-only templates. Exact duplicate text is removed globally. The method does not prove that semantic paraphrases or different renderings of the same source template cannot cross splits; manual and real-world evaluation remain important.

## Synthetic positives and negatives

Faker and Python's seeded random generator create fictional Spanish and English contexts. Fictional names, email addresses, phone numbers, identity numbers, bank identifiers, addresses, private paths, URLs, IPs and credentials receive labels when used as contextual personal/private values. `URL` includes private URL-like paths. DNI/NIE use a checksum; other identifiers such as the `ES00` IBAN are intentionally non-operational. The card number is a published test number. `SYNTHETIC_TEST_KEY_*`, `sk_test_SYNTHETIC_*` and `NOT_A_REAL_PASSWORD_*` are explicitly fictional but **positive `SECRET` labels** when presented as credentials. No real credentials are used.

Explicit placeholders (`${EMAIL}`, `YOUR_API_KEY`), documentation addresses such as `192.0.2.1`, an email shown as a generic example, company names, generic dates, ordinary numbers, technical hashes and cultural/bibliographic references are included as negatives. A fictional email address may instead be labeled when the surrounding text presents it as a person's contact. These are annotation policy choices, not runtime regex rules or guarantees about the model. The generator includes a private URL with a nested fake key to exercise the overlap policy.

Calibration estimates a **token-level** temperature on validation logits. Inference applies it before the geometric-mean span score; no claim of span-level probabilistic calibration follows without separate validation. Keep test and challenge splits out of calibration and model selection.

For NVIDIA Nemotron-PII, the source label mapping is in `configs/nemotron-labels.yaml`; see [its separate assessment](nemotron-pii-assessment.md). The isolated mixed preparation checks exact text, canonical templates, source groups and selected normalized high-signal values across splits and the reserved NVIDIA test. These heuristic checks do not prove complete persona-level independence. The NVIDIA dataset is not included in the current run.
