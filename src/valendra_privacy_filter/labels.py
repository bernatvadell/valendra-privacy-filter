"""Stable BIOES taxonomy."""

DEFAULT_CATEGORIES = (
    "PERSON", "EMAIL", "PHONE", "ADDRESS", "GOVERNMENT_ID",
    "FINANCIAL_ID", "SECRET", "IP_ADDRESS", "DATE_OF_BIRTH", "URL",
)


def label_names(categories=DEFAULT_CATEGORIES):
    categories = tuple(categories)
    if len(set(categories)) != len(categories) or any(not c or "-" in c for c in categories):
        raise ValueError("Categories must be distinct, nonempty and contain no hyphens")
    return ["O"] + [f"{prefix}-{category}" for category in categories for prefix in "BIES"]
