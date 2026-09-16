import json
from functools import lru_cache

from src.config import get_settings


@lru_cache
def load_test_set() -> dict[str, dict]:
    """Load the bundled test documents and their hand-written answers, keyed by document id."""
    try:
        labels_path = get_settings().test_set_dir / "labels.json"
        with open(labels_path, encoding="utf-8") as f:
            labels = json.load(f)
    except FileNotFoundError:
        return {}

    mapping = {}
    for doc_id, info in labels.items():
        doc_path = get_settings().test_set_dir / f"{doc_id}.txt"
        try:
            with open(doc_path, encoding="utf-8") as f:
                doc_text = f.read()
        except FileNotFoundError:
            continue
        mapping[doc_id] = {"doc_text": doc_text, **info}
    return mapping
