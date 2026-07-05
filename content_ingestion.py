"""
CONTENT INGESTION + UNIVERSAL STORE
====================================
Turns raw content (CSV export, JSON, or DB query rows — from anyone: your
dataset, a college partner, a future second college) into NormalizedResource
objects, and gives you one searchable store that your recommender can query
regardless of source.

Usage once you have his data:
    store = ContentStore()
    store.add_many(ingest_from_csv("mait_notes_export.csv", college_name="MAIT"))
    results = store.search("boolean algebra", top_n=5)
"""

import csv
import json
import difflib
from pathlib import Path
from typing import Optional

from universal_resource_schema import NormalizedResource, ResourceKind


# ---------------------------------------------------------------------------
# INGESTION: raw formats -> NormalizedResource
# ---------------------------------------------------------------------------

DEFAULT_CSV_MAPPING = {
    "id": "id",
    "title": "title",
    "subject": "subject",
    "topic": "topic",
    "resource_kind": "type",
    "url": "url",
    "unit": "unit",
    "year": "year",
}


def _row_to_resource(row: dict, college_name: str, source_tag: str, mapping: dict) -> Optional[NormalizedResource]:
    """Map one raw row (from CSV/JSON/DB) into a NormalizedResource using a column mapping."""
    try:
        kind_raw = str(row.get(mapping["resource_kind"], "notes")).strip().lower()
        # Be forgiving about how the source labels resource types.
        kind_lookup = {
            "note": ResourceKind.PDF_NOTES, "notes": ResourceKind.PDF_NOTES, "pdf": ResourceKind.PDF_NOTES,
            "pyq": ResourceKind.PYQ, "previous year question": ResourceKind.PYQ, "paper": ResourceKind.PYQ,
            "syllabus": ResourceKind.SYLLABUS,
            "practical": ResourceKind.PRACTICAL, "lab": ResourceKind.PRACTICAL,
            "video": ResourceKind.VIDEO,
            "course": ResourceKind.COURSE,
            "book": ResourceKind.BOOK,
        }
        resource_kind = kind_lookup.get(kind_raw, ResourceKind.PDF_NOTES)

        return NormalizedResource(
            id=str(row.get(mapping["id"], "")).strip() or f"{source_tag}_{hash(str(row))}",
            title=str(row.get(mapping["title"], "")).strip() or "Untitled resource",
            subject=str(row.get(mapping["subject"], "")).strip() or "General",
            topic=str(row.get(mapping["topic"], "")).strip() or str(row.get(mapping["title"], "")).strip(),
            resource_kind=resource_kind,
            url=str(row.get(mapping["url"], "")).strip(),
            source=source_tag,
            college=college_name,
            unit=str(row.get(mapping.get("unit", ""), "")).strip() or None,
            year=str(row.get(mapping.get("year", ""), "")).strip() or None,
        )
    except Exception as e:
        print(f"⚠️ Skipped a row during ingestion ({source_tag}): {e}")
        return None


def ingest_from_csv(filepath: str, college_name: str, mapping: dict = None) -> list[NormalizedResource]:
    """
    Ingest a CSV export (e.g. from the college's admin panel / spreadsheet).
    `mapping` lets you match whatever column names their export actually uses —
    you don't need them to reformat anything on their end.

    Example if his CSV has different column names:
        mapping = {
            "id": "sr_no", "title": "resource_name", "subject": "subject_name",
            "topic": "chapter", "resource_kind": "category", "url": "file_link",
            "unit": "unit_no", "year": "exam_year"
        }
    """
    mapping = mapping or DEFAULT_CSV_MAPPING
    resources = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            res = _row_to_resource(row, college_name, f"{college_name.lower()}_csv", mapping)
            if res:
                resources.append(res)

    print(f"✅ Ingested {len(resources)} resources from {filepath} ({college_name})")
    return resources


def ingest_from_json(filepath: str, college_name: str, mapping: dict = None) -> list[NormalizedResource]:
    """Ingest a JSON export — same idea as CSV, just a different raw format."""
    mapping = mapping or DEFAULT_CSV_MAPPING
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {filepath}")

    with open(path, encoding="utf-8") as f:
        raw_items = json.load(f)

    resources = []
    for row in raw_items:
        res = _row_to_resource(row, college_name, f"{college_name.lower()}_json", mapping)
        if res:
            resources.append(res)

    print(f"✅ Ingested {len(resources)} resources from {filepath} ({college_name})")
    return resources


def ingest_from_db_rows(rows: list[dict], college_name: str, mapping: dict = None) -> list[NormalizedResource]:
    """
    Ingest rows straight from a DB query (e.g. cursor.fetchall() mapped to dicts,
    or an ORM query's .values()). Same mapping logic — use this if he gives you
    DB read access instead of a file export.
    """
    mapping = mapping or DEFAULT_CSV_MAPPING
    resources = []
    for row in rows:
        res = _row_to_resource(row, college_name, f"{college_name.lower()}_db", mapping)
        if res:
            resources.append(res)

    print(f"✅ Ingested {len(resources)} resources from DB rows ({college_name})")
    return resources


# ---------------------------------------------------------------------------
# STORE: one searchable place for all normalized content, from any source
# ---------------------------------------------------------------------------

class ContentStore:
    """
    Holds NormalizedResource objects from every source (your dataset, his
    college, a future second college) and searches across all of them the
    same way. This is what your recommender should query going forward.
    """

    def __init__(self):
        self._items: list[NormalizedResource] = []

    def add_many(self, resources: list[NormalizedResource]) -> None:
        self._items.extend(resources)

    def add(self, resource: NormalizedResource) -> None:
        self._items.append(resource)

    def count(self) -> int:
        return len(self._items)

    def search(
        self,
        query: str,
        resource_kind: Optional[ResourceKind] = None,
        college: Optional[str] = None,
        top_n: int = 5,
    ) -> list[NormalizedResource]:
        """
        Simple, fast keyword + fuzzy scoring — no embeddings needed for this
        scale. Filters by resource_kind/college first if given, then ranks by
        relevance to `query`.
        """
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())

        candidates = self._items
        if resource_kind:
            candidates = [r for r in candidates if r.resource_kind == resource_kind]
        if college:
            candidates = [r for r in candidates if r.college == college]

        scored = []
        for r in candidates:
            word_overlap = len(query_words & set(r.search_blob.split()))
            fuzzy_score = difflib.SequenceMatcher(None, query_lower, r.search_blob).ratio()
            score = word_overlap * 2 + fuzzy_score  # word overlap weighted higher than fuzzy similarity
            if score > 0.15:  # cutoff to avoid returning irrelevant noise
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_n]]

    def save_cache(self, filepath: str = "content_store_cache.json") -> None:
        """Persist to disk so you don't re-run ingestion every server restart."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in self._items], f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(self._items)} resources to {filepath}")

    def load_cache(self, filepath: str = "content_store_cache.json") -> None:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️ No cache found at {filepath} — skipping load")
            return
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self._items = [NormalizedResource(**item) for item in raw]
        print(f"✅ Loaded {len(self._items)} resources from {filepath}")


# ---------------------------------------------------------------------------
# EXAMPLE — how you'll actually wire this into your app
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    store = ContentStore()

    # Once he gives you an export, this is the entire integration step:
    # store.add_many(ingest_from_csv("mait_content_export.csv", college_name="MAIT"))

    # For now, a couple of fake rows to prove it works end-to-end:
    sample_rows = [
        {"id": "1", "title": "Boolean Algebra Notes", "subject": "Digital Electronics",
         "topic": "Boolean Algebra", "type": "notes", "url": "https://example.com/n1.pdf", "unit": "Unit 2"},
        {"id": "2", "title": "DE PYQ 2023", "subject": "Digital Electronics",
         "topic": "Boolean Algebra", "type": "pyq", "url": "https://example.com/pyq1.pdf", "year": "2023"},
    ]
    store.add_many(ingest_from_db_rows(sample_rows, college_name="MAIT"))

    results = store.search("boolean algebra")
    for r in results:
        print(f"- [{r.resource_kind}] {r.title} ({r.college})")