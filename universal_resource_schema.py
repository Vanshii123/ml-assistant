"""
UNIVERSAL RESOURCE SCHEMA
=========================
The single shape every piece of content gets normalized into, regardless of
where it came from: your internal course/PDF dataset, a college's notes/PYQ/
syllabus export, or a future second/third college's content.

Why this matters: your recommender should never need to know or care whether
a resource came from a CSV your college partner gave you, or from your own
dataset. It just queries NormalizedResource objects. Onboarding a new content
source later becomes "write one ingestion function," not "touch the engine."
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ResourceKind(str, Enum):
    """What kind of study material this is."""
    COURSE = "course"
    PDF_NOTES = "notes"
    PYQ = "pyq"                # previous year questions
    SYLLABUS = "syllabus"
    PRACTICAL = "practical"
    VIDEO = "video"
    BOOK = "book"


class NormalizedResource(BaseModel):
    """
    One resource, in a shape your recommender always understands —
    no matter which source it came from.
    """
    id: str
    title: str
    subject: str                         # e.g. "Digital Electronics"
    topic: str                           # e.g. "Boolean Algebra" — used for matching
    resource_kind: ResourceKind
    url: str
    source: str                          # e.g. "internal_dataset", "ipu_mait_site"
    college: Optional[str] = None        # e.g. "MAIT", "IPU" — None for generic/internal content
    unit: Optional[str] = None           # e.g. "Unit 3" — useful for syllabus-mapped content
    year: Optional[str] = None           # relevant for PYQs, e.g. "2023"
    search_blob: str = Field(default="")  # precomputed lowercase text used for matching

    def model_post_init(self, __context) -> None:
        # Build once at ingestion time so search doesn't recompute this every query.
        if not self.search_blob:
            self.search_blob = " ".join(
                str(x).lower() for x in [self.title, self.subject, self.topic, self.unit or ""]
            )