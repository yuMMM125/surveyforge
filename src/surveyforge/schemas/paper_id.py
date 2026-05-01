"""Shared validator for `paper_id` fields across schemas.

Per Decision 0.3: paper_id must use prefix form `arxiv:` / `s2:` / `web:` to
encode the source and avoid cross-source id collisions. This validator is
the contract enforcement point — Pydantic v2 `AfterValidator` lets each
schema annotate its `paper_id` field with `PaperId` to inherit the check.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator

VALID_PAPER_ID_PREFIXES = ("arxiv:", "s2:", "web:")


def validate_paper_id_prefix(v: str) -> str:
    if not any(v.startswith(p) for p in VALID_PAPER_ID_PREFIXES):
        raise ValueError(
            f"paper_id must start with one of {VALID_PAPER_ID_PREFIXES}, got {v!r}"
        )
    return v


PaperId = Annotated[str, AfterValidator(validate_paper_id_prefix)]
