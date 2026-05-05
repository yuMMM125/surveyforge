"""Pydantic schemas for Synthesizer output.

`SynthesizerOutput` is the structured contract between Researcher-Deep's
EvidenceItem rows and Writer's prose generation. Per spec § 1.3 P3, this
is THE differentiator from STORM: cross-paper structured extraction with
evidence-traceable comparison matrix and taxonomy.

8 fields total:
  - section_id: echoes input section
  - papers_cited: dedup'd paper_id list
  - claims: pass-through backwards-compat shape
  - paper_facts: per-paper extracted method/dataset/metric/result
  - comparison_matrix: cross-paper structured table; cells carry evidence_ids
  - taxonomy: per-section paper grouping; categories carry rationale_evidence_ids
  - cross_paper_synthesis: multi-paper narrative claims with evidence refs
  - coverage_gaps: must_find_evidence items not covered + papers dropped to budget

Every paper_id field uses the canonical `PaperId` Annotated type (prefix-form
enforced via shared validator). Every "value + evidence_ids" pair defends
against LLM fabricating content not grounded in EvidenceItems.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from litweave.schemas.paper_id import PaperId


class PaperFacts(BaseModel):
    """Per-paper extracted facts. Empty/unknown fields use empty string ""."""

    model_config = ConfigDict(frozen=True)

    paper_id: PaperId
    method: str = Field(description="Primary method/algorithm/architecture")
    dataset: str = Field(description="Primary dataset/benchmark used")
    metric: str = Field(description="Primary evaluation metric")
    result: str = Field(description="Headline result (e.g., 'F1 0.84 on RULER')")
    evidence_ids: list[str] = Field(min_length=1, description="EvidenceIds backing this extraction")


class MatrixCell(BaseModel):
    """One cell of a comparison_matrix row. Value MUST be traceable to evidence_ids."""

    model_config = ConfigDict(frozen=True)

    value: str = Field(description="Short cell text; '' if unknown")
    evidence_ids: list[str] = Field(min_length=1, description="EvidenceIds backing this cell value")


class MatrixRow(BaseModel):
    """One paper's row across all comparison_matrix dimensions."""

    model_config = ConfigDict(frozen=True)

    paper_id: PaperId
    cells: dict[str, MatrixCell] = Field(description="Maps dimension name → MatrixCell")


class ComparisonMatrix(BaseModel):
    """Cross-paper structured table. Dimensions are fixed."""

    model_config = ConfigDict(frozen=True)

    dimensions: list[str] = Field(default=["method", "dataset", "metric", "result"], min_length=1)
    rows: list[MatrixRow] = Field(default_factory=list)


class TaxonomyCategory(BaseModel):
    """One per-section taxonomy category. rationale_evidence_ids defends against
    ungrounded categorisation (per AD #3 host-side grounding)."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="Short category label, e.g. 'training-time methods'")
    description: str = Field(description="One-line category definition")
    paper_ids: list[PaperId] = Field(default_factory=list)
    rationale_evidence_ids: list[str] = Field(
        default_factory=list,
        description="EvidenceIds explaining WHY these papers cluster here. "
        "MUST be non-empty if paper_ids is non-empty.",
    )


class SectionTaxonomy(BaseModel):
    """Per-section paper grouping. Cross-section taxonomy mutual-exclusion is a
    later Critic-final concern."""

    model_config = ConfigDict(frozen=True)

    categories: list[TaxonomyCategory] = Field(default_factory=list)


class SynthesisClaim(BaseModel):
    """Multi-paper narrative claim with traceable evidence. The Synthesizer's
    core cross-paper synthesis output (per Open Decision #1; not Writer's
    responsibility)."""

    model_config = ConfigDict(frozen=True)

    claim_text: str = Field(min_length=1, description="1-3 sentence cross-paper claim")
    paper_ids: list[PaperId] = Field(min_length=2, description="≥ 2 papers supporting this claim")
    evidence_ids: list[str] = Field(min_length=2, description="≥ 1 evidence_id per supporting paper")


class CoverageGap(BaseModel):
    """A must_find_evidence item not satisfied by current evidence, OR a paper
    dropped during top_k rerank fallback. reason='missing_evidence' set by LLM;
    reason='budget_truncation' set by host after top_k rerank."""

    model_config = ConfigDict(frozen=True)

    must_find_evidence_item: str = Field(description="Text from section.must_find_evidence")
    reason: str = Field(description="'missing_evidence' or 'budget_truncation'")
    description: str = Field(default="", description="Optional context, e.g. dropped paper_ids")


class ClaimRef(BaseModel):
    """Backwards-compat claim shape — preserved so the existing stub Writer +
    downstream code don't break when the real Synthesizer lands."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    paper_id: PaperId
    claim: str
    confidence: float = Field(ge=0.0, le=1.0)


class SynthesizerOutput(BaseModel):
    """Synthesizer's structured output per section.

    Spec § 1.3 P3 differentiator: cross-paper schema-guided extraction.
    8 fields; every value-bearing field with cross-reference also carries
    evidence_ids for traceability.
    """

    model_config = ConfigDict(frozen=True)

    section_id: str = Field(min_length=1)
    papers_cited: list[PaperId] = Field(default_factory=list)
    claims: list[ClaimRef] = Field(default_factory=list)
    paper_facts: list[PaperFacts] = Field(default_factory=list)
    comparison_matrix: ComparisonMatrix = Field(default_factory=ComparisonMatrix)
    taxonomy: SectionTaxonomy = Field(default_factory=SectionTaxonomy)
    cross_paper_synthesis: list[SynthesisClaim] = Field(default_factory=list)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)
