from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AUTHORITY_STATEMENT = (
    "Informational report only; the deterministic LFighter pipeline retains all "
    "security decision authority."
)


class CitedStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=360)
    evidence_refs: list[str] = Field(min_length=1, max_length=4)


class Interpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=360)
    confidence: Literal["low", "moderate", "high"]
    evidence_refs: list[str] = Field(min_length=1, max_length=4)


class ForensicReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"]
    record_id: str = Field(min_length=1, max_length=160)
    report_title: str = Field(min_length=1, max_length=120)
    executive_summary: str = Field(min_length=1, max_length=600)
    facts: list[CitedStatement] = Field(min_length=3, max_length=4)
    interpretations: list[Interpretation] = Field(min_length=1, max_length=2)
    uncertainties: list[str] = Field(min_length=2, max_length=4)
    refusals: list[str] = Field(min_length=2, max_length=4)
    authority_statement: Literal[
        "Informational report only; the deterministic LFighter pipeline retains all security decision authority."
    ]
