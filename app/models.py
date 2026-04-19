from typing import Literal, Dict, Any, Optional
from pydantic import BaseModel, Field


class ProteinRequest(BaseModel):
    sequence: str = Field(..., min_length=5, max_length=10_000)
    mode: Literal["light", "full"] = Field("light")


class RiskScore(BaseModel):
    toxicity: float = Field(ge=0.0, le=1.0)
    allergenicity: float = Field(ge=0.0, le=1.0)
    bcell_epitope: float = Field(ge=0.0, le=1.0)
    tcell_epitope: float = Field(ge=0.0, le=1.0)
    instability: float = Field(ge=0.0, le=1.0)
    overall_risk: float = Field(ge=0.0, le=1.0)


class PropertyScore(BaseModel):
    molecular_weight: float
    isoelectric_point: float
    aromaticity: float
    instability_index: float
    gravy: float
    charge: float
    flexibility: float
    amino_acid_composition: Dict[str, int]
    half_life: Optional[float] = None           # ← ИСПРАВЛЕНО
    aliphatic_index: Optional[float] = None     # ← ИСПРАВЛЕНО


class ScoreResponse(BaseModel):
    sequence_len: int
    risk: RiskScore
    properties: PropertyScore
    raw: Dict[str, Any]