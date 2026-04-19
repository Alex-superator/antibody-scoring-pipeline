# app/main.py

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.mlflow_loader import registry
from app.models import ProteinRequest, ScoreResponse, RiskScore, PropertyScore
from app.pipeline import ScorePipeline


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


pipeline = ScorePipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        registry.load_models(dict(os.environ))
    except Exception as e:
        app.state.model_load_error = str(e)
    else:
        app.state.model_load_error = None

    yield

    registry.toxicity_model = None
    registry.allergenicity_model = None
    registry.bcell_model = None
    registry.tcell_model = None
    registry.stability_model = None


app = FastAPI(
    title="Antibody Safety & Property Scoring",
    version="1.0.0",
    description="Unified FastAPI pipeline for toxicity, allergenicity, epitopes, stability and ProtParam properties.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return index_file.read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": {
            "toxicity": registry.toxicity_model is not None,
            "allergenicity": registry.allergenicity_model is not None,
            "bcell": registry.bcell_model is not None,
            "tcell": registry.tcell_model is not None,
            "stability": registry.stability_model is not None,
        },
        "model_load_error": getattr(app.state, "model_load_error", None),
    }


@app.post("/score", response_model=ScoreResponse)
async def score_protein(req: ProteinRequest):
    seq = req.sequence.strip().upper()

    if not seq.isalpha():
        raise HTTPException(status_code=400, detail="Sequence must contain only letters.")
    if len(seq) < 5 or len(seq) > 10000:
        raise HTTPException(status_code=400, detail="Sequence length must be between 5 and 10000.")

    try:
        raw = pipeline(seq, mode=req.mode)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # все свойства в корне raw
    properties = PropertyScore(
        molecular_weight=float(raw.get("molecular_weight", 0.0)),
        isoelectric_point=float(raw.get("isoelectric_point", 0.0)),
        aromaticity=float(raw.get("aromaticity", 0.0)),
        instability_index=float(raw.get("instability_index", 0.0)),
        aliphatic_index=raw.get("aliphatic_index"),  # None OK
        gravy=float(raw.get("gravy", 0.0)),
        half_life=raw.get("half_life"),              # None OK
        charge=float(raw.get("charge", 0.0)),
        flexibility=float(raw.get("flexibility", 0.0)),
        amino_acid_composition=dict(raw.get("amino_acid_composition", {})),
    )

    risk = RiskScore(
        toxicity=float(raw.get("toxicity", 0.0)),
        allergenicity=float(raw.get("allergenicity", 0.0)),
        bcell_epitope=float(raw.get("bcell_epitope", 0.0)),
        tcell_epitope=float(raw.get("tcell_epitope", 0.0)),
        instability=float(raw.get("instability", 0.0)),
        overall_risk=float(raw.get("overall_risk", 0.0)),
    )

    return ScoreResponse(
        sequence_len=int(raw.get("sequence_len", len(seq))),
        risk=risk,
        properties=properties,
        raw=raw,
    )