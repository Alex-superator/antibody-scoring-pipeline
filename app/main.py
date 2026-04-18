# app/main.py

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.mlflow_loader import registry
from app.models import ProteinRequest, ScoreResponse
from app.pipeline import ScorePipeline


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


pipeline = ScorePipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        registry.load_models(dict(os.environ))
    except Exception as e:
        # Приложение стартует даже если MLflow временно недоступен,
        # но full-mode тогда будет падать с понятной ошибкой.
        app.state.model_load_error = str(e)
    else:
        app.state.model_load_error = None

    yield

    # Здесь можно освобождать ресурсы, если будут GPU/сессии/соединения.
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

    risk = {
        "toxicity": raw["toxicity"],
        "allergenicity": raw["allergenicity"],
        "bcell_epitope": raw["bcell_epitope"],
        "tcell_epitope": raw["tcell_epitope"],
        "instability": raw["instability"],
        "overall_risk": raw["overall_risk"],
    }

    properties = {
        "molecular_weight": raw["molecular_weight"],
        "isoelectric_point": raw["isoelectric_point"],
        "aromaticity": raw["aromaticity"],
        "instability_index": raw["instability_index"],
        "aliphatic_index": raw["aliphatic_index"],
        "gravy": raw["gravy"],
        "half_life": raw["half_life"],
        "charge": raw["charge"],
        "flexibility": raw["flexibility"],
        "amino_acid_composition": raw["amino_acid_composition"],
    }

    return ScoreResponse(
        sequence_len=raw["sequence_len"],
        risk=risk,
        properties=properties,
        raw=raw,
    )
