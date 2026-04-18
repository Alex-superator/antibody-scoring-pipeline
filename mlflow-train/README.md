# MLflow Training Scripts

This directory contains training scripts for SOTA models used in the FastAPI web app:

- `train_toxicity.py`        → StrucToxNet‑like toxicity classifier  
- `train_allergenicity.py`   → AllergenAI / pLM4Alg‑style allergenicity classifier  
- `train_epitopes.py`        → B‑ and T‑cell epitope predictors  
- `train_stability.py`       → Stability Oracle‑style stability predictor  

Each script:
- loads training data,
- computes features (matching `app/scoring/...`),
- trains a model,
- logs params/metrics/artifacts to MLflow,
- registers the model in MLflow Registry.

The registered model names (e.g., `StrucToxNet`, `AllergenAI`, `BcellPred`, `TcellPred`, `StabilityOracle`)
must match the `*_MODEL_NAME` environment variables used in the FastAPI app.
