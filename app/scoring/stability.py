import numpy as np
from typing import Dict, Callable
from app.scoring.protparam import calc_protparam
from app.mlflow_loader import registry


def sequence_to_stability_features(seq: str, use_struct_features: bool = False) -> np.ndarray:
    pt = calc_protparam(seq)
    return np.array([
        pt["molecular_weight"],
        pt["isoelectric_point"],
        pt["instability_index"],
        pt["gravy"],
        pt["aliphatic_index"],
        pt["aromaticity"],
        0.0,  # placeholder: delta‑pH‑stability
        0.0,  # placeholder: plddt_avg
        0.0,  # placeholder: rs_accessibility
    ])


def score_stability(sequence: str, mode: str = "light") -> float:
    pt = calc_protparam(sequence)
    instability_idx = pt["instability_index"]

    if mode != "full" or not registry.stability_model:
        if instability_idx < 40:
            return 0.1 + 0.05 * np.random.random()
        elif instability_idx < 60:
            return 0.4 + 0.1 * np.random.random()
        else:
            return 0.7 + 0.1 * np.random.random()

    features = sequence_to_stability_features(sequence, use_struct_features=False)
    X = np.array([features]).astype("float32")
    raw_pred = registry.stability_model.predict(X)
    raw_score = float(raw_pred[0])
    return min(max(raw_score, 0.0), 1.0)
