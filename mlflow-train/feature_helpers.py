# mlflow-train/feature_helpers.py
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# общие feature‑функции, повторяющиеся в app/scoring/*
def sequence_to_features_toxicity(seq: str) -> np.ndarray:
    # same logic as in app/scoring/toxicity
    # ... (details)
    pass

def sequence_to_features_allergenicity(seq: str) -> np.ndarray:
    # same as in app/scoring/allergenicity
    # ... (details)
    pass

def sequence_to_bcell_features(seq: str) -> np.ndarray:
    pass

def sequence_to_tcell_features(seq: str) -> np.ndarray:
    pass

def sequence_to_stability_features(seq: str) -> np.ndarray:
    # same as in app/scoring/stability
    pass
