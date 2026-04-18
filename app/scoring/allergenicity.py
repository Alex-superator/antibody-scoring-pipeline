import numpy as np
from app.mlflow_loader import registry


def sequence_to_allergen_features(seq: str) -> np.ndarray:
    return np.array([
        seq.count("P") / len(seq),
        seq.count("I") / len(seq),
        seq.count("L") / len(seq),
        seq.count("V") / len(seq),
        seq.count("F") / len(seq),
        seq.count("Y") / len(seq),
        seq.count("W") / len(seq),
        seq.count("Q") / len(seq),
        seq.count("N") / len(seq),
        len(seq),
    ])


def score_allergenicity(sequence: str, mode: str = "light") -> float:
    if mode != "full":
        patchy = sum(sequence.count(aa) for aa in "PILVFYW")
        tox = 0.15 + 0.05 * len(sequence) / 1000 + 0.7 * patchy / len(sequence)
        return min(tox, 1.0)

    if not registry.allergenicity_model:
        raise RuntimeError("MLflow model 'allergenicity_model' not loaded!")

    features = sequence_to_allergen_features(sequence)
    X = np.array([features]).astype("float32")
    raw_pred = registry.allergenicity_model.predict(X)
    return float(raw_pred[0])
