import numpy as np
from app.mlflow_loader import registry


def sequence_to_features(seq: str) -> np.ndarray:
    return np.array([
        seq.count("C") / len(seq),
        seq.count("M") / len(seq),
        seq.count("R") / len(seq),
        seq.count("P") / len(seq),
        seq.count("D") / len(seq),
        len(seq),
        (seq.count("K") + seq.count("R")) / len(seq),
        (seq.count("E") + seq.count("D")) / len(seq),
    ])


def score_toxicity(sequence: str, mode: str = "light") -> float:
    if mode != "full":
        tox = 0.1 + 0.05 * len(sequence) / 1000
        tox += 0.3 * sequence.count("C") / len(sequence)
        tox += 0.2 * sequence.count("M") / len(sequence)
        tox += 0.2 * sequence.count("R") / len(sequence)
        return min(tox, 1.0)

    if not registry.toxicity_model:
        raise RuntimeError("MLflow model 'toxicity_model' not loaded!")

    row = sequence_to_features(sequence)
    X = np.array([row]).astype("float32")
    raw_pred = registry.toxicity_model.predict(X)
    return float(raw_pred[0])
