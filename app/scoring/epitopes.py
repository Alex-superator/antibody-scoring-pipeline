import numpy as np
from typing import Literal
from app.mlflow_loader import registry


def sequence_to_bcell_features(seq: str, window: int = 15) -> np.ndarray:
    if len(seq) < window:
        return np.array([0.0] * 10)

    motifs = [seq[i:i+window] for i in range(len(seq) - window + 1)]
    p_patches = sum(m.count("P") for m in motifs)
    hydrophobic = sum(1 for m in motifs if m.count("F") + m.count("W") + m.count("Y") + m.count("L") + m.count("I") + m.count("V") > 0)
    flexibility = sum(m.count("P") for m in motifs)

    return np.array([
        len(motifs),
        p_patches / len(motifs),
        hydrophobic / len(motifs),
        flexibility / len(motifs),
        seq.count("K") / len(seq),
        seq.count("R") / len(seq),
        seq.count("E") / len(seq),
        seq.count("D") / len(seq),
        (seq.count("K") + seq.count("R")) / len(seq),
        (seq.count("E") + seq.count("D")) / len(seq),
    ])


def score_bcell_epitope(sequence: str, mode: Literal["light", "full"] = "light") -> float:
    if mode != "full":
        p_patches = sum(1 for i in range(len(sequence) - 3)
                        if sequence[i] == "P" and sequence[i + 3] == "P")
        flexibility = sequence.count("P")
        polar = sum(sequence.count(aa) for aa in "QNKDEST")
        tox = 0.2 + 0.5 * p_patches / len(sequence) + 0.3 * flexibility / len(sequence)
        return min(tox, 1.0)

    if not registry.bcell_model:
        raise RuntimeError("MLflow model 'bcell_model' not loaded!")

    features = sequence_to_bcell_features(sequence)
    X = np.array([features]).astype("float32")
    raw_pred = registry.bcell_model.predict(X)
    return float(raw_pred[0])


def sequence_to_tcell_features(
    seq: str,
    window: int = 9,
    hla_alleles: list[str] = ["HLA-A*02:01", "HLA-A*01:01"]
) -> np.ndarray:
    if len(seq) < window:
        return np.array([0.0] * (8 + len(hla_alleles)))

    motifs = [seq[i:i+window] for i in range(len(seq) - window + 1)]
    hydrophobic = sum(1 for m in motifs if m.count("F") + m.count("Y") + m.count("L") + m.count("V") + m.count("I") > 0)
    anchor = sum(1 for m in motifs if m[0] in "FYLIV" and m[-1] in "FYLIVA")

    base = [
        len(motifs),
        hydrophobic / len(motifs),
        anchor / len(motifs),
        seq.count("F") / len(seq),
        seq.count("Y") / len(seq),
        seq.count("L") / len(seq),
        seq.count("V") / len(seq),
        seq.count("I") / len(seq),
    ]
    base.extend([0.1] * len(hla_alleles))
    return np.array(base)


def score_tcell_epitope(sequence: str, mode: Literal["light", "full"] = "light") -> float:
    if mode != "full":
        p9 = 9
        motifs = [sequence[i:i+p9] for i in range(len(sequence) - p9 + 1)]
        hydrophobic = sum(1 for m in motifs if m.count("F") + m.count("Y") + m.count("L") + m.count("V") + m.count("I") > 0)
        anchor = sum(1 for m in motifs if m[0] in "FYLIV" and m[-1] in "FYLIVA")
        tox = 0.25 + 0.4 * hydrophobic / len(motifs) + 0.3 * anchor / len(motifs)
        return min(tox, 1.0)

    if not registry.tcell_model:
        raise RuntimeError("MLflow model 'tcell_model' not loaded!")

    features = sequence_to_tcell_features(sequence)
    X = np.array([features]).astype("float32")
    raw_pred = registry.tcell_model.predict(X)
    return float(raw_pred[0])
