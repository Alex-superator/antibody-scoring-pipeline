from typing import Literal, Dict, Any
from app.scoring import toxicity, allergenicity, epitopes, stability, protparam


WEIGHTS = {
    "toxicity": 0.25,
    "allergenicity": 0.20,
    "bcell": 0.15,
    "tcell": 0.15,
    "instability": 0.25,
}


class ScorePipeline:
    def __call__(self, sequence: str, mode: Literal["light", "full"] = "light") -> Dict[str, Any]:
        seq = sequence.strip().upper()

        pt = protparam.calc_protparam(seq)
        bcell = epitopes.score_bcell_epitope(seq, mode=mode)
        tcell = epitopes.score_tcell_epitope(seq, mode=mode)
        tox = toxicity.score_toxicity(seq, mode=mode)
        allergen = allergenicity.score_allergenicity(seq, mode=mode)
        instability = stability.score_stability(seq, mode=mode)

        w = WEIGHTS
        risk = (
            w["toxicity"] * tox +
            w["allergenicity"] * allergen +
            w["bcell"] * bcell +
            w["tcell"] * tcell +
            w["instability"] * instability
        )

        return dict(
            sequence_len=len(seq),
            toxicity=tox,
            allergenicity=allergen,
            bcell_epitope=bcell,
            tcell_epitope=tcell,
            instability=instability,
            **pt,
            overall_risk=risk,
        )
