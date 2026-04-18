from Bio.SeqUtils.ProtParam import ProteinAnalysis
from typing import Dict


def calc_protparam(seq: str) -> Dict:
    p = ProteinAnalysis(seq)
    return {
        "molecular_weight": p.molecular_weight(),
        "isoelectric_point": p.isoelectric_point(),
        "aromaticity": p.aromaticity(),
        "instability_index": p.instability_index(),
        "aliphatic_index": p.aliphatic_index(),
        "gravy": p.gravy(),
        "half_life": p.half_life(),
        "charge": p.charge_at_pH(7.4),
        "flexibility": sum(p.flexibility()) / len(p.flexibility()) if p.flexibility() else 0.0,
        "amino_acid_composition": dict(p.count_amino_acids()),
    }
