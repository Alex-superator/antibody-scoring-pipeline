# app/scoring/protparam.py — ПОЛНАЯ ЗАЩИТА ОТ ВСЕХ ОШИБОК

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from typing import Dict, Any


def calc_protparam(seq: str) -> Dict[str, Any]:
    """
    Полностью безопасный расчёт свойств белка.
    Защищает от division by zero, ValueError, TypeError для КОРОТКИХ последовательностей.
    """
    if not seq or len(seq) < 1:
        return {
            "molecular_weight": 0.0,
            "isoelectric_point": 7.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "gravy": 0.0,
            "half_life": 0.0,
            "aliphatic_index": 0.0,
            "charge": 0.0,
            "flexibility": 0.0,
            "amino_acid_composition": {},
        }

    try:
        p = ProteinAnalysis(seq)
    except Exception:
        return {
            "molecular_weight": 0.0,
            "isoelectric_point": 7.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "gravy": 0.0,
            "half_life": 0.0,
            "aliphatic_index": 0.0,
            "charge": 0.0,
            "flexibility": 0.0,
            "amino_acid_composition": {},
        }

    result = {}

    # безопасные вызовы — НЕ падают НИ на чём
    params_to_try = [
        ("molecular_weight", p.molecular_weight),
        ("isoelectric_point", p.isoelectric_point),
        ("aromaticity", p.aromaticity),
        ("instability_index", p.instability_index),
        ("gravy", p.gravy),
    ]

    for param, func in params_to_try:
        try:
            result[param] = float(func())
        except (ZeroDivisionError, ValueError, TypeError, OverflowError):
            result[param] = 0.0

    # charge_at_pH(7.4)
    try:
        result["charge"] = float(p.charge_at_pH(7.4))
    except (ZeroDivisionError, ValueError, TypeError, OverflowError):
        result["charge"] = 0.0

    # half_life
    try:
        hl_result = p.half_life()
        result["half_life"] = float(hl_result) if isinstance(hl_result, (int, float)) else None
    except:
        result["half_life"] = None

    # aliphatic_index
    try:
        ai_result = p.aliphatic_index()
        result["aliphatic_index"] = float(ai_result) if isinstance(ai_result, (int, float)) else None
    except:
        result["aliphatic_index"] = None

    # flexibility — с защитой от пустого списка
    try:
        flex_values = p.flexibility()
        if flex_values and len(flex_values) > 0:
            result["flexibility"] = float(sum(flex_values) / len(flex_values))
        else:
            result["flexibility"] = 0.0
    except:
        result["flexibility"] = 0.0

    # amino_acid_composition — всегда работает
    try:
        result["amino_acid_composition"] = dict(p.count_amino_acids())
    except:
        result["amino_acid_composition"] = {}

    print(f"DEBUG protparam: seq_len={len(seq)}, flexibility_len={len(flex_values) if 'flex_values' in locals() else 0}")
    
    return result