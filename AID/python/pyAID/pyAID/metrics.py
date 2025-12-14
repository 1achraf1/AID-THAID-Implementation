from __future__ import annotations

def variance_reduction(parent_sse: float, left_sse: float, right_sse: float) -> float:
    """Réduction de SSE (within-node sum of squares) due à un split."""
    return float(parent_sse - (left_sse + right_sse))


def bss_from_sse(tss: float, sse: float) -> float:
    """Between Sum of Squares (BSS) = TSS - SSE."""
    return float(tss - sse)


def r_squared(tss: float, sse: float) -> float:
    """R² = 1 - SSE/TSS (peut être négatif si l'arbre est pire que la moyenne)."""
    if tss == 0.0:
        return 0.0
    return float(1.0 - (sse / tss))


def r_squared_clipped(tss: float, sse: float) -> float:
    """R² tronqué à [0, 1] (option d'affichage uniquement)."""
    return max(0.0, min(1.0, r_squared(tss, sse)))
