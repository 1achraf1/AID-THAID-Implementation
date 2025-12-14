"""pyAID: Automatic Interaction Detection (AID) implementation.

This package provides an educational yet efficient implementation of the
original AID algorithm (Morgan & Sonquist, 1963) focused on regression tasks.
"""

from .aid import AIDRegressor

__all__ = ["AIDRegressor"]
