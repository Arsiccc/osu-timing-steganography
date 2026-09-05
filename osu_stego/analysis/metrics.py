"""Merenje BER (Bit Error Rate) -- primarna metrika za Encoder/Decoder par."""

from __future__ import annotations

import numpy as np


def bit_error_rate(original: np.ndarray, decoded: np.ndarray) -> float:
    """Računa BER kao udeo bitova koji se ne poklapaju.

    Radi podjednako sa bipolarnom (-1/+1) i binarnom (0/1) reprezentacijom,
    pod uslovom da su original i decoded u ISTOJ reprezentaciji.

    Raises
    ------
    ValueError
        Ako nizovi nisu iste dužine.
    """
    if len(original) != len(decoded):
        raise ValueError(
            f"original ({len(original)}) i decoded ({len(decoded)}) "
            "moraju biti iste dužine."
        )
    if len(original) == 0:
        return 0.0

    mismatches = np.sum(original != decoded)
    return float(mismatches / len(original))
