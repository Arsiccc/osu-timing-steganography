"""Konverzija poruke između bytes, bit-niza (0/1) i bipolarne (+1/-1) forme.

Bipolarna konverzija (0 -> -1, 1 -> +1) direktno odgovara formuli iz Faze 2
predloga: "binarna poruka se najpre konvertuje u biполarni niz vrednosti".
"""

from __future__ import annotations

import numpy as np


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Konvertuje bytes u niz bitova (0/1), MSB prvi, jedan bit po elementu."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Konvertuje niz bitova (0/1) nazad u bytes.

    Raises
    ------
    ValueError
        Ako broj bitova nije deljiv sa 8 (ne može se rekonstruisati ceo bajt).
    """
    if len(bits) % 8 != 0:
        raise ValueError(
            f"Broj bitova ({len(bits)}) mora biti deljiv sa 8 da bi se "
            "rekonstruisali celi bajtovi."
        )
    return np.packbits(bits.astype(np.uint8)).tobytes()


def bits_to_bipolar(bits: np.ndarray) -> np.ndarray:
    """Konvertuje bit-niz (0/1) u bipolarnu formu (-1/+1). 0 -> -1, 1 -> +1."""
    return np.where(bits == 0, -1, 1).astype(np.int8)


def bipolar_to_bits(bipolar: np.ndarray) -> np.ndarray:
    """Konvertuje bipolarni niz (-1/+1) nazad u bit-niz (0/1). Vrednost 0 se
    tretira kao bit=1 (granica odluke pri dekodovanju korelacije)."""
    return np.where(bipolar < 0, 0, 1).astype(np.uint8)
