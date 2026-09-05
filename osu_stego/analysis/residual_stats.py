"""Ručna implementacija deskriptivne statistike tajming rezidualâ.

Namerno ne koristimo scipy.stats za ove metrike (iako je dostupan u
requirements.txt za druge potrebe) -- ovde je prioritet potpuna kontrola i
transparentnost nad formulama koje direktno odgovaraju definicijama iz
predloga (Faza 1), a ne oslanjanje na podrazumevana ponašanja biblioteke
(npr. scipy.stats.kurtosis ima 'bias' i 'fisher' parametre čije difoltne
vrednosti nisu očigledne na prvi pogled).

Sve funkcije rade nad *populacijom* rezidualâ jednog replay-a (N u
imeniocu, ne N-1), pošto posmatramo kompletan skup rezidualâ tog fajla, ne
uzorak izvučen iz veće populacije.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualStatistics:
    """Deskriptivna statistika raspodele tajming rezidualâ jednog replay-a."""

    n: int
    mean: float
    std: float
    excess_kurtosis: float
    autocorrelation_lag1: float


def compute_mean(residuals: np.ndarray) -> float:
    """Aritmetička sredina rezidualâ."""
    return float(np.sum(residuals) / len(residuals))


def compute_std(residuals: np.ndarray, mean: float | None = None) -> float:
    """Standardna devijacija (populaciona, deljenje sa N).

    Parameters
    ----------
    residuals : np.ndarray
        Niz residual[i] vrednosti.
    mean : float, optional
        Prosek, ako je već izračunat (izbegava duplo računanje). Ako nije
        prosleđen, računa se interno.
    """
    if mean is None:
        mean = compute_mean(residuals)
    n = len(residuals)
    variance = np.sum((residuals - mean) ** 2) / n
    return float(np.sqrt(variance))


def compute_excess_kurtosis(residuals: np.ndarray, mean: float | None = None) -> float:
    """Fisher-ov excess kurtosis: (m4 / m2^2) - 3, populacioni momenti.

    m2 i m4 su drugi i četvrti centralni moment (deljeno sa N, ne N-1).
    Normalna raspodela ima excess_kurtosis = 0; pozitivna vrednost znači
    "teže repove" (leptokurtična raspodela) od normalne.

    Raises
    ------
    ValueError
        Ako je varijansa nula (svi rezidualy identični) -- kurtosis nije
        definisan u tom slučaju (deljenje nulom), radije eksplicitno
        signaliziramo grešku nego da vratimo NaN/inf bez objašnjenja.
    """
    if mean is None:
        mean = compute_mean(residuals)
    n = len(residuals)
    centered = residuals - mean
    m2 = np.sum(centered**2) / n
    m4 = np.sum(centered**4) / n

    if m2 == 0:
        raise ValueError(
            "Varijansa rezidualâ je nula -- kurtosis nije definisan "
            "(svi rezidualy su identični, verovatno greška u podacima)."
        )

    return float(m4 / (m2**2) - 3.0)


def compute_autocorrelation(
    residuals: np.ndarray, lag: int = 1, mean: float | None = None
) -> float:
    """Normalizovana autokorelacija rezidualâ na zadatom pomeraju (lag).

    Formula: sum_{i=0}^{n-lag-1} (x_i - mean)(x_{i+lag} - mean) / sum (x_i - mean)^2

    Vrednost je u opsegu [-1, 1] (analogno Pearson korelaciji, samo između
    niza i njegove pomerene verzije). Vrednost blizu 0 ukazuje na odsustvo
    periodičnog obrasca na tom lag-u -- relevantno za Fazu 4 (PN sequence
    može ostaviti slab periodičan trag).

    Raises
    ------
    ValueError
        Ako je lag >= n (nema dovoljno podataka za pomeraj) ili je
        varijansa nula.
    """
    n = len(residuals)
    if lag >= n:
        raise ValueError(f"lag ({lag}) mora biti manji od broja rezidualâ ({n}).")

    if mean is None:
        mean = compute_mean(residuals)
    centered = residuals - mean

    numerator = np.sum(centered[: n - lag] * centered[lag:])
    denominator = np.sum(centered**2)

    if denominator == 0:
        raise ValueError(
            "Varijansa rezidualâ je nula -- autokorelacija nije definisana."
        )

    return float(numerator / denominator)


def compute_residual_statistics(residuals: np.ndarray) -> ResidualStatistics:
    """Računa kompletan set deskriptivne statistike za dati niz rezidualâ."""
    mean = compute_mean(residuals)
    return ResidualStatistics(
        n=len(residuals),
        mean=mean,
        std=compute_std(residuals, mean=mean),
        excess_kurtosis=compute_excess_kurtosis(residuals, mean=mean),
        autocorrelation_lag1=compute_autocorrelation(residuals, lag=1, mean=mean),
    )
