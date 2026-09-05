"""Spread Spectrum encoder: ugradnja poruke u tajming rezidualove.

Formula iz predloga (Faza 2):
    residual_stego[i] = residual[i] + alpha * pn[i] * bit[i / N]

gde je bit[i/N] BIPOLARNA vrednost (+1/-1) odgovarajućeg bita poruke za
frejm i (svaki bit se prostire preko N uzastopnih frejmova/rezidualâ).
"""

from __future__ import annotations

import numpy as np

from .pn_sequence import generate_pn_sequence


def embed_message_indexed(
    note_residuals: np.ndarray,
    message_bipolar: np.ndarray,
    key: str | int,
    alpha: float,
    n_frames_per_bit: int,
    quantize: bool = True,
) -> np.ndarray:
    """Ugrađuje poruku u note-indeksirani residual niz (fiksne dužine, sa
    NaN na pozicijama promašenih nota -- videti match_keypresses_to_notes_indexed).

    Ključna razlika od embed_message: PN sekvenca je indeksirana po POZICIJI
    NOTE (ne po sažetoj listi uparenih vrednosti), pa ostaje poravnata sa
    dekoderom čak i ako se skup promašaja promeni nakon pisanja/čitanja
    stvarnog .osr fajla. Pozicije sa NaN residualom (promašaj -- nema frejma
    za izmenu) ostaju NaN u izlazu; te pozicije jednostavno ne doprinose
    korelacionoj sumi pri dekodovanju za blok kom pripadaju.

    Parameters
    ----------
    note_residuals : np.ndarray
        Niz dužine = broj nota, sa np.nan na promašenim pozicijama
        (izlaz iz match_keypresses_to_notes_indexed).

    Ostali parametri isti kao kod embed_message.

    Returns
    -------
    np.ndarray
        Note-indeksirani stego-residual niz, iste dužine kao note_residuals.
        NaN pozicije ostaju NaN.

    Raises
    ------
    ValueError
        Ako required_positions > len(note_residuals) (nedovoljan kapacitet).
    """
    required_positions = len(message_bipolar) * n_frames_per_bit
    if required_positions > len(note_residuals):
        raise ValueError(
            f"Poruka zahteva {required_positions} note-pozicija "
            f"({len(message_bipolar)} bita x {n_frames_per_bit} po bitu), "
            f"a dostupno je samo {len(note_residuals)} nota."
        )

    pn = generate_pn_sequence(key, required_positions).astype(np.float64)
    bit_per_position = np.repeat(message_bipolar, n_frames_per_bit).astype(np.float64)

    stego = note_residuals.copy()
    segment = note_residuals[:required_positions]
    valid_mask = ~np.isnan(segment)

    perturbed = segment.copy()
    perturbed[valid_mask] = segment[valid_mask] + (
        alpha * pn[valid_mask] * bit_per_position[valid_mask]
    )

    if quantize:
        perturbed[valid_mask] = np.round(perturbed[valid_mask])

    stego[:required_positions] = perturbed
    return stego


def max_payload_bits_indexed(note_residuals: np.ndarray, n_frames_per_bit: int) -> int:
    """Maksimalan broj bitova na osnovu UKUPNOG broja note-pozicija (uključujući
    promašaje -- promašaji i dalje "zauzimaju mesto" u bloku, samo ne
    doprinose korelaciji)."""
    return len(note_residuals) // n_frames_per_bit


def max_payload_bits(num_residuals: int, n_frames_per_bit: int) -> int:
    """Maksimalan broj bitova koji stane u dati broj rezidualâ za dato N."""
    return num_residuals // n_frames_per_bit


def embed_message(
    residuals: np.ndarray,
    message_bipolar: np.ndarray,
    key: str | int,
    alpha: float,
    n_frames_per_bit: int,
    quantize: bool = True,
) -> np.ndarray:
    """Ugrađuje bipolarnu poruku u niz tajming rezidualâ.

    Parameters
    ----------
    residuals : np.ndarray
        Originalni residual[i] niz (iz Faze 1).
    message_bipolar : np.ndarray
        Bipolarna (+1/-1) poruka za ugradnju, jedan element po bitu.
    key : str | int
        Tajni ključ za generisanje PN sekvence (mora biti isti pri dekodovanju).
    alpha : float
        Jačina ugradnje. Veće alpha -> otporniji signal na šum/kvantizaciju,
        ali veći rizik od statistički uočljive promene (videti Fazu 4).
    n_frames_per_bit : int
        Broj uzastopnih rezidualâ preko kojih se prostire jedan bit poruke
        (N u predlogu, preporučeno N >= 8 zbog integer kvantizacije).
    quantize : bool, default True
        Da li zaokružiti rezultat na cele brojeve. Ovo simulira stvarno
        ograničenje .osr formata (w[i] mora ostati int) -- postavljanjem na
        False može se meriti teorijski BER BEZ efekta kvantizacije, radi
        poređenja koliko kvantizacija sama po sebi degradira kanal.

    Returns
    -------
    np.ndarray
        Stego-residual niz, iste dužine kao ulazni `residuals`. Rezidualy
        van opsega poruke (indeks >= len(message_bipolar) * n_frames_per_bit)
        ostaju nepromenjeni.

    Raises
    ------
    ValueError
        Ako poruka ne stane u dati broj rezidualâ (nedovoljan kapacitet).
    """
    required_residuals = len(message_bipolar) * n_frames_per_bit
    if required_residuals > len(residuals):
        raise ValueError(
            f"Poruka zahteva {required_residuals} rezidualâ "
            f"({len(message_bipolar)} bita x {n_frames_per_bit} frejma), "
            f"a dostupno je samo {len(residuals)}."
        )

    pn = generate_pn_sequence(key, required_residuals).astype(np.float64)
    # Svaki bit poruke se ponavlja n_frames_per_bit puta da odgovara PN dužini.
    bit_per_frame = np.repeat(message_bipolar, n_frames_per_bit).astype(np.float64)

    stego = residuals.copy().astype(np.float64)
    stego[:required_residuals] = (
        residuals[:required_residuals] + alpha * pn * bit_per_frame
    )

    if quantize:
        stego = np.round(stego)

    return stego
