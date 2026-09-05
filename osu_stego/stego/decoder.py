"""Spread Spectrum decoder: ekstrakcija poruke korelacijom sa PN sekvencom.

Formula iz predloga (Faza 3):
    korelacija = sum(residual_stego[i] * pn[i]) za i u bloku od N frejmova
    pozitivna korelacija -> bit 1, negativna -> bit 0
"""

from __future__ import annotations

import numpy as np

from .pn_sequence import generate_pn_sequence


def extract_bipolar_message_indexed(
    stego_note_residuals: np.ndarray,
    key: str | int,
    n_frames_per_bit: int,
    num_bits: int,
) -> np.ndarray:
    """Ekstrahuje poruku iz note-indeksiranog stego niza (sa NaN na promašajima).

    Za razliku od extract_bipolar_message, PN sekvenca je poravnata sa
    POZICIJOM NOTE, ne sa sažetom listom uparenih vrednosti -- pozicije sa
    NaN (promašaj pri dekodovanju, koji NE MORA biti isti skup promašaja
    kao pri enkodovanju, videti napomenu u osr_writer.py) jednostavno ne
    ulaze u korelacionu sumu za svoj blok, umesto da pomere poravnanje
    svih narednih bitova.

    Returns
    -------
    np.ndarray
        Bipolarni niz (-1/+1) dekodovanih bitova, dužine num_bits.

    Napomena o degenerisanom slučaju
    ---------------------------------
    Ako SVE pozicije u bloku od N nota nedostaju (sve promašene), korelacija
    za taj blok je 0 -- trenutno se to tretira kao bit=1 (ista konvencija kao
    kod korelacije tačno 0 u neindeksiranoj verziji), ali ovo je efektivno
    NASUMIČNO tačan/netačan odgovor (50/50), ne stvarno dekodovanje. Ovo je
    poznato ograničenje -- pozivalac bi trebalo da poveća N ili redundansu
    (npr. error-correcting kod) ako je broj promašaja po bloku visok.
    """
    required = num_bits * n_frames_per_bit
    if required > len(stego_note_residuals):
        raise ValueError(
            f"Potrebno je {required} note-pozicija za {num_bits} bita "
            f"(N={n_frames_per_bit}), a dostupno je samo {len(stego_note_residuals)}."
        )

    pn = generate_pn_sequence(key, required)
    relevant = stego_note_residuals[:required]

    correlations = np.zeros(num_bits, dtype=np.float64)
    for b in range(num_bits):
        block = relevant[b * n_frames_per_bit : (b + 1) * n_frames_per_bit]
        block_pn = pn[b * n_frames_per_bit : (b + 1) * n_frames_per_bit]
        valid_mask = ~np.isnan(block)
        correlations[b] = np.sum(block[valid_mask] * block_pn[valid_mask])

    return np.where(correlations >= 0, 1, -1).astype(np.int8)


def extract_bipolar_message(
    stego_residuals: np.ndarray,
    key: str | int,
    n_frames_per_bit: int,
    num_bits: int,
) -> np.ndarray:
    """Ekstrahuje bipolarnu poruku iz stego-rezidualâ korelacionom analizom.

    Parameters
    ----------
    stego_residuals : np.ndarray
        Rezidualy (potencijalno modifikovani encoder-om, uključujući efekat
        kvantizacije ako je primenjena pri ugradnji).
    key : str | int
        Isti tajni ključ korišćen pri ugradnji -- neophodan da bi se
        regenerisala identična PN sekvenca.
    n_frames_per_bit : int
        Isti N korišćen pri ugradnji.
    num_bits : int
        Broj bitova poruke koji se očekuje (mora biti poznat dekoderu
        unapred -- u praktičnoj primeni ovo bi bilo deo protokola, npr.
        fiksna dužina zaglavlja koje enkodira dužinu poruke).

    Returns
    -------
    np.ndarray
        Bipolarni niz (-1/+1) dekodovanih bitova, dužine num_bits. Vrednost
        korelacije tačno 0 (retko, ali moguće) deterministički se tretira kao
        bipolarni bit +1, identično note-indeksiranom i DISTRIBUTED putu.

    Raises
    ------
    ValueError
        Ako stego_residuals nema dovoljno elemenata za traženi broj bitova.
    """
    required = num_bits * n_frames_per_bit
    if required > len(stego_residuals):
        raise ValueError(
            f"Potrebno je {required} rezidualâ za {num_bits} bita "
            f"(N={n_frames_per_bit}), a dostupno je samo {len(stego_residuals)}."
        )

    pn = generate_pn_sequence(key, required)
    relevant = stego_residuals[:required].astype(np.float64)

    correlations = np.array(
        [
            np.sum(
                relevant[b * n_frames_per_bit : (b + 1) * n_frames_per_bit]
                * pn[b * n_frames_per_bit : (b + 1) * n_frames_per_bit]
            )
            for b in range(num_bits)
        ]
    )

    return np.where(correlations >= 0, 1, -1).astype(np.int8)
