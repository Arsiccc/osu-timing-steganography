"""Generisanje pseudoslučajne PN (pseudo-noise) sekvence determinisane tajnim ključem.

Dizajn: ključ (proizvoljan string ili int) se prvo hešira preko SHA-256 da bi
se dobio numerički seed za numpy PRNG. Ovo obezbeđuje dve stvari:
1. Determinizam -- isti ključ uvek generiše identičnu PN sekvencu (neophodno
   da bi decoder mogao da regeneriše istu sekvencu bez pristupa encoderu).
2. Avalanche efekat -- i sitna razlika u ključu (npr. jedan karakter) daje
   potpuno drugačiji seed, pa samim tim i potpuno drugačiju PN sekvencu.
   Ovo je bitno za bezbednost šeme: bez poznavanja tačnog ključa, napadač
   ne može da pogodi PN sekvencu "približno".

Napomena: numpy PRNG (PCG64, podrazumevani generator iza np.random.default_rng)
NIJE kriptografski bezbedan generator. Za potrebe ovog istraživanja (statistička
nevidljivost i BER merenje) ovo je prihvatljivo, pošto se ne oslanjamo na
kriptografsku sigurnost PN sekvence protiv namernog kriptoanalitičkog napada
u samom Faza 2/3 delu -- Faza 4 (stegoanalizator) upravo testira da li se
uopšte može detektovati uграđena poruka bez znanja ključa, što je drugačiji
(statistički, ne kriptografski) model pretnje.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _derive_seed(key: str | int) -> int:
    """Determinisano izvodi 32-bitni seed iz proizvoljnog ključa preko SHA-256."""
    key_bytes = str(key).encode("utf-8")
    digest = hashlib.sha256(key_bytes).digest()
    # Uzimamo prva 4 bajta digest-a kao unsigned 32-bit int seed.
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def generate_pn_sequence(key: str | int, length: int) -> np.ndarray:
    """Generiše determinisanu bipolarnu (+1/-1) PN sekvencu date dužine.

    Parameters
    ----------
    key : str | int
        Tajni ključ. Isti ključ uvek daje identičnu sekvencu.
    length : int
        Broj elemenata sekvence.

    Returns
    -------
    np.ndarray
        Niz dtype=int8 sa vrednostima isključivo -1 ili +1.
    """
    if length <= 0:
        raise ValueError(f"length mora biti pozitivan broj, dobijeno: {length}")

    seed = _derive_seed(key)
    rng = np.random.default_rng(seed)
    uniform_samples = rng.random(length)
    return np.where(uniform_samples < 0.5, -1, 1).astype(np.int8)
