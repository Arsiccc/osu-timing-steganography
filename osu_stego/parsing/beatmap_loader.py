"""Učitavanje .osu beatmap fajlova i ekstrakcija beat_grid_expected[i]."""

from __future__ import annotations

import numpy as np
import slider
from slider.beatmap import Spinner


def load_hit_object_times(osu_path: str, include_spinners: bool = False) -> np.ndarray:
    """Učitava .osu fajl i vraća očekivano vreme (ms) svake note.

    Parameters
    ----------
    osu_path : str
        Putanja do .osu beatmap fajla.
    include_spinners : bool, default False
        Da li uključiti spinnere u rezultat. Podrazumevano isključeno:
        spinner se izvodi kontinuiranim okretanjem, ne preciznim klikom na
        određeni trenutak, pa njegovo "vreme starta" nije uporedivo sa
        tajming preciznošću koju merimo za circle/slider hit-ove. Uključivanje
        spinnera u residual analizu unelo bi šum nepovezan sa biološkim
        jitter-om koji je predmet istraživanja (potvrđeno na test fajlu:
        jedini spinner u mapi je bio "promašaj" u key-press uparivanju, jer
        igrač ne klikće na start spinnera na isti način kao na notu).

    Returns
    -------
    np.ndarray
        Sortiran niz beat_grid_expected[i] vrednosti u milisekundama.

    Napomena
    --------
    ``HitObject.time`` je tipa ``datetime.timedelta``, ne ``int`` — zato je
    neophodna konverzija preko ``total_seconds() * 1000``. Zaokružujemo na
    ceo milisekund da bi rezolucija odgovarala .osr formatu (koji takođe
    beleži vreme u celobrojnim ms), inače bi poređenje sa w[i] iz replay-a
    unosilo veštački bias u residual[i].
    """
    beatmap = slider.Beatmap.from_path(osu_path)
    hit_objects = beatmap.hit_objects(circles=True, sliders=True, spinners=True)

    if not include_spinners:
        hit_objects = [obj for obj in hit_objects if not isinstance(obj, Spinner)]

    times_ms = np.array(
        [round(obj.time.total_seconds() * 1000) for obj in hit_objects],
        dtype=np.int64,
    )
    return times_ms
