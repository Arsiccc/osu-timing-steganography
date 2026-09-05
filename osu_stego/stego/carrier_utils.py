"""Pomoćne funkcije za izbor fizički validnih stego carrier-a."""

from __future__ import annotations

import numpy as np


def deduplicate_note_frame_carriers(
    residuals: np.ndarray,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Jedan fizički replay frame može biti najviše jedan timing carrier.

    Ako se isti frame indeks pojavi na više note-pozicija, zadržava se prva
    pozicija; ostale se pretvaraju u NaN / -1. Note-indeksirana dužina ostaje
    ista, pa PN poravnanje nije pomereno.
    """
    if len(residuals) != len(frame_indices):
        raise ValueError("residuals i frame_indices moraju biti iste dužine.")

    cleaned_residuals = residuals.copy()
    cleaned_frames = frame_indices.copy()

    seen: set[int] = set()
    removed = 0

    for position, frame_index in enumerate(cleaned_frames.tolist()):
        frame_index = int(frame_index)

        if frame_index == -1:
            continue

        if frame_index in seen:
            cleaned_residuals[position] = np.nan
            cleaned_frames[position] = -1
            removed += 1
            continue

        seen.add(frame_index)

    return cleaned_residuals, cleaned_frames, removed
