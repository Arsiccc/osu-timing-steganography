"""Note-index-aware physical diagnostics for payload placement experiments."""

from __future__ import annotations

import numpy as np


QUARTERS = (1, 2, 3, 4)


def note_quarter_labels(num_notes: int) -> np.ndarray:
    """Return quarter labels using the same original-index split as features."""
    if num_notes < 0:
        raise ValueError("num_notes mora biti >= 0.")
    labels = np.empty(num_notes, dtype=np.int8)
    for quarter, indices in enumerate(
        np.array_split(np.arange(num_notes, dtype=np.int64), 4), 1
    ):
        labels[indices] = quarter
    return labels


def quarter_shift_diagnostics(
    *,
    num_notes: int,
    requested_positions: np.ndarray,
    active_positions: np.ndarray,
    active_shifts_ms: np.ndarray,
) -> dict[str, float | int]:
    """Summarize requested and writer-applied carriers by original note quarter."""
    requested = np.asarray(requested_positions, dtype=np.int64)
    active = np.asarray(active_positions, dtype=np.int64)
    shifts = np.asarray(active_shifts_ms, dtype=np.float64)
    if requested.ndim != 1 or active.ndim != 1 or shifts.ndim != 1:
        raise ValueError("Quarter diagnostics očekuje 1D nizove.")
    if len(active) != len(shifts):
        raise ValueError("Active positions i shifts moraju biti iste dužine.")
    if np.any(requested < 0) or np.any(requested >= num_notes):
        raise ValueError("Requested carrier je van note-indeksnog opsega.")
    if np.any(active < 0) or np.any(active >= num_notes):
        raise ValueError("Active carrier je van note-indeksnog opsega.")

    labels = note_quarter_labels(num_notes)
    total_energy = float(np.sum(shifts**2))
    result: dict[str, float | int] = {}
    for quarter in QUARTERS:
        requested_count = int(np.sum(labels[requested] == quarter))
        active_mask = labels[active] == quarter
        active_count = int(np.sum(active_mask))
        energy = float(np.sum(shifts[active_mask] ** 2))
        absolute = float(np.sum(np.abs(shifts[active_mask])))
        result[f"requested_carriers_q{quarter}"] = requested_count
        result[f"active_carriers_q{quarter}"] = active_count
        result[f"sum_squared_shift_ms2_q{quarter}"] = energy
        result[f"total_absolute_shift_ms_q{quarter}"] = absolute
        result[f"energy_fraction_q{quarter}"] = (
            energy / total_energy if total_energy > 0.0 else 0.0
        )
    if sum(int(result[f"requested_carriers_q{q}"]) for q in QUARTERS) != len(
        requested
    ):
        raise RuntimeError("Quarter requested zbir nije konzistentan.")
    if sum(int(result[f"active_carriers_q{q}"]) for q in QUARTERS) != len(active):
        raise RuntimeError("Quarter active zbir nije konzistentan.")
    if not np.isclose(
        sum(float(result[f"sum_squared_shift_ms2_q{q}"]) for q in QUARTERS),
        total_energy,
    ):
        raise RuntimeError("Quarter energy zbir nije konzistentan.")
    return result
