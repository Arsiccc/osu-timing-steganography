"""Sparse optimal monotone matching for osu! key presses and hit objects.

Lexicographic objective:
1. maximize number of matched notes;
2. among maximum-cardinality solutions, minimize total absolute timing error.

The implementation exploits the narrow hit window. Instead of O(N*M) dynamic
programming over every note/press pair, it considers only valid candidate
pairs and solves the same non-crossing matching problem with a Fenwick tree.
Complexity is O(K log M), where K is the number of candidate pairs inside the
hit window.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def hit_window_50_ms(overall_difficulty: float) -> float:
    return 200.0 - 10.0 * overall_difficulty


def _validate_inputs(
    key_press_times: np.ndarray,
    note_times: np.ndarray,
    hit_window_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    presses = np.asarray(key_press_times, dtype=np.int64)
    notes = np.asarray(note_times, dtype=np.int64)

    if presses.ndim != 1 or notes.ndim != 1:
        raise ValueError("key_press_times i note_times moraju biti jednodimenzioni nizovi.")
    if hit_window_ms < 0:
        raise ValueError(f"hit_window_ms mora biti >= 0, dobijeno {hit_window_ms}.")
    if len(presses) > 1 and np.any(presses[1:] < presses[:-1]):
        raise ValueError("key_press_times mora biti sortiran rastuće.")
    if len(notes) > 1 and np.any(notes[1:] < notes[:-1]):
        raise ValueError("note_times mora biti sortiran rastuće.")

    return presses, notes


@dataclass
class _Candidate:
    note_index: int
    press_index: int
    matches: int
    cost: int
    previous: int


def _better_candidate(
    candidate_id: int,
    best_id: int,
    candidates: list[_Candidate],
) -> bool:
    if candidate_id == -1:
        return False
    if best_id == -1:
        return True

    candidate = candidates[candidate_id]
    best = candidates[best_id]

    if candidate.matches != best.matches:
        return candidate.matches > best.matches
    if candidate.cost != best.cost:
        return candidate.cost < best.cost

    # Deterministic tie-break only; it does not change the optimization goal.
    if candidate.press_index != best.press_index:
        return candidate.press_index < best.press_index
    return candidate.note_index < best.note_index


class _FenwickBest:
    """Fenwick tree storing the best candidate id for each press prefix."""

    def __init__(self, size: int, candidates: list[_Candidate]) -> None:
        self._tree = [-1] * (size + 1)
        self._candidates = candidates

    def query(self, count: int) -> int:
        """Best candidate using press indices < count."""
        best_id = -1
        index = count

        while index > 0:
            stored_id = self._tree[index]
            if _better_candidate(stored_id, best_id, self._candidates):
                best_id = stored_id
            index -= index & -index

        return best_id

    def update(self, press_index: int, candidate_id: int) -> None:
        index = press_index + 1

        while index < len(self._tree):
            if _better_candidate(candidate_id, self._tree[index], self._candidates):
                self._tree[index] = candidate_id
            index += index & -index


def _optimal_monotone_match_indices(
    key_press_times: np.ndarray,
    note_times: np.ndarray,
    hit_window_ms: float,
) -> list[int]:
    presses, notes = _validate_inputs(key_press_times, note_times, hit_window_ms)
    n_notes = len(notes)
    n_presses = len(presses)

    if n_notes == 0:
        return []
    if n_presses == 0:
        return [-1] * n_notes

    candidates: list[_Candidate] = []
    fenwick = _FenwickBest(n_presses, candidates)
    window = float(hit_window_ms)

    for note_index, note_time_value in enumerate(notes):
        note_time = int(note_time_value)
        left = int(np.searchsorted(presses, note_time - window, side="left"))
        right = int(np.searchsorted(presses, note_time + window, side="right"))

        pending_updates: list[tuple[int, int]] = []

        for press_index in range(left, right):
            previous_id = fenwick.query(press_index)

            if previous_id == -1:
                previous_matches = 0
                previous_cost = 0
            else:
                previous = candidates[previous_id]
                previous_matches = previous.matches
                previous_cost = previous.cost

            candidate = _Candidate(
                note_index=note_index,
                press_index=press_index,
                matches=previous_matches + 1,
                cost=previous_cost + abs(int(presses[press_index]) - note_time),
                previous=previous_id,
            )
            candidate_id = len(candidates)
            candidates.append(candidate)
            pending_updates.append((press_index, candidate_id))

        # Important: update only after all candidates for this note are built,
        # otherwise one note could be used more than once in a chain.
        for press_index, candidate_id in pending_updates:
            fenwick.update(press_index, candidate_id)

    best_id = fenwick.query(n_presses)
    matched_press_idx = [-1] * n_notes

    while best_id != -1:
        candidate = candidates[best_id]
        matched_press_idx[candidate.note_index] = candidate.press_index
        best_id = candidate.previous

    return matched_press_idx


def match_keypresses_to_notes(
    key_press_times: np.ndarray,
    note_times: np.ndarray,
    hit_window_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    presses, notes = _validate_inputs(key_press_times, note_times, hit_window_ms)
    indices = _optimal_monotone_match_indices(presses, notes, hit_window_ms)

    matched_press = [presses[index] for index in indices if index != -1]
    matched_note = [note for index, note in zip(indices, notes) if index != -1]

    return (
        np.asarray(matched_press, dtype=np.int64),
        np.asarray(matched_note, dtype=np.int64),
    )


def match_keypresses_to_notes_with_frame_indices(
    key_press_times: np.ndarray,
    press_frame_indices: np.ndarray,
    note_times: np.ndarray,
    hit_window_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    presses, notes = _validate_inputs(key_press_times, note_times, hit_window_ms)
    frame_indices = np.asarray(press_frame_indices, dtype=np.int64)

    if frame_indices.ndim != 1 or len(presses) != len(frame_indices):
        raise ValueError(
            f"key_press_times ({len(presses)}) i press_frame_indices "
            f"({len(frame_indices)}) moraju biti jednodimenzioni i iste dužine."
        )

    indices = _optimal_monotone_match_indices(presses, notes, hit_window_ms)
    valid_indices = [index for index in indices if index != -1]

    return (
        np.asarray([presses[index] for index in valid_indices], dtype=np.int64),
        np.asarray([note for index, note in zip(indices, notes) if index != -1], dtype=np.int64),
        np.asarray([frame_indices[index] for index in valid_indices], dtype=np.int64),
    )


def match_keypresses_to_notes_indexed(
    key_press_times: np.ndarray,
    press_frame_indices: np.ndarray,
    note_times: np.ndarray,
    hit_window_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    presses, notes = _validate_inputs(key_press_times, note_times, hit_window_ms)
    frame_indices = np.asarray(press_frame_indices, dtype=np.int64)

    if frame_indices.ndim != 1 or len(presses) != len(frame_indices):
        raise ValueError(
            f"key_press_times ({len(presses)}) i press_frame_indices "
            f"({len(frame_indices)}) moraju biti jednodimenzioni i iste dužine."
        )

    indices = _optimal_monotone_match_indices(presses, notes, hit_window_ms)
    residuals = np.full(len(notes), np.nan, dtype=np.float64)
    frame_indices_out = np.full(len(notes), -1, dtype=np.int64)

    for note_index, press_index in enumerate(indices):
        if press_index == -1:
            continue
        residuals[note_index] = float(presses[press_index] - notes[note_index])
        frame_indices_out[note_index] = frame_indices[press_index]

    return residuals, frame_indices_out


def compute_residuals(
    key_press_times: np.ndarray,
    note_times: np.ndarray,
    overall_difficulty: float,
) -> np.ndarray:
    window = hit_window_50_ms(overall_difficulty)
    matched_press, matched_note = match_keypresses_to_notes(
        key_press_times,
        note_times,
        window,
    )
    return matched_press - matched_note
