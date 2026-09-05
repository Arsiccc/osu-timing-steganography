"""Upis stego timing shiftova uz očuvanje svih netarget keypress vremena.

Svi key-down frame-ovi su vremenski ANCHOR-i:
- target keydown: original_time + traženi shift;
- ostali keydown: original_time, bez promene;
- samo cursor-only frame-ovi između anchor-a smeju da se prilagode.

Ako traženi target shiftovi naruše hronologiju keypress anchor-a, writer ih
ne maskira tiho. Pozivalac može prvo pozvati ``make_chronology_safe_shifts``,
koji zadržava punu amplitudu ili carrier isključuje (shift=0).
"""

from __future__ import annotations

import copy

import numpy as np
import osrparse


_RNG_SENTINEL = -12345
_HIT_KEYS = (
    osrparse.Key.M1,
    osrparse.Key.M2,
    osrparse.Key.K1,
    osrparse.Key.K2,
)


def _gameplay_data(
    replay: osrparse.Replay,
) -> tuple[np.ndarray, list, np.ndarray]:
    indices: list[int] = []
    frames: list = []
    times: list[int] = []

    cumulative = 0
    sentinel_seen = False

    for original_index, frame in enumerate(replay.replay_data):
        delta = int(frame.time_delta)

        if delta == _RNG_SENTINEL:
            sentinel_seen = True
            continue

        if sentinel_seen:
            raise ValueError(
                "Gameplay frame posle RNG sentinel-a nije podržan."
            )

        cumulative += delta
        indices.append(original_index)
        frames.append(frame)
        times.append(cumulative)

    absolute = np.asarray(times, dtype=np.int64)

    # Legacy .osr može sadržati negativne/non-monotone pomoćne frame delte.
    # To samo po sebi nije razlog da odbacimo replay. Za stego timing je
    # relevantan redosled KEYDOWN anchor-a, koji se proverava zasebno.
    return (
        np.asarray(indices, dtype=np.int64),
        frames,
        absolute,
    )


def _keydown_original_indices(
    gameplay_indices: np.ndarray,
    frames: list,
) -> np.ndarray:
    output: set[int] = set()

    for key in _HIT_KEYS:
        previous = False

        for original_index, frame in zip(
            gameplay_indices.tolist(),
            frames,
        ):
            current = bool(frame.keys & key)

            if current and not previous:
                output.add(int(original_index))

            previous = current

    return np.asarray(sorted(output), dtype=np.int64)


def _normalize_targets(
    frame_indices: np.ndarray,
    shifts_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(frame_indices, dtype=np.int64)
    shifts = np.asarray(shifts_ms, dtype=np.int64)

    if indices.ndim != 1 or shifts.ndim != 1:
        raise ValueError("frame_indices i shifts_ms moraju biti 1D.")
    if len(indices) != len(shifts):
        raise ValueError("frame_indices i shifts_ms moraju biti iste dužine.")

    if len(indices) == 0:
        return indices, shifts

    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    shifts = shifts[order]

    unique_indices: list[int] = []
    unique_shifts: list[int] = []

    start = 0
    while start < len(indices):
        end = start + 1
        while end < len(indices) and indices[end] == indices[start]:
            end += 1

        group = np.unique(shifts[start:end])

        if len(group) != 1:
            raise ValueError(
                f"Frame {int(indices[start])} ima više različitih shiftova: "
                f"{group.tolist()}."
            )

        unique_indices.append(int(indices[start]))
        unique_shifts.append(int(group[0]))
        start = end

    return (
        np.asarray(unique_indices, dtype=np.int64),
        np.asarray(unique_shifts, dtype=np.int64),
    )


def make_chronology_safe_shifts(
    replay: osrparse.Replay,
    frame_indices: np.ndarray,
    shifts_ms: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Vraća shiftove iste dužine; konfliktni carrier-i postaju 0.

    Aktivni carrier zadržava PUNU zadatu amplitudu. Time je ``alpha`` i dalje
    stvarna amplituda aktivnog carrier-a, a ne clipping granica.
    """
    requested_indices = np.asarray(frame_indices, dtype=np.int64)
    requested_shifts = np.asarray(shifts_ms, dtype=np.int64)

    if len(requested_indices) != len(requested_shifts):
        raise ValueError("frame_indices i shifts_ms moraju biti iste dužine.")

    if len(requested_indices) == 0:
        return requested_shifts.copy(), 0

    unique_indices, unique_shifts = _normalize_targets(
        requested_indices,
        requested_shifts,
    )

    if len(unique_indices) != len(requested_indices):
        raise ValueError(
            "make_chronology_safe_shifts očekuje jedinstvene frame indekse."
        )

    gameplay_indices, frames, raw_times = _gameplay_data(replay)
    time_by_index = {
        int(index): int(time)
        for index, time in zip(
            gameplay_indices.tolist(),
            raw_times.tolist(),
        )
    }

    keydown_indices = _keydown_original_indices(
        gameplay_indices,
        frames,
    )
    keydown_set = set(keydown_indices.tolist())

    for frame_index in unique_indices.tolist():
        if int(frame_index) not in keydown_set:
            raise ValueError(
                f"Target frame {frame_index} nije key-down frame."
            )

    safe_by_index = {
        int(index): int(shift)
        for index, shift in zip(
            unique_indices.tolist(),
            unique_shifts.tolist(),
        )
    }

    # Originalna keydown vremena su monotona. Svaku lokalnu inverziju možemo
    # sigurno ukloniti nuliranjem svih target shiftova koji učestvuju u njoj.
    for _ in range(len(keydown_indices) + 1):
        desired = np.asarray(
            [
                time_by_index[int(index)]
                + safe_by_index.get(int(index), 0)
                for index in keydown_indices.tolist()
            ],
            dtype=np.int64,
        )

        bad = np.flatnonzero(desired[:-1] > desired[1:])

        if len(bad) == 0:
            break

        changed = False

        for position in bad.tolist():
            left = int(keydown_indices[position])
            right = int(keydown_indices[position + 1])

            if safe_by_index.get(left, 0) != 0:
                safe_by_index[left] = 0
                changed = True

            if safe_by_index.get(right, 0) != 0:
                safe_by_index[right] = 0
                changed = True

        if not changed:
            raise RuntimeError(
                "Hronološka inverzija postoji i bez target shiftova."
            )
    else:
        raise RuntimeError("Nije uspelo stabilizovanje target shiftova.")

    safe = np.asarray(
        [safe_by_index[int(index)] for index in requested_indices.tolist()],
        dtype=np.int64,
    )

    dropped = int(
        np.sum(
            (requested_shifts != 0)
            & (safe == 0)
        )
    )

    return safe, dropped


def apply_frame_time_shifts(
    replay: osrparse.Replay,
    frame_indices: np.ndarray,
    shifts_ms: np.ndarray,
) -> osrparse.Replay:
    target_indices, target_shifts = _normalize_targets(
        frame_indices,
        shifts_ms,
    )

    gameplay_indices, frames, original_times = _gameplay_data(replay)

    if len(target_indices) == 0:
        return copy.copy(replay)

    position_by_index = {
        int(index): position
        for position, index in enumerate(gameplay_indices.tolist())
    }
    original_time_by_index = {
        int(index): int(time)
        for index, time in zip(
            gameplay_indices.tolist(),
            original_times.tolist(),
        )
    }

    keydown_indices = _keydown_original_indices(
        gameplay_indices,
        frames,
    )
    keydown_set = set(keydown_indices.tolist())

    for frame_index in target_indices.tolist():
        if int(frame_index) not in keydown_set:
            raise ValueError(
                f"Target frame {frame_index} nije key-down frame."
            )

    shift_by_index = {
        int(index): int(shift)
        for index, shift in zip(
            target_indices.tolist(),
            target_shifts.tolist(),
        )
    }

    anchor_positions = np.asarray(
        [position_by_index[int(index)] for index in keydown_indices.tolist()],
        dtype=np.int64,
    )
    desired_anchor_times = np.asarray(
        [
            original_time_by_index[int(index)]
            + shift_by_index.get(int(index), 0)
            for index in keydown_indices.tolist()
        ],
        dtype=np.int64,
    )

    if (
        len(desired_anchor_times) > 1
        and np.any(desired_anchor_times[:-1] > desired_anchor_times[1:])
    ):
        raise ValueError(
            "Traženi shiftovi narušavaju redosled keypress frame-ova. "
            "Pozovi make_chronology_safe_shifts pre write_stego_replay."
        )

    adjusted = original_times.copy()
    adjusted[anchor_positions] = desired_anchor_times

    if len(anchor_positions) > 0:
        first_pos = int(anchor_positions[0])
        first_time = int(desired_anchor_times[0])

        if first_pos > 0:
            adjusted[:first_pos] = np.minimum(
                original_times[:first_pos],
                first_time,
            )

        for anchor_number in range(len(anchor_positions) - 1):
            left_pos = int(anchor_positions[anchor_number])
            right_pos = int(anchor_positions[anchor_number + 1])
            left_time = int(desired_anchor_times[anchor_number])
            right_time = int(desired_anchor_times[anchor_number + 1])

            if right_pos - left_pos > 1:
                adjusted[left_pos + 1 : right_pos] = np.clip(
                    original_times[left_pos + 1 : right_pos],
                    left_time,
                    right_time,
                )

        last_pos = int(anchor_positions[-1])
        last_time = int(desired_anchor_times[-1])

        if last_pos + 1 < len(adjusted):
            adjusted[last_pos + 1 :] = np.maximum(
                original_times[last_pos + 1 :],
                last_time,
            )

    # Ne zahtevamo globalnu monotonost svih pomoćnih cursor frame-ova:
    # legacy replay može već sadržati negativne delte. Ono što mora ostati
    # konzistentno jesu keydown anchor vremena.
    # Svi keydown frame-ovi moraju biti TAČNO na svojim anchor vremenima.
    if not np.array_equal(
        adjusted[anchor_positions],
        desired_anchor_times,
    ):
        raise RuntimeError("Keypress anchor vremena nisu očuvana.")

    new_deltas = np.empty_like(adjusted)

    if len(adjusted) > 0:
        new_deltas[0] = adjusted[0]
    if len(adjusted) > 1:
        new_deltas[1:] = np.diff(adjusted)

    delta_by_index = {
        int(index): int(delta)
        for index, delta in zip(
            gameplay_indices.tolist(),
            new_deltas.tolist(),
        )
    }

    new_frames = [
        osrparse.ReplayEventOsu(
            time_delta=delta_by_index.get(
                original_index,
                int(frame.time_delta),
            ),
            x=frame.x,
            y=frame.y,
            keys=frame.keys,
        )
        for original_index, frame in enumerate(replay.replay_data)
    ]

    output = copy.copy(replay)
    output.replay_data = new_frames
    return output


def write_stego_replay(
    original_osr_path: str,
    output_osr_path: str,
    frame_indices: np.ndarray,
    shifts_ms: np.ndarray,
) -> None:
    replay = osrparse.Replay.from_path(original_osr_path)
    stego = apply_frame_time_shifts(
        replay,
        frame_indices,
        shifts_ms,
    )
    stego.write_path(output_osr_path)
