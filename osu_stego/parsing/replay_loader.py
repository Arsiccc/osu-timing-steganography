"""Učitavanje osu!standard replay key-down događaja.

Vremenski model:
- replay frame-ovi čuvaju relativne ``time_delta`` vrednosti;
- RNG sentinel (-12345) nije gameplay frame;
- apsolutna RAW replay vremena dobijamo čistom kumulativnom sumom delta;
- map-time poravnanje se dobija dodavanjem JEDNOG fiksnog offseta po beatmapi.

Važno: ne nuliramo prvi delta i ne primenjujemo replay-specifične korekcije.
Map offset se kalibriše odvojeno, jednom po beatmapi.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import osrparse


_HIT_KEYS = (
    osrparse.Key.M1,
    osrparse.Key.M2,
    osrparse.Key.K1,
    osrparse.Key.K2,
)

_RNG_SEED_SENTINEL_DELTA = -12345


def load_map_offsets(path: str | Path) -> dict[str, int]:
    """Učitava beatmap_hash -> offset_ms mapiranje iz JSON fajla."""
    calibration_path = Path(path)
    data = json.loads(calibration_path.read_text(encoding="utf-8"))

    raw_offsets = data.get("offsets", data)
    if not isinstance(raw_offsets, dict):
        raise ValueError(
            "Offset JSON mora sadržati objekat 'offsets' "
            "ili direktno hash -> offset mapiranje."
        )

    return {
        str(beatmap_hash): int(offset)
        for beatmap_hash, offset in raw_offsets.items()
    }


def _raw_frame_times(
    replay_data: list,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Vraća RAW kumulativno vreme, frame-ove i njihove originalne indekse."""
    kept = [
        (original_index, frame)
        for original_index, frame in enumerate(replay_data)
        if int(frame.time_delta) != _RNG_SEED_SENTINEL_DELTA
    ]

    if not kept:
        return (
            np.array([], dtype=np.int64),
            [],
            np.array([], dtype=np.int64),
        )

    original_indices = np.array(
        [index for index, _ in kept],
        dtype=np.int64,
    )
    frames = [frame for _, frame in kept]

    deltas = np.array(
        [int(frame.time_delta) for frame in frames],
        dtype=np.int64,
    )

    return np.cumsum(deltas), frames, original_indices


def _extract_key_downs(
    times: np.ndarray,
    frames: list,
    original_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(frames) == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    time_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []

    for key in _HIT_KEYS:
        state = np.array(
            [bool(frame.keys & key) for frame in frames],
            dtype=bool,
        )

        previous = np.roll(state, 1)
        previous[0] = False

        key_down = state & ~previous

        time_parts.append(times[key_down])
        index_parts.append(original_indices[key_down])

    all_times = np.concatenate(time_parts)
    all_indices = np.concatenate(index_parts)

    order = np.argsort(all_times, kind="stable")
    return all_times[order], all_indices[order]


def load_key_press_events_raw(
    osr_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Key-down događaji na sirovoj replay vremenskoj osi."""
    replay = osrparse.Replay.from_path(str(osr_path))
    times, frames, indices = _raw_frame_times(replay.replay_data)
    return _extract_key_downs(times, frames, indices)


def load_key_press_events(
    osr_path: str | Path,
    time_offset_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Key-down događaji poravnati na beatmap time osu."""
    raw_times, frame_indices = load_key_press_events_raw(osr_path)
    return raw_times + int(time_offset_ms), frame_indices


def load_key_press_times(
    osr_path: str | Path,
    time_offset_ms: int,
) -> np.ndarray:
    times, _ = load_key_press_events(osr_path, time_offset_ms)
    return times


def load_key_press_events_from_offsets(
    osr_path: str | Path,
    offsets: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    replay = osrparse.Replay.from_path(str(osr_path))

    try:
        offset = offsets[replay.beatmap_hash]
    except KeyError as exc:
        raise KeyError(
            f"Nema kalibrisanog offseta za "
            f"beatmap_hash={replay.beatmap_hash}"
        ) from exc

    return load_key_press_events(osr_path, offset)
