"""Robusna kalibracija JEDNOG fiksnog replay time offseta po beatmapi.

Ne tražimo poseban offset za svaki replay.

Za svaku beatmapu:
1. uzmemo više replay-eva;
2. za svaki napravimo histogram (note_time - raw_press_time);
3. svaki replay histogram normalizujemo da ima jednak doprinos;
4. saberemo histograme;
5. dominantni ZAJEDNIČKI peak daje jedan offset za celu mapu;
6. fiksni offset proverimo na svim kalibracionim replay-evima.

Production analiza zatim koristi samo sačuvani map-level offset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from osu_stego.paths import CONFIG_DIR, DATASET_DIR

import numpy as np
import osrparse

from osu_stego.parsing.beatmap_loader import load_hit_object_times
from osu_stego.parsing.replay_loader import load_key_press_events_raw


DEFAULT_CATEGORIES = ["Elita", "Napredni", "Prosecni", "Pocetni"]


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_beatmap_index(
    dataset_dir: Path,
    categories: list[str],
) -> dict[str, Path]:
    index: dict[str, Path] = {}

    for category in categories:
        osu_dir = dataset_dir / category / "osu"
        if not osu_dir.is_dir():
            continue

        for osu_path in osu_dir.glob("*.osu"):
            try:
                index.setdefault(md5_file(osu_path), osu_path)
            except OSError:
                continue

    return index


def group_replays(
    dataset_dir: Path,
    categories: list[str],
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)

    for category in categories:
        osr_dir = dataset_dir / category / "osr"
        if not osr_dir.is_dir():
            continue

        for osr_path in sorted(osr_dir.glob("*.osr")):
            try:
                replay = osrparse.Replay.from_path(osr_path)
            except Exception:
                continue

            grouped[replay.beatmap_hash].append(osr_path)

    return grouped


def choose_replays(
    paths: list[Path],
    max_replays: int,
) -> list[Path]:
    paths = sorted(paths)

    if len(paths) <= max_replays:
        return paths

    indices = np.linspace(
        0,
        len(paths) - 1,
        max_replays,
        dtype=int,
    )
    return [paths[index] for index in indices]


def add_difference_histogram(
    histogram: np.ndarray,
    notes: np.ndarray,
    presses: np.ndarray,
    search_limit_ms: int,
    bin_width_ms: int,
    chunk_size: int = 256,
) -> None:
    """Dodaje NORMALIZOVAN histogram jednog replay-a u zajednički histogram.

    Chunking sprečava veliku notes x presses matricu u memoriji.
    """
    replay_hist = np.zeros_like(histogram, dtype=np.float64)

    for start in range(0, len(presses), chunk_size):
        chunk = presses[start : start + chunk_size]

        differences = (
            notes[:, None].astype(np.int64)
            - chunk[None, :].astype(np.int64)
        ).ravel()

        valid = differences[
            (differences >= -search_limit_ms)
            & (differences <= search_limit_ms)
        ]

        if len(valid) == 0:
            continue

        bin_indices = (
            (valid + search_limit_ms) // bin_width_ms
        ).astype(np.int64)

        bin_indices = bin_indices[
            (bin_indices >= 0)
            & (bin_indices < len(replay_hist))
        ]

        np.add.at(replay_hist, bin_indices, 1)

    peak = float(np.max(replay_hist))

    # Svaki replay doprinosi maksimalno 1.0 na svom najjačem peak-u.
    if peak > 0:
        replay_hist /= peak
        histogram += replay_hist


def estimate_common_offset(
    notes: np.ndarray,
    replay_paths: list[Path],
    search_limit_ms: int,
    coarse_bin_ms: int,
) -> tuple[int, np.ndarray]:
    num_bins = (
        (2 * search_limit_ms) // coarse_bin_ms
    ) + 1

    combined = np.zeros(num_bins, dtype=np.float64)

    for osr_path in replay_paths:
        presses, _ = load_key_press_events_raw(osr_path)

        if len(presses) == 0:
            continue

        add_difference_histogram(
            combined,
            notes,
            presses,
            search_limit_ms,
            coarse_bin_ms,
        )

    peak_index = int(np.argmax(combined))

    coarse_offset = (
        -search_limit_ms
        + peak_index * coarse_bin_ms
        + coarse_bin_ms // 2
    )

    # Precizna kalibracija samo oko zajedničkog coarse peak-a.
    fine_radius_ms = max(100, coarse_bin_ms * 4)
    fine_start = coarse_offset - fine_radius_ms
    fine_end = coarse_offset + fine_radius_ms

    fine_hist = np.zeros(
        fine_end - fine_start + 1,
        dtype=np.float64,
    )

    for osr_path in replay_paths:
        presses, _ = load_key_press_events_raw(osr_path)
        if len(presses) == 0:
            continue

        replay_fine = np.zeros_like(fine_hist)

        for start in range(0, len(presses), 256):
            chunk = presses[start : start + 256]

            differences = (
                notes[:, None].astype(np.int64)
                - chunk[None, :].astype(np.int64)
            ).ravel()

            valid = differences[
                (differences >= fine_start)
                & (differences <= fine_end)
            ]

            if len(valid) == 0:
                continue

            indices = (valid - fine_start).astype(np.int64)
            np.add.at(replay_fine, indices, 1)

        peak = float(np.max(replay_fine))
        if peak > 0:
            replay_fine /= peak
            fine_hist += replay_fine

    fine_peak = int(np.argmax(fine_hist))
    offset = fine_start + fine_peak

    return int(offset), fine_hist


def max_monotonic_matches(
    presses: np.ndarray,
    notes: np.ndarray,
    window_ms: int,
) -> int:
    i = 0
    j = 0
    matched = 0

    while i < len(presses) and j < len(notes):
        difference = int(presses[i]) - int(notes[j])

        if abs(difference) <= window_ms:
            matched += 1
            i += 1
            j += 1
        elif difference < -window_ms:
            i += 1
        else:
            j += 1

    return matched


def validate_fixed_offset(
    replay_paths: list[Path],
    notes: np.ndarray,
    offset_ms: int,
    window_ms: int,
) -> dict:
    ratios: list[float] = []

    for osr_path in replay_paths:
        raw_presses, _ = load_key_press_events_raw(osr_path)

        matches = max_monotonic_matches(
            raw_presses + offset_ms,
            notes,
            window_ms,
        )

        ratio = matches / len(notes) if len(notes) else 0.0
        ratios.append(ratio)

    if not ratios:
        return {
            "num_replays": 0,
            "median_match_ratio": 0.0,
            "min_match_ratio": 0.0,
            "max_match_ratio": 0.0,
        }

    return {
        "num_replays": len(ratios),
        "median_match_ratio": float(np.median(ratios)),
        "min_match_ratio": float(np.min(ratios)),
        "max_match_ratio": float(np.max(ratios)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robusna zajednička map-level replay time kalibracija."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CONFIG_DIR / "map_time_offsets.json",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
    )
    parser.add_argument(
        "--max-replays-per-map",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--search-limit-ms",
        type=int,
        default=60_000,
    )
    parser.add_argument(
        "--coarse-bin-ms",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--validation-window-ms",
        type=int,
        default=150,
    )
    args = parser.parse_args()

    beatmaps = build_beatmap_index(
        args.dataset_dir,
        args.categories,
    )
    grouped = group_replays(
        args.dataset_dir,
        args.categories,
    )

    offsets: dict[str, int] = {}
    diagnostics: dict[str, dict] = {}

    print("=" * 92)
    print("ROBUST MAP-LEVEL OFFSET CALIBRATION")
    print("=" * 92)

    for number, (beatmap_hash, osu_path) in enumerate(
        sorted(beatmaps.items()),
        1,
    ):
        all_replays = grouped.get(beatmap_hash, [])
        if not all_replays:
            continue

        calibration_replays = choose_replays(
            all_replays,
            args.max_replays_per_map,
        )

        notes = load_hit_object_times(str(osu_path))

        offset, _ = estimate_common_offset(
            notes,
            calibration_replays,
            args.search_limit_ms,
            args.coarse_bin_ms,
        )

        validation = validate_fixed_offset(
            calibration_replays,
            notes,
            offset,
            args.validation_window_ms,
        )

        offsets[beatmap_hash] = offset

        diagnostics[beatmap_hash] = {
            "beatmap_file": osu_path.name,
            "offset_ms": offset,
            "calibration_replays": len(calibration_replays),
            **validation,
        }

        median_ratio = validation["median_match_ratio"]
        minimum_ratio = validation["min_match_ratio"]

        flag = ""
        if median_ratio < 0.95:
            flag = "  [CHECK]"

        print(
            f"[{number:02d}] "
            f"offset={offset:+7d} ms | "
            f"replays={len(calibration_replays):2d} | "
            f"median={100 * median_ratio:6.2f}% | "
            f"min={100 * minimum_ratio:6.2f}% | "
            f"{osu_path.name}{flag}"
        )

    output = {
        "version": 2,
        "method": (
            "single fixed per-beatmap offset from normalized "
            "aggregate note-minus-raw-press histogram"
        ),
        "offsets": offsets,
        "diagnostics": diagnostics,
    }

    args.output.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    bad_maps = sum(
        1
        for info in diagnostics.values()
        if info["median_match_ratio"] < 0.95
    )

    print("\n" + "=" * 92)
    print("GOTOVO")
    print("=" * 92)
    print(f"Kalibrisano mapa: {len(offsets)}")
    print(f"Mapa sa CHECK:     {bad_maps}")
    print(f"Output:            {args.output}")


if __name__ == "__main__":
    main()
