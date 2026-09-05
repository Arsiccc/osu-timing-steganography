"""Pilot BER sweep za osu! replay Spread Spectrum projekat.

Eksperiment:
- 20 replay-eva po skill kategoriji (80 ukupno)
- alpha = [2, 5, 10, 15, 20] ms
- N = [8, 16, 32]
- 8 bita po replay-u / konfiguraciji
- ukupno 80 * 5 * 3 = 1200 end-to-end round-trip eksperimenata

Glavna BER metrika se meri TEK nakon:
encode -> .osr write -> reload -> rematching -> decode.

Uzorak je deterministički i preferira različite beatmape pre nego što
uzima više replay-eva iste mape.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from osu_stego.paths import (
    CONFIG_DIR,
    DATASET_DIR,
    METADATA_DIR,
    PILOT_BER_RESULTS_DIR,
)

import numpy as np
import osrparse
import pandas as pd
import slider

from osu_stego.parsing.beatmap_loader import load_hit_object_times
from osu_stego.stego.carrier_utils import deduplicate_note_frame_carriers
from osu_stego.stego.decoder import extract_bipolar_message_indexed
from osu_stego.stego.encoder import embed_message_indexed
from osu_stego.matching.matcher import hit_window_50_ms, match_keypresses_to_notes_indexed
from osu_stego.parsing.osr_writer import make_chronology_safe_shifts, write_stego_replay
from osu_stego.parsing.replay_loader import load_key_press_events, load_map_offsets


CATEGORIES = ["Elita", "Napredni", "Prosecni", "Pocetni"]

MOD_EASY = 2
MOD_HARD_ROCK = 16

DEFAULT_ALPHAS = [2.0, 5.0, 10.0, 15.0, 20.0]
DEFAULT_N_VALUES = [8, 16, 32]


@dataclass(frozen=True)
class BeatmapData:
    path: Path
    note_times: np.ndarray
    base_od: float


@dataclass
class ReplayContext:
    category: str
    replay_file: str
    player: str
    beatmap_hash: str
    osr_path: Path
    beatmap: BeatmapData
    replay: osrparse.Replay
    offset_ms: int
    hit_window_ms: float
    original_residuals: np.ndarray
    note_frame_indices: np.ndarray
    duplicate_carriers_removed: int


RESULT_FIELDS = [
    "status",
    "error",
    "category",
    "replay_file",
    "player",
    "beatmap_hash",
    "alpha",
    "n_frames_per_bit",
    "message_bits",
    "bit_errors_memory",
    "ber_memory",
    "bit_errors_roundtrip",
    "ber_roundtrip",
    "num_notes",
    "num_matched_before",
    "num_matched_after",
    "match_ratio_before",
    "match_ratio_after",
    "original_unmatched",
    "stego_unmatched",
    "delta_unmatched",
    "positive_new_unmatched",
    "changed_match_status_total",
    "changed_match_status_used",
    "duplicate_carriers_removed",
    "duplicate_carriers_removed_after",
    "requested_carriers",
    "active_carriers",
    "active_carrier_fraction",
    "dropped_for_hit_window",
    "dropped_for_chronology",
    "hit_window_ms",
    "runtime_seconds",
]


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def mod_value(replay: osrparse.Replay) -> int:
    mods = replay.mods

    if hasattr(mods, "value"):
        return int(mods.value)

    return int(mods)


def effective_od(base_od: float, replay: osrparse.Replay) -> float:
    """OD posle EZ/HR difficulty modova, u našoj beatmap-time osi."""
    value = float(base_od)
    mods = mod_value(replay)

    if mods & MOD_EASY:
        value *= 0.5

    if mods & MOD_HARD_ROCK:
        value = min(value * 1.4, 10.0)

    return value


def build_beatmap_index(
    dataset_dir: Path,
) -> dict[str, BeatmapData]:
    index: dict[str, BeatmapData] = {}

    for osu_path in dataset_dir.glob("*/osu/*.osu"):
        try:
            beatmap_hash = md5_file(osu_path)

            if beatmap_hash in index:
                continue

            beatmap = slider.Beatmap.from_path(str(osu_path))
            index[beatmap_hash] = BeatmapData(
                path=osu_path,
                note_times=load_hit_object_times(str(osu_path)),
                base_od=float(beatmap.od()),
            )
        except Exception:
            continue

    return index


def select_diverse_replays(
    results: pd.DataFrame,
    per_category: int,
    min_notes: int,
    seed: int,
) -> pd.DataFrame:
    """Deterministički uzorak, uz prioritet jedne instance po beatmapi."""
    selected_parts: list[pd.DataFrame] = []

    for category_index, category in enumerate(CATEGORIES):
        group = results[
            (results["skill_category"] == category)
            & (results["num_notes"] >= min_notes)
        ].copy()

        if len(group) < per_category:
            raise ValueError(
                f"{category}: samo {len(group)} replay-eva ima >= {min_notes} nota, "
                f"a traženo je {per_category}."
            )

        random_state = stable_seed(seed, category, "selection")
        shuffled = group.sample(
            frac=1.0,
            random_state=random_state,
        ).reset_index(drop=True)

        first_per_map = shuffled.drop_duplicates(
            subset=["beatmap_hash"],
            keep="first",
        )

        chosen = first_per_map.head(per_category)

        if len(chosen) < per_category:
            chosen_files = set(chosen["replay_file"].tolist())
            remaining = shuffled[
                ~shuffled["replay_file"].isin(chosen_files)
            ]
            needed = per_category - len(chosen)
            chosen = pd.concat(
                [chosen, remaining.head(needed)],
                ignore_index=True,
            )

        selected_parts.append(chosen)

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    return selected


def load_or_create_selection(
    results: pd.DataFrame,
    selection_path: Path,
    per_category: int,
    min_notes: int,
    seed: int,
    reset_selection: bool,
) -> pd.DataFrame:
    if selection_path.exists() and not reset_selection:
        selected = pd.read_csv(selection_path)

        required_columns = {
            "skill_category",
            "replay_file",
            "beatmap_hash",
            "num_notes",
        }

        if not required_columns.issubset(selected.columns):
            raise ValueError(
                f"{selection_path} nema očekivane kolone. "
                "Koristi --reset-selection."
            )

        print(f"Koristim postojeći pilot uzorak: {selection_path}")
        return selected

    selected = select_diverse_replays(
        results,
        per_category,
        min_notes,
        seed,
    )

    selected.to_csv(selection_path, index=False)
    print(f"Sačuvan pilot uzorak: {selection_path}")

    return selected


def deterministic_message(
    beatmap_hash: str,
    replay_file: str,
    seed: int,
    num_bits: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        stable_seed(
            seed,
            beatmap_hash,
            replay_file,
            "message",
        )
    )

    return rng.choice(
        np.array([-1, 1], dtype=np.int8),
        size=num_bits,
    )


def load_residuals(
    osr_path: Path,
    beatmap: BeatmapData,
    offset_ms: int,
    hit_window_ms: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    press_times, press_frames = load_key_press_events(
        osr_path,
        offset_ms,
    )

    residuals, note_frames = match_keypresses_to_notes_indexed(
        press_times,
        press_frames,
        beatmap.note_times,
        hit_window_ms,
    )

    return deduplicate_note_frame_carriers(
        residuals,
        note_frames,
    )


def prepare_replay(
    row,
    dataset_dir: Path,
    beatmaps: dict[str, BeatmapData],
    offsets: dict[str, int],
) -> ReplayContext:
    category = str(row.skill_category)
    replay_file = str(row.replay_file)
    beatmap_hash = str(row.beatmap_hash)

    osr_path = (
        dataset_dir
        / category
        / "osr"
        / replay_file
    )

    if not osr_path.is_file():
        raise FileNotFoundError(osr_path)

    if beatmap_hash not in beatmaps:
        raise KeyError(
            f"Nema .osu za beatmap_hash={beatmap_hash}"
        )

    if beatmap_hash not in offsets:
        raise KeyError(
            f"Nema map time offseta za beatmap_hash={beatmap_hash}"
        )

    beatmap = beatmaps[beatmap_hash]
    replay = osrparse.Replay.from_path(osr_path)
    offset_ms = int(offsets[beatmap_hash])

    od = effective_od(
        beatmap.base_od,
        replay,
    )
    hit_window_ms = float(
        hit_window_50_ms(od)
    )

    residuals, note_frames, duplicate_removed = load_residuals(
        osr_path,
        beatmap,
        offset_ms,
        hit_window_ms,
    )

    return ReplayContext(
        category=category,
        replay_file=replay_file,
        player=str(replay.username),
        beatmap_hash=beatmap_hash,
        osr_path=osr_path,
        beatmap=beatmap,
        replay=replay,
        offset_ms=offset_ms,
        hit_window_ms=hit_window_ms,
        original_residuals=residuals,
        note_frame_indices=note_frames,
        duplicate_carriers_removed=duplicate_removed,
    )


def build_error_row(
    context: ReplayContext | None,
    row,
    alpha: float,
    n_frames_per_bit: int,
    message_bits: int,
    error: Exception,
    runtime_seconds: float,
) -> dict:
    return {
        "status": "ERROR",
        "error": repr(error),
        "category": (
            context.category
            if context is not None
            else str(row.skill_category)
        ),
        "replay_file": (
            context.replay_file
            if context is not None
            else str(row.replay_file)
        ),
        "player": (
            context.player
            if context is not None
            else str(getattr(row, "player", ""))
        ),
        "beatmap_hash": (
            context.beatmap_hash
            if context is not None
            else str(row.beatmap_hash)
        ),
        "alpha": float(alpha),
        "n_frames_per_bit": int(n_frames_per_bit),
        "message_bits": int(message_bits),
        "runtime_seconds": float(runtime_seconds),
    }


def run_experiment(
    context: ReplayContext,
    alpha: float,
    n_frames_per_bit: int,
    message: np.ndarray,
    key: str,
    hit_margin_ms: float,
    temp_dir: Path,
) -> dict:
    start_time = time.perf_counter()

    original = context.original_residuals
    note_frames = context.note_frame_indices
    num_bits = len(message)

    required = num_bits * n_frames_per_bit

    if required > len(original):
        raise ValueError(
            f"Potrebno {required} note-pozicija, dostupno {len(original)}."
        )

    target_stego = embed_message_indexed(
        original,
        message,
        key,
        alpha,
        n_frames_per_bit,
        quantize=True,
    )

    requested_shifts = np.zeros(
        required,
        dtype=np.int64,
    )

    base_usable = (
        (note_frames[:required] != -1)
        & ~np.isnan(original[:required])
        & ~np.isnan(target_stego[:required])
    )

    requested_shifts[base_usable] = np.rint(
        target_stego[:required][base_usable]
        - original[:required][base_usable]
    ).astype(np.int64)

    originally_requested = (
        base_usable
        & (requested_shifts != 0)
    )

    # Ne dozvoli carrier-u da direktno pređe hit-window.
    limit = max(
        0.0,
        context.hit_window_ms - hit_margin_ms,
    )

    safe_for_hit_window = np.zeros(
        required,
        dtype=bool,
    )

    safe_for_hit_window[base_usable] = (
        np.abs(
            original[:required][base_usable]
            + requested_shifts[base_usable]
        )
        <= limit
    )

    dropped_hit_window = int(
        np.sum(
            originally_requested
            & ~safe_for_hit_window
        )
    )

    requested_shifts[
        originally_requested
        & ~safe_for_hit_window
    ] = 0

    target_positions = np.flatnonzero(
        base_usable
        & (requested_shifts != 0)
    )

    target_frames = note_frames[target_positions]
    target_requested_shifts = requested_shifts[target_positions]

    safe_target_shifts, dropped_chronology = (
        make_chronology_safe_shifts(
            context.replay,
            target_frames,
            target_requested_shifts,
        )
    )

    active_mask = safe_target_shifts != 0

    # BER u memoriji koristi identične fizički realizovane shiftove.
    physical_stego = original.copy()
    physical_stego[target_positions] = (
        original[target_positions]
        + safe_target_shifts
    )

    decoded_memory = extract_bipolar_message_indexed(
        physical_stego,
        key,
        n_frames_per_bit,
        num_bits,
    )

    bit_errors_memory = int(
        np.sum(message != decoded_memory)
    )

    output_path = temp_dir / (
        f"{context.osr_path.stem}"
        f"_a{alpha:g}"
        f"_n{n_frames_per_bit}.osr"
    )

    write_stego_replay(
        str(context.osr_path),
        str(output_path),
        target_frames[active_mask],
        safe_target_shifts[active_mask],
    )

    (
        stego_residuals,
        _,
        duplicate_after,
    ) = load_residuals(
        output_path,
        context.beatmap,
        context.offset_ms,
        context.hit_window_ms,
    )

    decoded_roundtrip = extract_bipolar_message_indexed(
        stego_residuals,
        key,
        n_frames_per_bit,
        num_bits,
    )

    bit_errors_roundtrip = int(
        np.sum(message != decoded_roundtrip)
    )

    original_valid = ~np.isnan(original)
    stego_valid = ~np.isnan(stego_residuals)

    original_unmatched = int(
        np.sum(~original_valid)
    )
    stego_unmatched = int(
        np.sum(~stego_valid)
    )

    changed_total = int(
        np.sum(original_valid != stego_valid)
    )
    changed_used = int(
        np.sum(
            original_valid[:required]
            != stego_valid[:required]
        )
    )

    num_notes = len(original)
    matched_before = int(
        np.sum(original_valid)
    )
    matched_after = int(
        np.sum(stego_valid)
    )

    requested_carriers = int(
        np.sum(originally_requested)
    )
    active_carriers = int(
        np.sum(active_mask)
    )

    runtime = time.perf_counter() - start_time

    return {
        "status": "OK",
        "error": "",
        "category": context.category,
        "replay_file": context.replay_file,
        "player": context.player,
        "beatmap_hash": context.beatmap_hash,
        "alpha": float(alpha),
        "n_frames_per_bit": int(n_frames_per_bit),
        "message_bits": int(num_bits),
        "bit_errors_memory": bit_errors_memory,
        "ber_memory": bit_errors_memory / num_bits,
        "bit_errors_roundtrip": bit_errors_roundtrip,
        "ber_roundtrip": bit_errors_roundtrip / num_bits,
        "num_notes": num_notes,
        "num_matched_before": matched_before,
        "num_matched_after": matched_after,
        "match_ratio_before": (
            matched_before / num_notes
            if num_notes
            else 0.0
        ),
        "match_ratio_after": (
            matched_after / num_notes
            if num_notes
            else 0.0
        ),
        "original_unmatched": original_unmatched,
        "stego_unmatched": stego_unmatched,
        "delta_unmatched": (
            stego_unmatched - original_unmatched
        ),
        "positive_new_unmatched": max(
            0,
            stego_unmatched - original_unmatched,
        ),
        "changed_match_status_total": changed_total,
        "changed_match_status_used": changed_used,
        "duplicate_carriers_removed": (
            context.duplicate_carriers_removed
        ),
        "duplicate_carriers_removed_after": duplicate_after,
        "requested_carriers": requested_carriers,
        "active_carriers": active_carriers,
        "active_carrier_fraction": (
            active_carriers / requested_carriers
            if requested_carriers
            else 0.0
        ),
        "dropped_for_hit_window": dropped_hit_window,
        "dropped_for_chronology": dropped_chronology,
        "hit_window_ms": context.hit_window_ms,
        "runtime_seconds": runtime,
    }


def existing_completed(
    output_path: Path,
) -> set[tuple[str, float, int]]:
    if not output_path.exists():
        return set()

    completed: set[tuple[str, float, int]] = set()

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        for row in csv.DictReader(file):
            if row.get("status") != "OK":
                continue

            try:
                completed.add(
                    (
                        row["replay_file"],
                        float(row["alpha"]),
                        int(row["n_frames_per_bit"]),
                    )
                )
            except Exception:
                continue

    return completed


def append_result(
    output_path: Path,
    row: dict,
) -> None:
    is_new = not output_path.exists()

    normalized = {
        field: row.get(field, "")
        for field in RESULT_FIELDS
    }

    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )

        if is_new:
            writer.writeheader()

        writer.writerow(normalized)


def write_summaries(
    output_path: Path,
    summary_path: Path,
    skill_summary_path: Path,
) -> None:
    frame = pd.read_csv(output_path)
    ok = frame[frame["status"] == "OK"].copy()

    if ok.empty:
        return

    def summarize(group: pd.DataFrame) -> pd.Series:
        total_bits = int(
            group["message_bits"].sum()
        )
        total_errors = int(
            group["bit_errors_roundtrip"].sum()
        )

        return pd.Series(
            {
                "experiments": len(group),
                "total_bits": total_bits,
                "bit_errors": total_errors,
                "ber": (
                    total_errors / total_bits
                    if total_bits
                    else np.nan
                ),
                "mean_ber_per_replay": (
                    group["ber_roundtrip"].mean()
                ),
                "median_ber_per_replay": (
                    group["ber_roundtrip"].median()
                ),
                "zero_ber_fraction": (
                    (group["ber_roundtrip"] == 0).mean()
                ),
                "ber_under_5pct_fraction": (
                    (group["ber_roundtrip"] < 0.05).mean()
                ),
                "mean_active_carrier_fraction": (
                    group["active_carrier_fraction"].mean()
                ),
                "total_drop_hit_window": int(
                    group["dropped_for_hit_window"].sum()
                ),
                "total_drop_chronology": int(
                    group["dropped_for_chronology"].sum()
                ),
                "mean_delta_unmatched": (
                    group["delta_unmatched"].mean()
                ),
                "total_positive_new_unmatched": int(
                    group["positive_new_unmatched"].sum()
                ),
            }
        )

    overall = (
        ok.groupby(
            ["alpha", "n_frames_per_bit"],
            as_index=False,
        )
        .apply(summarize)
        .reset_index()
    )

    # pandas verzije se razlikuju u ponašanju groupby.apply;
    # zadržavamo samo relevantne kolone ako se pojavi dodatni index.
    keep_overall = [
        column
        for column in [
            "alpha",
            "n_frames_per_bit",
            "experiments",
            "total_bits",
            "bit_errors",
            "ber",
            "mean_ber_per_replay",
            "median_ber_per_replay",
            "zero_ber_fraction",
            "ber_under_5pct_fraction",
            "mean_active_carrier_fraction",
            "total_drop_hit_window",
            "total_drop_chronology",
            "mean_delta_unmatched",
            "total_positive_new_unmatched",
        ]
        if column in overall.columns
    ]
    overall = overall[keep_overall]

    by_skill = (
        ok.groupby(
            [
                "category",
                "alpha",
                "n_frames_per_bit",
            ],
            as_index=False,
        )
        .apply(summarize)
        .reset_index()
    )

    keep_skill = [
        column
        for column in [
            "category",
            "alpha",
            "n_frames_per_bit",
            "experiments",
            "total_bits",
            "bit_errors",
            "ber",
            "mean_ber_per_replay",
            "median_ber_per_replay",
            "zero_ber_fraction",
            "ber_under_5pct_fraction",
            "mean_active_carrier_fraction",
            "total_drop_hit_window",
            "total_drop_chronology",
            "mean_delta_unmatched",
            "total_positive_new_unmatched",
        ]
        if column in by_skill.columns
    ]
    by_skill = by_skill[keep_skill]

    overall.to_csv(
        summary_path,
        index=False,
    )
    by_skill.to_csv(
        skill_summary_path,
        index=False,
    )


def print_summary(
    output_path: Path,
) -> None:
    frame = pd.read_csv(output_path)
    ok = frame[frame["status"] == "OK"].copy()
    errors = frame[frame["status"] != "OK"].copy()

    print("\n" + "=" * 100)
    print("PILOT BER SUMMARY")
    print("=" * 100)
    print(f"OK eksperimenata: {len(ok)}")
    print(f"Grešaka:          {len(errors)}")

    if ok.empty:
        return

    print()
    print(
        f"{'alpha':>7} {'N':>4} {'exp':>5} "
        f"{'BER':>8} {'zeroBER':>9} "
        f"{'active':>9} {'new_unm':>8}"
    )
    print("-" * 64)

    for (alpha, n_value), group in ok.groupby(
        ["alpha", "n_frames_per_bit"]
    ):
        total_bits = int(
            group["message_bits"].sum()
        )
        total_errors = int(
            group["bit_errors_roundtrip"].sum()
        )
        ber = (
            total_errors / total_bits
            if total_bits
            else float("nan")
        )

        print(
            f"{alpha:7.1f} "
            f"{int(n_value):4d} "
            f"{len(group):5d} "
            f"{ber:8.4f} "
            f"{(group['ber_roundtrip'] == 0).mean():9.1%} "
            f"{group['active_carrier_fraction'].mean():9.3f} "
            f"{int(group['positive_new_unmatched'].sum()):8d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot alpha x N BER sweep sa pravim .osr round-tripom."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=METADATA_DIR / "results_v3_clean.csv",
    )
    parser.add_argument(
        "--map-offsets",
        type=Path,
        default=CONFIG_DIR / "map_time_offsets.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PILOT_BER_RESULTS_DIR / "pilot_ber_results.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PILOT_BER_RESULTS_DIR / "pilot_ber_summary.csv",
    )
    parser.add_argument(
        "--skill-summary",
        type=Path,
        default=PILOT_BER_RESULTS_DIR / "pilot_ber_summary_by_skill.csv",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=PILOT_BER_RESULTS_DIR / "pilot_ber_selected_replays.csv",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=DEFAULT_N_VALUES,
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--hit-margin-ms",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--key",
        type=str,
        default="pilot-ber-key-v1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--reset-selection",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Ne postoji dataset: {args.dataset_dir}"
        )

    if not args.results.is_file():
        raise FileNotFoundError(
            f"Ne postoji results CSV: {args.results}"
        )

    if not args.map_offsets.is_file():
        raise FileNotFoundError(
            f"Ne postoji offset JSON: {args.map_offsets}"
        )

    if args.bits <= 0:
        raise ValueError("--bits mora biti > 0.")

    if args.per_category <= 0:
        raise ValueError("--per-category mora biti > 0.")

    if not args.n_values or min(args.n_values) <= 0:
        raise ValueError("Sve N vrednosti moraju biti > 0.")

    if not args.alphas or min(args.alphas) <= 0:
        raise ValueError("Sve alpha vrednosti moraju biti > 0.")

    if args.output.exists():
        if args.overwrite:
            args.output.unlink()
        elif not args.resume:
            raise FileExistsError(
                f"{args.output} već postoji. "
                "Koristi --resume ili --overwrite."
            )

    results = pd.read_csv(args.results)
    offsets = load_map_offsets(args.map_offsets)
    beatmaps = build_beatmap_index(args.dataset_dir)

    max_required_positions = (
        args.bits * max(args.n_values)
    )

    selected = load_or_create_selection(
        results=results,
        selection_path=args.selection,
        per_category=args.per_category,
        min_notes=max_required_positions,
        seed=args.seed,
        reset_selection=args.reset_selection,
    )

    # Selection fajl mora odgovarati trenutnim CLI parametrima.
    counts = selected.groupby(
        "skill_category"
    ).size()

    for category in CATEGORIES:
        count = int(counts.get(category, 0))
        if count != args.per_category:
            raise ValueError(
                f"Selection ima {count} replay-eva za {category}, "
                f"a --per-category={args.per_category}. "
                "Koristi --reset-selection."
            )

    too_short = selected[
        selected["num_notes"] < max_required_positions
    ]
    if not too_short.empty:
        raise ValueError(
            "Selection sadrži replay koji nema dovoljno nota za "
            f"{args.bits} bita pri N={max(args.n_values)}. "
            "Koristi --reset-selection."
        )

    total_replays = len(selected)
    total_configurations = (
        len(args.alphas)
        * len(args.n_values)
    )
    total_experiments = (
        total_replays
        * total_configurations
    )

    completed = (
        existing_completed(args.output)
        if args.resume
        else set()
    )

    print("=" * 100)
    print("PILOT BER SWEEP")
    print("=" * 100)
    print(f"Replay-eva:       {total_replays} ({args.per_category} po kategoriji)")
    print(f"Alpha:            {args.alphas}")
    print(f"N:                {args.n_values}")
    print(f"Poruka:           {args.bits} bita")
    print(f"Max note pozicija:{max_required_positions}")
    print(f"Ukupno pokušaja:  {total_experiments}")
    if completed:
        print(f"Već završeno:     {len(completed)}")
    print()

    experiment_counter = 0
    ok_counter = 0
    error_counter = 0
    global_start = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix="osu_pilot_ber_"
    ) as temp_name:
        temp_dir = Path(temp_name)

        for replay_number, row in enumerate(
            selected.itertuples(index=False),
            1,
        ):
            try:
                context = prepare_replay(
                    row,
                    args.dataset_dir,
                    beatmaps,
                    offsets,
                )
            except Exception as exc:
                context = None
                print(
                    f"[{replay_number:02d}/{total_replays:02d}] "
                    f"{row.skill_category} | {row.replay_file[:12]} | "
                    f"BASELINE ERROR: {exc!r}"
                )

            message = deterministic_message(
                str(row.beatmap_hash),
                str(row.replay_file),
                args.seed,
                args.bits,
            )

            print(
                f"[{replay_number:02d}/{total_replays:02d}] "
                f"{row.skill_category:10s} | "
                f"{str(row.replay_file)[:12]}"
            )

            for n_value in args.n_values:
                for alpha in args.alphas:
                    experiment_counter += 1

                    config_key = (
                        str(row.replay_file),
                        float(alpha),
                        int(n_value),
                    )

                    if config_key in completed:
                        continue

                    start = time.perf_counter()

                    try:
                        if context is None:
                            raise RuntimeError(
                                "Baseline priprema replay-a nije uspela."
                            )

                        result = run_experiment(
                            context=context,
                            alpha=float(alpha),
                            n_frames_per_bit=int(n_value),
                            message=message,
                            key=args.key,
                            hit_margin_ms=args.hit_margin_ms,
                            temp_dir=temp_dir,
                        )
                        ok_counter += 1
                    except Exception as exc:
                        result = build_error_row(
                            context=context,
                            row=row,
                            alpha=float(alpha),
                            n_frames_per_bit=int(n_value),
                            message_bits=args.bits,
                            error=exc,
                            runtime_seconds=(
                                time.perf_counter() - start
                            ),
                        )
                        error_counter += 1

                    append_result(
                        args.output,
                        result,
                    )

                    if (
                        experiment_counter % 25 == 0
                        or result["status"] != "OK"
                    ):
                        elapsed = (
                            time.perf_counter()
                            - global_start
                        )
                        print(
                            f"  progress "
                            f"{experiment_counter:4d}/{total_experiments} | "
                            f"OK={ok_counter} ERR={error_counter} | "
                            f"elapsed={elapsed:.1f}s"
                        )

    write_summaries(
        args.output,
        args.summary,
        args.skill_summary,
    )

    print_summary(args.output)

    elapsed = time.perf_counter() - global_start

    print("\n" + "=" * 100)
    print("GOTOVO")
    print("=" * 100)
    print(f"Ukupno vreme:      {elapsed:.1f} s")
    print(f"Detaljni rezultati:{args.output}")
    print(f"Overall summary:   {args.summary}")
    print(f"Skill summary:     {args.skill_summary}")
    print(f"Pilot uzorak:      {args.selection}")


if __name__ == "__main__":
    main()
