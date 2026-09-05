"""Payload sweep: real capacity vs BER vs steganalysis detectability.

Run from project root:

    python3 -m scripts.experiments.run_payload_sweep

Default configurations:
    alpha=20, N=4
    alpha=15, N=6
    alpha=10, N=16

Default payload fractions:
    10%, 25%, 50%, 75%, 100%

For each replay/config/fraction:
- choose message length as floor(nominal_capacity_bits * payload_fraction)
- deterministic random message for reproducibility
- encode with the same indexed spread-spectrum pipeline
- write real .osr
- reload + rematch
- decode and compute BER
- extract steganalysis features from clean/stego residuals

Then:
- aggregate BER by config/payload
- evaluate Random Forest ROC-AUC with GroupKFold
  (clean/stego versions of the same replay stay in the same fold)
- save overall and performance-group summaries

Important:
"100% payload" means 100% of the NOMINAL note-indexed capacity:
    floor(num_notes / N) bits.
It does not mean every physical carrier survives writer safety filters.

Outputs:
    results/payload/payload_sweep_results.csv
    results/payload/payload_sweep_features.csv
    results/payload/payload_sweep_summary.csv
    results/payload/payload_sweep_auc.csv
    results/payload/payload_sweep_by_performance.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import (
    CONFIG_DIR,
    DATASET_DIR,
    METADATA_DIR,
    RESULTS_DIR,
)
from osu_stego.stego.encoder import embed_message_indexed
from osu_stego.stego.decoder import extract_bipolar_message_indexed
from osu_stego.parsing.osr_writer import (
    make_chronology_safe_shifts,
    write_stego_replay,
)
from scripts.experiments.run_pilot_ber_sweep import (
    build_beatmap_index,
    load_map_offsets,
    load_residuals,
    prepare_replay,
    stable_seed,
)
from scripts.experiments.run_steganalysis import (
    FEATURE_COLUMNS,
    evaluate_subset,
    residual_features,
)


DEFAULT_CONFIGS = [
    (20.0, 4),
    (15.0, 6),
    (10.0, 16),
]

DEFAULT_PAYLOAD_FRACTIONS = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

PERFORMANCE_ORDER = [
    "Very Poor",
    "Poor",
    "Good",
    "Very Good",
]

RESULT_FIELDS = [
    "status",
    "error",
    "category",
    "replay_file",
    "player",
    "beatmap_hash",
    "alpha",
    "n_frames_per_bit",
    "payload_fraction",
    "nominal_capacity_bits",
    "message_bits",
    "bit_errors_memory",
    "ber_memory",
    "bit_errors_roundtrip",
    "ber_roundtrip",
    "num_notes",
    "num_matched_before",
    "num_matched_after",
    "original_unmatched",
    "stego_unmatched",
    "delta_unmatched",
    "positive_new_unmatched",
    "requested_carriers",
    "active_carriers",
    "active_carrier_fraction",
    "dropped_for_hit_window",
    "dropped_for_chronology",
    "runtime_seconds",
]

FEATURE_FIELDS = [
    "category",
    "replay_file",
    "player",
    "beatmap_hash",
    "alpha",
    "n_frames_per_bit",
    "payload_fraction",
    "message_bits",
    "label",
    "variance",
    "excess_kurtosis",
    "autocorrelation_lag1",
    "n_valid_residuals",
]


def parse_configs(values: list[str] | None) -> list[tuple[float, int]]:
    if not values:
        return DEFAULT_CONFIGS.copy()

    configs: list[tuple[float, int]] = []

    for value in values:
        try:
            alpha_text, n_text = value.split(":", maxsplit=1)
            alpha = float(alpha_text)
            n_value = int(n_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Neispravna konfiguracija '{value}'. "
                "Koristi format alpha:N, npr. 20:4."
            ) from exc

        if alpha <= 0 or n_value <= 0:
            raise ValueError("alpha i N moraju biti > 0.")

        configs.append((alpha, n_value))

    return configs


def parse_payloads(values: list[float] | None) -> list[float]:
    payloads = (
        DEFAULT_PAYLOAD_FRACTIONS.copy()
        if not values
        else [float(value) for value in values]
    )

    for value in payloads:
        if not (0.0 < value <= 1.0):
            raise ValueError(
                "Sve payload fractions moraju biti u intervalu (0, 1]."
            )

    return sorted(set(payloads))


def deterministic_message(
    replay_file: str,
    beatmap_hash: str,
    alpha: float,
    n_value: int,
    payload_fraction: float,
    num_bits: int,
    seed: int,
) -> np.ndarray:
    # Same deterministic bit stream for a replay across all configs/payloads.
    # Shorter payloads are prefixes of longer ones, which makes comparisons fairer.
    rng = np.random.default_rng(
        stable_seed(
            seed,
            replay_file,
            beatmap_hash,
            "payload-message",
        )
    )

    return rng.choice(
        np.array([-1, 1], dtype=np.int8),
        size=num_bits,
    )


def nominal_capacity_bits(
    num_notes: int,
    n_value: int,
) -> int:
    return num_notes // n_value


def message_length_for_fraction(
    capacity_bits: int,
    payload_fraction: float,
) -> int:
    if capacity_bits <= 0:
        return 0

    return max(
        1,
        int(np.floor(capacity_bits * payload_fraction)),
    )


def append_csv_row(
    path: Path,
    fieldnames: list[str],
    row: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    is_new = not path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if is_new:
            writer.writeheader()

        writer.writerow(
            {
                field: row.get(field, "")
                for field in fieldnames
            }
        )


def completed_experiments(
    results_path: Path,
) -> set[tuple[str, float, int, float]]:
    if not results_path.is_file():
        return set()

    frame = pd.read_csv(results_path)

    if frame.empty:
        return set()

    completed: set[
        tuple[str, float, int, float]
    ] = set()

    for row in frame.itertuples(index=False):
        if str(row.status) != "OK":
            continue

        completed.add(
            (
                str(row.replay_file),
                float(row.alpha),
                int(row.n_frames_per_bit),
                round(float(row.payload_fraction), 6),
            )
        )

    return completed


def feature_pair_completed(
    features_path: Path,
) -> set[tuple[str, float, int, float]]:
    if not features_path.is_file():
        return set()

    frame = pd.read_csv(features_path)

    if frame.empty:
        return set()

    completed: set[
        tuple[str, float, int, float]
    ] = set()

    grouped = frame.groupby(
        [
            "replay_file",
            "alpha",
            "n_frames_per_bit",
            "payload_fraction",
        ]
    )

    for keys, group in grouped:
        if set(group["label"].astype(int)) == {0, 1}:
            replay_file, alpha, n_value, payload_fraction = keys
            completed.add(
                (
                    str(replay_file),
                    float(alpha),
                    int(n_value),
                    round(float(payload_fraction), 6),
                )
            )

    return completed


def build_physical_stego(
    context,
    message: np.ndarray,
    alpha: float,
    n_value: int,
    hit_margin_ms: float,
    temp_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    original = context.original_residuals
    note_frames = context.note_frame_indices

    required = len(message) * n_value

    if required > len(original):
        raise ValueError(
            f"Potrebno {required} note pozicija, dostupno {len(original)}."
        )

    target_stego = embed_message_indexed(
        original,
        message,
        "payload-sweep-key-v1",
        alpha,
        n_value,
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

    limit = max(
        0.0,
        context.hit_window_ms - hit_margin_ms,
    )

    hit_safe = np.zeros(
        required,
        dtype=bool,
    )

    hit_safe[base_usable] = (
        np.abs(
            original[:required][base_usable]
            + requested_shifts[base_usable]
        )
        <= limit
    )

    dropped_hit_window = int(
        np.sum(
            originally_requested
            & ~hit_safe
        )
    )

    requested_shifts[
        originally_requested
        & ~hit_safe
    ] = 0

    target_positions = np.flatnonzero(
        base_usable
        & (requested_shifts != 0)
    )

    target_frames = note_frames[
        target_positions
    ]
    requested_target_shifts = requested_shifts[
        target_positions
    ]

    (
        safe_target_shifts,
        dropped_chronology,
    ) = make_chronology_safe_shifts(
        context.replay,
        target_frames,
        requested_target_shifts,
    )

    active_mask = (
        safe_target_shifts != 0
    )

    physical_stego = original.copy()

    physical_stego[
        target_positions
    ] = (
        original[target_positions]
        + safe_target_shifts
    )

    write_stego_replay(
        str(context.osr_path),
        str(temp_path),
        target_frames[active_mask],
        safe_target_shifts[active_mask],
    )

    stego_residuals, _, _ = load_residuals(
        temp_path,
        context.beatmap,
        context.offset_ms,
        context.hit_window_ms,
    )

    diagnostics = {
        "requested_carriers": int(
            np.sum(originally_requested)
        ),
        "active_carriers": int(
            np.sum(active_mask)
        ),
        "dropped_for_hit_window": (
            dropped_hit_window
        ),
        "dropped_for_chronology": int(
            dropped_chronology
        ),
    }

    return (
        physical_stego,
        stego_residuals,
        diagnostics,
    )


def run_one(
    context,
    alpha: float,
    n_value: int,
    payload_fraction: float,
    seed: int,
    hit_margin_ms: float,
    temp_dir: Path,
) -> tuple[dict, dict, dict]:
    start = time.perf_counter()

    original = context.original_residuals
    num_notes = len(original)

    capacity_bits = nominal_capacity_bits(
        num_notes,
        n_value,
    )

    num_bits = message_length_for_fraction(
        capacity_bits,
        payload_fraction,
    )

    if num_bits <= 0:
        raise ValueError(
            "Replay nema ni 1 bit nominalnog kapaciteta."
        )

    message = deterministic_message(
        replay_file=context.replay_file,
        beatmap_hash=context.beatmap_hash,
        alpha=alpha,
        n_value=n_value,
        payload_fraction=payload_fraction,
        num_bits=num_bits,
        seed=seed,
    )

    temp_path = temp_dir / (
        f"{context.osr_path.stem}"
        f"_a{alpha:g}"
        f"_n{n_value}"
        f"_p{int(round(payload_fraction * 100))}.osr"
    )

    (
        physical_stego,
        stego_residuals,
        diagnostics,
    ) = build_physical_stego(
        context=context,
        message=message,
        alpha=alpha,
        n_value=n_value,
        hit_margin_ms=hit_margin_ms,
        temp_path=temp_path,
    )

    decoded_memory = extract_bipolar_message_indexed(
        physical_stego,
        "payload-sweep-key-v1",
        n_value,
        num_bits,
    )

    decoded_roundtrip = extract_bipolar_message_indexed(
        stego_residuals,
        "payload-sweep-key-v1",
        n_value,
        num_bits,
    )

    memory_errors = int(
        np.sum(
            decoded_memory != message
        )
    )
    roundtrip_errors = int(
        np.sum(
            decoded_roundtrip != message
        )
    )

    original_valid = ~np.isnan(
        original
    )
    stego_valid = ~np.isnan(
        stego_residuals
    )

    requested = int(
        diagnostics[
            "requested_carriers"
        ]
    )
    active = int(
        diagnostics[
            "active_carriers"
        ]
    )

    result = {
        "status": "OK",
        "error": "",
        "category": context.category,
        "replay_file": context.replay_file,
        "player": context.player,
        "beatmap_hash": context.beatmap_hash,
        "alpha": float(alpha),
        "n_frames_per_bit": int(n_value),
        "payload_fraction": float(payload_fraction),
        "nominal_capacity_bits": int(capacity_bits),
        "message_bits": int(num_bits),
        "bit_errors_memory": memory_errors,
        "ber_memory": memory_errors / num_bits,
        "bit_errors_roundtrip": roundtrip_errors,
        "ber_roundtrip": roundtrip_errors / num_bits,
        "num_notes": int(num_notes),
        "num_matched_before": int(
            np.sum(original_valid)
        ),
        "num_matched_after": int(
            np.sum(stego_valid)
        ),
        "original_unmatched": int(
            np.sum(~original_valid)
        ),
        "stego_unmatched": int(
            np.sum(~stego_valid)
        ),
        "delta_unmatched": int(
            np.sum(~stego_valid)
            - np.sum(~original_valid)
        ),
        "positive_new_unmatched": max(
            0,
            int(
                np.sum(~stego_valid)
                - np.sum(~original_valid)
            ),
        ),
        "requested_carriers": requested,
        "active_carriers": active,
        "active_carrier_fraction": (
            active / requested
            if requested
            else 0.0
        ),
        "dropped_for_hit_window": int(
            diagnostics[
                "dropped_for_hit_window"
            ]
        ),
        "dropped_for_chronology": int(
            diagnostics[
                "dropped_for_chronology"
            ]
        ),
        "runtime_seconds": (
            time.perf_counter()
            - start
        ),
    }

    clean_stats = residual_features(
        original
    )
    stego_stats = residual_features(
        stego_residuals
    )

    feature_common = {
        "category": context.category,
        "replay_file": context.replay_file,
        "player": context.player,
        "beatmap_hash": context.beatmap_hash,
        "alpha": float(alpha),
        "n_frames_per_bit": int(n_value),
        "payload_fraction": float(payload_fraction),
        "message_bits": int(num_bits),
    }

    clean_feature = {
        **feature_common,
        **clean_stats,
        "label": 0,
    }

    stego_feature = {
        **feature_common,
        **stego_stats,
        "label": 1,
    }

    return (
        result,
        clean_feature,
        stego_feature,
    )


def write_ber_summary(
    results: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    ok = results[
        results["status"] == "OK"
    ].copy()

    summary = (
        ok.groupby(
            [
                "alpha",
                "n_frames_per_bit",
                "payload_fraction",
            ],
            as_index=False,
        )
        .agg(
            replay_count=("replay_file", "nunique"),
            total_bits=("message_bits", "sum"),
            bit_errors=("bit_errors_roundtrip", "sum"),
            mean_ber_per_replay=("ber_roundtrip", "mean"),
            median_ber_per_replay=("ber_roundtrip", "median"),
            zero_ber_fraction=(
                "ber_roundtrip",
                lambda values: (values == 0).mean(),
            ),
            mean_active_carrier_fraction=(
                "active_carrier_fraction",
                "mean",
            ),
            total_positive_new_unmatched=(
                "positive_new_unmatched",
                "sum",
            ),
        )
    )

    summary["ber"] = (
        summary["bit_errors"]
        / summary["total_bits"]
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    return summary


def evaluate_auc(
    features: pd.DataFrame,
    configs: list[tuple[float, int]],
    payloads: list[float],
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    for alpha, n_value in configs:
        for payload_fraction in payloads:
            subset = features[
                np.isclose(
                    features["payload_fraction"],
                    payload_fraction,
                )
            ].copy()

            summary, _ = evaluate_subset(
                frame=subset,
                scope="ALL",
                alpha=alpha,
                n_value=n_value,
                folds=folds,
                trees=trees,
                seed=seed,
                bootstrap_iterations=bootstrap_iterations,
            )

            summary[
                "payload_fraction"
            ] = float(
                payload_fraction
            )

            rows.append(summary)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "alpha",
                "n_frames_per_bit",
                "payload_fraction",
            ]
        )
        .reset_index(drop=True)
    )


def write_performance_summary(
    results: pd.DataFrame,
    auc_features: pd.DataFrame,
    performance: pd.DataFrame,
    configs: list[tuple[float, int]],
    payloads: list[float],
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
    output_path: Path,
) -> None:
    labels = performance[
        [
            "replay_file",
            "performance_category",
        ]
    ].copy()

    ber_merged = results.merge(
        labels,
        on="replay_file",
        how="left",
        validate="many_to_one",
    )

    ber_rows: list[dict] = []

    for (
        performance_category,
        alpha,
        n_value,
        payload_fraction,
    ), group in ber_merged[
        ber_merged["status"] == "OK"
    ].groupby(
        [
            "performance_category",
            "alpha",
            "n_frames_per_bit",
            "payload_fraction",
        ]
    ):
        total_bits = int(
            group["message_bits"].sum()
        )
        bit_errors = int(
            group[
                "bit_errors_roundtrip"
            ].sum()
        )

        ber_rows.append(
            {
                "performance_category": performance_category,
                "alpha": float(alpha),
                "n_frames_per_bit": int(n_value),
                "payload_fraction": float(payload_fraction),
                "replay_count": int(
                    group["replay_file"].nunique()
                ),
                "total_bits": total_bits,
                "bit_errors": bit_errors,
                "ber": (
                    bit_errors / total_bits
                    if total_bits
                    else np.nan
                ),
                "mean_active_carrier_fraction": float(
                    group[
                        "active_carrier_fraction"
                    ].mean()
                ),
            }
        )

    ber_frame = pd.DataFrame(
        ber_rows
    )

    feature_merged = auc_features.merge(
        labels,
        on="replay_file",
        how="left",
        validate="many_to_one",
    )

    feature_merged = feature_merged.copy()
    feature_merged["category"] = (
        feature_merged[
            "performance_category"
        ]
    )

    auc_rows: list[dict] = []

    for alpha, n_value in configs:
        for payload_fraction in payloads:
            payload_subset = feature_merged[
                np.isclose(
                    feature_merged[
                        "payload_fraction"
                    ],
                    payload_fraction,
                )
            ].copy()

            for performance_category in PERFORMANCE_ORDER:
                summary, _ = evaluate_subset(
                    frame=payload_subset,
                    scope=performance_category,
                    alpha=alpha,
                    n_value=n_value,
                    folds=folds,
                    trees=trees,
                    seed=seed,
                    bootstrap_iterations=bootstrap_iterations,
                )

                auc_rows.append(
                    {
                        "performance_category": performance_category,
                        "alpha": float(alpha),
                        "n_frames_per_bit": int(n_value),
                        "payload_fraction": float(payload_fraction),
                        "roc_auc": float(
                            summary["roc_auc"]
                        ),
                        "auc_ci95_low": float(
                            summary[
                                "auc_ci95_low"
                            ]
                        ),
                        "auc_ci95_high": float(
                            summary[
                                "auc_ci95_high"
                            ]
                        ),
                    }
                )

    auc_frame = pd.DataFrame(
        auc_rows
    )

    combined = ber_frame.merge(
        auc_frame,
        on=[
            "performance_category",
            "alpha",
            "n_frames_per_bit",
            "payload_fraction",
        ],
        how="inner",
        validate="one_to_one",
    )

    combined.to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Payload fraction sweep: BER + AUC "
            "at increasing real payload usage."
        )
    )

    payload_dir = (
        RESULTS_DIR
        / "payload"
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
        "--performance",
        type=Path,
        default=METADATA_DIR / "performance_groups.csv",
    )
    parser.add_argument(
        "--map-offsets",
        type=Path,
        default=CONFIG_DIR / "map_time_offsets.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=payload_dir / "payload_sweep_results.csv",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=payload_dir / "payload_sweep_features.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=payload_dir / "payload_sweep_summary.csv",
    )
    parser.add_argument(
        "--auc-output",
        type=Path,
        default=payload_dir / "payload_sweep_auc.csv",
    )
    parser.add_argument(
        "--performance-output",
        type=Path,
        default=payload_dir / "payload_sweep_by_performance.csv",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=None,
        help=(
            "Config list in alpha:N form, e.g. "
            "--config 20:4 15:6 10:16"
        ),
    )
    parser.add_argument(
        "--payloads",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Fractions in (0,1], e.g. "
            "--payloads 0.1 0.25 0.5 0.75 1.0"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--hit-margin-ms",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
    )

    args = parser.parse_args()

    configs = parse_configs(
        args.config
    )
    payloads = parse_payloads(
        args.payloads
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.features.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.evaluate_only:
        source = pd.read_csv(
            args.results
        )

        beatmaps = build_beatmap_index(
            args.dataset_dir
        )
        offsets = load_map_offsets(
            args.map_offsets
        )

        completed_results = completed_experiments(
            args.output
        )
        completed_features = feature_pair_completed(
            args.features
        )

        total = (
            len(source)
            * len(configs)
            * len(payloads)
        )

        print("=" * 100)
        print("PAYLOAD SWEEP")
        print("=" * 100)
        print(f"Replay-eva: {len(source)}")
        print(f"Configs:    {configs}")
        print(f"Payloads:   {payloads}")
        print(f"Pokušaja:   {total}")
        print(
            f"Resume OK:  {len(completed_results)} "
            "već završenih eksperimenata"
        )
        print()

        counter = 0
        new_ok = 0
        errors = 0
        global_start = time.perf_counter()

        with tempfile.TemporaryDirectory(
            prefix="osu_payload_sweep_"
        ) as temp_name:
            temp_dir = Path(
                temp_name
            )

            for replay_index, row in enumerate(
                source.itertuples(index=False),
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
                    print(
                        f"[{replay_index:03d}/{len(source):03d}] "
                        f"BASELINE ERROR {row.replay_file}: {exc!r}"
                    )
                    continue

                for alpha, n_value in configs:
                    for payload_fraction in payloads:
                        counter += 1

                        key = (
                            context.replay_file,
                            float(alpha),
                            int(n_value),
                            round(
                                float(payload_fraction),
                                6,
                            ),
                        )

                        if (
                            key in completed_results
                            and key in completed_features
                        ):
                            continue

                        try:
                            (
                                result,
                                clean_feature,
                                stego_feature,
                            ) = run_one(
                                context=context,
                                alpha=alpha,
                                n_value=n_value,
                                payload_fraction=payload_fraction,
                                seed=args.seed,
                                hit_margin_ms=args.hit_margin_ms,
                                temp_dir=temp_dir,
                            )

                            if key not in completed_results:
                                append_csv_row(
                                    args.output,
                                    RESULT_FIELDS,
                                    result,
                                )

                            if key not in completed_features:
                                append_csv_row(
                                    args.features,
                                    FEATURE_FIELDS,
                                    clean_feature,
                                )
                                append_csv_row(
                                    args.features,
                                    FEATURE_FIELDS,
                                    stego_feature,
                                )

                            new_ok += 1

                        except Exception as exc:
                            errors += 1

                            append_csv_row(
                                args.output,
                                RESULT_FIELDS,
                                {
                                    "status": "ERROR",
                                    "error": repr(exc),
                                    "category": context.category,
                                    "replay_file": context.replay_file,
                                    "player": context.player,
                                    "beatmap_hash": context.beatmap_hash,
                                    "alpha": alpha,
                                    "n_frames_per_bit": n_value,
                                    "payload_fraction": payload_fraction,
                                },
                            )

                            print(
                                f"ERROR {context.replay_file[:12]} "
                                f"a={alpha:g} N={n_value} "
                                f"p={payload_fraction:.2f}: {exc!r}"
                            )

                        if counter % 100 == 0:
                            elapsed = (
                                time.perf_counter()
                                - global_start
                            )

                            print(
                                f"progress {counter:5d}/{total} | "
                                f"new OK={new_ok} ERR={errors} | "
                                f"{elapsed / 60:.1f} min"
                            )

    results_frame = pd.read_csv(
        args.output
    )
    features_frame = pd.read_csv(
        args.features
    )

    summary = write_ber_summary(
        results_frame,
        args.summary,
    )

    auc_frame = evaluate_auc(
        features=features_frame,
        configs=configs,
        payloads=payloads,
        folds=args.folds,
        trees=args.trees,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )

    auc_frame.to_csv(
        args.auc_output,
        index=False,
    )

    performance = pd.read_csv(
        args.performance
    )

    write_performance_summary(
        results=results_frame,
        auc_features=features_frame,
        performance=performance,
        configs=configs,
        payloads=payloads,
        folds=args.folds,
        trees=args.trees,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
        output_path=args.performance_output,
    )

    final = summary.merge(
        auc_frame[
            [
                "alpha",
                "n_frames_per_bit",
                "payload_fraction",
                "roc_auc",
                "auc_ci95_low",
                "auc_ci95_high",
            ]
        ],
        on=[
            "alpha",
            "n_frames_per_bit",
            "payload_fraction",
        ],
        how="inner",
        validate="one_to_one",
    )

    print("\n" + "=" * 100)
    print("PAYLOAD SWEEP SUMMARY")
    print("=" * 100)

    print(
        final[
            [
                "alpha",
                "n_frames_per_bit",
                "payload_fraction",
                "total_bits",
                "ber",
                "roc_auc",
                "auc_ci95_low",
                "auc_ci95_high",
                "mean_active_carrier_fraction",
                "total_positive_new_unmatched",
            ]
        ].to_string(index=False)
    )

    print("\nFajlovi:")
    print(f"  {args.output}")
    print(f"  {args.features}")
    print(f"  {args.summary}")
    print(f"  {args.auc_output}")
    print(f"  {args.performance_output}")


if __name__ == "__main__":
    main()
