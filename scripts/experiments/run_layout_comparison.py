"""Controlled PREFIX vs DISTRIBUTED payload-layout experiment.

Run from project root:

    python3 -m scripts.experiments.run_layout_comparison

Default:
- exact 80-replay pilot cohort
- alpha=14, N=8
- payloads 7.5%, 10%, 15%
- layouts: prefix, distributed
- one fixed PN key and five independent layout keys

The same replay, message bits, PN key, alpha, N and payload length are used
for both layouts. Only block placement changes.

Outputs:
    results/payload/layout_multiseed_results.csv
    results/payload/layout_multiseed_features.csv
    results/payload/layout_multiseed_auc.csv
    results/payload/layout_multiseed_paired.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from osu_stego.analysis.layout_diagnostics import quarter_shift_diagnostics
from osu_stego.paths import CONFIG_DIR, DATASET_DIR, RESULTS_DIR
from osu_stego.parsing.osr_writer import (
    make_chronology_safe_shifts,
    write_stego_replay,
)
from osu_stego.stego.payload_layout import (
    embed_message_with_layout,
    extract_bipolar_message_with_layout,
    select_payload_blocks,
)
from scripts.experiments.run_payload_sweep import (
    deterministic_message,
    message_length_for_fraction,
    nominal_capacity_bits,
)
from scripts.experiments.run_pilot_ber_sweep import (
    build_beatmap_index,
    load_map_offsets,
    load_residuals,
    prepare_replay,
    stable_seed,
)
from scripts.experiments.run_steganalysis import (
    evaluate_subset,
    residual_features,
)


LAYOUTS = ("prefix", "distributed")
CV_GROUP_COLUMNS = ("replay_file", "beatmap_hash")
DEFAULT_PAYLOADS = (0.075, 0.10, 0.15)
DEFAULT_PN_KEY = "payload-pn-key-v1"
DEFAULT_LAYOUT_KEYS = tuple(
    f"payload-layout-seed-{index}"
    for index in range(5)
)
EXPERIMENT_VERSION = "layout-multiseed-v2"

RESULT_FIELDS = [
    "status",
    "error",
    "experiment_version",
    "config_id",
    "message_seed",
    "pn_key",
    "pn_key_id",
    "layout_key",
    "layout_key_id",
    "hit_margin_ms",
    "layout",
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
    "requested_carriers",
    "active_carriers",
    "active_carrier_fraction",
    "dropped_for_hit_window",
    "dropped_for_chronology",
    "positive_new_unmatched",
    "block_span_fraction",
    "occupied_quartiles",
    "runtime_seconds",
]

FEATURE_FIELDS = [
    "experiment_version",
    "config_id",
    "message_seed",
    "pn_key",
    "pn_key_id",
    "layout_key",
    "layout_key_id",
    "hit_margin_ms",
    "layout",
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


def key_id(key: str | int) -> str:
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16]


def experiment_config_id(**parameters: object) -> str:
    material = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _read_rows_with_schema(
    path: Path,
    fields: list[str],
) -> list[dict[str, str]]:
    if not path.is_file():
        return []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != fields:
            raise ValueError(
                f"{path} ima nekompatibilnu šemu. Koristi novi output fajl; "
                "stari rezultati se namerno ne mešaju sa v2 eksperimentom."
            )
        return list(reader)


def replace_configs_rows(
    path: Path,
    fields: list[str],
    rows: list[dict],
) -> None:
    """Atomically replace rows for every config represented in ``rows``."""
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_rows_with_schema(path, fields)
    config_ids = {str(row["config_id"]) for row in rows}
    retained = [
        row
        for row in existing
        if row.get("config_id") not in config_ids
    ]

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in fields}
                for row in [*retained, *rows]
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def replace_config_rows(
    path: Path,
    fields: list[str],
    config_id: str,
    rows: list[dict],
) -> None:
    if any(str(row.get("config_id")) != config_id for row in rows):
        raise ValueError("Svi redovi moraju pripadati prosleđenom config_id-u.")
    replace_configs_rows(path, fields, rows)


def completed_result_keys(path: Path) -> set[str]:
    frame = pd.DataFrame(_read_rows_with_schema(path, RESULT_FIELDS))
    if frame.empty:
        return set()
    return set(
        frame.loc[frame["status"] == "OK", "config_id"].astype(str)
    )


def completed_feature_keys(path: Path) -> set[str]:
    frame = pd.DataFrame(_read_rows_with_schema(path, FEATURE_FIELDS))
    if frame.empty:
        return set()

    completed: set[str] = set()
    for config_id, group in frame.groupby("config_id"):
        counts = group["label"].astype(int).value_counts().to_dict()
        if counts == {0: 1, 1: 1}:
            completed.add(str(config_id))
    return completed


def validate_evaluation_rows(
    results: pd.DataFrame,
    features: pd.DataFrame,
    expected_configs: int,
) -> None:
    if results["config_id"].duplicated().any():
        raise ValueError("Result CSV sadrži dupliran config_id.")

    ok = results[results["status"] == "OK"]
    if len(ok) != expected_configs:
        errors = int(np.sum(results["status"] != "OK"))
        raise RuntimeError(
            f"Evaluacija zahteva {expected_configs} kompletnih konfiguracija; "
            f"dostupno OK={len(ok)}, ERROR={errors}."
        )

    counts = features.groupby("config_id")["label"].agg(
        lambda values: tuple(sorted(values.astype(int).tolist()))
    )
    if len(counts) != expected_configs or not all(
        labels == (0, 1) for labels in counts.tolist()
    ):
        raise ValueError(
            "Svaka OK konfiguracija mora imati tačno jedan clean i jedan stego red."
        )

    # Prefix placement ignores layout_key. Repeated prefix controls across
    # layout seeds must therefore remain physically identical.
    prefix = ok[ok["layout"] == "prefix"]
    invariant_columns = [
        "message_bits",
        "bit_errors_memory",
        "bit_errors_roundtrip",
        "requested_carriers",
        "active_carriers",
        "dropped_for_hit_window",
        "dropped_for_chronology",
        "positive_new_unmatched",
    ]
    grouped = prefix.groupby(["replay_file", "payload_fraction"])
    for keys, group in grouped:
        if any(group[column].nunique(dropna=False) != 1 for column in invariant_columns):
            raise ValueError(
                f"PREFIX kontrola zavisi od layout ključa za {keys}."
            )


def layout_coverage(
    num_notes: int,
    n_value: int,
    num_bits: int,
    layout: str,
    layout_key: str | int,
) -> tuple[float, int]:
    capacity = nominal_capacity_bits(num_notes, n_value)
    blocks = select_payload_blocks(
        num_notes=num_notes,
        n_frames_per_bit=n_value,
        num_bits=num_bits,
        layout_key=layout_key,
        layout=layout,
    )

    if len(blocks) == 0 or capacity == 0:
        return 0.0, 0

    temporal = np.sort(blocks)
    span = (
        (int(temporal[-1]) - int(temporal[0]) + 1)
        / capacity
    )

    quartiles = np.minimum(
        (4 * blocks) // capacity,
        3,
    )
    occupied_quartiles = len(np.unique(quartiles))

    return float(span), int(occupied_quartiles)


def build_physical_stego(
    context,
    message: np.ndarray,
    alpha: float,
    n_value: int,
    layout: str,
    pn_key: str | int,
    layout_key: str | int,
    hit_margin_ms: float,
    temp_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    original = context.original_residuals
    note_frames = context.note_frame_indices

    target_stego = embed_message_with_layout(
        note_residuals=original,
        message_bipolar=message,
        pn_key=pn_key,
        layout_key=layout_key,
        alpha=alpha,
        n_frames_per_bit=n_value,
        layout=layout,
        quantize=True,
    )

    requested_shifts = np.zeros(len(original), dtype=np.int64)

    base_usable = (
        (note_frames != -1)
        & ~np.isnan(original)
        & ~np.isnan(target_stego)
    )

    requested_shifts[base_usable] = np.rint(
        target_stego[base_usable] - original[base_usable]
    ).astype(np.int64)

    originally_requested = base_usable & (requested_shifts != 0)

    limit = max(
        0.0,
        context.hit_window_ms - hit_margin_ms,
    )

    hit_safe = np.zeros(len(original), dtype=bool)
    hit_safe[base_usable] = (
        np.abs(
            original[base_usable]
            + requested_shifts[base_usable]
        )
        <= limit
    )

    dropped_hit_window = int(
        np.sum(originally_requested & ~hit_safe)
    )

    requested_shifts[
        originally_requested & ~hit_safe
    ] = 0

    target_positions = np.flatnonzero(
        base_usable & (requested_shifts != 0)
    )
    target_frames = note_frames[target_positions]
    target_shifts = requested_shifts[target_positions]

    safe_shifts, dropped_chronology = make_chronology_safe_shifts(
        context.replay,
        target_frames,
        target_shifts,
    )

    active_mask = safe_shifts != 0
    written_shifts = safe_shifts[active_mask].astype(
        np.float64,
        copy=False,
    )

    physical_stego = original.copy()
    physical_stego[target_positions] = (
        original[target_positions] + safe_shifts
    )

    write_stego_replay(
        str(context.osr_path),
        str(temp_path),
        target_frames[active_mask],
        safe_shifts[active_mask],
    )

    stego_residuals, _, _ = load_residuals(
        temp_path,
        context.beatmap,
        context.offset_ms,
        context.hit_window_ms,
    )

    original_unmatched = int(np.sum(np.isnan(original)))
    stego_unmatched = int(np.sum(np.isnan(stego_residuals)))
    original_valid = ~np.isnan(original)
    stego_valid = ~np.isnan(stego_residuals)

    if len(written_shifts) > 0:
        squared_shifts = written_shifts**2
        absolute_shifts = np.abs(written_shifts)
        sum_squared_shift_ms2 = float(np.sum(squared_shifts))
        mean_squared_shift_ms2 = float(np.mean(squared_shifts))
        rms_applied_shift_ms = float(np.sqrt(mean_squared_shift_ms2))
        total_absolute_shift_ms = float(np.sum(absolute_shifts))
        mean_absolute_applied_shift_ms = float(np.mean(absolute_shifts))
    else:
        sum_squared_shift_ms2 = 0.0
        mean_squared_shift_ms2 = 0.0
        rms_applied_shift_ms = 0.0
        total_absolute_shift_ms = 0.0
        mean_absolute_applied_shift_ms = 0.0

    diagnostics = {
        "requested_carriers": int(np.sum(originally_requested)),
        "active_carriers": int(np.sum(active_mask)),
        "dropped_for_hit_window": dropped_hit_window,
        "dropped_for_chronology": int(dropped_chronology),
        "positive_new_unmatched": max(
            0,
            stego_unmatched - original_unmatched,
        ),
        "new_unmatched_events": int(
            np.sum(original_valid & ~stego_valid)
        ),
        "new_matched_events": int(
            np.sum(~original_valid & stego_valid)
        ),
        "changed_match_status_total": int(
            np.sum(original_valid != stego_valid)
        ),
        "sum_squared_shift_ms2": sum_squared_shift_ms2,
        "mean_squared_shift_ms2": mean_squared_shift_ms2,
        "rms_applied_shift_ms": rms_applied_shift_ms,
        "mean_absolute_applied_shift_ms": mean_absolute_applied_shift_ms,
        "total_absolute_shift_ms": total_absolute_shift_ms,
        **quarter_shift_diagnostics(
            num_notes=len(original),
            requested_positions=np.flatnonzero(originally_requested),
            active_positions=target_positions[active_mask],
            active_shifts_ms=written_shifts,
        ),
    }

    return physical_stego, stego_residuals, diagnostics


def run_one(
    context,
    message: np.ndarray,
    alpha: float,
    n_value: int,
    payload_fraction: float,
    layout: str,
    pn_key: str | int,
    layout_key: str | int,
    message_seed: int,
    config_id: str,
    hit_margin_ms: float,
    temp_dir: Path,
) -> tuple[dict, dict, dict]:
    start = time.perf_counter()

    temp_path = temp_dir / (
        f"{context.osr_path.stem}"
        f"_a{alpha:g}_n{n_value}"
        f"_p{int(round(payload_fraction * 1000))}"
        f"_{layout}.osr"
    )

    physical, roundtrip, diagnostics = build_physical_stego(
        context=context,
        message=message,
        alpha=alpha,
        n_value=n_value,
        layout=layout,
        pn_key=pn_key,
        layout_key=layout_key,
        hit_margin_ms=hit_margin_ms,
        temp_path=temp_path,
    )

    decoded_memory = extract_bipolar_message_with_layout(
        stego_note_residuals=physical,
        pn_key=pn_key,
        layout_key=layout_key,
        n_frames_per_bit=n_value,
        num_bits=len(message),
        layout=layout,
    )
    decoded_roundtrip = extract_bipolar_message_with_layout(
        stego_note_residuals=roundtrip,
        pn_key=pn_key,
        layout_key=layout_key,
        n_frames_per_bit=n_value,
        num_bits=len(message),
        layout=layout,
    )

    memory_errors = int(np.sum(decoded_memory != message))
    roundtrip_errors = int(np.sum(decoded_roundtrip != message))

    requested = diagnostics["requested_carriers"]
    active = diagnostics["active_carriers"]

    span, quartiles = layout_coverage(
        num_notes=len(context.original_residuals),
        n_value=n_value,
        num_bits=len(message),
        layout=layout,
        layout_key=layout_key,
    )

    result = {
        "status": "OK",
        "error": "",
        "experiment_version": EXPERIMENT_VERSION,
        "config_id": config_id,
        "message_seed": message_seed,
        "pn_key": str(pn_key),
        "pn_key_id": key_id(pn_key),
        "layout_key": str(layout_key),
        "layout_key_id": key_id(layout_key),
        "hit_margin_ms": hit_margin_ms,
        "layout": layout,
        "category": context.category,
        "replay_file": context.replay_file,
        "player": context.player,
        "beatmap_hash": context.beatmap_hash,
        "alpha": alpha,
        "n_frames_per_bit": n_value,
        "payload_fraction": payload_fraction,
        "nominal_capacity_bits": nominal_capacity_bits(
            len(context.original_residuals),
            n_value,
        ),
        "message_bits": len(message),
        "bit_errors_memory": memory_errors,
        "ber_memory": memory_errors / len(message),
        "bit_errors_roundtrip": roundtrip_errors,
        "ber_roundtrip": roundtrip_errors / len(message),
        "requested_carriers": requested,
        "active_carriers": active,
        "active_carrier_fraction": (
            active / requested if requested else 0.0
        ),
        "dropped_for_hit_window": diagnostics[
            "dropped_for_hit_window"
        ],
        "dropped_for_chronology": diagnostics[
            "dropped_for_chronology"
        ],
        "positive_new_unmatched": diagnostics[
            "positive_new_unmatched"
        ],
        "new_unmatched_events": diagnostics[
            "new_unmatched_events"
        ],
        "new_matched_events": diagnostics[
            "new_matched_events"
        ],
        "changed_match_status_total": diagnostics[
            "changed_match_status_total"
        ],
        "sum_squared_shift_ms2": diagnostics[
            "sum_squared_shift_ms2"
        ],
        "mean_squared_shift_ms2": diagnostics[
            "mean_squared_shift_ms2"
        ],
        "rms_applied_shift_ms": diagnostics[
            "rms_applied_shift_ms"
        ],
        "mean_absolute_applied_shift_ms": diagnostics[
            "mean_absolute_applied_shift_ms"
        ],
        "total_absolute_shift_ms": diagnostics[
            "total_absolute_shift_ms"
        ],
        "block_span_fraction": span,
        "occupied_quartiles": quartiles,
        "runtime_seconds": time.perf_counter() - start,
    }

    clean_stats = residual_features(context.original_residuals)
    stego_stats = residual_features(roundtrip)

    common = {
        "experiment_version": EXPERIMENT_VERSION,
        "config_id": config_id,
        "message_seed": message_seed,
        "pn_key": str(pn_key),
        "pn_key_id": key_id(pn_key),
        "layout_key": str(layout_key),
        "layout_key_id": key_id(layout_key),
        "hit_margin_ms": hit_margin_ms,
        "layout": layout,
        "category": context.category,
        "replay_file": context.replay_file,
        "player": context.player,
        "beatmap_hash": context.beatmap_hash,
        "alpha": alpha,
        "n_frames_per_bit": n_value,
        "payload_fraction": payload_fraction,
        "message_bits": len(message),
    }

    clean_feature = {
        **common,
        **clean_stats,
        "label": 0,
    }
    stego_feature = {
        **common,
        **stego_stats,
        "label": 1,
    }

    return result, clean_feature, stego_feature


def bootstrap_weighted_ber_delta(
    paired: pd.DataFrame,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(paired)
    deltas = np.empty(iterations, dtype=np.float64)

    for index in range(iterations):
        sample = paired.iloc[rng.integers(0, n, size=n)]

        prefix_ber = (
            sample["errors_prefix"].sum()
            / sample["message_bits"].sum()
        )
        distributed_ber = (
            sample["errors_distributed"].sum()
            / sample["message_bits"].sum()
        )
        deltas[index] = distributed_ber - prefix_ber

    return (
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )


def bootstrap_auc_delta(
    prefix_predictions: pd.DataFrame,
    distributed_predictions: pd.DataFrame,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    merged = prefix_predictions[
        ["replay_file", "label", "oof_probability_stego"]
    ].merge(
        distributed_predictions[
            ["replay_file", "label", "oof_probability_stego"]
        ],
        on=["replay_file", "label"],
        suffixes=("_prefix", "_distributed"),
        validate="one_to_one",
    )

    groups = merged["replay_file"].astype(str).unique()
    grouped_indices = {
        group: np.flatnonzero(
            merged["replay_file"].astype(str).to_numpy() == group
        )
        for group in groups
    }

    rng = np.random.default_rng(seed)
    deltas: list[float] = []

    for _ in range(iterations):
        sampled_groups = rng.choice(
            groups,
            size=len(groups),
            replace=True,
        )
        indices = np.concatenate(
            [grouped_indices[group] for group in sampled_groups]
        )

        sample = merged.iloc[indices]
        y = sample["label"].to_numpy(dtype=np.int8)

        if len(np.unique(y)) < 2:
            continue

        auc_prefix = roc_auc_score(
            y,
            sample["oof_probability_stego_prefix"],
        )
        auc_distributed = roc_auc_score(
            y,
            sample["oof_probability_stego_distributed"],
        )
        deltas.append(float(auc_distributed - auc_prefix))

    if not deltas:
        raise RuntimeError("Bootstrap AUC delta nije proizveo validne uzorke.")

    return (
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )


def evaluate(
    results: pd.DataFrame,
    features: pd.DataFrame,
    payloads: list[float],
    alpha: float,
    n_value: int,
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    auc_rows: list[dict] = []
    predictions: dict[tuple[str, str, str, float], pd.DataFrame] = {}

    layout_key_ids = sorted(features["layout_key_id"].astype(str).unique())
    layout_key_by_id = {
        str(row.layout_key_id): str(row.layout_key)
        for row in features[
            ["layout_key_id", "layout_key"]
        ].drop_duplicates().itertuples(index=False)
    }

    for layout_key_id in layout_key_ids:
        key_features = features[
            features["layout_key_id"].astype(str) == layout_key_id
        ]
        for payload in payloads:
            payload_features = key_features[
                np.isclose(key_features["payload_fraction"], payload)
            ]

            for cv_group_column in CV_GROUP_COLUMNS:
                for layout in LAYOUTS:
                    subset = payload_features[
                        payload_features["layout"] == layout
                    ].copy()

                    summary, prediction = evaluate_subset(
                        frame=subset,
                        scope="ALL",
                        alpha=alpha,
                        n_value=n_value,
                        folds=folds,
                        trees=trees,
                        seed=seed,
                        bootstrap_iterations=bootstrap_iterations,
                        cv_group_column=cv_group_column,
                    )

                    summary["layout"] = layout
                    summary["experiment_version"] = EXPERIMENT_VERSION
                    summary["message_seed"] = seed
                    summary["pn_key"] = str(subset["pn_key"].iloc[0])
                    summary["pn_key_id"] = str(subset["pn_key_id"].iloc[0])
                    summary["layout_key"] = layout_key_by_id[layout_key_id]
                    summary["layout_key_id"] = layout_key_id
                    summary["payload_fraction"] = payload
                    summary["classifier_trees"] = trees
                    summary["bootstrap_iterations"] = bootstrap_iterations
                    auc_rows.append(summary)
                    predictions[
                        (cv_group_column, layout_key_id, layout, payload)
                    ] = prediction

    paired_rows: list[dict] = []

    ok = results[
        (results["status"] == "OK")
        & np.isclose(results["alpha"], alpha)
        & (results["n_frames_per_bit"] == n_value)
    ].copy()

    for layout_key_id in layout_key_ids:
        key_results = ok[
            ok["layout_key_id"].astype(str) == layout_key_id
        ]
        for payload in payloads:
            prefix = key_results[
                (key_results["layout"] == "prefix")
                & np.isclose(key_results["payload_fraction"], payload)
            ][
            [
                "replay_file",
                "message_bits",
                "bit_errors_roundtrip",
                "block_span_fraction",
                "occupied_quartiles",
            ]
            ].rename(
                columns={
                    "bit_errors_roundtrip": "errors_prefix",
                    "block_span_fraction": "span_prefix",
                    "occupied_quartiles": "quartiles_prefix",
                }
            )

            distributed = key_results[
                (key_results["layout"] == "distributed")
                & np.isclose(key_results["payload_fraction"], payload)
            ][
            [
                "replay_file",
                "message_bits",
                "bit_errors_roundtrip",
                "block_span_fraction",
                "occupied_quartiles",
            ]
            ].rename(
                columns={
                    "message_bits": "message_bits_distributed",
                    "bit_errors_roundtrip": "errors_distributed",
                    "block_span_fraction": "span_distributed",
                    "occupied_quartiles": "quartiles_distributed",
                }
            )

            paired = prefix.merge(
                distributed,
                on="replay_file",
                how="inner",
                validate="one_to_one",
            )

            if not np.array_equal(
                paired["message_bits"].to_numpy(),
                paired["message_bits_distributed"].to_numpy(),
            ):
                raise ValueError(
                    "PREFIX i DISTRIBUTED nemaju isti broj bitova po replay-u."
                )

            total_bits = int(paired["message_bits"].sum())
            ber_prefix = paired["errors_prefix"].sum() / total_bits
            ber_distributed = paired["errors_distributed"].sum() / total_bits

            ber_ci_low, ber_ci_high = bootstrap_weighted_ber_delta(
                paired,
                stable_seed(seed, layout_key_id, payload, "paired-ber"),
                bootstrap_iterations,
            )

            for cv_group_column in CV_GROUP_COLUMNS:
                auc_prefix_row = next(
                    row for row in auc_rows
                    if row["layout"] == "prefix"
                    and row["layout_key_id"] == layout_key_id
                    and row["cv_group_column"] == cv_group_column
                    and np.isclose(row["payload_fraction"], payload)
                )
                auc_distributed_row = next(
                    row for row in auc_rows
                    if row["layout"] == "distributed"
                    and row["layout_key_id"] == layout_key_id
                    and row["cv_group_column"] == cv_group_column
                    and np.isclose(row["payload_fraction"], payload)
                )

                auc_delta = (
                    auc_distributed_row["roc_auc"]
                    - auc_prefix_row["roc_auc"]
                )
                paired_auc_seed = (
                    stable_seed(seed, layout_key_id, payload, "paired-auc")
                    if cv_group_column == "replay_file"
                    else stable_seed(
                        seed,
                        layout_key_id,
                        payload,
                        cv_group_column,
                        "paired-auc",
                    )
                )
                auc_ci_low, auc_ci_high = bootstrap_auc_delta(
                    predictions[
                        (cv_group_column, layout_key_id, "prefix", payload)
                    ],
                    predictions[
                        (cv_group_column, layout_key_id, "distributed", payload)
                    ],
                    paired_auc_seed,
                    bootstrap_iterations,
                )

                paired_rows.append(
                    {
                        "experiment_version": EXPERIMENT_VERSION,
                        "message_seed": seed,
                        "pn_key": str(features["pn_key"].iloc[0]),
                        "pn_key_id": str(features["pn_key_id"].iloc[0]),
                        "layout_key": layout_key_by_id[layout_key_id],
                        "layout_key_id": layout_key_id,
                        "cv_group_column": cv_group_column,
                        "cv_groups": int(auc_prefix_row["cv_groups"]),
                        "classifier_trees": trees,
                        "folds": folds,
                        "bootstrap_iterations": bootstrap_iterations,
                        "alpha": alpha,
                        "n_frames_per_bit": n_value,
                        "payload_fraction": payload,
                        "replay_pairs": len(paired),
                        "total_bits": total_bits,
                        "ber_prefix": ber_prefix,
                        "ber_distributed": ber_distributed,
                        "delta_ber_distributed_minus_prefix": (
                            ber_distributed - ber_prefix
                        ),
                        "delta_ber_ci95_low": ber_ci_low,
                        "delta_ber_ci95_high": ber_ci_high,
                        "auc_prefix": auc_prefix_row["roc_auc"],
                        "auc_distributed": auc_distributed_row["roc_auc"],
                        "delta_auc_distributed_minus_prefix": auc_delta,
                        "delta_auc_ci95_low": auc_ci_low,
                        "delta_auc_ci95_high": auc_ci_high,
                        "mean_span_prefix": paired["span_prefix"].mean(),
                        "mean_span_distributed": paired["span_distributed"].mean(),
                        "mean_quartiles_prefix": paired["quartiles_prefix"].mean(),
                        "mean_quartiles_distributed": paired[
                            "quartiles_distributed"
                        ].mean(),
                    }
                )

    return pd.DataFrame(auc_rows), pd.DataFrame(paired_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled prefix vs distributed payload experiment."
    )

    payload_dir = RESULTS_DIR / "payload"

    parser.add_argument(
        "--results",
        type=Path,
        default=payload_dir / "fine_pilot_selected_replays.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Opcioni deterministički prefix kohorte za smoke/pilot proveru.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
    )
    parser.add_argument(
        "--map-offsets",
        type=Path,
        default=CONFIG_DIR / "map_time_offsets.json",
    )
    parser.add_argument("--alpha", type=float, default=14.0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument(
        "--payloads",
        nargs="+",
        type=float,
        default=list(DEFAULT_PAYLOADS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pn-key", default=DEFAULT_PN_KEY)
    parser.add_argument(
        "--layout-keys",
        nargs="+",
        default=list(DEFAULT_LAYOUT_KEYS),
        help="Nezavisni layout ključevi; PN ključ i poruka ostaju fiksni.",
    )
    parser.add_argument("--hit-margin-ms", type=float, default=5.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)

    parser.add_argument(
        "--output",
        type=Path,
        default=payload_dir / "layout_multiseed_results.csv",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=payload_dir / "layout_multiseed_features.csv",
    )
    parser.add_argument(
        "--auc-output",
        type=Path,
        default=payload_dir / "layout_multiseed_auc.csv",
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=payload_dir / "layout_multiseed_paired.csv",
    )

    args = parser.parse_args()
    payloads = sorted(set(float(value) for value in args.payloads))
    layout_keys = list(dict.fromkeys(str(value) for value in args.layout_keys))
    if not layout_keys:
        raise ValueError("Potreban je najmanje jedan layout ključ.")

    source = pd.read_csv(args.results)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit mora biti > 0.")
        source = source.head(args.limit).copy()
    beatmaps = build_beatmap_index(args.dataset_dir)
    offsets = load_map_offsets(args.map_offsets)

    completed_results = completed_result_keys(args.output)
    completed_features = completed_feature_keys(args.features)
    completed = completed_results & completed_features

    total = len(source) * len(payloads) * len(LAYOUTS) * len(layout_keys)
    counter = 0
    new_ok = 0

    print("=" * 100)
    print("PREFIX vs DISTRIBUTED")
    print("=" * 100)
    print(f"Replay-eva: {len(source)}")
    print(f"alpha/N:    {args.alpha:g}/{args.n}")
    print(f"payloads:   {payloads}")
    print(f"PN key id:  {key_id(args.pn_key)}")
    print(f"layout ids: {[key_id(value) for value in layout_keys]}")
    print(f"attempts:   {total}")
    print()

    with tempfile.TemporaryDirectory(
        prefix="osu_layout_comparison_"
    ) as temp_name:
        temp_dir = Path(temp_name)

        for replay_index, row in enumerate(
            source.itertuples(index=False),
            1,
        ):
            context = prepare_replay(
                row,
                args.dataset_dir,
                beatmaps,
                offsets,
            )
            replay_result_rows: list[dict] = []
            replay_feature_rows: list[dict] = []

            capacity = nominal_capacity_bits(
                len(context.original_residuals),
                args.n,
            )

            for payload in payloads:
                num_bits = message_length_for_fraction(
                    capacity,
                    payload,
                )

                message = deterministic_message(
                    replay_file=context.replay_file,
                    beatmap_hash=context.beatmap_hash,
                    alpha=args.alpha,
                    n_value=args.n,
                    payload_fraction=payload,
                    num_bits=num_bits,
                    seed=args.seed,
                )

                for layout_key in layout_keys:
                    for layout in LAYOUTS:
                        counter += 1
                        config_id = experiment_config_id(
                            experiment_version=EXPERIMENT_VERSION,
                            replay_file=context.replay_file,
                            beatmap_hash=context.beatmap_hash,
                            offset_ms=context.offset_ms,
                            hit_window_ms=context.hit_window_ms,
                            layout=layout,
                            alpha=float(args.alpha),
                            n_frames_per_bit=int(args.n),
                            payload_fraction=round(payload, 9),
                            message_bits=num_bits,
                            message_seed=int(args.seed),
                            pn_key_id=key_id(args.pn_key),
                            layout_key_id=key_id(layout_key),
                            hit_margin_ms=float(args.hit_margin_ms),
                        )

                        if config_id in completed:
                            continue

                        try:
                            result, clean_feature, stego_feature = run_one(
                                context=context,
                                message=message,
                                alpha=args.alpha,
                                n_value=args.n,
                                payload_fraction=payload,
                                layout=layout,
                                pn_key=args.pn_key,
                                layout_key=layout_key,
                                message_seed=args.seed,
                                config_id=config_id,
                                hit_margin_ms=args.hit_margin_ms,
                                temp_dir=temp_dir,
                            )

                            replay_result_rows.append(result)
                            replay_feature_rows.extend(
                                [clean_feature, stego_feature]
                            )
                            new_ok += 1

                        except Exception as exc:
                            replay_result_rows.append(
                                {
                                    "status": "ERROR",
                                    "error": repr(exc),
                                    "experiment_version": EXPERIMENT_VERSION,
                                    "config_id": config_id,
                                    "message_seed": args.seed,
                                    "pn_key": str(args.pn_key),
                                    "pn_key_id": key_id(args.pn_key),
                                    "layout_key": str(layout_key),
                                    "layout_key_id": key_id(layout_key),
                                    "hit_margin_ms": args.hit_margin_ms,
                                    "layout": layout,
                                    "category": context.category,
                                    "replay_file": context.replay_file,
                                    "player": context.player,
                                    "beatmap_hash": context.beatmap_hash,
                                    "alpha": args.alpha,
                                    "n_frames_per_bit": args.n,
                                    "payload_fraction": payload,
                                }
                            )
                            print(
                                f"ERROR {context.replay_file[:12]} "
                                f"{layout}/{key_id(layout_key)} "
                                f"p={payload}: {exc!r}"
                            )

            # One atomic rewrite per replay avoids per-config O(n^2) CSV I/O.
            # If interrupted between the two files, the next resume reruns and
            # replaces precisely the incomplete config ids.
            replace_configs_rows(
                args.output,
                RESULT_FIELDS,
                replay_result_rows,
            )
            replace_configs_rows(
                args.features,
                FEATURE_FIELDS,
                replay_feature_rows,
            )

            if replay_index % 10 == 0:
                print(
                    f"{replay_index:02d}/{len(source)} replay-eva | "
                    f"new OK={new_ok}"
                )

    results = pd.read_csv(args.output)
    features = pd.read_csv(args.features)

    requested_layout_ids = {key_id(value) for value in layout_keys}
    common_filter = (
        (results["experiment_version"] == EXPERIMENT_VERSION)
        & (results["message_seed"] == args.seed)
        & (results["pn_key_id"] == key_id(args.pn_key))
        & results["layout_key_id"].isin(requested_layout_ids)
        & np.isclose(results["hit_margin_ms"], args.hit_margin_ms)
        & np.isclose(results["alpha"], args.alpha)
        & (results["n_frames_per_bit"] == args.n)
        & results["payload_fraction"].apply(
            lambda value: any(np.isclose(value, payload) for payload in payloads)
        )
    )
    results = results[common_filter].copy()
    feature_filter = (
        (features["experiment_version"] == EXPERIMENT_VERSION)
        & (features["message_seed"] == args.seed)
        & (features["pn_key_id"] == key_id(args.pn_key))
        & features["layout_key_id"].isin(requested_layout_ids)
        & np.isclose(features["hit_margin_ms"], args.hit_margin_ms)
        & np.isclose(features["alpha"], args.alpha)
        & (features["n_frames_per_bit"] == args.n)
        & features["payload_fraction"].apply(
            lambda value: any(np.isclose(value, payload) for payload in payloads)
        )
    )
    features = features[feature_filter].copy()

    validate_evaluation_rows(
        results=results,
        features=features,
        expected_configs=total,
    )

    auc, paired = evaluate(
        results=results,
        features=features,
        payloads=payloads,
        alpha=args.alpha,
        n_value=args.n,
        folds=args.folds,
        trees=args.trees,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )

    auc.to_csv(args.auc_output, index=False)
    paired.to_csv(args.paired_output, index=False)

    print("\n" + "=" * 100)
    print("PAIRED RESULT")
    print("=" * 100)
    print(
        paired[
            [
                "layout_key_id",
                "cv_group_column",
                "payload_fraction",
                "ber_prefix",
                "ber_distributed",
                "delta_ber_distributed_minus_prefix",
                "auc_prefix",
                "auc_distributed",
                "delta_auc_distributed_minus_prefix",
                "mean_span_prefix",
                "mean_span_distributed",
            ]
        ].to_string(index=False)
    )

    print("\nInterpretation:")
    print("  delta BER < 0  => distributed ima manji BER")
    print("  delta AUC < 0  => distributed je teže detektovati")
    print("  Ako 95% CI za delta uključuje 0, pilot nije dao jasan dokaz razlike.")

    print("\nFajlovi:")
    print(f"  {args.output}")
    print(f"  {args.features}")
    print(f"  {args.auc_output}")
    print(f"  {args.paired_output}")


if __name__ == "__main__":
    main()
