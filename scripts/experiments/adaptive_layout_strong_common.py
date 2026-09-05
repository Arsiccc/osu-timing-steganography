"""Physical generation and schemas for adaptive-layout strong steganalysis."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.analysis.layout_diagnostics import QUARTERS
from osu_stego.analysis.timing_features import FEATURE_VERSION, FULL_FEATURES, residual_timing_features
from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import (
    alpha_from_sender_local_features,
    load_sender_local_policy,
)
from osu_stego.stego.payload_layout import extract_bipolar_message_with_layout
from osu_stego.stego.pn_sequence import generate_pn_sequence
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_layout_comparison import (
    build_physical_stego,
    experiment_config_id,
    key_id,
    layout_coverage,
)
from scripts.experiments.run_payload_sweep import (
    deterministic_message,
    message_length_for_fraction,
    nominal_capacity_bits,
)
from scripts.experiments.run_pilot_ber_sweep import (
    build_beatmap_index,
    load_map_offsets,
    prepare_replay,
)
from scripts.experiments.sender_local_common import load_partitioned_cohort


EXPERIMENT_VERSION = "adaptive-layout-strong-v1"
N_VALUE = 8
PAYLOAD_FRACTION = 0.075
MESSAGE_SEED = 42
PN_KEY = "adaptive-alpha-pn-v1"
PREFIX_LAYOUT_KEY = "prefix-layout-unused-v1"
LAYOUT_KEYS = tuple(f"adaptive-layout-seed-{index}" for index in range(5))
ALPHA_METHODS = ("fixed_15", "sender_local_adaptive")
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "adaptive_layout_strong_v1"
DEFAULT_POLICY = CONFIG_DIR / "adaptive_alpha_sender_local_v1.json"
DEFAULT_PARTITION = RESULTS_DIR / "adaptive_alpha_validation_v1" / "heldout_partition.csv"
STRONG_SOURCE_DIR = RESULTS_DIR / "strong_steganalysis_v1"
SENDER_SOURCE_DIR = RESULTS_DIR / "sender_local_adaptive_v1"

QUARTER_RESULT_FIELDS = [
    field
    for quarter in QUARTERS
    for field in (
        f"requested_carriers_q{quarter}",
        f"active_carriers_q{quarter}",
        f"sum_squared_shift_ms2_q{quarter}",
        f"total_absolute_shift_ms_q{quarter}",
        f"energy_fraction_q{quarter}",
    )
]

PROVENANCE_FIELDS = [
    "policy_sha256",
    "replay_sha256",
    "cohort_sha256",
    "performance_groups_sha256",
    "partition_sha256",
    "map_offsets_sha256",
    "payload_layout_sha256",
    "timing_features_sha256",
]

COMMON_FIELDS = [
    "experiment_version",
    "config_id",
    "method_config_hash",
    "partition",
    "alpha_method",
    "layout",
    "layout_seed",
    "layout_key_id",
    "message_seed",
    "message_sha256",
    "pn_key_id",
    "pn_sequence_sha256",
    "replay_file",
    "player",
    "beatmap_hash",
    "performance_category",
    "alpha",
    "n_frames_per_bit",
    "payload_fraction",
    "message_bits",
] + PROVENANCE_FIELDS

RESULT_FIELDS = [
    "status",
    "error",
] + COMMON_FIELDS + [
    "nominal_capacity_bits",
    "bit_errors_memory",
    "ber_memory",
    "bit_errors_roundtrip",
    "ber_roundtrip",
    "requested_carriers",
    "active_carriers",
    "active_carrier_fraction",
    "dropped_for_hit_window",
    "dropped_for_chronology",
    "new_unmatched_events",
    "new_matched_events",
    "changed_match_status_total",
    "sum_squared_shift_ms2",
    "mean_squared_shift_ms2",
    "rms_applied_shift_ms",
    "mean_absolute_applied_shift_ms",
    "total_absolute_shift_ms",
    "block_span_fraction",
    "occupied_quartiles",
] + QUARTER_RESULT_FIELDS + ["runtime_seconds"]

FEATURE_FIELDS = COMMON_FIELDS + [
    "feature_version",
    "label",
    "num_notes",
    "n_valid_residuals",
    "valid_residual_fraction",
] + list(FULL_FEATURES)


def message_sha256(message: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(message, dtype=np.int8).tobytes()).hexdigest()


def pn_sha256(num_bits: int) -> str:
    chips = generate_pn_sequence(PN_KEY, num_bits * N_VALUE)
    return hashlib.sha256(chips.tobytes()).hexdigest()


def method_configs() -> list[dict[str, str]]:
    configs: list[dict[str, str]] = []
    for alpha_method in ALPHA_METHODS:
        configs.append(
            {
                "alpha_method": alpha_method,
                "layout": "prefix",
                "layout_seed": "shared_prefix",
                "layout_key": PREFIX_LAYOUT_KEY,
            }
        )
        for seed_index, layout_key in enumerate(LAYOUT_KEYS):
            configs.append(
                {
                    "alpha_method": alpha_method,
                    "layout": "distributed",
                    "layout_seed": str(seed_index),
                    "layout_key": layout_key,
                }
            )
    return configs


def experiment_config(
    *,
    policy_path: Path,
    results_path: Path,
    performance_path: Path,
    partition_path: Path,
    map_offsets_path: Path,
    limit_per_partition: int | None,
    trees: int,
    folds: int,
    bootstrap_iterations: int,
) -> dict:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "scientific_status": "five_layout_keys_frozen_before_outcomes",
        "policy_sha256": file_sha256(policy_path),
        "N": N_VALUE,
        "payload_fraction": PAYLOAD_FRACTION,
        "message_seed": MESSAGE_SEED,
        "message_convention": "stable_seed(42,replay_file,beatmap_hash,payload-message)",
        "pn_key_id": key_id(PN_KEY),
        "prefix_layout_key_id": key_id(PREFIX_LAYOUT_KEY),
        "distributed_layout_keys": [
            {"seed": index, "key_id": key_id(value)}
            for index, value in enumerate(LAYOUT_KEYS)
        ],
        "feature_version": FEATURE_VERSION,
        "feature_sets": {
            "baseline": [
                "variance",
                "excess_kurtosis",
                "autocorrelation_lag1",
            ],
            "position_only": [
                name for name in FULL_FEATURES if name.startswith("quarter_")
            ],
            "full": list(FULL_FEATURES),
        },
        "rf": {
            "n_estimators": trees,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "n_jobs": 1,
            "folds": folds,
            "seed": MESSAGE_SEED,
        },
        "bootstrap_iterations": bootstrap_iterations,
        "limit_per_partition": limit_per_partition,
        "input_checksums_sha256": {
            "policy": file_sha256(policy_path),
            "cohort": file_sha256(results_path),
            "performance_groups": file_sha256(performance_path),
            "partition": file_sha256(partition_path),
            "map_offsets": file_sha256(map_offsets_path),
            "payload_layout_code": file_sha256(
                Path("osu_stego/stego/payload_layout.py")
            ),
            "timing_features_code": file_sha256(
                Path("osu_stego/analysis/timing_features.py")
            ),
        },
    }


def write_or_validate_config(path: Path, config: dict) -> None:
    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError("Postojeći config.json pripada drugom eksperimentu.")
        return
    path.write_text(serialized, encoding="utf-8")


def _read_rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != fields:
            raise ValueError(f"{path} ima nekompatibilnu šemu.")
        return list(reader)


def replace_config_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    ids = {str(row["config_id"]) for row in rows}
    existing = [row for row in _read_rows(path, fields) if row["config_id"] not in ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in fields}
                for row in [*existing, *rows]
            )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def completed_configs(result_path: Path, feature_path: Path) -> set[str]:
    results = pd.DataFrame(_read_rows(result_path, RESULT_FIELDS))
    features = pd.DataFrame(_read_rows(feature_path, FEATURE_FIELDS))
    if results.empty or features.empty:
        return set()
    ok = set(results.loc[results["status"] == "OK", "config_id"].astype(str))
    labels = features.groupby("config_id")["label"].apply(
        lambda values: tuple(sorted(values.astype(int).tolist()))
    )
    return ok & set(labels[labels == (0, 1)].index.astype(str))


def _source_prefix_lookups(partition_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    strong = pd.read_csv(STRONG_SOURCE_DIR / "features.csv")
    strong = strong[strong["partition"] == partition_name].set_index(
        ["method", "replay_file", "label"], drop=False
    )
    physical = pd.read_csv(
        SENDER_SOURCE_DIR / f"{partition_name}_alpha_grid_results.csv"
    ).set_index(["replay_file", "alpha"], drop=False)
    return strong, physical


def _validate_prefix_against_frozen(
    *,
    alpha_method: str,
    replay_file: str,
    alpha: float,
    result: dict,
    clean_features: dict,
    stego_features: dict,
    strong_lookup: pd.DataFrame,
    physical_lookup: pd.DataFrame,
) -> None:
    expected_method = alpha_method
    expected_clean = strong_lookup.loc[(expected_method, replay_file, 0)]
    expected_stego = strong_lookup.loc[(expected_method, replay_file, 1)]
    for name in FULL_FEATURES:
        if not np.isclose(float(clean_features[name]), float(expected_clean[name]), atol=1e-12):
            raise ValueError(f"PREFIX clean {name} nije frozen-ekvivalentan.")
        if not np.isclose(float(stego_features[name]), float(expected_stego[name]), atol=1e-12):
            raise ValueError(f"PREFIX stego {name} nije frozen-ekvivalentan.")
    expected_result = physical_lookup.loc[(replay_file, alpha)]
    for name in (
        "bit_errors_roundtrip",
        "requested_carriers",
        "active_carriers",
        "dropped_for_hit_window",
        "dropped_for_chronology",
        "new_unmatched_events",
        "new_matched_events",
        "changed_match_status_total",
        "sum_squared_shift_ms2",
        "total_absolute_shift_ms",
    ):
        if not np.isclose(float(result[name]), float(expected_result[name])):
            raise ValueError(f"PREFIX physical {name} nije frozen-ekvivalentan.")


def generate_partition(
    *,
    partition_name: str,
    cohort: pd.DataFrame,
    config: dict,
    policy_path: Path,
    dataset_dir: Path,
    map_offsets_path: Path,
    result_path: Path,
    feature_path: Path,
) -> int:
    policy = load_sender_local_policy(policy_path)
    beatmaps = build_beatmap_index(dataset_dir)
    offsets = load_map_offsets(map_offsets_path)
    completed = completed_configs(result_path, feature_path)
    strong_lookup, physical_lookup = _source_prefix_lookups(partition_name)
    provenance = {
        "policy_sha256": config["policy_sha256"],
        "cohort_sha256": config["input_checksums_sha256"]["cohort"],
        "performance_groups_sha256": config["input_checksums_sha256"][
            "performance_groups"
        ],
        "partition_sha256": config["input_checksums_sha256"]["partition"],
        "map_offsets_sha256": config["input_checksums_sha256"]["map_offsets"],
        "payload_layout_sha256": config["input_checksums_sha256"][
            "payload_layout_code"
        ],
        "timing_features_sha256": config["input_checksums_sha256"][
            "timing_features_code"
        ],
    }
    new_ok = 0
    pending_results: list[dict] = []
    pending_features: list[dict] = []
    requested_ids: set[str] = set()
    with tempfile.TemporaryDirectory(prefix=f"osu_adaptive_layout_{partition_name}_") as name:
        temp_dir = Path(name)
        for replay_index, row in enumerate(cohort.itertuples(index=False), 1):
            context = prepare_replay(row, dataset_dir, beatmaps, offsets)
            replay_sha = file_sha256(context.osr_path)
            # Reproduce the frozen sender-local experiment exactly.  Its policy
            # assignments were made from the checksummed cohort accuracy column.
            # Recomputing the mathematically equivalent ratio from the .osr can
            # differ by one ULP at a threshold and would change two assignments.
            adaptive_alpha = alpha_from_sender_local_features(
                policy,
                accuracy=float(row.accuracy),
            )
            capacity = nominal_capacity_bits(len(context.original_residuals), N_VALUE)
            num_bits = message_length_for_fraction(capacity, PAYLOAD_FRACTION)
            message = deterministic_message(
                replay_file=context.replay_file,
                beatmap_hash=context.beatmap_hash,
                alpha=0.0,
                n_value=N_VALUE,
                payload_fraction=PAYLOAD_FRACTION,
                num_bits=num_bits,
                seed=MESSAGE_SEED,
            )
            message_hash = message_sha256(message)
            pn_hash = pn_sha256(num_bits)
            clean_features = residual_timing_features(context.original_residuals)
            for method in method_configs():
                alpha = 15.0 if method["alpha_method"] == "fixed_15" else adaptive_alpha
                method_hash = experiment_config_id(
                    alpha_method=method["alpha_method"],
                    layout=method["layout"],
                    layout_seed=method["layout_seed"],
                    layout_key_id=key_id(method["layout_key"]),
                    policy_sha256=config["policy_sha256"],
                    N=N_VALUE,
                    payload_fraction=PAYLOAD_FRACTION,
                )
                config_id = experiment_config_id(
                    experiment_version=EXPERIMENT_VERSION,
                    partition=partition_name,
                    replay_file=context.replay_file,
                    replay_sha256=replay_sha,
                    beatmap_hash=context.beatmap_hash,
                    alpha=alpha,
                    method_config_hash=method_hash,
                    message_sha256=message_hash,
                    pn_sequence_sha256=pn_hash,
                    offset_ms=context.offset_ms,
                    hit_window_ms=context.hit_window_ms,
                    hit_margin_ms=float(policy["hit_margin_ms"]),
                    **provenance,
                )
                requested_ids.add(config_id)
                if config_id in completed:
                    continue
                start = time.perf_counter()
                try:
                    physical, roundtrip, diagnostics = build_physical_stego(
                        context=context,
                        message=message,
                        alpha=alpha,
                        n_value=N_VALUE,
                        layout=method["layout"],
                        pn_key=PN_KEY,
                        layout_key=method["layout_key"],
                        hit_margin_ms=float(policy["hit_margin_ms"]),
                        temp_path=temp_dir / f"{config_id}.osr",
                    )
                    memory_decoded = extract_bipolar_message_with_layout(
                        physical,
                        PN_KEY,
                        method["layout_key"],
                        N_VALUE,
                        num_bits,
                        method["layout"],
                    )
                    roundtrip_decoded = extract_bipolar_message_with_layout(
                        roundtrip,
                        PN_KEY,
                        method["layout_key"],
                        N_VALUE,
                        num_bits,
                        method["layout"],
                    )
                    memory_errors = int(np.sum(memory_decoded != message))
                    roundtrip_errors = int(np.sum(roundtrip_decoded != message))
                    stego_features = residual_timing_features(roundtrip)
                    span, occupied = layout_coverage(
                        len(context.original_residuals),
                        N_VALUE,
                        num_bits,
                        method["layout"],
                        method["layout_key"],
                    )
                    common = {
                        "experiment_version": EXPERIMENT_VERSION,
                        "config_id": config_id,
                        "method_config_hash": method_hash,
                        "partition": partition_name,
                        "alpha_method": method["alpha_method"],
                        "layout": method["layout"],
                        "layout_seed": method["layout_seed"],
                        "layout_key_id": key_id(method["layout_key"]),
                        "message_seed": MESSAGE_SEED,
                        "message_sha256": message_hash,
                        "pn_key_id": key_id(PN_KEY),
                        "pn_sequence_sha256": pn_hash,
                        "replay_file": context.replay_file,
                        "player": context.player,
                        "beatmap_hash": context.beatmap_hash,
                        "performance_category": str(row.performance_category),
                        "alpha": alpha,
                        "n_frames_per_bit": N_VALUE,
                        "payload_fraction": PAYLOAD_FRACTION,
                        "message_bits": num_bits,
                        "replay_sha256": replay_sha,
                        **provenance,
                    }
                    result = {
                        "status": "OK",
                        "error": "",
                        **common,
                        "nominal_capacity_bits": capacity,
                        "bit_errors_memory": memory_errors,
                        "ber_memory": memory_errors / num_bits,
                        "bit_errors_roundtrip": roundtrip_errors,
                        "ber_roundtrip": roundtrip_errors / num_bits,
                        "active_carrier_fraction": (
                            diagnostics["active_carriers"] / diagnostics["requested_carriers"]
                            if diagnostics["requested_carriers"]
                            else 0.0
                        ),
                        "block_span_fraction": span,
                        "occupied_quartiles": occupied,
                        "runtime_seconds": time.perf_counter() - start,
                        **diagnostics,
                    }
                    clean_row = {
                        **common,
                        "feature_version": FEATURE_VERSION,
                        "label": 0,
                        **clean_features,
                    }
                    stego_row = {
                        **common,
                        "feature_version": FEATURE_VERSION,
                        "label": 1,
                        **stego_features,
                    }
                    if method["layout"] == "prefix":
                        _validate_prefix_against_frozen(
                            alpha_method=method["alpha_method"],
                            replay_file=context.replay_file,
                            alpha=alpha,
                            result=result,
                            clean_features=clean_features,
                            stego_features=stego_features,
                            strong_lookup=strong_lookup,
                            physical_lookup=physical_lookup,
                        )
                    pending_results.append(result)
                    pending_features.extend([clean_row, stego_row])
                    new_ok += 1
                except Exception as exc:
                    pending_results.append(
                        {
                            "status": "ERROR",
                            "error": repr(exc),
                            "experiment_version": EXPERIMENT_VERSION,
                            "config_id": config_id,
                            "method_config_hash": method_hash,
                            "partition": partition_name,
                            "alpha_method": method["alpha_method"],
                            "layout": method["layout"],
                            "layout_seed": method["layout_seed"],
                            "replay_file": context.replay_file,
                            "beatmap_hash": context.beatmap_hash,
                            "alpha": alpha,
                            **provenance,
                        }
                    )
                    print(f"ERROR {partition_name} {context.replay_file} {method}: {exc!r}")
            if len(pending_results) >= 60 or replay_index == len(cohort):
                replace_config_rows(result_path, RESULT_FIELDS, pending_results)
                replace_config_rows(feature_path, FEATURE_FIELDS, pending_features)
                pending_results = []
                pending_features = []
            if replay_index % 10 == 0 or replay_index == len(cohort):
                print(
                    f"physical {partition_name}: {replay_index}/{len(cohort)} "
                    f"replays | new OK={new_ok}",
                    flush=True,
                )
    results = pd.read_csv(result_path)
    features = pd.read_csv(feature_path)
    results = results[results["config_id"].astype(str).isin(requested_ids)]
    features = features[features["config_id"].astype(str).isin(requested_ids)]
    validate_physical_outputs(results, features, len(cohort))
    return new_ok


def validate_physical_outputs(
    results: pd.DataFrame,
    features: pd.DataFrame,
    expected_replays: int,
) -> None:
    expected_configs = expected_replays * len(method_configs())
    if len(results) != expected_configs or set(results["status"]) != {"OK"}:
        raise ValueError("Physical results nisu kompletni i bez grešaka.")
    if results["config_id"].duplicated().any():
        raise ValueError("Duplirani physical config_id.")
    labels = features.groupby("config_id")["label"].apply(
        lambda values: tuple(sorted(values.astype(int).tolist()))
    )
    if len(labels) != expected_configs or not all(value == (0, 1) for value in labels):
        raise ValueError("Svaki physical config mora imati clean/stego par.")
    per_replay = results.groupby("replay_file").size()
    if not np.all(per_replay.to_numpy() == len(method_configs())):
        raise ValueError("Replay nema tačno sve zamrznute method/layout configove.")
    for _, replay_group in results.groupby("replay_file"):
        if replay_group["message_sha256"].nunique() != 1:
            raise ValueError("Message zavisi od alpha metode ili layout-a.")
        if replay_group["pn_sequence_sha256"].nunique() != 1:
            raise ValueError("PN sekvenca zavisi od alpha metode ili layout-a.")
        if replay_group["message_bits"].nunique() != 1:
            raise ValueError("Broj bitova zavisi od metode ili layout-a.")
    for _, group in results.groupby(["replay_file", "alpha_method"]):
        if group["alpha"].nunique() != 1 or group["message_sha256"].nunique() != 1:
            raise ValueError("Alpha ili message zavise od layout-a.")
        if group["pn_sequence_sha256"].nunique() != 1:
            raise ValueError("PN sekvenca zavisi od layout-a.")
    for field, total in (
        ("requested_carriers_q", "requested_carriers"),
        ("active_carriers_q", "active_carriers"),
        ("sum_squared_shift_ms2_q", "sum_squared_shift_ms2"),
    ):
        summed = sum(results[f"{field}{quarter}"] for quarter in QUARTERS)
        if not np.allclose(summed, results[total]):
            raise ValueError(f"Quarter zbir nije jednak {total}.")
    prefix = results[results["layout"] == "prefix"]
    if not np.allclose(prefix["energy_fraction_q1"], 1.0):
        raise ValueError("PREFIX energija nije potpuno u quarter 1.")
    if np.any(~np.isfinite(features[list(FULL_FEATURES)].to_numpy(dtype=float))):
        raise ValueError("Feature tabela sadrži NaN/inf.")
    clean = features[features["label"] == 0]
    for _, group in clean.groupby("replay_file"):
        if np.any(group[list(FULL_FEATURES)].nunique(dropna=False).to_numpy() != 1):
            raise ValueError("Clean feature-i nisu identični kroz sve configove.")
