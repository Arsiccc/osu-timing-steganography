"""Shared physical-grid infrastructure for sender-local alpha experiments."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.stego.adaptive_alpha import replay_local_quality
from scripts.experiments.run_adaptive_alpha_comparison import (
    N_VALUE,
    PAYLOAD_FRACTION,
    PN_KEY,
    PREFIX_LAYOUT_KEY,
)
from scripts.experiments.run_adaptive_alpha_validation import (
    ENERGY_FIELDS,
    ENERGY_MEASUREMENT_VERSION,
    file_sha256,
)
from scripts.experiments.run_layout_comparison import (
    FEATURE_FIELDS as BASE_FEATURE_FIELDS,
    RESULT_FIELDS as BASE_RESULT_FIELDS,
    experiment_config_id,
    key_id,
    replace_configs_rows,
    run_one,
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


GRID_VERSION = "sender-local-physical-alpha-grid-v1"
SENDER_FIELDS = [
    "count_300",
    "count_100",
    "count_50",
    "count_miss",
    "total_judgements",
    "accuracy",
    "miss_fraction",
]
PROVENANCE_FIELDS = [
    "grid_version",
    "partition",
    "replay_sha256",
    "cohort_sha256",
    "performance_groups_sha256",
    "partition_sha256",
    "map_offsets_sha256",
]
REFERENCE_FIELDS = ["performance_category"]
RESULT_FIELDS = (
    BASE_RESULT_FIELDS + ENERGY_FIELDS + SENDER_FIELDS + REFERENCE_FIELDS
    + PROVENANCE_FIELDS
)
FEATURE_FIELDS = BASE_FEATURE_FIELDS + SENDER_FIELDS + REFERENCE_FIELDS + PROVENANCE_FIELDS


def completed_config_ids(path: Path, fields: list[str], feature: bool) -> set[str]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    if list(frame.columns) != fields:
        raise ValueError(f"{path} ima nekompatibilnu šemu; koristi novi output.")
    if frame.empty:
        return set()
    if not feature:
        return set(frame.loc[frame["status"] == "OK", "config_id"].astype(str))
    labels = frame.groupby("config_id")["label"].apply(
        lambda values: tuple(sorted(values.astype(int).tolist()))
    )
    return set(labels[labels == (0, 1)].index.astype(str))


def load_partitioned_cohort(
    results_path: Path,
    performance_path: Path,
    partition_path: Path,
    partition_name: str,
    limit: int | None = None,
) -> pd.DataFrame:
    if partition_name not in ("development", "validation"):
        raise ValueError("partition mora biti development ili validation.")
    source = pd.read_csv(results_path)
    performance = pd.read_csv(performance_path)[
        [
            "replay_file",
            "beatmap_hash",
            "performance_category",
            "count_300",
            "count_100",
            "count_50",
            "count_miss",
            "total_judgements",
            "accuracy",
            "miss_rate",
        ]
    ].rename(columns={"miss_rate": "miss_fraction"})
    partition = pd.read_csv(partition_path)[["beatmap_hash", "partition"]]
    selected_maps = set(
        partition.loc[partition["partition"] == partition_name, "beatmap_hash"].astype(str)
    )
    cohort = source[source["beatmap_hash"].astype(str).isin(selected_maps)].merge(
        performance,
        on=["replay_file", "beatmap_hash"],
        how="inner",
        validate="one_to_one",
    )
    if set(cohort["beatmap_hash"].astype(str)) != selected_maps:
        raise ValueError(f"{partition_name} cohort ne pokriva sve zamrznute mape.")
    if cohort["replay_file"].duplicated().any():
        raise ValueError("Partition cohort sadrži duplirane replay-eve.")
    cohort["partition"] = partition_name
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit mora biti > 0.")
        ordered = cohort.sort_values(["beatmap_hash", "replay_file"])
        diverse = ordered.drop_duplicates("beatmap_hash").head(limit)
        if len(diverse) < limit:
            remaining = ordered[~ordered["replay_file"].isin(set(diverse["replay_file"]))]
            diverse = pd.concat([diverse, remaining.head(limit - len(diverse))])
        cohort = diverse.copy()
    return cohort.reset_index(drop=True)


def validate_grid(
    results: pd.DataFrame,
    features: pd.DataFrame,
    expected_configs: int,
    expected_alphas: tuple[float, ...],
) -> None:
    if len(results) != expected_configs or set(results["status"]) != {"OK"}:
        raise RuntimeError("Physical alpha grid nije kompletan i bez grešaka.")
    if results["config_id"].duplicated().any():
        raise ValueError("Physical alpha grid ima dupliran config_id.")
    labels = features.groupby("config_id")["label"].apply(
        lambda values: tuple(sorted(values.astype(int).tolist()))
    )
    if len(labels) != expected_configs or not all(value == (0, 1) for value in labels):
        raise ValueError("Svaka grid konfiguracija mora imati clean/stego par.")
    per_replay = results.groupby("replay_file")["alpha"].apply(
        lambda values: tuple(sorted(values.astype(float).tolist()))
    )
    if not all(value == tuple(sorted(expected_alphas)) for value in per_replay):
        raise ValueError("Replay nema tačno jednu konfiguraciju za svaki grid alpha.")
    if not np.allclose(
        results["sum_squared_shift_ms2"],
        results["active_carriers"] * results["alpha"] ** 2,
    ):
        raise ValueError("Grid energija nije iz stvarnih full-amplitude writer shiftova.")
    if np.any(results["new_unmatched_events"] < results["positive_new_unmatched"]):
        raise ValueError("Stari net unmatched pokazatelj prelazi stvarne događaje.")


def run_physical_alpha_grid(
    *,
    cohort: pd.DataFrame,
    alphas: tuple[float, ...],
    dataset_dir: Path,
    map_offsets_path: Path,
    source_results_path: Path,
    performance_path: Path,
    partition_path: Path,
    output_path: Path,
    feature_path: Path,
    seed: int,
    hit_margin_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    alphas = tuple(sorted(set(float(value) for value in alphas)))
    if not alphas or any(value < 10.0 or value > 20.0 for value in alphas):
        raise ValueError("Grid alpha mora biti neprazan i unutar [10, 20].")
    partitions = set(cohort["partition"])
    if len(partitions) != 1:
        raise ValueError("Jedan grid run sme sadržati samo jednu particiju.")
    partition_name = str(next(iter(partitions)))
    beatmaps = build_beatmap_index(dataset_dir)
    offsets = load_map_offsets(map_offsets_path)
    provenance = {
        "grid_version": GRID_VERSION,
        "partition": partition_name,
        "cohort_sha256": file_sha256(source_results_path),
        "performance_groups_sha256": file_sha256(performance_path),
        "partition_sha256": file_sha256(partition_path),
        "map_offsets_sha256": file_sha256(map_offsets_path),
    }
    completed = completed_config_ids(output_path, RESULT_FIELDS, False) & completed_config_ids(
        feature_path, FEATURE_FIELDS, True
    )
    requested_ids: set[str] = set()
    new_ok = 0
    with tempfile.TemporaryDirectory(prefix=f"osu_sender_local_{partition_name}_") as name:
        temp_dir = Path(name)
        for replay_index, row in enumerate(cohort.itertuples(index=False), 1):
            context = prepare_replay(row, dataset_dir, beatmaps, offsets)
            replay_hash = file_sha256(context.osr_path)
            quality = replay_local_quality(context.replay)
            for field in SENDER_FIELDS:
                if not np.isclose(float(quality[field]), float(getattr(row, field))):
                    raise ValueError(
                        f"Metadata {field} nije identičan single-.osr vrednosti za "
                        f"{context.replay_file}."
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
                seed=seed,
            )
            replay_results: list[dict] = []
            replay_features: list[dict] = []
            for alpha in alphas:
                config_id = experiment_config_id(
                    replay_file=context.replay_file,
                    replay_sha256=replay_hash,
                    beatmap_hash=context.beatmap_hash,
                    alpha=alpha,
                    n_frames_per_bit=N_VALUE,
                    payload_fraction=PAYLOAD_FRACTION,
                    message_bits=num_bits,
                    message_seed=seed,
                    pn_key=PN_KEY,
                    layout="prefix",
                    hit_margin_ms=hit_margin_ms,
                    offset_ms=context.offset_ms,
                    hit_window_ms=context.hit_window_ms,
                    **provenance,
                )
                requested_ids.add(config_id)
                if config_id in completed:
                    continue
                common = {
                    "experiment_version": GRID_VERSION,
                    "replay_sha256": replay_hash,
                    "performance_category": str(row.performance_category),
                    **quality,
                    **provenance,
                }
                try:
                    result, clean, stego = run_one(
                        context=context,
                        message=message,
                        alpha=alpha,
                        n_value=N_VALUE,
                        payload_fraction=PAYLOAD_FRACTION,
                        layout="prefix",
                        pn_key=PN_KEY,
                        layout_key=PREFIX_LAYOUT_KEY,
                        message_seed=seed,
                        config_id=config_id,
                        hit_margin_ms=hit_margin_ms,
                        temp_dir=temp_dir,
                    )
                    for value in (result, clean, stego):
                        value.update(common)
                    replay_results.append(result)
                    replay_features.extend([clean, stego])
                    new_ok += 1
                except Exception as exc:
                    replay_results.append(
                        {
                            **common,
                            "status": "ERROR",
                            "error": repr(exc),
                            "config_id": config_id,
                            "message_seed": seed,
                            "pn_key": PN_KEY,
                            "pn_key_id": key_id(PN_KEY),
                            "layout_key": PREFIX_LAYOUT_KEY,
                            "layout_key_id": key_id(PREFIX_LAYOUT_KEY),
                            "hit_margin_ms": hit_margin_ms,
                            "layout": "prefix",
                            "category": context.category,
                            "replay_file": context.replay_file,
                            "player": context.player,
                            "beatmap_hash": context.beatmap_hash,
                            "alpha": alpha,
                            "n_frames_per_bit": N_VALUE,
                            "payload_fraction": PAYLOAD_FRACTION,
                        }
                    )
            replace_configs_rows(output_path, RESULT_FIELDS, replay_results)
            replace_configs_rows(feature_path, FEATURE_FIELDS, replay_features)
            if replay_index % 25 == 0 or replay_index == len(cohort):
                print(
                    f"{partition_name}: {replay_index}/{len(cohort)} replay-eva | "
                    f"new OK={new_ok}"
                )
    results = pd.read_csv(output_path)
    features = pd.read_csv(feature_path)
    results = results[results["config_id"].astype(str).isin(requested_ids)].copy()
    features = features[features["config_id"].astype(str).isin(requested_ids)].copy()
    validate_grid(results, features, len(cohort) * len(alphas), alphas)
    return results, features, new_ok
