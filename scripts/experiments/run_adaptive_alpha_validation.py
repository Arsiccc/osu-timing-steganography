"""Physical-energy and frozen-policy validation for adaptive alpha.

This experiment preserves the validated physical replay round trip.  It adds
measurement only: energy is calculated from the non-zero chronology-safe
integer shifts passed directly to ``write_stego_replay``.

The held-out split is beatmap-disjoint and deterministic.  The current
map-relative performance category is intentionally treated as an oracle input:
its official hit counts are sender-visible, but its percentile requires a
reference population of other replays from the same beatmap.  Consequently the
held-out result is internal and does not validate deployability.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_comparison import (
    ADAPTIVE_ALPHA,
    FEATURE_FIELDS as ADAPTIVE_FEATURE_FIELDS,
    METHODS,
    N_VALUE,
    PAYLOAD_FRACTION,
    PERFORMANCE_ORDER,
    PN_KEY,
    POLICY_DEFINITION,
    POLICY_ID,
    PREFIX_LAYOUT_KEY,
    RESULT_FIELDS as ADAPTIVE_RESULT_FIELDS,
    alpha_for_method,
    load_experiment_cohort,
    validate_complete_rows,
)
from scripts.experiments.run_layout_comparison import (
    bootstrap_auc_delta,
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
    stable_seed,
)
from scripts.experiments.run_steganalysis import (
    FEATURE_COLUMNS,
    bootstrap_auc_by_group,
    evaluate_subset,
)


EXPERIMENT_VERSION = "adaptive-alpha-validation-v1"
ENERGY_MEASUREMENT_VERSION = "writer-input-safe-shifts-v1"
CATEGORY_AVAILABILITY = (
    "oracle-map-relative-percentile;official-hit-counts-sender-visible;"
    "reference-population-not-sender-local"
)
DEFAULT_VALIDATION_MAP_FRACTION = 0.30
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "adaptive_alpha_validation_v1"

ENERGY_FIELDS = [
    "sum_squared_shift_ms2",
    "mean_squared_shift_ms2",
    "rms_applied_shift_ms",
    "mean_absolute_applied_shift_ms",
    "total_absolute_shift_ms",
    "new_unmatched_events",
    "new_matched_events",
    "changed_match_status_total",
]
PROVENANCE_FIELDS = [
    "energy_measurement_version",
    "category_availability",
    "replay_sha256",
    "cohort_sha256",
    "performance_groups_sha256",
    "map_offsets_sha256",
]
RESULT_FIELDS = ADAPTIVE_RESULT_FIELDS + ENERGY_FIELDS + PROVENANCE_FIELDS
FEATURE_FIELDS = ADAPTIVE_FEATURE_FIELDS + PROVENANCE_FIELDS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_config_ids(path: Path, fields: list[str], feature: bool) -> set[str]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    if list(frame.columns) != fields:
        raise ValueError(f"{path} ima nekompatibilnu šemu; koristi novi output dir.")
    if frame.empty:
        return set()
    if not feature:
        return set(frame.loc[frame["status"] == "OK", "config_id"].astype(str))
    labels = frame.groupby("config_id")["label"].apply(
        lambda values: tuple(sorted(values.astype(int).tolist()))
    )
    return set(labels[labels == (0, 1)].index.astype(str))


def deterministic_beatmap_partition(
    beatmap_hashes: pd.Series,
    validation_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation map fraction mora biti u (0, 1).")
    maps = sorted(set(beatmap_hashes.astype(str)))
    if len(maps) < 4:
        raise ValueError("Potrebne su najmanje četiri beatmape za held-out podelu.")
    ordered = sorted(
        maps,
        key=lambda value: hashlib.sha256(
            f"{seed}|{EXPERIMENT_VERSION}|heldout-map|{value}".encode("utf-8")
        ).digest(),
    )
    validation_count = max(2, int(math.ceil(len(ordered) * validation_fraction)))
    validation_count = min(validation_count, len(ordered) - 2)
    validation_maps = set(ordered[:validation_count])
    return pd.DataFrame(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "partition_seed": seed,
            "validation_map_fraction_requested": validation_fraction,
            "beatmap_hash": maps,
            "partition": [
                "validation" if value in validation_maps else "development"
                for value in maps
            ],
            "category_availability": CATEGORY_AVAILABILITY,
        }
    )


def energy_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    active = int(frame["active_carriers"].sum())
    requested = int(frame["requested_carriers"].sum())
    sum_squared = float(frame["sum_squared_shift_ms2"].sum())
    total_absolute = float(frame["total_absolute_shift_ms"].sum())
    return {
        "replays": int(frame["replay_file"].nunique()),
        "requested_carriers": requested,
        "active_carriers": active,
        "active_carrier_fraction": active / requested if requested else 0.0,
        "sum_squared_shift_ms2": sum_squared,
        "mean_sum_squared_shift_ms2_per_replay": float(
            frame["sum_squared_shift_ms2"].mean()
        ),
        "mean_squared_shift_ms2_over_active_carriers": (
            sum_squared / active if active else 0.0
        ),
        "rms_applied_shift_ms": (
            math.sqrt(sum_squared / active) if active else 0.0
        ),
        "total_absolute_shift_ms": total_absolute,
        "mean_total_absolute_shift_ms_per_replay": float(
            frame["total_absolute_shift_ms"].mean()
        ),
        "mean_absolute_applied_shift_ms": (
            total_absolute / active if active else 0.0
        ),
        "dropped_for_hit_window": int(frame["dropped_for_hit_window"].sum()),
        "dropped_for_chronology": int(frame["dropped_for_chronology"].sum()),
        "new_unmatched_events": int(frame["new_unmatched_events"].sum()),
        "new_matched_events": int(frame["new_matched_events"].sum()),
        "changed_match_status_total": int(
            frame["changed_match_status_total"].sum()
        ),
    }


def _wide_pair(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "replay_file",
        "beatmap_hash",
        "message_bits",
        "bit_errors_roundtrip",
        "requested_carriers",
        "active_carriers",
        "sum_squared_shift_ms2",
        "total_absolute_shift_ms",
        "dropped_for_chronology",
        "new_unmatched_events",
        "positive_new_unmatched",
    ]
    control = results[results["method"] == "fixed_15"][columns]
    adaptive = results[results["method"] == "adaptive_v1"][columns]
    paired = control.merge(
        adaptive,
        on=["replay_file", "beatmap_hash"],
        suffixes=("_control", "_adaptive"),
        validate="one_to_one",
    )
    if not np.array_equal(
        paired["message_bits_control"], paired["message_bits_adaptive"]
    ):
        raise ValueError("Adaptive i fixed_15 nemaju isti payload po replay-u.")
    return paired


def _paired_value(frame: pd.DataFrame, metric: str, suffix: str) -> float:
    if metric == "weighted_ber":
        return float(
            frame[f"bit_errors_roundtrip_{suffix}"].sum()
            / frame[f"message_bits_{suffix}"].sum()
        )
    if metric == "mean_sum_squared_shift_ms2_per_replay":
        return float(frame[f"sum_squared_shift_ms2_{suffix}"].mean())
    if metric == "pooled_rms_applied_shift_ms":
        return float(
            np.sqrt(
                frame[f"sum_squared_shift_ms2_{suffix}"].sum()
                / frame[f"active_carriers_{suffix}"].sum()
            )
        )
    if metric == "pooled_mean_absolute_shift_ms":
        return float(
            frame[f"total_absolute_shift_ms_{suffix}"].sum()
            / frame[f"active_carriers_{suffix}"].sum()
        )
    if metric == "pooled_active_carrier_fraction":
        return float(
            frame[f"active_carriers_{suffix}"].sum()
            / frame[f"requested_carriers_{suffix}"].sum()
        )
    if metric in (
        "dropped_for_chronology",
        "new_unmatched_events",
        "positive_new_unmatched",
    ):
        return float(frame[f"{metric}_{suffix}"].mean())
    raise ValueError(f"Nepoznata paired metrika: {metric}.")


def bootstrap_paired_metric(
    paired: pd.DataFrame,
    metric: str,
    cluster_column: str,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    clusters = paired[cluster_column].astype(str).unique()
    pieces = {
        cluster: paired[paired[cluster_column].astype(str) == cluster]
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([pieces[cluster] for cluster in sampled], ignore_index=True)
        deltas[index] = _paired_value(sample, metric, "adaptive") - _paired_value(
            sample, metric, "control"
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def paired_metric_rows(
    paired: pd.DataFrame,
    cluster_column: str,
    seed: int,
    iterations: int,
    scope: str,
) -> list[dict]:
    metrics = (
        "weighted_ber",
        "mean_sum_squared_shift_ms2_per_replay",
        "pooled_rms_applied_shift_ms",
        "pooled_mean_absolute_shift_ms",
        "pooled_active_carrier_fraction",
        "dropped_for_chronology",
        "new_unmatched_events",
        "positive_new_unmatched",
    )
    rows: list[dict] = []
    for metric in metrics:
        control = _paired_value(paired, metric, "control")
        adaptive = _paired_value(paired, metric, "adaptive")
        low, high = bootstrap_paired_metric(
            paired,
            metric,
            cluster_column,
            stable_seed(seed, EXPERIMENT_VERSION, scope, metric, "paired"),
            iterations,
        )
        rows.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "scope": scope,
                "metric": metric,
                "control_method": "fixed_15",
                "candidate_method": "adaptive_v1",
                "control_value": control,
                "candidate_value": adaptive,
                "delta_adaptive_minus_control": adaptive - control,
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "bootstrap_unit": cluster_column,
                "bootstrap_iterations": iterations,
                "replay_pairs": len(paired),
                "cluster_count": paired[cluster_column].nunique(),
            }
        )
    return rows


def bootstrap_auc_delta_clustered(
    control: pd.DataFrame,
    adaptive: pd.DataFrame,
    cluster_column: str,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    paired = control[
        ["replay_file", "beatmap_hash", "label", "score"]
    ].merge(
        adaptive[["replay_file", "beatmap_hash", "label", "score"]],
        on=["replay_file", "beatmap_hash", "label"],
        suffixes=("_control", "_adaptive"),
        validate="one_to_one",
    )
    clusters = paired[cluster_column].astype(str).unique()
    pieces = {
        cluster: paired[paired[cluster_column].astype(str) == cluster]
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([pieces[cluster] for cluster in sampled], ignore_index=True)
        labels = sample["label"].to_numpy(dtype=np.int8)
        deltas.append(
            float(
                roc_auc_score(labels, sample["score_adaptive"])
                - roc_auc_score(labels, sample["score_control"])
            )
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def generalization_auc(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    method: str,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
) -> tuple[dict, pd.DataFrame]:
    train = development[development["method"] == method].copy()
    test = validation[validation["method"] == method].copy()
    if set(train["beatmap_hash"]) & set(test["beatmap_hash"]):
        raise ValueError("Beatmap leakage u development-to-validation evaluaciji.")
    classifier = RandomForestClassifier(
        n_estimators=trees,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=stable_seed(seed, EXPERIMENT_VERSION, "generalization-rf"),
        n_jobs=-1,
    )
    classifier.fit(
        train[FEATURE_COLUMNS].to_numpy(dtype=np.float64),
        train["label"].to_numpy(dtype=np.int8),
    )
    scores = classifier.predict_proba(
        test[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    )[:, 1]
    labels = test["label"].to_numpy(dtype=np.int8)
    groups = test["beatmap_hash"].astype(str).to_numpy()
    auc = float(roc_auc_score(labels, scores))
    low, high = bootstrap_auc_by_group(
        labels,
        scores,
        groups,
        stable_seed(seed, EXPERIMENT_VERSION, method, "generalization-bootstrap"),
        bootstrap_iterations,
    )
    predictions = test[
        ["replay_file", "beatmap_hash", "label"]
    ].copy()
    predictions["score"] = scores
    return (
        {
            "experiment_version": EXPERIMENT_VERSION,
            "scope": "development_to_validation_beatmap_generalization",
            "method": method,
            "roc_auc": auc,
            "auc_ci95_low": low,
            "auc_ci95_high": high,
            "bootstrap_unit": "beatmap_hash",
            "development_maps": train["beatmap_hash"].nunique(),
            "validation_maps": test["beatmap_hash"].nunique(),
            "validation_replays": test["replay_file"].nunique(),
            "classifier_trees": trees,
            "category_availability": CATEGORY_AVAILABILITY,
        },
        predictions,
    )


def write_analysis_outputs(
    results: pd.DataFrame,
    features: pd.DataFrame,
    partition: pd.DataFrame,
    output_dir: Path,
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
    hit_margin_ms: float,
    validation_map_fraction: float,
) -> None:
    first = results.iloc[0]
    common = {
        "experiment_version": EXPERIMENT_VERSION,
        "policy_id": POLICY_ID,
        "policy_definition": POLICY_DEFINITION,
        "energy_measurement_version": ENERGY_MEASUREMENT_VERSION,
        "category_availability": CATEGORY_AVAILABILITY,
        "message_seed": seed,
        "pn_key": PN_KEY,
        "pn_key_id": key_id(PN_KEY),
        "layout": "prefix",
        "layout_key": PREFIX_LAYOUT_KEY,
        "layout_key_id": key_id(PREFIX_LAYOUT_KEY),
        "n_frames_per_bit": N_VALUE,
        "payload_fraction": PAYLOAD_FRACTION,
        "hit_margin_ms": hit_margin_ms,
        "folds": folds,
        "classifier_trees": trees,
        "bootstrap_iterations": bootstrap_iterations,
        "validation_map_fraction": validation_map_fraction,
        "cohort_sha256": first["cohort_sha256"],
        "performance_groups_sha256": first["performance_groups_sha256"],
        "map_offsets_sha256": first["map_offsets_sha256"],
    }
    overall_rows = []
    group_rows = []
    for method, group in results.groupby("method", sort=False):
        overall_rows.append({**common, "method": method, **energy_summary(group)})
        for category in PERFORMANCE_ORDER:
            category_rows = group[group["performance_category"] == category]
            group_rows.append(
                {
                    **common,
                    "method": method,
                    "performance_category": category,
                    **energy_summary(category_rows),
                }
            )
    pd.DataFrame(overall_rows).to_csv(output_dir / "physical_energy_overall.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "physical_energy_by_group.csv", index=False)

    all_paired = _wide_pair(results)
    energy_paired = paired_metric_rows(
        all_paired,
        cluster_column="replay_file",
        seed=seed,
        iterations=bootstrap_iterations,
        scope="all_replays",
    )
    pd.DataFrame(energy_paired).to_csv(
        output_dir / "physical_energy_paired.csv", index=False
    )

    full_auc_rows: list[dict] = []
    full_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for cv_group_column in ("replay_file", "beatmap_hash"):
        for method in METHODS:
            summary, prediction = evaluate_subset(
                features[features["method"] == method],
                scope="ALL",
                alpha=None,
                n_value=N_VALUE,
                folds=folds,
                trees=trees,
                seed=seed,
                bootstrap_iterations=bootstrap_iterations,
                cv_group_column=cv_group_column,
            )
            bootstrap_unit = "replay_file"
            if cv_group_column == "beatmap_hash":
                low, high = bootstrap_auc_by_group(
                    prediction["label"].to_numpy(dtype=np.int8),
                    prediction["oof_probability_stego"].to_numpy(dtype=np.float64),
                    prediction["beatmap_hash"].astype(str).to_numpy(),
                    stable_seed(
                        seed,
                        EXPERIMENT_VERSION,
                        method,
                        "full-beatmap-auc-bootstrap",
                    ),
                    bootstrap_iterations,
                )
                summary["auc_ci95_low"] = low
                summary["auc_ci95_high"] = high
                bootstrap_unit = "beatmap_hash"
            full_auc_rows.append(
                {
                    **common,
                    "method": method,
                    "cv_group_column": cv_group_column,
                    "bootstrap_unit": bootstrap_unit,
                    **summary,
                }
            )
            full_predictions[(cv_group_column, method)] = prediction
    pd.DataFrame(full_auc_rows).to_csv(output_dir / "full_auc.csv", index=False)

    full_auc_paired: list[dict] = []
    for cv_group_column in ("replay_file", "beatmap_hash"):
        control_prediction = full_predictions[(cv_group_column, "fixed_15")]
        adaptive_prediction = full_predictions[(cv_group_column, "adaptive_v1")]
        control_auc = next(
            row for row in full_auc_rows
            if row["cv_group_column"] == cv_group_column
            and row["method"] == "fixed_15"
        )["roc_auc"]
        adaptive_auc = next(
            row for row in full_auc_rows
            if row["cv_group_column"] == cv_group_column
            and row["method"] == "adaptive_v1"
        )["roc_auc"]
        if cv_group_column == "replay_file":
            low, high = bootstrap_auc_delta(
                control_prediction,
                adaptive_prediction,
                stable_seed(seed, EXPERIMENT_VERSION, "full-replay-auc-delta"),
                bootstrap_iterations,
            )
            bootstrap_unit = "replay_file"
        else:
            low, high = bootstrap_auc_delta_clustered(
                control_prediction.rename(
                    columns={"oof_probability_stego": "score"}
                ),
                adaptive_prediction.rename(
                    columns={"oof_probability_stego": "score"}
                ),
                "beatmap_hash",
                stable_seed(seed, EXPERIMENT_VERSION, "full-beatmap-auc-delta"),
                bootstrap_iterations,
            )
            bootstrap_unit = "beatmap_hash"
        full_auc_paired.append(
            {
                **common,
                "cv_group_column": cv_group_column,
                "control_method": "fixed_15",
                "candidate_method": "adaptive_v1",
                "auc_control": control_auc,
                "auc_adaptive": adaptive_auc,
                "delta_auc_adaptive_minus_control": adaptive_auc - control_auc,
                "delta_auc_ci95_low": low,
                "delta_auc_ci95_high": high,
                "bootstrap_unit": bootstrap_unit,
                "replay_pairs": results["replay_file"].nunique(),
                "bootstrap_clusters": results[bootstrap_unit].nunique(),
            }
        )
    pd.DataFrame(full_auc_paired).to_csv(
        output_dir / "full_auc_paired.csv", index=False
    )

    partition_lookup = partition.set_index("beatmap_hash")["partition"]
    results = results.copy()
    features = features.copy()
    results["partition"] = results["beatmap_hash"].map(partition_lookup)
    features["partition"] = features["beatmap_hash"].map(partition_lookup)
    if results["partition"].isna().any() or features["partition"].isna().any():
        raise ValueError("Partition ne pokriva sve rezultate.")
    validation_results = results[results["partition"] == "validation"]
    validation_features = features[features["partition"] == "validation"]
    development_features = features[features["partition"] == "development"]

    heldout_method_rows = []
    replay_predictions: dict[str, pd.DataFrame] = {}
    generalization_predictions: dict[str, pd.DataFrame] = {}
    for method in ("fixed_15", "adaptive_v1"):
        method_results = validation_results[validation_results["method"] == method]
        bits = int(method_results["message_bits"].sum())
        replay_summary, prediction = evaluate_subset(
            validation_features[validation_features["method"] == method],
            scope="ALL",
            alpha=None,
            n_value=N_VALUE,
            folds=folds,
            trees=trees,
            seed=seed,
            bootstrap_iterations=bootstrap_iterations,
            cv_group_column="replay_file",
        )
        replay_predictions[method] = prediction
        heldout_method_rows.append(
            {
                **common,
                "scope": "validation_replay_grouped_cv",
                "method": method,
                "roundtrip_ber": float(
                    method_results["bit_errors_roundtrip"].sum() / bits
                ),
                **energy_summary(method_results),
                "roc_auc": replay_summary["roc_auc"],
                "auc_ci95_low": replay_summary["auc_ci95_low"],
                "auc_ci95_high": replay_summary["auc_ci95_high"],
                "bootstrap_unit": "replay_file",
                "development_maps": development_features["beatmap_hash"].nunique(),
                "validation_maps": validation_features["beatmap_hash"].nunique(),
                "validation_replays": method_results["replay_file"].nunique(),
            }
        )
        generalization_row, generalization_prediction = generalization_auc(
            development_features,
            validation_features,
            method,
            trees,
            seed,
            bootstrap_iterations,
        )
        heldout_method_rows.append({**common, **generalization_row})
        generalization_predictions[method] = generalization_prediction

    heldout_paired = paired_metric_rows(
        _wide_pair(validation_results),
        cluster_column="beatmap_hash",
        seed=seed,
        iterations=bootstrap_iterations,
        scope="validation_physical",
    )
    replay_low, replay_high = bootstrap_auc_delta(
        replay_predictions["fixed_15"],
        replay_predictions["adaptive_v1"],
        stable_seed(seed, EXPERIMENT_VERSION, "validation-replay-auc-delta"),
        bootstrap_iterations,
    )
    control_replay_auc = next(
        row["roc_auc"] for row in heldout_method_rows
        if row["scope"] == "validation_replay_grouped_cv"
        and row["method"] == "fixed_15"
    )
    adaptive_replay_auc = next(
        row["roc_auc"] for row in heldout_method_rows
        if row["scope"] == "validation_replay_grouped_cv"
        and row["method"] == "adaptive_v1"
    )
    heldout_paired.append(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "scope": "validation_replay_grouped_cv",
            "metric": "roc_auc",
            "control_method": "fixed_15",
            "candidate_method": "adaptive_v1",
            "control_value": control_replay_auc,
            "candidate_value": adaptive_replay_auc,
            "delta_adaptive_minus_control": adaptive_replay_auc - control_replay_auc,
            "delta_ci95_low": replay_low,
            "delta_ci95_high": replay_high,
            "bootstrap_unit": "replay_file",
            "bootstrap_iterations": bootstrap_iterations,
            "replay_pairs": validation_results["replay_file"].nunique(),
            "cluster_count": validation_results["replay_file"].nunique(),
        }
    )
    control_generalization = next(
        row for row in heldout_method_rows
        if row["scope"] == "development_to_validation_beatmap_generalization"
        and row["method"] == "fixed_15"
    )
    adaptive_generalization = next(
        row for row in heldout_method_rows
        if row["scope"] == "development_to_validation_beatmap_generalization"
        and row["method"] == "adaptive_v1"
    )
    gen_low, gen_high = bootstrap_auc_delta_clustered(
        generalization_predictions["fixed_15"],
        generalization_predictions["adaptive_v1"],
        "beatmap_hash",
        stable_seed(seed, EXPERIMENT_VERSION, "generalization-auc-delta"),
        bootstrap_iterations,
    )
    heldout_paired.append(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "scope": "development_to_validation_beatmap_generalization",
            "metric": "roc_auc",
            "control_method": "fixed_15",
            "candidate_method": "adaptive_v1",
            "control_value": control_generalization["roc_auc"],
            "candidate_value": adaptive_generalization["roc_auc"],
            "delta_adaptive_minus_control": (
                adaptive_generalization["roc_auc"] - control_generalization["roc_auc"]
            ),
            "delta_ci95_low": gen_low,
            "delta_ci95_high": gen_high,
            "bootstrap_unit": "beatmap_hash",
            "bootstrap_iterations": bootstrap_iterations,
            "replay_pairs": validation_results["replay_file"].nunique(),
            "cluster_count": validation_results["beatmap_hash"].nunique(),
        }
    )
    pd.DataFrame(heldout_method_rows).to_csv(
        output_dir / "heldout_method_summary.csv", index=False
    )
    pd.DataFrame(heldout_paired).to_csv(
        output_dir / "heldout_paired.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure physical shift energy and validate frozen adaptive alpha."
    )
    parser.add_argument(
        "--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv"
    )
    parser.add_argument(
        "--performance-groups",
        type=Path,
        default=METADATA_DIR / "performance_groups.csv",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-group-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hit-margin-ms", type=float, default=5.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument(
        "--validation-map-fraction",
        type=float,
        default=DEFAULT_VALIDATION_MAP_FRACTION,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "physical_energy_per_replay.csv"
    feature_path = args.output_dir / "validation_features.csv"
    partition_path = args.output_dir / "heldout_partition.csv"

    cohort = load_experiment_cohort(
        args.results, args.performance_groups, args.per_group_limit
    )
    partition = deterministic_beatmap_partition(
        cohort["beatmap_hash"], args.validation_map_fraction, args.seed
    )
    replay_counts = (
        cohort.groupby("beatmap_hash")["replay_file"]
        .nunique()
        .rename("replay_count")
    )
    partition["replay_count"] = partition["beatmap_hash"].map(replay_counts)
    partition.to_csv(partition_path, index=False)
    beatmaps = build_beatmap_index(args.dataset_dir)
    offsets = load_map_offsets(args.map_offsets)
    provenance = {
        "energy_measurement_version": ENERGY_MEASUREMENT_VERSION,
        "category_availability": CATEGORY_AVAILABILITY,
        "cohort_sha256": file_sha256(args.results),
        "performance_groups_sha256": file_sha256(args.performance_groups),
        "map_offsets_sha256": file_sha256(args.map_offsets),
    }
    completed = completed_config_ids(result_path, RESULT_FIELDS, False) & completed_config_ids(
        feature_path, FEATURE_FIELDS, True
    )
    requested_ids: set[str] = set()
    new_ok = 0

    print("=" * 100)
    print("ADAPTIVE ALPHA PHYSICAL ENERGY + FROZEN HELD-OUT VALIDATION")
    print("=" * 100)
    print(f"replays={len(cohort)} maps={cohort['beatmap_hash'].nunique()} methods={METHODS}")
    print(f"policy={ADAPTIVE_ALPHA}")
    print(f"category availability={CATEGORY_AVAILABILITY}")

    with tempfile.TemporaryDirectory(prefix="osu_adaptive_validation_") as temp_name:
        temp_dir = Path(temp_name)
        for replay_index, row in enumerate(cohort.itertuples(index=False), 1):
            context = prepare_replay(
                row, args.dataset_dir, beatmaps, offsets
            )
            replay_sha256 = file_sha256(context.osr_path)
            category = str(row.performance_category)
            capacity = nominal_capacity_bits(len(context.original_residuals), N_VALUE)
            num_bits = message_length_for_fraction(capacity, PAYLOAD_FRACTION)
            message = deterministic_message(
                replay_file=context.replay_file,
                beatmap_hash=context.beatmap_hash,
                alpha=0.0,
                n_value=N_VALUE,
                payload_fraction=PAYLOAD_FRACTION,
                num_bits=num_bits,
                seed=args.seed,
            )
            replay_results: list[dict] = []
            replay_features: list[dict] = []
            for method in METHODS:
                alpha = alpha_for_method(method, category)
                config_id = experiment_config_id(
                    experiment_version=EXPERIMENT_VERSION,
                    policy_id=POLICY_ID,
                    policy_definition=POLICY_DEFINITION,
                    method=method,
                    replay_file=context.replay_file,
                    replay_sha256=replay_sha256,
                    beatmap_hash=context.beatmap_hash,
                    performance_category=category,
                    alpha=alpha,
                    n_frames_per_bit=N_VALUE,
                    payload_fraction=PAYLOAD_FRACTION,
                    message_bits=num_bits,
                    message_seed=args.seed,
                    pn_key=PN_KEY,
                    layout="prefix",
                    hit_margin_ms=args.hit_margin_ms,
                    offset_ms=context.offset_ms,
                    hit_window_ms=context.hit_window_ms,
                    **provenance,
                )
                requested_ids.add(config_id)
                if config_id in completed:
                    continue
                common = {
                    "experiment_version": EXPERIMENT_VERSION,
                    "method": method,
                    "policy_id": POLICY_ID,
                    "policy_definition": POLICY_DEFINITION,
                    "performance_category": category,
                    "replay_sha256": replay_sha256,
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
                        message_seed=args.seed,
                        config_id=config_id,
                        hit_margin_ms=args.hit_margin_ms,
                        temp_dir=temp_dir,
                    )
                    result.update(common)
                    clean.update(common)
                    stego.update(common)
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
                            "message_seed": args.seed,
                            "pn_key": PN_KEY,
                            "pn_key_id": key_id(PN_KEY),
                            "layout_key": PREFIX_LAYOUT_KEY,
                            "layout_key_id": key_id(PREFIX_LAYOUT_KEY),
                            "hit_margin_ms": args.hit_margin_ms,
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
                    print(f"ERROR {context.replay_file[:12]} {method}: {exc!r}")
            replace_configs_rows(result_path, RESULT_FIELDS, replay_results)
            replace_configs_rows(feature_path, FEATURE_FIELDS, replay_features)
            if replay_index % 25 == 0 or replay_index == len(cohort):
                print(f"{replay_index}/{len(cohort)} replay-eva | new OK={new_ok}")

    results = pd.read_csv(result_path)
    features = pd.read_csv(feature_path)
    results = results[results["config_id"].astype(str).isin(requested_ids)].copy()
    features = features[features["config_id"].astype(str).isin(requested_ids)].copy()
    validate_complete_rows(results, features, len(cohort) * len(METHODS))

    expected_squared = results["active_carriers"] * results["alpha"] ** 2
    if not np.allclose(results["sum_squared_shift_ms2"], expected_squared):
        raise AssertionError("Writer-input energija nije puna alpha amplituda po carrier-u.")
    expected_absolute = results["active_carriers"] * results["alpha"].abs()
    if not np.allclose(results["total_absolute_shift_ms"], expected_absolute):
        raise AssertionError("Total absolute shift ne odgovara writer-input shiftovima.")
    nonzero = results["active_carriers"] > 0
    expected_mean_squared = (
        results.loc[nonzero, "sum_squared_shift_ms2"]
        / results.loc[nonzero, "active_carriers"]
    )
    if not np.allclose(
        results.loc[nonzero, "mean_squared_shift_ms2"], expected_mean_squared
    ):
        raise AssertionError("Mean squared shift nije računat preko aktivnih carrier-a.")
    if np.any(results["new_unmatched_events"] < results["positive_new_unmatched"]):
        raise AssertionError("Neto unmatched porast ne može preći broj novih događaja.")

    write_analysis_outputs(
        results,
        features,
        partition,
        args.output_dir,
        args.folds,
        args.trees,
        args.seed,
        args.bootstrap_iterations,
        args.hit_margin_ms,
        args.validation_map_fraction,
    )
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
