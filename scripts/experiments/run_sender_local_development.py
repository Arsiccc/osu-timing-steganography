"""Development-only sender-local adaptive-alpha policy search.

This module refuses to evaluate validation beatmaps.  It first builds a
resumable physical alpha grid on the frozen development partition, then creates
the pre-registered accuracy-only candidates by selecting the corresponding
physical result for each replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import ACCURACY_FORMULA
from scripts.experiments.run_adaptive_alpha_comparison import ADAPTIVE_ALPHA, N_VALUE, PAYLOAD_FRACTION
from scripts.experiments.run_adaptive_alpha_validation import (
    bootstrap_auc_delta_clustered,
    energy_summary,
)
from scripts.experiments.run_layout_comparison import bootstrap_auc_delta
from scripts.experiments.run_pilot_ber_sweep import stable_seed
from scripts.experiments.run_steganalysis import bootstrap_auc_by_group, evaluate_subset
from scripts.experiments.sender_local_common import (
    GRID_VERSION,
    load_partitioned_cohort,
    run_physical_alpha_grid,
)


EXPERIMENT_VERSION = "sender-local-development-v1"
ALPHA_GRID = (10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0)
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "sender_local_adaptive_v1"


def candidate_definitions(cohort: pd.DataFrame) -> list[dict]:
    accuracy = cohort["accuracy"].to_numpy(dtype=np.float64)
    quartiles = np.quantile(accuracy, [0.25, 0.50, 0.75]).tolist()
    median = float(quartiles[1])
    return [
        {
            "candidate_id": "fixed_14",
            "family": "fixed",
            "feature_names": [],
            "thresholds": [],
            "alpha_values": [14.0],
        },
        {
            "candidate_id": "fixed_15",
            "family": "fixed",
            "feature_names": [],
            "thresholds": [],
            "alpha_values": [15.0],
        },
        {
            "candidate_id": "oracle_map_relative",
            "family": "oracle_reference_not_deployable",
            "feature_names": ["performance_category"],
            "thresholds": [],
            "alpha_values": [20.0, 16.0, 12.0, 10.0],
            "category_mapping": ADAPTIVE_ALPHA,
        },
        {
            "candidate_id": "accuracy_q4_strong",
            "family": "accuracy_only_quartiles",
            "feature_names": ["accuracy"],
            "formula": ACCURACY_FORMULA,
            "thresholds": quartiles,
            "alpha_values": [20.0, 16.0, 12.0, 10.0],
        },
        {
            "candidate_id": "accuracy_q4_moderate",
            "family": "accuracy_only_quartiles",
            "feature_names": ["accuracy"],
            "formula": ACCURACY_FORMULA,
            "thresholds": quartiles,
            "alpha_values": [18.0, 16.0, 14.0, 12.0],
        },
        {
            "candidate_id": "accuracy_binary",
            "family": "accuracy_only_binary",
            "feature_names": ["accuracy"],
            "formula": ACCURACY_FORMULA,
            "thresholds": [median],
            "alpha_values": [18.0, 12.0],
        },
    ]


def assigned_alpha(definition: dict, row: pd.Series) -> float:
    family = definition["family"]
    if family == "fixed":
        return float(definition["alpha_values"][0])
    if family == "oracle_reference_not_deployable":
        return float(definition["category_mapping"][row["performance_category"]])
    thresholds = np.asarray(definition["thresholds"], dtype=np.float64)
    values = np.asarray(definition["alpha_values"], dtype=np.float64)
    return float(values[np.searchsorted(thresholds, float(row["accuracy"]), side="right")])


def select_candidate_rows(
    grid_results: pd.DataFrame,
    grid_features: pd.DataFrame,
    definition: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    replay = grid_results[
        ["replay_file", "accuracy", "performance_category"]
    ].drop_duplicates("replay_file")
    assignments = replay.copy()
    assignments["selected_alpha"] = assignments.apply(
        lambda row: assigned_alpha(definition, row), axis=1
    )
    selected = grid_results.merge(
        assignments[["replay_file", "selected_alpha"]],
        left_on=["replay_file", "alpha"],
        right_on=["replay_file", "selected_alpha"],
        how="inner",
        validate="many_to_one",
    )
    if selected["replay_file"].nunique() != grid_results["replay_file"].nunique():
        raise ValueError(f"Candidate {definition['candidate_id']} ne pokriva svaki replay.")
    features = grid_features[
        grid_features["config_id"].isin(set(selected["config_id"]))
    ].copy()
    labels = features.groupby("replay_file")["label"].apply(
        lambda values: tuple(sorted(values.astype(int).tolist()))
    )
    if not all(value == (0, 1) for value in labels):
        raise ValueError("Candidate features nemaju tačne clean/stego parove.")
    return selected, features


def bootstrap_numeric_delta(
    control: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str,
    seed: int,
    iterations: int,
) -> tuple[float, float, float, float]:
    columns = [
        "replay_file", "message_bits", "bit_errors_roundtrip", "active_carriers",
        "requested_carriers", "sum_squared_shift_ms2", "dropped_for_chronology",
        "new_unmatched_events",
    ]
    paired = control[columns].merge(
        candidate[columns], on="replay_file", suffixes=("_control", "_candidate"),
        validate="one_to_one",
    )

    def value(frame: pd.DataFrame, suffix: str) -> float:
        if metric == "weighted_ber":
            return float(frame[f"bit_errors_roundtrip_{suffix}"].sum() / frame[f"message_bits_{suffix}"].sum())
        if metric == "mean_energy_per_replay":
            return float(frame[f"sum_squared_shift_ms2_{suffix}"].mean())
        if metric == "pooled_rms_shift":
            return float(np.sqrt(frame[f"sum_squared_shift_ms2_{suffix}"].sum() / frame[f"active_carriers_{suffix}"].sum()))
        if metric == "active_carrier_fraction":
            return float(frame[f"active_carriers_{suffix}"].sum() / frame[f"requested_carriers_{suffix}"].sum())
        return float(frame[f"{metric}_{suffix}"].mean())

    control_value = value(paired, "control")
    candidate_value = value(paired, "candidate")
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
        deltas[index] = value(sample, "candidate") - value(sample, "control")
    low, high = np.percentile(deltas, [2.5, 97.5])
    return control_value, candidate_value, float(low), float(high)


def evaluate_candidates(
    grid_results: pd.DataFrame,
    grid_features: pd.DataFrame,
    definitions: list[dict],
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict] = []
    selected_results: dict[str, pd.DataFrame] = {}
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for definition in definitions:
        candidate_id = definition["candidate_id"]
        result, feature = select_candidate_rows(grid_results, grid_features, definition)
        selected_results[candidate_id] = result
        row = {
            "experiment_version": EXPERIMENT_VERSION,
            "candidate_id": candidate_id,
            "family": definition["family"],
            "feature_names": json.dumps(definition["feature_names"]),
            "thresholds": json.dumps(definition["thresholds"]),
            "alpha_values": json.dumps(definition["alpha_values"]),
            "replays": result["replay_file"].nunique(),
            "effective_transmitted_bits": int(result["message_bits"].sum()),
            "bit_errors_roundtrip": int(result["bit_errors_roundtrip"].sum()),
            "weighted_roundtrip_ber": float(
                result["bit_errors_roundtrip"].sum() / result["message_bits"].sum()
            ),
            **energy_summary(result),
            "alpha_assignment_counts": json.dumps(
                {str(k): int(v) for k, v in result["alpha"].value_counts().sort_index().items()},
                sort_keys=True,
            ),
        }
        for cv_group_column in ("replay_file", "beatmap_hash"):
            auc, prediction = evaluate_subset(
                feature, "ALL", None, N_VALUE, folds, trees, seed,
                bootstrap_iterations, cv_group_column=cv_group_column,
                classifier_jobs=1,
            )
            if cv_group_column == "beatmap_hash":
                low, high = bootstrap_auc_by_group(
                    prediction["label"].to_numpy(dtype=np.int8),
                    prediction["oof_probability_stego"].to_numpy(dtype=np.float64),
                    prediction["beatmap_hash"].astype(str).to_numpy(),
                    stable_seed(seed, EXPERIMENT_VERSION, candidate_id, "beatmap-bootstrap"),
                    bootstrap_iterations,
                )
                auc["auc_ci95_low"], auc["auc_ci95_high"] = low, high
            prefix = "replay" if cv_group_column == "replay_file" else "beatmap"
            row[f"{prefix}_grouped_auc"] = auc["roc_auc"]
            row[f"{prefix}_grouped_auc_ci95_low"] = auc["auc_ci95_low"]
            row[f"{prefix}_grouped_auc_ci95_high"] = auc["auc_ci95_high"]
            predictions[(candidate_id, cv_group_column)] = prediction
        summaries.append(row)

    paired_rows: list[dict] = []
    control = selected_results["fixed_15"]
    numeric_metrics = (
        "weighted_ber", "mean_energy_per_replay", "pooled_rms_shift",
        "active_carrier_fraction", "dropped_for_chronology", "new_unmatched_events",
    )
    for definition in definitions:
        candidate_id = definition["candidate_id"]
        if candidate_id == "fixed_15":
            continue
        for metric in numeric_metrics:
            control_value, candidate_value, low, high = bootstrap_numeric_delta(
                control, selected_results[candidate_id], metric,
                stable_seed(seed, EXPERIMENT_VERSION, candidate_id, metric),
                bootstrap_iterations,
            )
            paired_rows.append(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "candidate_id": candidate_id,
                    "control_method": "fixed_15",
                    "metric": metric,
                    "control_value": control_value,
                    "candidate_value": candidate_value,
                    "delta_candidate_minus_control": candidate_value - control_value,
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "bootstrap_unit": "replay_file",
                    "bootstrap_iterations": bootstrap_iterations,
                }
            )
        for cv_group_column in ("replay_file", "beatmap_hash"):
            control_prediction = predictions[("fixed_15", cv_group_column)]
            candidate_prediction = predictions[(candidate_id, cv_group_column)]
            if cv_group_column == "replay_file":
                low, high = bootstrap_auc_delta(
                    control_prediction, candidate_prediction,
                    stable_seed(seed, EXPERIMENT_VERSION, candidate_id, "replay-auc"),
                    bootstrap_iterations,
                )
                unit = "replay_file"
            else:
                low, high = bootstrap_auc_delta_clustered(
                    control_prediction.rename(columns={"oof_probability_stego": "score"}),
                    candidate_prediction.rename(columns={"oof_probability_stego": "score"}),
                    "beatmap_hash",
                    stable_seed(seed, EXPERIMENT_VERSION, candidate_id, "beatmap-auc"),
                    bootstrap_iterations,
                )
                unit = "beatmap_hash"
            control_auc = float(
                roc_auc_from_prediction(control_prediction)
            )
            candidate_auc = float(roc_auc_from_prediction(candidate_prediction))
            paired_rows.append(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "candidate_id": candidate_id,
                    "control_method": "fixed_15",
                    "metric": f"{cv_group_column}_auc",
                    "control_value": control_auc,
                    "candidate_value": candidate_auc,
                    "delta_candidate_minus_control": candidate_auc - control_auc,
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "bootstrap_unit": unit,
                    "bootstrap_iterations": bootstrap_iterations,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(paired_rows)


def roc_auc_from_prediction(prediction: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score

    return float(
        roc_auc_score(
            prediction["label"].to_numpy(dtype=np.int8),
            prediction["oof_probability_stego"].to_numpy(dtype=np.float64),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-only sender-local alpha search.")
    parser.add_argument("--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv")
    parser.add_argument("--performance-groups", type=Path, default=METADATA_DIR / "performance_groups.csv")
    parser.add_argument("--partition", type=Path, default=RESULTS_DIR / "adaptive_alpha_validation_v1" / "heldout_partition.csv")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hit-margin-ms", type=float, default=5.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_partitioned_cohort(
        args.results, args.performance_groups, args.partition, "development", args.limit
    )
    if args.limit is None and (len(cohort) != 611 or cohort["beatmap_hash"].nunique() != 23):
        raise ValueError("Frozen development cohort mora imati 611 replay-eva / 23 mape.")
    grid_results, grid_features, new_ok = run_physical_alpha_grid(
        cohort=cohort,
        alphas=ALPHA_GRID,
        dataset_dir=args.dataset_dir,
        map_offsets_path=args.map_offsets,
        source_results_path=args.results,
        performance_path=args.performance_groups,
        partition_path=args.partition,
        output_path=args.output_dir / "development_alpha_grid_results.csv",
        feature_path=args.output_dir / "development_alpha_grid_features.csv",
        seed=args.seed,
        hit_margin_ms=args.hit_margin_ms,
    )
    definitions = candidate_definitions(cohort)
    (args.output_dir / "development_candidate_definitions.json").write_text(
        json.dumps(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "grid_version": GRID_VERSION,
                "selection_partition": "development_only",
                "development_replays": len(cohort),
                "development_beatmaps": sorted(set(cohort["beatmap_hash"].astype(str))),
                "candidates": definitions,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    summary, paired = evaluate_candidates(
        grid_results, grid_features, definitions, args.folds, args.trees,
        args.seed, args.bootstrap_iterations,
    )
    summary.to_csv(args.output_dir / "development_policy_candidates.csv", index=False)
    paired.to_csv(args.output_dir / "development_policy_paired.csv", index=False)
    print(f"development replays={len(cohort)} maps={cohort['beatmap_hash'].nunique()} new OK={new_ok}")
    print(summary[["candidate_id", "weighted_roundtrip_ber", "replay_grouped_auc", "beatmap_grouped_auc", "rms_applied_shift_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
