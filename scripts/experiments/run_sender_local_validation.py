"""One-shot held-out validation of the frozen sender-local alpha policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import (
    alpha_from_sender_local_features,
    load_sender_local_policy,
)
from scripts.experiments.run_adaptive_alpha_comparison import ADAPTIVE_ALPHA, N_VALUE
from scripts.experiments.run_adaptive_alpha_validation import (
    bootstrap_auc_delta_clustered,
    energy_summary,
    file_sha256,
)
from scripts.experiments.run_layout_comparison import bootstrap_auc_delta
from scripts.experiments.run_pilot_ber_sweep import stable_seed
from scripts.experiments.run_steganalysis import (
    FEATURE_COLUMNS,
    bootstrap_auc_by_group,
    evaluate_subset,
)
from scripts.experiments.sender_local_common import (
    FEATURE_FIELDS,
    RESULT_FIELDS,
    load_partitioned_cohort,
    run_physical_alpha_grid,
)


EXPERIMENT_VERSION = "sender-local-heldout-validation-v1"
SHUFFLE_SEEDS = (101, 202, 303, 404, 505)
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "sender_local_adaptive_v1"
DEFAULT_POLICY = CONFIG_DIR / "adaptive_alpha_sender_local_v1.json"


def assignments_from_policy(cohort: pd.DataFrame, policy: dict) -> pd.DataFrame:
    assignments = cohort[["replay_file", "beatmap_hash", "accuracy"]].copy()
    assignments["selected_alpha"] = assignments["accuracy"].map(
        lambda value: alpha_from_sender_local_features(policy, accuracy=float(value))
    )
    return assignments


def oracle_assignments(cohort: pd.DataFrame) -> pd.DataFrame:
    assignments = cohort[
        ["replay_file", "beatmap_hash", "performance_category"]
    ].copy()
    assignments["selected_alpha"] = assignments["performance_category"].map(
        ADAPTIVE_ALPHA
    )
    return assignments


def shuffled_assignments(
    base: pd.DataFrame,
    shuffle_seed: int,
    partition_name: str,
) -> pd.DataFrame:
    output = base.sort_values("replay_file").reset_index(drop=True).copy()
    values = output["selected_alpha"].to_numpy(dtype=np.float64).copy()
    rng = np.random.default_rng(
        stable_seed(
            shuffle_seed,
            EXPERIMENT_VERSION,
            partition_name,
            "shuffled-alpha-assignment",
        )
    )
    rng.shuffle(values)
    output["selected_alpha"] = values
    if output["selected_alpha"].value_counts().to_dict() != base[
        "selected_alpha"
    ].value_counts().to_dict():
        raise AssertionError("Shuffle nije sačuvao alpha frekvencije.")
    return output


def select_method(
    grid_results: pd.DataFrame,
    grid_features: pd.DataFrame,
    assignments: pd.DataFrame,
    method: str,
    policy_sha256: str,
    shuffle_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = grid_results.merge(
        assignments[["replay_file", "selected_alpha"]],
        left_on=["replay_file", "alpha"],
        right_on=["replay_file", "selected_alpha"],
        how="inner",
        validate="many_to_one",
    )
    if selected["replay_file"].nunique() != assignments["replay_file"].nunique():
        raise ValueError(f"Method {method} ne pokriva svaki replay.")
    selected = selected.copy()
    selected["method"] = method
    selected["policy_sha256"] = policy_sha256
    selected["shuffle_seed"] = shuffle_seed
    features = grid_features[
        grid_features["config_id"].isin(set(selected["config_id"]))
    ].copy()
    features["method"] = method
    features["policy_sha256"] = policy_sha256
    features["shuffle_seed"] = shuffle_seed
    labels = features.groupby("replay_file")["label"].apply(
        lambda values: tuple(sorted(values.astype(int).tolist()))
    )
    if not all(value == (0, 1) for value in labels):
        raise ValueError(f"Method {method} nema tačne clean/stego parove.")
    return selected, features


def generalization_auc(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
    method: str,
) -> tuple[dict, pd.DataFrame]:
    if set(development["beatmap_hash"]) & set(validation["beatmap_hash"]):
        raise ValueError("Beatmap overlap u dev->validation evaluaciji.")
    classifier = RandomForestClassifier(
        n_estimators=trees,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=stable_seed(seed, EXPERIMENT_VERSION, "generalization-rf"),
        n_jobs=1,
    )
    classifier.fit(
        development[FEATURE_COLUMNS].to_numpy(dtype=np.float64),
        development["label"].to_numpy(dtype=np.int8),
    )
    scores = classifier.predict_proba(
        validation[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    )[:, 1]
    labels = validation["label"].to_numpy(dtype=np.int8)
    beatmaps = validation["beatmap_hash"].astype(str).to_numpy()
    auc = float(roc_auc_score(labels, scores))
    low, high = bootstrap_auc_by_group(
        labels,
        scores,
        beatmaps,
        stable_seed(seed, EXPERIMENT_VERSION, method, "generalization-bootstrap"),
        bootstrap_iterations,
    )
    prediction = validation[["replay_file", "beatmap_hash", "label"]].copy()
    prediction["score"] = scores
    return {
        "roc_auc": auc,
        "auc_ci95_low": low,
        "auc_ci95_high": high,
        "bootstrap_unit": "beatmap_hash",
    }, prediction


def paired_physical_rows(
    control: pd.DataFrame,
    candidate: pd.DataFrame,
    comparison: str,
    seed: int,
    iterations: int,
) -> list[dict]:
    columns = [
        "replay_file", "beatmap_hash", "message_bits", "bit_errors_roundtrip",
        "requested_carriers", "active_carriers", "sum_squared_shift_ms2",
        "total_absolute_shift_ms", "dropped_for_hit_window",
        "dropped_for_chronology", "new_unmatched_events", "new_matched_events",
        "changed_match_status_total",
    ]
    paired = control[columns].merge(
        candidate[columns],
        on=["replay_file", "beatmap_hash"],
        suffixes=("_control", "_candidate"),
        validate="one_to_one",
    )
    clusters = paired["beatmap_hash"].astype(str).unique()
    pieces = {
        value: paired[paired["beatmap_hash"].astype(str) == value]
        for value in clusters
    }

    def value(frame: pd.DataFrame, metric: str, suffix: str) -> float:
        if metric == "weighted_ber":
            return float(frame[f"bit_errors_roundtrip_{suffix}"].sum() / frame[f"message_bits_{suffix}"].sum())
        if metric == "mean_energy_per_replay":
            return float(frame[f"sum_squared_shift_ms2_{suffix}"].mean())
        if metric == "pooled_rms_shift":
            return float(np.sqrt(frame[f"sum_squared_shift_ms2_{suffix}"].sum() / frame[f"active_carriers_{suffix}"].sum()))
        if metric == "pooled_mean_absolute_shift":
            return float(frame[f"total_absolute_shift_ms_{suffix}"].sum() / frame[f"active_carriers_{suffix}"].sum())
        if metric == "active_carrier_fraction":
            return float(frame[f"active_carriers_{suffix}"].sum() / frame[f"requested_carriers_{suffix}"].sum())
        return float(frame[f"{metric}_{suffix}"].mean())

    metrics = (
        "weighted_ber", "mean_energy_per_replay", "pooled_rms_shift",
        "pooled_mean_absolute_shift", "active_carrier_fraction",
        "dropped_for_hit_window", "dropped_for_chronology",
        "new_unmatched_events", "new_matched_events", "changed_match_status_total",
    )
    rows = []
    for metric in metrics:
        control_value = value(paired, metric, "control")
        candidate_value = value(paired, metric, "candidate")
        rng = np.random.default_rng(
            stable_seed(seed, EXPERIMENT_VERSION, comparison, metric, "beatmap-bootstrap")
        )
        deltas = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            sample = pd.concat([pieces[item] for item in sampled], ignore_index=True)
            deltas[index] = value(sample, metric, "candidate") - value(sample, metric, "control")
        low, high = np.percentile(deltas, [2.5, 97.5])
        rows.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "comparison": comparison,
                "metric": metric,
                "control_value": control_value,
                "candidate_value": candidate_value,
                "delta_candidate_minus_control": candidate_value - control_value,
                "delta_ci95_low": float(low),
                "delta_ci95_high": float(high),
                "bootstrap_unit": "beatmap_hash",
                "bootstrap_iterations": iterations,
                "replay_pairs": len(paired),
                "beatmap_clusters": len(clusters),
            }
        )
    return rows


def create_or_validate_lock(
    lock_path: Path,
    policy_sha256: str,
    validation_maps: list[str],
) -> None:
    lock = {
        "experiment_version": EXPERIMENT_VERSION,
        "policy_sha256": policy_sha256,
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "validation_beatmap_hashes": validation_maps,
        "status": "one_shot_started",
    }
    if lock_path.is_file():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing != lock:
            raise ValueError("Held-out validation lock pripada drugoj konfiguraciji.")
        return
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot sender-local held-out validation.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv")
    parser.add_argument("--performance-groups", type=Path, default=METADATA_DIR / "performance_groups.csv")
    parser.add_argument("--partition", type=Path, default=RESULTS_DIR / "adaptive_alpha_validation_v1" / "heldout_partition.csv")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json")
    parser.add_argument("--development-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_sender_local_policy(args.policy)
    policy_sha256 = file_sha256(args.policy)
    if policy["policy_version"] != "adaptive-alpha-sender-local-v1" or not policy["frozen"]:
        raise ValueError("Held-out run zahteva frozen sender-local v1 policy.")
    cohort = load_partitioned_cohort(
        args.results, args.performance_groups, args.partition, "validation", args.limit
    )
    if args.limit is None and (len(cohort) != 338 or cohort["beatmap_hash"].nunique() != 11):
        raise ValueError("Frozen validation cohort mora imati 338 replay-eva / 11 mapa.")
    if args.limit is None:
        if set(cohort["beatmap_hash"].astype(str)) != set(policy["validation_beatmap_hashes"]):
            raise ValueError("Policy config validation mape nisu trenutna frozen particija.")
        create_or_validate_lock(
            args.output_dir / "validation_one_shot_lock.json",
            policy_sha256,
            sorted(set(cohort["beatmap_hash"].astype(str))),
        )

    sender_assignment = assignments_from_policy(cohort, policy)
    needed_alphas = tuple(
        sorted(
            set(sender_assignment["selected_alpha"].astype(float))
            | {15.0}
            | set(float(value) for value in ADAPTIVE_ALPHA.values())
        )
    )
    grid_results, grid_features, new_ok = run_physical_alpha_grid(
        cohort=cohort,
        alphas=needed_alphas,
        dataset_dir=args.dataset_dir,
        map_offsets_path=args.map_offsets,
        source_results_path=args.results,
        performance_path=args.performance_groups,
        partition_path=args.partition,
        output_path=args.output_dir / "validation_alpha_grid_results.csv",
        feature_path=args.output_dir / "validation_alpha_grid_features.csv",
        seed=args.seed,
        hit_margin_ms=float(policy["hit_margin_ms"]),
    )

    development_results = pd.read_csv(args.development_dir / "development_alpha_grid_results.csv")
    development_features = pd.read_csv(args.development_dir / "development_alpha_grid_features.csv")
    if set(development_results["partition"]) != {"development"}:
        raise ValueError("Training grid nije development-only.")
    development_replay = development_results[
        ["replay_file", "beatmap_hash", "accuracy", "performance_category"]
    ].drop_duplicates("replay_file")
    development_sender = assignments_from_policy(development_replay, policy)

    validation_assignments: dict[str, pd.DataFrame] = {
        "fixed_15": sender_assignment.assign(selected_alpha=15.0),
        "sender_local_adaptive": sender_assignment,
        "oracle_map_relative": oracle_assignments(cohort),
    }
    development_assignments: dict[str, pd.DataFrame] = {
        "fixed_15": development_sender.assign(selected_alpha=15.0),
        "sender_local_adaptive": development_sender,
        "oracle_map_relative": oracle_assignments(development_replay),
    }
    for shuffle_seed in SHUFFLE_SEEDS:
        method = f"shuffled_alpha_seed_{shuffle_seed}"
        validation_assignments[method] = shuffled_assignments(
            sender_assignment, shuffle_seed, "validation"
        )
        development_assignments[method] = shuffled_assignments(
            development_sender, shuffle_seed, "development"
        )

    validation_method_results: list[pd.DataFrame] = []
    validation_method_features: dict[str, pd.DataFrame] = {}
    development_method_features: dict[str, pd.DataFrame] = {}
    for method, assignment in validation_assignments.items():
        shuffle_seed = int(method.rsplit("_", 1)[-1]) if method.startswith("shuffled") else None
        result, feature = select_method(
            grid_results, grid_features, assignment, method, policy_sha256, shuffle_seed
        )
        validation_method_results.append(result)
        validation_method_features[method] = feature
        _, development_feature = select_method(
            development_results,
            development_features,
            development_assignments[method],
            method,
            policy_sha256,
            shuffle_seed,
        )
        development_method_features[method] = development_feature

    validation_results = pd.concat(validation_method_results, ignore_index=True)
    validation_results.to_csv(args.output_dir / "validation_results.csv", index=False)
    pd.concat(validation_method_features.values(), ignore_index=True).to_csv(
        args.output_dir / "validation_method_features.csv", index=False
    )

    method_rows: list[dict] = []
    replay_predictions: dict[str, pd.DataFrame] = {}
    generalization_predictions: dict[str, pd.DataFrame] = {}
    for method in validation_assignments:
        result = validation_results[validation_results["method"] == method]
        bits = int(result["message_bits"].sum())
        replay_auc, replay_prediction = evaluate_subset(
            validation_method_features[method], "ALL", None, N_VALUE,
            args.folds, args.trees, args.seed, args.bootstrap_iterations,
            cv_group_column="replay_file", classifier_jobs=1,
        )
        generalization, generalization_prediction = generalization_auc(
            development_method_features[method], validation_method_features[method],
            args.trees, args.seed, args.bootstrap_iterations, method,
        )
        replay_predictions[method] = replay_prediction
        generalization_predictions[method] = generalization_prediction
        method_rows.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "policy_sha256": policy_sha256,
                "method": method,
                "replays": result["replay_file"].nunique(),
                "bit_errors_roundtrip": int(result["bit_errors_roundtrip"].sum()),
                "total_bits": bits,
                "weighted_roundtrip_ber": float(result["bit_errors_roundtrip"].sum() / bits),
                "replay_grouped_auc": replay_auc["roc_auc"],
                "replay_grouped_auc_ci95_low": replay_auc["auc_ci95_low"],
                "replay_grouped_auc_ci95_high": replay_auc["auc_ci95_high"],
                "dev_to_validation_auc": generalization["roc_auc"],
                "dev_to_validation_auc_ci95_low": generalization["auc_ci95_low"],
                "dev_to_validation_auc_ci95_high": generalization["auc_ci95_high"],
                "alpha_assignment_counts": json.dumps(
                    {str(k): int(v) for k, v in result["alpha"].value_counts().sort_index().items()},
                    sort_keys=True,
                ),
                **energy_summary(result),
            }
        )
    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(args.output_dir / "validation_auc.csv", index=False)
    method_summary.to_csv(args.output_dir / "physical_energy.csv", index=False)

    paired_rows: list[dict] = []
    comparisons = (
        ("fixed_15", "sender_local_adaptive"),
        ("fixed_15", "oracle_map_relative"),
        ("oracle_map_relative", "sender_local_adaptive"),
    )
    for control_method, candidate_method in comparisons:
        control_result = validation_results[
            validation_results["method"] == control_method
        ]
        candidate_result = validation_results[
            validation_results["method"] == candidate_method
        ]
        comparison = f"{candidate_method}_minus_{control_method}"
        paired_rows.extend(
            paired_physical_rows(
                control_result, candidate_result, comparison,
                args.seed, args.bootstrap_iterations,
            )
        )
        for scope in ("replay_grouped_auc", "dev_to_validation_auc"):
            if scope == "replay_grouped_auc":
                low, high = bootstrap_auc_delta(
                    replay_predictions[control_method], replay_predictions[candidate_method],
                    stable_seed(args.seed, EXPERIMENT_VERSION, comparison, scope),
                    args.bootstrap_iterations,
                )
                control_auc = roc_auc_score(
                    replay_predictions[control_method]["label"],
                    replay_predictions[control_method]["oof_probability_stego"],
                )
                candidate_auc = roc_auc_score(
                    replay_predictions[candidate_method]["label"],
                    replay_predictions[candidate_method]["oof_probability_stego"],
                )
                unit = "replay_file"
            else:
                low, high = bootstrap_auc_delta_clustered(
                    generalization_predictions[control_method],
                    generalization_predictions[candidate_method],
                    "beatmap_hash",
                    stable_seed(args.seed, EXPERIMENT_VERSION, comparison, scope),
                    args.bootstrap_iterations,
                )
                control_auc = roc_auc_score(
                    generalization_predictions[control_method]["label"],
                    generalization_predictions[control_method]["score"],
                )
                candidate_auc = roc_auc_score(
                    generalization_predictions[candidate_method]["label"],
                    generalization_predictions[candidate_method]["score"],
                )
                unit = "beatmap_hash"
            paired_rows.append(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "comparison": comparison,
                    "metric": scope,
                    "control_value": float(control_auc),
                    "candidate_value": float(candidate_auc),
                    "delta_candidate_minus_control": float(candidate_auc - control_auc),
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "bootstrap_unit": unit,
                    "bootstrap_iterations": args.bootstrap_iterations,
                    "replay_pairs": len(cohort),
                    "beatmap_clusters": cohort["beatmap_hash"].nunique(),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.output_dir / "validation_paired.csv", index=False)

    shuffled_rows: list[dict] = []
    sender_result = validation_results[
        validation_results["method"] == "sender_local_adaptive"
    ]
    for shuffle_seed in SHUFFLE_SEEDS:
        method = f"shuffled_alpha_seed_{shuffle_seed}"
        shuffled_result = validation_results[validation_results["method"] == method]
        physical = paired_physical_rows(
            shuffled_result,
            sender_result,
            f"sender_local_minus_{method}",
            args.seed,
            args.bootstrap_iterations,
        )
        for row in physical:
            row["shuffle_seed"] = shuffle_seed
            shuffled_rows.append(row)
        for scope in ("replay_grouped_auc", "dev_to_validation_auc"):
            if scope == "replay_grouped_auc":
                low, high = bootstrap_auc_delta(
                    replay_predictions[method], replay_predictions["sender_local_adaptive"],
                    stable_seed(args.seed, EXPERIMENT_VERSION, method, scope),
                    args.bootstrap_iterations,
                )
                control_auc = roc_auc_score(
                    replay_predictions[method]["label"],
                    replay_predictions[method]["oof_probability_stego"],
                )
                candidate_auc = roc_auc_score(
                    replay_predictions["sender_local_adaptive"]["label"],
                    replay_predictions["sender_local_adaptive"]["oof_probability_stego"],
                )
                unit = "replay_file"
            else:
                low, high = bootstrap_auc_delta_clustered(
                    generalization_predictions[method],
                    generalization_predictions["sender_local_adaptive"],
                    "beatmap_hash",
                    stable_seed(args.seed, EXPERIMENT_VERSION, method, scope),
                    args.bootstrap_iterations,
                )
                control_auc = roc_auc_score(
                    generalization_predictions[method]["label"],
                    generalization_predictions[method]["score"],
                )
                candidate_auc = roc_auc_score(
                    generalization_predictions["sender_local_adaptive"]["label"],
                    generalization_predictions["sender_local_adaptive"]["score"],
                )
                unit = "beatmap_hash"
            shuffled_rows.append(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "comparison": f"sender_local_minus_{method}",
                    "shuffle_seed": shuffle_seed,
                    "metric": scope,
                    "control_value": float(control_auc),
                    "candidate_value": float(candidate_auc),
                    "delta_candidate_minus_control": float(candidate_auc - control_auc),
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "bootstrap_unit": unit,
                    "bootstrap_iterations": args.bootstrap_iterations,
                    "replay_pairs": len(cohort),
                    "beatmap_clusters": cohort["beatmap_hash"].nunique(),
                }
            )
    shuffled_frame = pd.DataFrame(shuffled_rows)
    shuffled_frame.to_csv(
        args.output_dir / "shuffled_alpha_control.csv", index=False
    )
    aggregate_rows = []
    for metric, group in shuffled_frame.groupby("metric", sort=False):
        aggregate_rows.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "metric": metric,
                "shuffle_seeds": json.dumps(list(SHUFFLE_SEEDS)),
                "seed_count": len(group),
                "mean_shuffled_control_value": float(group["control_value"].mean()),
                "min_shuffled_control_value": float(group["control_value"].min()),
                "max_shuffled_control_value": float(group["control_value"].max()),
                "sender_local_value": float(group["candidate_value"].iloc[0]),
                "mean_delta_sender_minus_shuffled": float(
                    group["delta_candidate_minus_control"].mean()
                ),
                "min_delta_sender_minus_shuffled": float(
                    group["delta_candidate_minus_control"].min()
                ),
                "max_delta_sender_minus_shuffled": float(
                    group["delta_candidate_minus_control"].max()
                ),
                "seeds_ci_strictly_below_zero": int(
                    np.sum(group["delta_ci95_high"] < 0.0)
                ),
                "seeds_ci_contains_zero": int(
                    np.sum(
                        (group["delta_ci95_low"] <= 0.0)
                        & (group["delta_ci95_high"] >= 0.0)
                    )
                ),
                "bootstrap_unit": str(group["bootstrap_unit"].iloc[0]),
            }
        )
    pd.DataFrame(aggregate_rows).to_csv(
        args.output_dir / "shuffled_alpha_control_summary.csv", index=False
    )
    print(f"validation replays={len(cohort)} maps={cohort['beatmap_hash'].nunique()} new OK={new_ok}")
    print(method_summary[["method", "weighted_roundtrip_ber", "replay_grouped_auc", "dev_to_validation_auc", "rms_applied_shift_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
