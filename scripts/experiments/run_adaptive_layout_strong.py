"""Multi-seed PREFIX vs DISTRIBUTED test under frozen strong features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from osu_stego.analysis.layout_diagnostics import QUARTERS
from osu_stego.analysis.timing_features import (
    BASELINE_FEATURES,
    FULL_FEATURES,
    POSITION_FEATURES,
)
from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR
from scripts.experiments.adaptive_layout_strong_common import (
    ALPHA_METHODS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARTITION,
    DEFAULT_POLICY,
    EXPERIMENT_VERSION,
    FEATURE_FIELDS,
    LAYOUT_KEYS,
    MESSAGE_SEED,
    N_VALUE,
    RESULT_FIELDS,
    SENDER_SOURCE_DIR,
    experiment_config,
    generate_partition,
    method_configs,
    validate_physical_outputs,
    write_or_validate_config,
)
from scripts.experiments.run_adaptive_alpha_validation import (
    bootstrap_auc_delta_clustered,
)
from scripts.experiments.run_layout_comparison import bootstrap_auc_delta
from scripts.experiments.run_pilot_ber_sweep import stable_seed
from scripts.experiments.run_steganalysis import bootstrap_auc_by_group, shuffled_group_ids
from scripts.experiments.sender_local_common import load_partitioned_cohort


FEATURE_ABLATIONS = {
    "baseline": tuple(BASELINE_FEATURES),
    "position_only": tuple(POSITION_FEATURES),
    "full": tuple(FULL_FEATURES),
}
LEGACY_STRONG_EXPERIMENT = "sender-local-heldout-validation-v1"


def frozen_ordered_subset(
    features: pd.DataFrame,
    partition: str,
    alpha_method: str,
    layout: str,
    layout_seed: str,
) -> pd.DataFrame:
    subset = features[
        (features["partition"] == partition)
        & (features["alpha_method"] == alpha_method)
        & (features["layout"] == layout)
        & (features["layout_seed"].astype(str) == str(layout_seed))
    ].copy()
    source = pd.read_csv(
        SENDER_SOURCE_DIR / f"{partition}_alpha_grid_features.csv"
    )
    replay_order: dict[tuple[str, int], int] = {}
    for index, row in enumerate(source.itertuples(index=False)):
        replay_order.setdefault((str(row.replay_file), int(row.label)), index)
    keys = list(zip(subset["replay_file"].astype(str), subset["label"].astype(int)))
    try:
        subset["_source_order"] = [replay_order[key] for key in keys]
    except KeyError as exc:
        raise ValueError("Layout feature red nije u frozen source cohort-u.") from exc
    return subset.sort_values("_source_order").drop(columns="_source_order").reset_index(
        drop=True
    )


def _rf(random_state: int, trees: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trees,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )


def replay_auc(
    subset: pd.DataFrame,
    feature_names: tuple[str, ...],
    feature_set: str,
    alpha_method: str,
    layout: str,
    layout_seed: str,
    folds: int,
    trees: int,
    iterations: int,
) -> tuple[dict, pd.DataFrame]:
    X = subset[list(feature_names)].to_numpy(dtype=float)
    labels = subset["label"].to_numpy(dtype=np.int8)
    replay_groups = subset["replay_file"].astype(str).to_numpy()
    groups = shuffled_group_ids(
        replay_groups,
        stable_seed(MESSAGE_SEED, "ALL", None, N_VALUE, "folds"),
    )
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    scores = np.full(len(subset), np.nan, dtype=float)
    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(X, labels, groups), 1
    ):
        classifier = _rf(
            stable_seed(MESSAGE_SEED, "ALL", None, N_VALUE, fold_index, "rf"),
            trees,
        )
        classifier.fit(X[train_index], labels[train_index])
        scores[test_index] = classifier.predict_proba(X[test_index])[:, 1]
    auc = float(roc_auc_score(labels, scores))
    low, high = bootstrap_auc_by_group(
        labels,
        scores,
        replay_groups,
        stable_seed(
            MESSAGE_SEED,
            EXPERIMENT_VERSION,
            alpha_method,
            layout,
            layout_seed,
            feature_set,
            "replay-ci",
        ),
        iterations,
    )
    prediction = subset[["replay_file", "beatmap_hash", "label"]].copy()
    prediction["oof_probability_stego"] = scores
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "scope": "validation_replay_grouped",
        "feature_set": feature_set,
        "alpha_method": alpha_method,
        "layout": layout,
        "layout_seed": layout_seed,
        "roc_auc": auc,
        "auc_ci95_low": low,
        "auc_ci95_high": high,
        "bootstrap_unit": "replay_file",
        "replay_pairs": subset["replay_file"].nunique(),
        "folds": splitter.n_splits,
        "trees": trees,
        "rf_n_jobs": 1,
    }, prediction


def generalization_auc(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: tuple[str, ...],
    feature_set: str,
    alpha_method: str,
    layout: str,
    layout_seed: str,
    trees: int,
    iterations: int,
) -> tuple[dict, pd.DataFrame]:
    if set(development["beatmap_hash"]) & set(validation["beatmap_hash"]):
        raise ValueError("Development/validation beatmap overlap.")
    classifier = _rf(
        stable_seed(MESSAGE_SEED, LEGACY_STRONG_EXPERIMENT, "generalization-rf"),
        trees,
    )
    classifier.fit(
        development[list(feature_names)].to_numpy(dtype=float),
        development["label"].to_numpy(dtype=np.int8),
    )
    scores = classifier.predict_proba(
        validation[list(feature_names)].to_numpy(dtype=float)
    )[:, 1]
    labels = validation["label"].to_numpy(dtype=np.int8)
    beatmaps = validation["beatmap_hash"].astype(str).to_numpy()
    auc = float(roc_auc_score(labels, scores))
    low, high = bootstrap_auc_by_group(
        labels,
        scores,
        beatmaps,
        stable_seed(
            MESSAGE_SEED,
            EXPERIMENT_VERSION,
            alpha_method,
            layout,
            layout_seed,
            feature_set,
            "map-ci",
        ),
        iterations,
    )
    prediction = validation[["replay_file", "beatmap_hash", "label"]].copy()
    prediction["score"] = scores
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "scope": "development_to_validation",
        "feature_set": feature_set,
        "alpha_method": alpha_method,
        "layout": layout,
        "layout_seed": layout_seed,
        "roc_auc": auc,
        "auc_ci95_low": low,
        "auc_ci95_high": high,
        "bootstrap_unit": "beatmap_hash",
        "development_maps": development["beatmap_hash"].nunique(),
        "validation_maps": validation["beatmap_hash"].nunique(),
        "validation_replays": validation["replay_file"].nunique(),
        "trees": trees,
        "rf_n_jobs": 1,
    }, prediction


def evaluate_auc(
    features: pd.DataFrame,
    output_dir: Path,
    folds: int,
    trees: int,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    replay_predictions: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    general_predictions: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    for feature_set, names in FEATURE_ABLATIONS.items():
        for method in method_configs():
            key = (
                method["alpha_method"],
                method["layout"],
                method["layout_seed"],
                feature_set,
            )
            development = frozen_ordered_subset(
                features,
                "development",
                method["alpha_method"],
                method["layout"],
                method["layout_seed"],
            )
            validation = frozen_ordered_subset(
                features,
                "validation",
                method["alpha_method"],
                method["layout"],
                method["layout_seed"],
            )
            replay_row, replay_prediction = replay_auc(
                validation,
                names,
                feature_set,
                method["alpha_method"],
                method["layout"],
                method["layout_seed"],
                folds,
                trees,
                iterations,
            )
            general_row, general_prediction = generalization_auc(
                development,
                validation,
                names,
                feature_set,
                method["alpha_method"],
                method["layout"],
                method["layout_seed"],
                trees,
                iterations,
            )
            rows.extend([replay_row, general_row])
            replay_predictions[key] = replay_prediction
            general_predictions[key] = general_prediction

    paired_replay = []
    paired_general = []
    for feature_set in FEATURE_ABLATIONS:
        for alpha_method in ALPHA_METHODS:
            prefix_key = (alpha_method, "prefix", "shared_prefix", feature_set)
            prefix_replay = replay_predictions[prefix_key]
            prefix_general = general_predictions[prefix_key]
            for seed_index in range(len(LAYOUT_KEYS)):
                distributed_key = (
                    alpha_method,
                    "distributed",
                    str(seed_index),
                    feature_set,
                )
                distributed_replay = replay_predictions[distributed_key]
                distributed_general = general_predictions[distributed_key]
                prefix_auc = float(
                    roc_auc_score(
                        prefix_replay["label"],
                        prefix_replay["oof_probability_stego"],
                    )
                )
                distributed_auc = float(
                    roc_auc_score(
                        distributed_replay["label"],
                        distributed_replay["oof_probability_stego"],
                    )
                )
                low, high = bootstrap_auc_delta(
                    prefix_replay,
                    distributed_replay,
                    stable_seed(
                        MESSAGE_SEED,
                        EXPERIMENT_VERSION,
                        alpha_method,
                        seed_index,
                        feature_set,
                        "paired-replay-auc",
                    ),
                    iterations,
                )
                paired_replay.append(
                    {
                        "experiment_version": EXPERIMENT_VERSION,
                        "feature_set": feature_set,
                        "alpha_method": alpha_method,
                        "layout_seed": seed_index,
                        "prefix_auc": prefix_auc,
                        "distributed_auc": distributed_auc,
                        "delta_distributed_minus_prefix": distributed_auc - prefix_auc,
                        "delta_ci95_low": low,
                        "delta_ci95_high": high,
                        "bootstrap_unit": "replay_file",
                        "bootstrap_iterations": iterations,
                    }
                )
                prefix_gen_auc = float(
                    roc_auc_score(prefix_general["label"], prefix_general["score"])
                )
                distributed_gen_auc = float(
                    roc_auc_score(
                        distributed_general["label"], distributed_general["score"]
                    )
                )
                low, high = bootstrap_auc_delta_clustered(
                    prefix_general,
                    distributed_general,
                    "beatmap_hash",
                    stable_seed(
                        MESSAGE_SEED,
                        EXPERIMENT_VERSION,
                        alpha_method,
                        seed_index,
                        feature_set,
                        "paired-map-auc",
                    ),
                    iterations,
                )
                paired_general.append(
                    {
                        "experiment_version": EXPERIMENT_VERSION,
                        "feature_set": feature_set,
                        "alpha_method": alpha_method,
                        "layout_seed": seed_index,
                        "prefix_auc": prefix_gen_auc,
                        "distributed_auc": distributed_gen_auc,
                        "delta_distributed_minus_prefix": (
                            distributed_gen_auc - prefix_gen_auc
                        ),
                        "delta_ci95_low": low,
                        "delta_ci95_high": high,
                        "bootstrap_unit": "beatmap_hash",
                        "bootstrap_iterations": iterations,
                    }
                )
    all_auc = pd.DataFrame(rows)
    paired_replay_frame = pd.DataFrame(paired_replay)
    paired_general_frame = pd.DataFrame(paired_general)
    all_auc.to_csv(output_dir / "layout_feature_ablation.csv", index=False)
    all_auc[
        (all_auc["feature_set"] == "full")
        & (all_auc["scope"] == "validation_replay_grouped")
    ].to_csv(output_dir / "auc_by_seed.csv", index=False)
    paired_replay_frame[paired_replay_frame["feature_set"] == "full"].to_csv(
        output_dir / "auc_paired.csv", index=False
    )
    all_auc[
        (all_auc["feature_set"] == "full")
        & (all_auc["scope"] == "development_to_validation")
    ].to_csv(output_dir / "dev_to_validation_auc_by_seed.csv", index=False)
    paired_general_frame[paired_general_frame["feature_set"] == "full"].to_csv(
        output_dir / "dev_to_validation_paired.csv", index=False
    )
    return all_auc, paired_replay_frame, paired_general_frame, pd.concat(
        [
            frame.assign(prediction_scope=scope)
            for scope, prediction_dict in (
                ("replay", replay_predictions),
                ("generalization", general_predictions),
            )
            for key, frame in prediction_dict.items()
            if key[-1] == "full"
        ],
        ignore_index=True,
    )


def _paired_results(
    results: pd.DataFrame,
    partition: str,
    alpha_method: str,
    seed_index: int,
) -> pd.DataFrame:
    columns = [
        "replay_file",
        "beatmap_hash",
        "message_bits",
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
        "rms_applied_shift_ms",
    ]
    prefix = results[
        (results["partition"] == partition)
        & (results["alpha_method"] == alpha_method)
        & (results["layout"] == "prefix")
    ][columns]
    distributed = results[
        (results["partition"] == partition)
        & (results["alpha_method"] == alpha_method)
        & (results["layout"] == "distributed")
        & (results["layout_seed"].astype(str) == str(seed_index))
    ][columns]
    return prefix.merge(
        distributed,
        on=["replay_file", "beatmap_hash"],
        suffixes=("_prefix", "_distributed"),
        validate="one_to_one",
    )


def _physical_value(frame: pd.DataFrame, metric: str, suffix: str) -> float:
    if metric == "weighted_roundtrip_ber":
        return float(
            frame[f"bit_errors_roundtrip_{suffix}"].sum()
            / frame[f"message_bits_{suffix}"].sum()
        )
    if metric == "active_carrier_fraction":
        return float(
            frame[f"active_carriers_{suffix}"].sum()
            / frame[f"requested_carriers_{suffix}"].sum()
        )
    if metric == "pooled_rms_shift_ms":
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
    if metric == "mean_energy_per_replay":
        return float(frame[f"sum_squared_shift_ms2_{suffix}"].mean())
    return float(frame[f"{metric}_{suffix}"].mean())


def physical_paired_rows(
    results: pd.DataFrame,
    iterations: int,
) -> pd.DataFrame:
    metrics = (
        "weighted_roundtrip_ber",
        "active_carrier_fraction",
        "dropped_for_hit_window",
        "dropped_for_chronology",
        "new_unmatched_events",
        "new_matched_events",
        "changed_match_status_total",
        "mean_energy_per_replay",
        "pooled_rms_shift_ms",
        "pooled_mean_absolute_shift_ms",
    )
    rows = []
    for partition in ("development", "validation"):
        for alpha_method in ALPHA_METHODS:
            for seed_index in range(len(LAYOUT_KEYS)):
                paired = _paired_results(results, partition, alpha_method, seed_index)
                clusters = paired["beatmap_hash"].astype(str).unique()
                pieces = {
                    cluster: paired[paired["beatmap_hash"].astype(str) == cluster]
                    for cluster in clusters
                }
                for metric in metrics:
                    prefix_value = _physical_value(paired, metric, "prefix")
                    distributed_value = _physical_value(paired, metric, "distributed")
                    rng = np.random.default_rng(
                        stable_seed(
                            MESSAGE_SEED,
                            EXPERIMENT_VERSION,
                            partition,
                            alpha_method,
                            seed_index,
                            metric,
                            "physical-bootstrap",
                        )
                    )
                    deltas = []
                    for _ in range(iterations):
                        sampled = rng.choice(clusters, len(clusters), replace=True)
                        sample = pd.concat([pieces[value] for value in sampled])
                        deltas.append(
                            _physical_value(sample, metric, "distributed")
                            - _physical_value(sample, metric, "prefix")
                        )
                    low, high = np.percentile(deltas, [2.5, 97.5])
                    rows.append(
                        {
                            "experiment_version": EXPERIMENT_VERSION,
                            "partition": partition,
                            "alpha_method": alpha_method,
                            "layout_seed": seed_index,
                            "metric": metric,
                            "prefix_value": prefix_value,
                            "distributed_value": distributed_value,
                            "delta_distributed_minus_prefix": (
                                distributed_value - prefix_value
                            ),
                            "delta_ci95_low": float(low),
                            "delta_ci95_high": float(high),
                            "bootstrap_unit": "beatmap_hash",
                            "bootstrap_iterations": iterations,
                            "replay_pairs": len(paired),
                            "beatmap_clusters": len(clusters),
                        }
                    )
    return pd.DataFrame(rows)


def _cluster_interval(
    frame: pd.DataFrame,
    value_column: str,
    statistic: str,
    seed_parts: tuple[object, ...],
    iterations: int,
) -> tuple[float, float]:
    clusters = frame["beatmap_hash"].astype(str).unique()
    pieces = {
        value: frame[frame["beatmap_hash"].astype(str) == value] for value in clusters
    }
    rng = np.random.default_rng(stable_seed(*seed_parts))
    values = []
    for _ in range(iterations):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        sample = pd.concat([pieces[value] for value in sampled])
        vector = sample[value_column].to_numpy(dtype=float)
        values.append(float(np.mean(vector) if statistic == "mean" else np.median(vector)))
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def quarter_feature_rows(features: pd.DataFrame, iterations: int) -> pd.DataFrame:
    rows = []
    for partition in ("development", "validation"):
        for alpha_method in ALPHA_METHODS:
            prefix = frozen_ordered_subset(
                features, partition, alpha_method, "prefix", "shared_prefix"
            )
            for seed_index in range(len(LAYOUT_KEYS)):
                distributed = frozen_ordered_subset(
                    features, partition, alpha_method, "distributed", str(seed_index)
                )
                delta_frames = {}
                for layout, frame in (("prefix", prefix), ("distributed", distributed)):
                    clean = frame[frame["label"] == 0].set_index("replay_file")
                    stego = frame[frame["label"] == 1].set_index("replay_file")
                    delta = stego[list(POSITION_FEATURES)] - clean[list(POSITION_FEATURES)]
                    delta["beatmap_hash"] = stego["beatmap_hash"]
                    delta["replay_file"] = delta.index
                    delta_frames[layout] = delta.reset_index(drop=True)
                comparison = delta_frames["prefix"].merge(
                    delta_frames["distributed"],
                    on=["replay_file", "beatmap_hash"],
                    suffixes=("_prefix", "_distributed"),
                    validate="one_to_one",
                )
                for layout, delta in delta_frames.items():
                    if layout == "prefix" and seed_index > 0:
                        continue
                    for feature in POSITION_FEATURES:
                        for statistic in ("mean", "median"):
                            point = float(
                                delta[feature].mean()
                                if statistic == "mean"
                                else delta[feature].median()
                            )
                            low, high = _cluster_interval(
                                delta,
                                feature,
                                statistic,
                                (
                                    MESSAGE_SEED,
                                    EXPERIMENT_VERSION,
                                    partition,
                                    alpha_method,
                                    seed_index,
                                    layout,
                                    feature,
                                    statistic,
                                ),
                                iterations,
                            )
                            row = {
                                "experiment_version": EXPERIMENT_VERSION,
                                "partition": partition,
                                "alpha_method": alpha_method,
                                "layout": layout,
                                "layout_seed": (
                                    "shared_prefix" if layout == "prefix" else seed_index
                                ),
                                "feature": feature,
                                "statistic": statistic,
                                "clean_to_stego_delta": point,
                                "delta_ci95_low": low,
                                "delta_ci95_high": high,
                                "bootstrap_unit": "beatmap_hash",
                                "replay_pairs": len(delta),
                            }
                            if layout == "distributed":
                                difference_column = f"{feature}_difference"
                                comparison[difference_column] = (
                                    comparison[f"{feature}_distributed"]
                                    - comparison[f"{feature}_prefix"]
                                )
                                comp_point = float(
                                    comparison[difference_column].mean()
                                    if statistic == "mean"
                                    else comparison[difference_column].median()
                                )
                                comp_low, comp_high = _cluster_interval(
                                    comparison,
                                    difference_column,
                                    statistic,
                                    (
                                        MESSAGE_SEED,
                                        EXPERIMENT_VERSION,
                                        partition,
                                        alpha_method,
                                        seed_index,
                                        feature,
                                        statistic,
                                        "distributed-minus-prefix",
                                    ),
                                    iterations,
                                )
                                row.update(
                                    {
                                        "difference_from_prefix": comp_point,
                                        "difference_ci95_low": comp_low,
                                        "difference_ci95_high": comp_high,
                                    }
                                )
                            rows.append(row)
    return pd.DataFrame(rows)


def quarter_energy_rows(results: pd.DataFrame, iterations: int) -> pd.DataFrame:
    rows = []
    for partition in ("development", "validation"):
        for alpha_method in ALPHA_METHODS:
            for layout, seed_values in (
                ("prefix", ["shared_prefix"]),
                ("distributed", [str(value) for value in range(len(LAYOUT_KEYS))]),
            ):
                for layout_seed in seed_values:
                    subset = results[
                        (results["partition"] == partition)
                        & (results["alpha_method"] == alpha_method)
                        & (results["layout"] == layout)
                        & (results["layout_seed"].astype(str) == layout_seed)
                    ]
                    for quarter in QUARTERS:
                        energy_column = f"sum_squared_shift_ms2_q{quarter}"
                        fraction_column = f"energy_fraction_q{quarter}"
                        rows.append(
                            {
                                "experiment_version": EXPERIMENT_VERSION,
                                "partition": partition,
                                "alpha_method": alpha_method,
                                "layout": layout,
                                "layout_seed": layout_seed,
                                "quarter": quarter,
                                "active_carriers": int(
                                    subset[f"active_carriers_q{quarter}"].sum()
                                ),
                                "sum_squared_shift_ms2": float(subset[energy_column].sum()),
                                "pooled_energy_fraction": float(
                                    subset[energy_column].sum()
                                    / subset["sum_squared_shift_ms2"].sum()
                                ),
                                "mean_replay_energy_fraction": float(
                                    subset[fraction_column].mean()
                                ),
                                "median_replay_energy_fraction": float(
                                    subset[fraction_column].median()
                                ),
                                "replays": subset["replay_file"].nunique(),
                            }
                        )
    return pd.DataFrame(rows)


def seed_stability_rows(
    replay_paired: pd.DataFrame,
    general_paired: pd.DataFrame,
    physical_paired: pd.DataFrame,
    quarter_energy: pd.DataFrame,
    quarter_features: pd.DataFrame,
) -> pd.DataFrame:
    raw_rows = []
    for alpha_method in ALPHA_METHODS:
        for scope, frame in (
            ("replay_auc", replay_paired),
            ("dev_to_validation_auc", general_paired),
        ):
            subset = frame[
                (frame["alpha_method"] == alpha_method)
                & (frame["feature_set"] == "full")
            ]
            for _, row in subset.iterrows():
                raw_rows.extend(
                    [
                        {
                            "alpha_method": alpha_method,
                            "metric": scope,
                            "layout_seed": row.layout_seed,
                            "value": row.distributed_auc,
                        },
                        {
                            "alpha_method": alpha_method,
                            "metric": f"delta_{scope}_distributed_minus_prefix",
                            "layout_seed": row.layout_seed,
                            "value": row.delta_distributed_minus_prefix,
                        },
                    ]
                )
        physical = physical_paired[
            (physical_paired["partition"] == "validation")
            & (physical_paired["alpha_method"] == alpha_method)
        ]
        for _, row in physical.iterrows():
            raw_rows.append(
                {
                    "alpha_method": alpha_method,
                    "metric": f"distributed_{row.metric}",
                    "layout_seed": row.layout_seed,
                    "value": row.distributed_value,
                }
            )
            raw_rows.append(
                {
                    "alpha_method": alpha_method,
                    "metric": f"delta_{row.metric}_distributed_minus_prefix",
                    "layout_seed": row.layout_seed,
                    "value": row.delta_distributed_minus_prefix,
                }
            )
        energy = quarter_energy[
            (quarter_energy["partition"] == "validation")
            & (quarter_energy["alpha_method"] == alpha_method)
            & (quarter_energy["layout"] == "distributed")
            & (quarter_energy["quarter"] == 1)
        ]
        for _, row in energy.iterrows():
            raw_rows.append(
                {
                    "alpha_method": alpha_method,
                    "metric": "distributed_q1_pooled_energy_fraction",
                    "layout_seed": row.layout_seed,
                    "value": row.pooled_energy_fraction,
                }
            )
        feature = quarter_features[
            (quarter_features["partition"] == "validation")
            & (quarter_features["alpha_method"] == alpha_method)
            & (quarter_features["layout"] == "distributed")
            & (quarter_features["statistic"] == "mean")
            & quarter_features["feature"].isin(
                (
                    "quarter_1_variance",
                    "quarter_1_autocorrelation_lag1",
                    "quarter_1_mean_absolute_residual",
                )
            )
        ]
        for _, row in feature.iterrows():
            raw_rows.append(
                {
                    "alpha_method": alpha_method,
                    "metric": f"distributed_minus_prefix_{row.feature}_mean_delta",
                    "layout_seed": row.layout_seed,
                    "value": row.difference_from_prefix,
                }
            )
    raw = pd.DataFrame(raw_rows)
    summaries = []
    for (alpha_method, metric), group in raw.groupby(["alpha_method", "metric"]):
        summaries.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "alpha_method": alpha_method,
                "metric": metric,
                "seed_count": len(group),
                "mean": float(group["value"].mean()),
                "std": float(group["value"].std(ddof=1)),
                "min": float(group["value"].min()),
                "max": float(group["value"].max()),
            }
        )
    return pd.DataFrame(summaries)


def diagnostics_rows(
    results: pd.DataFrame,
    features: pd.DataFrame,
    replay_paired: pd.DataFrame,
    general_paired: pd.DataFrame,
    new_ok: int,
) -> pd.DataFrame:
    development_maps = set(
        results.loc[results["partition"] == "development", "beatmap_hash"]
    )
    validation_maps = set(
        results.loc[results["partition"] == "validation", "beatmap_hash"]
    )
    prefix = results[results["layout"] == "prefix"]
    return pd.DataFrame(
        [
            {"diagnostic": "new_ok_this_run", "value": new_ok},
            {"diagnostic": "physical_configs", "value": len(results)},
            {"diagnostic": "feature_rows", "value": len(features)},
            {
                "diagnostic": "development_replays",
                "value": results.loc[
                    results["partition"] == "development", "replay_file"
                ].nunique(),
            },
            {
                "diagnostic": "validation_replays",
                "value": results.loc[
                    results["partition"] == "validation", "replay_file"
                ].nunique(),
            },
            {
                "diagnostic": "development_validation_map_overlap",
                "value": len(development_maps & validation_maps),
            },
            {
                "diagnostic": "prefix_non_q1_energy_ms2",
                "value": float(
                    sum(prefix[f"sum_squared_shift_ms2_q{q}"].sum() for q in (2, 3, 4))
                ),
            },
            {
                "diagnostic": "all_features_finite",
                "value": int(
                    np.all(np.isfinite(features[list(FULL_FEATURES)].to_numpy(float)))
                ),
            },
            {
                "diagnostic": "replay_full_seeds_ci_below_zero_adaptive",
                "value": int(
                    np.sum(
                        (replay_paired["alpha_method"] == "sender_local_adaptive")
                        & (replay_paired["feature_set"] == "full")
                        & (replay_paired["delta_ci95_high"] < 0.0)
                    )
                ),
            },
            {
                "diagnostic": "generalization_full_seeds_ci_below_zero_adaptive",
                "value": int(
                    np.sum(
                        (general_paired["alpha_method"] == "sender_local_adaptive")
                        & (general_paired["feature_set"] == "full")
                        & (general_paired["delta_ci95_high"] < 0.0)
                    )
                ),
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json")
    parser.add_argument("--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv")
    parser.add_argument("--performance-groups", type=Path, default=METADATA_DIR / "performance_groups.csv")
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--limit-per-partition", type=int, default=None)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.generate_only and args.evaluate_only:
        raise ValueError("Ne mogu zajedno --generate-only i --evaluate-only.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = experiment_config(
        policy_path=args.policy,
        results_path=args.results,
        performance_path=args.performance_groups,
        partition_path=args.partition,
        map_offsets_path=args.map_offsets,
        limit_per_partition=args.limit_per_partition,
        trees=args.trees,
        folds=args.folds,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    write_or_validate_config(args.output_dir / "config.json", config)
    result_path = args.output_dir / "physical_results.csv"
    feature_path = args.output_dir / "features.csv"
    new_ok = 0
    if not args.evaluate_only:
        for partition_name in ("development", "validation"):
            cohort = load_partitioned_cohort(
                args.results,
                args.performance_groups,
                args.partition,
                partition_name,
                args.limit_per_partition,
            )
            new_ok += generate_partition(
                partition_name=partition_name,
                cohort=cohort,
                config=config,
                policy_path=args.policy,
                dataset_dir=args.dataset_dir,
                map_offsets_path=args.map_offsets,
                result_path=result_path,
                feature_path=feature_path,
            )
        print(f"physical generation new OK={new_ok}", flush=True)
    if args.generate_only:
        return
    results = pd.read_csv(result_path)
    features = pd.read_csv(feature_path)
    expected_replays = 949 if args.limit_per_partition is None else 2 * args.limit_per_partition
    validate_physical_outputs(results, features, expected_replays)
    all_auc, replay_paired, general_paired, _ = evaluate_auc(
        features,
        args.output_dir,
        args.folds,
        args.trees,
        args.bootstrap_iterations,
    )
    physical = physical_paired_rows(results, args.bootstrap_iterations)
    physical.to_csv(args.output_dir / "ber_paired.csv", index=False)
    quarter_energy = quarter_energy_rows(results, args.bootstrap_iterations)
    quarter_energy.to_csv(args.output_dir / "quarter_energy.csv", index=False)
    quarter_features = quarter_feature_rows(features, args.bootstrap_iterations)
    quarter_features.to_csv(
        args.output_dir / "quarter_feature_deltas.csv", index=False
    )
    stability = seed_stability_rows(
        replay_paired,
        general_paired,
        physical,
        quarter_energy,
        quarter_features,
    )
    stability.to_csv(args.output_dir / "seed_stability.csv", index=False)
    diagnostics_rows(
        results, features, replay_paired, general_paired, new_ok
    ).to_csv(args.output_dir / "diagnostics.csv", index=False)
    print("adaptive layout strong evaluation complete", flush=True)


if __name__ == "__main__":
    main()
