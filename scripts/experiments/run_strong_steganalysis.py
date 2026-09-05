"""Controlled feature-ablation study for frozen sender-local adaptive alpha v1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from osu_stego.analysis.timing_features import (
    BASELINE_FEATURES,
    FEATURE_SETS,
    FEATURE_VERSION,
    FULL_FEATURES,
    GLOBAL_FEATURES,
    POSITION_FEATURES,
    TEMPORAL_FEATURES,
    residual_timing_features,
)
from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import load_sender_local_policy
from scripts.experiments.run_adaptive_alpha_comparison import N_VALUE
from scripts.experiments.run_adaptive_alpha_validation import (
    bootstrap_auc_delta_clustered,
    file_sha256,
)
from scripts.experiments.run_layout_comparison import (
    bootstrap_auc_delta,
    build_physical_stego,
    experiment_config_id,
    key_id,
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
from scripts.experiments.run_sender_local_validation import (
    assignments_from_policy,
    select_method,
)
from scripts.experiments.run_steganalysis import (
    bootstrap_auc_by_group,
    residual_features,
    shuffled_group_ids,
)
from scripts.experiments.sender_local_common import load_partitioned_cohort


EXPERIMENT_VERSION = "strong-steganalysis-v1"
LEGACY_EXPERIMENT_VERSION = "sender-local-heldout-validation-v1"
METHODS = ("fixed_15", "sender_local_adaptive")
PAYLOAD_FRACTION = 0.075
PN_KEY = "adaptive-alpha-pn-v1"
LAYOUT_KEY = "prefix-layout-unused-v1"
LAYOUT = "prefix"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "strong_steganalysis_v1"
DEFAULT_POLICY = CONFIG_DIR / "adaptive_alpha_sender_local_v1.json"
DEFAULT_PARTITION = RESULTS_DIR / "adaptive_alpha_validation_v1" / "heldout_partition.csv"
SOURCE_DIR = RESULTS_DIR / "sender_local_adaptive_v1"

RF_PARAMETERS = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "n_jobs": 1,
}

METADATA_FIELDS = [
    "experiment_version",
    "feature_version",
    "feature_config_id",
    "source_config_id",
    "method_config_hash",
    "partition",
    "method",
    "replay_file",
    "beatmap_hash",
    "performance_category",
    "label",
    "alpha",
    "n_frames_per_bit",
    "payload_fraction",
    "message_bits",
    "num_notes",
    "n_valid_residuals",
    "valid_residual_fraction",
    "replay_sha256",
    "policy_sha256",
    "partition_sha256",
    "map_offsets_sha256",
    "source_grid_features_sha256",
    "source_grid_results_sha256",
]
FEATURE_FIELDS = METADATA_FIELDS + list(FULL_FEATURES)


def feature_schema(
    policy_sha256: str,
    partition_sha256: str,
    map_offsets_sha256: str,
) -> dict:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
        "feature_groups": {
            "baseline": list(BASELINE_FEATURES),
            "global_additions": list(GLOBAL_FEATURES),
            "temporal_additions": list(TEMPORAL_FEATURES),
            "position_additions": list(POSITION_FEATURES),
        },
        "missing_value_rules": {
            "quarter_assignment": "np.array_split on original note-index axis before dropping NaN",
            "lag_pairs": "indices exactly lag apart and both finite",
            "difference_pairs": "adjacent original note indices and both finite",
            "undefined_statistic": 0.0,
            "n_valid_is_model_feature": False,
        },
        "model_inputs_excluded": [
            "message",
            "pn_key",
            "BER",
            "label-derived preprocessing",
            "population statistics",
        ],
        "rf_parameters": RF_PARAMETERS,
        "seed": 42,
        "folds": 5,
        "methods": list(METHODS),
        "method_configuration": {
            "N": N_VALUE,
            "payload_fraction": PAYLOAD_FRACTION,
            "layout": LAYOUT,
            "message_seed": 42,
            "pn_key_id": key_id(PN_KEY),
            "layout_key_id": key_id(LAYOUT_KEY),
        },
        "policy_sha256": policy_sha256,
        "partition_sha256": partition_sha256,
        "map_offsets_sha256": map_offsets_sha256,
    }


def write_or_validate_schema(path: Path, schema: dict) -> None:
    serialized = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError("Postojeći feature_schema.json pripada drugoj konfiguraciji.")
        return
    path.write_text(serialized, encoding="utf-8")


def completed_feature_configs(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    if list(frame.columns) != FEATURE_FIELDS:
        raise ValueError("features.csv ima nekompatibilnu šemu; koristi novi output.")
    labels = frame.groupby("feature_config_id")["label"].apply(
        lambda values: tuple(sorted(values.astype(int).tolist()))
    )
    return set(labels[labels == (0, 1)].index.astype(str))


def replace_feature_rows(path: Path, rows: list[dict]) -> None:
    """Atomically replace complete feature configs in a resumable CSV."""
    if not rows:
        return
    existing: list[dict[str, str]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != FEATURE_FIELDS:
                raise ValueError("features.csv ima nekompatibilnu šemu.")
            existing = list(reader)
    config_ids = {str(row["feature_config_id"]) for row in rows}
    retained = [
        row for row in existing if row.get("feature_config_id") not in config_ids
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FEATURE_FIELDS)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in FEATURE_FIELDS}
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


def _selected_source_data(
    partition_name: str,
    cohort: pd.DataFrame,
    policy: dict,
    policy_sha256: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], Path, Path]:
    result_path = SOURCE_DIR / f"{partition_name}_alpha_grid_results.csv"
    feature_path = SOURCE_DIR / f"{partition_name}_alpha_grid_features.csv"
    grid_results = pd.read_csv(result_path)
    grid_features = pd.read_csv(feature_path)
    sender = assignments_from_policy(cohort, policy)
    assignments = {
        "fixed_15": sender.assign(selected_alpha=15.0),
        "sender_local_adaptive": sender,
    }
    results: dict[str, pd.DataFrame] = {}
    features: dict[str, pd.DataFrame] = {}
    for method in METHODS:
        selected_result, selected_feature = select_method(
            grid_results,
            grid_features,
            assignments[method],
            method,
            policy_sha256,
            None,
        )
        results[method] = selected_result
        features[method] = selected_feature
    return results, features, result_path, feature_path


def _assert_legacy_baseline(
    extracted: dict[str, float | int],
    residuals: np.ndarray,
    expected: pd.Series,
) -> None:
    legacy = residual_features(residuals)
    for name in BASELINE_FEATURES:
        if float(extracted[name]) != float(legacy[name]):
            raise ValueError(f"Baseline feature nije reprodukovan tačno: {name}.")
        if not np.isclose(
            float(extracted[name]), float(expected[name]), rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"Regenerisan {name} se ne slaže sa frozen CSV-om.")


def generate_partition_features(
    *,
    partition_name: str,
    cohort: pd.DataFrame,
    policy: dict,
    policy_sha256: str,
    dataset_dir: Path,
    map_offsets_path: Path,
    partition_path: Path,
    output_path: Path,
    seed: int,
) -> int:
    selected_results, selected_features, result_path, source_feature_path = (
        _selected_source_data(partition_name, cohort, policy, policy_sha256)
    )
    selected_by_method = {
        method: frame.set_index("replay_file", drop=False)
        for method, frame in selected_results.items()
    }
    expected_by_method = {
        method: frame.set_index(["replay_file", "label"], drop=False)
        for method, frame in selected_features.items()
    }
    completed = completed_feature_configs(output_path)
    requested: set[str] = set()
    beatmaps = build_beatmap_index(dataset_dir)
    offsets = load_map_offsets(map_offsets_path)
    provenance = {
        "policy_sha256": policy_sha256,
        "partition_sha256": file_sha256(partition_path),
        "map_offsets_sha256": file_sha256(map_offsets_path),
        "source_grid_features_sha256": file_sha256(source_feature_path),
        "source_grid_results_sha256": file_sha256(result_path),
    }
    pending_rows: list[dict] = []
    new_pairs = 0
    with tempfile.TemporaryDirectory(prefix=f"osu_strong_{partition_name}_") as name:
        temp_dir = Path(name)
        for replay_index, row in enumerate(cohort.itertuples(index=False), 1):
            context = prepare_replay(row, dataset_dir, beatmaps, offsets)
            replay_sha256 = file_sha256(context.osr_path)
            capacity = nominal_capacity_bits(len(context.original_residuals), N_VALUE)
            message_bits = message_length_for_fraction(capacity, PAYLOAD_FRACTION)
            message = deterministic_message(
                replay_file=context.replay_file,
                beatmap_hash=context.beatmap_hash,
                alpha=0.0,
                n_value=N_VALUE,
                payload_fraction=PAYLOAD_FRACTION,
                num_bits=message_bits,
                seed=seed,
            )
            for method in METHODS:
                source = selected_by_method[method].loc[context.replay_file]
                alpha = float(source["alpha"])
                method_config_hash = experiment_config_id(
                    method=method,
                    alpha=alpha,
                    N=N_VALUE,
                    payload_fraction=PAYLOAD_FRACTION,
                    layout=LAYOUT,
                    policy_sha256=policy_sha256,
                )
                feature_config_id = experiment_config_id(
                    experiment_version=EXPERIMENT_VERSION,
                    feature_version=FEATURE_VERSION,
                    source_config_id=str(source["config_id"]),
                    method_config_hash=method_config_hash,
                    replay_sha256=replay_sha256,
                    **provenance,
                )
                requested.add(feature_config_id)
                if feature_config_id in completed:
                    continue
                _, roundtrip, diagnostics = build_physical_stego(
                    context=context,
                    message=message,
                    alpha=alpha,
                    n_value=N_VALUE,
                    layout=LAYOUT,
                    pn_key=PN_KEY,
                    layout_key=LAYOUT_KEY,
                    hit_margin_ms=float(policy["hit_margin_ms"]),
                    temp_path=temp_dir / f"{context.osr_path.stem}_{method}.osr",
                )
                if not np.isclose(
                    float(diagnostics["sum_squared_shift_ms2"]),
                    float(source["sum_squared_shift_ms2"]),
                ):
                    raise ValueError("Fizička energija se ne slaže sa frozen grid-om.")
                clean = residual_timing_features(context.original_residuals)
                stego = residual_timing_features(roundtrip)
                _assert_legacy_baseline(
                    clean,
                    context.original_residuals,
                    expected_by_method[method].loc[(context.replay_file, 0)],
                )
                _assert_legacy_baseline(
                    stego,
                    roundtrip,
                    expected_by_method[method].loc[(context.replay_file, 1)],
                )
                common = {
                    "experiment_version": EXPERIMENT_VERSION,
                    "feature_version": FEATURE_VERSION,
                    "feature_config_id": feature_config_id,
                    "source_config_id": str(source["config_id"]),
                    "method_config_hash": method_config_hash,
                    "partition": partition_name,
                    "method": method,
                    "replay_file": context.replay_file,
                    "beatmap_hash": context.beatmap_hash,
                    "performance_category": str(row.performance_category),
                    "alpha": alpha,
                    "n_frames_per_bit": N_VALUE,
                    "payload_fraction": PAYLOAD_FRACTION,
                    "message_bits": message_bits,
                    "replay_sha256": replay_sha256,
                    **provenance,
                }
                pending_rows.extend(
                    [
                        {**common, **clean, "label": 0},
                        {**common, **stego, "label": 1},
                    ]
                )
                new_pairs += 1
            if len(pending_rows) >= 100 or replay_index == len(cohort):
                replace_feature_rows(output_path, pending_rows)
                pending_rows = []
            if replay_index % 25 == 0 or replay_index == len(cohort):
                print(
                    f"features {partition_name}: {replay_index}/{len(cohort)} "
                    f"replay-eva | new pairs={new_pairs}"
                )
    frame = pd.read_csv(output_path)
    frame = frame[frame["feature_config_id"].astype(str).isin(requested)].copy()
    validate_feature_frame(frame, expected_replays=len(cohort))
    return new_pairs


def validate_feature_frame(frame: pd.DataFrame, expected_replays: int | None = None) -> None:
    if list(frame.columns) != FEATURE_FIELDS:
        raise ValueError("Strong feature tabela nema zamrznutu šemu.")
    if set(frame["method"].astype(str)) != set(METHODS):
        raise ValueError("Nedostaje primarni metod u strong feature tabeli.")
    labels = frame.groupby("feature_config_id")["label"].apply(
        lambda values: tuple(sorted(values.astype(int).tolist()))
    )
    if not all(value == (0, 1) for value in labels):
        raise ValueError("Svaki strong feature config mora imati clean/stego par.")
    replay_sets = {
        method: set(group["replay_file"].astype(str))
        for method, group in frame.groupby("method")
    }
    if replay_sets[METHODS[0]] != replay_sets[METHODS[1]]:
        raise ValueError("Metodi ne koriste isti replay subset.")
    if expected_replays is not None and len(replay_sets[METHODS[0]]) != expected_replays:
        raise ValueError("Strong feature tabela nema očekivani broj replay-eva.")
    if np.any(~np.isfinite(frame[list(FULL_FEATURES)].to_numpy(dtype=float))):
        raise ValueError("Strong feature tabela sadrži NaN/inf.")
    clean = frame[frame["label"] == 0]
    for _, group in clean.groupby("replay_file"):
        if np.any(group[list(FULL_FEATURES)].nunique(dropna=False).to_numpy() != 1):
            raise ValueError("Clean feature-i nisu identični između metoda.")


def _rf(seed: int, trees: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trees,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=1,
    )


def source_ordered_subset(
    frame: pd.DataFrame,
    partition_name: str,
    method: str,
) -> pd.DataFrame:
    """Restore the exact frozen grid row order used by the baseline evaluator."""
    subset = frame[frame["method"] == method].copy()
    source = pd.read_csv(SOURCE_DIR / f"{partition_name}_alpha_grid_features.csv")
    source_order = {
        (str(row.config_id), int(row.label)): index
        for index, row in enumerate(source.itertuples(index=False))
    }
    keys = list(
        zip(
            subset["source_config_id"].astype(str),
            subset["label"].astype(int),
        )
    )
    try:
        subset["_source_order"] = [source_order[key] for key in keys]
    except KeyError as exc:
        raise ValueError("Strong feature red nema frozen source-grid par.") from exc
    return subset.sort_values("_source_order").drop(columns="_source_order").reset_index(
        drop=True
    )


def replay_grouped_evaluation(
    validation: pd.DataFrame,
    feature_names: tuple[str, ...],
    method: str,
    feature_set: str,
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    subset = source_ordered_subset(validation, "validation", method)
    X = subset[list(feature_names)].to_numpy(dtype=float)
    y = subset["label"].to_numpy(dtype=np.int8)
    replay = subset["replay_file"].astype(str).to_numpy()
    groups = shuffled_group_ids(
        replay,
        stable_seed(seed, "ALL", None, N_VALUE, "folds"),
    )
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    scores = np.full(len(subset), np.nan, dtype=float)
    importances = []
    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups), 1
    ):
        classifier = _rf(
            stable_seed(seed, "ALL", None, N_VALUE, fold_index, "rf"), trees
        )
        classifier.fit(X[train_index], y[train_index])
        scores[test_index] = classifier.predict_proba(X[test_index])[:, 1]
        importances.append(classifier.feature_importances_.astype(float))
    auc = float(roc_auc_score(y, scores))
    low, high = bootstrap_auc_by_group(
        y,
        scores,
        replay,
        stable_seed(seed, "ALL", None, N_VALUE, "bootstrap"),
        bootstrap_iterations,
    )
    prediction = subset[
        ["replay_file", "beatmap_hash", "performance_category", "label"]
    ].copy()
    prediction["score"] = scores
    prediction["method"] = method
    prediction["feature_set"] = feature_set
    return (
        {
            "experiment_version": EXPERIMENT_VERSION,
            "feature_set": feature_set,
            "method": method,
            "roc_auc": auc,
            "auc_ci95_low": low,
            "auc_ci95_high": high,
            "bootstrap_unit": "replay_file",
            "replay_pairs": subset["replay_file"].nunique(),
            "folds": splitter.n_splits,
            "trees": trees,
            "rf_n_jobs": 1,
        },
        prediction,
        np.mean(np.vstack(importances), axis=0),
    )


def generalization_evaluation(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: tuple[str, ...],
    method: str,
    feature_set: str,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    train = source_ordered_subset(development, "development", method)
    test = source_ordered_subset(validation, "validation", method)
    if set(train["beatmap_hash"]) & set(test["beatmap_hash"]):
        raise ValueError("Beatmap leakage u development-to-validation evaluaciji.")
    classifier = _rf(
        stable_seed(seed, LEGACY_EXPERIMENT_VERSION, "generalization-rf"), trees
    )
    classifier.fit(
        train[list(feature_names)].to_numpy(dtype=float),
        train["label"].to_numpy(dtype=np.int8),
    )
    scores = classifier.predict_proba(
        test[list(feature_names)].to_numpy(dtype=float)
    )[:, 1]
    labels = test["label"].to_numpy(dtype=np.int8)
    beatmaps = test["beatmap_hash"].astype(str).to_numpy()
    auc = float(roc_auc_score(labels, scores))
    low, high = bootstrap_auc_by_group(
        labels,
        scores,
        beatmaps,
        stable_seed(
            seed,
            LEGACY_EXPERIMENT_VERSION,
            method,
            "generalization-bootstrap",
        ),
        bootstrap_iterations,
    )
    prediction = test[
        ["replay_file", "beatmap_hash", "performance_category", "label"]
    ].copy()
    prediction["score"] = scores
    prediction["method"] = method
    prediction["feature_set"] = feature_set
    return (
        {
            "experiment_version": EXPERIMENT_VERSION,
            "feature_set": feature_set,
            "method": method,
            "roc_auc": auc,
            "auc_ci95_low": low,
            "auc_ci95_high": high,
            "bootstrap_unit": "beatmap_hash",
            "development_maps": train["beatmap_hash"].nunique(),
            "validation_maps": test["beatmap_hash"].nunique(),
            "validation_replays": test["replay_file"].nunique(),
            "trees": trees,
            "rf_n_jobs": 1,
        },
        prediction,
        classifier.feature_importances_.astype(float),
    )


def paired_prediction_row(
    control: pd.DataFrame,
    adaptive: pd.DataFrame,
    feature_set: str,
    scope: str,
    seed: int,
    iterations: int,
) -> dict:
    control_auc = float(roc_auc_score(control["label"], control["score"]))
    adaptive_auc = float(roc_auc_score(adaptive["label"], adaptive["score"]))
    if scope == "replay_grouped":
        renamed_control = control.rename(columns={"score": "oof_probability_stego"})
        renamed_adaptive = adaptive.rename(columns={"score": "oof_probability_stego"})
        low, high = bootstrap_auc_delta(
            renamed_control,
            renamed_adaptive,
            stable_seed(
                seed,
                LEGACY_EXPERIMENT_VERSION,
                "sender_local_adaptive_minus_fixed_15",
                "replay_grouped_auc",
            ),
            iterations,
        )
        unit = "replay_file"
    else:
        low, high = bootstrap_auc_delta_clustered(
            control,
            adaptive,
            "beatmap_hash",
            stable_seed(
                seed,
                LEGACY_EXPERIMENT_VERSION,
                "sender_local_adaptive_minus_fixed_15",
                "dev_to_validation_auc",
            ),
            iterations,
        )
        unit = "beatmap_hash"
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "feature_set": feature_set,
        "comparison": "sender_local_adaptive_minus_fixed_15",
        "control_auc": control_auc,
        "adaptive_auc": adaptive_auc,
        "delta_auc_adaptive_minus_fixed15": adaptive_auc - control_auc,
        "delta_ci95_low": low,
        "delta_ci95_high": high,
        "bootstrap_unit": unit,
        "bootstrap_iterations": iterations,
    }


def detector_strength_rows(
    predictions: dict[tuple[str, str], pd.DataFrame],
    scope: str,
    seed: int,
    iterations: int,
) -> list[dict]:
    rows = []
    for method in METHODS:
        baseline = predictions[(method, "baseline")]
        full = predictions[(method, "full")]
        if scope == "replay_grouped":
            low, high = bootstrap_auc_delta(
                baseline.rename(columns={"score": "oof_probability_stego"}),
                full.rename(columns={"score": "oof_probability_stego"}),
                stable_seed(seed, EXPERIMENT_VERSION, method, scope, "strength"),
                iterations,
            )
            unit = "replay_file"
        else:
            low, high = bootstrap_auc_delta_clustered(
                baseline,
                full,
                "beatmap_hash",
                stable_seed(seed, EXPERIMENT_VERSION, method, scope, "strength"),
                iterations,
            )
            unit = "beatmap_hash"
        baseline_auc = float(roc_auc_score(baseline["label"], baseline["score"]))
        full_auc = float(roc_auc_score(full["label"], full["score"]))
        rows.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "scope": scope,
                "method": method,
                "baseline_auc": baseline_auc,
                "full_auc": full_auc,
                "delta_full_minus_baseline": full_auc - baseline_auc,
                "delta_ci95_low": low,
                "delta_ci95_high": high,
                "bootstrap_unit": unit,
                "bootstrap_iterations": iterations,
            }
        )
    return rows


def robustness_outputs(
    development: pd.DataFrame,
    replay_predictions: dict[tuple[str, str], pd.DataFrame],
    generalization_predictions: dict[tuple[str, str], pd.DataFrame],
    output_dir: Path,
) -> None:
    correlations = development[list(FULL_FEATURES)].corr(method="spearman")
    correlation_rows = []
    for left_index, left in enumerate(FULL_FEATURES):
        for right in FULL_FEATURES[left_index + 1 :]:
            correlation_rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "spearman": float(correlations.loc[left, right]),
                    "absolute_spearman": abs(float(correlations.loc[left, right])),
                }
            )
    pd.DataFrame(correlation_rows).sort_values(
        "absolute_spearman", ascending=False
    ).to_csv(output_dir / "feature_correlations.csv", index=False)

    diagnostic_rows = []
    extreme_rows = []
    quality_rows = []
    per_map_rows = []
    for method in METHODS:
        replay = replay_predictions[(method, "full")]
        full_auc = float(roc_auc_score(replay["label"], replay["score"]))
        influences = []
        for replay_file in replay["replay_file"].unique():
            kept = replay[replay["replay_file"] != replay_file]
            auc_without = float(roc_auc_score(kept["label"], kept["score"]))
            influences.append((str(replay_file), auc_without - full_auc))
        influences.sort(key=lambda item: abs(item[1]), reverse=True)
        for replay_file, delta in influences[:10]:
            extreme_rows.append(
                {
                    "method": method,
                    "replay_file": replay_file,
                    "full_auc": full_auc,
                    "auc_without_replay": full_auc + delta,
                    "delta_when_removed": delta,
                }
            )
        diagnostic_rows.append(
            {
                "diagnostic": "maximum_absolute_leave_one_replay_out_auc_change",
                "method": method,
                "value": max(abs(item[1]) for item in influences),
                "detail": influences[0][0],
            }
        )
        for quality, group in replay.groupby("performance_category"):
            quality_rows.append(
                {
                    "method": method,
                    "performance_category": quality,
                    "replay_pairs": group["replay_file"].nunique(),
                    "roc_auc": float(roc_auc_score(group["label"], group["score"])),
                    "analysis_role": "descriptive_only",
                }
            )
        generalization = generalization_predictions[(method, "full")]
        for beatmap_hash, group in generalization.groupby("beatmap_hash"):
            per_map_rows.append(
                {
                    "method": method,
                    "feature_set": "full",
                    "beatmap_hash": beatmap_hash,
                    "replay_pairs": group["replay_file"].nunique(),
                    "roc_auc": float(roc_auc_score(group["label"], group["score"])),
                }
            )
    development_maps = set(development["beatmap_hash"].astype(str))
    validation_reference = replay_predictions[("fixed_15", "full")]
    validation_maps = set(validation_reference["beatmap_hash"].astype(str))
    frozen = pd.read_csv(SOURCE_DIR / "validation_auc.csv").set_index("method")
    baseline_differences = []
    for method in METHODS:
        baseline_differences.extend(
            [
                abs(
                    roc_auc_score(
                        replay_predictions[(method, "baseline")]["label"],
                        replay_predictions[(method, "baseline")]["score"],
                    )
                    - float(frozen.loc[method, "replay_grouped_auc"])
                ),
                abs(
                    roc_auc_score(
                        generalization_predictions[(method, "baseline")]["label"],
                        generalization_predictions[(method, "baseline")]["score"],
                    )
                    - float(frozen.loc[method, "dev_to_validation_auc"])
                ),
            ]
        )
    map_auc = pd.DataFrame(per_map_rows).pivot(
        index="beatmap_hash", columns="method", values="roc_auc"
    )
    high_correlation_count = sum(
        row["absolute_spearman"] >= 0.95 for row in correlation_rows
    )
    diagnostic_rows.extend(
        [
            {
                "diagnostic": "development_feature_pairs_abs_spearman_ge_0_95",
                "method": "ALL",
                "value": high_correlation_count,
                "detail": "No feature selection performed",
            },
            {
                "diagnostic": "target_dependent_preprocessing",
                "method": "ALL",
                "value": 0,
                "detail": "Raw finite feature columns passed directly to RF",
            },
            {
                "diagnostic": "development_replays",
                "method": "ALL",
                "value": development["replay_file"].nunique(),
                "detail": f"maps={len(development_maps)}",
            },
            {
                "diagnostic": "validation_replays",
                "method": "ALL",
                "value": validation_reference["replay_file"].nunique(),
                "detail": f"maps={len(validation_maps)}",
            },
            {
                "diagnostic": "development_validation_map_overlap",
                "method": "ALL",
                "value": len(development_maps & validation_maps),
                "detail": "Frozen beatmap partition",
            },
            {
                "diagnostic": "maximum_absolute_baseline_auc_reproduction_error",
                "method": "ALL",
                "value": max(baseline_differences),
                "detail": "Four frozen replay/generalization AUC values",
            },
            {
                "diagnostic": "all_model_features_finite",
                "method": "ALL",
                "value": int(
                    np.all(
                        np.isfinite(
                            development[list(FULL_FEATURES)].to_numpy(dtype=float)
                        )
                    )
                ),
                "detail": "Development checked; validation enforced by validator",
            },
            {
                "diagnostic": "validation_maps_with_lower_adaptive_full_auc",
                "method": "sender_local_adaptive",
                "value": int(
                    np.sum(
                        map_auc["sender_local_adaptive"] < map_auc["fixed_15"]
                    )
                ),
                "detail": f"of {len(map_auc)} maps; descriptive only",
            },
            {
                "diagnostic": "layout_extended_recheck_supported_by_existing_csv",
                "method": "ALL",
                "value": 0,
                "detail": "Existing layout CSV stores baseline aggregates, not note-indexed residuals",
            },
        ]
    )
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / "diagnostics.csv", index=False)
    pd.DataFrame(extreme_rows).to_csv(
        output_dir / "extreme_replay_influence.csv", index=False
    )
    pd.DataFrame(quality_rows).to_csv(
        output_dir / "quality_group_auc.csv", index=False
    )
    pd.DataFrame(per_map_rows).to_csv(output_dir / "per_beatmap_auc.csv", index=False)


def evaluate(
    features: pd.DataFrame,
    output_dir: Path,
    folds: int,
    trees: int,
    seed: int,
    bootstrap_iterations: int,
) -> None:
    development = features[features["partition"] == "development"].copy()
    validation = features[features["partition"] == "validation"].copy()
    validate_feature_frame(development)
    validate_feature_frame(validation)
    if set(development["beatmap_hash"]) & set(validation["beatmap_hash"]):
        raise ValueError("Development/validation beatmap overlap.")

    replay_rows = []
    generalization_rows = []
    replay_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    generalization_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    importance_rows = []
    for feature_set, feature_names in FEATURE_SETS.items():
        for method in METHODS:
            replay_row, replay_prediction, replay_importance = replay_grouped_evaluation(
                validation,
                feature_names,
                method,
                feature_set,
                folds,
                trees,
                seed,
                bootstrap_iterations,
            )
            generalization_row, generalization_prediction, gen_importance = (
                generalization_evaluation(
                    development,
                    validation,
                    feature_names,
                    method,
                    feature_set,
                    trees,
                    seed,
                    bootstrap_iterations,
                )
            )
            replay_rows.append(replay_row)
            generalization_rows.append(generalization_row)
            replay_predictions[(method, feature_set)] = replay_prediction
            generalization_predictions[(method, feature_set)] = generalization_prediction
            if feature_set == "full":
                for scope, values in (
                    ("validation_replay_grouped_cv", replay_importance),
                    ("development_to_validation", gen_importance),
                ):
                    for name, value in zip(feature_names, values):
                        importance_rows.append(
                            {
                                "experiment_version": EXPERIMENT_VERSION,
                                "method": method,
                                "scope": scope,
                                "feature": name,
                                "importance": float(value),
                                "interpretation": "exploratory_noncausal",
                            }
                        )

    replay_frame = pd.DataFrame(replay_rows)
    generalization_frame = pd.DataFrame(generalization_rows)
    replay_frame.to_csv(output_dir / "ablation_auc.csv", index=False)
    generalization_frame.to_csv(output_dir / "dev_to_validation_auc.csv", index=False)

    replay_paired = []
    generalization_paired = []
    for feature_set in FEATURE_SETS:
        replay_paired.append(
            paired_prediction_row(
                replay_predictions[("fixed_15", feature_set)],
                replay_predictions[("sender_local_adaptive", feature_set)],
                feature_set,
                "replay_grouped",
                seed,
                bootstrap_iterations,
            )
        )
        generalization_paired.append(
            paired_prediction_row(
                generalization_predictions[("fixed_15", feature_set)],
                generalization_predictions[("sender_local_adaptive", feature_set)],
                feature_set,
                "development_to_validation",
                seed,
                bootstrap_iterations,
            )
        )
    pd.DataFrame(replay_paired).to_csv(
        output_dir / "ablation_auc_paired.csv", index=False
    )
    pd.DataFrame(generalization_paired).to_csv(
        output_dir / "dev_to_validation_paired.csv", index=False
    )
    pd.DataFrame(importance_rows).sort_values(
        ["method", "scope", "importance"], ascending=[True, True, False]
    ).to_csv(output_dir / "full_feature_importance.csv", index=False)
    strength = detector_strength_rows(
        replay_predictions, "replay_grouped", seed, bootstrap_iterations
    ) + detector_strength_rows(
        generalization_predictions,
        "development_to_validation",
        seed,
        bootstrap_iterations,
    )
    pd.DataFrame(strength).to_csv(output_dir / "detector_strength.csv", index=False)
    robustness_outputs(
        development,
        replay_predictions,
        generalization_predictions,
        output_dir,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.generate_only and args.evaluate_only:
        raise ValueError("Ne mogu zajedno --generate-only i --evaluate-only.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_sender_local_policy(args.policy)
    policy_sha256 = file_sha256(args.policy)
    schema = feature_schema(
        policy_sha256,
        file_sha256(args.partition),
        file_sha256(args.map_offsets),
    )
    schema["requested_limit_per_partition"] = args.limit_per_partition
    schema["rf_parameters"]["n_estimators"] = args.trees
    schema["folds"] = args.folds
    schema["bootstrap_iterations"] = args.bootstrap_iterations
    write_or_validate_schema(args.output_dir / "feature_schema.json", schema)
    feature_path = args.output_dir / "features.csv"

    if not args.evaluate_only:
        total_new = 0
        for partition_name in ("development", "validation"):
            cohort = load_partitioned_cohort(
                args.results,
                args.performance_groups,
                args.partition,
                partition_name,
                args.limit_per_partition,
            )
            total_new += generate_partition_features(
                partition_name=partition_name,
                cohort=cohort,
                policy=policy,
                policy_sha256=policy_sha256,
                dataset_dir=args.dataset_dir,
                map_offsets_path=args.map_offsets,
                partition_path=args.partition,
                output_path=feature_path,
                seed=args.seed,
            )
        print(f"feature generation new pairs={total_new}")

    if not args.generate_only:
        features = pd.read_csv(feature_path)
        evaluate(
            features,
            args.output_dir,
            args.folds,
            args.trees,
            args.seed,
            args.bootstrap_iterations,
        )
        print("strong steganalysis evaluation complete")


if __name__ == "__main__":
    main()
