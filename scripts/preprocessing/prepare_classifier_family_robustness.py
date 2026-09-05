"""Audit and freeze feature rows, folds, and classifier-family configuration."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from osu_stego.analysis.classifier_families import build_classifier, positive_class_scores
from osu_stego.analysis.timing_features import BASELINE_FEATURES, FULL_FEATURES, POSITION_FEATURES
from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from scripts.experiments.adaptive_layout_strong_common import MESSAGE_SEED, N_VALUE
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_adaptive_layout_strong import frozen_ordered_subset
from scripts.experiments.run_pilot_ber_sweep import stable_seed
from scripts.experiments.run_steganalysis import shuffled_group_ids


VERSION = "classifier-family-robustness-v1"
OUTPUT = RESULTS_DIR / "classifier_family_robustness_v1"
CONFIG = CONFIG_DIR / "steganalysis_classifier_robustness_v1.json"
DEV_SOURCE = RESULTS_DIR / "pn_key_robustness_v1" / "timing_features.csv"
NEW_MAP_SOURCE = RESULTS_DIR / "one_codeword_floor_validation_v1" / "timing_features.csv"
HISTORICAL_SOURCE = RESULTS_DIR / "adaptive_layout_strong_v1" / "features.csv"
HISTORICAL_AUC = RESULTS_DIR / "adaptive_layout_strong_v1" / "auc_by_seed.csv"
FEATURE_SCHEMA = RESULTS_DIR / "strong_steganalysis_v1" / "feature_schema.json"


def row_id(source: str, config_id: str, label: int) -> str:
    return hashlib.sha256(f"{VERSION}|{source}|{config_id}|{label}".encode()).hexdigest()


def normalize_source(frame: pd.DataFrame, partition: str, source_name: str) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    result = pd.DataFrame({
        "row_id": [row_id(source_name, str(row.config_id), int(row.label)) for row in frame.itertuples()],
        "pair_id": frame["config_id"].astype(str),
        "source_artifact": source_name,
        "source_config_id": frame["config_id"].astype(str),
        "partition": partition,
        "physical_method": "adaptive_distributed_normal_7_5_secded_historical_pn",
        "replay_file": frame["replay_file"].astype(str),
        "beatmap_hash": frame["beatmap_hash"].astype(str),
        "performance_category": frame.get("performance_category", pd.Series("UNAVAILABLE", index=frame.index)).fillna("UNAVAILABLE"),
        "layout_seed": frame["layout_seed"].astype(int),
        "layout_key_id": frame["layout_key_id"].astype(str),
        "label": frame["label"].astype(int),
        "feature_version": frame["feature_version"].astype(str),
    })
    for feature in FULL_FEATURES:
        result[feature] = frame[feature].to_numpy(dtype=float)
    return result


def build_dataset() -> pd.DataFrame:
    development = pd.read_csv(DEV_SOURCE)
    development = development[development["historical_pn"].astype(int) == 1].copy()
    unseen = pd.read_csv(NEW_MAP_SOURCE)
    unseen = unseen[unseen["capacity_class"] == "NORMAL_7_5"].copy()
    dataset = pd.concat([
        normalize_source(development, "development", str(DEV_SOURCE)),
        normalize_source(unseen, "unseen_new_maps", str(NEW_MAP_SOURCE)),
    ], ignore_index=True)
    return dataset.sort_values(
        ["partition", "replay_file", "layout_seed", "label"]
    ).reset_index(drop=True)


def validate_dataset(dataset: pd.DataFrame) -> None:
    if dataset["row_id"].duplicated().any() or dataset.duplicated(["pair_id", "label"]).any():
        raise ValueError("Duplicate feature row/pair identity.")
    pairs = dataset.groupby("pair_id")["label"].apply(lambda x: tuple(sorted(x.tolist())))
    if not np.all(pairs == (0, 1)):
        raise ValueError("Incomplete clean/stego pair.")
    if np.any(~np.isfinite(dataset[list(FULL_FEATURES)].to_numpy(dtype=float))):
        raise ValueError("NaN/inf in frozen model inputs.")
    expected = {"development": (680, 68, 15), "unseen_new_maps": (1350, 135, 9)}
    for partition, (rows, replays, maps) in expected.items():
        subset = dataset[dataset["partition"] == partition]
        if (len(subset), subset.replay_file.nunique(), subset.beatmap_hash.nunique()) != (rows, replays, maps):
            raise ValueError(f"Unexpected {partition} dimensions.")
    dev = dataset[dataset.partition == "development"]
    unseen = dataset[dataset.partition == "unseen_new_maps"]
    if set(dev.replay_file) & set(unseen.replay_file) or set(dev.beatmap_hash) & set(unseen.beatmap_hash):
        raise ValueError("Development/unseen identity or beatmap overlap.")
    if dataset.groupby(["replay_file", "layout_seed"]).size().nunique() != 1:
        raise ValueError("Replay/layout pair completeness changed.")
    if dataset.groupby("layout_seed")["layout_key_id"].nunique().max() != 1:
        raise ValueError("Layout seed maps to inconsistent keys.")
    feature_versions = set(dataset["feature_version"])
    if feature_versions != {"strong-steganalysis-features-v1"}:
        raise ValueError(f"Unexpected feature version: {feature_versions}")


def fold_manifest(dataset: pd.DataFrame) -> pd.DataFrame:
    replay_groups = dataset["replay_file"].astype(str).to_numpy()
    shuffled = shuffled_group_ids(
        replay_groups, stable_seed(MESSAGE_SEED, "ALL", None, N_VALUE, "folds")
    )
    fold_by_row = np.full(len(dataset), -1, dtype=int)
    dummy = np.zeros((len(dataset), 1))
    labels = dataset["label"].to_numpy(dtype=int)
    for fold, (_, test) in enumerate(GroupKFold(5).split(dummy, labels, shuffled)):
        fold_by_row[test] = fold
    rows = dataset[["replay_file", "beatmap_hash", "partition"]].copy()
    rows["fold"] = fold_by_row
    if rows.groupby("replay_file")["fold"].nunique().max() != 1:
        raise ValueError("A replay spans CV folds.")
    return rows.drop_duplicates("replay_file").sort_values("replay_file").reset_index(drop=True)


def rf_reproduction() -> pd.DataFrame:
    features = pd.read_csv(HISTORICAL_SOURCE)
    expected = pd.read_csv(HISTORICAL_AUC)
    expected = expected[
        (expected["alpha_method"] == "sender_local_adaptive")
        & (expected["layout"] == "distributed")
        & (expected["feature_set"] == "full")
    ]
    rows = []
    for layout_seed in range(5):
        subset = frozen_ordered_subset(
            features, "validation", "sender_local_adaptive", "distributed", str(layout_seed)
        )
        groups = shuffled_group_ids(
            subset.replay_file.astype(str).to_numpy(),
            stable_seed(MESSAGE_SEED, "ALL", None, N_VALUE, "folds"),
        )
        values = subset[list(FULL_FEATURES)].to_numpy(dtype=float)
        labels = subset.label.to_numpy(dtype=int)
        scores = np.empty(len(subset), dtype=float)
        for fold, (train, test) in enumerate(GroupKFold(5).split(values, labels, groups), 1):
            model = build_classifier(
                "random_forest",
                stable_seed(MESSAGE_SEED, "ALL", None, N_VALUE, fold, "rf"),
            )
            model.fit(values[train], labels[train])
            scores[test] = positive_class_scores(model, values[test])
        observed = float(roc_auc_score(labels, scores))
        reference = float(expected[expected.layout_seed.astype(str) == str(layout_seed)].roc_auc.iloc[0])
        rows.append({
            "layout_seed": layout_seed,
            "observed_auc": observed,
            "frozen_auc": reference,
            "difference": observed - reference,
            "exact_match": int(observed == reference),
        })
    result = pd.DataFrame(rows)
    if not np.all(result["exact_match"] == 1):
        raise RuntimeError("RF reproduction gate failed; stop before alternative models.")
    return result


def classifier_config() -> dict:
    return {
        "version": VERSION,
        "scientific_status": "frozen_before_alternative_classifier_outcomes",
        "classifiers": {
            "random_forest": {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 2, "n_jobs": 1},
            "logistic_regression": {"pipeline": ["StandardScaler", "LogisticRegression"], "C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000, "tol": 1e-4},
            "rbf_svm": {"pipeline": ["StandardScaler", "SVC"], "kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": False},
            "hist_gradient_boosting": {"constructor": "HistGradientBoostingClassifier(random_state=fold_seed)", "other_parameters": "sklearn 1.9.0 defaults"},
        },
        "score_outputs": {"random_forest": "predict_proba[:,positive_class]", "logistic_regression": "decision_function", "rbf_svm": "decision_function", "hist_gradient_boosting": "predict_proba[:,positive_class]"},
        "feature_sets": {"baseline": list(BASELINE_FEATURES), "position_only": list(POSITION_FEATURES), "full": list(FULL_FEATURES)},
        "primary_physical_method": "adaptive distributed, N=8, normal 7.5%, SECDED(8,4), historical PN, five frozen layouts",
        "replay_grouped_scope": "pooled development plus unseen-new-map exact-system rows",
        "generalization_scope": "development exact-system rows -> disjoint unseen-map NORMAL_7_5 exact-system rows",
        "cv": {"folds": 5, "group": "replay_file", "fold_seed_rule": "historical shuffled_group_ids seed", "same_fold_manifest_all_models": True},
        "bootstrap": {"iterations": 2000, "replay_grouped_unit": "replay_file", "generalization_unit": "beatmap_hash", "paired_model_differences": True, "seed": 20260902},
        "random_state_rule": "stable_seed(42,classifier-family-robustness-v1,scope,classifier,feature_set,fold_or_fit)",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "scipy": scipy.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "input_sha256": {"development_features": file_sha256(DEV_SOURCE), "unseen_features": file_sha256(NEW_MAP_SOURCE), "historical_features": file_sha256(HISTORICAL_SOURCE), "historical_auc": file_sha256(HISTORICAL_AUC), "feature_schema": file_sha256(FEATURE_SCHEMA)},
        "prohibitions": ["no hyperparameter search", "no new features", "no validation-based selection", "no physical replay generation", "no host-aware PN primary rows"],
    }


def main() -> None:
    if OUTPUT.exists() or CONFIG.exists():
        raise FileExistsError("Classifier robustness output/config exists; refusing overwrite.")
    OUTPUT.mkdir(parents=True)
    dataset = build_dataset()
    validate_dataset(dataset)
    dataset.to_csv(OUTPUT / "dataset_manifest.csv", index=False)
    folds = fold_manifest(dataset)
    folds.to_csv(OUTPUT / "fold_manifest.csv", index=False)
    reproduction = rf_reproduction()
    reproduction.to_csv(OUTPUT / "rf_reproduction.csv", index=False)
    config = classifier_config()
    CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "classifier_config.json").write_bytes(CONFIG.read_bytes())
    audit = f"""# Classifier-family robustness data audit

- Historical adaptive+DISTRIBUTED matrix: `adaptive_layout_strong_v1/features.csv`; 7.5% uncoded message bits, so it is not physically equivalent to the final SECDED method. It is used only for exact RF reproduction and historical scope documentation.
- Primary final-system development matrix: historical-PN rows from `pn_key_robustness_v1/timing_features.csv`; 68 replays, 15 maps, five layouts, adaptive alpha v2, normal 7.5%, SECDED(8,4).
- Primary final-system unseen-map matrix: NORMAL_7_5 rows from `one_codeword_floor_validation_v1/timing_features.csv`; 135 replays, 9 new maps, five matching layout keys, identical method semantics.
- Development/unseen replay and beatmap overlap: zero. Dataset total: {len(dataset)} rows, {dataset.replay_file.nunique()} replays, {dataset.beatmap_hash.nunique()} maps.
- Every physical config has exactly one clean and one stego row. No duplicate row/pair identity and no NaN/inf among the frozen 29 model inputs.
- Only frozen timing features are copied into the model matrix; PN/layout IDs, message, alpha, BER, decoder, ECC outcome, integrity status, and ground truth are metadata and never model inputs.
- Fold grouping is by replay and shared across all classifiers. Scaling exists only inside Logistic/SVM sklearn Pipelines and is therefore fit within each training fold or development-only fit.
- Frozen RF FULL AUC reproduced exactly for all five historical distributed layout seeds.
- No `.osr` file was written, reloaded, or rematched for this branch.
"""
    (OUTPUT / "audit.md").write_text(audit, encoding="utf-8")
    print(f"dataset_rows={len(dataset)} replays={dataset.replay_file.nunique()} maps={dataset.beatmap_hash.nunique()}")
    print(f"rf_max_abs_difference={reproduction.difference.abs().max():.17g}")
    for path in (CONFIG, OUTPUT / "dataset_manifest.csv", OUTPUT / "fold_manifest.csv"):
        print(f"{path}: {file_sha256(path)}")


if __name__ == "__main__":
    main()
