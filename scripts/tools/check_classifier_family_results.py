"""Audit completed classifier-family robustness outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from osu_stego.analysis.classifier_families import CLASSIFIER_NAMES
from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256


OUTPUT = RESULTS_DIR / "classifier_family_robustness_v1"
CONFIG = CONFIG_DIR / "steganalysis_classifier_robustness_v1.json"
FEATURE_SETS = ("baseline", "position_only", "full")


def assert_auc_table(
    predictions: pd.DataFrame,
    table_path: Path,
    scope: str,
) -> None:
    """Recompute every reported point AUC from the saved predictions."""
    table = pd.read_csv(table_path).set_index(["classifier", "feature_set"])
    scoped = predictions[predictions.scope == scope]
    for classifier in CLASSIFIER_NAMES:
        for feature_set in FEATURE_SETS:
            rows = scoped[
                (scoped.classifier == classifier)
                & (scoped.feature_set == feature_set)
            ]
            observed = float(roc_auc_score(rows.label, rows.score))
            reported = float(table.loc[(classifier, feature_set), "roc_auc"])
            if not np.isclose(observed, reported, rtol=0.0, atol=1e-14):
                raise AssertionError(
                    f"AUC mismatch for {scope}/{classifier}/{feature_set}: "
                    f"{observed} != {reported}"
                )


def main() -> None:
    lock = json.loads((OUTPUT / "experiment_lock.json").read_text())
    locked_files = (
        (CONFIG, "classifier_config_sha256"),
        (OUTPUT / "dataset_manifest.csv", "dataset_manifest_sha256"),
        (OUTPUT / "fold_manifest.csv", "fold_manifest_sha256"),
        (Path("scripts/experiments/run_classifier_family_robustness.py"), "runner_sha256"),
        (Path("osu_stego/analysis/classifier_families.py"), "classifier_module_sha256"),
    )
    for path, key in locked_files:
        if file_sha256(path) != lock[key]:
            raise AssertionError(f"Locked hash mismatch: {path}")

    config = json.loads(CONFIG.read_text())
    dataset = pd.read_csv(OUTPUT / "dataset_manifest.csv")
    folds = pd.read_csv(OUTPUT / "fold_manifest.csv")
    predictions = pd.read_csv(OUTPUT / "predictions.csv")

    if len(dataset) != 2030 or dataset.replay_file.nunique() != 203:
        raise AssertionError("Unexpected primary dataset size.")
    if dataset.row_id.duplicated().any():
        raise AssertionError("Duplicate primary row_id.")
    if folds.replay_file.duplicated().any() or folds.replay_file.nunique() != 203:
        raise AssertionError("Invalid fold manifest.")
    if folds.fold.nunique() != 5:
        raise AssertionError("Expected five common replay folds.")
    if set(dataset[dataset.partition == "development"].beatmap_hash) & set(
        dataset[dataset.partition == "unseen_new_maps"].beatmap_hash
    ):
        raise AssertionError("Development/unseen beatmap leakage.")
    if len(config["feature_sets"]["full"]) != 29:
        raise AssertionError("FULL feature schema changed.")

    expected_rows = {
        "replay_grouped": 2030,
        "development_to_unseen": 1350,
    }
    for scope, expected in expected_rows.items():
        scoped = predictions[predictions.scope == scope]
        sizes = scoped.groupby(["classifier", "feature_set"]).row_id.nunique()
        if set(sizes.index.get_level_values("classifier")) != set(CLASSIFIER_NAMES):
            raise AssertionError(f"Classifier mismatch in {scope}.")
        if set(sizes.index.get_level_values("feature_set")) != set(FEATURE_SETS):
            raise AssertionError(f"Feature-set mismatch in {scope}.")
        if not np.all(sizes.to_numpy() == expected):
            raise AssertionError(f"Model-specific row dropping in {scope}.")
        if scoped.duplicated(["classifier", "feature_set", "row_id"]).any():
            raise AssertionError(f"Duplicate predictions in {scope}.")
        identities = scoped.groupby(["classifier", "feature_set"]).row_id.apply(set)
        if any(value != identities.iloc[0] for value in identities.iloc[1:]):
            raise AssertionError(f"Models did not receive identical rows in {scope}.")

    if not np.isfinite(predictions.score.to_numpy(float)).all():
        raise AssertionError("Non-finite prediction score.")
    labels_per_row = predictions.groupby(["scope", "row_id"]).label.nunique()
    if labels_per_row.max() != 1:
        raise AssertionError("Inconsistent labels across models.")

    assert_auc_table(
        predictions, OUTPUT / "auc_replay_grouped.csv", "replay_grouped"
    )
    assert_auc_table(
        predictions,
        OUTPUT / "auc_dev_to_validation.csv",
        "development_to_unseen",
    )
    reproduction = pd.read_csv(OUTPUT / "rf_reproduction.csv")
    if not np.all(reproduction.exact_match == 1) or reproduction.difference.abs().max() != 0:
        raise AssertionError("Historical RF reproduction is no longer exact.")

    expected_outputs = (
        "audit.md", "classifier_config.json", "experiment_lock.json",
        "dataset_manifest.csv", "fold_manifest.csv", "rf_reproduction.csv",
        "auc_replay_grouped.csv", "auc_dev_to_validation.csv",
        "auc_paired_vs_rf.csv", "predictions.csv",
        "classifier_score_correlations.csv", "per_beatmap_auc.csv",
        "quality_group_auc.csv", "logistic_coefficients.csv",
        "logistic_coefficients_summary.csv", "exploratory_feature_importance.csv",
        "pn_subset_auc.csv", "pn_subset_summary.csv", "train_vs_test_auc.csv",
        "diagnostics.csv", "classifier_robustness_report.md",
    )
    missing = [name for name in expected_outputs if not (OUTPUT / name).exists()]
    if missing:
        raise AssertionError(f"Missing result outputs: {missing}")
    print("classifier-family completed-results audit: PASS")


if __name__ == "__main__":
    main()
