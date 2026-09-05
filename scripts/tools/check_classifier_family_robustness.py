"""Pre-outcome leakage, reproduction, and classifier plumbing checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from osu_stego.analysis.classifier_families import CLASSIFIER_NAMES, build_classifier, positive_class_scores
from osu_stego.paths import CONFIG_DIR, RESULTS_DIR


OUTPUT = RESULTS_DIR / "classifier_family_robustness_v1"


def main() -> None:
    config = json.loads((CONFIG_DIR / "steganalysis_classifier_robustness_v1.json").read_text())
    dataset = pd.read_csv(OUTPUT / "dataset_manifest.csv")
    folds = pd.read_csv(OUTPUT / "fold_manifest.csv")
    reproduction = pd.read_csv(OUTPUT / "rf_reproduction.csv")
    full = config["feature_sets"]["full"]
    if len(full) != 29 or len(set(full)) != 29:
        raise AssertionError("FULL feature order is not exactly 29 unique fields.")
    if not np.all(reproduction.exact_match == 1) or reproduction.difference.abs().max() != 0:
        raise AssertionError("Historical RF reproduction gate failed.")
    if dataset.row_id.duplicated().any() or dataset.duplicated(["pair_id", "label"]).any():
        raise AssertionError("Duplicate sample identity.")
    pairs = dataset.groupby("pair_id").label.apply(lambda x: tuple(sorted(x.tolist())))
    if not np.all(pairs == (0, 1)):
        raise AssertionError("Incomplete clean/stego pair.")
    if np.any(~np.isfinite(dataset[full].to_numpy(float))):
        raise AssertionError("NaN/inf model inputs.")
    merged = dataset.merge(folds[["replay_file", "fold"]], on="replay_file", validate="many_to_one")
    if merged.groupby("replay_file").fold.nunique().max() != 1:
        raise AssertionError("Replay leakage across folds.")
    if set(dataset[dataset.partition == "development"].beatmap_hash) & set(dataset[dataset.partition == "unseen_new_maps"].beatmap_hash):
        raise AssertionError("Development/unseen beatmap leakage.")

    pilot_replays = sorted(dataset.replay_file.unique())[:20]
    pilot = dataset[dataset.replay_file.isin(pilot_replays)].reset_index(drop=True)
    values = pilot[full].to_numpy(float)
    labels = pilot.label.to_numpy(int)
    for name in CLASSIFIER_NAMES:
        first = build_classifier(name, 20260902)
        second = build_classifier(name, 20260902)
        first.fit(values, labels)
        second.fit(values, labels)
        score_a = positive_class_scores(first, values)
        score_b = positive_class_scores(second, values)
        if not np.array_equal(score_a, score_b):
            raise AssertionError(f"Non-deterministic repeated pilot: {name}")
        if tuple(first.classes_.tolist()) != (0, 1):
            raise AssertionError(f"Unexpected class orientation: {name}")
        if name in ("logistic_regression", "rbf_svm") and not isinstance(first, Pipeline):
            raise AssertionError(f"Scaling is not inside Pipeline: {name}")
        if name == "logistic_regression":
            fitted = first.named_steps["classifier"]
            if int(fitted.n_iter_[0]) >= int(fitted.max_iter):
                raise AssertionError("Logistic regression did not converge in pilot.")
    print("classifier-family pre-outcome checks: PASS")


if __name__ == "__main__":
    main()
