"""Evaluate frozen classifier families on existing final-system timing features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from osu_stego.analysis.classifier_families import CLASSIFIER_NAMES, build_classifier, positive_class_scores
from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_pilot_ber_sweep import stable_seed


VERSION = "classifier-family-robustness-v1"
OUTPUT = RESULTS_DIR / "classifier_family_robustness_v1"
CONFIG = CONFIG_DIR / "steganalysis_classifier_robustness_v1.json"


def clustered_auc_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    unique = np.asarray(sorted(set(groups.astype(str))))
    indices = {group: np.flatnonzero(groups.astype(str) == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        chosen = np.concatenate([indices[group] for group in sampled])
        if len(np.unique(labels[chosen])) == 2:
            values.append(roc_auc_score(labels[chosen], scores[chosen]))
    if not values:
        raise RuntimeError("No valid clustered bootstrap samples.")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def seed_for(scope: str, classifier: str, feature_set: str, fold: str | int) -> int:
    return stable_seed(42, VERSION, scope, classifier, feature_set, fold)


def model_diagnostics(model, feature_names: list[str], scope: str, classifier: str, fold: str | int) -> tuple[list[dict], list[dict]]:
    coefficients: list[dict] = []
    importances: list[dict] = []
    if classifier == "logistic_regression":
        values = model.named_steps["classifier"].coef_[0]
        coefficients = [
            {"scope": scope, "fold": fold, "feature": name, "standardized_coefficient": float(value)}
            for name, value in zip(feature_names, values)
        ]
    elif classifier == "random_forest":
        importances = [
            {"scope": scope, "classifier": classifier, "fold": fold, "feature": name, "importance": float(value), "importance_kind": "impurity"}
            for name, value in zip(feature_names, model.feature_importances_)
        ]
    return coefficients, importances


def replay_grouped(
    dataset: pd.DataFrame,
    folds: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    iterations: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    frame = dataset.merge(folds[["replay_file", "fold"]], on="replay_file", validate="many_to_one")
    auc_rows: list[dict] = []
    prediction_rows: list[dict] = []
    train_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    importance_rows: list[dict] = []
    for feature_set, feature_names in feature_sets.items():
        values = frame[feature_names].to_numpy(dtype=float)
        labels = frame["label"].to_numpy(dtype=int)
        for classifier in CLASSIFIER_NAMES:
            scores = np.empty(len(frame), dtype=float)
            fold_train_auc = []
            fold_test_auc = []
            for fold in sorted(frame["fold"].unique()):
                train = frame["fold"].to_numpy() != fold
                test = ~train
                model = build_classifier(
                    classifier, seed_for("replay_grouped", classifier, feature_set, int(fold))
                )
                model.fit(values[train], labels[train])
                train_score = positive_class_scores(model, values[train])
                test_score = positive_class_scores(model, values[test])
                scores[test] = test_score
                fold_train_auc.append(roc_auc_score(labels[train], train_score))
                fold_test_auc.append(roc_auc_score(labels[test], test_score))
                if feature_set == "full":
                    coefs, imports = model_diagnostics(
                        model, feature_names, "replay_grouped", classifier, int(fold)
                    )
                    coefficient_rows.extend(coefs)
                    importance_rows.extend(imports)
            auc = float(roc_auc_score(labels, scores))
            low, high = clustered_auc_interval(
                labels, scores, frame.replay_file.astype(str).to_numpy(),
                seed_for("bootstrap_replay", classifier, feature_set, "ci"), iterations,
            )
            auc_rows.append({
                "scope": "replay_grouped", "classifier": classifier,
                "feature_set": feature_set, "roc_auc": auc,
                "auc_ci95_low": low, "auc_ci95_high": high,
                "bootstrap_unit": "replay_file", "bootstrap_iterations": iterations,
                "rows": len(frame), "replay_pairs": frame.replay_file.nunique(),
                "beatmaps": frame.beatmap_hash.nunique(), "folds": 5,
            })
            train_rows.append({
                "scope": "replay_grouped", "classifier": classifier,
                "feature_set": feature_set,
                "mean_fold_train_auc": float(np.mean(fold_train_auc)),
                "mean_fold_test_auc": float(np.mean(fold_test_auc)),
                "pooled_test_auc": auc,
            })
            for row, score in zip(frame.itertuples(index=False), scores.tolist()):
                prediction_rows.append({
                    "scope": "replay_grouped", "classifier": classifier,
                    "feature_set": feature_set, "row_id": row.row_id,
                    "pair_id": row.pair_id, "replay_file": row.replay_file,
                    "beatmap_hash": row.beatmap_hash, "partition": row.partition,
                    "performance_category": row.performance_category,
                    "layout_seed": row.layout_seed, "label": row.label,
                    "fold": row.fold, "score": score,
                })
    return auc_rows, prediction_rows, train_rows, coefficient_rows, importance_rows


def generalization(
    dataset: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    iterations: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    development = dataset[dataset.partition == "development"].reset_index(drop=True)
    unseen = dataset[dataset.partition == "unseen_new_maps"].reset_index(drop=True)
    auc_rows: list[dict] = []
    prediction_rows: list[dict] = []
    train_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    importance_rows: list[dict] = []
    for feature_set, feature_names in feature_sets.items():
        train_values = development[feature_names].to_numpy(dtype=float)
        train_labels = development.label.to_numpy(dtype=int)
        test_values = unseen[feature_names].to_numpy(dtype=float)
        test_labels = unseen.label.to_numpy(dtype=int)
        for classifier in CLASSIFIER_NAMES:
            model = build_classifier(
                classifier, seed_for("development_to_unseen", classifier, feature_set, "fit")
            )
            model.fit(train_values, train_labels)
            train_score = positive_class_scores(model, train_values)
            test_score = positive_class_scores(model, test_values)
            auc = float(roc_auc_score(test_labels, test_score))
            low, high = clustered_auc_interval(
                test_labels, test_score, unseen.beatmap_hash.astype(str).to_numpy(),
                seed_for("bootstrap_map", classifier, feature_set, "ci"), iterations,
            )
            auc_rows.append({
                "scope": "development_to_unseen", "classifier": classifier,
                "feature_set": feature_set, "roc_auc": auc,
                "auc_ci95_low": low, "auc_ci95_high": high,
                "bootstrap_unit": "beatmap_hash", "bootstrap_iterations": iterations,
                "development_rows": len(development), "development_replays": development.replay_file.nunique(),
                "development_maps": development.beatmap_hash.nunique(), "test_rows": len(unseen),
                "test_replays": unseen.replay_file.nunique(), "test_maps": unseen.beatmap_hash.nunique(),
            })
            train_rows.append({
                "scope": "development_to_unseen", "classifier": classifier,
                "feature_set": feature_set,
                "mean_fold_train_auc": float(roc_auc_score(train_labels, train_score)),
                "mean_fold_test_auc": auc, "pooled_test_auc": auc,
            })
            if feature_set == "full":
                coefs, imports = model_diagnostics(
                    model, feature_names, "development_to_unseen", classifier, "development_fit"
                )
                coefficient_rows.extend(coefs)
                importance_rows.extend(imports)
            for row, score in zip(unseen.itertuples(index=False), test_score.tolist()):
                prediction_rows.append({
                    "scope": "development_to_unseen", "classifier": classifier,
                    "feature_set": feature_set, "row_id": row.row_id,
                    "pair_id": row.pair_id, "replay_file": row.replay_file,
                    "beatmap_hash": row.beatmap_hash, "partition": row.partition,
                    "performance_category": row.performance_category,
                    "layout_seed": row.layout_seed, "label": row.label,
                    "fold": "unseen", "score": score,
                })
    return auc_rows, prediction_rows, train_rows, coefficient_rows, importance_rows


def pn_subset(folds: pd.DataFrame) -> pd.DataFrame:
    subset_lock = json.loads(
        (RESULTS_DIR / "pn_key_robustness_phase2_v1" / "detector_subset_lock.json").read_text()
    )
    phase1 = pd.read_csv(RESULTS_DIR / "pn_key_robustness_v1" / "timing_features.csv")
    phase1 = phase1[(phase1.historical_pn == 1) & (phase1.layout_seed < 3)].copy()
    phase2 = pd.read_csv(RESULTS_DIR / "pn_key_robustness_phase2_v1" / "timing_features.csv")
    phase2 = phase2[
        phase2.pn_key_id.isin(subset_lock["phase2_key_ids"])
        & phase2.layout_seed.isin(subset_lock["layout_seeds"])
    ].copy()
    frame = pd.concat([phase1, phase2], ignore_index=True, sort=False)
    frame = frame.merge(folds[["replay_file", "fold"]], on="replay_file", validate="many_to_one")
    rows = []
    features = json.loads(CONFIG.read_text())["feature_sets"]["full"]
    for pn_key_id, group in frame.groupby("pn_key_id"):
        group = group.reset_index(drop=True)
        values = group[features].to_numpy(dtype=float)
        labels = group.label.to_numpy(dtype=int)
        for classifier in CLASSIFIER_NAMES:
            scores = np.empty(len(group), dtype=float)
            for fold in sorted(group.fold.unique()):
                train = group.fold.to_numpy() != fold
                test = ~train
                model = build_classifier(
                    classifier, seed_for("pn_subset", classifier, "full", int(fold))
                )
                model.fit(values[train], labels[train])
                scores[test] = positive_class_scores(model, values[test])
            rows.append({
                "pn_key_id": pn_key_id,
                "historical_pn": int(pn_key_id == "54ca9c579c4d4b9d"),
                "classifier": classifier,
                "feature_set": "full",
                "roc_auc": float(roc_auc_score(labels, scores)),
                "rows": len(group), "replays": group.replay_file.nunique(),
                "layouts": group.layout_seed.nunique(),
            })
    return pd.DataFrame(rows)


def run(output: Path) -> None:
    core = [output / "predictions.csv", output / "auc_replay_grouped.csv", output / "auc_dev_to_validation.csv"]
    if all(path.exists() for path in core):
        print("classifier-family evaluation already complete; new fits=0")
        return
    if any(path.exists() for path in core):
        raise RuntimeError("Partial core outputs found; refusing to mix runs.")
    config = json.loads(CONFIG.read_text())
    dataset = pd.read_csv(output / "dataset_manifest.csv")
    folds = pd.read_csv(output / "fold_manifest.csv")
    feature_sets = config["feature_sets"]
    iterations = int(config["bootstrap"]["iterations"])
    replay = replay_grouped(dataset, folds, feature_sets, iterations)
    general = generalization(dataset, feature_sets, iterations)
    pd.DataFrame(replay[0]).to_csv(output / "auc_replay_grouped.csv", index=False)
    pd.DataFrame(general[0]).to_csv(output / "auc_dev_to_validation.csv", index=False)
    pd.DataFrame([*replay[1], *general[1]]).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame([*replay[2], *general[2]]).to_csv(output / "train_vs_test_auc.csv", index=False)
    pd.DataFrame([*replay[3], *general[3]]).to_csv(output / "logistic_coefficients.csv", index=False)
    pd.DataFrame([*replay[4], *general[4]]).to_csv(output / "exploratory_feature_importance.csv", index=False)
    pn_subset(folds).to_csv(output / "pn_subset_auc.csv", index=False)
    print("classifier-family evaluation complete; new fits>0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--lock", type=Path, default=OUTPUT / "experiment_lock.json")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    for path, field in (
        (args.config, "classifier_config_sha256"),
        (args.output_dir / "dataset_manifest.csv", "dataset_manifest_sha256"),
        (args.output_dir / "fold_manifest.csv", "fold_manifest_sha256"),
        (Path(__file__), "runner_sha256"),
        (Path("osu_stego/analysis/classifier_families.py"), "classifier_module_sha256"),
    ):
        if file_sha256(path) != lock[field]:
            raise RuntimeError(f"Locked hash mismatch: {path}")
    run(args.output_dir)


if __name__ == "__main__":
    main()
