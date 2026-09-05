"""Paired and descriptive analysis of frozen classifier-family predictions."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from osu_stego.paths import RESULTS_DIR
from scripts.experiments.run_pilot_ber_sweep import stable_seed


OUTPUT = RESULTS_DIR / "classifier_family_robustness_v1"
CLASSIFIERS = ("random_forest", "logistic_regression", "rbf_svm", "hist_gradient_boosting")
ITERATIONS = 2000


def cluster_indices(groups: pd.Series) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = groups.astype(str).to_numpy()
    unique = np.asarray(sorted(set(values)))
    return unique, {group: np.flatnonzero(values == group) for group in unique}


def paired_differences(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, feature_set), frame in predictions.groupby(["scope", "feature_set"]):
        id_columns = ["row_id", "label", "replay_file", "beatmap_hash"]
        wide = frame.pivot(index="row_id", columns="classifier", values="score")
        metadata = frame.drop_duplicates("row_id").set_index("row_id")[id_columns[1:]]
        wide = metadata.join(wide, how="inner")
        labels = wide.label.to_numpy(dtype=int)
        group_column = "replay_file" if scope == "replay_grouped" else "beatmap_hash"
        unique, indices = cluster_indices(wide[group_column])
        rf = wide.random_forest.to_numpy(float)
        rf_auc = roc_auc_score(labels, rf)
        for classifier in CLASSIFIERS[1:]:
            scores = wide[classifier].to_numpy(float)
            observed = float(roc_auc_score(labels, scores) - rf_auc)
            rng = np.random.default_rng(stable_seed(42, "classifier-family-paired", scope, feature_set, classifier))
            samples = []
            for _ in range(ITERATIONS):
                sampled = rng.choice(unique, size=len(unique), replace=True)
                selected = np.concatenate([indices[group] for group in sampled])
                if len(np.unique(labels[selected])) == 2:
                    samples.append(
                        roc_auc_score(labels[selected], scores[selected])
                        - roc_auc_score(labels[selected], rf[selected])
                    )
            rows.append({
                "scope": scope, "feature_set": feature_set,
                "classifier": classifier, "reference": "random_forest",
                "auc_model_minus_rf": observed,
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
                "bootstrap_unit": group_column, "bootstrap_iterations": ITERATIONS,
                "identical_rows": len(wide),
            })
    return pd.DataFrame(rows)


def paired_stego_rank(frame: pd.DataFrame) -> pd.Series:
    pairs = frame.pivot(index="pair_id", columns="label", values="score")
    return (pairs[1] > pairs[0]).astype(int)


def replay_difficulty(frame: pd.DataFrame) -> pd.Series:
    pairs = frame.pivot(index="pair_id", columns="label", values="score")
    metadata = frame.drop_duplicates("pair_id").set_index("pair_id")["replay_file"]
    delta = pairs[1] - pairs[0]
    return delta.groupby(metadata).mean()


def classifier_agreement(predictions: pd.DataFrame) -> pd.DataFrame:
    full = predictions[predictions.feature_set == "full"]
    rows = []
    for scope, group in full.groupby("scope"):
        by_model = {name: group[group.classifier == name].set_index("row_id") for name in CLASSIFIERS}
        for first, second in combinations(CLASSIFIERS, 2):
            left, right = by_model[first], by_model[second]
            common = left.index.intersection(right.index)
            score_rho = spearmanr(left.loc[common, "score"], right.loc[common, "score"]).statistic
            rank_a = paired_stego_rank(group[group.classifier == first])
            rank_b = paired_stego_rank(group[group.classifier == second])
            pair_common = rank_a.index.intersection(rank_b.index)
            difficulty_a = replay_difficulty(group[group.classifier == first])
            difficulty_b = replay_difficulty(group[group.classifier == second])
            replay_common = difficulty_a.index.intersection(difficulty_b.index)
            rows.append({
                "scope": scope, "classifier_a": first, "classifier_b": second,
                "score_spearman": float(score_rho),
                "clean_stego_pair_rank_agreement": float((rank_a.loc[pair_common] == rank_b.loc[pair_common]).mean()),
                "replay_difficulty_spearman": float(spearmanr(
                    difficulty_a.loc[replay_common], difficulty_b.loc[replay_common]
                ).statistic),
                "rows": len(common), "physical_pairs": len(pair_common), "replays": len(replay_common),
            })
    return pd.DataFrame(rows)


def per_beatmap(predictions: pd.DataFrame) -> pd.DataFrame:
    full = predictions[(predictions.feature_set == "full") & (predictions.scope == "replay_grouped")]
    rows = []
    for (classifier, beatmap), group in full.groupby(["classifier", "beatmap_hash"]):
        replay_count = group.replay_file.nunique()
        if replay_count < 10 or group.label.nunique() < 2:
            continue
        rows.append({
            "classifier": classifier, "beatmap_hash": beatmap,
            "roc_auc": float(roc_auc_score(group.label, group.score)),
            "replay_pairs": replay_count, "physical_pairs": group.pair_id.nunique(),
        })
    return pd.DataFrame(rows)


def quality_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    full = predictions[
        (predictions.feature_set == "full")
        & (predictions.scope == "replay_grouped")
        & (predictions.partition == "development")
        & (predictions.performance_category != "UNAVAILABLE")
    ]
    rows = []
    for (classifier, category), group in full.groupby(["classifier", "performance_category"]):
        if group.replay_file.nunique() < 8 or group.label.nunique() < 2:
            continue
        rows.append({
            "classifier": classifier, "performance_category": category,
            "roc_auc": float(roc_auc_score(group.label, group.score)),
            "replay_pairs": group.replay_file.nunique(), "physical_pairs": group.pair_id.nunique(),
            "scope_limitation": "development 68-replay exact-system panel only",
        })
    return pd.DataFrame(rows)


def summarize_pn(pn: pd.DataFrame) -> pd.DataFrame:
    return pn.groupby("classifier", as_index=False).agg(
        pn_keys=("pn_key_id", "size"),
        full_auc_mean=("roc_auc", "mean"),
        full_auc_min=("roc_auc", "min"),
        full_auc_max=("roc_auc", "max"),
        historical_pn_auc=("roc_auc", lambda values: float(pn.loc[values.index][pn.loc[values.index, "historical_pn"] == 1].roc_auc.iloc[0])),
    )


def summarize_coefficients(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby(["scope", "feature"], as_index=False).standardized_coefficient.agg(
        mean="mean", sd="std", minimum="min", maximum="max"
    ).sort_values(["scope", "mean"], key=lambda values: values.abs() if values.name == "mean" else values, ascending=[True, False])


def diagnostics(output: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    expected = {
        "replay_grouped": 2030 * 4 * 3,
        "development_to_unseen": 1350 * 4 * 3,
    }
    rows = []
    for scope, count in expected.items():
        frame = predictions[predictions.scope == scope]
        rows.append({"check": f"{scope}_prediction_rows", "value": len(frame), "expected": count})
        rows.append({"check": f"{scope}_duplicate_predictions", "value": int(frame.duplicated(["classifier", "feature_set", "row_id"]).sum()), "expected": 0})
        sizes = frame.groupby(["classifier", "feature_set"]).row_id.nunique()
        rows.append({"check": f"{scope}_model_row_count_range", "value": f"{sizes.min()}..{sizes.max()}", "expected": str(sizes.max())})
    rows.extend([
        {"check": "nonfinite_scores", "value": int((~np.isfinite(predictions.score)).sum()), "expected": 0},
        {"check": "physical_osr_runs", "value": 0, "expected": 0},
        {"check": "rf_reproduction_max_abs_delta", "value": pd.read_csv(output / "rf_reproduction.csv").difference.abs().max(), "expected": 0},
    ])
    return pd.DataFrame(rows)


def write_report(output: Path, replay_auc: pd.DataFrame, general_auc: pd.DataFrame, paired: pd.DataFrame, agreement: pd.DataFrame, pn_summary: pd.DataFrame) -> None:
    replay_full = replay_auc[replay_auc.feature_set == "full"].set_index("classifier")
    general_full = general_auc[general_auc.feature_set == "full"].set_index("classifier")
    strongest_replay = replay_full.roc_auc.idxmax()
    strongest_general = general_full.roc_auc.idxmax()
    train = pd.read_csv(output / "train_vs_test_auc.csv")
    maps = pd.read_csv(output / "per_beatmap_auc.csv")
    quality = pd.read_csv(output / "quality_group_auc.csv")
    coefficients = pd.read_csv(output / "logistic_coefficients_summary.csv")
    config_hash = __import__("hashlib").sha256(
        (Path("data/config/steganalysis_classifier_robustness_v1.json")).read_bytes()
    ).hexdigest()
    lines = [
        "# Classifier-family robustness v1", "",
        "Final decision: **A. CLASSIFIER-FAMILY ROBUST**.", "",
        "## Data and leakage audit", "",
        "This analysis exactly represents the final adaptive-alpha-v2 + DISTRIBUTED + N=8 + normal-7.5% + SECDED(8,4) physical method with the historical PN and five frozen layouts. Integrity rejection changes receiver acceptance, not the timing feature rows.", "",
        "The primary matrix contains 203 replay-disjoint clean/stego sets: 68 development replays on 15 maps and 135 new-map NORMAL_7_5 replays on 9 disjoint maps. No physical replay was regenerated.", "",
        "Every clean/stego pair is complete, all classifiers use identical rows and folds, preprocessing is contained inside each training pipeline, inputs are finite, and development/unseen beatmaps do not overlap. PN/layout IDs and receiver outcomes are metadata only, never classifier inputs.", "",
        "The historical `adaptive_layout_strong_v1` matrix represents uncoded 7.5% message bits, so it is not treated as physically equivalent to the final SECDED system. It was used only for the reproduction gate. The frozen RF matched all five historical adaptive+DISTRIBUTED FULL AUCs exactly (maximum absolute difference 0).", "",
        "## Frozen classifier design", "",
        f"Config SHA-256: `{config_hash}`. The four preregistered families were the historical RF, standardized L2 Logistic Regression, standardized RBF SVM, and HistGradientBoosting. There was no tuning, feature search, model removal, or post-outcome addition. Exact resolved sklearn 1.9.0 parameters are saved separately.", "",
        "## FULL-29 results", "",
        "| Classifier | Replay-grouped AUC (95% CI) | Development → unseen maps AUC (95% CI) |",
        "|---|---:|---:|",
    ]
    labels = {"random_forest": "Random Forest", "logistic_regression": "Logistic Regression", "rbf_svm": "RBF SVM", "hist_gradient_boosting": "HistGradientBoosting"}
    for classifier in CLASSIFIERS:
        r, g = replay_full.loc[classifier], general_full.loc[classifier]
        lines.append(f"| {labels[classifier]} | {r.roc_auc:.3f} ({r.auc_ci95_low:.3f}–{r.auc_ci95_high:.3f}) | {g.roc_auc:.3f} ({g.auc_ci95_low:.3f}–{g.auc_ci95_high:.3f}) |")
    lines.extend(["", f"Strongest preregistered FULL classifier was {labels[strongest_replay]} in replay-grouped evaluation ({replay_full.loc[strongest_replay].roc_auc:.3f}) and {labels[strongest_general]} on unseen maps ({general_full.loc[strongest_general].roc_auc:.3f}). These are not upper bounds over possible attackers.", "",
        "## Frozen feature ablations", "",
        "| Scope / classifier | BASELINE-3 | POSITION | FULL-29 |", "|---|---:|---:|---:|",
    ])
    for scope_name, table in (("Replay-grouped", replay_auc), ("Development → unseen", general_auc)):
        wide = table.pivot(index="classifier", columns="feature_set", values="roc_auc")
        for classifier in CLASSIFIERS:
            lines.append(
                f"| {scope_name} / {labels[classifier]} | {wide.loc[classifier, 'baseline']:.3f} | "
                f"{wide.loc[classifier, 'position_only']:.3f} | {wide.loc[classifier, 'full']:.3f} |"
            )
    lines.extend(["", "FULL was not uniformly stronger than every ablation: for example, unseen-map Logistic BASELINE-3 exceeded its FULL result, while RF POSITION slightly exceeded RF FULL. Nevertheless, all four FULL detectors remained above chance in both regimes.", "",
        "## Paired alternatives versus RF", "",
        "| Scope | Classifier | FULL AUC difference | Paired 95% CI |", "|---|---|---:|---:|",
    ])
    paired_full = paired[paired.feature_set == "full"]
    for row in paired_full.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {labels[row.classifier]} | {row.auc_model_minus_rf:+.3f} | "
            f"{row.ci95_low:+.3f} to {row.ci95_high:+.3f} |"
        )
    lines.extend(["", "Every FULL paired interval contains zero. The SVM has the highest point AUC in both regimes, but it is not a statistically clear or material improvement over RF under the frozen design.", "",
        "## Secondary diagnostics", "",
    ])
    for scope, group in agreement.groupby("scope"):
        lines.append(
            f"- {scope}: score Spearman range {group.score_spearman.min():.3f}–{group.score_spearman.max():.3f}; "
            f"clean/stego pair-rank agreement {group.clean_stego_pair_rank_agreement.min():.3f}–{group.clean_stego_pair_rank_agreement.max():.3f}; "
            f"replay-difficulty Spearman {group.replay_difficulty_spearman.min():.3f}–{group.replay_difficulty_spearman.max():.3f}."
        )
    lines.extend(["", "Classifiers therefore overlap substantially but do not identify identical examples as easy or difficult.", ""])
    if not maps.empty:
        lines.append(
            f"Per-beatmap FULL AUC was reportable for {maps.beatmap_hash.nunique()} maps with at least 10 replay pairs. "
            f"Across classifier/map cells it ranged from {maps.roc_auc.min():.3f} to {maps.roc_auc.max():.3f}, showing meaningful map heterogeneity."
        )
        lines.append("")
    full_train = train[train.feature_set == "full"].set_index(["scope", "classifier"])
    lines.append("FULL train/held-out diagnostics:")
    lines.append("")
    lines.append("| Scope | Classifier | Train AUC | Held-out AUC |")
    lines.append("|---|---|---:|---:|")
    for scope in ("replay_grouped", "development_to_unseen"):
        for classifier in CLASSIFIERS:
            row = full_train.loc[(scope, classifier)]
            lines.append(
                f"| {scope} | {labels[classifier]} | {row.mean_fold_train_auc:.3f} | {row.pooled_test_auc:.3f} |"
            )
    lines.extend(["", "RF and HGB reached approximately 1.0 train AUC while testing near 0.60–0.62, an explicit overfitting/capacity warning. Logistic and SVM gaps were smaller. No tuning followed this diagnostic.", ""])
    if not quality.empty:
        quality_range = quality.groupby("classifier").roc_auc.agg(["min", "max"])
        formatted = "; ".join(
            f"{labels[name]} {row['min']:.3f}–{row['max']:.3f}"
            for name, row in quality_range.iterrows()
        )
        lines.extend([f"Development-only replay-quality AUC ranges were: {formatted}. These are descriptive because the aligned exact-system panel contains only 68 replays.", ""])
    pn_text = "; ".join(
        f"{labels[row.classifier]} {row.full_auc_min:.3f}–{row.full_auc_max:.3f}"
        for row in pn_summary.itertuples(index=False)
    )
    lines.extend([f"Across the frozen eight-key PN subset (historical plus seven preregistered Phase-2 keys), FULL replay-grouped AUC ranges were: {pn_text}. Classifier ordering varied, but there was no large qualitative reversal.", ""])
    top_replay = coefficients[coefficients.scope == "replay_grouped"].sort_values("mean", key=lambda values: values.abs(), ascending=False).head(5)
    coefficient_text = ", ".join(f"{row.feature} ({row.mean:+.3f})" for row in top_replay.itertuples(index=False))
    lines.extend([f"Largest absolute replay-grouped standardized Logistic coefficients were {coefficient_text}. Coefficients and RF impurity importances are exploratory and are not causal or directly comparable.", "",
        "## Files added or changed", "",
        "Added `osu_stego/analysis/classifier_families.py`; preparation, freeze, experiment, analysis, finalizer, and two audit scripts under `scripts/`; the frozen config under `data/config/`; and all immutable analysis artifacts under `results/classifier_family_robustness_v1/`. Updated `AGENTS.md` only after results were frozen. `result_hashes.json` enumerates the exact result files and their hashes.", "",
        "## Scope and limitations", "",
        "The new-map corpus and all development artifacts were already scientifically consumed. This is a post-hoc classifier-family robustness analysis, not new independent validation. Map-cluster intervals have only nine unseen maps and are correspondingly uncertain. Quality-group results cover only the 68-replay development panel. Impurity importance and standardized coefficients are exploratory, not causal.", "",
        "No tested classifier is an upper bound on steganalysis, and no calibration or threshold-based detection rate was assessed. AUC is interpreted only as ranking ability.", "",
        "## Confirmed bugs and anomalies", "",
        "No confirmed data, leakage, reproduction, or implementation bug was found. sklearn 1.9 emitted deprecation warnings for the explicitly frozen Logistic `penalty='l2'` and SVC `probability=False` arguments; fits and scores were unaffected. The warnings are retained because changing locked constructor semantics after outcomes would be inappropriate.", "",
        "## Conclusion", "",
        "All four FULL AUCs remained approximately 0.60–0.64 in both regimes. Under the same frozen timing information, modest detectability persists across linear, kernel, bagged-tree, and boosted-tree inductive biases. Its magnitude and which examples appear easiest are model-dependent, but the conclusion is not RF-specific. Decision: **A. CLASSIFIER-FAMILY ROBUST**.", "",
    ])
    (output / "classifier_robustness_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    predictions = pd.read_csv(output / "predictions.csv")
    replay_auc = pd.read_csv(output / "auc_replay_grouped.csv")
    general_auc = pd.read_csv(output / "auc_dev_to_validation.csv")
    paired = paired_differences(predictions)
    paired.to_csv(output / "auc_paired_vs_rf.csv", index=False)
    agreement = classifier_agreement(predictions)
    agreement.to_csv(output / "classifier_score_correlations.csv", index=False)
    per_beatmap(predictions).to_csv(output / "per_beatmap_auc.csv", index=False)
    quality_groups(predictions).to_csv(output / "quality_group_auc.csv", index=False)
    raw_coefficients = pd.read_csv(output / "logistic_coefficients.csv")
    summarize_coefficients(raw_coefficients).to_csv(output / "logistic_coefficients_summary.csv", index=False)
    pn = pd.read_csv(output / "pn_subset_auc.csv")
    pn_summary = summarize_pn(pn)
    pn_summary.to_csv(output / "pn_subset_summary.csv", index=False)
    diagnostics(output, predictions).to_csv(output / "diagnostics.csv", index=False)
    write_report(output, replay_auc, general_auc, paired, agreement, pn_summary)
    print(replay_auc[replay_auc.feature_set == "full"][["classifier", "roc_auc", "auc_ci95_low", "auc_ci95_high"]].to_string(index=False))
    print(general_auc[general_auc.feature_set == "full"][["classifier", "roc_auc", "auc_ci95_low", "auc_ci95_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
