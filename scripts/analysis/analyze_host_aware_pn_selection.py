"""Analyze the locked host-aware PN-selection holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from osu_stego.analysis.timing_features import (
    BASELINE_FEATURES,
    FULL_FEATURES,
    POSITION_FEATURES,
)
from osu_stego.paths import CONFIG_DIR, DATASET_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import alpha_from_replay, load_sender_local_policy
from osu_stego.stego.ecc import encode_hamming_8_4_secded
from osu_stego.stego.host_aware_pn import (
    MESSAGE_FAMILIES,
    deterministic_message_family,
    predicted_margins,
    select_candidate,
    summarize_margins,
)
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS
from scripts.experiments.run_pilot_ber_sweep import (
    build_beatmap_index,
    load_map_offsets,
    prepare_replay,
)


OUTPUT = RESULTS_DIR / "host_aware_pn_selection_v1"
K_VALUES = (1, 2, 4, 8)
BOOTSTRAP_ITERATIONS = 2000


def outcome_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        errors = int(group["raw_bit_errors"].sum())
        bits = int(group["coded_bits"].sum())
        accepted = int(group["correct_accept"].sum() + group["wrong_accept"].sum())
        rows.append({
            **row,
            "configs": len(group),
            "raw_bit_errors": errors,
            "coded_bits_total": bits,
            "weighted_raw_ber": errors / bits,
            "post_ecc_bit_errors": int(group["post_ecc_bit_errors"].sum()),
            "useful_bits_total": int(group["useful_bits"].sum()),
            "post_ecc_ber": group["post_ecc_bit_errors"].sum() / group["useful_bits"].sum(),
            "correct_accept": int(group["correct_accept"].sum()),
            "wrong_accept": int(group["wrong_accept"].sum()),
            "reject": int(group["reject"].sum()),
            "full_message_recovery_rate": float((group["post_ecc_bit_errors"] == 0).mean()),
            "accepted_message_accuracy": (
                float(group["correct_accept"].sum() / accepted) if accepted else np.nan
            ),
        })
    return pd.DataFrame(rows)


def replay_cluster_bootstrap(known: pd.DataFrame) -> pd.DataFrame:
    replay_ids = np.asarray(sorted(known["replay_file"].unique()))
    rng = np.random.default_rng(20260902)
    rows = []
    for k_value in (2, 4, 8):
        subset = known[known["K"].isin((1, k_value))]
        observed = {}
        for key, group in subset.groupby("K"):
            observed[int(key)] = group["raw_bit_errors"].sum() / group["coded_bits"].sum()
        samples = np.empty(BOOTSTRAP_ITERATIONS)
        for iteration in range(BOOTSTRAP_ITERATIONS):
            selected = rng.choice(replay_ids, size=len(replay_ids), replace=True)
            totals = {1: [0, 0], k_value: [0, 0]}
            for replay_id in selected:
                replay = subset[subset["replay_file"] == replay_id]
                for key, group in replay.groupby("K"):
                    totals[int(key)][0] += int(group["raw_bit_errors"].sum())
                    totals[int(key)][1] += int(group["coded_bits"].sum())
            samples[iteration] = (
                totals[k_value][0] / totals[k_value][1]
                - totals[1][0] / totals[1][1]
            )
        delta = observed[k_value] - observed[1]
        rows.append({
            "comparison": f"K={k_value} minus K=1",
            "ber_k1": observed[1],
            "ber_k": observed[k_value],
            "delta_ber": delta,
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
            "bootstrap_unit": "replay",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        })
    return pd.DataFrame(rows)


def margin_validation(bits: pd.DataFrame) -> pd.DataFrame:
    rows = [{
        "section": "overall",
        "bin": "all",
        "bits": len(bits),
        "errors": int(bits["is_error"].sum()),
        "error_rate": float(bits["is_error"].mean()),
        "negative_margin_error_auc": float(roc_auc_score(
            bits["is_error"], -bits["predicted_margin"]
        )),
        "median_margin_correct": float(bits.loc[bits["is_error"] == 0, "predicted_margin"].median()),
        "median_margin_error": float(bits.loc[bits["is_error"] == 1, "predicted_margin"].median()),
    }]
    ranked = bits.copy()
    ranked["margin_bin"] = pd.qcut(
        ranked["predicted_margin"], 10, duplicates="drop"
    ).astype(str)
    for name, group in ranked.groupby("margin_bin", observed=True, sort=False):
        rows.append({
            "section": "margin_bin", "bin": name, "bits": len(group),
            "errors": int(group["is_error"].sum()),
            "error_rate": float(group["is_error"].mean()),
            "negative_margin_error_auc": np.nan,
            "median_margin_correct": np.nan, "median_margin_error": np.nan,
        })
    return pd.DataFrame(rows)


def margin_config_validation(
    scores: pd.DataFrame, physical: pd.DataFrame
) -> pd.DataFrame:
    selected = physical.merge(
        scores,
        left_on=["unit_id", "selected_pn_id"],
        right_on=["unit_id", "pn_key_id"],
        suffixes=("", "_score"),
        validate="one_to_one",
    )
    rows = []
    for metric, risk in (
        ("nonpositive_count", selected["nonpositive_count"]),
        ("minimum_margin", -selected["minimum_margin"]),
        ("mean_margin", -selected["mean_margin"]),
    ):
        rows.append({
            "metric": metric,
            "selected_physical_configs": len(selected),
            "spearman_risk_vs_actual_ber": float(
                spearmanr(risk, selected["raw_ber"]).statistic
            ),
        })
    return pd.DataFrame(rows)


def replay_grouped_auc(frame: pd.DataFrame, features: tuple[str, ...]) -> float:
    groups = frame["replay_file"].astype(str).to_numpy()
    labels = frame["label"].astype(int).to_numpy()
    values = frame[list(features)].to_numpy(dtype=float)
    predictions = np.empty(len(frame), dtype=float)
    for train, test in GroupKFold(n_splits=5).split(values, labels, groups):
        model = RandomForestClassifier(
            n_estimators=300, max_features="sqrt", min_samples_leaf=2,
            random_state=42, n_jobs=1,
        )
        model.fit(values[train], labels[train])
        predictions[test] = model.predict_proba(values[test])[:, 1]
    return float(roc_auc_score(labels, predictions))


def detector_analysis(known: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lookup = features.set_index(["physical_id", "label"], drop=False)
    historical = pd.read_csv(RESULTS_DIR / "adaptive_layout_strong_v1" / "features.csv")
    historical = historical[
        (historical["partition"] == "development")
        & (historical["alpha_method"] == "sender_local_adaptive")
        & (historical["layout"] == "distributed")
        & ~historical["replay_file"].isin(set(known["replay_file"]))
    ]
    for k_value in K_VALUES:
        conditions = known[known["K"] == k_value]
        selected_rows = []
        for physical_id in conditions["physical_id"]:
            selected_rows.extend([
                lookup.loc[(physical_id, 0)], lookup.loc[(physical_id, 1)]
            ])
        frame = pd.DataFrame(selected_rows).reset_index(drop=True)
        for name, columns in (
            ("BASELINE-3", BASELINE_FEATURES),
            ("POSITION-only", POSITION_FEATURES),
            ("FULL-29", FULL_FEATURES),
        ):
            rows.append({
                "K": k_value, "feature_set": name,
                "attacker": "key_aware_replay_grouped_cv", "layout_seed": "pooled",
                "replay_grouped_auc": replay_grouped_auc(frame, tuple(columns)),
                "rows": len(frame), "replays": frame["replay_file"].nunique(),
            })
            for layout_seed in (0, 1, 2):
                test = frame[frame["layout_seed"].astype(int) == layout_seed]
                train = historical[
                    historical["layout_seed"].astype(str) == str(layout_seed)
                ]
                model = RandomForestClassifier(
                    n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                    random_state=42, n_jobs=1,
                )
                model.fit(train[list(columns)], train["label"])
                score = model.predict_proba(test[list(columns)])[:, 1]
                rows.append({
                    "K": k_value, "feature_set": name,
                    "attacker": "historical_pn_trained_replay_disjoint",
                    "layout_seed": layout_seed,
                    "replay_grouped_auc": float(roc_auc_score(test["label"], score)),
                    "rows": len(test), "replays": test["replay_file"].nunique(),
                })
    return pd.DataFrame(rows)


def oracle_design(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, phase_group in candidates.groupby("design_phase"):
        for k_value in K_VALUES:
            configs = phase_group[phase_group["candidate_order"] < k_value]
            selected_rows, oracle_rows, baseline_rows = [], [], []
            for _, group in configs.groupby(["replay_file", "layout_seed"]):
                selected_rows.append(group.sort_values(
                    ["nonpositive_count", "minimum_margin", "mean_margin", "pn_key_id"],
                    ascending=[True, False, False, True],
                ).iloc[0])
                oracle_rows.append(group.sort_values(
                    ["actual_ber", "pn_key_id"]
                ).iloc[0])
                baseline_rows.append(group.sort_values("candidate_order").iloc[0])
            summaries = {}
            for name, values in (
                ("baseline", pd.DataFrame(baseline_rows)),
                ("selector", pd.DataFrame(selected_rows)),
                ("oracle", pd.DataFrame(oracle_rows)),
            ):
                summaries[name] = values["actual_bit_errors"].sum() / values["coded_bits"].sum()
            rows.append({"design_phase": phase, "K": k_value, **summaries})
    return pd.DataFrame(rows)


def physical_by_k(known: pd.DataFrame, physical: pd.DataFrame) -> pd.DataFrame:
    diagnostics = [
        "active_carrier_fraction", "dropped_for_hit_window", "dropped_for_chronology",
        "new_unmatched_events", "new_matched_events", "changed_match_status_total",
        "sum_squared_shift_ms2", "rms_applied_shift_ms", "mean_absolute_applied_shift_ms",
    ]
    joined = known[["condition_id", "K", "physical_id"]].merge(
        physical[["physical_id", *diagnostics]], on="physical_id", validate="many_to_one"
    )
    return joined.groupby("K", as_index=False)[diagnostics].mean()


def side_information_cost(known: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k_value, group in known.groupby("K"):
        side_bits = int(np.ceil(np.log2(k_value)))
        useful_total = int(group["useful_bits"].sum())
        rows.append({
            "K": k_value,
            "side_information_bits_per_message": side_bits,
            "mean_useful_bits_per_message": float(group["useful_bits"].mean()),
            "total_side_information_bits": side_bits * len(group),
            "total_useful_bits": useful_total,
            "side_bits_as_fraction_of_useful_bits": (
                side_bits * len(group) / useful_total
            ),
            "index_physically_embedded_in_this_experiment": 0,
        })
    return pd.DataFrame(rows)


def selection_analysis(known: pd.DataFrame) -> pd.DataFrame:
    frequency = known.groupby(["K", "selected_pn_id"], as_index=False).size()
    frequency["fraction"] = frequency["size"] / frequency.groupby("K")["size"].transform("sum")
    return frequency


def selection_variation(known: pd.DataFrame) -> pd.DataFrame:
    rows = []
    definitions = (
        ("across_messages", ["replay_file", "layout_seed", "K"]),
        ("across_layouts", ["replay_file", "message_family", "K"]),
        ("across_replays", ["message_family", "layout_seed", "K"]),
    )
    for comparison, grouping in definitions:
        counts = known.groupby(grouping)["selected_pn_id"].nunique()
        for k_value in K_VALUES:
            values = counts.xs(k_value, level="K")
            rows.append({
                "comparison": comparison,
                "K": k_value,
                "groups": len(values),
                "fraction_with_multiple_selected_candidates": float((values > 1).mean()),
                "mean_distinct_selected_candidates": float(values.mean()),
            })
    return pd.DataFrame(rows)


def host_vs_message_awareness(output: Path) -> pd.DataFrame:
    """Compare frozen true-message ranking with an all-zero dummy ranking."""
    holdout = pd.read_csv(output / "holdout_selection.csv")
    pn_config = json.loads(
        (CONFIG_DIR / "host_aware_pn_candidates_v1.json").read_text(encoding="utf-8")
    )
    candidates = pn_config["candidates"]
    policy = load_sender_local_policy(CONFIG_DIR / "adaptive_alpha_sender_local_v2.json")
    beatmaps = build_beatmap_index(DATASET_DIR)
    offsets = load_map_offsets(CONFIG_DIR / "map_time_offsets.json")
    rows = []
    for source in holdout.itertuples(index=False):
        context = prepare_replay(source, DATASET_DIR, beatmaps, offsets)
        alpha = float(alpha_from_replay(policy, context.replay))
        dummy_useful = deterministic_message_family(
            "all_zero", int(source.useful_bits), context.replay_file, context.beatmap_hash
        )
        dummy_encoded = encode_hamming_8_4_secded(dummy_useful)
        for layout_seed in (0, 1, 2):
            layout_key = LAYOUT_KEYS[layout_seed]
            dummy_scores = {
                str(candidate["key_id"]): summarize_margins(predicted_margins(
                    context.original_residuals, dummy_encoded, str(candidate["key"]),
                    layout_key, alpha, 8,
                ))
                for candidate in candidates
            }
            for family in MESSAGE_FAMILIES:
                useful = deterministic_message_family(
                    family, int(source.useful_bits), context.replay_file, context.beatmap_hash
                )
                encoded = encode_hamming_8_4_secded(useful)
                actual_scores = {
                    str(candidate["key_id"]): summarize_margins(predicted_margins(
                        context.original_residuals, encoded, str(candidate["key"]),
                        layout_key, alpha, 8,
                    ))
                    for candidate in candidates
                }
                for k_value in K_VALUES:
                    ids = [str(candidate["key_id"]) for candidate in candidates[:k_value]]
                    actual = select_candidate([(value, actual_scores[value]) for value in ids])
                    dummy = select_candidate([(value, dummy_scores[value]) for value in ids])
                    rows.append({
                        "replay_file": context.replay_file,
                        "beatmap_hash": context.beatmap_hash,
                        "message_family": family,
                        "layout_seed": layout_seed,
                        "K": k_value,
                        "true_message_selected_pn_id": actual,
                        "dummy_message_selected_pn_id": dummy,
                        "same_selection": int(actual == dummy),
                    })
    return pd.DataFrame(rows)


def blind_summary(blind: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k_value, group in blind.groupby("K"):
        accepted = int(group["correct_accept"].sum() + group["wrong_accept"].sum())
        rows.append({
            "K": k_value, "configs": len(group),
            "correct_accept": int(group["correct_accept"].sum()),
            "wrong_accept": int(group["wrong_accept"].sum()),
            "reject": int(group["reject"].sum()),
            "ambiguous_candidate_reject": int(group["ambiguous_reject"].sum()),
            "zero_candidate_reject": int(group["zero_candidate_reject"].sum()),
            "mean_accepted_hypotheses": float(group["accepted_hypotheses"].mean()),
            "accepted_message_accuracy": group["correct_accept"].sum() / accepted if accepted else np.nan,
        })
    return pd.DataFrame(rows)


def wrong_hypothesis_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    wrong = diagnostics[diagnostics["is_selected_hypothesis"] == 0]
    return wrong.groupby("K", as_index=False).agg(
        wrong_hypotheses=("hypothesis_pn_id", "size"),
        wrong_pn_integrity_pass=("integrity_accepted", "sum"),
        wrong_pn_same_truth_message=("message_matches_truth", lambda x: int(((x == 1) & (wrong.loc[x.index, "integrity_accepted"] == 1)).sum())),
        wrong_pn_different_message=("message_matches_truth", lambda x: int(((x == 0) & (wrong.loc[x.index, "integrity_accepted"] == 1)).sum())),
    )


def write_report(
    reliability: pd.DataFrame,
    paired: pd.DataFrame,
    margin: pd.DataFrame,
    blind: pd.DataFrame,
    detector: pd.DataFrame,
    physical_count: int,
) -> None:
    r = reliability.set_index("K")
    b = blind.set_index("K")
    m = margin.iloc[0]
    clear = paired[(paired["delta_ber"] < 0) & (paired["ci_high"] < 0)]
    blind_gain = b.loc[8, "correct_accept"] > b.loc[1, "correct_accept"]
    auc = detector[
        (detector["feature_set"] == "FULL-29")
        & (detector["attacker"] == "key_aware_replay_grouped_cv")
    ].set_index("K")["replay_grouped_auc"]
    serious_auc = auc.loc[8] - auc.loc[1] > 0.03
    if len(clear) == 0:
        decision = "C. ANALYTICAL SELECTOR HAS WEAK / NO GENERALIZATION"
    elif serious_auc:
        decision = "E. SELECTION HARMS STEALTH / INTEGRITY"
    elif blind_gain:
        decision = "A. HOST-AWARE PN SELECTION PROMISING AND DEPLOYABLE"
    else:
        decision = "B. WORKS ONLY WITH PN INDEX SIDE INFORMATION"
    lines = [
        "# Host-aware PN selection v1", "",
        f"Final decision: **{decision}**", "",
        "The exact frozen sender score is `m_j = b_j*sum(clean_valid r_i*p_i) + alpha*n_clean_valid`. "
        "Margins at exactly zero count as non-positive; ranking minimizes their count, then maximizes minimum margin, mean margin, and finally uses PN-ID order.",
        "",
        f"The replay-disjoint internal holdout completed {physical_count} deduplicated physical round trips. "
        "It is not beatmap-disjoint external validation.", "",
        "## Index-known reliability", "",
        "| K | Raw BER | Correct accept | Wrong accept | Reject | Side bits |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k_value in K_VALUES:
        row = r.loc[k_value]
        lines.append(
            f"| {k_value} | {row.weighted_raw_ber:.4%} | {int(row.correct_accept)} | "
            f"{int(row.wrong_accept)} | {int(row.reject)} | {int(np.ceil(np.log2(k_value)))} |"
        )
    lines.extend(["", "## Mechanism and deployability", "",
        f"Heldout negative-margin error AUC was {m.negative_margin_error_auc:.3f}; median predicted margins were "
        f"{m.median_margin_correct:.1f} for correct bits and {m.median_margin_error:.1f} for errors.",
        f"Blind K=1/K=8 correct accepts were {int(b.loc[1,'correct_accept'])}/{int(b.loc[8,'correct_accept'])}; "
        f"K=8 ambiguous and zero-candidate rejections were {int(b.loc[8,'ambiguous_candidate_reject'])} and {int(b.loc[8,'zero_candidate_reject'])}.",
        f"Replay-grouped FULL-29 AUC was {auc.loc[1]:.3f} at K=1 and {auc.loc[8]:.3f} at K=8.",
        "", "Paired replay-cluster intervals and all message/layout breakdowns are in the accompanying CSV files. "
        "Oracle results are design-only upper bounds and were not used to modify the frozen selector.",
    ])
    (OUTPUT / "host_aware_pn_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output_dir
    known = pd.read_csv(OUTPUT / "index_known_results.csv")
    blind = pd.read_csv(OUTPUT / "blind_decoder_results.csv")
    blind_diag = pd.read_csv(OUTPUT / "blind_candidate_diagnostics.csv")
    physical = pd.read_csv(OUTPUT / "physical_results.csv")
    bits = pd.read_csv(OUTPUT / "bit_results.csv")
    features = pd.read_csv(OUTPUT / "timing_features.csv")
    candidates = pd.read_csv(OUTPUT / "selector_candidates.csv")
    heldout_scores = pd.read_csv(OUTPUT / "candidate_scores.csv")
    reliability = outcome_summary(known, ["K"])
    reliability.to_csv(OUTPUT / "reliability_by_k.csv", index=False)
    paired = replay_cluster_bootstrap(known)
    paired.to_csv(OUTPUT / "paired_reliability.csv", index=False)
    outcome_summary(known, ["message_family", "K"]).to_csv(
        OUTPUT / "reliability_by_message.csv", index=False
    )
    outcome_summary(known, ["layout_seed", "K"]).to_csv(
        OUTPUT / "reliability_by_layout.csv", index=False
    )
    selection_analysis(known).to_csv(OUTPUT / "selection_frequency.csv", index=False)
    selection_variation(known).to_csv(OUTPUT / "selection_variation.csv", index=False)
    host_vs_message_awareness(OUTPUT).to_csv(
        OUTPUT / "host_vs_message_awareness.csv", index=False
    )
    margin = margin_validation(bits)
    margin.to_csv(OUTPUT / "margin_validation.csv", index=False)
    margin_config_validation(heldout_scores, physical).to_csv(
        OUTPUT / "margin_config_validation.csv", index=False
    )
    blind_table = blind_summary(blind)
    blind_table.to_csv(OUTPUT / "blind_by_k.csv", index=False)
    wrong_hypothesis_summary(blind_diag).to_csv(
        OUTPUT / "wrong_candidate_false_acceptance.csv", index=False
    )
    detector = detector_analysis(known, features)
    detector.to_csv(OUTPUT / "detector_by_k.csv", index=False)
    physical_by_k(known, physical).to_csv(OUTPUT / "physical_by_k.csv", index=False)
    side_information_cost(known).to_csv(
        OUTPUT / "side_information_cost.csv", index=False
    )
    oracle_design(candidates).to_csv(OUTPUT / "oracle_design_upper_bound.csv", index=False)
    write_report(reliability, paired, margin, blind_table, detector, len(physical))
    print(reliability[["K", "weighted_raw_ber", "correct_accept", "wrong_accept", "reject"]].to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
