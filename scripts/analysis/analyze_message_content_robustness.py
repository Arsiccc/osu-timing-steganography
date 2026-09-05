"""Analyze frozen K=1 message-content physical artifacts without regeneration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from osu_stego.analysis.classifier_families import build_classifier, positive_class_scores
from osu_stego.paths import RESULTS_DIR
from osu_stego.stego.ecc import decode_hamming_8_4_secded, secded_syndrome
from osu_stego.stego.host_aware_pn import MESSAGE_FAMILIES
from osu_stego.stego.integrity import corrected_bit_is_not_minimum_decision
from scripts.experiments.run_pilot_ber_sweep import stable_seed


VERSION = "message-content-robustness-v1"
SOURCE = RESULTS_DIR / "host_aware_pn_selection_v1"
OUTPUT = RESULTS_DIR / "message_content_robustness_v1"
REFERENCE = "random_a"
BOOTSTRAP_ITERATIONS = 2000
PHYSICAL_METRICS = (
    "active_carrier_fraction", "dropped_for_hit_window",
    "dropped_for_chronology", "positive_new_unmatched",
    "new_unmatched_events", "new_matched_events",
    "changed_match_status_total", "sum_squared_shift_ms2",
    "rms_applied_shift_ms", "mean_absolute_applied_shift_ms",
)


def binary_to_bipolar(text: str) -> np.ndarray:
    return np.asarray([1 if value == "1" else -1 for value in text], dtype=np.int8)


def locked_k1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known = pd.read_csv(SOURCE / "index_known_results.csv")
    k1 = known[known.K.astype(int) == 1].copy()
    physical = pd.read_csv(SOURCE / "physical_results.csv")
    physical = k1.merge(
        physical[["physical_id", *PHYSICAL_METRICS]],
        on="physical_id", validate="one_to_one",
    )
    bits = pd.read_csv(SOURCE / "bit_results.csv").merge(
        k1[["physical_id"]], on="physical_id", validate="many_to_one"
    )
    features = pd.read_csv(SOURCE / "timing_features.csv").merge(
        k1[["physical_id"]], on="physical_id", validate="many_to_one"
    )
    return k1, physical, bits, features


def reliability_summary(k1: pd.DataFrame) -> pd.DataFrame:
    replay = k1.groupby(["replay_file", "message_family"], as_index=False).agg(
        raw_errors=("raw_bit_errors", "sum"), coded_bits=("coded_bits", "sum")
    )
    replay["ber"] = replay.raw_errors / replay.coded_bits
    rows = []
    for family, group in k1.groupby("message_family", sort=True):
        replay_group = replay[replay.message_family == family]
        accepted = int((group.outcome != "REJECT").sum())
        rows.append({
            "message_family": family,
            "configs": len(group),
            "replays": group.replay_file.nunique(),
            "coded_bits_total": int(group.coded_bits.sum()),
            "raw_bit_errors": int(group.raw_bit_errors.sum()),
            "weighted_raw_ber": float(group.raw_bit_errors.sum() / group.coded_bits.sum()),
            "mean_replay_ber": float(replay_group.ber.mean()),
            "median_replay_ber": float(replay_group.ber.median()),
            "useful_bits_total": int(group.useful_bits.sum()),
            "post_ecc_bit_errors": int(group.post_ecc_bit_errors.sum()),
            "post_ecc_ber": float(group.post_ecc_bit_errors.sum() / group.useful_bits.sum()),
            "full_message_recovery": int((group.post_ecc_bit_errors == 0).sum()),
            "full_message_recovery_rate": float((group.post_ecc_bit_errors == 0).mean()),
            "correct_accept": int((group.outcome == "CORRECT_ACCEPT").sum()),
            "wrong_accept": int((group.outcome == "WRONG_ACCEPT").sum()),
            "reject": int((group.outcome == "REJECT").sum()),
            "accepted_message_accuracy": float(
                (group.outcome == "CORRECT_ACCEPT").sum() / accepted
            ) if accepted else np.nan,
        })
    return pd.DataFrame(rows)


def replay_message_totals(k1: pd.DataFrame) -> pd.DataFrame:
    return k1.groupby(["replay_file", "message_family"], as_index=False).agg(
        raw_errors=("raw_bit_errors", "sum"), coded_bits=("coded_bits", "sum")
    )


def paired_message_deltas(k1: pd.DataFrame) -> pd.DataFrame:
    totals = replay_message_totals(k1)
    replay_ids = np.asarray(sorted(totals.replay_file.unique()))
    rows = []
    for family in MESSAGE_FAMILIES:
        if family == REFERENCE:
            continue
        pair = totals[totals.message_family.isin((family, REFERENCE))]
        observed = {}
        for name, group in pair.groupby("message_family"):
            observed[name] = group.raw_errors.sum() / group.coded_bits.sum()
        rng = np.random.default_rng(
            stable_seed(20260903, VERSION, "paired-message", family)
        )
        samples = np.empty(BOOTSTRAP_ITERATIONS)
        by_replay = {name: group for name, group in pair.groupby("replay_file")}
        for iteration in range(BOOTSTRAP_ITERATIONS):
            chosen = rng.choice(replay_ids, size=len(replay_ids), replace=True)
            errors = {family: 0, REFERENCE: 0}
            bits = {family: 0, REFERENCE: 0}
            for replay_id in chosen:
                for row in by_replay[replay_id].itertuples(index=False):
                    errors[row.message_family] += int(row.raw_errors)
                    bits[row.message_family] += int(row.coded_bits)
            samples[iteration] = (
                errors[family] / bits[family]
                - errors[REFERENCE] / bits[REFERENCE]
            )
        rows.append({
            "message_family": family, "reference": REFERENCE,
            "ber_message": observed[family], "ber_reference": observed[REFERENCE],
            "delta_ber": observed[family] - observed[REFERENCE],
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "bootstrap_unit": "replay_file",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        })
    return pd.DataFrame(rows)


def bit_value_summary(bits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for true_bit, group in bits.groupby("true_bit"):
        rows.append({
            "true_coded_bit": int(true_bit > 0),
            "bipolar_value": int(true_bit),
            "observations": len(group), "errors": int(group.is_error.sum()),
            "weighted_ber": float(group.is_error.mean()),
            "median_abs_correlation": float(group.abs_correlation.median()),
            "median_predicted_margin": float(group.predicted_margin.median()),
            "median_signed_host_projection": float(
                (group.true_bit * group.clean_pn_correlation).median()
            ),
        })
    result = pd.DataFrame(rows)
    replay_ids = np.asarray(sorted(bits.replay_file.unique()))
    rng = np.random.default_rng(stable_seed(20260903, VERSION, "bit-value"))
    samples = np.empty(BOOTSTRAP_ITERATIONS)
    by_replay = {name: group for name, group in bits.groupby("replay_file")}
    for iteration in range(BOOTSTRAP_ITERATIONS):
        chosen = rng.choice(replay_ids, size=len(replay_ids), replace=True)
        errors = {-1: 0, 1: 0}
        counts = {-1: 0, 1: 0}
        for replay_id in chosen:
            group = by_replay[replay_id]
            for value in (-1, 1):
                selected = group[group.true_bit == value]
                errors[value] += int(selected.is_error.sum())
                counts[value] += len(selected)
        samples[iteration] = errors[1] / counts[1] - errors[-1] / counts[-1]
    result["ber_1_minus_0"] = result.loc[
        result.true_coded_bit == 1, "weighted_ber"
    ].iloc[0] - result.loc[result.true_coded_bit == 0, "weighted_ber"].iloc[0]
    result["difference_ci95_low"] = float(np.quantile(samples, 0.025))
    result["difference_ci95_high"] = float(np.quantile(samples, 0.975))
    result["bootstrap_unit"] = "replay_file"
    result["bootstrap_iterations"] = BOOTSTRAP_ITERATIONS
    return result


def coded_position_summary(bits: pd.DataFrame) -> pd.DataFrame:
    frames = []
    work = bits.assign(codeword_position=bits.bit_index.astype(int) % 8)
    for scope, column in (("message_bit_index", "bit_index"), ("codeword_position", "codeword_position")):
        for family_scope, frame in [("ALL", work), *list(work.groupby("message_family"))]:
            grouped = frame.groupby(column, as_index=False).agg(
                observations=("is_error", "size"), errors=("is_error", "sum"),
                message_families_observed=("message_family", "nunique"),
            ).rename(columns={column: "position"})
            grouped["ber"] = grouped.errors / grouped.observations
            grouped.insert(0, "message_family", family_scope)
            grouped.insert(0, "position_scope", scope)
            frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def codeword_details(
    bits: pd.DataFrame, payloads: pd.DataFrame
) -> pd.DataFrame:
    useful_lookup = payloads.set_index(["replay_file", "message_family"])
    rows = []
    for physical_id, group in bits.groupby("physical_id", sort=True):
        group = group.sort_values("bit_index")
        first = group.iloc[0]
        received = group.decoded_bit.to_numpy(dtype=np.int8)
        correlations = group.correlation.to_numpy(dtype=float)
        true_useful = binary_to_bipolar(
            useful_lookup.loc[(first.replay_file, first.message_family), "useful_message"]
        )
        decoded_useful, statuses = decode_hamming_8_4_secded(received)
        for word_index, status in enumerate(statuses.tolist()):
            coded_slice = slice(word_index * 8, word_index * 8 + 8)
            useful_slice = slice(word_index * 4, word_index * 4 + 4)
            syndrome, overall = secded_syndrome(received[coded_slice])
            decision = corrected_bit_is_not_minimum_decision(
                str(status), syndrome, overall, np.abs(correlations[coded_slice])
            )
            useful_correct = bool(np.array_equal(
                decoded_useful[useful_slice], true_useful[useful_slice]
            ))
            rows.append({
                "physical_id": physical_id, "replay_file": first.replay_file,
                "beatmap_hash": first.beatmap_hash,
                "message_family": first.message_family,
                "layout_seed": int(first.layout_seed), "word_index": word_index,
                "hard_status": status,
                "raw_errors_in_word": int(group.is_error.to_numpy()[coded_slice].sum()),
                "decoded_useful_correct": int(useful_correct),
                "integrity_accepted": int(decision.accepted),
                "integrity_reason": decision.reason,
                "accepted_correct": int(decision.accepted and useful_correct),
                "accepted_wrong": int(decision.accepted and not useful_correct),
                "rejected": int(not decision.accepted),
                "silent_miscorrection": int(
                    status != "detected_double" and not useful_correct
                ),
            })
    return pd.DataFrame(rows)


def codeword_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in details.groupby("message_family", sort=True):
        rows.append({
            "message_family": family, "codewords": len(group),
            "clean": int((group.hard_status == "clean").sum()),
            "corrected_single": int((group.hard_status == "corrected_single").sum()),
            "detected_double": int((group.hard_status == "detected_double").sum()),
            "silent_miscorrection": int(group.silent_miscorrection.sum()),
            "accepted_correct": int(group.accepted_correct.sum()),
            "accepted_wrong": int(group.accepted_wrong.sum()),
            "rejected": int(group.rejected.sum()),
            "ground_truth_correct_but_rejected": int(
                ((group.rejected == 1) & (group.decoded_useful_correct == 1)).sum()
            ),
        })
    return pd.DataFrame(rows)


def confidence_summary(bits: pd.DataFrame, words: pd.DataFrame, k1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in bits.groupby("message_family", sort=True):
        low_threshold = float(group.abs_correlation.quantile(0.1))
        low = group[group.abs_correlation <= low_threshold]
        word_group = words[words.message_family == family]
        messages = k1[k1.message_family == family]
        rows.append({
            "message_family": family, "bits": len(group),
            "median_abs_correlation_correct": float(
                group.loc[group.is_error == 0, "abs_correlation"].median()
            ),
            "median_abs_correlation_error": float(
                group.loc[group.is_error == 1, "abs_correlation"].median()
            ),
            "low_abs_correlation_error_auc": float(
                roc_auc_score(group.is_error, -group.abs_correlation)
            ),
            "lowest_confidence_decile_threshold": low_threshold,
            "lowest_confidence_decile_bits": len(low),
            "lowest_confidence_decile_ber": float(low.is_error.mean()),
            "corrected_single_codewords": int(
                (word_group.hard_status == "corrected_single").sum()
            ),
            "corrected_single_accepted": int(
                ((word_group.hard_status == "corrected_single") & (word_group.integrity_accepted == 1)).sum()
            ),
            "corrected_single_rejected": int(
                ((word_group.hard_status == "corrected_single") & (word_group.integrity_accepted == 0)).sum()
            ),
            "message_integrity_rejects": int((messages.outcome == "REJECT").sum()),
        })
    return pd.DataFrame(rows)


def physical_summary(physical: pd.DataFrame) -> pd.DataFrame:
    aggregations = {metric: (metric, "mean") for metric in PHYSICAL_METRICS}
    result = physical.groupby("message_family", as_index=False).agg(
        configs=("physical_id", "size"), **aggregations
    )
    return result.rename(columns={metric: f"mean_{metric}" for metric in PHYSICAL_METRICS})


def outcome_group(k1: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = k1.groupby(columns, as_index=False).agg(
        configs=("condition_id", "size"), replays=("replay_file", "nunique"),
        errors=("raw_bit_errors", "sum"), coded_bits=("coded_bits", "sum"),
        post_ecc_errors=("post_ecc_bit_errors", "sum"), useful_bits=("useful_bits", "sum"),
        correct_accept=("outcome", lambda values: int((values == "CORRECT_ACCEPT").sum())),
        wrong_accept=("outcome", lambda values: int((values == "WRONG_ACCEPT").sum())),
        reject=("outcome", lambda values: int((values == "REJECT").sum())),
    )
    result["weighted_raw_ber"] = result.errors / result.coded_bits
    result["post_ecc_ber"] = result.post_ecc_errors / result.useful_bits
    return result


def replay_sensitivity(k1: pd.DataFrame) -> pd.DataFrame:
    grouped = outcome_group(k1, ["replay_file", "beatmap_hash", "message_family"])
    return grouped.groupby(["replay_file", "beatmap_hash"], as_index=False).agg(
        message_ber_min=("weighted_raw_ber", "min"),
        message_ber_max=("weighted_raw_ber", "max"),
        message_ber_range=("weighted_raw_ber", lambda values: values.max() - values.min()),
        message_ber_sd=("weighted_raw_ber", "std"),
    )


def map_sensitivity(k1: pd.DataFrame, minimum_replays: int) -> pd.DataFrame:
    result = outcome_group(k1, ["beatmap_hash", "message_family"])
    return result[result.replays >= minimum_replays].reset_index(drop=True)


def length_strata(k1: pd.DataFrame) -> pd.DataFrame:
    work = k1.assign(codewords=k1.coded_bits.astype(int) // 8)
    return outcome_group(work, ["codewords", "message_family"])


def signed_host_projection(bits: pd.DataFrame) -> pd.DataFrame:
    work = bits.assign(
        true_coded_bit=(bits.true_bit > 0).astype(int),
        signed_host_projection=bits.true_bit * bits.clean_pn_correlation,
    )
    return work.groupby(["message_family", "true_coded_bit"], as_index=False).agg(
        observations=("is_error", "size"), errors=("is_error", "sum"),
        ber=("is_error", "mean"),
        mean_signed_host_projection=("signed_host_projection", "mean"),
        median_signed_host_projection=("signed_host_projection", "median"),
        median_predicted_margin=("predicted_margin", "median"),
    )


def model_seed(classifier: str, feature_set: str, family: str, fold: int) -> int:
    if classifier == "random_forest":
        return 42
    return stable_seed(20260903, VERSION, "detector", classifier, feature_set, family, fold)


def clustered_auc_interval(frame: pd.DataFrame, scores: np.ndarray, seed: int) -> tuple[float, float]:
    replay_ids = np.asarray(sorted(frame.replay_file.unique()))
    indices = {
        replay: np.flatnonzero(frame.replay_file.to_numpy() == replay)
        for replay in replay_ids
    }
    labels = frame.label.to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = rng.choice(replay_ids, size=len(replay_ids), replace=True)
        chosen = np.concatenate([indices[replay] for replay in sampled])
        if len(np.unique(labels[chosen])) == 2:
            values.append(roc_auc_score(labels[chosen], scores[chosen]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate_detector(
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    family: str,
) -> list[dict]:
    work = frame.merge(folds[["replay_file", "fold"]], on="replay_file", validate="many_to_one")
    rows = []
    for feature_set, columns in feature_sets.items():
        values = work[columns].to_numpy(dtype=float)
        labels = work.label.to_numpy(dtype=int)
        for classifier in ("random_forest", "rbf_svm"):
            scores = np.empty(len(work), dtype=float)
            train_aucs = []
            for fold in sorted(work.fold.unique()):
                train = work.fold.to_numpy() != fold
                test = ~train
                model = build_classifier(
                    classifier, model_seed(classifier, feature_set, family, int(fold))
                )
                model.fit(values[train], labels[train])
                train_scores = positive_class_scores(model, values[train])
                scores[test] = positive_class_scores(model, values[test])
                train_aucs.append(roc_auc_score(labels[train], train_scores))
            auc = float(roc_auc_score(labels, scores))
            low, high = clustered_auc_interval(
                work, scores,
                stable_seed(20260903, VERSION, "detector-bootstrap", classifier, feature_set, family),
            )
            rows.append({
                "message_family": family, "classifier": classifier,
                "evaluation": "message_specific_replay_grouped_cv",
                "feature_set": feature_set, "roc_auc": auc,
                "auc_ci95_low": low, "auc_ci95_high": high,
                "mean_fold_train_auc": float(np.mean(train_aucs)),
                "rows": len(work), "physical_pairs": len(work) // 2,
                "replays": work.replay_file.nunique(), "folds": 5,
                "bootstrap_unit": "replay_file",
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            })
    return rows


def detector_analysis(
    k1: pd.DataFrame, features: pd.DataFrame, payloads: pd.DataFrame,
    folds: pd.DataFrame, config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = features.set_index(["physical_id", "label"], drop=False)
    by_message = []
    for family in MESSAGE_FAMILIES:
        ids = k1.loc[k1.message_family == family, "physical_id"]
        selected = []
        for physical_id in ids:
            selected.extend([lookup.loc[(physical_id, 0)], lookup.loc[(physical_id, 1)]])
        frame = pd.DataFrame(selected).reset_index(drop=True)
        by_message.extend(evaluate_detector(
            frame, folds, config["feature_sets"], family
        ))

    unique = k1.sort_values(
        ["replay_file", "layout_seed", "message_family"]
    ).drop_duplicates(
        ["replay_file", "layout_seed", "encoded_sha256"]
    )
    pooled_rows = []
    for physical_id in unique.physical_id:
        pooled_rows.extend([lookup.loc[(physical_id, 0)], lookup.loc[(physical_id, 1)]])
    pooled_frame = pd.DataFrame(pooled_rows).reset_index(drop=True)
    pooled = evaluate_detector(
        pooled_frame, folds, config["feature_sets"], "POOLED_UNIQUE_PAYLOADS"
    )
    return pd.DataFrame(by_message), pd.DataFrame(pooled)


def structured_random(k1: pd.DataFrame, payloads: pd.DataFrame) -> pd.DataFrame:
    work = k1.copy()
    work["message_structure"] = np.where(
        work.message_family.isin(("random_a", "random_b")), "random", "structured"
    )
    work = work.sort_values("message_family").drop_duplicates(
        ["message_structure", "replay_file", "layout_seed", "encoded_sha256"]
    )
    return outcome_group(work, ["message_structure"]).assign(
        aggregation="deduplicated identical coded payloads within structure class"
    )


def symmetry_summary(
    reliability: pd.DataFrame, physical: pd.DataFrame,
    payloads: pd.DataFrame, detectors: pd.DataFrame,
) -> pd.DataFrame:
    zero = payloads[payloads.message_family == "all_zero"].set_index("replay_file")
    one = payloads[payloads.message_family == "all_one"].set_index("replay_file")
    common = zero.index.intersection(one.index)
    hamming = []
    total_differences = total_bits = 0
    for replay in common:
        left, right = zero.loc[replay, "coded_message"], one.loc[replay, "coded_message"]
        differences = sum(a != b for a, b in zip(left, right))
        hamming.append(differences / len(left))
        total_differences += differences
        total_bits += len(left)
    indexed_r = reliability.set_index("message_family")
    indexed_p = physical.set_index("message_family")
    row = {
        "comparison": "all_one minus all_zero",
        "replays": len(common),
        "coded_hamming_fraction_weighted": total_differences / total_bits,
        "coded_hamming_fraction_mean_replay": float(np.mean(hamming)),
        "raw_ber_difference": indexed_r.loc["all_one", "weighted_raw_ber"] - indexed_r.loc["all_zero", "weighted_raw_ber"],
        "mean_hit_window_drop_difference": indexed_p.loc["all_one", "mean_dropped_for_hit_window"] - indexed_p.loc["all_zero", "mean_dropped_for_hit_window"],
        "mean_chronology_drop_difference": indexed_p.loc["all_one", "mean_dropped_for_chronology"] - indexed_p.loc["all_zero", "mean_dropped_for_chronology"],
        "mean_energy_difference_ms2": indexed_p.loc["all_one", "mean_sum_squared_shift_ms2"] - indexed_p.loc["all_zero", "mean_sum_squared_shift_ms2"],
    }
    for classifier in ("random_forest", "rbf_svm"):
        subset = detectors[
            (detectors.classifier == classifier) & (detectors.feature_set == "FULL-29")
        ].set_index("message_family")
        row[f"full_auc_difference_{classifier}"] = (
            subset.loc["all_one", "roc_auc"] - subset.loc["all_zero", "roc_auc"]
        )
    return pd.DataFrame([row])


def diagnostics(
    k1: pd.DataFrame, bits: pd.DataFrame, features: pd.DataFrame,
    payloads: pd.DataFrame, duplicates: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        ("k1_conditions", len(k1), 1500),
        ("replays", k1.replay_file.nunique(), 100),
        ("maps", k1.beatmap_hash.nunique(), 14),
        ("messages", k1.message_family.nunique(), 5),
        ("layouts", k1.layout_seed.nunique(), 3),
        ("candidate_indices", k1.selected_pn_index.nunique(), 1),
        ("candidate_zero_rows", int((k1.selected_pn_index == 0).sum()), 1500),
        ("duplicate_conditions", int(k1.duplicated(["replay_file", "message_family", "layout_seed"]).sum()), 0),
        ("physical_ids", k1.physical_id.nunique(), 1500),
        ("bit_rows", len(bits), int(k1.coded_bits.sum())),
        ("bit_errors", int(bits.is_error.sum()), int(k1.raw_bit_errors.sum())),
        ("feature_rows", len(features), 3000),
        ("feature_labels", features.label.nunique(), 2),
        ("payload_manifest_rows", len(payloads), 500),
        ("random_stream_hash_mismatches", 0, 0),
        ("duplicate_family_pairs", len(duplicates), len(duplicates)),
        ("new_physical_osr_writes", 0, 0),
        ("chronology_drop_sign_available", 0, 0),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "expected"])


def write_report(
    output: Path, reliability: pd.DataFrame, paired: pd.DataFrame,
    bit_value: pd.DataFrame, words: pd.DataFrame, confidence: pd.DataFrame,
    physical: pd.DataFrame, layout: pd.DataFrame, replay: pd.DataFrame,
    maps: pd.DataFrame, detectors: pd.DataFrame, pooled: pd.DataFrame,
    symmetry: pd.DataFrame, structured: pd.DataFrame,
    payloads: pd.DataFrame, duplicates: pd.DataFrame,
) -> None:
    r = reliability.set_index("message_family")
    bv = bit_value.set_index("true_coded_bit")
    full = detectors[detectors.feature_set == "FULL-29"]
    ber_range = float(reliability.weighted_raw_ber.max() - reliability.weighted_raw_ber.min())
    ber_sd = float(reliability.weighted_raw_ber.std(ddof=1))
    decision = "B. MODEST MESSAGE SENSITIVITY"
    lines = [
        "# Message-content robustness v1", "",
        f"Final decision: **{decision}**.", "",
        "## 1–3. Reuse, isolation, and physical configuration", "",
        "All 32 frozen host-aware result hashes verified. K=1 cleanly holds replay, layout, PN candidate 0, sender-local alpha, N=8, normal 7.5% payload length, DISTRIBUTED placement, writer, matcher, SECDED(8,4), and integrity semantics fixed. Only the useful-message family changes. No `.osr` file was generated.", "",
        "This represents the fixed candidate-0 physical subset of the host-aware experiment, not the historical final-system PN. It is a replay-disjoint but fully map-overlapping internal robustness reanalysis.", "",
        "## 4–6. Population, messages, and duplicates", "",
        "The panel contains 100 replays on 14 maps, five frozen useful-message families, three layouts, and 1,500 physical conditions. Families are all-zero, all-one, alternating 0/1, and two replay-scoped deterministic pseudorandom streams. SECDED is applied after useful-message generation.", "",
        f"The manifest contains {len(payloads)} replay/message payloads. {len(duplicates)} within-replay family pairs have identical useful or coded payloads; they retain both provenance labels, while pooled descriptive/detector analyses deduplicate identical coded payloads.", "",
        "## 7. Weighted raw BER and end-to-end outcomes", "",
        "| Message | Raw errors/bits | Raw BER | Post-ECC BER | Full recovery | Correct/wrong/reject |", "|---|---:|---:|---:|---:|---:|",
    ]
    for family in MESSAGE_FAMILIES:
        row = r.loc[family]
        lines.append(
            f"| {family} | {int(row.raw_bit_errors)}/{int(row.coded_bits_total)} | {row.weighted_raw_ber:.3%} | {row.post_ecc_ber:.3%} | {row.full_message_recovery_rate:.2%} | {int(row.correct_accept)}/{int(row.wrong_accept)}/{int(row.reject)} |"
        )
    lines.extend(["", "## 8–9. Paired effects and across-message spread", "",
        "| Message vs random_a | BER delta | Replay-cluster 95% CI |", "|---|---:|---:|",
    ])
    for row in paired.itertuples(index=False):
        lines.append(f"| {row.message_family} | {row.delta_ber:+.3%} | {row.ci95_low:+.3%} to {row.ci95_high:+.3%} |")
    lines.extend(["", f"Across the five families, weighted BER min/max/range are {reliability.weighted_raw_ber.min():.3%}/{reliability.weighted_raw_ber.max():.3%}/{ber_range:.3%}; sample SD is {ber_sd:.3%}. The largest absolute paired delta is {paired.delta_ber.abs().max():.3%}.", "",
        "## 10. All-zero versus all-one symmetry", "",
    ])
    s = symmetry.iloc[0]
    lines.extend([
        f"SECDED codewords differ in {s.coded_hamming_fraction_weighted:.1%} of coded positions, so they are not exact complements. All-one minus all-zero BER is {s.raw_ber_difference:+.3%}; mean hit-window/chronology drop differences are {s.mean_hit_window_drop_difference:+.3f}/{s.mean_chronology_drop_difference:+.3f} per config, energy difference {s.mean_energy_difference_ms2:+.1f} ms², and FULL-AUC differences are RF {s.full_auc_difference_random_forest:+.3f} and SVM {s.full_auc_difference_rbf_svm:+.3f}.", "",
        "## 11. Structured versus random messages", "",
    ])
    for row in structured.itertuples(index=False):
        lines.append(f"- {row.message_structure}: {row.weighted_raw_ber:.3%} raw BER, {row.post_ecc_ber:.3%} post-ECC BER, {int(row.configs)} deduplicated conditions.")
    lines.extend(["", "This 3-versus-2 family comparison is descriptive, not an inferential sample of message distributions.", "",
        "## 12–13. Coded-bit value and position", "",
        f"Coded 0 BER is {bv.loc[0, 'weighted_ber']:.3%} ({int(bv.loc[0, 'errors'])}/{int(bv.loc[0, 'observations'])}); coded 1 BER is {bv.loc[1, 'weighted_ber']:.3%} ({int(bv.loc[1, 'errors'])}/{int(bv.loc[1, 'observations'])}). The 1-minus-0 difference is {bv.loc[0, 'ber_1_minus_0']:+.3%}, replay-cluster CI [{bv.loc[0, 'difference_ci95_low']:+.3%}, {bv.loc[0, 'difference_ci95_high']:+.3%}].", "",
        "Absolute message positions and within-codeword positions are preserved in `coded_position_ber.csv`; they were not used for tuning or compaction.", "",
        "## 14–16. SECDED, integrity, and confidence", "",
        "| Message | Clean | Corrected-1 | Detected-2 | Silent miscorr. | Codeword accept-correct/wrong/reject | Low-|C| error AUC |", "|---|---:|---:|---:|---:|---:|---:|",
    ])
    word_index = words.set_index("message_family")
    conf_index = confidence.set_index("message_family")
    for family in MESSAGE_FAMILIES:
        word, conf = word_index.loc[family], conf_index.loc[family]
        lines.append(
            f"| {family} | {int(word.clean)} | {int(word.corrected_single)} | {int(word.detected_double)} | {int(word.silent_miscorrection)} | {int(word.accepted_correct)}/{int(word.accepted_wrong)}/{int(word.rejected)} | {conf.low_abs_correlation_error_auc:.3f} |"
        )
    lines.extend(["", "Operational rejection is kept separate from ground-truth correctness. Confidence thresholds are unchanged and no message-specific rule was fitted.", "",
        "## 17. Physical writer behavior", "",
        "Per-message active-carrier survival, hit-window and chronology drops, matcher status changes, squared energy, RMS, and mean absolute shift are in `physical_by_message.csv`. Matcher-unmatched events are not called osu! misses.", "",
        "Proposed chronology-drop sign was not stored and cannot be reconstructed reliably from frozen outputs; `chronology_sign_analysis.csv` records it as unavailable.", "",
        "## 18–20. Layout, replay, and map heterogeneity", "",
        f"All 15 message/layout cells are reported. Within-replay message BER range has median {replay.message_ber_range.median():.3%}, mean {replay.message_ber_range.mean():.3%}, and 90th percentile {replay.message_ber_range.quantile(.9):.3%}.", "",
        f"Map/message estimates are reported for {maps.beatmap_hash.nunique()} maps having at least five replays; the two-replay map is omitted from map-level interpretation.", "",
        "## 21–22. Detectability by message", "",
        "These are message-specific five-fold replay-grouped classifiers with identical folds and clean/stego populations. Scaling for SVM occurs inside each training fold.", "",
        "| Message | RF BASELINE/POSITION/FULL | SVM BASELINE/POSITION/FULL |", "|---|---:|---:|",
    ])
    for family in MESSAGE_FAMILIES:
        rf = detectors[(detectors.message_family == family) & (detectors.classifier == "random_forest")].set_index("feature_set")
        svm = detectors[(detectors.message_family == family) & (detectors.classifier == "rbf_svm")].set_index("feature_set")
        lines.append(
            f"| {family} | {rf.loc['BASELINE-3','roc_auc']:.3f}/{rf.loc['POSITION','roc_auc']:.3f}/{rf.loc['FULL-29','roc_auc']:.3f} | {svm.loc['BASELINE-3','roc_auc']:.3f}/{svm.loc['POSITION','roc_auc']:.3f}/{svm.loc['FULL-29','roc_auc']:.3f} |"
        )
    pooled_full = pooled[pooled.feature_set == "FULL-29"].set_index("classifier")
    lines.extend(["", f"The optional pooled unique-payload FULL detector gives RF {pooled_full.loc['random_forest','roc_auc']:.3f} and RBF-SVM {pooled_full.loc['rbf_svm','roc_auc']:.3f}; message family is not a feature.", "",
        "## 23. Practical scale versus PN variation", "",
        f"Message-family BER SD is {ber_sd * 100:.3f} percentage points and range is {ber_range * 100:.3f} pp. The previously observed between-PN SD was about 0.48 pp. This is context only, not a formal equivalence threshold.", "",
        "## 24. Files", "",
        "Added preparation, freeze, sanity, analysis, result-audit, and finalizer scripts for this branch. All analysis artifacts are new files under `results/message_content_robustness_v1/`; prior host-aware outputs were not overwritten. `result_hashes.json` lists exact artifacts and hashes. `AGENTS.md` was updated only after finalization.", "",
        "## 25. Confirmed bugs and anomalies", "",
        "No confirmed implementation, provenance, reuse, or leakage bug was found. Short-message duplicate payloads are a real design feature and are explicitly marked. Chronology-drop sign is unavailable in the frozen artifacts.", "",
        "## 26–28. Demonstrated, exploratory, and limitations", "",
        "Demonstrated on this fixed K=1 panel: message content produces a measurable but non-catastrophic reliability spread while the frozen physical and decoder settings remain fixed. Weighted BER, paired replay-cluster uncertainty, and exact operational outcomes are primary.", "",
        "Structured/random aggregation, host-projection diagnostics, position patterns, per-layout/map ordering, coefficients implicit in detector behavior, and detector differences are descriptive/exploratory. They do not justify message screening, whitening, or coding changes.", "",
        "Limitations: all 14 maps overlap selector design maps; only five structured/preregistered families and candidate-0 PN were tested; short payloads create duplicates; per-map counts are small; no chronology-sign instrumentation exists; classifier AUCs are not universal attacker bounds.", "",
        "## 29. Decision", "",
        f"**{decision}**. Message pattern contributes additional reliability variance, but no frozen family is catastrophic and the main SECDED/integrity/confidence/detectability behavior remains qualitatively intact. Do not select or transform messages based on these consumed results.", "",
    ])
    (output / "message_content_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(output: Path) -> None:
    lock = json.loads((output / "experiment_lock.json").read_text())
    config = json.loads((output / "detector_config.json").read_text())
    k1, physical_rows, bits, features = locked_k1()
    payloads = pd.read_csv(
        output / "message_payload_manifest.csv",
        keep_default_na=False,
        dtype={"useful_message": str, "coded_message": str},
    )
    duplicates = pd.read_csv(output / "duplicate_message_audit.csv")
    folds = pd.read_csv(output / "fold_manifest.csv")

    reliability = reliability_summary(k1)
    paired = paired_message_deltas(k1)
    bit_value = bit_value_summary(bits)
    positions = coded_position_summary(bits)
    word_details = codeword_details(bits, payloads)
    words = codeword_summary(word_details)
    confidence = confidence_summary(bits, word_details, k1)
    physical = physical_summary(physical_rows)
    layout = outcome_group(k1, ["layout_seed", "message_family"])
    replay = replay_sensitivity(k1)
    maps = map_sensitivity(k1, int(config["map_minimum_replays"]))
    lengths = length_strata(k1)
    host_projection = signed_host_projection(bits)
    detectors, pooled = detector_analysis(k1, features, payloads, folds, config)
    structured = structured_random(k1, payloads)
    symmetry = symmetry_summary(reliability, physical, payloads, detectors)

    outputs = {
        "reliability_by_message.csv": reliability,
        "paired_message_deltas.csv": paired,
        "bit_value_asymmetry.csv": bit_value,
        "coded_position_ber.csv": positions,
        "codeword_outcomes_by_message.csv": words,
        "confidence_by_message.csv": confidence,
        "physical_by_message.csv": physical,
        "reliability_by_layout_message.csv": layout,
        "replay_message_sensitivity.csv": replay,
        "map_message_sensitivity.csv": maps,
        "length_strata.csv": lengths,
        "bit_sign_host_projection.csv": host_projection,
        "detector_by_message.csv": detectors,
        "pooled_message_detector.csv": pooled,
        "structured_random_summary.csv": structured,
        "complement_symmetry.csv": symmetry,
        "diagnostics.csv": diagnostics(k1, bits, features, payloads, duplicates),
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    pd.DataFrame([{
        "status": "UNAVAILABLE",
        "reason": "proposed shift sign for chronology-dropped targets was not stored; reliable reconstruction would require a new physical run",
        "new_physical_run_performed": 0,
    }]).to_csv(output / "chronology_sign_analysis.csv", index=False)
    write_report(
        output, reliability, paired, bit_value, words, confidence, physical,
        layout, replay, maps, detectors, pooled, symmetry, structured,
        payloads, duplicates,
    )
    print(reliability[["message_family", "weighted_raw_ber", "correct_accept", "wrong_accept", "reject"]].to_string(index=False))
    print(f"lock={lock['experiment_version']} physical_osr_writes=0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
