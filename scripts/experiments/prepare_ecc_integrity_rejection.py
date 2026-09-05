"""Analyze design SECDED failures, choose one rejection rule, and freeze it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import CONFIG_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.integrity import (
    IntegrityDecision,
    baseline_integrity_decision,
    corrected_bit_is_not_minimum_decision,
    corrected_position,
    message_integrity_decision,
    multiple_low_bits_decision,
)
from scripts.experiments.adaptive_layout_strong_common import write_or_validate_config
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_ecc_pilot import DEFAULT_PARTITION
from scripts.experiments.run_payload_sweep import (
    message_length_for_fraction,
    nominal_capacity_bits,
)
from scripts.experiments.sender_local_common import load_partitioned_cohort


EXPERIMENT_VERSION = "ecc-integrity-rejection-v1"
OUTPUT_DIR = RESULTS_DIR / "ecc_integrity_rejection_v1"
DESIGN_DIR = RESULTS_DIR / "ecc_equal_payload_v1"
CONFIDENCE_DIR = RESULTS_DIR / "confidence_analysis_v1"
DESIGN_SELECTION = RESULTS_DIR / "ecc_pilot_v1" / "pilot_selection.csv"
FROZEN_CONFIG = CONFIG_DIR / "ecc_integrity_rejection_v1.json"
POLICY = CONFIG_DIR / "adaptive_alpha_sender_local_v2.json"
N_VALUE = 8
PAYLOAD_FRACTION = 0.075
SELECTION_SEED = 20260902


def parse_array(value: str, dtype: type = float) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=dtype)


def frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def deterministic_order(replay_file: str) -> str:
    return hashlib.sha256(
        f"{EXPERIMENT_VERSION}|holdout|{SELECTION_SEED}|{replay_file}".encode("utf-8")
    ).hexdigest()


def decision_for_rule(row: pd.Series, rule: dict) -> IntegrityDecision:
    confidence = parse_array(row.abs_correlations)
    if rule["kind"] == "baseline_status":
        return baseline_integrity_decision(str(row.hard_status))
    if rule["kind"] == "multiple_low_bits":
        return multiple_low_bits_decision(
            str(row.hard_status), confidence, float(rule["threshold"])
        )
    if rule["kind"] == "corrected_bit_not_minimum":
        return corrected_bit_is_not_minimum_decision(
            str(row.hard_status), int(row.syndrome), int(row.overall_parity), confidence
        )
    raise ValueError(f"Nepoznata integrity rule vrsta: {rule['kind']}")


def candidate_rules(thresholds: dict[str, float]) -> list[dict]:
    rules = [
        {
            "rule_id": "R0_hard_operational",
            "kind": "baseline_status",
            "description": "existing operational hard SECDED",
            "selection_eligible": False,
        },
        {
            "rule_id": "R1_explicit_status_control",
            "kind": "baseline_status",
            "description": "reject any SECDED detected_double codeword",
            "selection_eligible": True,
        },
    ]
    for label in ("q05", "q10", "q15", "q20"):
        rules.append({
            "rule_id": f"R2_multiple_low_{label}",
            "kind": "multiple_low_bits",
            "threshold": thresholds[label],
            "description": (
                "R1 plus reject corrected_single when count(|C| <= T) >= 2; "
                "equivalent to second-lowest |C| <= T"
            ),
            "selection_eligible": True,
        })
    rules.append({
        "rule_id": "R4_corrected_bit_not_minimum",
        "kind": "corrected_bit_not_minimum",
        "description": (
            "R1 plus reject corrected_single when syndrome-corrected bit has |C| "
            "strictly above the codeword minimum"
        ),
        "selection_eligible": True,
    })
    return rules


def word_signature_rows(words: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    word_rows: list[dict] = []
    bit_rows: list[dict] = []
    for _, row in words.iterrows():
        confidence = parse_array(row.abs_correlations)
        encoded = parse_array(row.encoded_bits, int)
        received = parse_array(row.hard_received_bits, int)
        wrong = received != encoded
        ranks = np.argsort(np.argsort(confidence, kind="stable"), kind="stable")
        position = corrected_position(int(row.syndrome), int(row.overall_parity))
        if row.hard_status != "corrected_single":
            position = None
        ordered = np.sort(confidence)
        if bool(row.silent_miscorrection):
            category = "D_silent_miscorrected"
        elif row.hard_status == "clean" and not bool(row.hard_failed):
            category = "A_correct_clean"
        elif row.hard_status == "corrected_single" and not bool(row.hard_failed):
            category = "B_correct_repaired_single"
        elif row.hard_status == "detected_double":
            category = "C_detected_double_rejected"
        else:
            category = "other"
        record = {
            "config_id": row.config_id,
            "replay_file": row.replay_file,
            "beatmap_hash": row.beatmap_hash,
            "layout_seed": int(row.layout_seed),
            "codeword_index": int(row.codeword_index),
            "alpha": float(row.alpha),
            "comparison_category": category,
            "physical_error_count": int(row.raw_error_count),
            "physical_error_positions": row.raw_error_positions,
            "syndrome": int(row.syndrome),
            "overall_parity": int(row.overall_parity),
            "decoder_action": row.hard_status,
            "decoder_corrected_position": -1 if position is None else position,
            "min_abs_correlation": float(ordered[0]),
            "second_lowest_abs_correlation": float(ordered[1]),
            "median_abs_correlation": float(np.median(confidence)),
            "confidence_gap_lowest_second": float(ordered[1] - ordered[0]),
            "corrected_bit_abs_correlation": np.nan if position is None else float(confidence[position]),
            "corrected_bit_confidence_rank": -1 if position is None else int(ranks[position]),
        }
        for label, threshold in thresholds.items():
            record[f"count_below_{label}"] = int(np.sum(confidence <= threshold))
        word_rows.append(record)
        if bool(row.silent_miscorrection):
            for bit_position in range(8):
                bit_rows.append({
                    "config_id": row.config_id,
                    "replay_file": row.replay_file,
                    "beatmap_hash": row.beatmap_hash,
                    "layout_seed": int(row.layout_seed),
                    "codeword_index": int(row.codeword_index),
                    "alpha": float(row.alpha),
                    "physical_error_count": int(row.raw_error_count),
                    "syndrome": int(row.syndrome),
                    "overall_parity": int(row.overall_parity),
                    "decoder_action": row.hard_status,
                    "decoder_corrected_position": -1 if position is None else position,
                    "bit_position": bit_position,
                    "abs_correlation": float(confidence[bit_position]),
                    "confidence_rank": int(ranks[bit_position]),
                    "physically_wrong": int(wrong[bit_position]),
                    "changed_by_secded": int(position == bit_position),
                })
    return pd.DataFrame(word_rows), pd.DataFrame(bit_rows)


def signature_summary(signatures: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "min_abs_correlation", "second_lowest_abs_correlation",
        "median_abs_correlation", "confidence_gap_lowest_second",
        "count_below_q05", "count_below_q10", "count_below_q15", "count_below_q20",
        "corrected_bit_abs_correlation", "corrected_bit_confidence_rank",
    ]
    rows = []
    for category, group in signatures.groupby("comparison_category"):
        for metric in metrics:
            values = group[metric].replace(-1, np.nan).dropna().to_numpy(float)
            rows.append({
                "comparison_category": category,
                "codewords": len(group),
                "metric": metric,
                "observations": len(values),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "q25": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                "q75": float(np.quantile(values, 0.75)) if len(values) else np.nan,
            })
    return pd.DataFrame(rows)


def evaluate_design_rule(words: pd.DataFrame, physical: pd.DataFrame, rule: dict) -> tuple[dict, pd.DataFrame]:
    decisions = []
    for _, row in words.iterrows():
        decision = decision_for_rule(row, rule)
        decisions.append({
            "config_id": row.config_id,
            "accepted": decision.accepted,
            "reason": decision.reason,
            "word_wrong": bool(row.hard_failed),
        })
    word_decisions = pd.DataFrame(decisions)
    messages = []
    physical_by_id = physical.set_index("config_id")
    for config_id, group in word_decisions.groupby("config_id", sort=False):
        message_decision = message_integrity_decision([
            IntegrityDecision(bool(row.accepted), str(row.reason))
            for row in group.itertuples(index=False)
        ])
        wrong = bool(group.word_wrong.any())
        if not message_decision.accepted:
            outcome = "REJECT"
        elif wrong:
            outcome = "WRONG_ACCEPT"
        else:
            outcome = "CORRECT_ACCEPT"
        source = physical_by_id.loc[config_id]
        messages.append({
            "config_id": config_id,
            "replay_file": source.replay_file,
            "layout_seed": int(source.layout_seed),
            "useful_bits": int(source.useful_bits),
            "outcome": outcome,
        })
    message_frame = pd.DataFrame(messages)
    counts = message_frame.outcome.value_counts()
    correct = int(counts.get("CORRECT_ACCEPT", 0))
    wrong = int(counts.get("WRONG_ACCEPT", 0))
    rejected = int(counts.get("REJECT", 0))
    accepted = correct + wrong
    summary = {
        **rule,
        "configs": len(message_frame),
        "correct_accept": correct,
        "wrong_accept": wrong,
        "reject": rejected,
        "correct_accept_rate": correct / len(message_frame),
        "silent_error_rate": wrong / len(message_frame),
        "rejection_rate": rejected / len(message_frame),
        "accepted_message_accuracy": correct / accepted if accepted else np.nan,
        "useful_correctly_accepted_bits_per_config": float(
            (message_frame.useful_bits * (message_frame.outcome == "CORRECT_ACCEPT")).mean()
        ),
    }
    return summary, message_frame


def select_holdout(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    cohort = load_partitioned_cohort(
        args.results, args.performance_groups, args.partition, "development"
    ).copy()
    cohort["nominal_payload_bits"] = cohort.num_notes.map(
        lambda count: message_length_for_fraction(
            nominal_capacity_bits(int(count), N_VALUE), PAYLOAD_FRACTION
        )
    )
    eligible = cohort[cohort.nominal_payload_bits >= 8].copy()
    design = pd.read_csv(args.design_selection)
    remaining = eligible[~eligible.replay_file.isin(design.replay_file)].copy()
    disjoint = remaining[~remaining.beatmap_hash.isin(set(design.beatmap_hash))]
    if len(disjoint) >= args.holdout_size:
        pool = disjoint
        selection_basis = "beatmap_disjoint_deterministic_hash"
    else:
        pool = remaining
        selection_basis = "replay_disjoint_deterministic_hash; beatmap-disjoint infeasible"
    pool = pool.assign(selection_key=pool.replay_file.map(deterministic_order))
    selected = pool.sort_values(["selection_key", "replay_file"]).head(args.holdout_size).copy()
    selected.insert(0, "selection_order", np.arange(len(selected)))
    selected["selection_basis"] = selection_basis
    if selected.replay_file.duplicated().any() or set(selected.replay_file) & set(design.replay_file):
        raise ValueError("Holdout izbor nije replay-disjoint/jedinstven.")
    diagnostics = {
        "development_eligible_replays": len(eligible),
        "remaining_replay_disjoint": len(remaining),
        "beatmap_disjoint_available": len(disjoint),
        "selected_replays": len(selected),
        "design_beatmaps": int(design.beatmap_hash.nunique()),
        "selected_beatmaps": int(selected.beatmap_hash.nunique()),
        "overlapping_beatmaps": len(set(selected.beatmap_hash) & set(design.beatmap_hash)),
        "selection_basis": selection_basis,
    }
    return selected, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--design-dir", type=Path, default=DESIGN_DIR)
    parser.add_argument("--confidence-dir", type=Path, default=CONFIDENCE_DIR)
    parser.add_argument("--design-selection", type=Path, default=DESIGN_SELECTION)
    parser.add_argument("--frozen-config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv")
    parser.add_argument("--performance-groups", type=Path, default=METADATA_DIR / "performance_groups.csv")
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--holdout-size", type=int, default=80)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    words_path = args.design_dir / "ecc_codeword_results.csv"
    physical_path = args.design_dir / "physical_results.csv"
    words = pd.read_csv(words_path)
    physical = pd.read_csv(physical_path)
    physical = physical[physical.condition == "ecc"].copy()
    confidence_config = json.loads(
        (args.confidence_dir / "config.json").read_text(encoding="utf-8")
    )
    raw_thresholds = confidence_config["erasure_candidate_thresholds"]
    thresholds = {
        f"q{int(float(fraction) * 100):02d}": float(value)
        for fraction, value in raw_thresholds.items()
    }
    signatures, silent_bits = word_signature_rows(words, thresholds)
    if int((signatures.comparison_category == "D_silent_miscorrected").sum()) != 17:
        raise ValueError("Design silent-miscorrection count se promenio.")
    silent_words = signatures[
        signatures.comparison_category == "D_silent_miscorrected"
    ].copy()
    silent_words.to_csv(args.output_dir / "design_silent_miscorrections.csv", index=False)
    silent_bits.to_csv(args.output_dir / "design_silent_miscorrection_bits.csv", index=False)
    signature_summary(signatures).to_csv(
        args.output_dir / "design_codeword_signature_summary.csv", index=False
    )

    summaries = []
    message_frames: dict[str, pd.DataFrame] = {}
    rules = candidate_rules(thresholds)
    for rule in rules:
        summary, messages = evaluate_design_rule(words, physical, rule)
        summaries.append(summary)
        message_frames[rule["rule_id"]] = messages
    candidates = pd.DataFrame(summaries)
    r0 = candidates[candidates.rule_id == "R0_hard_operational"].iloc[0]
    r1 = candidates[candidates.rule_id == "R1_explicit_status_control"].iloc[0]
    for metric in ("correct_accept", "wrong_accept", "reject"):
        if int(r0[metric]) != int(r1[metric]):
            raise ValueError("R1 ne reprodukuje operational hard SECDED.")
    baseline_messages = message_frames["R1_explicit_status_control"].set_index("config_id")
    enriched = []
    for summary in summaries:
        messages = message_frames[summary["rule_id"]].set_index("config_id")
        baseline_wrong = baseline_messages.outcome == "WRONG_ACCEPT"
        baseline_correct = baseline_messages.outcome == "CORRECT_ACCEPT"
        summary = dict(summary)
        summary["baseline_silent_captured"] = int(
            (baseline_wrong & (messages.outcome == "REJECT")).sum()
        )
        summary["additional_correct_rejected"] = int(
            (baseline_correct & (messages.outcome == "REJECT")).sum()
        )
        enriched.append(summary)
    candidates = pd.DataFrame(enriched)
    selectable = candidates[candidates.selection_eligible.astype(bool)].copy()
    selectable = selectable.sort_values(
        ["silent_error_rate", "correct_accept_rate", "rejection_rate", "rule_id"],
        ascending=[True, False, True, True],
    )
    selected_id = str(selectable.iloc[0].rule_id)
    candidates["frozen_selected"] = (candidates.rule_id == selected_id).astype(int)
    candidates.to_csv(args.output_dir / "design_rule_candidates.csv", index=False)
    selected_rule = next(rule for rule in rules if rule["rule_id"] == selected_id)

    holdout, selection_diagnostics = select_holdout(args)
    holdout_path = args.output_dir / "holdout_selection.csv"
    holdout.to_csv(holdout_path, index=False)
    frozen = {
        "experiment_version": EXPERIMENT_VERSION,
        "rule_id": selected_rule["rule_id"],
        "rule_kind": selected_rule["kind"],
        "rule_definition": selected_rule["description"],
        "confidence_threshold": selected_rule.get("threshold"),
        "codeword_decision_semantics": "baseline SECDED reject plus frozen observable rule",
        "message_rejection_semantics": "reject entire message if any codeword rejects",
        "selection_objective": (
            "lexicographic: minimize design silent WRONG_ACCEPT rate; then maximize "
            "CORRECT_ACCEPT rate; then minimize rejection rate"
        ),
        "selection_partition": "development_design_only",
        "design_selection_sha256": file_sha256(args.design_selection),
        "design_words_sha256": file_sha256(words_path),
        "design_physical_sha256": file_sha256(physical_path),
        "confidence_config_sha256": file_sha256(args.confidence_dir / "config.json"),
        "adaptive_policy_sha256": file_sha256(args.policy),
        "development_partition_sha256": file_sha256(args.partition),
        "holdout_selection_sha256": file_sha256(holdout_path),
        "holdout_selection_frame_sha256": frame_sha256(holdout),
        "holdout_selection_created_before_physical_evaluation": True,
        "N": N_VALUE,
        "payload_fraction": PAYLOAD_FRACTION,
        "selection_diagnostics": selection_diagnostics,
    }
    write_or_validate_config(args.frozen_config, frozen)
    write_or_validate_config(args.output_dir / "frozen_rule.json", frozen)
    diagnostics = {
        "design_codewords": len(words),
        "design_configs": physical.config_id.nunique(),
        "design_silent_codewords": int(words.silent_miscorrection.sum()),
        "frozen_rule_id": selected_id,
        "frozen_config_sha256": file_sha256(args.frozen_config),
        **selection_diagnostics,
    }
    pd.DataFrame(
        [{"diagnostic": key, "value": value} for key, value in diagnostics.items()]
    ).to_csv(args.output_dir / "design_diagnostics.csv", index=False)
    print(f"design frozen | rule={selected_id}")
    print(f"frozen_config_sha256={file_sha256(args.frozen_config)}")
    print(
        f"holdout={len(holdout)} replay-disjoint | "
        f"beatmap_overlap={selection_diagnostics['overlapping_beatmaps']}"
    )


if __name__ == "__main__":
    main()
