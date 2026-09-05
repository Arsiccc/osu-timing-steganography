"""Frozen physical SECDED holdout and receiver-only integrity evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import CONFIG_DIR, DATASET_DIR, METADATA_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import alpha_from_replay, load_sender_local_policy
from osu_stego.stego.ecc import (
    decode_hamming_8_4_secded,
    encode_hamming_8_4_secded,
    secded_syndrome,
)
from osu_stego.stego.integrity import (
    IntegrityDecision,
    baseline_integrity_decision,
    corrected_bit_is_not_minimum_decision,
    message_integrity_decision,
    multiple_low_bits_decision,
)
from osu_stego.stego.payload_layout import (
    extract_bipolar_message_with_layout,
    message_correlations_with_layout,
)
from scripts.experiments.adaptive_layout_strong_common import (
    LAYOUT_KEYS,
    MESSAGE_SEED,
    N_VALUE,
    PAYLOAD_FRACTION,
    PN_KEY,
    replace_config_rows,
    write_or_validate_config,
)
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_ecc_equal_payload import array_hash, key_id
from scripts.experiments.run_ecc_pilot import DEFAULT_PARTITION, DEFAULT_POLICY
from scripts.experiments.run_layout_comparison import build_physical_stego
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
from scripts.experiments.sender_local_common import load_partitioned_cohort


EXPERIMENT_VERSION = "ecc-integrity-holdout-v1"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "ecc_integrity_rejection_v1"
DEFAULT_FROZEN_RULE = CONFIG_DIR / "ecc_integrity_rejection_v1.json"
PHYSICAL_FIELDS = [
    "experiment_version", "config_id", "replay_file", "beatmap_hash",
    "performance_category", "layout_seed", "layout_key_id", "alpha", "N",
    "payload_fraction", "nominal_payload_bits", "useful_bits", "physical_bits",
    "message_sha256", "encoded_message_sha256", "raw_bit_errors", "raw_ber",
    "post_ecc_bit_errors", "post_ecc_ber", "ground_truth_full_message_correct",
    "codewords", "clean_codewords", "corrected_single_codewords",
    "detected_double_codewords", "silent_miscorrected_codewords",
    "requested_carriers", "active_carriers", "active_carrier_fraction",
    "dropped_for_hit_window", "dropped_for_chronology", "new_unmatched_events",
    "new_matched_events", "changed_match_status_total", "sum_squared_shift_ms2",
    "rms_applied_shift_ms", "mean_absolute_applied_shift_ms", "policy_sha256",
    "replay_sha256", "pn_key_id", "frozen_rule_sha256",
]
RAW_WORD_FIELDS = [
    "config_id", "replay_file", "beatmap_hash", "layout_seed", "alpha",
    "codeword_index", "true_info_bits", "encoded_bits", "hard_received_bits",
    "hard_decoded_info_bits", "raw_error_count", "raw_error_positions",
    "hard_status", "syndrome", "overall_parity", "hard_failed",
    "silent_miscorrection", "abs_correlations",
]


def identifier(
    replay_file: str,
    replay_sha256: str,
    layout_seed: int,
    encoded: np.ndarray,
    policy_sha256: str,
    frozen_rule_sha256: str,
) -> str:
    material = "|".join((
        EXPERIMENT_VERSION, replay_file, replay_sha256, str(layout_seed),
        array_hash(encoded), policy_sha256, frozen_rule_sha256,
        key_id(PN_KEY), key_id(LAYOUT_KEYS[layout_seed]), str(N_VALUE),
        str(PAYLOAD_FRACTION),
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def completed_configs(
    physical_path: Path, raw_word_path: Path
) -> set[str]:
    if not physical_path.is_file() or not raw_word_path.is_file():
        return set()
    physical = pd.read_csv(physical_path)
    words = pd.read_csv(raw_word_path)
    counts = words.groupby("config_id").size()
    return {
        str(row.config_id)
        for row in physical.itertuples(index=False)
        if int(counts.get(str(row.config_id), -1)) == int(row.codewords)
    }


def frozen_decision(row: pd.Series, frozen: dict) -> IntegrityDecision:
    confidence = np.asarray(json.loads(row.abs_correlations), dtype=np.float64)
    if frozen["rule_kind"] == "corrected_bit_not_minimum":
        return corrected_bit_is_not_minimum_decision(
            str(row.hard_status), int(row.syndrome), int(row.overall_parity), confidence
        )
    if frozen["rule_kind"] == "multiple_low_bits":
        return multiple_low_bits_decision(
            str(row.hard_status), confidence, float(frozen["confidence_threshold"])
        )
    if frozen["rule_kind"] == "baseline_status":
        return baseline_integrity_decision(str(row.hard_status))
    raise ValueError("Frozen integrity rule nije podržan.")


def cluster_bootstrap_delta(
    messages: pd.DataFrame,
    metric: str,
    iterations: int = 5000,
) -> tuple[float, float, float]:
    wide = messages.pivot(
        index=["replay_file", "layout_seed"], columns="method", values=metric
    ).reset_index()
    replay_delta = (
        wide.assign(delta=wide.integrity - wide.baseline)
        .groupby("replay_file").delta.mean().to_numpy(float)
    )
    delta = float(replay_delta.mean())
    rng = np.random.default_rng(stable_seed(42, EXPERIMENT_VERSION, metric))
    indices = rng.integers(0, len(replay_delta), size=(iterations, len(replay_delta)))
    samples = replay_delta[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return delta, float(low), float(high)


def classify_word(row: pd.Series, method: str, frozen: dict) -> dict:
    decision = (
        baseline_integrity_decision(str(row.hard_status))
        if method == "baseline"
        else frozen_decision(row, frozen)
    )
    wrong = bool(row.hard_failed)
    if not decision.accepted:
        evaluation = "REJECT"
    elif wrong:
        evaluation = "WRONG_ACCEPT"
    else:
        evaluation = "CORRECT_ACCEPT"
    if not decision.accepted:
        action = (
            "secded_rejected"
            if decision.reason == "secded_detected_uncorrectable"
            else "confidence_rejected"
        )
    elif row.hard_status == "corrected_single":
        action = "corrected_accepted"
    else:
        action = "accepted_unchanged"
    return {
        "method": method,
        "accepted": int(decision.accepted),
        "rejection_reason": decision.reason,
        "operational_action": action,
        "evaluation_outcome": evaluation,
        "ground_truth_correct": int(not wrong),
        "silent_miscorrection": int(decision.accepted and wrong),
    }


def build_message_results(
    codewords: pd.DataFrame, physical: pd.DataFrame
) -> pd.DataFrame:
    physical_by_id = physical.set_index("config_id")
    rows = []
    for (method, config_id), group in codewords.groupby(["method", "config_id"]):
        decisions = [
            IntegrityDecision(bool(row.accepted), str(row.rejection_reason))
            for row in group.itertuples(index=False)
        ]
        decision = message_integrity_decision(decisions)
        wrong = bool((group.ground_truth_correct == 0).any())
        if not decision.accepted:
            outcome = "REJECT"
        elif wrong:
            outcome = "WRONG_ACCEPT"
        else:
            outcome = "CORRECT_ACCEPT"
        source = physical_by_id.loc[config_id]
        rows.append({
            "method": method,
            "config_id": config_id,
            "replay_file": source.replay_file,
            "beatmap_hash": source.beatmap_hash,
            "layout_seed": int(source.layout_seed),
            "alpha": float(source.alpha),
            "useful_bits": int(source.useful_bits),
            "physical_bits": int(source.physical_bits),
            "accepted": int(decision.accepted),
            "rejection_reason": decision.reason,
            "outcome": outcome,
            "correct_accept": int(outcome == "CORRECT_ACCEPT"),
            "wrong_accept": int(outcome == "WRONG_ACCEPT"),
            "reject": int(outcome == "REJECT"),
            "ground_truth_full_message_correct": int(not wrong),
            "useful_correctly_accepted_bits": int(source.useful_bits) * int(
                outcome == "CORRECT_ACCEPT"
            ),
        })
    return pd.DataFrame(rows)


def message_summary(messages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in messages.groupby("method"):
        accepted = int(group.accepted.sum())
        correct = int(group.correct_accept.sum())
        wrong = int(group.wrong_accept.sum())
        rows.append({
            "method": method,
            "configs": len(group),
            "replays": group.replay_file.nunique(),
            "correct_accept": correct,
            "wrong_accept": wrong,
            "reject": int(group.reject.sum()),
            "correct_accept_rate": correct / len(group),
            "silent_error_rate": wrong / len(group),
            "rejection_rate": float(group.reject.mean()),
            "accepted_message_accuracy": correct / accepted if accepted else np.nan,
            "ground_truth_full_message_correctness": float(
                group.ground_truth_full_message_correct.mean()
            ),
            "useful_correctly_accepted_bits_per_config": float(
                group.useful_correctly_accepted_bits.mean()
            ),
            "useful_correctly_accepted_bits_per_replay_across_five_seeds": float(
                group.groupby("replay_file").useful_correctly_accepted_bits.sum().mean()
            ),
        })
    return pd.DataFrame(rows)


def grouped_results(messages: pd.DataFrame, group_field: str) -> pd.DataFrame:
    rows = []
    for value, group in messages.groupby(group_field):
        wide = group.pivot(index="config_id", columns="method")
        rows.append({
            group_field: value,
            "configs": len(wide),
            "replays": group.replay_file.nunique(),
            "baseline_silent_error_rate": float(wide["wrong_accept"].baseline.mean()),
            "integrity_silent_error_rate": float(wide["wrong_accept"].integrity.mean()),
            "baseline_correct_accept_rate": float(wide["correct_accept"].baseline.mean()),
            "integrity_correct_accept_rate": float(wide["correct_accept"].integrity.mean()),
            "baseline_rejection_rate": float(wide["reject"].baseline.mean()),
            "integrity_rejection_rate": float(wide["reject"].integrity.mean()),
        })
    return pd.DataFrame(rows)


def evaluate(
    physical_path: Path,
    raw_word_path: Path,
    output_dir: Path,
    frozen: dict,
    expected_replays: int,
) -> None:
    physical = pd.read_csv(physical_path)
    raw_words = pd.read_csv(raw_word_path)
    expected_configs = expected_replays * len(LAYOUT_KEYS)
    if len(physical) != expected_configs or physical.config_id.nunique() != expected_configs:
        raise ValueError("Holdout physical rezultati nisu kompletni/jedinstveni.")
    if raw_words.duplicated(["config_id", "codeword_index"]).any():
        raise ValueError("Duplikat holdout codeword reda.")
    rows = []
    for _, row in raw_words.iterrows():
        common = row.to_dict()
        for method in ("baseline", "integrity"):
            rows.append({**common, **classify_word(row, method, frozen)})
    codewords = pd.DataFrame(rows)
    codewords.to_csv(output_dir / "codeword_results.csv", index=False)
    messages = build_message_results(codewords, physical)
    messages.to_csv(output_dir / "message_results.csv", index=False)
    summary = message_summary(messages)
    summary.to_csv(output_dir / "message_summary.csv", index=False)

    paired_rows = []
    for metric in (
        "wrong_accept", "correct_accept", "reject", "useful_correctly_accepted_bits"
    ):
        delta, low, high = cluster_bootstrap_delta(messages, metric)
        paired_rows.append({
            "metric": metric,
            "delta_integrity_minus_baseline": delta,
            "ci95_low": low,
            "ci95_high": high,
            "bootstrap_unit": "replay_file",
            "bootstrap_iterations": 5000,
        })
    pd.DataFrame(paired_rows).to_csv(
        output_dir / "paired_integrity_results.csv", index=False
    )

    wide_outcomes = messages.pivot(
        index=["config_id", "replay_file", "layout_seed"],
        columns="method", values="outcome"
    ).reset_index()
    observed_transitions = (
        wide_outcomes.groupby(["baseline", "integrity"]).size()
        .rename("configs")
    )
    transition_index = pd.MultiIndex.from_product(
        [
            ("CORRECT_ACCEPT", "WRONG_ACCEPT", "REJECT"),
            ("CORRECT_ACCEPT", "WRONG_ACCEPT", "REJECT"),
        ],
        names=["baseline", "integrity"],
    )
    transitions = observed_transitions.reindex(transition_index, fill_value=0).reset_index()
    transitions.to_csv(output_dir / "silent_error_transitions.csv", index=False)
    grouped_results(messages, "layout_seed").to_csv(
        output_dir / "results_by_seed.csv", index=False
    )
    grouped_results(messages, "useful_bits").to_csv(
        output_dir / "results_by_message_length.csv", index=False
    )

    codeword_diagnostics = (
        codewords.groupby(
            ["method", "operational_action", "evaluation_outcome"], dropna=False
        ).size().rename("codewords").reset_index()
    )
    codeword_diagnostics.to_csv(
        output_dir / "codeword_diagnostics.csv", index=False
    )
    status_rows = []
    for method, group in codewords.groupby("method"):
        status_rows.extend([
            {"method": method, "status": "accepted_unchanged_correct", "codewords": int(((group.operational_action == "accepted_unchanged") & (group.evaluation_outcome == "CORRECT_ACCEPT")).sum())},
            {"method": method, "status": "corrected_accepted_correct", "codewords": int(((group.operational_action == "corrected_accepted") & (group.evaluation_outcome == "CORRECT_ACCEPT")).sum())},
            {"method": method, "status": "secded_rejected", "codewords": int((group.operational_action == "secded_rejected").sum())},
            {"method": method, "status": "confidence_rejected", "codewords": int((group.operational_action == "confidence_rejected").sum())},
            {"method": method, "status": "silent_miscorrected", "codewords": int(group.silent_miscorrection.sum())},
        ])
    pd.DataFrame(status_rows).to_csv(
        output_dir / "codeword_status_counts.csv", index=False
    )
    baseline = messages[messages.method == "baseline"].set_index("config_id")
    integrity = messages[messages.method == "integrity"].set_index("config_id")
    baseline_silent = baseline.wrong_accept.astype(bool)
    baseline_correct = baseline.correct_accept.astype(bool)
    caught = int((baseline_silent & integrity.reject.astype(bool)).sum())
    remaining = int((baseline_silent & integrity.wrong_accept.astype(bool)).sum())
    false_rejects = int((baseline_correct & integrity.reject.astype(bool)).sum())
    diagnostics = [
        {"diagnostic": "physical_configs", "value": len(physical)},
        {"diagnostic": "unique_physical_configs", "value": physical.config_id.nunique()},
        {"diagnostic": "raw_codewords", "value": len(raw_words)},
        {"diagnostic": "baseline_silent_messages", "value": int(baseline_silent.sum())},
        {"diagnostic": "baseline_silent_to_reject", "value": caught},
        {"diagnostic": "baseline_silent_still_wrong_accept", "value": remaining},
        {
            "diagnostic": "silent_error_capture_rate",
            "value": caught / int(baseline_silent.sum()) if baseline_silent.any() else np.nan,
        },
        {"diagnostic": "baseline_correct_messages", "value": int(baseline_correct.sum())},
        {"diagnostic": "additional_false_rejections", "value": false_rejects},
        {
            "diagnostic": "false_rejection_rate_among_baseline_correct",
            "value": false_rejects / int(baseline_correct.sum()),
        },
        {
            "diagnostic": "silent_prevented_per_additional_correct_rejected",
            "value": caught / false_rejects if false_rejects else np.nan,
        },
        {"diagnostic": "detectability_rerun", "value": "not run; physical replay unchanged by decoder rule"},
    ]
    pd.DataFrame(diagnostics).to_csv(output_dir / "diagnostics.csv", index=False)


def generate(args: argparse.Namespace, frozen: dict, run_config: dict) -> None:
    selection = pd.read_csv(args.selection)
    if args.limit_replays is not None:
        selection = selection.head(args.limit_replays)
    cohort = load_partitioned_cohort(
        args.results, args.performance_groups, args.partition, "development"
    )
    selected = selection[["replay_file"]].merge(
        cohort, on="replay_file", how="left", validate="one_to_one"
    )
    if selected.beatmap_hash.isna().any():
        raise ValueError("Holdout selection nije development podskup.")
    design = pd.read_csv(args.design_selection)
    if set(selected.replay_file) & set(design.replay_file):
        raise ValueError("Holdout se preklapa sa design replay-evima.")
    beatmaps = build_beatmap_index(args.dataset_dir)
    offsets = load_map_offsets(args.map_offsets)
    policy = load_sender_local_policy(args.policy)
    physical_path = args.output_dir / "holdout_physical_results.csv"
    raw_word_path = args.output_dir / "holdout_raw_codewords.csv"
    complete = completed_configs(physical_path, raw_word_path)
    pending_physical: list[dict] = []
    pending_words: list[dict] = []
    new = 0
    with tempfile.TemporaryDirectory(prefix="osu_integrity_holdout_") as name:
        temp_dir = Path(name)
        for replay_index, row in enumerate(selected.itertuples(index=False), 1):
            context = prepare_replay(row, args.dataset_dir, beatmaps, offsets)
            replay_hash = file_sha256(context.osr_path)
            alpha = alpha_from_replay(policy, context.replay)
            nominal = message_length_for_fraction(
                nominal_capacity_bits(len(context.original_residuals), N_VALUE),
                PAYLOAD_FRACTION,
            )
            codeword_count = nominal // 8
            useful_count = codeword_count * 4
            physical_count = codeword_count * 8
            useful = deterministic_message(
                context.replay_file, context.beatmap_hash, 0.0, N_VALUE,
                PAYLOAD_FRACTION, useful_count, MESSAGE_SEED,
            )
            encoded = encode_hamming_8_4_secded(useful)
            for seed, layout_key in enumerate(LAYOUT_KEYS):
                config_id = identifier(
                    context.replay_file, replay_hash, seed, encoded,
                    run_config["policy_sha256"], run_config["frozen_rule_sha256"],
                )
                if config_id in complete:
                    continue
                _, roundtrip, diagnostics = build_physical_stego(
                    context, encoded, alpha, N_VALUE, "distributed", PN_KEY,
                    layout_key, float(policy["hit_margin_ms"]),
                    temp_dir / f"{config_id}.osr",
                )
                raw = extract_bipolar_message_with_layout(
                    roundtrip, PN_KEY, layout_key, N_VALUE, physical_count, "distributed"
                )
                correlations = message_correlations_with_layout(
                    roundtrip, PN_KEY, layout_key, N_VALUE, physical_count, "distributed"
                )
                decoded, statuses = decode_hamming_8_4_secded(raw)
                raw_errors = int(np.sum(raw != encoded))
                post_errors = int(np.sum(decoded != useful))
                true_words = useful.reshape(-1, 4)
                decoded_words = decoded.reshape(-1, 4)
                failed_words = np.any(decoded_words != true_words, axis=1)
                for word_index in range(codeword_count):
                    code_slice = slice(word_index * 8, (word_index + 1) * 8)
                    info_slice = slice(word_index * 4, (word_index + 1) * 4)
                    errors = np.flatnonzero(raw[code_slice] != encoded[code_slice])
                    syndrome, parity = secded_syndrome(raw[code_slice])
                    pending_words.append({
                        "config_id": config_id,
                        "replay_file": context.replay_file,
                        "beatmap_hash": context.beatmap_hash,
                        "layout_seed": seed,
                        "alpha": alpha,
                        "codeword_index": word_index,
                        "true_info_bits": json.dumps(useful[info_slice].tolist()),
                        "encoded_bits": json.dumps(encoded[code_slice].tolist()),
                        "hard_received_bits": json.dumps(raw[code_slice].tolist()),
                        "hard_decoded_info_bits": json.dumps(decoded[info_slice].tolist()),
                        "raw_error_count": len(errors),
                        "raw_error_positions": json.dumps(errors.tolist()),
                        "hard_status": str(statuses[word_index]),
                        "syndrome": syndrome,
                        "overall_parity": parity,
                        "hard_failed": int(failed_words[word_index]),
                        "silent_miscorrection": int(
                            failed_words[word_index]
                            and statuses[word_index] != "detected_double"
                        ),
                        "abs_correlations": json.dumps(
                            np.abs(correlations[code_slice]).tolist()
                        ),
                    })
                pending_physical.append({
                    "experiment_version": EXPERIMENT_VERSION,
                    "config_id": config_id,
                    "replay_file": context.replay_file,
                    "beatmap_hash": context.beatmap_hash,
                    "performance_category": row.performance_category,
                    "layout_seed": seed,
                    "layout_key_id": key_id(layout_key),
                    "alpha": alpha,
                    "N": N_VALUE,
                    "payload_fraction": PAYLOAD_FRACTION,
                    "nominal_payload_bits": nominal,
                    "useful_bits": useful_count,
                    "physical_bits": physical_count,
                    "message_sha256": array_hash(useful),
                    "encoded_message_sha256": array_hash(encoded),
                    "raw_bit_errors": raw_errors,
                    "raw_ber": raw_errors / physical_count,
                    "post_ecc_bit_errors": post_errors,
                    "post_ecc_ber": post_errors / useful_count,
                    "ground_truth_full_message_correct": int(post_errors == 0),
                    "codewords": codeword_count,
                    "clean_codewords": int(np.sum(statuses == "clean")),
                    "corrected_single_codewords": int(np.sum(statuses == "corrected_single")),
                    "detected_double_codewords": int(np.sum(statuses == "detected_double")),
                    "silent_miscorrected_codewords": int(
                        np.sum(failed_words & (statuses != "detected_double"))
                    ),
                    "active_carrier_fraction": (
                        diagnostics["active_carriers"] / diagnostics["requested_carriers"]
                        if diagnostics["requested_carriers"] else 0.0
                    ),
                    "policy_sha256": run_config["policy_sha256"],
                    "replay_sha256": replay_hash,
                    "pn_key_id": key_id(PN_KEY),
                    "frozen_rule_sha256": run_config["frozen_rule_sha256"],
                    **diagnostics,
                })
                new += 1
                if new % 10 == 0:
                    replace_config_rows(physical_path, PHYSICAL_FIELDS, pending_physical)
                    replace_config_rows(raw_word_path, RAW_WORD_FIELDS, pending_words)
                    pending_physical, pending_words = [], []
            if replay_index % 10 == 0 or replay_index == len(selected):
                print(
                    f"integrity holdout: {replay_index}/{len(selected)} | new configs={new}",
                    flush=True,
                )
        replace_config_rows(physical_path, PHYSICAL_FIELDS, pending_physical)
        replace_config_rows(raw_word_path, RAW_WORD_FIELDS, pending_words)
    print(f"holdout generation complete | new configs={new}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection", type=Path, default=DEFAULT_OUTPUT_DIR / "holdout_selection.csv")
    parser.add_argument("--frozen-rule", type=Path, default=DEFAULT_FROZEN_RULE)
    parser.add_argument("--design-selection", type=Path, default=RESULTS_DIR / "ecc_pilot_v1" / "pilot_selection.csv")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv")
    parser.add_argument("--performance-groups", type=Path, default=METADATA_DIR / "performance_groups.csv")
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--limit-replays", type=int, default=None)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.generate_only and args.analyze_only:
        raise ValueError("--generate-only i --analyze-only ne mogu zajedno.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.frozen_rule.read_text(encoding="utf-8"))
    run_config = {
        "experiment_version": EXPERIMENT_VERSION,
        "frozen_rule_sha256": file_sha256(args.frozen_rule),
        "holdout_selection_sha256": file_sha256(args.selection),
        "policy_sha256": file_sha256(args.policy),
        "partition_sha256": file_sha256(args.partition),
        "map_offsets_sha256": file_sha256(args.map_offsets),
        "cohort_sha256": file_sha256(args.results),
        "design_selection_sha256": file_sha256(args.design_selection),
        "N": N_VALUE,
        "payload_fraction": PAYLOAD_FRACTION,
        "pn_key_id": key_id(PN_KEY),
        "layout_key_ids": [key_id(value) for value in LAYOUT_KEYS],
        "message_seed": MESSAGE_SEED,
        "limit_replays": args.limit_replays,
        "physical_replay_shared_by_all_decoder_rules": True,
    }
    write_or_validate_config(args.output_dir / "holdout_config.json", run_config)
    if not args.analyze_only:
        generate(args, frozen, run_config)
    if not args.generate_only:
        expected_replays = (
            args.limit_replays
            if args.limit_replays is not None
            else len(pd.read_csv(args.selection))
        )
        evaluate(
            args.output_dir / "holdout_physical_results.csv",
            args.output_dir / "holdout_raw_codewords.csv",
            args.output_dir,
            frozen,
            expected_replays,
        )
        print("holdout integrity analysis complete")


if __name__ == "__main__":
    main()
