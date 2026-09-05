"""One-shot unseen-beatmap validation of the frozen SECDED integrity rule."""

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
from osu_stego.stego.ecc import decode_hamming_8_4_secded, encode_hamming_8_4_secded, secded_syndrome
from osu_stego.stego.integrity import corrected_position
from osu_stego.stego.payload_layout import extract_bipolar_message_with_layout, message_correlations_with_layout
from scripts.experiments.adaptive_layout_strong_common import (
    LAYOUT_KEYS, MESSAGE_SEED, N_VALUE, PAYLOAD_FRACTION, PN_KEY,
    replace_config_rows, write_or_validate_config,
)
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_ecc_equal_payload import array_hash, key_id
from scripts.experiments.run_ecc_integrity_holdout import (
    PHYSICAL_FIELDS, RAW_WORD_FIELDS, completed_configs, evaluate,
)
from scripts.experiments.run_ecc_pilot import DEFAULT_PARTITION, DEFAULT_POLICY
from scripts.experiments.run_layout_comparison import build_physical_stego
from scripts.experiments.run_payload_sweep import deterministic_message, message_length_for_fraction, nominal_capacity_bits
from scripts.experiments.run_pilot_ber_sweep import build_beatmap_index, load_map_offsets, prepare_replay
from scripts.experiments.sender_local_common import load_partitioned_cohort


EXPECTED_RULE_SHA256 = "f49467d3b69e18b76ba88816d38270fc9c948583166c3a99d9c5ada2bc67ae42"
EXPERIMENT_VERSION = "ecc-integrity-validation-v1"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "ecc_integrity_validation_v1"
DEFAULT_RULE = CONFIG_DIR / "ecc_integrity_rejection_v1.json"


def validation_identifier(
    replay_file: str,
    replay_sha256: str,
    layout_seed: int,
    encoded: np.ndarray,
    policy_sha256: str,
    frozen_rule_sha256: str,
) -> str:
    """Build a resume identity namespaced to this validation experiment."""
    material = "|".join((
        EXPERIMENT_VERSION, replay_file, replay_sha256, str(layout_seed),
        array_hash(encoded), policy_sha256, frozen_rule_sha256,
        key_id(PN_KEY), key_id(LAYOUT_KEYS[layout_seed]), str(N_VALUE),
        str(PAYLOAD_FRACTION),
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def generate(args: argparse.Namespace, frozen: dict, run_config: dict) -> None:
    selection = pd.read_csv(args.selection)
    if args.limit_replays is not None:
        selection = selection.head(args.limit_replays)
    cohort = load_partitioned_cohort(
        args.results, args.performance_groups, args.partition, "validation"
    )
    selected = selection[["replay_file"]].merge(
        cohort, on="replay_file", how="left", validate="one_to_one"
    )
    if selected.beatmap_hash.isna().any():
        raise ValueError("Validation selection nije podskup frozen validation particije.")
    physical_path = args.output_dir / "physical_provenance.csv"
    raw_word_path = args.output_dir / "validation_raw_codewords.csv"
    complete = completed_configs(physical_path, raw_word_path)
    beatmaps = build_beatmap_index(args.dataset_dir)
    offsets = load_map_offsets(args.map_offsets)
    policy = load_sender_local_policy(args.policy)
    pending_physical: list[dict] = []
    pending_words: list[dict] = []
    new = 0
    with tempfile.TemporaryDirectory(prefix="osu_integrity_validation_") as name:
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
            if codeword_count < 1:
                raise ValueError("Validation lock sadrži SECDED-ineligible replay.")
            useful_count = codeword_count * 4
            physical_count = codeword_count * 8
            useful = deterministic_message(
                context.replay_file, context.beatmap_hash, 0.0, N_VALUE,
                PAYLOAD_FRACTION, useful_count, MESSAGE_SEED,
            )
            encoded = encode_hamming_8_4_secded(useful)
            for seed, layout_key in enumerate(LAYOUT_KEYS):
                config_id = validation_identifier(
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
                        "config_id": config_id, "replay_file": context.replay_file,
                        "beatmap_hash": context.beatmap_hash, "layout_seed": seed,
                        "alpha": alpha, "codeword_index": word_index,
                        "true_info_bits": json.dumps(useful[info_slice].tolist()),
                        "encoded_bits": json.dumps(encoded[code_slice].tolist()),
                        "hard_received_bits": json.dumps(raw[code_slice].tolist()),
                        "hard_decoded_info_bits": json.dumps(decoded[info_slice].tolist()),
                        "raw_error_count": len(errors),
                        "raw_error_positions": json.dumps(errors.tolist()),
                        "hard_status": str(statuses[word_index]),
                        "syndrome": syndrome, "overall_parity": parity,
                        "hard_failed": int(failed_words[word_index]),
                        "silent_miscorrection": int(
                            failed_words[word_index] and statuses[word_index] != "detected_double"
                        ),
                        "abs_correlations": json.dumps(np.abs(correlations[code_slice]).tolist()),
                    })
                pending_physical.append({
                    "experiment_version": EXPERIMENT_VERSION,
                    "config_id": config_id, "replay_file": context.replay_file,
                    "beatmap_hash": context.beatmap_hash,
                    "performance_category": row.performance_category,
                    "layout_seed": seed, "layout_key_id": key_id(layout_key),
                    "alpha": alpha, "N": N_VALUE, "payload_fraction": PAYLOAD_FRACTION,
                    "nominal_payload_bits": nominal, "useful_bits": useful_count,
                    "physical_bits": physical_count, "message_sha256": array_hash(useful),
                    "encoded_message_sha256": array_hash(encoded),
                    "raw_bit_errors": raw_errors, "raw_ber": raw_errors / physical_count,
                    "post_ecc_bit_errors": post_errors, "post_ecc_ber": post_errors / useful_count,
                    "ground_truth_full_message_correct": int(post_errors == 0),
                    "codewords": codeword_count,
                    "clean_codewords": int(np.sum(statuses == "clean")),
                    "corrected_single_codewords": int(np.sum(statuses == "corrected_single")),
                    "detected_double_codewords": int(np.sum(statuses == "detected_double")),
                    "silent_miscorrected_codewords": int(np.sum(failed_words & (statuses != "detected_double"))),
                    "active_carrier_fraction": (
                        diagnostics["active_carriers"] / diagnostics["requested_carriers"]
                        if diagnostics["requested_carriers"] else 0.0
                    ),
                    "policy_sha256": run_config["policy_sha256"],
                    "replay_sha256": replay_hash, "pn_key_id": key_id(PN_KEY),
                    "frozen_rule_sha256": run_config["frozen_rule_sha256"],
                    **diagnostics,
                })
                new += 1
                if new % 10 == 0:
                    replace_config_rows(physical_path, PHYSICAL_FIELDS, pending_physical)
                    replace_config_rows(raw_word_path, RAW_WORD_FIELDS, pending_words)
                    pending_physical, pending_words = [], []
            if replay_index % 10 == 0 or replay_index == len(selected):
                print(f"validation: {replay_index}/{len(selected)} | new configs={new}", flush=True)
        replace_config_rows(physical_path, PHYSICAL_FIELDS, pending_physical)
        replace_config_rows(raw_word_path, RAW_WORD_FIELDS, pending_words)
    print(f"validation generation complete | new configs={new}")


def add_validation_analyses(output_dir: Path) -> None:
    messages = pd.read_csv(output_dir / "message_results.csv")
    words = pd.read_csv(output_dir / "codeword_results.csv")
    pd.read_csv(output_dir / "message_summary.csv").to_csv(output_dir / "summary.csv", index=False)
    pd.read_csv(output_dir / "paired_integrity_results.csv").to_csv(output_dir / "paired_deltas.csv", index=False)
    pd.read_csv(output_dir / "silent_error_transitions.csv").to_csv(output_dir / "transition_table.csv", index=False)

    baseline = messages[messages.method == "baseline"].set_index("config_id")
    integrity = messages[messages.method == "integrity"].set_index("config_id")
    baseline_wrong = baseline.wrong_accept.astype(bool)
    baseline_correct = baseline.correct_accept.astype(bool)
    caught = baseline_wrong & integrity.reject.astype(bool)
    remaining = baseline_wrong & integrity.wrong_accept.astype(bool)
    false_reject = baseline_correct & integrity.reject.astype(bool)
    capture = pd.DataFrame([{
        "baseline_wrong_accept": int(baseline_wrong.sum()),
        "converted_to_reject": int(caught.sum()),
        "remained_wrong_accept": int(remaining.sum()),
        "converted_to_correct_accept": int((baseline_wrong & integrity.correct_accept.astype(bool)).sum()),
        "capture_rate": float(caught.sum() / baseline_wrong.sum()) if baseline_wrong.any() else np.nan,
    }])
    capture.to_csv(output_dir / "silent_error_capture.csv", index=False)
    pd.DataFrame([{
        "baseline_correct_accept": int(baseline_correct.sum()),
        "additional_correct_rejects": int(false_reject.sum()),
        "false_rejection_rate_among_baseline_correct": float(false_reject.sum() / baseline_correct.sum()),
        "additional_correct_rejects_per_prevented_silent": float(false_reject.sum() / caught.sum()) if caught.any() else np.nan,
    }]).to_csv(output_dir / "false_rejection.csv", index=False)

    integrity_wrong = messages[(messages.method == "integrity") & (messages.wrong_accept == 1)]
    config_count = int((messages.method == "integrity").sum())
    replay_count = messages.replay_file.nunique()
    if len(integrity_wrong) == 0:
        config_upper = 1.0 - 0.05 ** (1.0 / config_count)
        replay_upper = 1.0 - 0.05 ** (1.0 / replay_count)
    else:
        config_upper = np.nan
        replay_upper = np.nan
    pd.DataFrame([
        {
            "bound": "one_sided_95pct_exact_binomial_config_independence",
            "events": len(integrity_wrong), "units": config_count,
            "upper_bound": config_upper,
            "dependence_note": "ignores five-seed within-replay dependence",
        },
        {
            "bound": "one_sided_95pct_replay_cluster_event_conservative",
            "events": integrity_wrong.replay_file.nunique(), "units": replay_count,
            "upper_bound": replay_upper,
            "dependence_note": "bounds probability a replay has at least one silent seed event",
        },
    ]).to_csv(output_dir / "zero_event_bounds.csv", index=False)

    baseline_words = words[words.method == "baseline"].copy()
    rank_rows = []
    silent_rows = []
    for _, row in baseline_words.iterrows():
        if row.hard_status != "corrected_single":
            if bool(row.silent_miscorrection):
                silent_rows.append({
                    "config_id": row.config_id, "replay_file": row.replay_file,
                    "beatmap_hash": row.beatmap_hash, "layout_seed": int(row.layout_seed),
                    "codeword_index": int(row.codeword_index), "raw_error_count": int(row.raw_error_count),
                    "syndrome": int(row.syndrome), "overall_parity": int(row.overall_parity),
                    "corrected_bit_index": -1, "corrected_bit_rank": -1,
                    "minimum_abs_correlation": min(json.loads(row.abs_correlations)),
                    "corrected_bit_abs_correlation": np.nan,
                    "integrity_caught": 0,
                })
            continue
        confidence = np.asarray(json.loads(row.abs_correlations), dtype=float)
        position = corrected_position(int(row.syndrome), int(row.overall_parity))
        if position is None:
            raise ValueError("corrected_single bez corrected position.")
        ranks = np.argsort(np.argsort(confidence, kind="stable"), kind="stable")
        category = "silent_corrected_single" if bool(row.silent_miscorrection) else "correctly_repaired_single"
        rank_rows.append({
            "category": category, "corrected_bit_rank": int(ranks[position]),
            "corrected_bit_abs_correlation": float(confidence[position]),
            "minimum_abs_correlation": float(confidence.min()),
        })
        if bool(row.silent_miscorrection):
            integrity_row = words[
                (words.method == "integrity")
                & (words.config_id == row.config_id)
                & (words.codeword_index == row.codeword_index)
            ].iloc[0]
            silent_rows.append({
                "config_id": row.config_id, "replay_file": row.replay_file,
                "beatmap_hash": row.beatmap_hash, "layout_seed": int(row.layout_seed),
                "codeword_index": int(row.codeword_index), "raw_error_count": int(row.raw_error_count),
                "syndrome": int(row.syndrome), "overall_parity": int(row.overall_parity),
                "corrected_bit_index": position, "corrected_bit_rank": int(ranks[position]),
                "minimum_abs_correlation": float(confidence.min()),
                "corrected_bit_abs_correlation": float(confidence[position]),
                "integrity_caught": int(integrity_row.evaluation_outcome == "REJECT"),
            })
    ranks = pd.DataFrame(rank_rows)
    analysis_rows = []
    for category, group in ranks.groupby("category"):
        analysis_rows.append({
            "category": category, "codewords": len(group),
            "rank_mean": float(group.corrected_bit_rank.mean()),
            "rank_median": float(group.corrected_bit_rank.median()),
            "rank_q25": float(group.corrected_bit_rank.quantile(0.25)),
            "rank_q75": float(group.corrected_bit_rank.quantile(0.75)),
            "rank_zero_fraction": float((group.corrected_bit_rank == 0).mean()),
            "rank_counts": json.dumps({str(k): int(v) for k, v in group.corrected_bit_rank.value_counts().sort_index().items()}),
        })
    pd.DataFrame(analysis_rows).to_csv(output_dir / "corrected_bit_rank_analysis.csv", index=False)
    pd.DataFrame(silent_rows).to_csv(output_dir / "validation_silent_codewords.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection", type=Path, default=DEFAULT_OUTPUT_DIR / "validation_selection.csv")
    parser.add_argument("--validation-lock", type=Path, default=DEFAULT_OUTPUT_DIR / "validation_lock.json")
    parser.add_argument("--lock-amendment", type=Path, default=DEFAULT_OUTPUT_DIR / "validation_lock_amendment.json")
    parser.add_argument("--frozen-rule", type=Path, default=DEFAULT_RULE)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--map-offsets", type=Path, default=CONFIG_DIR / "map_time_offsets.json")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--results", type=Path, default=METADATA_DIR / "results_v3_clean.csv")
    parser.add_argument("--performance-groups", type=Path, default=METADATA_DIR / "performance_groups.csv")
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--limit-replays", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.generate_only and args.analyze_only:
        raise ValueError("--generate-only i --analyze-only ne mogu zajedno.")
    if file_sha256(args.frozen_rule) != EXPECTED_RULE_SHA256:
        raise ValueError("Frozen integrity rule hash nije očekivani final-validation hash.")
    lock = json.loads(args.validation_lock.read_text(encoding="utf-8"))
    if lock["frozen_rule_sha256"] != EXPECTED_RULE_SHA256:
        raise ValueError("Validation lock nema očekivani rule hash.")
    locked_runner_hash = lock["source_sha256"][
        "scripts/experiments/run_ecc_integrity_validation.py"
    ]
    current_runner_hash = file_sha256(Path(__file__))
    amendment_hash = None
    if current_runner_hash != locked_runner_hash:
        amendment = json.loads(args.lock_amendment.read_text(encoding="utf-8"))
        if (
            amendment["original_validation_lock_sha256"] != file_sha256(args.validation_lock)
            or amendment["locked_source_sha256"] != locked_runner_hash
            or amendment["amended_source_sha256"] != current_runner_hash
            or amendment["frozen_rule_changed"]
            or amendment["selection_changed"]
            or amendment["main_validation_outcome_rows_before_amendment"] != 0
        ):
            raise ValueError("Validation lock amendment nije konzistentan.")
        amendment_hash = file_sha256(args.lock_amendment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "experiment_version": EXPERIMENT_VERSION,
        "validation_lock_sha256": file_sha256(args.validation_lock),
        "validation_lock_amendment_sha256": amendment_hash,
        "frozen_rule_sha256": file_sha256(args.frozen_rule),
        "selection_sha256": file_sha256(args.selection),
        "policy_sha256": file_sha256(args.policy),
        "partition_sha256": file_sha256(args.partition),
        "map_offsets_sha256": file_sha256(args.map_offsets),
        "cohort_sha256": file_sha256(args.results),
        "N": N_VALUE, "payload_fraction": PAYLOAD_FRACTION,
        "pn_key_id": key_id(PN_KEY),
        "layout_key_ids": [key_id(value) for value in LAYOUT_KEYS],
        "message_seed": MESSAGE_SEED, "limit_replays": args.limit_replays,
        "rule_cli_override_available": False,
        "physical_replay_shared_by_both_decoders": True,
    }
    write_or_validate_config(args.output_dir / "config.json", run_config)
    frozen = json.loads(args.frozen_rule.read_text(encoding="utf-8"))
    if not args.analyze_only:
        generate(args, frozen, run_config)
    if not args.generate_only:
        expected_replays = args.limit_replays or len(pd.read_csv(args.selection))
        evaluate(
            args.output_dir / "physical_provenance.csv",
            args.output_dir / "validation_raw_codewords.csv",
            args.output_dir, frozen, expected_replays,
        )
        add_validation_analyses(args.output_dir)
        print("final validation analysis complete")


if __name__ == "__main__":
    main()
