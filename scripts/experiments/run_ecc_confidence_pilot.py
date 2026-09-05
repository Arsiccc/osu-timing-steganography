"""Decoder-only confidence-aware SECDED pilot on frozen development artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import RESULTS_DIR
from osu_stego.stego.ecc import decode_hamming_8_4_one_erasure
from scripts.experiments.adaptive_layout_strong_common import write_or_validate_config
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_ecc_equal_payload import cluster_bootstrap_delta


EXPERIMENT_VERSION = "ecc-confidence-pilot-v1"
SOURCE_DIR = RESULTS_DIR / "ecc_equal_payload_v1"
CONFIDENCE_DIR = RESULTS_DIR / "confidence_analysis_v1"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "ecc_confidence_pilot_v1"


def decode_word(
    row: pd.Series,
    threshold: float | None,
) -> dict:
    true_info = np.asarray(json.loads(row.true_info_bits), dtype=np.int8)
    hard_info = np.asarray(json.loads(row.hard_decoded_info_bits), dtype=np.int8)
    received = np.asarray(json.loads(row.hard_received_bits), dtype=np.int8)
    confidence = np.asarray(json.loads(row.abs_correlations), dtype=np.float64)
    erased = False
    erasure_index = -1
    output = hard_info
    status = str(row.hard_status)
    declared_failure = status == "detected_double"
    if threshold is not None:
        candidate = int(np.argmin(confidence))
        if confidence[candidate] <= threshold:
            erased = True
            erasure_index = candidate
            decoded, erasure_status = decode_hamming_8_4_one_erasure(
                received, candidate
            )
            status = erasure_status
            if decoded is None:
                declared_failure = True
            else:
                output = decoded
                declared_failure = False
    bit_errors = int(np.sum(output != true_info))
    return {
        "erasure_applied": int(erased),
        "erasure_index": erasure_index,
        "erased_abs_correlation": float(confidence[erasure_index]) if erased else np.nan,
        "decoder_status": status,
        "declared_failure": int(declared_failure),
        "post_ecc_bit_errors": bit_errors,
        "ground_truth_info_correct": int(bit_errors == 0),
        "silent_miscorrection": int(bit_errors > 0 and not declared_failure),
        "decoder_output_info_bits": json.dumps(output.tolist()),
    }


def aggregate_configs(decoder_words: pd.DataFrame, physical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (decoder, config_id), group in decoder_words.groupby(["decoder", "config_id"]):
        source = physical.loc[config_id]
        post_errors = int(group.post_ecc_bit_errors.sum())
        declared_failures = int(group.declared_failure.sum())
        rows.append({
            "decoder": decoder, "config_id": config_id,
            "replay_file": source.replay_file, "beatmap_hash": source.beatmap_hash,
            "layout_seed": int(source.layout_seed), "useful_bits": int(source.useful_bits),
            "physical_bits": int(source.physical_bits),
            "post_ecc_bit_errors": post_errors,
            "post_ecc_ber": post_errors / int(source.useful_bits),
            "ground_truth_full_message_correct": int(post_errors == 0),
            "operational_full_message_recovered": int(
                post_errors == 0 and declared_failures == 0
            ),
            "declared_failed_codewords": declared_failures,
            "silent_miscorrected_codewords": int(group.silent_miscorrection.sum()),
            "erased_codewords": int(group.erasure_applied.sum()),
            "codewords": len(group),
        })
    return pd.DataFrame(rows)


def comparison_rows(configs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    for decoder, group in configs.groupby("decoder"):
        summaries.append({
            "decoder": decoder, "configs": len(group),
            "post_ecc_bit_errors": int(group.post_ecc_bit_errors.sum()),
            "useful_bits": int(group.useful_bits.sum()),
            "post_ecc_ber": float(group.post_ecc_bit_errors.sum() / group.useful_bits.sum()),
            "ground_truth_full_message_recovery": float(group.ground_truth_full_message_correct.mean()),
            "operational_full_message_recovery": float(group.operational_full_message_recovered.mean()),
            "declared_failed_codewords": int(group.declared_failed_codewords.sum()),
            "silent_miscorrected_codewords": int(group.silent_miscorrected_codewords.sum()),
            "erased_codewords": int(group.erased_codewords.sum()),
            "physical_bits": int(group.physical_bits.sum()),
            "useful_bits_per_physical_bit": float(group.useful_bits.sum() / group.physical_bits.sum()),
        })
    paired = []
    for decoder in sorted(set(configs.decoder) - {"hard"}):
        subset = configs[configs.decoder.isin(("hard", decoder))].rename(
            columns={"decoder": "condition"}
        )
        for metric in (
            "ground_truth_full_message_correct",
            "operational_full_message_recovered",
        ):
            delta, low, high = cluster_bootstrap_delta(
                subset, decoder, "hard", metric
            )
            paired.append({
                "decoder": decoder, "metric": metric,
                "delta_vs_hard": delta, "ci95_low": low, "ci95_high": high,
                "bootstrap_unit": "replay_file", "iterations": 5000,
            })
    return pd.DataFrame(summaries), pd.DataFrame(paired)


def miscorrection_comparison(words: pd.DataFrame) -> pd.DataFrame:
    hard_silent_ids = set(
        words.loc[
            (words.decoder == "hard") & (words.silent_miscorrection == 1),
            ["config_id", "codeword_index"],
        ].itertuples(index=False, name=None)
    )
    rows = []
    for decoder in sorted(set(words.decoder) - {"hard"}):
        candidate = words[words.decoder == decoder].set_index(
            ["config_id", "codeword_index"]
        )
        counts = {
            "corrected": 0, "detected": 0, "still_miscorrected": 0,
            "not_erased_remained_miscorrected": 0,
        }
        for key in hard_silent_ids:
            row = candidate.loc[key]
            if row.ground_truth_info_correct:
                counts["corrected"] += 1
            elif row.declared_failure:
                counts["detected"] += 1
            elif row.silent_miscorrection:
                if row.erasure_applied:
                    counts["still_miscorrected"] += 1
                else:
                    counts["not_erased_remained_miscorrected"] += 1
        rows.append({"decoder": decoder, **counts, "hard_silent_total": len(hard_silent_ids)})
    return pd.DataFrame(rows)


def status_transition_rows(words: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for decoder in sorted(set(words.decoder) - {"hard"}):
        group = words[words.decoder == decoder]
        for hard_status, subset in group.groupby("hard_status"):
            rows.append({
                "decoder": decoder, "hard_status": hard_status,
                "codewords": len(subset),
                "candidate_correct": int(subset.ground_truth_info_correct.sum()),
                "candidate_declared_failure": int(subset.declared_failure.sum()),
                "candidate_silent_miscorrection": int(subset.silent_miscorrection.sum()),
                "erasure_applied": int(subset.erasure_applied.sum()),
            })
    return pd.DataFrame(rows)


def erasure_theory_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"pattern": "one_erasure_no_unknown_error", "cases": 128, "correct": 128, "ambiguous": 0, "wrong_unique": 0},
        {"pattern": "one_erasure_one_unknown_error", "cases": 896, "correct": 896, "ambiguous": 0, "wrong_unique": 0},
        {"pattern": "two_erasures_no_unknown_error", "cases": 448, "correct": 448, "ambiguous": 0, "wrong_unique": 0},
        {"pattern": "three_erasures_no_unknown_error", "cases": 896, "correct": 896, "ambiguous": 0, "wrong_unique": 0},
        {"pattern": "one_erasure_two_unknown_errors", "cases": 2688, "correct": 0, "ambiguous": 0, "wrong_unique": 2688},
        {"pattern": "four_erasures_no_unknown_error", "cases": 1120, "correct": 896, "ambiguous": 224, "wrong_unique": 0},
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--confidence-dir", type=Path, default=CONFIDENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confidence_config = json.loads(
        (args.confidence_dir / "config.json").read_text(encoding="utf-8")
    )
    thresholds = {
        f"erase_q{int(float(fraction) * 100):02d}": float(value)
        for fraction, value in confidence_config["erasure_candidate_thresholds"].items()
    }
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "source_bits_sha256": file_sha256(args.source_dir / "ecc_bit_results.csv"),
        "source_words_sha256": file_sha256(args.source_dir / "ecc_codeword_results.csv"),
        "source_physical_sha256": file_sha256(args.source_dir / "physical_results.csv"),
        "confidence_config_sha256": file_sha256(args.confidence_dir / "config.json"),
        "thresholds": thresholds,
        "threshold_selection": "development_bit_level_quantiles_only",
        "physical_reembedding": False,
        "erasure_rule": "erase one lowest-|C| code bit iff its |C| <= threshold",
    }
    write_or_validate_config(args.output_dir / "config.json", config)
    source_words = pd.read_csv(args.source_dir / "ecc_codeword_results.csv")
    rows = []
    decoder_thresholds: dict[str, float | None] = {"hard": None, **thresholds}
    for _, row in source_words.iterrows():
        for decoder, threshold in decoder_thresholds.items():
            rows.append({
                "decoder": decoder, "threshold": threshold,
                "config_id": row.config_id, "replay_file": row.replay_file,
                "beatmap_hash": row.beatmap_hash, "layout_seed": int(row.layout_seed),
                "alpha": float(row.alpha), "codeword_index": int(row.codeword_index),
                "raw_error_count": int(row.raw_error_count),
                "hard_status": row.hard_status, "hard_silent_miscorrection": int(row.silent_miscorrection),
                **decode_word(row, threshold),
            })
    decoder_words = pd.DataFrame(rows)
    decoder_words.to_csv(args.output_dir / "decoder_results.csv", index=False)
    physical = pd.read_csv(args.source_dir / "physical_results.csv")
    physical = physical[physical.condition == "ecc"].set_index("config_id")
    configs = aggregate_configs(decoder_words, physical)
    configs.to_csv(args.output_dir / "config_results.csv", index=False)
    summary, paired = comparison_rows(configs)
    summary.to_csv(args.output_dir / "threshold_comparison.csv", index=False)
    paired.to_csv(args.output_dir / "recovery_paired.csv", index=False)
    miscorrection_comparison(decoder_words).to_csv(
        args.output_dir / "miscorrection_comparison.csv", index=False
    )
    status_transition_rows(decoder_words).to_csv(
        args.output_dir / "status_transitions.csv", index=False
    )
    erasure_theory_rows().to_csv(
        args.output_dir / "erasure_theory.csv", index=False
    )
    print("ECC confidence pilot complete")


if __name__ == "__main__":
    main()
