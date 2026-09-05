"""Development-frozen PN-correlation confidence calibration and generalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from osu_stego.paths import RESULTS_DIR
from scripts.experiments.adaptive_layout_strong_common import write_or_validate_config
from scripts.experiments.run_adaptive_alpha_validation import file_sha256


EXPERIMENT_VERSION = "confidence-analysis-v1"
DEFAULT_BITS = RESULTS_DIR / "decoder_error_analysis_v1" / "bit_level_results.csv"
DEFAULT_WORDS = RESULTS_DIR / "ecc_equal_payload_v1" / "ecc_codeword_results.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "confidence_analysis_v1"
ERASE_FRACTIONS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
THRESHOLD_CANDIDATE_FRACTIONS = (0.05, 0.10, 0.15, 0.20)


def confidence_bin_rows(
    bits: pd.DataFrame,
    partition: str,
    edges: np.ndarray,
) -> pd.DataFrame:
    frame = bits[bits.partition == partition].copy()
    frame["confidence_bin"] = np.searchsorted(
        edges[1:-1], frame.abs_correlation.to_numpy(float), side="right"
    )
    rows = []
    for bin_index in range(10):
        group = frame[frame.confidence_bin == bin_index]
        rows.append({
            "partition": partition, "bin_index": bin_index,
            "development_threshold_low": float(edges[bin_index]),
            "development_threshold_high": float(edges[bin_index + 1]),
            "bits": len(group), "errors": int(group.is_error.sum()),
            "ber": float(group.is_error.mean()) if len(group) else np.nan,
            "mean_abs_correlation": float(group.abs_correlation.mean()) if len(group) else np.nan,
            "median_abs_correlation": float(group.abs_correlation.median()) if len(group) else np.nan,
            "mean_valid_carriers": float(group.roundtrip_valid_carriers.mean()) if len(group) else np.nan,
            "alpha_counts": json.dumps(
                {str(key): int(value) for key, value in group.alpha.value_counts().sort_index().items()},
                sort_keys=True,
            ),
        })
    return pd.DataFrame(rows)


def coverage_rows(
    bits: pd.DataFrame,
    thresholds: dict[float, float],
) -> pd.DataFrame:
    rows = []
    for partition in ("development", "validation"):
        frame = bits[bits.partition == partition]
        total_errors = int(frame.is_error.sum())
        for fraction in ERASE_FRACTIONS:
            threshold = thresholds[fraction]
            erased = np.zeros(len(frame), dtype=bool) if fraction == 0.0 else (
                frame.abs_correlation.to_numpy(float) <= threshold
            )
            retained = frame[~erased]
            captured = int(frame.loc[erased, "is_error"].sum())
            rows.append({
                "partition": partition, "target_erased_fraction": fraction,
                "development_frozen_threshold": threshold,
                "actual_erased_fraction": float(np.mean(erased)),
                "retained_coverage": float(np.mean(~erased)),
                "retained_bits": len(retained),
                "retained_errors": int(retained.is_error.sum()),
                "retained_ber": float(retained.is_error.mean()),
                "erased_errors": captured,
                "fraction_all_errors_captured": captured / total_errors,
            })
    return pd.DataFrame(rows)


def within_replay_rows(bits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = bits.copy()
    medians = frame.groupby(["partition", "replay_file"])["abs_correlation"].transform("median")
    frame["replay_normalized_confidence"] = frame.abs_correlation / medians.replace(0.0, np.nan)
    rows = []
    for (partition, replay_file), group in frame.groupby(["partition", "replay_file"]):
        correct = group[group.is_error == 0].abs_correlation
        error = group[group.is_error == 1].abs_correlation
        if len(correct) and len(error):
            rows.append({
                "partition": partition, "replay_file": replay_file,
                "bits": len(group), "errors": len(error),
                "median_correct_abs_correlation": float(correct.median()),
                "median_error_abs_correlation": float(error.median()),
                "error_minus_correct_median": float(error.median() - correct.median()),
                "error_median_is_lower": int(error.median() < correct.median()),
            })
    summaries = []
    for partition, group in frame.groupby("partition"):
        valid = np.isfinite(group.replay_normalized_confidence)
        summaries.append({
            "partition": partition, "bits": len(group), "errors": int(group.is_error.sum()),
            "raw_low_confidence_auc": float(roc_auc_score(group.is_error, -group.abs_correlation)),
            "replay_normalized_low_confidence_auc": float(
                roc_auc_score(
                    group.loc[valid, "is_error"],
                    -group.loc[valid, "replay_normalized_confidence"],
                )
            ),
            "eligible_replays_with_correct_and_error_bits": sum(
                row["partition"] == partition for row in rows
            ),
            "fraction_eligible_replays_error_median_lower": float(
                np.mean([
                    row["error_median_is_lower"]
                    for row in rows if row["partition"] == partition
                ])
            ),
        })
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def miscorrection_rows(words: pd.DataFrame) -> pd.DataFrame:
    silent = words[words.silent_miscorrection == 1].copy()
    rows = []
    for raw_errors, group in silent.groupby("raw_error_count"):
        ranks = [
            rank
            for encoded in group.wrong_bit_confidence_ranks
            for rank in json.loads(encoded)
        ]
        rows.append({
            "row_type": "hard_silent_summary", "raw_error_count": int(raw_errors),
            "codewords": len(group), "wrong_physical_bits": len(ranks),
            "wrong_bits_lowest_confidence_count": int(sum(rank == 0 for rank in ranks)),
            "mean_wrong_bit_confidence_rank": float(np.mean(ranks)) if ranks else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=Path, default=DEFAULT_BITS)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bits = pd.read_csv(args.bits)
    development = bits[bits.partition == "development"]
    edges = np.quantile(development.abs_correlation, np.linspace(0.0, 1.0, 11))
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("Development confidence decile pragovi nisu strogo rastući.")
    thresholds = {
        fraction: (-1.0 if fraction == 0.0 else float(
            np.quantile(development.abs_correlation, fraction)
        ))
        for fraction in ERASE_FRACTIONS
    }
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "bits_sha256": file_sha256(args.bits),
        "words_sha256": file_sha256(args.words),
        "threshold_selection_partition": "development_only",
        "development_decile_edges": edges.tolist(),
        "coverage_thresholds": {str(key): value for key, value in thresholds.items()},
        "erasure_candidate_fractions": list(THRESHOLD_CANDIDATE_FRACTIONS),
        "erasure_candidate_thresholds": {
            str(value): thresholds[value] for value in THRESHOLD_CANDIDATE_FRACTIONS
        },
        "replay_normalization": "abs_correlation / replay_median_abs_correlation",
    }
    write_or_validate_config(args.output_dir / "config.json", config)
    confidence_bin_rows(bits, "development", edges).to_csv(
        args.output_dir / "confidence_bins_development.csv", index=False
    )
    confidence_bin_rows(bits, "validation", edges).to_csv(
        args.output_dir / "confidence_bins_validation.csv", index=False
    )
    coverage_rows(bits, thresholds).to_csv(
        args.output_dir / "confidence_coverage_curve.csv", index=False
    )
    within, summary = within_replay_rows(bits)
    within.to_csv(args.output_dir / "within_replay_confidence.csv", index=False)
    summary.to_csv(args.output_dir / "within_replay_summary.csv", index=False)
    miscorrection_rows(pd.read_csv(args.words)).to_csv(
        args.output_dir / "miscorrection_analysis.csv", index=False
    )
    print("confidence analysis complete")


if __name__ == "__main__":
    main()
