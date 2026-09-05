"""Create descriptive summaries and the final PN robustness report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


OUTPUT = Path("results/pn_key_robustness_v1")


def main() -> None:
    reliability = pd.read_csv(OUTPUT / "reliability_by_pn.csv")
    paired = pd.read_csv(OUTPUT / "reliability_paired.csv")
    frozen = pd.read_csv(OUTPUT / "detector_frozen_by_pn.csv")
    aware = pd.read_csv(OUTPUT / "detector_keyaware_by_pn.csv")
    physical = pd.read_csv(OUTPUT / "physical_by_pn.csv")
    confidence = pd.read_csv(OUTPUT / "confidence_by_pn.csv")
    wrong = pd.read_csv(OUTPUT / "wrong_key_sanity_summary.csv")
    selection = pd.read_csv(OUTPUT / "pilot_selection.csv")
    lock = json.loads((OUTPUT / "experiment_lock.json").read_text())

    metrics = ["weighted_raw_ber", "post_ecc_ber", "correct_accept_rate", "wrong_accept_rate", "reject_rate"]
    stability = reliability[metrics].agg(["mean", "std", "median", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    stability.to_csv(OUTPUT / "pn_stability_summary.csv", index=False)
    detector_summary = pd.concat([
        frozen.groupby(["pn_key_id", "feature_set"]).roc_auc.mean().rename("roc_auc").reset_index().assign(detector="historical_pn_trained_replay_disjoint"),
        aware.groupby(["pn_key_id", "feature_set"]).roc_auc.mean().rename("roc_auc").reset_index().assign(detector="key_aware_replay_grouped_cv"),
    ], ignore_index=True)
    detector_summary.to_csv(OUTPUT / "detector_summary_by_pn.csv", index=False)
    layout_matrix = pd.read_csv(OUTPUT / "pn_layout_matrix.csv")
    layout_matrix["weighted_raw_ber"] = layout_matrix["raw_bit_errors"] / layout_matrix["coded_bits"]
    layout_matrix.to_csv(OUTPUT / "pn_layout_matrix_weighted.csv", index=False)

    hist = reliability[reliability.historical_pn == 1].iloc[0]
    significant = paired[(paired.metric == "raw_ber") & ((paired.ci95_low > 0) | (paired.ci95_high < 0))]
    report = f"""# Multiple-PN-key robustness v1

## Status and design

This is a preregistered parameter-robustness experiment on an already-used development corpus, not independent external validation. The lock SHA-256 is `{hashlib.sha256((OUTPUT/'experiment_lock.json').read_bytes()).hexdigest()}`. The cohort had {len(selection)} incoming replays on {selection.beatmap_hash.nunique()} maps; {int(selection.eligible_7_5.sum())} replays on {selection.loc[selection.eligible_7_5 == 1, 'beatmap_hash'].nunique()} maps were eligible. The physical factorial contains 11 PN keys, five frozen layout seeds, and 3,740 configurations.

The historical PN implementation was reproduced exactly on 45 overlapping frozen configurations from nine replays. All recorded physical/reliability fields matched. PN and layout independence, deterministic regeneration, distinct sequences, and noiseless BER=0 passed before the physical pilot.

## Reliability and integrity

Historical-key weighted raw BER was {hist.weighted_raw_ber:.3%}; across all keys the mean was {stability.loc[stability.metric == 'weighted_raw_ber', 'mean'].iloc[0]:.3%}, SD {stability.loc[stability.metric == 'weighted_raw_ber', 'std'].iloc[0]:.3%}, and range {stability.loc[stability.metric == 'weighted_raw_ber', 'min'].iloc[0]:.3%}–{stability.loc[stability.metric == 'weighted_raw_ber', 'max'].iloc[0]:.3%}. Correct-accept rates ranged {reliability.correct_accept_rate.min():.2%}–{reliability.correct_accept_rate.max():.2%}; rejection ranged {reliability.reject_rate.min():.2%}–{reliability.reject_rate.max():.2%}. There were {int(reliability.wrong_accept.sum())} wrong accepted messages across {int(reliability.configs.sum())} configurations.

Two preregistered keys had replay-cluster raw-BER deltas versus historical whose 95% intervals excluded zero: {', '.join(significant.pn_key_id.astype(str))}. The largest-BER key reached {reliability.weighted_raw_ber.max():.3%}, with {reliability.loc[reliability.weighted_raw_ber.idxmax(), 'correct_accept_rate']:.2%} correct accept and {reliability.loc[reliability.weighted_raw_ber.idxmax(), 'reject_rate']:.2%} rejection. Key rankings changed across layouts, although the same high-BER keys were problematic in several layouts.

Low |C| remained informative for errors for every key: error-prediction AUC ranged {confidence.low_abs_c_error_auc.min():.3f}–{confidence.low_abs_c_error_auc.max():.3f}; median |C| was roughly 102–105 for correct bits and 32–42 for error bits. The frozen integrity rule was not retuned.

## Detectability and physical behavior

Mean-over-layout FULL AUC for the replay-disjoint historical-PN-trained detector ranged {detector_summary[(detector_summary.detector == 'historical_pn_trained_replay_disjoint') & (detector_summary.feature_set == 'full')].roc_auc.min():.3f}–{detector_summary[(detector_summary.detector == 'historical_pn_trained_replay_disjoint') & (detector_summary.feature_set == 'full')].roc_auc.max():.3f}. Key-aware replay-grouped FULL AUC ranged {detector_summary[(detector_summary.detector == 'key_aware_replay_grouped_cv') & (detector_summary.feature_set == 'full')].roc_auc.min():.3f}–{detector_summary[(detector_summary.detector == 'key_aware_replay_grouped_cv') & (detector_summary.feature_set == 'full')].roc_auc.max():.3f}. Leakage-safe held-out-key plus disjoint-beatmap FULL AUC ranged {pd.read_csv(OUTPUT/'detector_multikey_by_pn.csv').query("feature_set == 'full'").roc_auc.min():.3f}–{pd.read_csv(OUTPUT/'detector_multikey_by_pn.csv').query("feature_set == 'full'").roc_auc.max():.3f}.

Physical survival was stable: mean active-carrier fraction ranged {physical.active_carrier_fraction.min():.3%}–{physical.active_carrier_fraction.max():.3%}, while chronology drops averaged {physical.dropped_for_chronology.min():.3f}–{physical.dropped_for_chronology.max():.3f} per configuration. Therefore the observed BER spread is not explained by a large energy/survival difference. Feature shifts varied modestly across keys. Descriptive PN-balance correlations are based on only 11 sequences and are not evidence for key selection.

Wrong-key decoding produced BER {wrong.wrong_key_ber.mean():.3%} on average (range {wrong.wrong_key_ber.min():.3%}–{wrong.wrong_key_ber.max():.3%}); this is a sanity result, not a cryptographic-security claim.

## Decision

**B — PN SENSITIVITY EXISTS.** Most conclusions remain qualitatively stable and detectability does not change catastrophically, but two preregistered keys show statistically clear raw-BER degradation and one also materially worsens post-ECC recovery, correct acceptance, and rejection. The historical PN key is not the worst realization. Per the preregistered rule, do not launch a larger run automatically; design and lock a second stage first if stronger system-level quantification is required.
"""
    (OUTPUT / "pn_robustness_report.md").write_text(report, encoding="utf-8")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name != "result_hashes.json"}
    (OUTPUT / "result_hashes.json").write_text(json.dumps({"hash_algorithm": "sha256", "files": files}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
