"""Apply transparent post-lock corrections to message-content reporting only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_pilot_ber_sweep import stable_seed


VERSION = "message-content-robustness-v1"
OUTPUT = RESULTS_DIR / "message_content_robustness_v1"
SOURCE = RESULTS_DIR / "host_aware_pn_selection_v1"
ITERATIONS = 2000


def complement_interval() -> tuple[float, float, float]:
    known = pd.read_csv(SOURCE / "index_known_results.csv")
    pair = known[
        (known.K.astype(int) == 1)
        & known.message_family.isin(("all_zero", "all_one"))
    ]
    totals = pair.groupby(["replay_file", "message_family"], as_index=False).agg(
        errors=("raw_bit_errors", "sum"), bits=("coded_bits", "sum")
    )
    observed = totals.groupby("message_family").agg(
        errors=("errors", "sum"), bits=("bits", "sum")
    )
    delta = (
        observed.loc["all_one", "errors"] / observed.loc["all_one", "bits"]
        - observed.loc["all_zero", "errors"] / observed.loc["all_zero", "bits"]
    )
    replay_ids = np.asarray(sorted(totals.replay_file.unique()))
    by_replay = {name: group for name, group in totals.groupby("replay_file")}
    rng = np.random.default_rng(stable_seed(20260903, VERSION, "all-one-minus-all-zero"))
    samples = np.empty(ITERATIONS)
    for iteration in range(ITERATIONS):
        selected = rng.choice(replay_ids, size=len(replay_ids), replace=True)
        errors = {"all_zero": 0, "all_one": 0}
        bits = {"all_zero": 0, "all_one": 0}
        for replay_id in selected:
            for row in by_replay[replay_id].itertuples(index=False):
                errors[row.message_family] += int(row.errors)
                bits[row.message_family] += int(row.bits)
        samples[iteration] = (
            errors["all_one"] / bits["all_one"]
            - errors["all_zero"] / bits["all_zero"]
        )
    return delta, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    lock = json.loads((OUTPUT / "experiment_lock.json").read_text())
    locked_analysis_hash = lock["source_code_sha256"]["analysis"]
    symmetry_path = OUTPUT / "complement_symmetry.csv"
    symmetry = pd.read_csv(symmetry_path)
    if not np.isclose(symmetry.coded_hamming_fraction_weighted.iloc[0], 1.0):
        raise RuntimeError("Correction expects exact coded complementarity.")
    delta, low, high = complement_interval()
    if not np.isclose(delta, symmetry.raw_ber_difference.iloc[0], atol=1e-15):
        raise RuntimeError("Complement BER delta does not match frozen summary.")
    symmetry["raw_ber_difference_ci95_low"] = low
    symmetry["raw_ber_difference_ci95_high"] = high
    symmetry["bootstrap_unit"] = "replay_file"
    symmetry["bootstrap_iterations"] = ITERATIONS
    symmetry.to_csv(symmetry_path, index=False)

    report_path = OUTPUT / "message_content_report.md"
    report = report_path.read_text(encoding="utf-8")
    old = (
        "SECDED codewords differ in 100.0% of coded positions, so they are not exact complements. "
        "All-one minus all-zero BER is -1.362%;"
    )
    new = (
        "SECDED codewords differ in 100.0% of coded positions and therefore are exact coded-bit complements. "
        f"All-one minus all-zero BER is {delta:+.3%}, with replay-cluster 95% CI "
        f"[{low:+.3%}, {high:+.3%}];"
    )
    if old not in report and new not in report:
        raise RuntimeError("Expected complement sentence is missing from report.")
    report = report.replace(old, new)
    report = report.replace(
        "No confirmed implementation, provenance, reuse, or leakage bug was found. Short-message duplicate payloads are a real design feature and are explicitly marked. Chronology-drop sign is unavailable in the frozen artifacts.",
        "Three reporting/plumbing bugs were confirmed and corrected transparently: pandas inferred short binary strings as integers; an unnecessary pooled merge created suffixed coded-hash columns; and the first report misinterpreted 100% Hamming distance as non-complementarity. The first two stopped before any outcome file was written and are recorded in the lock; the third changed wording and added the prespecified paired complement CI, not the underlying measurements. No provenance, reuse, physical-data, or leakage bug was found. Short-message duplicates are explicitly marked, and chronology-drop sign remains unavailable."
    )
    report = report.replace(
        "The manifest contains 500 replay/message payloads. 30 within-replay family pairs have identical useful or coded payloads; they retain both provenance labels, while pooled descriptive/detector analyses deduplicate identical coded payloads.",
        "The manifest contains 500 replay/message payloads. Thirty within-replay family pairs on 25 replays have identical useful and coded payloads: 29 pairs occur at four useful bits and one at 12 bits. No coded-only collision exists because the SECDED mapping is injective. Both provenance labels are retained; pooled descriptive/detector analyses reduce 1,500 nominal conditions to 1,416 unique replay/layout/coded-payload conditions."
    )
    report = report.replace(
        "Absolute message positions and within-codeword positions are preserved in `coded_position_ber.csv`; they were not used for tuning or compaction.",
        "Within-codeword BER was highest at positions 3, 1, and 2 (8.93%, 8.80%, and 8.12%) and lowest at position 0 (4.40%), with 2,340 observations at every position. Absolute-position BER is also preserved in `coded_position_ber.csv`, but later positions have smaller denominators because payload lengths differ. These patterns were not used for tuning or compaction."
    )
    report = report.replace(
        "Operational rejection is kept separate from ground-truth correctness. Confidence thresholds are unchanged and no message-specific rule was fitted.",
        "Operational rejection is kept separate from ground-truth correctness. Across messages, correct/error median |C| ranges were 105–106 and 23–40; low-|C| error-prediction AUC was 0.797–0.864 and lowest-confidence-decile BER 26.3–41.0%. Confidence remains useful for every family. Thresholds are unchanged and no message-specific rule was fitted."
    )
    report = report.replace(
        "Per-message active-carrier survival, hit-window and chronology drops, matcher status changes, squared energy, RMS, and mean absolute shift are in `physical_by_message.csv`. Matcher-unmatched events are not called osu! misses.",
        "Physical behavior was nearly invariant: mean active-carrier fraction ranged 99.47–99.62%, hit-window drops 0.050–0.063/config, chronology drops 0.273–0.400/config, matcher-status changes 2.087–2.187/config, and squared energy 24,745–24,786 ms²; RMS and mean absolute applied shift were 15.28 ms for every family. Matcher-unmatched events are not called osu! misses."
    )
    report = report.replace(
        "All 15 message/layout cells are reported. Within-replay message BER range has median 4.167%, mean 6.479%, and 90th percentile 16.667%.",
        "All 15 message/layout cells are reported. All-one BER was below all-zero BER on each layout; all-zero was the highest-BER family on layouts 0 and 2, but not layout 1. Natural length strata contain 67 one-codeword, 10 two-codeword, and 23 three-codeword replays. Within-replay message BER range has median 4.167%, mean 6.479%, and 90th percentile 16.667%; 18/100 replays had zero range and 19/100 had at least 10 pp range."
    )
    report = report.replace(
        "Map/message estimates are reported for 13 maps having at least five replays; the two-replay map is omitted from map-level interpretation.",
        "Map/message estimates are reported for 13 maps having at least five replays; the two-replay map is omitted from map-level interpretation. Within-map family spread ranged 0.52–7.14 pp (median 2.60 pp), and all-one beat all-zero on 11/13 reported maps, so the complement effect is not confined to one map."
    )
    report_path.write_text(report, encoding="utf-8")

    correction = {
        "version": VERSION,
        "kind": "post-lock reporting correction",
        "locked_analysis_sha256": locked_analysis_hash,
        "correction_script_sha256": file_sha256(Path(__file__)),
        "changes": [
            "Correct interpretation of coded Hamming fraction 1.0: exact complement, not non-complement.",
            "Add the prespecified replay-cluster interval for all_one minus all_zero raw BER.",
            "Document two pre-outcome plumbing failures and this reporting error.",
            "Expand prespecified duplicate, confidence, physical, position, layout, length, replay, and map findings from already generated tables.",
        ],
        "physical_data_changed": False,
        "model_parameters_changed": False,
        "outcome_selection_changed": False,
        "all_one_minus_all_zero_raw_ber": delta,
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_unit": "replay_file",
        "bootstrap_iterations": ITERATIONS,
    }
    (OUTPUT / "post_lock_corrections.json").write_text(
        json.dumps(correction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"complement correction: delta={delta:.6f}, "
        f"ci=[{low:.6f}, {high:.6f}]"
    )


if __name__ == "__main__":
    main()
