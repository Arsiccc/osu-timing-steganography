"""Audit completed message-content robustness summaries and hashes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256


OUTPUT = RESULTS_DIR / "message_content_robustness_v1"
SOURCE = RESULTS_DIR / "host_aware_pn_selection_v1"


def main() -> None:
    lock = json.loads((OUTPUT / "experiment_lock.json").read_text())
    for name, expected in lock["source_artifact_sha256"].items():
        if file_sha256(SOURCE / name) != expected:
            raise AssertionError(f"Frozen source changed: {name}")

    known = pd.read_csv(SOURCE / "index_known_results.csv")
    k1 = known[known.K.astype(int) == 1]
    reliability = pd.read_csv(OUTPUT / "reliability_by_message.csv")
    recomputed = k1.groupby("message_family").agg(
        errors=("raw_bit_errors", "sum"), bits=("coded_bits", "sum")
    )
    recomputed["ber"] = recomputed.errors / recomputed.bits
    reported = reliability.set_index("message_family")
    for family, row in recomputed.iterrows():
        if int(row.errors) != int(reported.loc[family, "raw_bit_errors"]):
            raise AssertionError(f"Raw-error mismatch: {family}")
        if int(row.bits) != int(reported.loc[family, "coded_bits_total"]):
            raise AssertionError(f"Coded-bit mismatch: {family}")
        if not np.isclose(row.ber, reported.loc[family, "weighted_raw_ber"], atol=1e-15):
            raise AssertionError(f"Weighted BER mismatch: {family}")

    bits = pd.read_csv(SOURCE / "bit_results.csv").merge(
        k1[["physical_id"]], on="physical_id", validate="many_to_one"
    )
    bit_values = pd.read_csv(OUTPUT / "bit_value_asymmetry.csv")
    if int(bit_values.observations.sum()) != len(bits):
        raise AssertionError("Bit-value denominator mismatch.")
    if int(bit_values.errors.sum()) != int(bits.is_error.sum()):
        raise AssertionError("Bit-value error mismatch.")
    if int(reliability.coded_bits_total.sum()) != len(bits):
        raise AssertionError("Primary coded-bit denominator mismatch.")

    detectors = pd.read_csv(OUTPUT / "detector_by_message.csv")
    expected_cells = 5 * 2 * 3
    if len(detectors) != expected_cells or detectors.roc_auc.isna().any():
        raise AssertionError("Incomplete per-message detector matrix.")
    if not np.all(detectors.rows == 600) or not np.all(detectors.replays == 100):
        raise AssertionError("Detector datasets differ across messages/models.")
    if set(detectors.classifier) != {"random_forest", "rbf_svm"}:
        raise AssertionError("Unexpected detector family.")
    if set(detectors.feature_set) != {"BASELINE-3", "POSITION", "FULL-29"}:
        raise AssertionError("Unexpected detector feature set.")

    diagnostics = pd.read_csv(OUTPUT / "diagnostics.csv")
    mismatches = diagnostics[diagnostics.value.astype(str) != diagnostics.expected.astype(str)]
    if len(mismatches):
        raise AssertionError(f"Diagnostic mismatch:\n{mismatches}")
    chronology = pd.read_csv(OUTPUT / "chronology_sign_analysis.csv")
    if chronology.status.iloc[0] != "UNAVAILABLE" or chronology.new_physical_run_performed.iloc[0] != 0:
        raise AssertionError("Chronology-sign limitation not preserved.")

    required = (
        "source_artifact_audit.md", "experiment_lock.json",
        "message_payload_manifest.csv", "duplicate_message_audit.csv",
        "reliability_by_message.csv", "paired_message_deltas.csv",
        "bit_value_asymmetry.csv", "coded_position_ber.csv",
        "codeword_outcomes_by_message.csv", "confidence_by_message.csv",
        "physical_by_message.csv", "chronology_sign_analysis.csv",
        "reliability_by_layout_message.csv", "replay_message_sensitivity.csv",
        "map_message_sensitivity.csv", "length_strata.csv",
        "detector_by_message.csv", "pooled_message_detector.csv",
        "diagnostics.csv", "message_content_report.md",
    )
    missing = [name for name in required if not (OUTPUT / name).exists()]
    if missing:
        raise AssertionError(f"Missing outputs: {missing}")
    print("message-content completed-results audit: PASS")


if __name__ == "__main__":
    main()
