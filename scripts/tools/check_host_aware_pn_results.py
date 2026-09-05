"""Integrity and completeness audit for host-aware PN-selection results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.analysis.timing_features import FULL_FEATURES
from osu_stego.paths import RESULTS_DIR


OUTPUT = RESULTS_DIR / "host_aware_pn_selection_v1"


def unique(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    if frame.duplicated(columns).any():
        raise AssertionError(f"Duplicate {name} identity: {columns}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    lock = json.loads((output / "experiment_lock.json").read_text())
    scores = pd.read_csv(output / "candidate_scores.csv")
    physical = pd.read_csv(output / "physical_results.csv")
    bits = pd.read_csv(output / "bit_results.csv")
    features = pd.read_csv(output / "timing_features.csv")
    known = pd.read_csv(output / "index_known_results.csv")
    blind = pd.read_csv(output / "blind_decoder_results.csv")
    blind_diag = pd.read_csv(output / "blind_candidate_diagnostics.csv")
    units = pd.read_csv(output / "completed_units.csv")

    unique(units, ["unit_id"], "unit")
    unique(scores, ["unit_id", "pn_key_id"], "candidate score")
    unique(physical, ["physical_id"], "physical")
    unique(bits, ["physical_id", "bit_index"], "bit")
    unique(features, ["physical_id", "label"], "feature")
    unique(known, ["condition_id"], "index-known condition")
    unique(blind, ["condition_id"], "blind condition")
    unique(blind_diag, ["condition_id", "hypothesis_pn_id"], "blind hypothesis")

    expected_units = int(lock["expected_units"])
    if not args.allow_partial and len(units) != expected_units:
        raise AssertionError(f"Expected {expected_units} units, found {len(units)}")
    if len(scores) != len(units) * 8:
        raise AssertionError("Every unit must contain eight candidate scores.")
    if len(known) != len(units) * 4 or len(blind) != len(units) * 4:
        raise AssertionError("Every unit must contain four K conditions.")
    expected_hypotheses = int(sum((1, 2, 4, 8)) * len(units))
    if len(blind_diag) != expected_hypotheses:
        raise AssertionError("Blind candidate hypothesis count mismatch.")
    if len(features) != 2 * len(physical):
        raise AssertionError("Every physical run must have one clean/stego feature pair.")
    if len(bits) != int(physical["coded_bits"].sum()):
        raise AssertionError("Bit-result count does not match physical coded-bit total.")
    if set(known["physical_id"]) - set(physical["physical_id"]):
        raise AssertionError("Index-known result references a missing physical run.")
    if set(blind["physical_id"]) - set(physical["physical_id"]):
        raise AssertionError("Blind result references a missing physical run.")
    if not np.all(known["selected_pn_index"].astype(int) < known["K"].astype(int)):
        raise AssertionError("Selected PN lies outside its nested K subset.")
    if not np.all(known.loc[known["K"] == 1, "selected_pn_index"].astype(int) == 0):
        raise AssertionError("K=1 does not always select candidate zero.")
    per_condition = blind_diag.groupby("condition_id").size()
    expected = blind.set_index("condition_id")["K"].astype(int)
    if not per_condition.sort_index().equals(expected.sort_index()):
        raise AssertionError("Blind decoder did not evaluate exactly K hypotheses.")
    if not set(blind["outcome"]).issubset({"CORRECT_ACCEPT", "WRONG_ACCEPT", "REJECT"}):
        raise AssertionError("Unknown blind outcome.")
    if np.any(~np.isfinite(features[list(FULL_FEATURES)].to_numpy(dtype=float))):
        raise AssertionError("Feature values contain NaN/inf.")
    print(
        f"host-aware result audit: PASS units={len(units)} physical={len(physical)} "
        f"bits={len(bits)} known={len(known)} blind_hypotheses={len(blind_diag)}"
    )


if __name__ == "__main__":
    main()
