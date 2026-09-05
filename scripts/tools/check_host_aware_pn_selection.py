"""Focused pre-holdout sanity checks for host-aware PN selection."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from osu_stego.stego.host_aware_pn import (
    MESSAGE_FAMILIES,
    deterministic_message_family,
    predicted_margins,
    select_candidate,
    summarize_margins,
)
from osu_stego.stego.payload_layout import (
    embed_message_with_layout,
    message_correlations_with_layout,
    select_payload_blocks,
)
from osu_stego.stego.pn_sequence import _derive_seed
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS
from scripts.experiments.run_host_aware_pn_selection import blind_decision


OUTPUT = RESULTS_DIR / "host_aware_pn_selection_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-frozen-files", action="store_true")
    args = parser.parse_args()

    residuals = np.array([2.0, -3.0, np.nan, 1.0, 4.0, -2.0, 0.0, 3.0] * 4)
    coded = np.array([-1, 1, -1, 1], dtype=np.int8)
    pn_key = "host-aware-sanity-pn"
    layout_key = LAYOUT_KEYS[0]
    alpha = 12.0
    margins = predicted_margins(residuals, coded, pn_key, layout_key, alpha, 8)
    embedded = embed_message_with_layout(
        residuals, coded, pn_key, layout_key, alpha, 8, "distributed", quantize=True
    )
    correlations = message_correlations_with_layout(
        embedded, pn_key, layout_key, 8, len(coded), "distributed"
    )
    if not np.array_equal(margins, coded * correlations):
        raise AssertionError("Predicted margin does not equal ideal embedded signed correlation.")
    if not np.array_equal(
        margins, predicted_margins(residuals, coded, pn_key, layout_key, alpha, 8)
    ):
        raise AssertionError("Candidate score is not deterministic.")

    scores = [("b", summarize_margins(np.array([1.0, 2.0]))),
              ("a", summarize_margins(np.array([1.0, 2.0])))]
    if select_candidate(scores) != "a":
        raise AssertionError("PN-ID tie-break is not deterministic.")
    if select_candidate(scores[:1]) != "b":
        raise AssertionError("K=1 does not select its only candidate.")
    blocks_small = select_payload_blocks(256, 8, 4, layout_key, "distributed")
    blocks_large = select_payload_blocks(256, 8, 8, layout_key, "distributed")
    if not np.array_equal(blocks_small, blocks_large[:4]):
        raise AssertionError("Distributed layout nesting changed.")

    messages = [
        deterministic_message_family(name, 8, "r.osr", "map")
        for name in MESSAGE_FAMILIES
    ]
    if any(len(message) != 8 for message in messages):
        raise AssertionError("Message-family length changed.")
    if not np.array_equal(messages[3], deterministic_message_family(
        "random_a", 8, "r.osr", "map"
    )):
        raise AssertionError("Message-family generation is not deterministic.")
    if "selected" in inspect.signature(blind_decision).parameters:
        raise AssertionError("Blind decoder accepts selected-PN identity as input.")

    if args.require_frozen_files:
        pn = json.loads((CONFIG_DIR / "host_aware_pn_candidates_v1.json").read_text())
        candidates = pn["candidates"]
        seeds = [int(value["effective_seed_32"]) for value in candidates]
        if seeds != [_derive_seed(value["key"]) for value in candidates]:
            raise AssertionError("Stored effective PN seed mismatch.")
        if len(seeds) != 8 or len(set(seeds)) != 8:
            raise AssertionError("New effective PN seeds are not eight distinct values.")
        for k_value in (1, 2, 4, 8):
            if pn["nested_candidate_subsets"][str(k_value)] != list(range(k_value)):
                raise AssertionError("Candidate subsets are not exact nested prefixes.")
        holdout = pd.read_csv(OUTPUT / "holdout_selection.csv")
        design = pd.read_csv(RESULTS_DIR / "pn_key_robustness_phase2_v1" / "replay_panel.csv")
        design_ids = set(design.loc[design["eligible_7_5"] == 1, "replay_file"])
        if set(holdout["replay_file"]) & design_ids:
            raise AssertionError("Holdout overlaps the eligible design panel.")
        if len(holdout) != 100 or holdout["replay_file"].duplicated().any():
            raise AssertionError("Holdout must contain 100 unique replays.")
    print("host-aware PN sanity checks: PASS")


if __name__ == "__main__":
    main()
