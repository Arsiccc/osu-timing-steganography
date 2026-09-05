"""Focused tests for receiver-only SECDED integrity decisions and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from osu_stego.stego.integrity import (
    IntegrityDecision,
    baseline_integrity_decision,
    corrected_bit_is_not_minimum_decision,
    message_integrity_decision,
    multiple_low_bits_decision,
)
from scripts.experiments.run_adaptive_alpha_validation import file_sha256


OUTPUT_DIR = RESULTS_DIR / "ecc_integrity_rejection_v1"
CONFIG_PATH = CONFIG_DIR / "ecc_integrity_rejection_v1.json"


def synthetic_checks() -> None:
    confidence = np.asarray([5, 10, 20, 30, 40, 50, 60, 70], dtype=float)
    assert baseline_integrity_decision("clean").accepted
    assert baseline_integrity_decision("corrected_single").accepted
    assert not baseline_integrity_decision("detected_double").accepted
    assert not multiple_low_bits_decision(
        "corrected_single", confidence, 10.0
    ).accepted
    assert multiple_low_bits_decision("corrected_single", confidence, 5.0).accepted
    assert multiple_low_bits_decision("clean", confidence, 70.0).accepted
    assert corrected_bit_is_not_minimum_decision(
        "corrected_single", 1, 1, confidence
    ).accepted
    assert not corrected_bit_is_not_minimum_decision(
        "corrected_single", 2, 1, confidence
    ).accepted
    before = confidence.copy()
    corrected_bit_is_not_minimum_decision("corrected_single", 2, 1, confidence)
    assert np.array_equal(before, confidence), "Integrity odluka ne sme menjati input."
    assert message_integrity_decision([
        IntegrityDecision(True, "accepted"), IntegrityDecision(True, "accepted")
    ]).accepted
    assert not message_integrity_decision([
        IntegrityDecision(True, "accepted"), IntegrityDecision(False, "reason")
    ]).accepted


def artifact_checks() -> None:
    if not CONFIG_PATH.is_file():
        return
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mirror = json.loads(
        (OUTPUT_DIR / "frozen_rule.json").read_text(encoding="utf-8")
    )
    assert config == mirror
    assert config["selection_partition"] == "development_design_only"
    assert config["holdout_selection_created_before_physical_evaluation"] is True
    selection = pd.read_csv(OUTPUT_DIR / "holdout_selection.csv")
    design = pd.read_csv(RESULTS_DIR / "ecc_pilot_v1" / "pilot_selection.csv")
    assert len(selection) == 80 and selection.replay_file.is_unique
    assert not set(selection.replay_file) & set(design.replay_file)
    assert file_sha256(OUTPUT_DIR / "holdout_selection.csv") == config[
        "holdout_selection_sha256"
    ]
    candidates = pd.read_csv(OUTPUT_DIR / "design_rule_candidates.csv")
    assert int(candidates.frozen_selected.sum()) == 1
    r0 = candidates[candidates.rule_id == "R0_hard_operational"].iloc[0]
    r1 = candidates[candidates.rule_id == "R1_explicit_status_control"].iloc[0]
    assert (r0.correct_accept, r0.wrong_accept, r0.reject) == (
        r1.correct_accept, r1.wrong_accept, r1.reject
    ) == (353, 15, 32)
    assert len(pd.read_csv(OUTPUT_DIR / "design_silent_miscorrections.csv")) == 17
    assert len(pd.read_csv(OUTPUT_DIR / "design_silent_miscorrection_bits.csv")) == 136
    physical_path = OUTPUT_DIR / "holdout_physical_results.csv"
    if not physical_path.is_file():
        return
    run_config = json.loads(
        (OUTPUT_DIR / "holdout_config.json").read_text(encoding="utf-8")
    )
    assert run_config["frozen_rule_sha256"] == file_sha256(CONFIG_PATH)
    assert run_config["holdout_selection_sha256"] == file_sha256(
        OUTPUT_DIR / "holdout_selection.csv"
    )
    assert run_config["physical_replay_shared_by_all_decoder_rules"] is True
    physical = pd.read_csv(physical_path)
    words = pd.read_csv(OUTPUT_DIR / "holdout_raw_codewords.csv")
    messages = pd.read_csv(OUTPUT_DIR / "message_results.csv")
    assert len(physical) == 400 and physical.config_id.is_unique
    assert physical.groupby("replay_file").layout_seed.nunique().eq(5).all()
    assert physical.groupby("replay_file").alpha.nunique().eq(1).all()
    assert physical.groupby("replay_file").message_sha256.nunique().eq(1).all()
    assert physical.groupby("replay_file").encoded_message_sha256.nunique().eq(1).all()
    assert physical.groupby("replay_file").layout_key_id.nunique().eq(5).all()
    assert physical.pn_key_id.nunique() == 1
    assert physical.frozen_rule_sha256.nunique() == 1
    assert physical.frozen_rule_sha256.iloc[0] == file_sha256(CONFIG_PATH)
    assert not words.duplicated(["config_id", "codeword_index"]).any()
    word_counts = words.groupby("config_id").size().reindex(physical.config_id)
    assert np.array_equal(word_counts.to_numpy(), physical.codewords.to_numpy())
    assert len(messages) == 800
    assert not messages.duplicated(["method", "config_id"]).any()
    assert messages.groupby("config_id").method.nunique().eq(2).all()
    assert messages[["correct_accept", "wrong_accept", "reject"]].sum(axis=1).eq(1).all()
    summary = pd.read_csv(OUTPUT_DIR / "message_summary.csv").set_index("method")
    assert tuple(summary.loc["baseline", ["correct_accept", "wrong_accept", "reject"]]) == (
        367, 6, 27
    )
    assert tuple(summary.loc["integrity", ["correct_accept", "wrong_accept", "reject"]]) == (
        338, 0, 62
    )
    transitions = pd.read_csv(OUTPUT_DIR / "silent_error_transitions.csv").set_index(
        ["baseline", "integrity"]
    )
    assert int(transitions.loc[("WRONG_ACCEPT", "REJECT"), "configs"]) == 6
    assert int(transitions.loc[("WRONG_ACCEPT", "WRONG_ACCEPT"), "configs"]) == 0


def main() -> None:
    synthetic_checks()
    artifact_checks()
    print("ECC INTEGRITY CHECK OK")
    if CONFIG_PATH.is_file():
        print(f"frozen_config_sha256={file_sha256(CONFIG_PATH)}")


if __name__ == "__main__":
    main()
