"""Final provenance and semantic checks for integrity validation v1."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import CONFIG_DIR, RESULTS_DIR
from osu_stego.stego.ecc import decode_hamming_8_4_secded
from osu_stego.stego.integrity import corrected_bit_is_not_minimum_decision
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_ecc_integrity_validation import (
    EXPECTED_RULE_SHA256,
    validation_identifier,
)


OUTPUT_DIR = RESULTS_DIR / "ecc_integrity_validation_v1"


def main() -> None:
    rule_path = CONFIG_DIR / "ecc_integrity_rejection_v1.json"
    lock_path = OUTPUT_DIR / "validation_lock.json"
    amendment_path = OUTPUT_DIR / "validation_lock_amendment.json"
    assert file_sha256(rule_path) == EXPECTED_RULE_SHA256
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    config = json.loads((OUTPUT_DIR / "config.json").read_text(encoding="utf-8"))
    assert lock["frozen_rule_sha256"] == EXPECTED_RULE_SHA256
    assert amendment["original_validation_lock_sha256"] == file_sha256(lock_path)
    assert amendment["amended_source_sha256"] == file_sha256(
        Path("scripts/experiments/run_ecc_integrity_validation.py")
    )
    assert config["validation_lock_amendment_sha256"] == file_sha256(amendment_path)
    assert config["rule_cli_override_available"] is False
    assert config["physical_replay_shared_by_both_decoders"] is True
    assert lock["threshold_search"] is False
    assert lock["ground_truth_used_for_operational_decision"] is False
    assert "true" not in inspect.signature(
        corrected_bit_is_not_minimum_decision
    ).parameters

    selection = pd.read_csv(OUTPUT_DIR / "validation_selection.csv")
    design = pd.read_csv(RESULTS_DIR / "ecc_pilot_v1" / "pilot_selection.csv")
    holdout = pd.read_csv(RESULTS_DIR / "ecc_integrity_rejection_v1" / "holdout_selection.csv")
    development_maps = set(design.beatmap_hash) | set(holdout.beatmap_hash)
    assert len(selection) == 167 and selection.replay_file.is_unique
    assert selection.beatmap_hash.nunique() == 5
    assert not set(selection.replay_file) & (set(design.replay_file) | set(holdout.replay_file))
    assert not set(selection.beatmap_hash) & development_maps

    physical = pd.read_csv(OUTPUT_DIR / "physical_provenance.csv")
    raw_words = pd.read_csv(OUTPUT_DIR / "validation_raw_codewords.csv")
    codewords = pd.read_csv(OUTPUT_DIR / "codeword_results.csv")
    messages = pd.read_csv(OUTPUT_DIR / "message_results.csv")
    assert len(physical) == 167 * 5 and physical.config_id.is_unique
    assert physical.groupby("replay_file").layout_seed.nunique().eq(5).all()
    assert physical.groupby("replay_file").layout_key_id.nunique().eq(5).all()
    assert physical.groupby("replay_file").alpha.nunique().eq(1).all()
    assert physical.groupby("replay_file").message_sha256.nunique().eq(1).all()
    assert physical.pn_key_id.nunique() == 1
    assert physical.frozen_rule_sha256.eq(EXPECTED_RULE_SHA256).all()
    assert not raw_words.duplicated(["config_id", "codeword_index"]).any()
    counts = raw_words.groupby("config_id").size().reindex(physical.config_id)
    assert np.array_equal(counts.to_numpy(), physical.codewords.to_numpy())
    assert not codewords.duplicated(["method", "config_id", "codeword_index"]).any()
    assert not messages.duplicated(["method", "config_id"]).any()
    assert messages.groupby("config_id").method.nunique().eq(2).all()
    assert messages[["correct_accept", "wrong_accept", "reject"]].sum(axis=1).eq(1).all()

    for row in raw_words.itertuples(index=False):
        received = np.asarray(json.loads(row.hard_received_bits), dtype=np.int8)
        decoded, statuses = decode_hamming_8_4_secded(received)
        assert statuses.tolist() == [row.hard_status]
        assert decoded.tolist() == json.loads(row.hard_decoded_info_bits)
    paired_words = codewords.pivot(
        index=["config_id", "codeword_index"], columns="method",
        values="hard_decoded_info_bits",
    )
    assert (paired_words.baseline == paired_words.integrity).all()

    sample = physical.iloc[0]
    encoded = np.asarray(
        json.loads(raw_words[raw_words.config_id == sample.config_id].iloc[0].encoded_bits),
        dtype=np.int8,
    )
    assert sample.config_id == validation_identifier(
        sample.replay_file, sample.replay_sha256, int(sample.layout_seed), encoded,
        sample.policy_sha256, sample.frozen_rule_sha256,
    )
    summary = pd.read_csv(OUTPUT_DIR / "summary.csv").set_index("method")
    assert tuple(summary.loc["baseline", ["correct_accept", "wrong_accept", "reject"]]) == (767, 14, 54)
    assert tuple(summary.loc["integrity", ["correct_accept", "wrong_accept", "reject"]]) == (716, 1, 118)
    transitions = pd.read_csv(OUTPUT_DIR / "transition_table.csv").set_index(["baseline", "integrity"])
    assert int(transitions.loc[("CORRECT_ACCEPT", "WRONG_ACCEPT"), "configs"]) == 0
    assert int(transitions.loc[("REJECT", "WRONG_ACCEPT"), "configs"]) == 0
    print("ECC INTEGRITY FINAL VALIDATION CHECK OK")
    print("eligible_replays=167 configs=835 baseline_wrong=14 integrity_wrong=1")


if __name__ == "__main__":
    main()
