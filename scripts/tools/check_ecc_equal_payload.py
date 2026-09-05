"""Focused integrity checks for the frozen equal-payload ECC pilot artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from osu_stego.paths import RESULTS_DIR
from scripts.experiments.run_adaptive_alpha_validation import file_sha256
from scripts.experiments.run_ecc_equal_payload import CONDITIONS, identifier


OUTPUT_DIR = RESULTS_DIR / "ecc_equal_payload_v1"
SOURCE_DIR = RESULTS_DIR / "ecc_pilot_v1"


def main() -> None:
    config = json.loads((OUTPUT_DIR / "config.json").read_text(encoding="utf-8"))
    physical = pd.read_csv(OUTPUT_DIR / "physical_results.csv")
    bits = pd.read_csv(OUTPUT_DIR / "ecc_bit_results.csv")
    words = pd.read_csv(OUTPUT_DIR / "ecc_codeword_results.csv")

    assert config["source_ecc_results_sha256"] == file_sha256(
        SOURCE_DIR / "physical_results.csv"
    ), "Frozen source ECC results changed after paired regeneration."
    assert len(physical) == 80 * 5 * len(CONDITIONS)
    assert physical.config_id.is_unique
    assert bits.config_id.notna().all() and words.config_id.notna().all()
    assert not bits.duplicated(["config_id", "bit_index"]).any()
    assert not words.duplicated(["config_id", "codeword_index"]).any()

    pairs = physical.pivot(
        index=["replay_file", "layout_seed"], columns="condition"
    )
    assert len(pairs) == 80 * 5
    for field in ("alpha", "N", "payload_fraction", "pn_key_id", "replay_sha256"):
        values = pairs[field]
        assert (values.nunique(axis=1) == 1).all(), field
    assert (pairs["useful_message_sha256"].nunique(axis=1) == 1).all()
    assert (
        pairs["layout_key_id"].nunique(axis=1) == 1
    ).all(), "Layout provenance differs inside a paired configuration."

    ecc = physical[physical.condition == "ecc"]
    equal_useful = physical[physical.condition == "uncoded_equal_useful"]
    equal_physical = physical[physical.condition == "uncoded_equal_physical"]
    assert (ecc.physical_bits == 2 * ecc.useful_bits).all()
    assert (equal_useful.physical_bits == equal_useful.useful_bits).all()
    assert (equal_physical.physical_bits == equal_physical.useful_bits).all()
    assert set(ecc.useful_bits.unique()) == {4, 8, 12}
    assert (ecc.nominal_payload_bits // 8 * 4 == ecc.useful_bits).all()
    assert (ecc.nominal_payload_bits - ecc.physical_bits).between(0, 7).all()
    assert (bits.groupby("config_id").size().reindex(ecc.config_id).to_numpy() == ecc.physical_bits).all()
    assert (words.groupby("config_id").size().reindex(ecc.config_id).to_numpy() == ecc.codewords).all()

    sample = ecc.iloc[0]
    message = np.array([1, -1, 1, -1], dtype=np.int8)
    first = identifier(
        "ecc", sample.replay_file, sample.replay_sha256,
        int(sample.layout_seed), message, config["policy_sha256"],
    )
    assert first == identifier(
        "ecc", sample.replay_file, sample.replay_sha256,
        int(sample.layout_seed), message.copy(), config["policy_sha256"],
    )
    assert first != identifier(
        "uncoded_equal_useful", sample.replay_file, sample.replay_sha256,
        int(sample.layout_seed), message, config["policy_sha256"],
    )
    assert first != identifier(
        "ecc", sample.replay_file, sample.replay_sha256,
        int(sample.layout_seed), -message, config["policy_sha256"],
    )

    correlations = np.array([5.0, 1.0, 1.0, 3.0])
    order_a = np.argsort(np.abs(correlations), kind="stable")
    order_b = np.argsort(np.abs(correlations.copy()), kind="stable")
    assert np.array_equal(order_a, order_b)
    assert order_a.tolist() == [1, 2, 3, 0]
    print("ECC EQUAL-PAYLOAD INTEGRITY CHECK OK")
    print(f"physical_rows={len(physical)} bit_rows={len(bits)} codewords={len(words)}")


if __name__ == "__main__":
    main()
