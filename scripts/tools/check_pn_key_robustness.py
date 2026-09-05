"""Focused pre-outcome checks for the frozen multi-PN experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from osu_stego.stego.payload_layout import (
    embed_message_with_layout,
    extract_bipolar_message_with_layout,
    select_payload_blocks,
)
from osu_stego.stego.pn_sequence import generate_pn_sequence
from scripts.experiments.adaptive_layout_strong_common import LAYOUT_KEYS, PN_KEY
from scripts.experiments.run_layout_comparison import key_id


CONFIG = Path("data/config/pn_key_robustness_v1.json")


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    keys = [config["historical_reference"]["key"]] + [row["key"] for row in config["new_keys"]]
    assert keys[0] == PN_KEY and len(keys) == 11
    assert len(set(keys)) == len(keys)
    assert len({key_id(value) for value in keys}) == len(keys)
    assert all(np.array_equal(generate_pn_sequence(key, 256), generate_pn_sequence(key, 256)) for key in keys)
    assert len({hashlib.sha256(generate_pn_sequence(key, 256).tobytes()).hexdigest() for key in keys}) == len(keys)

    residuals = np.zeros(1024, dtype=float)
    message = np.asarray([-1, 1, 1, -1, 1, -1, -1, 1], dtype=np.int8)
    blocks = select_payload_blocks(len(residuals), 8, len(message), LAYOUT_KEYS[0], "distributed")
    for key in keys:
        stego = embed_message_with_layout(residuals, message, key, LAYOUT_KEYS[0], 12, 8, "distributed")
        decoded = extract_bipolar_message_with_layout(stego, key, LAYOUT_KEYS[0], 8, len(message), "distributed")
        assert np.array_equal(decoded, message)
        assert np.array_equal(blocks, select_payload_blocks(len(residuals), 8, len(message), LAYOUT_KEYS[0], "distributed"))
    assert not np.array_equal(blocks, select_payload_blocks(len(residuals), 8, len(message), LAYOUT_KEYS[1], "distributed"))
    chips = generate_pn_sequence(keys[1], len(message) * 8)
    assert np.array_equal(chips, generate_pn_sequence(keys[1], len(message) * 8))
    assert not np.array_equal(chips, generate_pn_sequence(keys[2], len(message) * 8))
    print("PN KEY ROBUSTNESS CHECK OK")
    print("keys=11 deterministic distinct noiseless_ber=0 pn_layout_independent=true")


if __name__ == "__main__":
    main()
