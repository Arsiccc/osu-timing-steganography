"""Boundary and legacy-equivalence checks for the frozen floor policy."""

from __future__ import annotations

import json
from pathlib import Path

from osu_stego.stego.payload_policy import one_codeword_floor_decision
from scripts.experiments.run_payload_sweep import message_length_for_fraction


POLICY_PATH = Path("data/config/payload_policy_one_codeword_floor_v1.json")


def main() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    expected = {
        63: ("NO_CAPACITY", 4, 0),
        64: ("FLOOR_12_5", 4, 8),
        106: ("FLOOR_12_5", 7, 8),
        107: ("NORMAL_7_5", 8, 8),
        213: ("NORMAL_7_5", 15, 8),
        214: ("NORMAL_7_5", 16, 16),
    }
    for capacity, values in expected.items():
        decision = one_codeword_floor_decision(policy, capacity * 8)
        observed = (
            decision.capacity_class,
            decision.normal_allocated_bits,
            decision.transmitted_coded_bits,
        )
        if observed != values:
            raise AssertionError((capacity, observed, values))
    for capacity in range(1, 10000):
        decision = one_codeword_floor_decision(policy, capacity * 8)
        legacy = message_length_for_fraction(capacity, 0.075)
        if decision.normal_allocated_bits != legacy:
            raise AssertionError(f"Legacy allocation mismatch at C={capacity}.")
        if legacy >= 8 and decision.transmitted_coded_bits != (legacy // 8) * 8:
            raise AssertionError(f"NORMAL behavior changed at C={capacity}.")
        if decision.capacity_class == "FLOOR_12_5" and decision.transmitted_coded_bits != 8:
            raise AssertionError(f"Floor is not exactly one word at C={capacity}.")
    print("ONE-CODEWORD FLOOR POLICY CHECK OK")
    print("NO_CAPACITY C<=63 | FLOOR_12_5 C=64..106 | NORMAL_7_5 C>=107")


if __name__ == "__main__":
    main()
