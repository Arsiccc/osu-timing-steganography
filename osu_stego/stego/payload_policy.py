"""Exact integer payload policies layered over nominal coded-bit capacity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PayloadDecision:
    capacity_class: str
    nominal_capacity_bits: int
    normal_allocated_bits: int
    transmitted_coded_bits: int
    useful_bits: int


def _fractional_allocation(capacity: int, numerator: int, denominator: int) -> int:
    if capacity < 0:
        raise ValueError("Nominal capacity mora biti >= 0.")
    if numerator <= 0 or denominator <= 0 or numerator > denominator:
        raise ValueError("Payload fraction mora biti u intervalu (0, 1].")
    return 0 if capacity == 0 else max(1, (capacity * numerator) // denominator)


def one_codeword_floor_decision(policy: dict[str, Any], num_note_positions: int) -> PayloadDecision:
    """Apply the frozen rational 7.5% + capped one-codeword-floor policy."""
    if num_note_positions < 0:
        raise ValueError("Broj note-index pozicija mora biti >= 0.")
    n_value = int(policy["n_frames_per_bit"])
    codeword_bits = int(policy["secded_codeword_bits"])
    useful_per_word = int(policy["secded_useful_bits_per_codeword"])
    normal = policy["normal_payload_fraction"]
    cap = policy["floor_max_fraction"]
    capacity = num_note_positions // n_value
    allocated = _fractional_allocation(
        capacity, int(normal["numerator"]), int(normal["denominator"])
    )
    complete_words = allocated // codeword_bits
    if complete_words >= 1:
        return PayloadDecision(
            "NORMAL_7_5", capacity, allocated,
            complete_words * codeword_bits, complete_words * useful_per_word,
        )
    floor_permitted = capacity * int(cap["numerator"]) >= codeword_bits * int(cap["denominator"])
    if floor_permitted:
        return PayloadDecision(
            "FLOOR_12_5", capacity, allocated, codeword_bits, useful_per_word
        )
    return PayloadDecision("NO_CAPACITY", capacity, allocated, 0, 0)
