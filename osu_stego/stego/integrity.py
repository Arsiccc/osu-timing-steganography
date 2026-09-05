"""Observable-only accept/reject decisions layered on hard SECDED decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntegrityDecision:
    accepted: bool
    reason: str


def corrected_position(syndrome: int, overall_parity: int) -> int | None:
    """Return the bit position changed by SECDED, or None if none was changed."""
    if overall_parity != 1:
        return None
    return syndrome - 1 if syndrome else 7


def baseline_integrity_decision(hard_status: str) -> IntegrityDecision:
    """Reject only SECDED's algebraically detected uncorrectable words."""
    if hard_status == "detected_double":
        return IntegrityDecision(False, "secded_detected_uncorrectable")
    if hard_status not in ("clean", "corrected_single"):
        raise ValueError(f"Nepoznat hard SECDED status: {hard_status}")
    return IntegrityDecision(True, "accepted")


def multiple_low_bits_decision(
    hard_status: str,
    abs_correlations: np.ndarray,
    threshold: float,
) -> IntegrityDecision:
    """Reject corrected words with evidence for at least two weak bits."""
    baseline = baseline_integrity_decision(hard_status)
    if not baseline.accepted:
        return baseline
    confidence = _validated_confidence(abs_correlations)
    if hard_status == "corrected_single" and int(np.sum(confidence <= threshold)) >= 2:
        return IntegrityDecision(False, "confidence_multiple_low_bits")
    return baseline


def corrected_bit_is_not_minimum_decision(
    hard_status: str,
    syndrome: int,
    overall_parity: int,
    abs_correlations: np.ndarray,
) -> IntegrityDecision:
    """Reject when SECDED corrects a bit that is not the weakest observed bit."""
    baseline = baseline_integrity_decision(hard_status)
    if not baseline.accepted:
        return baseline
    confidence = _validated_confidence(abs_correlations)
    if hard_status != "corrected_single":
        return baseline
    position = corrected_position(syndrome, overall_parity)
    if position is None:
        raise ValueError("corrected_single mora imati observable corrected position.")
    if confidence[position] > np.min(confidence):
        return IntegrityDecision(False, "confidence_corrected_bit_not_minimum")
    return baseline


def message_integrity_decision(
    codeword_decisions: list[IntegrityDecision],
) -> IntegrityDecision:
    """Reject the entire message if any codeword is rejected."""
    if not codeword_decisions:
        raise ValueError("Poruka mora sadržati bar jedan SECDED codeword.")
    for decision in codeword_decisions:
        if not decision.accepted:
            return IntegrityDecision(False, decision.reason)
    return IntegrityDecision(True, "accepted")


def _validated_confidence(abs_correlations: np.ndarray) -> np.ndarray:
    confidence = np.asarray(abs_correlations, dtype=np.float64)
    if confidence.shape != (8,) or not np.all(np.isfinite(confidence)):
        raise ValueError("Integrity odluka očekuje osam konačnih |C| vrednosti.")
    if np.any(confidence < 0.0):
        raise ValueError("|C| ne sme biti negativan.")
    return confidence
