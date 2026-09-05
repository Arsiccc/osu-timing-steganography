"""Sender-side analytical scoring for reliability-aware PN selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from osu_stego.stego.ecc import decode_hamming_8_4_secded, secded_syndrome
from osu_stego.stego.integrity import corrected_bit_is_not_minimum_decision
from osu_stego.stego.payload_layout import select_payload_blocks
from osu_stego.stego.pn_sequence import generate_pn_sequence


MESSAGE_FAMILIES = ("all_zero", "all_one", "alternating", "random_a", "random_b")


@dataclass(frozen=True)
class CandidateScore:
    nonpositive_count: int
    minimum_margin: float
    mean_margin: float

    def rank_key(self, candidate_id: str) -> tuple[int, float, float, str]:
        """Lower tuple is better; candidate ID gives a deterministic tie-break."""
        return (
            self.nonpositive_count,
            -self.minimum_margin,
            -self.mean_margin,
            candidate_id,
        )


@dataclass(frozen=True)
class IntegrityDecode:
    useful_bits: np.ndarray
    accepted: bool
    rejection_reason: str
    hard_statuses: tuple[str, ...]


def deterministic_message_family(
    family: str,
    useful_length: int,
    replay_file: str,
    beatmap_hash: str,
) -> np.ndarray:
    """Return one preregistered bipolar useful-message family."""
    if family not in MESSAGE_FAMILIES:
        raise ValueError(f"Unknown message family: {family}")
    if useful_length <= 0 or useful_length % 4:
        raise ValueError("Useful length must be positive and divisible by four.")
    if family == "all_zero":
        return np.full(useful_length, -1, dtype=np.int8)
    if family == "all_one":
        return np.ones(useful_length, dtype=np.int8)
    if family == "alternating":
        return np.where(np.arange(useful_length) % 2, 1, -1).astype(np.int8)
    material = "|".join(
        ("host-aware-pn-selection-v1", family, replay_file, beatmap_hash)
    )
    seed = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=useful_length)


def predicted_margins(
    clean_residuals: np.ndarray,
    coded_bits: np.ndarray,
    pn_key: str | int,
    layout_key: str | int,
    alpha: float,
    n_frames_per_bit: int,
) -> np.ndarray:
    """Compute exact ideal pre-write signed margins from sender-known inputs.

    For bit j and clean-valid note positions V_j, this is
    ``b_j * sum(r_i * p_i) + alpha * |V_j|``.  It deliberately excludes
    hit-window, chronology, write, reload, and rematching outcomes.
    """
    residuals = np.asarray(clean_residuals, dtype=np.float64)
    message = np.asarray(coded_bits, dtype=np.int8)
    if message.ndim != 1 or not np.all(np.isin(message, (-1, 1))):
        raise ValueError("Coded bits must be a one-dimensional bipolar array.")
    blocks = select_payload_blocks(
        len(residuals), n_frames_per_bit, len(message), layout_key, "distributed"
    )
    pn = generate_pn_sequence(pn_key, len(message) * n_frames_per_bit)
    margins = np.empty(len(message), dtype=np.float64)
    for bit_index, block_index in enumerate(blocks.tolist()):
        start = block_index * n_frames_per_bit
        block = residuals[start : start + n_frames_per_bit]
        chips = pn[
            bit_index * n_frames_per_bit : (bit_index + 1) * n_frames_per_bit
        ]
        valid = ~np.isnan(block)
        host_correlation = float(np.sum(block[valid] * chips[valid]))
        margins[bit_index] = (
            float(message[bit_index]) * host_correlation
            + float(alpha) * int(np.sum(valid))
        )
    return margins


def summarize_margins(margins: np.ndarray) -> CandidateScore:
    values = np.asarray(margins, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Margins must be a non-empty finite one-dimensional array.")
    return CandidateScore(
        nonpositive_count=int(np.sum(values <= 0.0)),
        minimum_margin=float(np.min(values)),
        mean_margin=float(np.mean(values)),
    )


def select_candidate(scores: list[tuple[str, CandidateScore]]) -> str:
    """Apply the frozen lexicographic selector to a non-empty candidate set."""
    if not scores:
        raise ValueError("At least one PN candidate is required.")
    ids = [candidate_id for candidate_id, _ in scores]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique.")
    return min(scores, key=lambda item: item[1].rank_key(item[0]))[0]


def decode_with_frozen_integrity(
    received_coded_bits: np.ndarray,
    correlations: np.ndarray,
) -> IntegrityDecode:
    """Hard SECDED decode followed by the unchanged frozen integrity rule."""
    received = np.asarray(received_coded_bits, dtype=np.int8)
    scores = np.asarray(correlations, dtype=np.float64)
    if received.shape != scores.shape or len(received) == 0 or len(received) % 8:
        raise ValueError("Received bits/correlations must have equal positive 8k length.")
    useful, statuses = decode_hamming_8_4_secded(received)
    accepted = True
    reason = "accepted"
    for word_index, status_value in enumerate(statuses.tolist()):
        coded_slice = slice(word_index * 8, word_index * 8 + 8)
        syndrome, overall = secded_syndrome(received[coded_slice])
        decision = corrected_bit_is_not_minimum_decision(
            str(status_value), syndrome, overall, np.abs(scores[coded_slice])
        )
        if accepted and not decision.accepted:
            accepted = False
            reason = decision.reason
    return IntegrityDecode(
        useful_bits=useful,
        accepted=accepted,
        rejection_reason=reason,
        hard_statuses=tuple(str(value) for value in statuses.tolist()),
    )
