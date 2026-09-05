"""Small deterministic block codes for controlled channel pilots."""

from __future__ import annotations

import itertools

import numpy as np


def secded_syndrome(code_bipolar: np.ndarray) -> tuple[int, int]:
    """Return Hamming syndrome and overall parity for one received word."""
    code = np.asarray(code_bipolar, dtype=np.int8)
    if code.shape != (8,) or not np.all(np.isin(code, (-1, 1))):
        raise ValueError("SECDED syndrome očekuje jedan bipolarni 8-bitni word.")
    word = (code > 0).astype(np.int8)
    syndrome = int(
        (word[0] ^ word[2] ^ word[4] ^ word[6])
        + 2 * (word[1] ^ word[2] ^ word[5] ^ word[6])
        + 4 * (word[3] ^ word[4] ^ word[5] ^ word[6])
    )
    return syndrome, int(np.bitwise_xor.reduce(word))


def encode_hamming_8_4_secded(info_bipolar: np.ndarray) -> np.ndarray:
    """Encode bipolar information bits with extended Hamming SECDED (8, 4)."""
    info = np.asarray(info_bipolar, dtype=np.int8)
    if info.ndim != 1 or len(info) % 4:
        raise ValueError("SECDED (8,4) zahteva 1D broj info bitova deljiv sa 4.")
    if not np.all(np.isin(info, (-1, 1))):
        raise ValueError("Info bitovi moraju biti bipolarni.")
    binary = (info > 0).astype(np.int8).reshape(-1, 4)
    code = np.zeros((len(binary), 8), dtype=np.int8)
    code[:, [2, 4, 5, 6]] = binary
    code[:, 0] = code[:, 2] ^ code[:, 4] ^ code[:, 6]
    code[:, 1] = code[:, 2] ^ code[:, 5] ^ code[:, 6]
    code[:, 3] = code[:, 4] ^ code[:, 5] ^ code[:, 6]
    code[:, 7] = np.bitwise_xor.reduce(code[:, :7], axis=1)
    return np.where(code.reshape(-1) == 1, 1, -1).astype(np.int8)


def decode_hamming_8_4_secded(
    code_bipolar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode SECDED words and return bipolar info plus per-word status.

    Status values are ``clean``, ``corrected_single`` and
    ``detected_double``. Three-or-more errors are not reliably detectable.
    """
    code = np.asarray(code_bipolar, dtype=np.int8)
    if code.ndim != 1 or len(code) % 8:
        raise ValueError("SECDED (8,4) zahteva 1D broj code bitova deljiv sa 8.")
    if not np.all(np.isin(code, (-1, 1))):
        raise ValueError("Code bitovi moraju biti bipolarni.")
    binary = (code > 0).astype(np.int8).reshape(-1, 8).copy()
    statuses: list[str] = []
    for word in binary:
        syndrome, overall = secded_syndrome(
            np.where(word == 1, 1, -1).astype(np.int8)
        )
        if syndrome == 0 and overall == 0:
            statuses.append("clean")
        elif overall == 1:
            word[syndrome - 1 if syndrome else 7] ^= 1
            statuses.append("corrected_single")
        else:
            statuses.append("detected_double")
    info = binary[:, [2, 4, 5, 6]].reshape(-1)
    return np.where(info == 1, 1, -1).astype(np.int8), np.asarray(statuses)


def decode_hamming_8_4_with_erasures(
    received_bipolar: np.ndarray,
    erasure_indices: tuple[int, ...],
    *,
    max_unknown_errors: int,
) -> tuple[np.ndarray | None, str]:
    """Decode by exhaustive unique-codeword search over known positions.

    The caller must explicitly state the supported number of additional
    unknown hard errors. A result is returned only when exactly one codeword is
    within that distance on non-erased positions.
    """
    received = np.asarray(received_bipolar, dtype=np.int8)
    if received.shape != (8,) or not np.all(np.isin(received, (-1, 1))):
        raise ValueError("Erasure SECDED očekuje jedan bipolarni 8-bitni word.")
    erasures = tuple(sorted(set(int(value) for value in erasure_indices)))
    if not erasures or any(value < 0 or value >= 8 for value in erasures):
        raise ValueError("Erasure indeksi moraju biti neprazan podskup [0, 7].")
    if max_unknown_errors < 0:
        raise ValueError("max_unknown_errors mora biti >= 0.")
    known = np.ones(8, dtype=bool)
    known[list(erasures)] = False
    candidates: list[tuple[np.ndarray, int]] = []
    for values in itertools.product((-1, 1), repeat=4):
        info = np.asarray(values, dtype=np.int8)
        code = encode_hamming_8_4_secded(info)
        distance = int(np.sum(code[known] != received[known]))
        if distance <= max_unknown_errors:
            candidates.append((info, distance))
    if len(candidates) != 1:
        return None, "erasure_unrecoverable"
    info, distance = candidates[0]
    return info, (
        "erasures_recovered" if distance == 0 else "erasures_plus_error_corrected"
    )


def decode_hamming_8_4_one_erasure(
    received_bipolar: np.ndarray,
    erasure_index: int,
) -> tuple[np.ndarray | None, str]:
    """Correct one erasure and at most one additional unknown error."""
    return decode_hamming_8_4_with_erasures(
        received_bipolar,
        (erasure_index,),
        max_unknown_errors=1,
    )
