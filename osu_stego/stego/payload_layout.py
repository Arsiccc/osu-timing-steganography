"""Keyed payload placement for note-indexed spread-spectrum embedding.

This module does not replace the existing encoder/decoder. It adds a layout
layer so the same message and PN sequence can be placed either:

- "prefix": existing behaviour, first B*N note positions
- "distributed": B non-overlapping N-note blocks spread across the full replay

For a fair comparison, the PN sequence is identical for both layouts.
Only physical block locations change.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np

from osu_stego.stego.pn_sequence import generate_pn_sequence


PayloadLayout = Literal["prefix", "distributed"]


def _layout_seed(layout_key: str | int) -> int:
    """Derive a deterministic RNG seed from the independent layout key."""
    material = f"{layout_key}|payload-layout-v1".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def select_payload_blocks(
    num_notes: int,
    n_frames_per_bit: int,
    num_bits: int,
    layout_key: str | int,
    layout: PayloadLayout,
) -> np.ndarray:
    """Return one non-overlapping N-note block index for each message bit.

    Block k corresponds to note positions:
        [k*N, (k+1)*N)

    "prefix"
        Returns blocks 0, 1, ..., num_bits-1.

    "distributed"
        Builds one key-determined permutation of every available N-note block
        across the full replay and uses the first `num_bits` entries.

        This gives an important experimental invariant: for the same replay,
        layout key and N, a smaller payload is a strict subset of a larger
        payload.
        Only the number of selected blocks changes.

    The decoder needs the same num_notes, N, num_bits, layout key and layout.
    """
    if num_notes < 0:
        raise ValueError("num_notes mora biti >= 0.")
    if n_frames_per_bit <= 0:
        raise ValueError("n_frames_per_bit mora biti > 0.")
    if num_bits < 0:
        raise ValueError("num_bits mora biti >= 0.")
    if layout not in ("prefix", "distributed"):
        raise ValueError(
            f"Nepoznat layout '{layout}'. Očekivano: prefix ili distributed."
        )

    capacity_blocks = num_notes // n_frames_per_bit

    if num_bits > capacity_blocks:
        raise ValueError(
            f"Poruka zahteva {num_bits} blokova, "
            f"a dostupno je {capacity_blocks}."
        )

    if num_bits == 0:
        return np.empty(0, dtype=np.int64)

    if layout == "prefix":
        return np.arange(num_bits, dtype=np.int64)

    rng = np.random.default_rng(_layout_seed(layout_key))

    # Generate the same full block permutation regardless of payload length.
    # Taking a prefix makes payload levels nested and directly comparable.
    permutation = rng.permutation(capacity_blocks).astype(
        np.int64,
        copy=False,
    )

    return permutation[:num_bits]


def embed_message_with_layout(
    note_residuals: np.ndarray,
    message_bipolar: np.ndarray,
    pn_key: str | int,
    layout_key: str | int,
    alpha: float,
    n_frames_per_bit: int,
    layout: PayloadLayout,
    quantize: bool = True,
) -> np.ndarray:
    """Embed a bipolar message using prefix or distributed block placement."""
    residuals = np.asarray(note_residuals, dtype=np.float64)
    message = np.asarray(message_bipolar, dtype=np.int8)

    block_indices = select_payload_blocks(
        num_notes=len(residuals),
        n_frames_per_bit=n_frames_per_bit,
        num_bits=len(message),
        layout_key=layout_key,
        layout=layout,
    )

    required_chips = len(message) * n_frames_per_bit
    if required_chips == 0:
        return residuals.copy()

    pn = generate_pn_sequence(
        pn_key,
        required_chips,
    ).astype(np.float64)

    stego = residuals.copy()

    for bit_index, block_index in enumerate(block_indices.tolist()):
        note_start = block_index * n_frames_per_bit
        note_end = note_start + n_frames_per_bit

        pn_start = bit_index * n_frames_per_bit
        pn_end = pn_start + n_frames_per_bit

        block = residuals[note_start:note_end]
        valid = ~np.isnan(block)

        modified = block.copy()
        modified[valid] = (
            block[valid]
            + alpha
            * pn[pn_start:pn_end][valid]
            * float(message[bit_index])
        )

        if quantize:
            modified[valid] = np.round(modified[valid])

        stego[note_start:note_end] = modified

    return stego


def message_correlations_with_layout(
    stego_note_residuals: np.ndarray,
    pn_key: str | int,
    layout_key: str | int,
    n_frames_per_bit: int,
    num_bits: int,
    layout: PayloadLayout,
) -> np.ndarray:
    """Return one PN correlation score per message bit."""
    residuals = np.asarray(
        stego_note_residuals,
        dtype=np.float64,
    )

    block_indices = select_payload_blocks(
        num_notes=len(residuals),
        n_frames_per_bit=n_frames_per_bit,
        num_bits=num_bits,
        layout_key=layout_key,
        layout=layout,
    )

    required_chips = num_bits * n_frames_per_bit
    if required_chips == 0:
        return np.empty(0, dtype=np.float64)

    pn = generate_pn_sequence(
        pn_key,
        required_chips,
    ).astype(np.float64)

    correlations = np.zeros(
        num_bits,
        dtype=np.float64,
    )

    for bit_index, block_index in enumerate(block_indices.tolist()):
        note_start = block_index * n_frames_per_bit
        note_end = note_start + n_frames_per_bit

        pn_start = bit_index * n_frames_per_bit
        pn_end = pn_start + n_frames_per_bit

        block = residuals[note_start:note_end]
        block_pn = pn[pn_start:pn_end]
        valid = ~np.isnan(block)

        correlations[bit_index] = np.sum(
            block[valid] * block_pn[valid]
        )

    return correlations


def extract_bipolar_message_with_layout(
    stego_note_residuals: np.ndarray,
    pn_key: str | int,
    layout_key: str | int,
    n_frames_per_bit: int,
    num_bits: int,
    layout: PayloadLayout,
) -> np.ndarray:
    """Decode a bipolar message from the selected layout blocks."""
    correlations = message_correlations_with_layout(
        stego_note_residuals=stego_note_residuals,
        pn_key=pn_key,
        layout_key=layout_key,
        n_frames_per_bit=n_frames_per_bit,
        num_bits=num_bits,
        layout=layout,
    )

    return np.where(
        correlations >= 0,
        1,
        -1,
    ).astype(np.int8)
