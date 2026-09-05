"""Sanity checks for keyed payload placement.

Run:
    python3 -m scripts.tools.check_payload_layout
"""

from __future__ import annotations

import numpy as np

from osu_stego.stego.encoder import embed_message_indexed
from osu_stego.stego.decoder import extract_bipolar_message_indexed
from osu_stego.stego.payload_layout import (
    embed_message_with_layout,
    extract_bipolar_message_with_layout,
    select_payload_blocks,
)


def main() -> None:
    pn_key = "pn-check-key"
    layout_key = "layout-check-key"
    alpha = 14.0
    n_value = 8

    rng = np.random.default_rng(12345)
    residuals = rng.integers(
        -25,
        26,
        size=800,
    ).astype(np.float64)

    message = rng.choice(
        np.array([-1, 1], dtype=np.int8),
        size=10,
    )

    # 1) New prefix implementation must be exactly equivalent to the old one.
    old_prefix = embed_message_indexed(
        residuals,
        message,
        pn_key,
        alpha,
        n_value,
        quantize=True,
    )
    new_prefix = embed_message_with_layout(
        residuals,
        message,
        pn_key,
        layout_key,
        alpha,
        n_value,
        layout="prefix",
        quantize=True,
    )

    if not np.array_equal(old_prefix, new_prefix):
        raise AssertionError(
            "BUG: novi prefix embedding nije identičan starom encoderu."
        )

    old_decoded = extract_bipolar_message_indexed(
        old_prefix,
        pn_key,
        n_value,
        len(message),
    )
    new_decoded = extract_bipolar_message_with_layout(
        new_prefix,
        pn_key,
        layout_key,
        n_value,
        len(message),
        layout="prefix",
    )

    if not np.array_equal(old_decoded, new_decoded):
        raise AssertionError(
            "BUG: novi prefix decoder nije identičan starom decoderu."
        )

    # 2) Distributed layout invariants.
    blocks_a = select_payload_blocks(
        num_notes=len(residuals),
        n_frames_per_bit=n_value,
        num_bits=len(message),
        layout_key=layout_key,
        layout="distributed",
    )
    blocks_b = select_payload_blocks(
        num_notes=len(residuals),
        n_frames_per_bit=n_value,
        num_bits=len(message),
        layout_key=layout_key,
        layout="distributed",
    )

    if not np.array_equal(blocks_a, blocks_b):
        raise AssertionError("BUG: distributed layout nije determinističan.")

    if len(np.unique(blocks_a)) != len(blocks_a):
        raise AssertionError("BUG: distributed blokovi se preklapaju.")

    capacity = len(residuals) // n_value
    if np.any(blocks_a < 0) or np.any(blocks_a >= capacity):
        raise AssertionError("BUG: distributed blok van kapaciteta.")

    # 2b) Payload levels must be nested for a fair payload sweep.
    smaller = select_payload_blocks(
        num_notes=len(residuals),
        n_frames_per_bit=n_value,
        num_bits=5,
        layout_key=layout_key,
        layout="distributed",
    )
    if not np.array_equal(smaller, blocks_a[:5]):
        raise AssertionError(
            "BUG: manji distributed payload nije prefiks većeg layout-a."
        )

    # 3) Noiseless distributed round-trip must be exact.
    zero_host = np.zeros(800, dtype=np.float64)

    distributed = embed_message_with_layout(
        zero_host,
        message,
        pn_key,
        layout_key,
        alpha,
        n_value,
        layout="distributed",
        quantize=True,
    )
    decoded = extract_bipolar_message_with_layout(
        distributed,
        pn_key,
        layout_key,
        n_value,
        len(message),
        layout="distributed",
    )

    if not np.array_equal(message, decoded):
        raise AssertionError(
            "BUG: distributed noiseless round-trip nije 0 BER."
        )

    # 4) Missing carriers keep note alignment in both directions.
    residuals_with_nan = zero_host.copy()
    residuals_with_nan[[3, 17, 89, 301, 650]] = np.nan
    distributed_nan = embed_message_with_layout(
        residuals_with_nan,
        message,
        pn_key,
        layout_key,
        alpha,
        n_value,
        layout="distributed",
        quantize=True,
    )
    decoded_nan = extract_bipolar_message_with_layout(
        distributed_nan,
        pn_key,
        layout_key,
        n_value,
        len(message),
        layout="distributed",
    )
    if not np.array_equal(message, decoded_nan):
        raise AssertionError(
            "BUG: distributed NaN round-trip nije očuvao note poravnanje."
        )

    # 5) PN and layout keys must be independently controllable.
    other_layout_key = "other-layout-key"
    other_blocks = select_payload_blocks(
        num_notes=len(residuals),
        n_frames_per_bit=n_value,
        num_bits=len(message),
        layout_key=other_layout_key,
        layout="distributed",
    )
    if np.array_equal(blocks_a, other_blocks):
        raise AssertionError("BUG: promena layout ključa nije promenila blokove.")

    first_signal = embed_message_with_layout(
        zero_host,
        message,
        pn_key,
        layout_key,
        alpha,
        n_value,
        layout="distributed",
    )
    second_signal = embed_message_with_layout(
        zero_host,
        message,
        pn_key,
        other_layout_key,
        alpha,
        n_value,
        layout="distributed",
    )

    def placed_chips(signal: np.ndarray, blocks: np.ndarray) -> np.ndarray:
        chips: list[np.ndarray] = []
        for bit_index, block_index in enumerate(blocks.tolist()):
            start = block_index * n_value
            end = start + n_value
            chips.append(
                signal[start:end]
                / (alpha * float(message[bit_index]))
            )
        return np.concatenate(chips)

    first_chips = placed_chips(first_signal, blocks_a)
    second_chips = placed_chips(second_signal, other_blocks)
    if not np.array_equal(first_chips, second_chips):
        raise AssertionError("BUG: layout ključ je promenio PN chip sekvencu.")

    print("PAYLOAD LAYOUT CHECK OK")
    print(f"capacity blocks: {capacity}")
    print(f"message bits:    {len(message)}")
    print(f"prefix blocks:   {list(range(len(message)))}")
    print(f"distributed:     {blocks_a.tolist()}")
    temporal = np.sort(blocks_a)
    print(
        "temporal span:   "
        f"{temporal[0] / capacity:.1%} .. "
        f"{(temporal[-1] + 1) / capacity:.1%}"
    )


if __name__ == "__main__":
    main()
