"""Regression check for the frozen exact-zero PN-correlation decision."""

from __future__ import annotations

import numpy as np

from osu_stego.stego.decoder import (
    extract_bipolar_message,
    extract_bipolar_message_indexed,
)
from osu_stego.stego.payload_layout import extract_bipolar_message_with_layout


def main() -> None:
    zeros = np.zeros(16, dtype=np.float64)
    indexed = zeros.copy()
    indexed[::3] = np.nan
    assert np.array_equal(extract_bipolar_message(zeros, "tie", 8, 2), [1, 1])
    assert np.array_equal(extract_bipolar_message_indexed(indexed, "tie", 8, 2), [1, 1])
    assert np.array_equal(
        extract_bipolar_message_with_layout(zeros, "tie", "layout", 8, 2, "distributed"),
        [1, 1],
    )
    print("DECODER ZERO-TIE CHECK OK: exact zero -> +1")


if __name__ == "__main__":
    main()
