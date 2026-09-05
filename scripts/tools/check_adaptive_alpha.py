"""Sanity checks for the pre-specified adaptive-alpha policy."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from osu_stego.paths import METADATA_DIR
from osu_stego.stego.pn_sequence import generate_pn_sequence
from scripts.experiments.run_adaptive_alpha_comparison import (
    ADAPTIVE_ALPHA,
    PERFORMANCE_ORDER,
    PN_KEY,
    alpha_for_method,
)
from scripts.experiments.run_payload_sweep import deterministic_message


def main() -> None:
    if tuple(inspect.signature(alpha_for_method).parameters) != (
        "method",
        "performance_category",
    ):
        raise AssertionError("Alpha policy prima nedozvoljen target signal.")

    values = [ADAPTIVE_ALPHA[category] for category in PERFORMANCE_ORDER]
    if values != sorted(values, reverse=True):
        raise AssertionError("Adaptive alpha nije monotono opadajući sa kvalitetom.")

    for category, expected in ADAPTIVE_ALPHA.items():
        if alpha_for_method("adaptive_v1", category) != expected:
            raise AssertionError(f"Pogrešan alpha za {category}.")

    metadata = pd.read_csv(METADATA_DIR / "performance_groups.csv")
    applied = metadata["performance_category"].map(ADAPTIVE_ALPHA)
    if applied.isna().any():
        raise AssertionError("Policy ne pokriva sve aktivne replay-eve.")

    message_a = deterministic_message(
        "replay.osr", "map", 10.0, 8, 0.075, 32, 42
    )
    message_b = deterministic_message(
        "replay.osr", "map", 20.0, 8, 0.075, 32, 42
    )
    if not np.array_equal(message_a, message_b):
        raise AssertionError("Promena alpha je promenila skrivenu poruku.")

    pn_a = generate_pn_sequence(PN_KEY, 256)
    pn_b = generate_pn_sequence(PN_KEY, 256)
    if not np.array_equal(pn_a, pn_b):
        raise AssertionError("PN sekvenca nije deterministična.")

    print("ADAPTIVE ALPHA CHECK OK")
    print(f"policy: {ADAPTIVE_ALPHA}")
    print(f"mean alpha: {applied.mean():.4f}")
    print(f"RMS alpha:  {np.sqrt(np.mean(applied**2)):.4f}")


if __name__ == "__main__":
    main()
