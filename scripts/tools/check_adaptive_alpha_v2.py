"""Focused checks for canonical integer/rational adaptive-alpha inference."""

from __future__ import annotations

from pathlib import Path

import osrparse
import pandas as pd

from osu_stego.paths import DATASET_DIR, RESULTS_DIR
from osu_stego.stego.adaptive_alpha import (
    alpha_from_accuracy_ratio,
    alpha_from_replay,
    load_sender_local_policy,
)


POLICY_PATH = Path("data/config/adaptive_alpha_sender_local_v2.json")
FROZEN_RESULTS = RESULTS_DIR / "adaptive_layout_strong_v1" / "physical_results.csv"
BOUNDARY_REPLAYS = (
    "d40016cfb4b5d944edeebb6437eb0ddf.osr",
    "47fdd7b758ad17f824cf0d0fcb93cc9e.osr",
)


def replay_path(replay_file: str) -> Path:
    matches = list(DATASET_DIR.glob(f"*/osr/{replay_file}"))
    if len(matches) != 1:
        raise AssertionError(f"Nije pronađen jedinstven replay: {replay_file}")
    return matches[0]


def main() -> None:
    policy = load_sender_local_policy(POLICY_PATH)
    alphas = [float(value) for value in policy["alpha_values"]]
    for index, ratio in enumerate(policy["threshold_ratios"]):
        numerator = int(ratio["numerator"])
        denominator = int(ratio["denominator"])
        below = alpha_from_accuracy_ratio(
            policy, numerator=numerator - 1, denominator=denominator
        )
        exact = alpha_from_accuracy_ratio(
            policy, numerator=numerator, denominator=denominator
        )
        above = alpha_from_accuracy_ratio(
            policy, numerator=numerator + 1, denominator=denominator
        )
        if (below, exact, above) != (alphas[index], alphas[index + 1], alphas[index + 1]):
            raise AssertionError(f"Neispravna semantika oko praga {index}.")

    frozen = pd.read_csv(FROZEN_RESULTS)
    frozen = frozen[
        (frozen["alpha_method"] == "sender_local_adaptive")
        & (frozen["layout"] == "prefix")
    ].set_index("replay_file")
    observed: dict[str, float] = {}
    for replay_file, expected in frozen["alpha"].items():
        replay = osrparse.Replay.from_path(replay_path(replay_file))
        observed[replay_file] = alpha_from_replay(policy, replay)
        if observed[replay_file] != float(expected):
            raise AssertionError(f"V2 menja frozen alpha za {replay_file}.")

    for replay_file in BOUNDARY_REPLAYS:
        path = replay_path(replay_file)
        repeated = [
            alpha_from_replay(policy, osrparse.Replay.from_path(path))
            for _ in range(5)
        ]
        if repeated != [12.0] * 5:
            raise AssertionError(f"Boundary replay nije deterministički alpha=12: {replay_file}")

    print("ADAPTIVE ALPHA V2 CHECK OK")
    print(f"frozen_assignments_unchanged={len(observed)}")
    print("boundary_exact_29_over_31=alpha_12")


if __name__ == "__main__":
    main()
