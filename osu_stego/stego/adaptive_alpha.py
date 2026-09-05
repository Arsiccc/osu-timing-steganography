"""Sender-local adaptive-alpha policy helpers.

Only values stored in one replay header are accepted as inference inputs.  The
policy never receives beatmap notes, timing residuals, population ranks, BER, or
steganalysis outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import osrparse


ACCURACY_FORMULA = (
    "(300*count_300 + 100*count_100 + 50*count_50) / "
    "(300*(count_300 + count_100 + count_50 + count_miss))"
)
ALLOWED_POLICY_FEATURES = ("accuracy", "miss_fraction")


def accuracy_ratio_from_counts(
    count_300: int,
    count_100: int,
    count_50: int,
    count_miss: int,
) -> tuple[int, int]:
    """Return the exact osu!standard accuracy numerator and denominator."""
    counts = tuple(
        int(value) for value in (count_300, count_100, count_50, count_miss)
    )
    if any(value < 0 for value in counts):
        raise ValueError("Replay hit counts moraju biti nenegativni.")
    total = sum(counts)
    if total <= 0:
        raise ValueError("Replay nema nijedan judgement; quality nije definisan.")
    numerator = 300 * counts[0] + 100 * counts[1] + 50 * counts[2]
    denominator = 300 * total
    return numerator, denominator


def replay_local_quality_from_counts(
    count_300: int,
    count_100: int,
    count_50: int,
    count_miss: int,
) -> dict[str, float | int]:
    counts = tuple(int(value) for value in (count_300, count_100, count_50, count_miss))
    numerator, denominator = accuracy_ratio_from_counts(*counts)
    total = sum(counts)
    accuracy = numerator / denominator
    return {
        "count_300": counts[0],
        "count_100": counts[1],
        "count_50": counts[2],
        "count_miss": counts[3],
        "total_judgements": total,
        "accuracy": float(accuracy),
        "miss_fraction": float(counts[3] / total),
    }


def replay_local_quality(replay: osrparse.Replay) -> dict[str, float | int]:
    return replay_local_quality_from_counts(
        replay.count_300,
        replay.count_100,
        replay.count_50,
        replay.count_miss,
    )


def load_sender_local_policy(path: str | Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_sender_local_policy(policy)
    return policy


def validate_sender_local_policy(policy: dict[str, Any]) -> None:
    features = tuple(policy.get("feature_names", ()))
    if features != ("accuracy",):
        raise ValueError(
            "Sender-local v1 podržava samo unapred odabranu accuracy-only politiku."
        )
    thresholds = np.asarray(policy.get("thresholds", ()), dtype=np.float64)
    alphas = np.asarray(policy.get("alpha_values", ()), dtype=np.float64)
    if thresholds.ndim != 1 or alphas.ndim != 1:
        raise ValueError("Policy thresholds/alpha_values moraju biti 1D.")
    if len(alphas) != len(thresholds) + 1:
        raise ValueError("Broj alpha vrednosti mora biti broj pragova + 1.")
    if len(thresholds) and np.any(thresholds[1:] <= thresholds[:-1]):
        raise ValueError("Accuracy pragovi moraju biti strogo rastući.")
    if np.any(~np.isfinite(thresholds)) or np.any(~np.isfinite(alphas)):
        raise ValueError("Policy sadrži NaN/inf.")
    if np.any((thresholds <= 0.0) | (thresholds >= 1.0)):
        raise ValueError("Accuracy pragovi moraju biti u (0, 1).")
    if np.any((alphas < 10.0) | (alphas > 20.0)):
        raise ValueError("Alpha mora ostati u pre-registered opsegu [10, 20].")
    if policy.get("formula") != ACCURACY_FORMULA:
        raise ValueError("Policy ne zapisuje tačnu projektnu accuracy formulu.")
    ratios = policy.get("threshold_ratios")
    if ratios is not None:
        if len(ratios) != len(thresholds):
            raise ValueError("Broj racionalnih i float pragova se razlikuje.")
        previous_num, previous_den = 0, 1
        for ratio in ratios:
            numerator = int(ratio["numerator"])
            denominator = int(ratio["denominator"])
            if numerator <= 0 or denominator <= 0 or numerator >= denominator:
                raise ValueError("Racionalni accuracy prag mora biti u (0, 1).")
            if numerator * previous_den <= previous_num * denominator:
                raise ValueError("Racionalni accuracy pragovi nisu strogo rastući.")
            previous_num, previous_den = numerator, denominator


def alpha_from_accuracy_ratio(
    policy: dict[str, Any],
    *,
    numerator: int,
    denominator: int,
) -> float:
    """Apply a canonical policy using exact integer cross-products."""
    validate_sender_local_policy(policy)
    numerator = int(numerator)
    denominator = int(denominator)
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError("Accuracy razlomak mora biti konačan odnos u [0, 1].")
    ratios = policy.get("threshold_ratios")
    if ratios is None:
        raise ValueError("Policy nema kanonske racionalne pragove.")
    bin_index = 0
    for ratio in ratios:
        threshold_num = int(ratio["numerator"])
        threshold_den = int(ratio["denominator"])
        if numerator * threshold_den < threshold_num * denominator:
            break
        bin_index += 1
    return float(policy["alpha_values"][bin_index])


def alpha_from_sender_local_counts(
    policy: dict[str, Any],
    *,
    count_300: int,
    count_100: int,
    count_50: int,
    count_miss: int,
) -> float:
    """Apply canonical sender-local inference directly to replay hit counts."""
    numerator, denominator = accuracy_ratio_from_counts(
        count_300,
        count_100,
        count_50,
        count_miss,
    )
    return alpha_from_accuracy_ratio(
        policy,
        numerator=numerator,
        denominator=denominator,
    )


def alpha_from_sender_local_features(
    policy: dict[str, Any],
    *,
    accuracy: float,
) -> float:
    validate_sender_local_policy(policy)
    value = float(accuracy)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("accuracy mora biti konačan i u [0, 1].")
    thresholds = np.asarray(policy["thresholds"], dtype=np.float64)
    alphas = np.asarray(policy["alpha_values"], dtype=np.float64)
    bin_index = int(np.searchsorted(thresholds, value, side="right"))
    return float(alphas[bin_index])


def alpha_from_replay(policy: dict[str, Any], replay: osrparse.Replay) -> float:
    if policy.get("threshold_ratios") is not None:
        return alpha_from_sender_local_counts(
            policy,
            count_300=replay.count_300,
            count_100=replay.count_100,
            count_50=replay.count_50,
            count_miss=replay.count_miss,
        )
    quality = replay_local_quality(replay)
    return alpha_from_sender_local_features(
        policy,
        accuracy=float(quality["accuracy"]),
    )
