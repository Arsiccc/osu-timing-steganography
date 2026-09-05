"""Focused sanity checks for strong-steganalysis feature extraction."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from osu_stego.analysis.timing_features import (
    BASELINE_FEATURES,
    FEATURE_SETS,
    FULL_FEATURES,
    residual_timing_features,
)
from scripts.experiments.run_steganalysis import residual_features
from scripts.experiments.run_strong_steganalysis import (
    FEATURE_FIELDS,
    METHODS,
    validate_feature_frame,
)


def check_baseline_exactness() -> None:
    examples = (
        np.asarray([1.0, 2.0, 4.0, 8.0, 16.0]),
        np.asarray([1.0, np.nan, 4.0, 8.0, np.nan, 16.0]),
        np.asarray([-12.0, -3.0, 2.0, np.nan, 7.0, 21.0]),
    )
    for values in examples:
        legacy = residual_features(values)
        strong = residual_timing_features(values)
        for name in BASELINE_FEATURES:
            if float(legacy[name]) != float(strong[name]):
                raise AssertionError(f"Baseline nije tačno reprodukovan: {name}")


def check_note_index_semantics() -> None:
    separated = residual_timing_features(np.asarray([1.0, np.nan, -1.0]))
    if separated["autocorrelation_lag1"] != 0.0:
        raise AssertionError("NaN je napravio lažnu lag-1 susednost.")

    quartered = residual_timing_features(
        np.asarray([1.0, np.nan, np.nan, np.nan, 100.0, 100.0, 100.0, 100.0])
    )
    expected = (1.0, 0.0, 100.0, 100.0)
    actual = tuple(
        float(quartered[f"quarter_{index}_mean_absolute_residual"])
        for index in range(1, 5)
    )
    if actual != expected:
        raise AssertionError(f"Quarter-i nisu note-indeksirani: {actual}")


def check_degenerate_inputs() -> None:
    for values in (
        np.asarray([], dtype=float),
        np.asarray([np.nan]),
        np.asarray([5.0]),
        np.asarray([5.0, 5.0, np.nan]),
    ):
        features = residual_timing_features(values)
        vector = np.asarray([features[name] for name in FULL_FEATURES], dtype=float)
        if not np.all(np.isfinite(vector)):
            raise AssertionError("Degenerisan replay je proizveo NaN/inf.")


def check_schema_and_api() -> None:
    if tuple(FEATURE_SETS) != (
        "baseline",
        "baseline_global",
        "baseline_temporal",
        "baseline_position",
        "full",
    ):
        raise AssertionError("Feature-set redosled nije determinističan.")
    if tuple(FEATURE_SETS["full"]) != tuple(FULL_FEATURES):
        raise AssertionError("FULL redosled nije zamrznut.")
    if tuple(inspect.signature(residual_timing_features).parameters) != ("residuals",):
        raise AssertionError("Extractor prihvata nedozvoljene pomoćne ulaze.")
    forbidden = ("message", "pn", "ber", "label", "population", "skill")
    if any(token in name.lower() for name in FULL_FEATURES for token in forbidden):
        raise AssertionError("Feature schema sadrži zabranjeni signal.")


def check_pair_alignment() -> None:
    clean = residual_timing_features(np.asarray([1.0, 2.0, 3.0, 4.0]))
    stego = residual_timing_features(np.asarray([1.0, 3.0, 2.0, 5.0]))
    rows = []
    for method_index, method in enumerate(METHODS):
        common = {field: "" for field in FEATURE_FIELDS}
        common.update(
            {
                "experiment_version": "test",
                "feature_version": "test",
                "feature_config_id": f"config-{method_index}",
                "method": method,
                "partition": "validation",
                "replay_file": "one.osr",
                "beatmap_hash": "map",
            }
        )
        rows.extend(
            [
                {**common, **clean, "label": 0},
                {**common, **stego, "label": 1},
            ]
        )
    frame = pd.DataFrame(rows)[FEATURE_FIELDS]
    validate_feature_frame(frame, expected_replays=1)


def main() -> None:
    check_baseline_exactness()
    check_note_index_semantics()
    check_degenerate_inputs()
    check_schema_and_api()
    check_pair_alignment()
    print("STRONG STEGANALYSIS CHECK OK")
    print(f"feature_count={len(FULL_FEATURES)} feature_sets={list(FEATURE_SETS)}")


if __name__ == "__main__":
    main()
